/*
 Copyright 2024 Tomo Sasaki

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <iostream> // For std::cout, std::cerr
#include <iomanip>  // For std::setw
#include <memory>   // For std::unique_ptr
#include <map>      // For std::map
#include <cmath>    // For std::log
#include <Eigen/Dense>
#include <chrono>    // For timing
#include <execution> // For parallel execution policies
#include <future> // For multi-threading
#include <thread> // For multi-threading

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/helper.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/qp_solver.hpp"
#include "osqp++.h"

namespace cddp
{

// Solve the problem
CDDPSolution CDDP::solveASCDDP()
{
    // Initialize if not done
    if (!initialized_) {
        initializeCDDP();
    }

    if (!initialized_) {
        std::cerr << "CDDP: Initialization failed" << std::endl;
        throw std::runtime_error("CDDP: Initialization failed");
    }

    // Prepare solution struct
    CDDPSolution solution;
    solution.converged = false;
    solution.alpha = alpha_;
    solution.iterations = 0;
    solution.solve_time = 0.0;
    solution.time_sequence.reserve(horizon_ + 1);
    for (int t = 0; t <= horizon_; ++t) {
        solution.time_sequence.push_back(timestep_ * t);
    }
    solution.control_sequence.reserve(horizon_);
    solution.state_sequence.reserve(horizon_ + 1);
    solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency
    solution.lagrangian_sequence.reserve(options_.max_iterations); // Reserve space for efficiency
    solution.cost_sequence.push_back(J_);

    // Evaluate Lagrangian
    L_ = J_;
    solution.lagrangian_sequence.push_back(L_);

    if (options_.verbose) {
        printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); // Initial iteration information
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;

    // Main loop of CDDP
    while (iter < options_.max_iterations)
    {
        ++iter;
        solution.iterations = iter;

        // Check maximum CPU time
        if (options_.max_cpu_time > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
            if (duration.count() * 1e-6 > options_.max_cpu_time) {
                if (options_.verbose) {
                    std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                }
                break;
            }
        }

        // 1. Backward pass: Solve Riccati recursion to compute optimal control law
        bool backward_pass_success = false;
        while (!backward_pass_success)
        {
            backward_pass_success = solveASCDDPBackwardPass();

            if (!backward_pass_success) {
                if (options_.debug) {
                    std::cerr << "CDDP: Backward pass failed" << std::endl;
                }

                // Increase regularization
                increaseRegularization();

                if (isRegularizationLimitReached()) {
                    if (options_.verbose) {
                        std::cerr << "CDDP: Backward pass regularization limit reached" << std::endl;
                    }
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }

        // 2. Forward pass (either single-threaded or multi-threaded)
        ForwardPassResult best_result;
        best_result.cost = std::numeric_limits<double>::infinity();
        best_result.lagrangian = std::numeric_limits<double>::infinity();
        bool forward_pass_success = false;

        // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
        for (double alpha : alphas_) {
            ForwardPassResult result = solveASCDDPForwardPass(alpha);
            
            if (result.success && result.cost < best_result.cost) {
                best_result = result;
                forward_pass_success = true;
                
                // Check for early termination
                if (result.success) {
                    if (options_.debug) {
                        std::cout << "CDDP: Early termination due to successful forward pass" << std::endl;
                    }
                    break; // Exit if forward pass is successful
                }
            }
        }

        // Update solution if a feasible forward pass was found
        if (forward_pass_success) {
            if (options_.debug) {
                std::cout << "Best cost: " << best_result.cost << std::endl;
                std::cout << "Best alpha: " << best_result.alpha << std::endl;
            }
            X_ = best_result.state_sequence;
            U_ = best_result.control_sequence;
            dJ_ = J_ - best_result.cost;
            J_ = best_result.cost;
            dL_ = L_ - best_result.lagrangian;
            L_ = best_result.lagrangian;
            alpha_ = best_result.alpha;
            solution.cost_sequence.push_back(J_);
            solution.lagrangian_sequence.push_back(L_);

            // Decrease regularization
            decreaseRegularization();

            // // Check termination
            // if (dJ_ < options_.cost_tolerance) {
            //     solution.converged = true;
            //     break;
            // }
        } else {
            bool early_termination_flag = false; // TODO: Improve early termination
            // Increase regularization
            increaseRegularization();
            
            // If limit is reached treat as converged
            if (isRegularizationLimitReached()) {
                if ((dJ_ < options_.cost_tolerance * 1e2) ||
                    (optimality_gap_ < options_.grad_tolerance * 1e1)) 
                {
                    solution.converged = true;
                }  else
                {
                    // We are forced to large regularization but still not near local min
                    solution.converged = false;
                }
                if (options_.verbose) {
                    std::cerr << "CDDP: Regularization limit reached. "
                            << (solution.converged ? "Treating as converged." : "Not converged.") 
                            << std::endl;
                }
                break;
            }
        }

        // Check convergence
        if (iter > 0 && dJ_ < options_.cost_tolerance) {
            solution.converged = true;
            break;
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); 
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 

    // Finalize solution
    solution.state_sequence = X_;
    solution.control_sequence = U_;
    solution.alpha = alpha_;
    solution.solve_time = duration.count(); // Time in microseconds
    
    if (options_.verbose) {
        printSolution(solution); 
    }

    return solution;
}

// Backward pass
bool CDDP::solveASCDDPBackwardPass()
{
    // Initialize variables
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    const int dual_dim = getTotalDualDim() - control_dim;
    const auto active_set_tol = 1e-6;

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Terminal cost and its derivatives]
    Eigen::VectorXd V_x = objective_->getFinalCostGradient(X_.back());
    Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());

    // Pre-allocate matrices
    Eigen::MatrixXd A(state_dim, state_dim);
    Eigen::MatrixXd B(state_dim, control_dim);
    Eigen::VectorXd Q_x(state_dim);
    Eigen::VectorXd Q_u(control_dim);
    Eigen::MatrixXd Q_xx(state_dim, state_dim);
    Eigen::MatrixXd Q_uu(control_dim, control_dim);
    Eigen::MatrixXd Q_uu_reg(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::MatrixXd Q_ux_reg(control_dim, state_dim);
    Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);
    Eigen::MatrixXd active_constraint_tabl(2 * (control_dim), horizon_);
    double Qu_error = 0.0;

    // Backward Riccati recursion
    for (int t = horizon_ - 1; t >= 0; --t)
    {
        // Get state and control
        const Eigen::VectorXd &x = X_[t];
        const Eigen::VectorXd &u = U_[t];

        // Get continuous dynamics Jacobians
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx;
        A.diagonal().array() += 1.0; // More efficient way to add identity
        B = timestep_ * Fu;

        // Get cost and its derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // Compute Q-function matrices
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;
        Q_ux = l_ux + B.transpose() * V_xx * A;
        Q_uu = l_uu + B.transpose() * V_xx * B;

        if (options_.regularization_type == "state" || options_.regularization_type == "both") {
            Q_ux_reg = l_ux + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * A;
            Q_uu_reg = l_uu + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * B;
        } else {
            Q_ux_reg = Q_ux;
            Q_uu_reg = Q_uu;
        } 

        if (options_.regularization_type == "control" || options_.regularization_type == "both") {
            Q_uu_reg.diagonal().array() += regularization_control_;
        }

        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
        const Eigen::VectorXd& eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            if (options_.debug) {
                std::cerr << "CDDP: Q_uu is still not positive definite" << std::endl;
            }
            return false;
        }

        /*  --- Identify Active Constraint --- */
        int active_constraint_index = 0;
        Eigen::MatrixXd C(dual_dim, control_dim); // Control constraint matrix
        Eigen::MatrixXd D(dual_dim, state_dim);   // State constraint matrix

        // Identify constrol constraints
        for (int j = 0; j < control_dim; j++)
        {
            if (u(j) <= control_box_constraint->getLowerBound()(j) + active_set_tol)
            {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
                e(j) = 1.0;
                C.row(active_constraint_index) = -e; // Note the negative sign
                D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
                active_constraint_index += 1;
            }
            else if (u(j) >= control_box_constraint->getUpperBound()(j) - active_set_tol)
            {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
                e(j) = 1.0; // No negative here
                C.row(active_constraint_index) = e;
                D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
                active_constraint_index += 1;
            }
        }

        // Identify state constraints
        if (t < horizon_ - 1)
        {
            for (auto &constraint : constraint_set_)
            {
                if (constraint.first == "ControlBoxConstraint")
                {
                    continue;
                }

                Eigen::VectorXd constraint_vals = constraint.second->evaluate(X_[t + 1], U_[t + 1]) - constraint.second->getUpperBound();
                Eigen::MatrixXd C_state = constraint.second->getStateJacobian(X_[t + 1], U_[t + 1]);
                Eigen::MatrixXd D_state = constraint.second->getControlJacobian(X_[t + 1], U_[t + 1]);
                
                for (int j = 0; j < constraint_vals.size(); j++)
                {
                    if (std::abs(constraint_vals(j)) <= active_set_tol) {  
                        C.row(active_constraint_index) = C_state.row(j) * Fu;  
                        D.row(active_constraint_index) = -D_state.row(j) * Fx; 
                        active_constraint_index++;
                    }
                }
            }
        }

        Eigen::MatrixXd active_constraint_table = Eigen::MatrixXd::Zero(2 * (control_dim), horizon_);
        if (active_constraint_index == 0)
        { // No active constraints
            const Eigen::MatrixXd &H = Q_uu_reg.inverse();
            k = -H * Q_u;
            K = -H * Q_ux_reg;
        }
        else
        {
            // Extract identified active constraints
            Eigen::MatrixXd grad_x_g = D.topRows(active_constraint_index);
            Eigen::MatrixXd grad_u_g = C.topRows(active_constraint_index);

            // Calculate Lagrange multipliers
            Eigen::MatrixXd Q_uu_inv = Q_uu_reg.inverse();
            Eigen::MatrixXd lambda = -(grad_u_g * Q_uu_inv * grad_u_g.transpose()).inverse() * (grad_u_g * Q_uu_inv * Q_u);

            // Find indices where lambda is non-negative
            std::vector<int> active_indices;
            for (int i = 0; i < lambda.rows(); ++i)
            {
                if (lambda(i) >= 0)
                {
                    active_indices.push_back(i);
                }
            }
            int active_count_new = active_indices.size();

            // Create new constraint matrices
            Eigen::MatrixXd C_new = Eigen::MatrixXd::Zero(active_count_new, control_dim);
            Eigen::MatrixXd D_new = Eigen::MatrixXd::Zero(active_count_new, state_dim);

            if (active_count_new > 0)
            {
                // Fill new constraint matrices with active constraints
                for (int i = 0; i < active_count_new; ++i)
                {
                    C_new.row(i) = grad_u_g.row(active_indices[i]);
                    D_new.row(i) = grad_x_g.row(active_indices[i]);
                }

                // Calculate feedback gains
                Eigen::MatrixXd W = -(C_new * Q_uu_inv * C_new.transpose()).inverse() * (C_new * Q_uu_inv);
                Eigen::MatrixXd H = Q_uu_inv * (Eigen::MatrixXd::Identity(control_dim, control_dim) - C_new.transpose() * W);
                k = -H * Q_u;
                K = -H * Q_ux_reg + W.transpose() * D_new;
            }
            else
            {
                // If no active constraints remain, revert to unconstrained solution
                Eigen::MatrixXd H = Q_uu_reg.inverse();
                K = -H * Q_ux_reg;
                k = -H * Q_u;
            }
        }

        // Store Q-function matrices
        Q_UU_[t] = Q_uu_reg;
        Q_UX_[t] = Q_ux_reg;
        Q_U_[t] = Q_u;

        // Compute value function approximation
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ = dV_ + dV_step;
        V_x = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize Hessian

        // Compute optimality gap (Inf-norm) for convergence check
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

        // TODO: Add constraint optimality gap analysis
        optimality_gap_ = Qu_error;
    }

    if (options_.debug) {
        std::cout << "Qu_error: " << Qu_error << std::endl;
        std::cout << "dV: " << dV_.transpose() << std::endl;
    }

    return true;
}

// Forward pass
ForwardPassResult CDDP::solveASCDDPForwardPass(double alpha)
{   
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.lagrangian = std::numeric_limits<double>::infinity();
    result.alpha = alpha;

    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    const int dual_dim = getTotalDualDim() - control_dim;

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Initialize trajectories with current state and control histories.
    std::vector<Eigen::VectorXd> X_new = X_;
    std::vector<Eigen::VectorXd> U_new = U_;
    X_new[0] = initial_state_;
    double J_new = 0.0;

    // Loop over the horizon
    for (int t = 0; t < horizon_; ++t) {
        const Eigen::VectorXd &x = X_new[t];
        const Eigen::VectorXd &u = U_new[t];
        Eigen::VectorXd delta_x = x - X_[t];

        // Extract Q-function matrices computed in the backward pass.
        const Eigen::VectorXd &Q_u = Q_U_[t];
        const Eigen::MatrixXd &Q_uu = Q_UU_[t];
        const Eigen::MatrixXd &Q_ux = Q_UX_[t];

        // Create QP problem
        Eigen::SparseMatrix<double> P(Q_uu.rows(), Q_uu.cols()); // Hessian of QP objective
        int numNonZeros = Q_uu.nonZeros(); 
        P.reserve(numNonZeros);
        for (int i = 0; i < Q_uu.rows(); ++i) {
            for (int j = 0; j < Q_uu.cols(); ++j) {
                if (Q_uu(i, j) != 0) {
                    P.insert(i, j) = Q_uu(i, j);
                }
            }
        }
        P.makeCompressed(); // Important for efficient storage and operations

        // Form the gradient of the QP objective:
        // q = alpha * Q_u + Q_ux * delta_x
        Eigen::VectorXd q = alpha * Q_u + Q_ux * delta_x;

        // Create QP constraints
        Eigen::MatrixXd A_dense = Eigen::MatrixXd::Identity(control_dim, control_dim);
        Eigen::VectorXd lb_dense = control_box_constraint->getLowerBound() - u;
        Eigen::VectorXd ub_dense = control_box_constraint->getUpperBound() - u;

        Eigen::MatrixXd A_aug = Eigen::MatrixXd::Zero(dual_dim, control_dim);
        Eigen::VectorXd lb_aug = Eigen::VectorXd::Zero(dual_dim);
        Eigen::VectorXd ub_aug = Eigen::VectorXd::Zero(dual_dim);   

        // First block: control constraints
        A_aug.topRows(control_dim) = A_dense;
        lb_aug.head(control_dim) = lb_dense;
        ub_aug.head(control_dim) = ub_dense;

        // Second block: state constraints
        int row_index = control_dim;
        if (t < horizon_ - 1) {
            auto [fx, fu] = system_->getJacobians(x, u);
            Eigen::MatrixXd Fu = timestep_ * fu;

            // Predicted next state
            Eigen::VectorXd x_next = system_->getDiscreteDynamics(x, u);

            for (const auto &constraint : constraint_set_) {
                if (constraint.first == "ControlBoxConstraint") {
                    continue;
                }
                Eigen::VectorXd cons_vals = constraint.second->evaluate(x_next, u) - constraint.second->getUpperBound();
                Eigen::MatrixXd cons_jac_x = constraint.second->getStateJacobian(x_next, u);
                
                int m = cons_vals.size();
                A_aug.block(row_index, 0, m, control_dim) = cons_jac_x * Fu;
                lb_aug.segment(row_index, m).setConstant(-std::numeric_limits<double>::infinity());
                ub_aug.segment(row_index, m) = -cons_vals;
                row_index += m;

            }
        }

        // Convert augmented constraint matrix to sparse format.
        Eigen::SparseMatrix<double> A_sparse = A_aug.sparseView();

        // Initialize QP solver
        osqp::OsqpInstance instance;

        // Set the objective
        instance.objective_matrix = P;
        instance.objective_vector = q;
        instance.constraint_matrix = A_sparse;
        instance.lower_bounds = lb_aug;
        instance.upper_bounds = ub_aug;

        // Solve the QP problem
        osqp::OsqpSolver osqp_solver;
        osqp::OsqpSettings settings;
        settings.warm_start = true;
        settings.verbose = false;

        osqp_solver.Init(instance, settings);
        osqp::OsqpExitCode exit_code = osqp_solver.Solve();

        if (exit_code != osqp::OsqpExitCode::kOptimal) {
            if (options_.debug) {
                std::cerr << "CDDP: QP solver failed at time step " << t << std::endl;
            }
            result.success = false;
            return result;
        }

        // Update control using the QP solution delta_u.
        Eigen::VectorXd delta_u = osqp_solver.primal_solution();
        U_new[t] += delta_u;

        // Compute running cost and propagate state.
        J_new += objective_->running_cost(x, U_new[t], t);
        X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);
    }
    // Add terminal cost.
    J_new += objective_->terminal_cost(X_new.back());

    // Compute actual cost reduction and the predicted improvement.
    double dJ = J_ - J_new;
    double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
    double reduction_ratio = (expected > 0.0) ? dJ / expected : std::copysign(1.0, dJ);

    // --- Acceptance Criterion ---
    if (dJ <= 0) {
        if (options_.debug) {
            std::cerr << "CDDP: Forward pass did not yield sufficient decrease (dJ: " 
                      << dJ << ", reduction_ratio: " << reduction_ratio << ")" << std::endl;
        }
        result.success = false;
    } else {
        result.success = true;
        result.state_sequence = X_new;
        result.control_sequence = U_new;
        result.cost = J_new;
        result.lagrangian = J_new; // For this example we use the cost as the Lagrangian.
    }
    return result;
}

} // namespace cddp