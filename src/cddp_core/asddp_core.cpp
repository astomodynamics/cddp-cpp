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
    // // Loop over horizon
    // for (int t = 0; t < 1; ++t) {
    //     // Evaluate state constraint violation
    //     for (const auto& constraint : constraint_set_) {
    //         if (constraint.first == "ControlBoxConstraint") {
    //             L_ += getLogBarrierCost(*constraint.second, X_[t], U_[t], barrier_coeff_, options_.relaxation_coeff);
    //             // Eigen::VectorXd constraint_violation = constraint.second->evaluate(X_[t], U_[t]);
    //             // if (constraint_violation.minCoeff() < 0) {
    //             //     std::cerr << "CDDP: Constraint violation at time " << t << std::endl;
    //             //     std::cerr << "Constraint violation: " << constraint_violation.transpose() << std::endl;
    //             //     throw std::runtime_error("Constraint violation");
    //             // }
    //         }

    //     }
    // }

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

            if (!backward_pass_success)
            {
                std::cerr << "CDDP: Backward pass failed" << std::endl;

                // Increase regularization
                if (options_.regularization_type == "state") {
                    regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                    regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                } else if (options_.regularization_type == "control") {
                    regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                    regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);
                } else if (options_.regularization_type == "both") {
                    regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                    regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                    regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                    regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);
                } 

                if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
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
                    break;
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
            if (options_.regularization_type == "state") {
                regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
                regularization_state_ *= regularization_state_step_;
                if (regularization_state_ < options_.regularization_state_min) {
                    regularization_state_ = options_.regularization_state_min;
                }
            } else if (options_.regularization_type == "control") {
                regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
                regularization_control_ *= regularization_control_step_;
                if (regularization_control_ < options_.regularization_control_min) {
                    regularization_control_ = options_.regularization_control_min;
                }
            } else if (options_.regularization_type == "both") {
                regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
                regularization_state_ *= regularization_state_step_;
                if (regularization_state_ < options_.regularization_state_min) {
                    regularization_state_ = options_.regularization_state_min;
                }
                regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
                regularization_control_ *= regularization_control_step_;
                if (regularization_control_ < options_.regularization_control_min) {
                    regularization_control_ = options_.regularization_control_min;
                }
            }

            // Check termination
            if (dJ_ < options_.cost_tolerance) {
                solution.converged = true;
                break;
            }
        } else {
            bool early_termination_flag = false; // TODO: Improve early termination
            // Increase regularization
            if (options_.regularization_type == "state") {
                if (regularization_state_ < 1e-2) {
                    early_termination_flag = true; // Early termination if regularization is fairly small
                }
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_state_ = std::min(regularization_state_ * regularization_state_step_, options_.regularization_state_max);
                
            } else if (options_.regularization_type == "control") {
                if (regularization_control_ < 1e-2) {
                    early_termination_flag = true; // Early termination if regularization is fairly small
                }
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                regularization_control_ = std::min(regularization_control_ * regularization_control_step_, options_.regularization_control_max);
            } else if (options_.regularization_type == "both") {
                if (regularization_state_ < 1e-2 ||  
                    regularization_control_ < 1e-2) {
                    early_termination_flag = true; // Early termination if regularization is fairly small
                }
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
            } else {
                early_termination_flag = true;
            }

            // Check early termination
            if (options_.early_termination && early_termination_flag) {
                if (dJ_ < options_.cost_tolerance * 1e2 ||
                    (optimality_gap_ < options_.grad_tolerance * 1e1)) 
                {
                    solution.converged = true;
                    if (options_.verbose) {
                        std::cerr << "CDDP: Early termination due to small cost reduction" << std::endl;
                    }
                    break;
                }
            }
            
            // Check regularization limit
            if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
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

        // TODO: Implement log barrier cost and its derivatives // FIXME: Implement log barrier cost and its derivatives
        // Get log barrier cost and its derivatives
        // for (const auto& constraint : constraint_set_) {
        //     if (constraint.first == "ControlBoxConstraint") {
        //         const double barrier_cost = getLogBarrierCost(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
        //         l += barrier_cost;
        //         const auto [l_x_barrier, l_u_barrier] = getLogBarrierCostGradients(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
        //         const auto [l_xx_barrier, l_uu_barrier, l_ux_barrier] = getLogBarrierCostHessians(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
        //         l_x += l_x_barrier;
        //         l_u += l_u_barrier;
        //         // l_xx += l_xx_barrier;
        //         // l_uu += l_uu_barrier;
        //         // l_ux += l_ux_barrier;
        //     }
        // }

        // Compute Q-function matrices
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;
        Q_ux = l_ux + B.transpose() * V_xx * A;
        Q_uu = l_uu + B.transpose() * V_xx * B;

        // TODO: Apply Cholesky decomposition to Q_uu later?
        // // Symmetrize Q_uu for Cholesky decomposition
        // Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

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
        Eigen::MatrixXd C(control_dim, control_dim); // Control constraint matrix
        Eigen::MatrixXd D(control_dim, state_dim);   // State constraint matrix

        // TODO: Implement active set method
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

    return true;
}

// Forward pass
ForwardPassResult CDDP::solveASCDDPForwardPass(double alpha)
{   
    // Prepare result struct
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.lagrangian = std::numeric_limits<double>::infinity();
    result.alpha = alpha;

    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Initialize trajectories
    std::vector<Eigen::VectorXd> X_new = X_;
    std::vector<Eigen::VectorXd> U_new = U_;
    X_new[0] = initial_state_;
    double J_new = 0.0;

    for (int t = 0; t < horizon_; ++t) {
        const Eigen::VectorXd& x = X_new[t];
        const Eigen::VectorXd& u = U_new[t];
        const Eigen::VectorXd& delta_x = x - X_[t];

        // Extract Q-function matrices
        const Eigen::VectorXd &Q_u = Q_U_[t];
        const Eigen::MatrixXd &Q_uu = Q_UU_[t];
        const Eigen::MatrixXd &Q_ux = Q_UX_[t];

        const Eigen::VectorXd &q = alpha * Q_u + Q_ux * delta_x; // Gradient of QP objective

        // Lower and upper bounds
        const Eigen::VectorXd& lb = control_box_constraint->getLowerBound() - u;
        const Eigen::VectorXd& ub = control_box_constraint->getUpperBound() - u;

        // Create b vector
        Eigen::VectorXd b(2 * control_dim);
        b << ub, -lb;

        // Create A matrix
        Eigen::MatrixXd A(2 * control_dim, control_dim); // Upper is identity, lower is -identity
        A.setIdentity();
        A.block(control_dim, 0, control_dim, control_dim) = -Eigen::MatrixXd::Identity(control_dim, control_dim);

        QPSolverOptions options;
        options.verbose = false;
        QPSolver solver(options);

        solver.setDimensions(state_dim, 2*control_dim);
        solver.setHessian(Q_uu);
        solver.setGradient(q);
        solver.setConstraints(A, b);

        QPResult qp_result = solver.solve();

        if (qp_result.status != QPStatus::OPTIMAL) {
            if (options_.debug) {
                std::cerr << "CDDP: QP solver failed" << std::endl;
            }
            result.success = false;
            break;
        }

        // Extract solution
        const Eigen::VectorXd &delta_u = qp_result.x;

        // Extract solution
        U_new[t] += delta_u;

        // Compute cost
        J_new += objective_->running_cost(x, U_new[t], t);

        // Compute new state
        X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);
    }
    J_new += objective_->terminal_cost(X_new.back());
    double dJ = J_ - J_new;
    double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
    double reduction_ratio = expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

    // Check if cost reduction is sufficient
    result.success = true;
    result.state_sequence = X_new;
    result.control_sequence = U_new;
    result.cost = J_new;
    result.lagrangian = J_new;

    return result;
}
} // namespace cddp