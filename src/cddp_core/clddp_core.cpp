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
#include <iomanip> // For std::setw
#include <memory> // For std::unique_ptr
#include <map>    // For std::map
#include <cmath>  // For std::log
#include <Eigen/Dense>
#include <chrono> // For timing
#include <execution> // For parallel execution policies
#include "osqp++.h"
// TODO: gurobi solver compatibility
// #include "gurobi_c++.h"

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/helper.hpp"
#include "cddp_core/boxqp.hpp"

namespace cddp {


CDDPSolution CDDP::solveCLDDP() {
    // Initialize CDDP solver
    initializeCDDP();

    if (options_.verbose) {
        printOptions(options_);
    }

    // Initialize solution
    CDDPSolution solution;
    solution.converged = false;
    solution.time_sequence.reserve(horizon_ + 1);
    for (int t = 0; t <= horizon_; ++t) {
        solution.time_sequence.push_back(timestep_ * t);
    }
    solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency

    // Evaluate initial cost
    J_ = objective_->evaluate(X_, U_);
    solution.cost_sequence.push_back(J_);

    if (options_.verbose) {
        printIteration(0, J_, 0.0, optimality_gap_, regularization_state_, regularization_control_, alpha_); // Initial iteration information
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;

    // Main loop of CDDP
    while (iter < options_.max_iterations)
    {
        ++iter;

        // Check maximum CPU time
        if (options_.max_cpu_time > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
            if (duration.count() * 1e-6 > options_.max_cpu_time) {
                std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                break;
            }
        }

        // 1. Backward pass: Solve Riccati recursion to compute optimal control law
        bool backward_pass_success = false;
        while (!backward_pass_success) {
            backward_pass_success = solveCLDDPBackwardPass();

            if (!backward_pass_success) {
                std::cerr << "CDDP: Backward pass failed" << std::endl;

                // Increase regularization
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);

                if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                    std::cerr << "CDDP: Regularization limit reached" << std::endl;
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }
        
        // Check termination due to small cost improvement
        if (optimality_gap_ < options_.grad_tolerance && regularization_state_ < 1e-4 && regularization_control_ < 1e-4) {
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_;
            if (regularization_state_ <= options_.regularization_state_min) {
                regularization_state_ = 0.0;
            }
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_;
            if (regularization_control_ <= options_.regularization_control_min) {
                regularization_control_ = 0.0;
            }

            solution.converged = true;
            break;
        }

        bool forward_pass_success = false;
        // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
        if (backward_pass_success) {
            forward_pass_success = solveCLDDPForwardPass();
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, 0.0, optimality_gap_, regularization_state_, regularization_control_, alpha_); 
        }
       
       if (forward_pass_success) {
            // Decrease regularization
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_;
            if (regularization_state_ <= options_.regularization_state_min) {
                regularization_state_ = 0.0;
            }
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_;
            if (regularization_control_ <= options_.regularization_control_min) {
                regularization_control_ = 0.0;
            }

            // Append Latest Cost
            solution.cost_sequence.push_back(J_);

            if (dJ_ < options_.cost_tolerance) {
                solution.converged = true;
                solution.iterations = iter;
                break;
            }
        }
    }

    // Finalize solution
    solution.control_sequence = U_;
    solution.state_sequence = X_;
    solution.iterations = solution.converged ? iter : options_.max_iterations;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
    solution.solve_time = duration.count(); // Time in microseconds
    printSolution(solution);

    return solution;
}

bool CDDP::solveCLDDPBackwardPass() {
    // Initialize variables
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    const auto active_set_tol = options_.active_set_tolerance;

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Terminal cost and its derivatives
    V_X_.back() = objective_->getFinalCostGradient(X_.back());
    V_XX_.back() = objective_->getFinalCostHessian(X_.back());

    // Pre-allocate matrices
    Eigen::MatrixXd A(state_dim, state_dim);
    Eigen::MatrixXd B(state_dim, control_dim);
    Eigen::VectorXd Q_x(state_dim);
    Eigen::VectorXd Q_u(control_dim);
    Eigen::MatrixXd Q_xx(state_dim, state_dim);
    Eigen::MatrixXd Q_uu(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);
    Eigen::SparseMatrix<double> P(control_dim, control_dim); // Hessian of QP objective

    // Create BoxQP solver
    cddp::BoxQPOptions qp_options;
    qp_options.verbose = false;
    qp_options.maxIter = 100;
    cddp::BoxQPSolver qp_solver(qp_options);

    double Qu_error = 0.0;

    // Backward Riccati recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get state and control
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Extract value function approximation
        const Eigen::VectorXd& V_x = V_X_[t + 1];
        const Eigen::MatrixXd& V_xx = V_XX_[t + 1];

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
if (t > 400) {
std::cout << "Q_x: " << Q_x.transpose() << std::endl;
std::cout << "Q_u: " << Q_u.transpose() << std::endl;

}

        if (options_.regularization_type == "state" || options_.regularization_type == "both") {
            Q_ux = l_ux + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * A;
            Q_uu = l_uu + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * B;
        } else {
            Q_ux = l_ux + B.transpose() * V_xx * A;
            Q_uu = l_uu + B.transpose() * V_xx * B;
        }

        // Symmetrize Q_uu for cholensky decomposition
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Control Regularization
        if (options_.regularization_type == "control" || options_.regularization_type == "both") {
            Q_uu += options_.regularization_control * Eigen::MatrixXd::Identity(control_dim, control_dim);
        } 
// std::cout << "Q_uu: " << Q_uu << std::endl;
        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            // Add regularization
            // Q_uu.diagonal() += 1e-6;
            std::cout << "Q_uu is not positive definite at " << t << std::endl;

            eigenvalues = es.eigenvalues().real();
            if (eigenvalues.minCoeff() <= 0) {
                std::cout << "Q_uu is still not positive definite" << std::endl;
                return false;
            }
        }

        // Store Q-function matrices
        Q_UU_[t] = Q_uu;
        Q_UX_[t] = Q_ux;
        Q_U_[t] = Q_u;

        if (control_box_constraint == nullptr) {
            const Eigen::MatrixXd &H = Q_uu.inverse();
            k = -H * Q_u;
            K = -H * Q_ux;
        } else {
            // TODO: Migrate to BoxQP solver
            // // Solve Box QP Problem using BoxQPSolver
            // Eigen::VectorXd lower = control_box_constraint->getLowerBound() - u;
            // Eigen::VectorXd upper = control_box_constraint->getUpperBound() - u;

            // cddp::BoxQPResult qp_result = qp_solver.solve(Q_uu, Q_u, lower, upper, u);

            // // if (qp_result.status != cddp::BoxQPStatus::SMALL_GRADIENT && 
            // //     qp_result.status != cddp::BoxQPStatus::SMALL_IMPROVEMENT) {   
            // //     std::cout << "BoxQP solver failed with status: " << static_cast<int>(qp_result.status) << std::endl;
            // //     return false;
            // // }

            // // Extract solution
            // k = qp_result.x;  // Feedforward term

            // const auto &H = qp_result.Hfree;
            // const auto &free = qp_result.free;

            // const Eigen::MatrixXd &Q_ux_free = Q_ux(free, Eigen::all);

            // Compute gain for free dimensions
            // Eigen::MatrixXd K_free = H.solve(-Q_ux_free);


            // // Fill in the gain matrix
            // K.setZero();
            // K(Eigen::all, free) = K_free;


            /*    Solve Box QP Problem    */   
            int numNonZeros = Q_uu.nonZeros(); 
            P.reserve(numNonZeros);
            P.setZero();
            for (int i = 0; i < Q_uu.rows(); ++i) {
                for (int j = 0; j < Q_uu.cols(); ++j) {
                    if (Q_uu(i, j) != 0) {
                        P.insert(i, j) = Q_uu(i, j);
                    }
                }
            }
            P.makeCompressed(); // Important for efficient storage and operations
            osqp_solver_.UpdateObjectiveMatrix(P);

            const Eigen::VectorXd& q = Q_u; // Gradient of QP objective
            osqp_solver_.SetObjectiveVector(q);  

            // Lower and upper bounds
            Eigen::VectorXd lb = 1.0 * (control_box_constraint->getLowerBound() - u);
            Eigen::VectorXd ub = 1.0 * (control_box_constraint->getUpperBound() - u);    
            osqp_solver_.SetBounds(lb, ub);

            // Solve the QP problem TODO: Use SDQP instead of OSQP
            osqp::OsqpExitCode exit_code = osqp_solver_.Solve();

            // Extract solution
            double optimal_objective = osqp_solver_.objective_value();
            k = osqp_solver_.primal_solution();

            // // Compute gain matrix
            K = -Q_uu.inverse() * Q_ux;

        }

        // Store feedforward and feedback gain
        k_[t] = k;
        K_[t] = K;

        // Compute value function approximation
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ = dV_ + dV_step;
// std::cout << "dV: " << dV_.transpose() << std::endl;
        V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_XX_[t] = 0.5 * (V_XX_[t] + V_XX_[t].transpose()); // Symmetrize Hessian

        // Compute optimality gap (Inf-norm) for convergence check
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

        // TODO: Add constraint optimality gap analysis
        optimality_gap_ = Qu_error;
    }

    return true;
}

bool CDDP::solveCLDDPForwardPass() {
    bool is_feasible = false;
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    int iter = 0;
    double alpha = options_.backtracking_coeff;

    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Line-search iteration 
    for (iter = 0; iter < options_.max_line_search_iterations; ++iter) {
        // Initialize cost and constraints
        double J_new = 0.0, dJ = 0.0, expected_dV = 0.0, gradient_norm = 0.0;

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        X_new[0] = initial_state_;

        for (int t = 0; t < horizon_; ++t) {
            // Get state and control
            const Eigen::VectorXd& x = X_new[t];
            const Eigen::VectorXd& u = U_new[t];

            // Deviation from the nominal trajectory
            const Eigen::VectorXd& delta_x = x - X_[t];

            // Extract control laws
            const Eigen::VectorXd& k = k_[t];
            const Eigen::MatrixXd& K = K_[t];

            // Create a new solution
            U_new[t] = u + alpha * k + K * delta_x;

            // Clamp control input
            if (control_box_constraint != nullptr) {
                U_new[t] = control_box_constraint->clamp(U_new[t]);
            }

            // Compute cost
            J_new += objective_->running_cost(x, U_new[t], t);

            // Compute new state
            X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);

        }
        J_new += objective_->terminal_cost(X_new.back());

        // Calculate Cost Reduction
        dJ = J_ - J_new;

        double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));

        double reduction_ratio = 0.0;
        if (expected > 0.0) {
            reduction_ratio = dJ / expected;
        } else {
            reduction_ratio = std::copysign(1.0, dJ);
            std::cout << "Expected improvement is not positive" << std::endl;
        }

        if (reduction_ratio > options_.minimum_reduction_ratio) {
            // Update state and control
            X_ = X_new;
            U_ = U_new;
            J_ = J_new;
            dJ_ = dJ;
            alpha_ = alpha;
            return true;
        } else {
            alpha *= options_.backtracking_factor;
        }

    }
    return false;
}


} // namespace cddp