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
#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"
// TODO: gurobi solver compatibility
// #include "gurobi_c++.h"

#include "cddp_core/cddp_core.hpp"

namespace cddp {


CDDPSolution CDDP::solveLogCDDP() {
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
    solution.lagrangian_sequence.reserve(options_.max_iterations); // Reserve space for efficiency

    // Evaluate initial cost
    J_ = objective_->evaluate(X_, U_);
    solution.cost_sequence.push_back(J_);
    
    // Evaluate Lagrangian
    L_ = J_;

    // Loop over horizon # TODO: Multi-threading?
    for (int t = 0; t < 1; ++t) {
        // Evaluate state constraint violation
        for (const auto& constraint : constraint_set_) {
            if (constraint.first == "ControlBoxConstraint") {
                L_ += getLogBarrierCost(*constraint.second, X_[t], U_[t], barrier_coeff_, options_.relaxation_coeff);
                // Eigen::VectorXd constraint_violation = constraint.second->evaluate(X_[t], U_[t]);
                // if (constraint_violation.minCoeff() < 0) {
                //     std::cerr << "CDDP: Constraint violation at time " << t << std::endl;
                //     std::cerr << "Constraint violation: " << constraint_violation.transpose() << std::endl;
                //     throw std::runtime_error("Constraint violation");
                // }
            }

        }
    }

    // if (options_.verbose) {
    //     printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_); // Initial iteration information
    // }

    // // Start timer
    // auto start_time = std::chrono::high_resolution_clock::now();
    // int iter = 0;

    // // Main loop of CDDP
    // while (iter < options_.max_iterations) {
    //     ++iter;
        
    //     // Check maximum CPU time
    //     if (options_.max_cpu_time > 0) {
    //         auto end_time = std::chrono::high_resolution_clock::now();
    //         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
    //         if (duration.count() * 1e-6 > options_.max_cpu_time) {
    //             std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
    //             break;
    //         }
    //     }

    //     // 1. Backward pass: Solve Riccati recursion to compute optimal control law
    //     bool backward_pass_success = false;
    //     while (!backward_pass_success) {
    //         backward_pass_success = solveBackwardPass();

    //         if (!backward_pass_success) {
    //             std::cerr << "CDDP: Backward pass failed" << std::endl;

    //             // Increase regularization
    //             regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
    //             regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
    //             regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
    //             regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);

    //             if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
    //                 std::cerr << "CDDP: Regularization limit reached" << std::endl;
    //                 break; // Exit if regularization limit reached
    //             }
    //             continue; // Continue if backward pass fails
    //         }
    //     }
        
    //     // Check termination due to small cost improvement
    //     if (optimality_gap_ < options_.grad_tolerance && regularization_state_ < 1e-4 && regularization_control_ < 1e-4) {
    //         regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
    //         regularization_state_ *= regularization_state_step_;
    //         if (regularization_state_ <= options_.regularization_state_min) {
    //             regularization_state_ = 0.0;
    //         }
    //         regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
    //         regularization_control_ *= regularization_control_step_;
    //         if (regularization_control_ <= options_.regularization_control_min) {
    //             regularization_control_ = 0.0;
    //         }

    //         solution.converged = true;
    //         break;
    //     }

    //     bool forward_pass_success = false;
    //     // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
    //     if (backward_pass_success) {
    //         forward_pass_success = solveForwardPass();
    //     }

    //     // Print iteration information
    //     if (options_.verbose) {
    //         printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_); 
    //     }
       
    //    if (forward_pass_success) {
    //         // Decrease regularization
    //         regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
    //         regularization_state_ *= regularization_state_step_;
    //         if (regularization_state_ <= options_.regularization_state_min) {
    //             regularization_state_ = 0.0;
    //         }
    //         regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
    //         regularization_control_ *= regularization_control_step_;
    //         if (regularization_control_ <= options_.regularization_control_min) {
    //             regularization_control_ = 0.0;
    //         }

    //         // Append Latest Cost
    //         solution.cost_sequence.push_back(J_);

    //         if (dJ_ < options_.cost_tolerance) {
    //             solution.converged = true;
    //             solution.iterations = iter;
    //             break;
    //         }
    //     }
    // }

    // // Finalize solution
    // solution.control_sequence = U_;
    // solution.state_sequence = X_;
    // solution.iterations = solution.converged ? iter : options_.max_iterations;

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
    // solution.solve_time = duration.count(); // Time in microseconds
    // printSolution(solution);

    return solution;
}

bool CDDP::solveLogCDDPBackwardPass() {
    // Initialize variables
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Get control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Terminal cost and derivatives
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
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);

    double Qu_error = 0.0;

    // Backward pass
    for (int t = horizon_ - 1; t >= 0; --t) {
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Get value function
        const Eigen::VectorXd& V_x = V_X_[t + 1];
        const Eigen::MatrixXd& V_xx = V_XX_[t + 1];

        // Get dynamics Jacobians
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Discrete dynamics
        A = timestep_ * Fx;
        A.diagonal().array() += 1.0;
        B = timestep_ * Fu;

        // Get cost derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // Add log barrier terms
        const double barrier_cost = getLogBarrierCost(*control_box_constraint, x, u, barrier_coeff_, options_.relaxation_coeff);
        l += barrier_cost;

        // Get barrier gradients
        auto [barrier_x, barrier_u] = getLogBarrierCostGradients(*control_box_constraint, x, u, barrier_coeff_, options_.relaxation_coeff);
        l_x += barrier_x;
        l_u += barrier_u;

        // Get barrier Hessians  
        auto [barrier_xx, barrier_uu, barrier_ux] = getLogBarrierCostHessians(*control_box_constraint, x, u, barrier_coeff_, options_.relaxation_coeff);
        l_xx += barrier_xx;
        l_uu += barrier_uu;
        l_ux += barrier_ux;

        // Q-function matrices
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;
        Q_ux = l_ux + B.transpose() * V_xx * A;
        Q_uu = l_uu + B.transpose() * V_xx * B;

        // Regularization
        if (options_.regularization_type == "state" || options_.regularization_type == "both") {
            Q_uu += regularization_control_ * Eigen::MatrixXd::Identity(control_dim, control_dim);
        }

        // Make Q_uu symmetric
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Check positive definiteness
        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu);
        if (llt.info() != Eigen::Success) {
            std::cout << "Q_uu not positive definite" << std::endl;
            return false;
        }

        // Compute gains
        k = -Q_uu.ldlt().solve(Q_u);
        K = -Q_uu.ldlt().solve(Q_ux);

        // Store gains
        k_[t] = k;
        K_[t] = K;

        // Update value function approximation
        dV_(0) += Q_u.dot(k);
        dV_(1) += 0.5 * k.dot(Q_uu * k);
        
        V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_XX_[t] = 0.5 * (V_XX_[t] + V_XX_[t].transpose());

        // Update optimality error
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());
    }

    optimality_gap_ = Qu_error;
    return true;
}

bool CDDP::solveLogCDDPForwardPass() {
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    double alpha = options_.backtracking_coeff;
    bool success = false;

    for (int iter = 0; iter < options_.max_line_search_iterations; ++iter) {
        double J_new = 0.0;
        double L_new = 0.0;

        // Create new trajectories
        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        X_new[0] = initial_state_;

        // Forward simulation
        for (int t = 0; t < horizon_; ++t) {
            const Eigen::VectorXd& x = X_new[t];
            const Eigen::VectorXd& u = U_new[t];
            
            // State deviation 
            Eigen::VectorXd dx = x - X_[t];

            // Compute control update
            Eigen::VectorXd du = k_[t] + K_[t] * dx;
            U_new[t] = u + alpha * du;

            // Get costs including barriers
            double step_cost = objective_->running_cost(x, U_new[t], t);
            double barrier_cost = getLogBarrierCost(*control_box_constraint, x, U_new[t], barrier_coeff_, options_.relaxation_coeff);
            
            J_new += step_cost;
            L_new += step_cost + barrier_cost;

            // Forward step
            X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);
        }

        // Terminal costs
        J_new += objective_->terminal_cost(X_new.back());
        L_new += objective_->terminal_cost(X_new.back());

        // Calculate improvements
        double dJ = J_ - J_new;
        double dL = L_ - L_new;
        double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));

        if (expected > 0) {
            double ratio = dL / expected;
            if (ratio > options_.minimum_reduction_ratio) {
                X_ = X_new;
                U_ = U_new; 
                J_ = J_new;
                L_ = L_new;
                dJ_ = dJ;
                dL_ = dL;
                barrier_coeff_ *= options_.barrier_factor;
                success = true;
                break;
            }
        }

        alpha *= options_.backtracking_factor;
    }

    return success;
}

} // namespace cddp