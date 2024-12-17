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
#include <future>
#include <thread>
#include "osqp++.h"
#include "sdqp.hpp"

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
                if (options_.verbose) {
                    std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                }
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
                    if (options_.verbose) {
                        std::cerr << "CDDP: Regularization limit reached" << std::endl;
                    }
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }
        
        // Check termination due to small cost improvement
        if (optimality_gap_ < options_.grad_tolerance) {
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_ * (regularization_state_ > options_.regularization_state_min ? 1.0 : 0.0);
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_ * (regularization_control_ > options_.regularization_control_min ? 1.0 : 0.0);
            
            if (regularization_state_ < 1e-5 && regularization_control_ < 1e-5) {
                solution.converged = true;
                break;
            }
            
        }

        // 2. Forward pass (either single-threaded or multi-threaded)
        ForwardPassResult best_result;
        best_result.cost = std::numeric_limits<double>::infinity();
        bool forward_pass_feasible = false;
        bool forward_pass_success = false;

        if (!options_.use_parallel) {
            // Single-threaded execution with early termination
            for (double alpha : alphas_) {
                ForwardPassResult result = solveCLDDPForwardPass(alpha);
                
                if (result.success && result.cost < best_result.cost) {
                    best_result = result;
                    forward_pass_feasible = true;
                    
                    // Check for early termination
                    double expected_cost_reduction = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
                    if (expected_cost_reduction > 0.0) {
                        double cost_reduction_ratio = (J_ - result.cost) / expected_cost_reduction;
                        if (cost_reduction_ratio > options_.minimum_reduction_ratio) {
                            if (options_.debug) {
                                std::cout << "CDDP: Early termination due to successful forward pass" << std::endl;
                            }
                            break;
                        }
                    }
                }
            }
        } else { 
            // TODO: Improve multi-threaded execution
            // Multi-threaded execution 
            std::vector<std::future<ForwardPassResult>> futures;
            futures.reserve(alphas_.size());
            
            // Launch all forward passes in parallel
            for (double alpha : alphas_) {
                futures.push_back(std::async(std::launch::async, 
                    [this, alpha]() { return solveCLDDPForwardPass(alpha); }));
            }
            
            // Collect results from all threads
            for (auto& future : futures) {
                try {
                    if (future.valid()) {
                        ForwardPassResult result = future.get();
                        if (result.success && result.cost < best_result.cost) {
                            best_result = result;
                            forward_pass_feasible = true;
                        }
                    }
                } catch (const std::exception& e) {
                    if (options_.verbose) {
                        std::cerr << "CDDP: Forward pass thread failed: " << e.what() << std::endl;
                    }
                    continue;
                }
            }
        }

        // Update solution if a feasible forward pass was found
        if (forward_pass_feasible) {
            if (options_.debug) {
                std::cout << "Best cost: " << best_result.cost << std::endl;
                std::cout << "Best alpha: " << best_result.alpha << std::endl;
            }

            double expected_cost_reduction = -best_result.alpha * (dV_(0) + 0.5 * best_result.alpha * dV_(1));
            double cost_reduction_ratio;
            
            // Check if cost reduction is positive
            if (expected_cost_reduction > 0.0) {
                cost_reduction_ratio = (J_ - best_result.cost) / expected_cost_reduction;
            } else {
                cost_reduction_ratio = std::copysign(1.0, J_ - best_result.cost);
                if (options_.debug) {
                    std::cerr << "CDDP: Expected cost reduction is non-positive" << std::endl;
                }
            } 

            // Check if cost reduction is sufficient
            if (cost_reduction_ratio > options_.minimum_reduction_ratio) {
                forward_pass_success = true;
            } else {
                alpha_ = std::numeric_limits<double>::infinity();
                if (options_.debug) {
                    std::cerr << "CDDP: Cost reduction ratio is too small" << std::endl;
                }
            }
        }

        if (forward_pass_success) {
            X_ = best_result.state_sequence;
            U_ = best_result.control_sequence;
            dJ_ = J_ - best_result.cost;
            J_ = best_result.cost;
            alpha_ = best_result.alpha;
            // solution.lagrangian_sequence.push_back(best_result.lagrangian);
            solution.cost_sequence.push_back(J_);
            
            // Decrease regularization
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_ * (regularization_state_ > options_.regularization_state_min ? 1.0 : 0.0);
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_ * (regularization_control_ > options_.regularization_control_min ? 1.0 : 0.0);

            // Check termination
            if (dJ_ < options_.cost_tolerance) {
                solution.converged = true;
                break;
            }
        } else {
            // Increase regularization
            regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
            regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_max);
            regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
            regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_max);

            // Check regularization limit
            if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                std::cerr << "CDDP: Regularization limit reached" << std::endl;
                solution.converged = false;
                break;
            }
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, 0.0, optimality_gap_, expected_, regularization_control_, alpha_); 
        }


    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 

    // Finalize solution
    solution.state_sequence = X_;
    solution.control_sequence = U_;
    solution.alpha = alpha_;
    solution.iterations = iter;
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
    Eigen::MatrixXd Q_uu_reg(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::MatrixXd Q_ux_reg(control_dim, state_dim);
    Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);
    Eigen::SparseMatrix<double> P(control_dim, control_dim); // Hessian of QP objective
    dV_ = Eigen::Vector2d::Zero();

    // Create BoxQP solver
    cddp::BoxQPOptions qp_options;
    qp_options.verbose = false;
    qp_options.maxIter = 1000;
    // qp_options.eps_abs = 1e-3;
    // qp_options.eps_rel = 1e-2;
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
        Q_ux = l_ux + B.transpose() * V_xx * A;
        Q_uu = l_uu + B.transpose() * V_xx * B;

        // Symmetrize Q_uu for cholensky decomposition
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

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
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            eigenvalues = es.eigenvalues().real();

            if (options_.debug) {
                std::cerr << "CDDP: Q_uu is still not positive definite" << std::endl;
            }
            return false;
        }

        // Store Q-function matrices
        Q_UU_[t] = Q_uu;
        Q_UX_[t] = Q_ux;
        Q_U_[t] = Q_u;

        if (control_box_constraint == nullptr) {
            const Eigen::MatrixXd &H = Q_uu_reg.inverse();
            k = -H * Q_u;
            K = -H * Q_ux_reg;
            if (options_.debug) {
                std::cout << "No control box constraint" << std::endl;
            }
        } else {
            // Solve QP by boxQP
            Eigen::VectorXd lb = control_box_constraint->getLowerBound() - u;
            Eigen::VectorXd ub = control_box_constraint->getUpperBound() - u;
            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(control_dim); // Initial guess
            
            cddp::BoxQPResult qp_result = qp_solver.solve(Q_uu_reg, Q_u, lb, ub, x0);
            
            if (qp_result.status == BoxQPStatus::HESSIAN_NOT_PD || 
                qp_result.status == BoxQPStatus::NO_DESCENT) {
                    if (options_.debug) {
                        std::cerr << "CDDP: BoxQP failed at time step " << t << std::endl;
                    }
                return false;
            }
            
            // Extract solution
            k = qp_result.x;

            // Compute feedback gain matrix
            K = Eigen::MatrixXd::Zero(control_dim, state_dim);
            if (qp_result.free.sum() > 0) {
                // Get indices of free variables
                std::vector<int> free_idx;
                for (int i = 0; i < control_dim; i++) {
                    if (qp_result.free(i)) {
                        free_idx.push_back(i);
                    }
                }

                // Extract relevant parts of Q_ux for free variables
                Eigen::MatrixXd Q_ux_free(free_idx.size(), state_dim);
                for (size_t i = 0; i < free_idx.size(); i++) {
                    Q_ux_free.row(i) = Q_ux_reg.row(free_idx[i]);
                }

                // Compute gains for free variables using the LDLT factorization
                Eigen::MatrixXd K_free = -qp_result.Hfree.solve(Q_ux_free);

                // Put back into full K matrix
                for (size_t i = 0; i < free_idx.size(); i++) {
                    K.row(free_idx[i]) = K_free.row(i);
                }
            }
        }

        // Store feedforward and feedback gain
        k_[t] = k;
        K_[t] = K;

        // Compute value function approximation
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ = dV_ + dV_step;
        V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_XX_[t] = 0.5 * (V_XX_[t] + V_XX_[t].transpose()); // Symmetrize Hessian

        // Compute optimality gap (Inf-norm) for convergence check
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

        // TODO: Add constraint optimality gap analysis
        optimality_gap_ = Qu_error;
    }

    expected_ = dV_(0);

    if (options_.debug) {
        std::cout << "Qu_error: " << Qu_error << std::endl;
        std::cout << "dV: " << dV_.transpose() << std::endl;
    }
    
    return true;
}

ForwardPassResult CDDP::solveCLDDPForwardPass(double alpha) {
    ForwardPassResult result;
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Initialize trajectories
    std::vector<Eigen::VectorXd> X_new = X_;
    std::vector<Eigen::VectorXd> U_new = U_;
    X_new[0] = initial_state_;
    double J_new = 0.0;

    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    for (int t = 0; t < horizon_; ++t) {
        const Eigen::VectorXd& x = X_new[t];
        const Eigen::VectorXd& u = U_new[t];
        const Eigen::VectorXd& delta_x = x - X_[t];

        U_new[t] = u + alpha * k_[t] + K_[t] * delta_x;

        if (control_box_constraint != nullptr) {
            U_new[t] = control_box_constraint->clamp(U_new[t]);
        }

        J_new += objective_->running_cost(x, U_new[t], t);
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
    result.alpha = alpha;

    return result;
}


} // namespace cddp