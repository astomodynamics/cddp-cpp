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
#include <future>    // For multi-threading
#include <thread>    // For multi-threading

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/helper.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/qp_solver.hpp"

namespace cddp
{

// Solve the problem using Augmented Lagrangian DDP (ALDDP)
CDDPSolution CDDP::solveALDDP()
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
    solution.cost_sequence.reserve(options_.max_iterations);
    solution.lagrangian_sequence.reserve(options_.max_iterations);
    solution.cost_sequence.push_back(J_);

    // Initialize dual variables and penalty parameter for Augmented Lagrangian
    std::map<std::string, std::vector<Eigen::VectorXd>> lambda_; // Dual variables (Lagrange multipliers)
    double rho = options_.mu_initial; // Penalty parameter (start with initial barrier coefficient)
    
    // Initialize dual variables for each constraint
    for (const auto& constraint_pair : constraint_set_) {
        const std::string& constraint_name = constraint_pair.first;
        const auto& constraint = constraint_pair.second;
        int dual_dim = constraint->getDualDim();
        
        lambda_[constraint_name].resize(horizon_);
        for (int t = 0; t < horizon_; ++t) {
            lambda_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        }
    }

    // Initialize constraint violation
    constraint_violation_ = computeConstraintViolation(X_, U_);
    
    // Evaluate Lagrangian (initially same as cost)
    L_ = J_;
    solution.lagrangian_sequence.push_back(L_);

    if (options_.verbose) {
        printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, rho, constraint_violation_);
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;
    int al_iter = 0; // Augmented Lagrangian iteration counter
    double max_constraint_violation = constraint_violation_;

    // Main loop of ALDDP - ALTRO approach
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
                    std::cerr << "ALDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                }
                break;
            }
        }

        // 1. Backward pass: Solve Riccati recursion to compute optimal control law
        bool backward_pass_success = false;
        while (!backward_pass_success)
        {
            backward_pass_success = solveALDDPBackwardPass(lambda_, rho);

            if (!backward_pass_success) {
                if (options_.debug) {
                    std::cerr << "ALDDP: Backward pass failed" << std::endl;
                }

                // Increase regularization
                increaseRegularization();

                if (isRegularizationLimitReached()) {
                    if (options_.verbose) {
                        std::cerr << "ALDDP: Backward pass regularization limit reached" << std::endl;
                    }
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }

        // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
        ForwardPassResult best_result;
        best_result.cost = std::numeric_limits<double>::infinity();
        best_result.lagrangian = std::numeric_limits<double>::infinity();
        bool forward_pass_success = false;

        // Try different step sizes
        for (double alpha : alphas_) {
            ForwardPassResult result = solveALDDPForwardPass(alpha, lambda_, rho);
            
            if (result.success && result.cost < best_result.cost) {
                best_result = result;
                forward_pass_success = true;
                
                // Early termination for successful forward pass
                if (result.success) {
                    if (options_.debug) {
                        std::cout << "ALDDP: Early termination due to successful forward pass" << std::endl;
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
                std::cout << "Constraint violation: " << best_result.constraint_violation << std::endl;
            }
            X_ = best_result.state_sequence;
            U_ = best_result.control_sequence;
            G_ = best_result.constraint_sequence;
            dJ_ = J_ - best_result.cost;
            J_ = best_result.cost;
            dL_ = L_ - best_result.lagrangian;
            L_ = best_result.lagrangian;
            alpha_ = best_result.alpha;
            constraint_violation_ = best_result.constraint_violation;
            solution.cost_sequence.push_back(J_);
            solution.lagrangian_sequence.push_back(L_);

            // Decrease regularization
            decreaseRegularization();

            // Augmented Lagrangian parameter updates
            // Check if inner loop has converged
            bool inner_converged = (dJ_ < options_.cost_tolerance) && (optimality_gap_ < options_.grad_tolerance);
            
            // Update Lagrange multipliers and penalty parameter based on ALTRO approach
            if (inner_converged || (iter - al_iter > 5)) {
                // Update AL iteration counter
                al_iter = iter;
                
                // Update Lagrange multipliers
                for (const auto& constraint_pair : constraint_set_) {
                    const std::string& constraint_name = constraint_pair.first;
                    const auto& constraint = constraint_pair.second;
                    
                    for (int t = 0; t < horizon_; ++t) {
                        // Update multipliers: λ = λ + ρ*c(x,u)
                        lambda_[constraint_name][t] += rho * G_[constraint_name][t];
                    }
                }
                
                // Update penalty parameter based on constraint violation
                if (constraint_violation_ > options_.constraint_tolerance) {
                    if (constraint_violation_ > max_constraint_violation * options_.mu_reduction_ratio) {
                        // If violation didn't decrease enough, increase penalty
                        rho = std::min(rho * (1.0/options_.mu_reduction_ratio), options_.mu_max);
                        if (options_.debug) {
                            std::cout << "Increasing penalty to: " << rho << std::endl;
                        }
                    }
                    max_constraint_violation = constraint_violation_;
                }
                
                // Reset cost improvement to ensure we do multiple iterations with new parameters
                dJ_ = std::numeric_limits<double>::infinity();
            }
        } else {
            // Increase regularization if no successful forward pass
            increaseRegularization();
            
            // Check if regularization limit is reached
            if (isRegularizationLimitReached()) {
                if ((dJ_ < options_.cost_tolerance * 1e2) ||
                    (optimality_gap_ < options_.grad_tolerance * 1e1)) 
                {
                    solution.converged = true;
                } else {
                    // We are forced to large regularization but still not near local min
                    solution.converged = false;
                }
                if (options_.verbose) {
                    std::cerr << "ALDDP: Regularization limit reached. "
                            << (solution.converged ? "Treating as converged." : "Not converged.") 
                            << std::endl;
                }
                break;
            }
        }

        // Check convergence of the overall problem
        if (dJ_ < options_.cost_tolerance && 
            constraint_violation_ < options_.constraint_tolerance && 
            optimality_gap_ < options_.grad_tolerance) {
            solution.converged = true;
            break;
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, rho, constraint_violation_);
        }
    }

    // Finalize solution
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    solution.solve_time = duration.count() * 1e-6; // convert to seconds
    
    solution.state_sequence = X_;
    solution.control_sequence = U_;
    
    // Store control gains
    solution.control_gain = K_u_;
    
    if (options_.header_and_footer && options_.verbose) {
        printSolution(solution);
    }
    
    return solution;
}

// Backward pass of ALDDP
bool CDDP::solveALDDPBackwardPass(const std::map<std::string, std::vector<Eigen::VectorXd>>& lambda, double rho)
{
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    
    // Resize feedback gain matrices
    k_u_.resize(horizon_);
    K_u_.resize(horizon_);
    
    // Initialize value function approximation at terminal state
    Eigen::VectorXd V_x = Eigen::VectorXd::Zero(state_dim);
    Eigen::MatrixXd V_xx = Eigen::MatrixXd::Zero(state_dim, state_dim);
    
    // Get terminal cost derivatives
    objective_->getTerminalDerivatives(X_[horizon_], V_x, V_xx);
    
    // Add regularization to terminal cost Hessian
    if (regularization_state_ > 0) {
        V_xx += regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim);
    }
    
    // Working variables
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
    
    // Store optimal policy and Q matrices for line search
    Q_UU_.resize(horizon_);
    Q_UX_.resize(horizon_);
    Q_U_.resize(horizon_);
    
    // Backward pass from N-1 to 0
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get dynamics and cost derivatives
        system_->linearize(X_[t], U_[t], A, B);
        objective_->getDerivatives(t, X_[t], U_[t], Q_x, Q_u, Q_xx, Q_uu, Q_ux);
        
        // Add augmented Lagrangian terms for each constraint
        for (const auto& constraint_pair : constraint_set_) {
            const std::string& constraint_name = constraint_pair.first;
            const auto& constraint = constraint_pair.second;
            const int dual_dim = constraint->getDualDim();
            
            // Get constraint Jacobians and constraint values
            Eigen::MatrixXd C_x, C_u;
            Eigen::VectorXd c;
            constraint->getDerivatives(X_[t], U_[t], C_x, C_u, c);
            
            // Augmented Lagrangian contribution to derivatives
            // For constraint c(x,u) with Lagrange multiplier λ and penalty parameter ρ
            // L_AL = λ^T c(x,u) + (ρ/2)||c(x,u)||^2
            
            // First-order derivatives
            Q_x += C_x.transpose() * lambda.at(constraint_name)[t];
            Q_u += C_u.transpose() * lambda.at(constraint_name)[t];
            
            // Add penalty term: (ρ/2)||c(x,u)||^2
            Q_x += rho * C_x.transpose() * c;
            Q_u += rho * C_u.transpose() * c;
            
            // Second-order derivatives (approximation)
            Q_xx += rho * C_x.transpose() * C_x;
            Q_uu += rho * C_u.transpose() * C_u;
            Q_ux += rho * C_u.transpose() * C_x;
        }
        
        // Q-terms of the Q-function approximation
        Q_x += A.transpose() * V_x;
        Q_u += B.transpose() * V_x;
        Q_xx += A.transpose() * V_xx * A;
        Q_ux += B.transpose() * V_xx * A;
        Q_uu += B.transpose() * V_xx * B;
        
        // Add regularization to ensure Q_uu is positive definite
        Q_uu_reg = Q_uu;
        Q_ux_reg = Q_ux;
        
        if (regularization_control_ > 0) {
            Q_uu_reg += regularization_control_ * Eigen::MatrixXd::Identity(control_dim, control_dim);
        }
        
        // Check if Q_uu_reg is positive definite and invert
        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
        if (llt.info() == Eigen::NumericalIssue) {
            // Not positive definite - try EigenSolver approach
            Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
            Eigen::VectorXd eigenvalues = es.eigenvalues().real();
            double min_eig = eigenvalues.minCoeff();
            
            if (min_eig < 1e-6) {
                if (options_.debug) {
                    std::cerr << "ALDDP: Q_uu not positive definite at time " << t << ", min eigenvalue: " << min_eig << std::endl;
                }
                return false; // Backward pass failed
            }
        }
        
        // Compute inverse (Q_uu_reg is guaranteed to be positive definite now)
        Q_uu_inv = Q_uu_reg.ldlt().solve(Eigen::MatrixXd::Identity(control_dim, control_dim));
        
        // Feedback and feedforward terms
        k = -Q_uu_inv * Q_u;
        K = -Q_uu_inv * Q_ux_reg;
        
        // Store gains for forward pass
        k_u_[t] = k;
        K_u_[t] = K;
        
        // Store Q matrices for line search
        Q_UU_[t] = Q_uu_reg;
        Q_UX_[t] = Q_ux_reg;
        Q_U_[t] = Q_u;
        
        // Update value function for next iteration
        V_x = Q_x + K.transpose() * Q_u + Q_ux.transpose() * k + K.transpose() * Q_uu * k;
        V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        
        // Add regularization to value function Hessian
        if (regularization_state_ > 0) {
            V_xx += regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim);
        }
        
        // Ensure V_xx is symmetric for numerical stability
        V_xx = 0.5 * (V_xx + V_xx.transpose());
    }
    
    // Calculate expected cost reduction (for convergence check)
    dV_ = Eigen::VectorXd(2);
    dV_(0) = 0.0;
    dV_(1) = 0.0;
    
    for (int t = 0; t < horizon_; ++t) {
        dV_(0) += k_u_[t].dot(Q_U_[t]);
        dV_(1) += 0.5 * k_u_[t].dot(Q_UU_[t] * k_u_[t]);
    }
    
    // Expected improvement should be positive for a valid backward pass
    optimality_gap_ = std::abs(dV_(0));
    
    return true;
}

// Forward pass of ALDDP
ForwardPassResult CDDP::solveALDDPForwardPass(double alpha, const std::map<std::string, std::vector<Eigen::VectorXd>>& lambda, double rho)
{
    ForwardPassResult result;
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    
    // Initialize state and control sequences
    result.state_sequence.resize(horizon_ + 1);
    result.control_sequence.resize(horizon_);
    result.state_sequence[0] = X_[0]; // Start with current initial state
    
    // Initialize constraint results for each constraint
    for (const auto& constraint_pair : constraint_set_) {
        const std::string& constraint_name = constraint_pair.first;
        const auto& constraint = constraint_pair.second;
        int dual_dim = constraint->getDualDim();
        
        result.constraint_sequence[constraint_name].resize(horizon_);
    }
    
    // Forward simulation with new control policy
    double new_cost = 0.0;
    double new_lagrangian = 0.0;
    double constraint_violation = 0.0;
    
    for (int t = 0; t < horizon_; ++t) {
        // Compute control deviation using feedback policy
        Eigen::VectorXd state_diff = result.state_sequence[t] - X_[t];
        result.control_sequence[t] = U_[t] + alpha * k_u_[t] + K_u_[t] * state_diff;
        
        // Evaluate constraints
        for (const auto& constraint_pair : constraint_set_) {
            const std::string& constraint_name = constraint_pair.first;
            const auto& constraint = constraint_pair.second;
            
            // Evaluate constraint
            Eigen::VectorXd c = constraint->evaluate(result.state_sequence[t], result.control_sequence[t]);
            result.constraint_sequence[constraint_name][t] = c;
            
            // Compute constraint violation norm
            constraint_violation += c.squaredNorm();
        }
        
        // Forward simulate dynamics
        Eigen::VectorXd next_state;
        system_->step(result.state_sequence[t], result.control_sequence[t], next_state);
        result.state_sequence[t + 1] = next_state;
        
        // Accumulate stage cost
        new_cost += objective_->evaluateRunningCost(t, result.state_sequence[t], result.control_sequence[t]);
    }
    
    // Add terminal cost
    new_cost += objective_->evaluateTerminalCost(result.state_sequence[horizon_]);
    
    // Calculate constraint violation
    constraint_violation = std::sqrt(constraint_violation);
    result.constraint_violation = constraint_violation;
    
    // Calculate Lagrangian
    new_lagrangian = new_cost;
    
    // Add augmented Lagrangian terms
    for (int t = 0; t < horizon_; ++t) {
        for (const auto& constraint_pair : constraint_set_) {
            const std::string& constraint_name = constraint_pair.first;
            const Eigen::VectorXd& c = result.constraint_sequence[constraint_name][t];
            
            // Add λ^T c(x,u) term
            new_lagrangian += lambda.at(constraint_name)[t].dot(c);
            
            // Add (ρ/2)||c(x,u)||^2 term
            new_lagrangian += 0.5 * rho * c.squaredNorm();
        }
    }
    
    // Check if the forward pass is successful
    double expected_improvement = alpha * dV_(0) + alpha * alpha * dV_(1);
    double actual_improvement = J_ - new_cost;
    double improvement_ratio = 0.0;
    
    if (std::abs(expected_improvement) > 1e-10) {
        improvement_ratio = actual_improvement / expected_improvement;
    }
    
    // A successful step should have improvement_ratio > minimum_reduction_ratio
    result.success = (improvement_ratio > options_.minimum_reduction_ratio);
    result.cost = new_cost;
    result.lagrangian = new_lagrangian;
    result.alpha = alpha;
    
    return result;
}
}
