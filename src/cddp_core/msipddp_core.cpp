/*
 Copyright 2025 Tomo Sasaki

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

namespace cddp 
{
    void CDDP::initializeMSIPDDP() 
    {
        if (!system_)
        {
            initialized_ = false;
            if (options_.verbose)
            {
                std::cerr << "MSIPDDP::initializeMSIPDDP: No dynamical system provided." << std::endl; // Updated message
            }
            return;
        }

        if (!objective_)
        {
            initialized_ = false;
            if (options_.verbose)
            {
                std::cerr << "MSIPDDP::initializeMSIPDDP: No objective function provided." << std::endl; // Updated message
            }
            return;
        }

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();

        // Check if reference_state in objective and reference_state in IPDDP are the same
        if ((reference_state_ - objective_->getReferenceState()).norm() > 1e-6) 
        {
            std::cerr << "MSIPDDP: Initial state and goal state in the objective function do not match" << std::endl;
            throw std::runtime_error("Initial state and goal state in the objective function do not match");
        }

        // Initialize trajectories (X_ and U_ are std::vectors of Eigen::VectorXd)
        if (X_.size() != horizon_ + 1 && U_.size() != horizon_)
        {
            X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim)); 
            U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));

            // Create X_ initial guess using initial_state and reference_state by interpolating between them
            for (int t = 0; t < horizon_ + 1; ++t)
            {
                X_[t] = initial_state_ + t * (reference_state_ - initial_state_) / horizon_;
            }
        }
        else if (X_.size() != horizon_ + 1)
        {
            X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
        }
        else if (U_.size() != horizon_)
        {
            U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
        }

        Lambda_.resize(horizon_, Eigen::VectorXd::Zero(state_dim));
        for (int t = 0; t < horizon_; ++t)
        {
            Lambda_[t] = 0.1 * Eigen::VectorXd::Ones(state_dim);
        }

        k_u_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
        K_u_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
        k_x_.resize(horizon_, Eigen::VectorXd::Zero(state_dim));
        K_x_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, state_dim));
        k_lambda_.resize(horizon_, Eigen::VectorXd::Zero(state_dim));
        K_lambda_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, state_dim));

        G_.clear(); // Constraint value
        Y_.clear(); // Dual variable for path constraints
        S_.clear(); // Slack variable for path constraints
        k_y_.clear();
        K_y_.clear();
        k_s_.clear();
        K_s_.clear();

        for (const auto &constraint : constraint_set_)
        {
            std::string constraint_name = constraint.first;
            int dual_dim = constraint.second->getDualDim();

            G_[constraint_name].resize(horizon_);
            Y_[constraint_name].resize(horizon_);
            S_[constraint_name].resize(horizon_);
            k_y_[constraint_name].resize(horizon_);
            K_y_[constraint_name].resize(horizon_);
            k_s_[constraint_name].resize(horizon_);
            K_s_[constraint_name].resize(horizon_);

            for (int t = 0; t < horizon_; ++t)
            {
                G_[constraint_name][t] = constraint.second->evaluate(X_[t], U_[t]); 
                Y_[constraint_name][t] = options_.dual_scale * Eigen::VectorXd::Ones(dual_dim);
                S_[constraint_name][t] = options_.slack_scale * Eigen::VectorXd::Ones(dual_dim);

                // Gains set to zero.
                k_y_[constraint_name][t].setZero(dual_dim);
                K_y_[constraint_name][t].setZero(dual_dim, state_dim);
                k_s_[constraint_name][t].setZero(dual_dim);
                K_s_[constraint_name][t].setZero(dual_dim, state_dim);
            }
        }

        // Initialize cost
        J_ = objective_->evaluate(X_, U_); 

        // Initialize line search parameters
        alphas_.clear();
        alpha_ = options_.backtracking_coeff;
        for (int i = 0; i < options_.max_line_search_iterations; ++i)
        {
            alphas_.push_back(alpha_);
            alpha_ *= options_.backtracking_factor;
        }
        alpha_ = options_.backtracking_coeff;
        dV_.resize(2); // Cost improvement

        // Initialize regularization parameters
        if (options_.regularization_type == "state" || options_.regularization_type == "both")
        {
            regularization_state_ = options_.regularization_state;
            regularization_state_step_ = options_.regularization_state_step;
        }
        else
        {
            regularization_state_ = 0.0;
            regularization_state_step_ = 1.0;
        }

        if (options_.regularization_type == "control" || options_.regularization_type == "both")
        {
            regularization_control_ = options_.regularization_control;
            regularization_control_step_ = options_.regularization_control_step;
        }
        else
        {
            regularization_control_ = 0.0;
            regularization_control_step_ = 1.0;
        }

        if (constraint_set_.empty()) // Check if *path* constraints are empty
        {
            mu_ = 1e-8;
        }
        else
        {
            mu_ = options_.barrier_coeff;
        }

        // Now initialized
        initialized_ = true;
    }
    
    CDDPSolution CDDP::solveMSIPDDP() 
    {
        // Initialize if not done or if Lambda_ is not defined
        if (!initialized_ || Lambda_.empty())
        {
            initializeMSIPDDP(); 
        }

        if (!initialized_)
        {
             // Updated message
            std::cerr << "MSIPDDP: Initialization failed" << std::endl;
            throw std::runtime_error("MSIPDDP: Initialization failed");
        }

        // Prepare solution struct
        CDDPSolution solution;
        solution.converged = false;
        solution.alpha = alpha_;
        solution.iterations = 0;
        solution.solve_time = 0.0;
        solution.time_sequence.reserve(horizon_ + 1);
        for (int t = 0; t <= horizon_; ++t)
        {
            solution.time_sequence.push_back(timestep_ * t);
        }
        solution.control_sequence.reserve(horizon_);
        solution.state_sequence.reserve(horizon_ + 1); // Stores initial states of intervals
        solution.cost_sequence.reserve(options_.max_iterations);
        solution.lagrangian_sequence.reserve(options_.max_iterations);

        // Initialize trajectories and gaps
        initialMSIPDDPRollout(); // J_ is computed inside this function
        solution.cost_sequence.push_back(J_);

        // Reset MSIPDDP filter 
        resetMSIPDDPFilter(); // L_ is computed inside this function
        solution.lagrangian_sequence.push_back(L_);

        // Reset regularization
        resetMSIPDDPRegularization();

        if (options_.verbose)
        {
            // TODO: Update printIteration to include gap violation norm
            printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); // Initial iteration information
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        ForwardPassResult best_result; 
        ipddp_regularization_counter_ = 0; // Reset regularization counter

        // Main loop of MSIPDDP
        while (iter < options_.max_iterations)
        {
            ++iter;
            solution.iterations = iter;

            // Check maximum CPU time
            if (options_.max_cpu_time > 0)
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                if (duration.count() * 1e-6 > options_.max_cpu_time)
                {
                    if (options_.debug)
                    {
                        std::cerr << "MSCDDP: Maximum CPU time reached. Returning current solution" << std::endl; // Updated message
                    }
                    // TODO: Treat as convergence
                    solution.converged = true;
                    break;
                }
            }

            // 1. Backward pass: Solve Riccati recursion for MSIPDDP
            bool backward_pass_success = false;
            while (!backward_pass_success)
            {
                backward_pass_success = solveMSIPDDPBackwardPass(); 

                if (!backward_pass_success)
                {
                    if (options_.debug)
                    {
                        std::cerr << "MSIPDDP: Backward pass failed" << std::endl; 
                    }

                    // Increase regularization
                    increaseRegularization(); 

                    if (isRegularizationLimitReached())
                    {
                        if (options_.verbose)
                        {
                            std::cerr << "MSIPDDP: Backward pass regularization limit reached!" << std::endl; 
                        }
                        // TODO: Treat as convergence 
                        solution.converged = true;
                        break; // Exit if regularization limit reached
                    }
                    continue; // Continue if backward pass fails
                }
            }
            
            // Check if already converged due to regularization limit in backward pass
            if (solution.converged) {
                break;
            }


            // 2. Forward pass (either single-threaded or multi-threaded) for MSIPDDP
            best_result.success = false;
            best_result.cost = std::numeric_limits<double>::infinity();
            best_result.lagrangian = std::numeric_limits<double>::infinity();
            // TODO: Add gap violation to result struct
            best_result.constraint_violation = 0.0; // Path constraint violation
            // double gap_violation = 0.0; // Gap violation
            
            bool forward_pass_success = false;

            if (!options_.use_parallel)
            {
                // Single-threaded execution with early termination
                for (double alpha : alphas_)
                {
                    ForwardPassResult result = solveMSIPDDPForwardPass(alpha); 

                    if (result.success) // Success criteria might differ for MS
                    {
                        best_result = result;
                        forward_pass_success = true;

                        // Check for early termination
                        if (result.success) 
                        {
                            break;
                        }
                    }
                }
            }
            else
            {
                // Multi-threaded execution
                std::vector<std::future<ForwardPassResult>> futures;
                futures.reserve(alphas_.size());

                // Launch all forward passes in parallel
                for (double alpha : alphas_)
                {
                    futures.push_back(std::async(std::launch::async,
                                                 [this, alpha]()
                                                 {
                                                    return solveMSIPDDPForwardPass(alpha); 
                                                 }));
                }

                // Collect results from all threads
                for (auto &future : futures)
                {
                    try
                    {
                        if (future.valid())
                        {
                            ForwardPassResult result = future.get();
                            if (result.success && result.lagrangian < best_result.lagrangian) 
                            {
                                best_result = result;
                                forward_pass_success = true;
                            }
                        }
                    }
                    catch (const std::exception &e)
                    {
                        if (options_.debug)
                        {
                             // Updated message
                            std::cerr << "MSCDDP: Forward pass thread failed: " << e.what() << std::endl;
                        }
                        continue;
                    }
                }
            }

            // Update solution if a feasible forward pass was found
            if (forward_pass_success)
            {
                if (options_.debug)
                {
                    // TODO: Update debug output for MSIPDDP
                    std::cout << "[MSIPDDP: Forward pass] " << std::endl; // Updated message
                    std::cout << "    cost: " << best_result.cost << std::endl;
                    std::cout << "    logcost: " << best_result.lagrangian << std::endl;
                    std::cout << "    alpha: " << best_result.alpha << std::endl;
                    std::cout << "    rp_err: " << best_result.constraint_violation << std::endl; // Path constraints
                    // std::cout << "    gap_err: " << best_result.gap_violation << std::endl; // Gap constraints
                }
                X_ = best_result.state_sequence; // Initial states of intervals
                U_ = best_result.control_sequence;
                Y_ = best_result.dual_sequence;    // Path constraint duals
                S_ = best_result.slack_sequence;   // Path constraint slacks
                G_ = best_result.constraint_sequence; // Path constraint values
                // TODO: Update gaps_ and gap_multipliers_ from best_result
                // gaps_ = best_result.gap_sequence;
                // gap_multipliers_ = best_result.gap_multiplier_sequence;

                dJ_ = J_ - best_result.cost;
                J_ = best_result.cost;
                dL_ = L_ - best_result.lagrangian; // Lagrangian includes path & gap terms
                L_ = best_result.lagrangian;
                alpha_ = best_result.alpha;
                // TODO: Update gap_violation_ member variable
                // gap_violation_ = best_result.gap_violation;
                constraint_violation_ = best_result.constraint_violation; // Path constraint violation

                solution.cost_sequence.push_back(J_);
                solution.lagrangian_sequence.push_back(L_);

                // Decrease regularization
                decreaseRegularization(); 
            }
            else
            {
                // Increase regularization
                increaseRegularization(); 

                if (isRegularizationLimitReached())
                {
                    if (options_.debug)
                    {
                         // Updated message
                        std::cerr << "MSCDDP: Forward Pass regularization limit reached" << std::endl;
                    }

                    // TODO: Treat as convergence
                    solution.converged = true;

                    break;
                }
            }

            // Print iteration information
            if (options_.verbose)
            {
                // TODO: Update printIteration call
                printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_/*, gap_violation_*/);
            }

            // Check termination (needs to include gap feasibility)
             // TODO: Update termination check
            if (std::max(optimality_gap_, mu_) <= options_.cost_tolerance /* && gap_violation_ <= options_.gap_tolerance */ )
            {
                solution.converged = true;
                break;
            }

            // From original IPDDP implementation - Barrier update logic
            // TODO: Review barrier update strategy for MSIPDDP
            if (optimality_gap_ <= 0.2 * mu_) 
            {   
                // if (constraint_set_.empty()) { // Still have gap constraints
                // }
                // else {
                    // Update mu based on path and gap constraints combined?
                    mu_ = std::max(options_.cost_tolerance / 10.0, std::min(0.2 * mu_, std::pow(mu_, 1.2)));
                // }
                resetMSIPDDPFilter();
                resetMSIPDDPRegularization(); 
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Finalize solution
        solution.state_sequence = X_; // Initial states of intervals
        // TODO: Add integrated full state trajectory to solution if needed
        solution.control_sequence = U_;
        // TODO: Add gap sequence to solution?
        solution.alpha = alpha_;
        solution.solve_time = duration.count(); // Time in microseconds

        if (options_.header_and_footer)
        {
            printSolution(solution); // TODO: Update printSolution if needed
        }

        return solution;
    }

    bool CDDP::solveMSIPDDPBackwardPass() 
    {
        // Setup
        // Initialize variables
        const int state_dim = getStateDim();
        const int control_dim = getControlDim();
        const int total_dual_dim = getTotalDualDim(); // Dual dim for path constraints

        // Terminal cost and derivatives (V_x, V_xx at t=N)
        Eigen::VectorXd V_x = objective_->getFinalCostGradient(X_.back());
        Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

        dV_ = Eigen::Vector2d::Zero();
        double Qu_err = 0.0;
        double rp_err = 0.0; // primal feasibility for path constraints
        double rd_err = 0.0; // dual feasibility for path constraints
        double rf_err = 0.0; // gap feasibility for defect constraints 

        // If no path constraints, use simpler recursion (still needs gap handling)
        if (constraint_set_.empty()) 
        {
             for (int t = horizon_ - 1; t >= 0; --t)
            {
                const Eigen::VectorXd &x = X_[t]; // Initial state of interval t
                const Eigen::VectorXd &u = U_[t]; // Control for interval t
                const Eigen::VectorXd &lambda = Lambda_[t]; // Costate for interval t
                const Eigen::VectorXd &d = system_->getDiscreteDynamics(x, u) - X_[t+1]; // Defect

                // Dynamics Jacobians at (x_t, u_t)
                const auto [Fx, Fu] = system_->getJacobians(x, u);
                // Discretized Jacobians (assuming Euler integration for simplicity)
                Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx; 
                Eigen::MatrixXd B = timestep_ * Fu;

                // Get dynamics hessians if not using iLQR
                std::vector<Eigen::MatrixXd> Fxx, Fuu, Fux;
                if (!options_.is_ilqr) {
                    const auto hessians = system_->getHessians(x, u);
                    Fxx = std::get<0>(hessians);
                    Fuu = std::get<1>(hessians);
                    Fux = std::get<2>(hessians);
                }

                // Cost derivatives at (x_t, u_t)
                double l = objective_->running_cost(x, u, t);
                auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

                Eigen::VectorXd Q_x = l_x + A.transpose() * lambda;
                Eigen::VectorXd Q_u = l_u + B.transpose() * lambda;

                Eigen::MatrixXd Q_xx = l_xx;
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_xx += timestep_ * lambda(i) * Fxx[i];
                    }
                }

                Eigen::MatrixXd Q_ux = l_ux;
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_ux += timestep_ * lambda(i) * Fux[i];
                    }
                }

                Eigen::MatrixXd Q_uu = l_uu;
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_uu += timestep_ * lambda(i) * Fuu[i];
                    }
                }

                Eigen::VectorXd Q_w = - lambda + V_x;
                Eigen::MatrixXd Q_ww = V_xx;

                // Regularize Q_uu and Q_ux
                Eigen::MatrixXd Q_ux_reg = Q_ux;
                Eigen::MatrixXd Q_uu_reg = Q_uu;
            

                // Form KKT matrix H = [ Q_uu    0    f_u^T ]
                //                      [ 0    Q_ww    -I    ]
                //                      [ f_u     -I     - rho * I    ]
                Eigen::MatrixXd H_kkt(control_dim + state_dim + state_dim, control_dim + state_dim + state_dim);
                H_kkt.setZero(); // Initialize with zeros
                // Add regularization to Q_uu
                Q_uu_reg += regularization_control_ * Eigen::MatrixXd::Identity(control_dim, control_dim);
                
                // Construct the KKT matrix for MSIPDDP
                // Block (0,0): Q_uu_reg
                H_kkt.block(0, 0, control_dim, control_dim) = Q_uu_reg;
                // Block (0,1): Zero (already zero)
                // Block (0,2): B^T
                H_kkt.block(0, control_dim + state_dim, control_dim, state_dim) = B.transpose();
                // Block (1,0): Zero (already zero)
                // Block (1,1): Q_ww
                H_kkt.block(control_dim, control_dim, state_dim, state_dim) = Q_ww;
                // Block (1,2): -I
                H_kkt.block(control_dim, control_dim + state_dim, state_dim, state_dim) = -Eigen::MatrixXd::Identity(state_dim, state_dim);
                // Block (2,0): B
                H_kkt.block(control_dim + state_dim, 0, state_dim, control_dim) = B;
                // Block (2,1): -I
                H_kkt.block(control_dim + state_dim, control_dim, state_dim, state_dim) = -Eigen::MatrixXd::Identity(state_dim, state_dim);
                // Block (2,2): - rho * I
                H_kkt.block(control_dim + state_dim, control_dim + state_dim, state_dim, state_dim) = - regularization_control_ * Eigen::MatrixXd::Identity(state_dim, state_dim);

                // std::cout << "H_kkt: " << std::endl << H_kkt << std::endl;

                // Form RHS for k = [k_u; k_lambda]: G = [Q_u; 0]
                Eigen::VectorXd G_rhs(control_dim + state_dim + state_dim);
                G_rhs.head(control_dim) = Q_u; 
                G_rhs.segment(control_dim, state_dim) = Q_w;
                G_rhs.tail(state_dim) = d;

                // Form RHS for K = [K_u; K_lambda]: J = [Q_ux; A]
                Eigen::MatrixXd J_rhs(control_dim + state_dim + state_dim, state_dim);
                J_rhs.topRows(control_dim) = Q_ux_reg; // Use regularized Q_ux
                J_rhs.block(control_dim, 0, state_dim, state_dim) = Eigen::MatrixXd::Zero(state_dim, state_dim);
                J_rhs.bottomRows(state_dim) = A;

                // Solve H * k = -G and H * K = -J using LDLT
                Eigen::LDLT<Eigen::MatrixXd> ldlt_H(H_kkt);
                if (ldlt_H.info() != Eigen::Success) {
                    if(options_.debug) { std::cerr << "MSIPDDP: KKT LDLT factorization failed at t=" << t << std::endl; }
                    // Consider increasing regularization here before returning false
                    return false; 
                }

                Eigen::VectorXd k = ldlt_H.solve(-G_rhs);
                Eigen::MatrixXd K_sol = ldlt_H.solve(-J_rhs);

                // Check solve status (important after solve call)
                if (ldlt_H.info() != Eigen::Success) { 
                    if(options_.debug) { std::cerr << "MSIPDDP: KKT solve failed at t=" << t << std::endl; }
                     // Consider increasing regularization here before returning false
                    return false;
                }

                // Extract control gains
                Eigen::VectorXd k_u = k.head(control_dim);
                Eigen::MatrixXd K_u = K_sol.topRows(control_dim);
                
                // Store gains
                k_u_[t] = k_u;
                K_u_[t] = K_u;
                k_x_[t] = k.segment(control_dim, state_dim);
                K_x_[t] = K_sol.block(control_dim, 0, state_dim, state_dim);
                k_lambda_[t] = k.segment(control_dim + state_dim, state_dim);
                K_lambda_[t] = K_sol.block(control_dim + state_dim, 0, state_dim, state_dim);

                // Update Value function derivatives 
                V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
                V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;
                // Symmetrize V_xx
                V_xx = 0.5 * (V_xx + V_xx.transpose());

                // --- MSIPDDP Modification Ends Here (for unconstrained case) ---

                // Accumulate cost improvement
                dV_[0] += k_u.dot(Q_u);
                dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

                // Error tracking (Qu_err for control stationarity)
                Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
                
                // TODO: Calculate gap error contribution if needed for optimality_gap_
                // Eigen::VectorXd current_gap = X_[t+1] - system_->getDiscreteDynamics(x, u);
                // gap_err = std::max(gap_err, current_gap.lpNorm<Eigen::Infinity>());

            } // end for t (unconstrained)

            // TODO: Update optimality gap calculation for MSIPDDP
            optimality_gap_ = Qu_err; // Needs to include gap feasibility term
            return true;
        }
        else // Has path constraints
        {
            // TODO: Implement backward pass for MSIPDDP *with* path constraints
            // This combines the IPDDP logic (handling y, s) with MS logic (handling gaps/lambda)
            
            // Placeholder: Using original IPDDP logic - NEEDS REPLACEMENT
            for (int t = horizon_ - 1; t >= 0; --t)
            {
                 // Expand cost around (x[t], u[t]) - interval start state and control
                const Eigen::VectorXd &x = X_[t]; 
                const Eigen::VectorXd &u = U_[t];

                // Continuous dynamics Jacobians
                const auto [Fx, Fu] = system_->getJacobians(x, u);

                // Get dynamics hessians if not using iLQR
                std::vector<Eigen::MatrixXd> Fxx, Fuu, Fux;
                if (!options_.is_ilqr) {
                    const auto hessians = system_->getHessians(x, u);
                    Fxx = std::get<0>(hessians);
                    Fuu = std::get<1>(hessians);
                    Fux = std::get<2>(hessians);
                }

                // Discretize (e.g., Euler)
                Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx;
                Eigen::MatrixXd B = timestep_ * Fu;

                // Gather path constraint duals (y), slacks (s), values (g)
                Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd s = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim); // g_u
                Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim); // g_x

                int offset = 0; 
                for (auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    auto &constraint = cKV.second;
                    int dual_dim = constraint->getDualDim();

                    // Variables at time t
                    const Eigen::VectorXd &y_vec = Y_[cname][t]; 
                    const Eigen::VectorXd &s_vec = S_[cname][t]; 
                    const Eigen::VectorXd &g_vec = G_[cname][t]; 
                     // Constraint Jacobians evaluated at (x_t, u_t)
                    const Eigen::MatrixXd &g_x = constraint->getStateJacobian(x, u);
                    const Eigen::MatrixXd &g_u = constraint->getControlJacobian(x, u);

                    // Insert into aggregate vectors/matrices
                    y.segment(offset, dual_dim) = y_vec;
                    s.segment(offset, dual_dim) = s_vec;
                    g.segment(offset, dual_dim) = g_vec;
                    Q_yx.block(offset, 0, dual_dim, state_dim) = g_x;
                    Q_yu.block(offset, 0, dual_dim, control_dim) = g_u;

                    offset += dual_dim;
                }

                // Cost & derivatives at (x_t, u_t)
                double l = objective_->running_cost(x, u, t);
                auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

                // --- MSIPDDP Modification Starts Here (Constrained) ---
                // Let p_{t+1} = V_x (costate passed backward from end of interval t+1)

                // Q expansions including path constraints and costate p_{t+1}
                // Q_x = l_x + Q_yx^T * y + A^T * p_{t+1} 
                // Q_u = l_u + Q_yu^T * y + B^T * p_{t+1}
                // Q_xx = l_xx + A^T * V_xx * A  // V_xx from t+1
                // Q_ux = l_ux + B^T * V_xx * A
                // Q_uu = l_uu + B^T * V_xx * B
                
                // TODO: Add Hessian terms from dynamics (Fxx, Fuu, Fux) if !is_ilqr, multiplied by p_{t+1}
                // TODO: Add Hessian terms from path constraints (g_xx, g_uu, g_ux) multiplied by y ? (Check reference)

                // Using IPDDP version as placeholder - NEEDS REPLACEMENT
                Eigen::VectorXd Q_x = l_x + Q_yx.transpose() * y + A.transpose() * V_x; 
                Eigen::VectorXd Q_u = l_u + Q_yu.transpose() * y + B.transpose() * V_x;
                
                Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_xx += timestep_ * V_x(i) * Fxx[i]; // Needs adjustment for p_{t+1}
                    }
                     // TODO: Add g_xx terms
                }
                
                Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
                 if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_ux += timestep_ * V_x(i) * Fux[i]; // Needs adjustment for p_{t+1}
                    }
                    // TODO: Add g_ux terms
                }
                
                Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;
                 if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_uu += timestep_ * V_x(i) * Fuu[i]; // Needs adjustment for p_{t+1}
                    }
                    // TODO: Add g_uu terms
                }

                // Path constraint related matrices
                Eigen::MatrixXd Y = y.asDiagonal();  
                Eigen::MatrixXd S = s.asDiagonal();  
                // Eigen::MatrixXd G = g.asDiagonal(); // Not typically needed directly
                Eigen::MatrixXd S_inv = S.inverse(); 
                Eigen::MatrixXd YSinv = Y * S_inv;   

                // Path constraint residuals
                Eigen::VectorXd r_p = g + s; // Primal feasibility for path constraints
                Eigen::VectorXd r_d = y.cwiseProduct(s).array() - mu_; // Dual feasibility for path constraints
                Eigen::VectorXd rhat = y.cwiseProduct(r_p) - r_d; // Combined residual term for path constraints

                // Regularization (Applied similarly to MS Q-functions)
                Eigen::MatrixXd Q_ux_reg = Q_ux;
                Eigen::MatrixXd Q_uu_reg = Q_uu;

                if (options_.regularization_type == "state" ||
                    options_.regularization_type == "both")
                {
                    Eigen::MatrixXd V_xx_reg = V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim); // V_xx from t+1
                    // Recompute Q_ux_reg, Q_uu_reg using V_xx_reg
                    Q_ux_reg = l_ux + B.transpose() * V_xx_reg * A; // Simplified - needs constraint Hessians?
                    Q_uu_reg = l_uu + B.transpose() * V_xx_reg * B; // Simplified - needs constraint Hessians?
                    if (!options_.is_ilqr) {
                         // TODO: Add hessian terms with V_x / p_{t+1}
                    }
                }
                // (Control regularization added below)


                // --- KKT System Solve for MSIPDDP ---
                // This is the core difference. The system couples (du, dx_t) and possibly dx_{t+1} or lambda_t updates.
                // The structure depends heavily on the exact formulation (implicit vs explicit gap handling).
                
                // Using IPDDP system solve as placeholder - NEEDS REPLACEMENT
                if (options_.regularization_type == "control" ||
                    options_.regularization_type == "both")
                {
                    Q_uu_reg.diagonal().array() += regularization_control_;
                }
                 // Combine Q_uu_reg with path constraint terms
                Eigen::MatrixXd KKT_uu = Q_uu_reg + Q_yu.transpose() * YSinv * Q_yu;
                KKT_uu = 0.5 * (KKT_uu + KKT_uu.transpose()); // symmetrize

                Eigen::LLT<Eigen::MatrixXd> llt(KKT_uu);
                if (llt.info() != Eigen::Success)
                {
                    if (options_.debug)
                    {
                         // Updated message
                        std::cerr << "MSCDDP: Backward pass KKT solve failed at time " << t << std::endl;
                    }
                    return false;
                }

                // RHS combines control gradient Q_u and path constraint residual rhat
                Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
                bigRHS.col(0) = Q_u + Q_yu.transpose() * S_inv * rhat;
                 // Matrix M combines Q_ux and path constraint coupling terms
                Eigen::MatrixXd M = Q_ux + Q_yu.transpose() * YSinv * Q_yx;
                for (int col = 0; col < state_dim; col++)
                {
                    bigRHS.col(col + 1) = M.col(col);
                }

                // Solve for kK = [k_u, K_u] (feedback on x_t)
                Eigen::MatrixXd R = llt.matrixU();
                Eigen::MatrixXd z = R.triangularView<Eigen::Upper>().solve(bigRHS);
                Eigen::MatrixXd kK = -R.transpose().triangularView<Eigen::Lower>().solve(z);

                // Extract gains
                Eigen::VectorXd k_u = kK.col(0); 
                Eigen::MatrixXd K_u = kK.block(0, 1, control_dim, state_dim);

                // Save gains
                k_u_[t] = k_u;
                K_u_[t] = K_u;

                // Compute gains for path constraints (y, s) - depend on k_u, K_u
                Eigen::VectorXd k_y = S_inv * (rhat + Y * Q_yu * k_u);
                Eigen::MatrixXd K_y = YSinv * (Q_yx + Q_yu * K_u);
                Eigen::VectorXd k_s = -r_p - Q_yu * k_u;
                Eigen::MatrixXd K_s = -Q_yx - Q_yu * K_u;

                // Store path constraint gains
                offset = 0;
                for (auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    int dual_dim = cKV.second->getDualDim();
                    k_y_[cname][t] = k_y.segment(offset, dual_dim);
                    K_y_[cname][t] = K_y.block(offset, 0, dual_dim, state_dim);
                    k_s_[cname][t] = k_s.segment(offset, dual_dim);
                    K_s_[cname][t] = K_s.block(offset, 0, dual_dim, state_dim);
                    offset += dual_dim;
                }

                 // TODO: Compute gains for gap multipliers (lambda) if needed
                 // k_lambda = ...
                 // K_lambda = ...

                // Update Q expansions (effective gradients/Hessians after solving for du)
                // This is primarily for calculating V_x, V_xx propagation, not directly used in solve
                // Q_u_eff = Q_u + Q_yu.transpose() * S_inv * rhat; // Effective Q_u before solve
                // Q_x_eff = Q_x + Q_yx.transpose() * S_inv * rhat; // Effective Q_x before solve
                // Q_xx_eff = Q_xx + Q_yx.transpose() * YSinv * Q_yx;
                // Q_ux_eff = Q_ux + Q_yx.transpose() * YSinv * Q_yu;
                // Q_uu_eff = Q_uu + Q_yu.transpose() * YSinv * Q_yu; // Note: This is KKT_uu without regularization

                // Update cost improvement
                // dV_[0] += k_u.dot(Q_u_eff); 
                // dV_[1] += 0.5 * k_u.dot(Q_uu_eff * k_u); // Approximation

                // Update Value function derivatives / Costate propagation for MSIPDDP
                // V_x_t = Q_x_eff + K_u^T * Q_u_eff + Q_ux_eff^T * k_u + K_u^T * Q_uu_eff * k_u
                // V_xx_t = Q_xx_eff + K_u^T * Q_ux_eff + Q_ux_eff^T * K_u + K_u^T * Q_uu_eff * K_u
                // These become the V_x, V_xx for the *next* iteration (t-1)
                
                // Using IPDDP update as placeholder - NEEDS REPLACEMENT
                Q_u += Q_yu.transpose() * S_inv * rhat; // Modify Q before using in V update
                Q_x += Q_yx.transpose() * S_inv * rhat; 
                Q_xx += Q_yx.transpose() * YSinv * Q_yx;
                Q_ux += Q_yx.transpose() * YSinv * Q_yu;
                Q_uu += Q_yu.transpose() * YSinv * Q_yu; // This is KKT_uu (ignoring regularization difference)
                
                dV_[0] += k_u.dot(Q_u); // Uses modified Q_u
                dV_[1] += 0.5 * k_u.dot(Q_uu * k_u); // Uses modified Q_uu

                V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
                V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;
                V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

                // Error tracking
                // Qu_err needs modification for MS? Stationarity w.r.t u AND x_{t+1}?
                // Using IPDDP version as placeholder
                Qu_err = std::max(Qu_err, (Q_u + Q_uu*k_u + Q_ux*Eigen::VectorXd::Zero(state_dim)).lpNorm<Eigen::Infinity>()); // Approx L_u = 0
                rp_err = std::max(rp_err, r_p.lpNorm<Eigen::Infinity>()); // Path constraint primal feasibility
                rd_err = std::max(rd_err, r_d.lpNorm<Eigen::Infinity>()); // Path constraint dual feasibility
                // TODO: Calculate gap error contribution
                // Eigen::VectorXd current_gap = X_[t+1] - system_->getDiscreteDynamics(x, u);
                // gap_err = std::max(gap_err, current_gap.lpNorm<Eigen::Infinity>());

            } // end for t (constrained)

            // Compute optimality gap (combine path, gap, and stationarity errors)
            // TODO: Update optimality gap calculation
            optimality_gap_ = std::max(Qu_err, std::max(rp_err, rd_err)); 

            if (options_.debug)
            {
                 // TODO: Update debug output for MSIPDDP
                std::cout << "[MSIPDDP Backward Pass]\n" // Updated message
                          << "    Qu_err:  " << Qu_err << "\n" // Control stationarity component
                          << "    rp_err:  " << rp_err << "\n" // Path constraint primal feasibility
                          << "    rd_err:  " << rd_err << "\n" // Path constraint dual feasibility
                          // << "    gap_err: " << gap_err << "\n" // Gap feasibility
                          << "    dV:      " << dV_.transpose() << std::endl;
            }
            return true;
        }
        // return false; // Should not be reached if logic is correct
    } // end solveMSIPDDPBackwardPass


     // TODO: Rename solveIPDDPForwardPass -> solveMSIPDDPForwardPass
    ForwardPassResult CDDP::solveMSIPDDPForwardPass(double alpha) 
    {
        // Prepare result structure 
        ForwardPassResult result; // TODO: Potentially MSForwardPassResult
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.lagrangian = std::numeric_limits<double>::infinity();
        // TODO: Add gap violation field
        // result.gap_violation = std::numeric_limits<double>::infinity();
        result.alpha = alpha;

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();
        const int dual_dim = getTotalDualDim(); // Path constraint dual dim

        // Define tau for feasibility check
        double tau = std::max(0.99, 1.0 - mu_);

        // Copy the current (old) trajectories:
        // X_: initial states of intervals
        // U_: controls for intervals
        // Y_, S_: path constraint duals/slacks
        // G_: path constraint values
        // TODO: Copy gaps_ and gap_multipliers_
        std::vector<Eigen::VectorXd> X_new = X_; 
        std::vector<Eigen::VectorXd> U_new = U_;
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
        std::map<std::string, std::vector<Eigen::VectorXd>> G_new = G_;
        // std::vector<Eigen::VectorXd> gaps_new = gaps_; 
        // std::vector<Eigen::VectorXd> gap_multipliers_new = gap_multipliers_;


        // Set the initial state (boundary condition)
        X_new[0] = initial_state_;

        // Initialize cost accumulators and measures.
        double cost_new = 0.0;
        double log_cost_new = 0.0;
        double rp_err = 0.0; // Path constraint violation (L1 norm)
        double gap_err = 0.0; // Gap violation (L-inf norm for now)

        // The current filter point (includes path and gap violations)
        // TODO: Update filter point definition if needed
        // FilterPoint current{L_, constraint_violation_ + gap_violation_}; // Combine violations? Or 2D filter?
        cddp::FilterPoint current{L_, constraint_violation_}; // Placeholder // Qualified with cddp::

        // --- Forward pass loop for MSIPDDP ---
        // We update x_t, u_t, y_t, s_t, lambda_t simultaneously? Or sequentially?
        // Update Rule (Conceptual):
        // dx_t = x_new[t] - x_old[t]
        // du_t = alpha * k_u[t] + K_u[t] * dx_t
        // dy_t = alpha * k_y[t] + K_y[t] * dx_t
        // ds_t = alpha * k_s[t] + K_s[t] * dx_t
        // dlambda_t = alpha * k_lambda[t] + K_lambda[t] * dx_t (?)
        // dx_{t+1} = x_new[t+1] - x_old[t+1] -> This needs calculation based on du, dx_t and potentially gap feedback
        
        // Alternative: Update x_t, u_t first, then simulate dynamics to get x_{t+1} and gaps.
        
        // Let's try the simulation approach:
        // 1. Update control u_new[t] based on x_new[t]
        // 2. Update path constraint duals/slacks y_new[t], s_new[t] based on x_new[t]
        // 3. Simulate dynamics from x_new[t] with u_new[t] to get simulated_x_next
        // 4. Update the *next* interval's initial state x_new[t+1] based on some rule (e.g., using gap feedback?)
        //    OR: x_new[t+1] is updated based on dx_{t+1} = alpha*k_x[t+1] + K_x[t+1]*dx_t ??? (Needs clarification from reference)
        // 5. Calculate gap: gaps_new[t] = x_new[t+1] - simulated_x_next
        
        // Let's assume x_new[t+1] is updated using the linear feedback law from the state x_t.
        // This requires gains K_x which are typically Identity in standard DDP but might differ in MS.
        // Assume dx_{t+1} is implicitly handled or computed during backward pass?
        // For simplicity, let's stick to the standard DDP/IPDDP forward pass structure first and see where it breaks for MS.
        // The key issue is how X_new[t+1] is determined. In standard DDP, it's the result of simulation.
        // In MS, X_new[t+1] is a decision variable updated similarly to U_new[t].

        // Let's assume a simplified update for x_{t+1} (Needs verification):
        // dx_{t+1} = alpha * k_x_{t+1} + K_x_{t+1} * dx_t  (where k_x, K_x might be derived in backward pass)
        // If we don't have k_x, K_x, maybe dx_{t+1} = A*dx_t + B*du_t ? (This connects it to simulation result)
        
        // --- Forward Pass Attempt 1: Update x, u, y, s based on local feedback ---
         if (constraint_set_.empty()) // No path constraints
        {
            // Unconstrained MS forward pass
             for (int t = 0; t < horizon_; ++t)
             {
                // State difference at start of interval t
                Eigen::VectorXd dx = X_new[t] - X_[t]; 
                
                // Update control based on feedback from x_new[t]
                U_new[t] = U_[t] + alpha * k_u_[t] + K_u_[t] * dx;
                
                // Simulate dynamics from x_new[t] with u_new[t]
                Eigen::VectorXd simulated_x_next = system_->getDiscreteDynamics(X_new[t], U_new[t]);
                
                // --- How to update X_new[t+1]? ---
                // Option A: Just use the simulated value (like standard DDP) - This ignores the MS nature
                // X_new[t + 1] = simulated_x_next; 
                
                // Option B: Update X_new[t+1] using its own feedback law (if derived)?
                // Assume dx_{t+1} = A*dx_t + B*du_t (Linearized propagation of difference)
                 Eigen::VectorXd du = U_new[t] - U_[t];
                 // Need Jacobians A, B at (X_[t], U_[t])
                 const auto [Fx, Fu] = system_->getJacobians(X_[t], U_[t]);
                 Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx; 
                 Eigen::MatrixXd B = timestep_ * Fu;
                 Eigen::VectorXd dx_next = A * dx + B * du;
                 X_new[t+1] = X_[t+1] + dx_next; // Update next state based on linearized change


                // Accumulate stage cost at (x_new[t], u_new[t])
                cost_new += objective_->running_cost(X_new[t], U_new[t], t);
                
                // Calculate gap based on the updated X_new[t+1] and the simulation result
                Eigen::VectorXd current_gap = X_new[t+1] - simulated_x_next;
                gap_err = std::max(gap_err, current_gap.lpNorm<Eigen::Infinity>());

                // TODO: Update gap multipliers lambda_new[t]? Requires gains k_lambda, K_lambda.
             }
             cost_new += objective_->terminal_cost(X_new.back()); // Terminal cost at X_new[N]
             
             // Lagrangian = cost - mu * sum(log(gaps?)) -> Implicit barrier on gaps?
             // If gaps are equality constraints, they shouldn't be in log barrier.
             // Lagrangian might just be the cost if gaps are handled by multipliers.
             log_cost_new = cost_new; // Placeholder

             // Check improvement and feasibility (cost reduction + gap feasibility)
             // TODO: Define success criteria for MSIPDDP forward pass
             // Needs sufficient cost reduction AND acceptable gap violation?
             // Using DDP check as placeholder
             double dJ = J_ - cost_new;
             double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1)); // Expected cost reduction
             double reduction_ratio = expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

             // Success = Cost reduction achieved AND gap violation is acceptable
             result.success = (reduction_ratio > options_.minimum_reduction_ratio) && (gap_err <= options_.grad_tolerance); // Example criteria

             if (result.success) {
                 result.state_sequence = X_new; // Initial states of intervals
                 result.control_sequence = U_new;
                 result.cost = cost_new;
                 result.lagrangian = log_cost_new;
                 // result.gap_violation = gap_err;
                 result.constraint_violation = 0.0; // No path constraints
             }
             return result;

        } else { // Has path constraints

            // Constrained MS forward pass
             for (int t = 0; t < horizon_; ++t)
             {
                // State difference at start of interval t
                Eigen::VectorXd dx = X_new[t] - X_[t]; 

                // Update path constraint duals and slacks
                bool feasible = true;
                for (auto &ckv : constraint_set_)
                {
                    const std::string &cname = ckv.first;
                    int current_dual_dim = ckv.second->getDualDim();

                    const Eigen::VectorXd &y_old = Y_[cname][t];
                    const Eigen::VectorXd &s_old = S_[cname][t];
                    
                    // Update based on feedback from dx
                    Eigen::VectorXd y_new = y_old + alpha * k_y_[cname][t] + K_y_[cname][t] * dx;
                    Eigen::VectorXd s_new = s_old + alpha * k_s_[cname][t] + K_s_[cname][t] * dx;

                    // Enforce minimal feasibility for y, s (positivity)
                    Eigen::VectorXd y_min = (1.0 - tau) * y_old;
                    Eigen::VectorXd s_min = (1.0 - tau) * s_old;
                    for (int i = 0; i < current_dual_dim; i++)
                    {
                        if (y_new[i] < y_min[i] || s_new[i] < s_min[i])
                        {
                            feasible = false;
                            break; 
                        }
                    }
                    if (!feasible) break; 

                    // Store feasible updates
                    Y_new[cname][t] = y_new;
                    S_new[cname][t] = s_new;
                } // end loop over path constraints

                if (!feasible) {
                    // Early exit if y or s feasibility violated
                    return result; // Return default failure result
                }

                // Update control based on feedback from dx
                U_new[t] = U_[t] + alpha * k_u_[t] + K_u_[t] * dx;

                // Simulate dynamics from x_new[t] with u_new[t]
                Eigen::VectorXd simulated_x_next = system_->getDiscreteDynamics(X_new[t], U_new[t]);

                // --- How to update X_new[t+1]? --- (Using linearized update again)
                 Eigen::VectorXd du = U_new[t] - U_[t];
                 const auto [Fx, Fu] = system_->getJacobians(X_[t], U_[t]);
                 Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx; 
                 Eigen::MatrixXd B = timestep_ * Fu;
                 Eigen::VectorXd dx_next = A * dx + B * du;
                 X_new[t+1] = X_[t+1] + dx_next; 

                // Calculate gap
                Eigen::VectorXd current_gap = X_new[t+1] - simulated_x_next;
                gap_err = std::max(gap_err, current_gap.lpNorm<Eigen::Infinity>());

                 // TODO: Update gap multipliers lambda_new[t]?

             } // end for t (constrained forward pass)

             // --- Compute Cost and Path Constraint Violations ---
             cost_new = 0.0;
             log_cost_new = 0.0;
             rp_err = 0.0; // Path constraint violation (L1 norm)
             
             for (int t = 0; t < horizon_; ++t)
             {
                // Stage cost at (x_new[t], u_new[t])
                cost_new += objective_->running_cost(X_new[t], U_new[t], t);

                // Evaluate path constraints and update log barrier / primal error
                for (const auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    // Evaluate constraint at (x_new[t], u_new[t])
                    Eigen::VectorXd g_vec = cKV.second->evaluate(X_new[t], U_new[t]); 

                    // Store constraint value relative to upper bound
                    G_new[cname][t] = g_vec - cKV.second->getUpperBound(); 
                    
                    // Get the updated slack s_new
                    const Eigen::VectorXd &s_vec = S_new[cname][t];
                    
                    // Add log-barrier term for path constraints
                    log_cost_new -= mu_ * s_vec.array().log().sum();

                    // Compute primal feasibility residual for path constraints
                    Eigen::VectorXd r_p = G_new[cname][t] + s_vec; // g(x_new, u_new) + s_new
                    rp_err += r_p.lpNorm<1>(); // Accumulate L1 norm
                }
             }

             // Add terminal cost
             cost_new += objective_->terminal_cost(X_new.back());
             log_cost_new += cost_new; // Add cost to log-barrier cost

             // --- Filter Check for MSIPDDP ---
             // Need a filter that considers log_cost, path_violation (rp_err), and gap_violation (gap_err)
             // Option 1: Combine violations: total_violation = rp_err + w * gap_err
             // Option 2: Use a 2D filter: (rp_err, gap_err)
             
             // Using combined violation as placeholder
             // double total_violation = std::max(rp_err, options_.cost_tolerance) + std::max(gap_err, options_.gap_tolerance); // Example combination
             double current_rp_err = std::max(rp_err, options_.cost_tolerance); // Ensure non-zero for filter
             double current_gap_err = std::max(gap_err, options_.grad_tolerance); // Ensure non-zero for filter


             // Build candidate filter point {Lagrangian, CombinedViolation}
             // FilterPoint candidate{log_cost_new, total_violation};
             // TODO: Define MSFilterPoint struct { log_cost, path_violation, gap_violation }

             // Check against filter (needs update for MS filter definition)
             bool candidateDominated = false;
             // for (const auto &fp : ms_filter_) // Assuming ms_filter_ exists
             // {
             //     if (candidate.log_cost >= fp.log_cost && 
             //         candidate.path_violation >= fp.path_violation &&
             //         candidate.gap_violation >= fp.gap_violation) 
             //     {
             //         candidateDominated = true;
             //         return result; // Dominated, return failure
             //     }
             // }
             
             // Placeholder: Using original filter logic based on log_cost and rp_err only
             cddp::FilterPoint candidate{log_cost_new, current_rp_err}; // Qualified with cddp::
             for (const auto &fp : filter_) {
                 if (candidate.log_cost >= fp.log_cost && candidate.violation >= fp.violation) {
                     candidateDominated = true;
                     return result; 
                 }
             }


             if (!candidateDominated)
             {
                 // Remove points dominated by candidate (Needs update for MS filter)
                 // for (auto it = ms_filter_.begin(); it != ms_filter_.end();) {
                 //     if (candidate.log_cost <= it->log_cost && 
                 //         candidate.path_violation <= it->path_violation &&
                 //         candidate.gap_violation <= it->gap_violation) {
                 //         it = ms_filter_.erase(it);
                 //     } else {
                 //         ++it;
                 //     }
                 // }
                 // ms_filter_.push_back(candidate); 
                 
                 // Placeholder: Using original filter update
                  for (auto it = filter_.begin(); it != filter_.end();) {
                    if (candidate.log_cost <= it->log_cost && candidate.violation <= it->violation) {
                        it = filter_.erase(it);
                    } else {
                        ++it;
                    }
                 }
                 filter_.push_back(candidate);


                 // Check reduction ratio (optional, filter acceptance is primary)
                 // double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
                 // double reduction_ratio = expected > 0.0 ? (L_ - log_cost_new) / expected : std::copysign(1.0, L_ - log_cost_new);

                 // Update result if filter accepted AND gap constraint met
                 // result.success = (gap_err <= options_.gap_tolerance); // Filter acceptance implies sufficient progress
                 result.success = true; // Placeholder: Assume filter acceptance is enough

                 if (result.success) {
                     result.state_sequence = X_new;
                     result.control_sequence = U_new;
                     result.dual_sequence = Y_new;    // Path constraint duals
                     result.slack_sequence = S_new;   // Path constraint slacks
                     result.constraint_sequence = G_new; // Path constraint values
                     // TODO: Add gaps_new, gap_multipliers_new to result
                     result.cost = cost_new;
                     result.lagrangian = log_cost_new;
                     result.constraint_violation = rp_err; // Path constraint violation
                     // result.gap_violation = gap_err; // Gap violation
                 }
             }
             return result;
         } // end constrained forward pass

    } // end solveMSIPDDPForwardPass


     // TODO: Rename resetIPDDPFilter -> resetMSIPDDPFilter
    void CDDP::resetMSIPDDPFilter() 
    {
        // Evaluate log-barrier cost (includes path constraints)
        L_ = J_; // Assume J_ (total cost) is computed from a rollout
        double rp_err = 0.0; // Path constraint violation
        double gap_err = 0.0; // Gap violation
        filter_ = {}; // TODO: Use ms_filter_

        // Calculate path constraint terms and violation
        for (int t = 0; t < horizon_; ++t)
        {
            for (const auto &cKV : constraint_set_) // Loop over path constraints
            {
                const std::string &cname = cKV.first;
                const Eigen::VectorXd &s_vec = S_[cname][t];
                const Eigen::VectorXd &g_vec = G_[cname][t]; // Assumes G_ is updated

                L_ -= mu_ * s_vec.array().log().sum(); // Barrier term for path constraints
                rp_err += (s_vec + g_vec).lpNorm<1>(); // Primal violation for path constraints
            }
            
            // Calculate gap violation (needs gaps_ to be computed/stored)
            // Eigen::VectorXd current_gap = X_[t+1] - system_->getDiscreteDynamics(X_[t], U_[t]); // Recompute gap
            // Eigen::VectorXd current_gap = gaps_[t]; // Assumes gaps_ stores current gaps
            // gap_err = std::max(gap_err, current_gap.lpNorm<Eigen::Infinity>()); 
            
             // TODO: Add barrier term for implicit gap constraints if formulation requires it
             // L_ -= mu_ * sum(log(...)) for gaps?
        }

        // Apply tolerances
        if (rp_err < options_.cost_tolerance) rp_err = 0.0;
        if (gap_err < options_.grad_tolerance) gap_err = 0.0; 
        
        constraint_violation_ = rp_err; // Store path violation
        // gap_violation_ = gap_err; // Store gap violation

        // Update filter (needs modification for MS filter)
        // ms_filter_.push_back(MSFilterPoint(L_, rp_err, gap_err));
        filter_.push_back(cddp::FilterPoint(L_, rp_err)); // Placeholder // Qualified with cddp::
        return;
    }

     // TODO: Rename initialIPDDPRollout -> initialMSIPDDPRollout
    void CDDP::initialMSIPDDPRollout() 
    {
        // For MSIPDDP, this needs to simulate the trajectory based on initial X_ and U_
        // and compute the initial gaps and total cost.
        
        // X_[0] is the fixed initial state.
        // X_[1]...X_[N] are initial guesses for states at interval boundaries.
        // U_[0]...U_[N-1] are initial guesses for controls.

        double cost = 0.0;
        // std::vector<Eigen::VectorXd> simulated_X(horizon_ + 1); // Store simulated trajectory
        // simulated_X[0] = initial_state_;
        
        // Rollout dynamics and calculate cost and gaps
        for (int t = 0; t < horizon_; ++t)
        {
            // State and control for interval t
            const Eigen::VectorXd &x_t_initial = X_[t]; // Initial state guess for interval t
            const Eigen::VectorXd &u_t = U_[t];         // Control guess for interval t

            // Compute stage cost using the guessed state/control
            cost += objective_->running_cost(x_t_initial, u_t, t);

            // Evaluate path constraints at guessed state/control
            for (const auto &cKV : constraint_set_)
            {
                const std::string &cname = cKV.first;
                Eigen::VectorXd g_vec = cKV.second->evaluate(x_t_initial, u_t); 
                G_[cname][t] = g_vec - cKV.second->getUpperBound(); // Store initial path constraint values
            }

            // Simulate the dynamics *within* interval t starting from x_t_initial
            Eigen::VectorXd x_t_final_simulated = system_->getDiscreteDynamics(x_t_initial, u_t);
            // Store the simulated trajectory if needed for other purposes
            // simulated_X[t+1] = x_t_final_simulated;

            // Calculate the initial gap for interval t
            // gap = x_{t+1}_initial_guess - x_{t+1}_simulated
             // TODO: Initialize gaps_ vector if it exists
            // gaps_[t] = X_[t+1] - x_t_final_simulated; 

        } // end for t

        // Add terminal cost based on the final *guessed* state X_[N]
        cost += objective_->terminal_cost(X_.back());

        // Store the initial total cost.
        J_ = cost;

        // Note: This rollout calculates the initial cost based on the *guessed* states X_
        // and the corresponding initial *gaps* between guessed and simulated states.
        // The initial slacks S_ and duals Y_, gap_multipliers_ should also be initialized (e.g., in initializeMSIPDDP).

        return;
    }

     // TODO: Rename resetIPDDPRegularization -> resetMSIPDDPRegularization if needed
    void CDDP::resetMSIPDDPRegularization() 
    {
        ipddp_regularization_counter_ = 0;
        // TODO: Reset any MS-specific regularization parameters?
        return;
    }

} // namespace pipddp