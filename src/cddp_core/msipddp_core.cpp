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

        // Resize linearized dynamics storage
        F_.resize(horizon_, Eigen::VectorXd::Zero(state_dim));
        Fx_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, state_dim));
        Fu_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, control_dim));
        A_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, state_dim));
        B_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, control_dim));
        if (!options_.is_ilqr)
        {
            Fxx_.resize(horizon_, std::vector<Eigen::MatrixXd>(state_dim, Eigen::MatrixXd::Zero(state_dim, state_dim)));
            Fuu_.resize(horizon_, std::vector<Eigen::MatrixXd>(control_dim, Eigen::MatrixXd::Zero(control_dim, control_dim)));
            Fux_.resize(horizon_, std::vector<Eigen::MatrixXd>(control_dim, Eigen::MatrixXd::Zero(state_dim, control_dim)));
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
            regularization_state_ = 0.0;
            regularization_state_step_ = 1.0;

            if (options_.verbose)
            {
                std::cout << "IPDDP: State regularization is not enabled for IPDDP" << std::endl;
            }
        }

        if (options_.regularization_type == "control" || options_.regularization_type == "both")
        {
            regularization_control_ = options_.regularization_control;
            regularization_control_step_ = options_.regularization_control_step;

            if (options_.verbose)
            {
                std::cout << "IPDDP: Control regularization is enabled for IPDDP" << std::endl;
            }
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

        // Initialize defect penalty
        defect_violation_penalty_ = options_.defect_violation_penalty_initial;
        ms_segment_length_ = options_.ms_segment_length;

        // Check if ms_segment_length_ is valid
        if (ms_segment_length_ < 0)
        {
            std::cerr << "MSIPDDP: ms_segment_length_ must be non-negative" << std::endl;
            throw std::runtime_error("MSIPDDP: ms_segment_length_ must be non-negative");
        }

        if (options_.ms_rollout_type != "linear" && options_.ms_rollout_type != "nonlinear" && options_.ms_rollout_type != "hybrid")
        {
            std::cerr << "MSIPDDP: Invalid ms_rollout_type: " << options_.ms_rollout_type << std::endl;
            throw std::runtime_error("MSIPDDP: Invalid ms_rollout_type");
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
            if (solution.converged)
            {
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
                Y_ = best_result.dual_sequence;       // Path constraint duals
                S_ = best_result.slack_sequence;      // Path constraint slacks
                G_ = best_result.constraint_sequence; // Path constraint values
                F_ = best_result.dynamics_sequence;   // Dynamics
                // TODO: Update gaps_ and gap_multipliers_ from best_result
                // gaps_ = best_result.gap_sequence;
                // gap_multipliers_ = best_result.gap_multiplier_sequence;

                dJ_ = J_ - best_result.cost;
                J_ = best_result.cost;
                dL_ = L_ - best_result.lagrangian; // Lagrangian includes path & gap terms
                L_ = best_result.lagrangian;
                alpha_ = best_result.alpha;
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
                printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_);
            }

            // Check termination
            if (std::max(optimality_gap_, mu_) <= options_.grad_tolerance)
            {   
                if (options_.debug)
                {
                    std::cout << "IPDDP: Converged due to small change in cost and Lagrangian." << std::endl;
                }
                solution.converged = true;
                break;
            }
            if (abs(dJ_) < options_.cost_tolerance && abs(dL_) < options_.cost_tolerance)
            {
                if (options_.debug)
                {
                    std::cout << "IPDDP: Converged due to small change in cost and Lagrangian." << std::endl;
                }
                solution.converged = true;
                break;
            }

            // Barrier update logic 
            if (optimality_gap_ <= 0.2 * mu_)
            {
                if (constraint_set_.empty())
                {
                }
                else
                {
                    mu_ = std::max(options_.cost_tolerance / 10.0, std::min(0.2 * mu_, std::pow(mu_, 1.2)));
                }
                resetIPDDPFilter();
                resetIPDDPRegularization();
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Finalize solution
        solution.state_sequence = X_; // Initial states of intervals
        solution.control_sequence = U_;
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

        // --- Pre-computation Phase: Compute and store linearized dynamics ---
        for (int t = 0; t < horizon_; ++t)
        {
            const Eigen::VectorXd &x = X_[t];
            const Eigen::VectorXd &u = U_[t];

            // Dynamics Jacobians
            const auto [Fx, Fu] = system_->getJacobians(x, u);
            Fx_[t] = Fx;
            Fu_[t] = Fu;

            // Linearized dynamics matrices
            A_[t] = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx_[t];
            B_[t] = timestep_ * Fu_[t];

            // Dynamics Hessians
            if (!options_.is_ilqr)
            {
                const auto hessians = system_->getHessians(x, u);
                Fxx_[t] = std::get<0>(hessians);
                Fuu_[t] = std::get<1>(hessians);
                Fux_[t] = std::get<2>(hessians);
            }
        }
        // --- End Pre-computation Phase ---

        // DDP Backward Pass
        int t = horizon_ - 1;
        if (constraint_set_.empty())
        {   
            // Unconstrained case
            while (t >= 0)
            {
                const Eigen::VectorXd &x = X_[t];           // Initial state of interval t
                const Eigen::VectorXd &u = U_[t];           // Control for interval t
                const Eigen::VectorXd &lambda = Lambda_[t]; // Costate for interval t
                const Eigen::VectorXd &f = F_[t];           // Dynamics at interval t
                const Eigen::VectorXd &d = f - X_[t + 1];   // Defect
                const Eigen::MatrixXd &A = A_[t];
                const Eigen::MatrixXd &B = B_[t];

                // Cost derivatives at (x_t, u_t)
                double l = objective_->running_cost(x, u, t);
                auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

                Eigen::VectorXd Q_x = l_x + A.transpose() * (V_x + V_xx * d);
                Eigen::VectorXd Q_u = l_u + B.transpose() * (V_x + V_xx * d);
                Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
                Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
                Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

                // Add state hessian term if not using iLQR
                if (!options_.is_ilqr)
                {
                    for (int i = 0; i < state_dim; ++i)
                    {
                        Q_xx += timestep_ * V_x(i) * Fxx_[t][i];
                        Q_ux += timestep_ * V_x(i) * Fux_[t][i];
                        Q_uu += timestep_ * V_x(i) * Fuu_[t][i];
                    }
                }

                // Eigen::Matrix Q_w = - lambda + V_x;
                // Eigen::Matrix Q_ww = V_xx;

                // Regularization
                Eigen::MatrixXd Q_uu_reg = Q_uu;
                // Apply regularization 
                Q_uu_reg.diagonal().array() += regularization_control_;
                Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

                Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
                if (llt.info() != Eigen::Success)
                {
                    if (options_.debug)
                    {
                        std::cerr << "MSIPDDP: Backward pass failed at time " << t << std::endl;
                    }
                    return false;
                }

                // Solve for gains k_u, K_u
                Eigen::VectorXd k_u = -llt.solve(Q_u);
                Eigen::MatrixXd K_u = -llt.solve(Q_ux); 
                k_u_[t] = k_u;
                K_u_[t] = K_u;

                // Update Value function derivatives
                V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
                V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;
                V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize V_xx

                // Accumulate cost improvement
                dV_[0] += k_u.dot(Q_u);
                dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

                // Error tracking 
                Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
                rf_err = std::max(rf_err, d.lpNorm<Eigen::Infinity>());

                t--;
            } // end while t (unconstrained)

            optimality_gap_ = std::max(Qu_err, rf_err);
            if (options_.debug)
            {
                std::cout << "[MSIPDDP Backward Pass]\n"
                          << "    Qu_err:  " << Qu_err << "\n"
                          << "    rf_err:  " << rf_err << "\n"
                          << "    dV:      " << dV_.transpose() << std::endl;
            }
            return true;
        }


        return false; 
    } // end solveMSIPDDPBackwardPass

    // TODO: Rename solveIPDDPForwardPass -> solveMSIPDDPForwardPass
    ForwardPassResult CDDP::solveMSIPDDPForwardPass(double alpha)
    {
        // Prepare result structure
        ForwardPassResult result;
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.lagrangian = std::numeric_limits<double>::infinity();
        result.alpha = alpha;

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();
        const int dual_dim = getTotalDualDim(); 

        // Define tau for feasibility check
        double tau = std::max(0.99, 1.0 - mu_);

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        std::vector<Eigen::VectorXd> F_new = F_;
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
        std::map<std::string, std::vector<Eigen::VectorXd>> G_new = G_;

        // Set the initial state 
        X_new[0] = initial_state_;

        // Initialize cost_new for this alpha trial
        double cost_new = 0.0;
        double logcost_new = 0.0;
        double rp_err = 0.0;
        double rf_err = 0.0;

        // Rollout loop
        for (int t = 0; t < horizon_; ++t)
        {
            const Eigen::VectorXd dx = X_new[t] - X_[t];

            // Update control
            const Eigen::VectorXd du = alpha * k_u_[t] + K_u_[t] * dx;
            U_new[t] = U_[t] + du;

            // --- Rollout based on options_.ms_rollout_type ---
            Eigen::VectorXd f_new_t;
            if (options_.ms_rollout_type == "nonlinear")
            {
                f_new_t = system_->getDiscreteDynamics(X_new[t], U_new[t]);
                X_new[t + 1] = f_new_t;
            }
            else if (options_.ms_rollout_type == "linear")
            {
            
                f_new_t = F_[t] + A_[t] * dx + B_[t] * du; 
                X_new[t + 1] = X_[t] + A_[t] * dx + B_[t] * du;
            }
            else // Hybrid (default)
            {
                    f_new_t = system_->getDiscreteDynamics(X_new[t], U_new[t]);
                X_new[t + 1] = f_new_t;
            }
            F_new[t] = f_new_t;


            // Determine if the *next* step (t+1) starts a new segment boundary
            bool apply_gap_closing_strategy = (ms_segment_length_ > 0) &&
                                              ((t + 1) % ms_segment_length_ == 0) &&
                                              (t + 1 < horizon_);

            if (apply_gap_closing_strategy)
            {
                const Eigen::VectorXd defect_at_segment_end = F_[t] - X_[t + 1]; // Defect from the *previous* iteration's plan

                if (options_.ms_rollout_type == "nonlinear")
                {
                    X_new[t + 1] = f_new_t + alpha * defect_at_segment_end;
                }
                else if (options_.ms_rollout_type == "linear")
                {
                    X_new[t + 1] = X_[t] + A_[t] * dx + alpha * B_[t] * k_u_[t] + alpha * defect_at_segment_end;
                }
                else // Hybrid (default)
                {
                    X_new[t + 1] = X_[t] + A_[t] * dx + alpha * B_[t] * k_u_[t] + alpha * defect_at_segment_end;
                }
            }
            else // Not a segment boundary
            {
                X_new[t + 1] = f_new_t;
            }

            // --- Robustness Check during Rollout ---
            if (!X_new[t + 1].allFinite() || !U_new[t].allFinite())
            {
                if (options_.debug)
                {
                    std::cerr << "[MSIPDDP Forward Pass] NaN/Inf detected during HYBRID rollout (nonlinear within, linear between) at t=" << t
                              << " for alpha=" << alpha << std::endl; // Updated debug message
                }
                result.success = false;
                cost_new = std::numeric_limits<double>::infinity();
                return result;
            }
        } // End of rollout loop

        // --- Cost and Defect Evaluation for New Trajectory ---
        cost_new = 0.0;
        for (int t = 0; t < horizon_; ++t)
        {
            cost_new += objective_->running_cost(X_new[t], U_new[t], t);
            Eigen::VectorXd d = F_new[t] - X_new[t + 1];
            logcost_new += defect_violation_penalty_ * d.lpNorm<Eigen::Infinity>();
            rf_err += std::max(rf_err, d.lpNorm<Eigen::Infinity>());
        } // End of cost and defect evaluation loop
        cost_new += objective_->terminal_cost(X_new.back());
        logcost_new += cost_new;

        // Build a candidate filter point from the computed cost metrics.
        FilterPoint candidate{logcost_new, rf_err};

        // Check if candidate is dominated by any existing filter point
        bool candidateDominated = false;
        for (const auto &fp : filter_)
        {
            // If the candidate is dominated by an existing filter point, early exit.
            if (candidate.log_cost >= fp.log_cost && candidate.violation >= fp.violation)
            {
                candidateDominated = true;
                return result;
            }
        }

        if (!candidateDominated)
        {
            // Remove any filter points that are dominated by the candidate.
            for (auto it = filter_.begin(); it != filter_.end();)
            {
                if (candidate.log_cost <= it->log_cost && candidate.violation <= it->violation)
                {
                    // Candidate dominates this point, so erase it.
                    it = filter_.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            // Append the candidate to the filter set.
            filter_.push_back(candidate);

            // Update the result with the new trajectories and metrics.
            result.success = true;
            result.state_sequence = X_new;
            result.control_sequence = U_new;
            result.dynamics_sequence = F_new;
            result.dual_sequence = Y_new;
            result.slack_sequence = S_new;
            result.constraint_sequence = G_new;
            result.cost = cost_new;
            result.lagrangian = logcost_new;
            result.constraint_violation = rf_err;
        }

        return result;
    } // end solveMSIPDDPForwardPass

    // TODO: Rename resetIPDDPFilter -> resetMSIPDDPFilter
    void CDDP::resetMSIPDDPFilter()
    {
        // Evaluate log-barrier cost (includes path constraints)
        L_ = J_;             // Assume J_ (total cost) is computed from a rollout
        double rp_err = 0.0; // Path constraint violation
        double rf_err = 0.0; // Gap violation
        filter_ = {};        // TODO: Use ms_filter_

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

            // Add defect violation penalty
            Eigen::VectorXd d = F_[t] - X_[t + 1];
            L_ += defect_violation_penalty_ * d.lpNorm<1>();
            rf_err = std::max(rf_err, d.lpNorm<Eigen::Infinity>());
        }

        // Apply tolerances
        if (rp_err < options_.cost_tolerance)
            rp_err = 0.0;
        if (rf_err < options_.grad_tolerance)
            rf_err = 0.0;

        constraint_violation_ = std::max(rp_err, rf_err);

        // Update filter
        filter_.push_back(cddp::FilterPoint(L_, constraint_violation_)); // Placeholder // Qualified with cddp::
        return;
    }

    // TODO: Rename initialIPDDPRollout -> initialMSIPDDPRollout
    void CDDP::initialMSIPDDPRollout()
    {
        double cost = 0.0;

        // Rollout dynamics and calculate cost and gaps
        for (int t = 0; t < horizon_; ++t)
        {
            // State and control for interval t
            const Eigen::VectorXd &x_t = X_[t]; // Initial state guess for interval t
            const Eigen::VectorXd &u_t = U_[t]; // Control guess for interval t

            // Compute stage cost using the guessed state/control
            cost += objective_->running_cost(x_t, u_t, t);

            // Evaluate path constraints at guessed state/control
            for (const auto &cKV : constraint_set_)
            {
                const std::string &cname = cKV.first;
                Eigen::VectorXd g_vec = cKV.second->evaluate(x_t, u_t);
                G_[cname][t] = g_vec - cKV.second->getUpperBound(); // Store initial path constraint values
            }

            // Compute defect
            Eigen::VectorXd f = system_->getDiscreteDynamics(x_t, u_t);
            F_[t] = f;
        }

        // Add terminal cost based on the final *guessed* state X_[N]
        cost += objective_->terminal_cost(X_.back());

        // Store the initial total cost.
        J_ = cost;

        return;
    }

    // TODO: Rename resetIPDDPRegularization -> resetMSIPDDPRegularization if needed
    void CDDP::resetMSIPDDPRegularization()
    {
        ipddp_regularization_counter_ = 0;
        // TODO: Reset any MS-specific regularization parameters?
        return;
    }

    double CDDP::calculate_defect_norm(const std::vector<Eigen::VectorXd> &X,
                                       const std::vector<Eigen::VectorXd> &U,
                                       const std::vector<Eigen::VectorXd> &F) const
    {
        double total_defect_norm_l1 = 0.0;
        // Add basic size checks for safety
        if (X.size() != horizon_ + 1 || F.size() != horizon_ || U.size() != horizon_)
        {
            std::cerr << "ERROR: Inconsistent sizes provided to calculate_defect_norm. X: " << X.size()
                      << ", U: " << U.size() << ", F: " << F.size() << ", Horizon: " << horizon_ << std::endl;
            // Returning infinity might be too harsh, consider alternatives or exceptions
            return std::numeric_limits<double>::infinity();
        }

        for (int t = 0; t < horizon_; ++t)
        {
            Eigen::VectorXd defect = F[t] - X[t + 1];
            total_defect_norm_l1 += defect.lpNorm<1>(); // Sum of L1 norms
        }
        return total_defect_norm_l1;
    }

} // namespace cddp
