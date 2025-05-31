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

        Lambda_.resize(horizon_, Eigen::VectorXd::Zero(state_dim)); // Resize, will be filled in 1st backward pass
        ms_lambda_initialization_ = true;
        // for (int t = 0; t < horizon_; ++t)
        // {
        //     Lambda_[t] = options_.lambda_scale * Eigen::VectorXd::Ones(state_dim);
        // }

        // Resize linearized dynamics storage
        F_.resize(horizon_, Eigen::VectorXd::Zero(state_dim));
        Fx_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, state_dim));
        Fu_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, control_dim));
        A_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, state_dim));
        B_.resize(horizon_, Eigen::MatrixXd::Zero(state_dim, control_dim));
        if (!options_.is_ilqr)
        {
            Fxx_.resize(horizon_, std::vector<Eigen::MatrixXd>(state_dim, Eigen::MatrixXd::Zero(state_dim, state_dim)));
            Fuu_.resize(horizon_, std::vector<Eigen::MatrixXd>(state_dim, Eigen::MatrixXd::Zero(control_dim, control_dim)));
            Fux_.resize(horizon_, std::vector<Eigen::MatrixXd>(state_dim, Eigen::MatrixXd::Zero(control_dim, state_dim)));
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

        // Determine initial mu for the heuristic s*y = mu
        double heuristic_initial_mu = options_.barrier_coeff; // Default: 1.0
        mu_ = heuristic_initial_mu;                           // Set initial mu

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
                // Evaluate g(x,u) = evaluate(x,u) - getUpperBound()
                Eigen::VectorXd g_at_xt_ut = constraint.second->evaluate(X_[t], U_[t]) - constraint.second->getUpperBound();
                G_[constraint_name][t] = g_at_xt_ut;

                Eigen::VectorXd s_init_t = Eigen::VectorXd::Zero(dual_dim);
                Eigen::VectorXd y_init_t = Eigen::VectorXd::Zero(dual_dim);

                for (int i = 0; i < dual_dim; ++i)
                {
                    // Initialize s_i = max(options.slack_scale, -g_i) to ensure s_i > 0
                    s_init_t(i) = std::max(options_.slack_scale, -g_at_xt_ut(i));
                
                    // Initialize y_i = heuristic_initial_mu / s_i to satisfy s_i * y_i = heuristic_initial_mu
                    if (s_init_t(i) < 1e-12)
                    { // Safeguard
                        y_init_t(i) = heuristic_initial_mu / 1e-12;
                    }
                    else
                    {
                        y_init_t(i) = heuristic_initial_mu / s_init_t(i);
                    }
                    // Ensure y_i is also not too small
                    y_init_t(i) = std::max(options_.dual_scale * 0.1, std::min(y_init_t(i), options_.dual_scale * 10.0)); // 10% of dual_scale 
                }
                Y_[constraint_name][t] = y_init_t;
                S_[constraint_name][t] = s_init_t;

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

            if (options_.debug)
            {
                std::cout << "IPDDP: State regularization is not enabled for IPDDP" << std::endl;
            }
        }

        if (options_.regularization_type == "control" || options_.regularization_type == "both")
        {
            regularization_control_ = options_.regularization_control;
            regularization_control_step_ = options_.regularization_control_step;

            if (options_.debug)
            {
                std::cout << "IPDDP: Control regularization is enabled for IPDDP" << std::endl;
            }
        }
        else
        {
            regularization_control_ = 0.0;
            regularization_control_step_ = 1.0;
        }

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
        if (!initialized_ || Lambda_.empty() || S_.empty())
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
        solution.state_sequence.reserve(horizon_ + 1);
        solution.cost_sequence.reserve(options_.max_iterations);
        solution.lagrangian_sequence.reserve(options_.max_iterations);

        // Initialize trajectories and gaps
        initialMSIPDDPRollout(); // J_ is computed inside this function
        solution.cost_sequence.push_back(J_);

        // Reset MSIPDDP filter
        resetMSIPDDPFilter(); // L_ is computed inside this function
        solution.lagrangian_sequence.push_back(L_);

        if (options_.verbose)
        {
            // TODO: Update printIteration to include gap violation norm
            printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); // Initial iteration information
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        ForwardPassResult best_result;

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
            // TODO: Remove this
            if (solution.converged)
            {
                break;
            }

            // 2. Forward pass (either single-threaded or multi-threaded) for MSIPDDP
            best_result.success = false;
            best_result.cost = std::numeric_limits<double>::infinity();
            best_result.lagrangian = std::numeric_limits<double>::infinity();
            best_result.constraint_violation = 0.0; // Path constraint violation

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
                }
                X_ = best_result.state_sequence; // Initial states of intervals
                U_ = best_result.control_sequence;
                Y_ = best_result.dual_sequence;       // Path constraint duals
                S_ = best_result.slack_sequence;      // Path constraint slacks
                G_ = best_result.constraint_sequence; // Path constraint values
                F_ = best_result.dynamics_sequence;   // Dynamics

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
                        std::cerr << "MSIPDDP: Forward Pass regularization limit reached" << std::endl;
                    }

                    solution.converged = false;

                    break;
                }
            }

            // Print iteration information
            if (options_.verbose)
            {
                printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_);
            }

            // Check termination
            double mu_vs_kkt_error = std::max(mu_, kkt_error_);
            if (std::max(optimality_gap_, mu_vs_kkt_error) <= options_.grad_tolerance)
            {
                if (options_.debug)
                {
                    std::cout << "MSIPDDP: Converged due to optimality gap and mu." << std::endl;
                }
                solution.converged = true;
                break;
            }

            // TODO: This should be removed
            if (abs(dJ_) < options_.cost_tolerance && abs(dL_) < options_.cost_tolerance && mu_vs_kkt_error <= options_.grad_tolerance)
            {
                if (options_.debug)
                {
                    std::cout << "MSIPDDP: Converged due to small change in cost and Lagrangian." << std::endl;
                }
                solution.converged = true;
                break;
            }

            // Barrier update logic
            if (kkt_error_ <= options_.barrier_update_factor * mu_)
            {
                double linear_reduction_target_factor = options_.barrier_update_factor;
                if (mu_ > 1e-12)
                {
                    double kkt_progress_metric = kkt_error_ / mu_;
                    // Satisfying the KKT conditions for the current mu.
                    // So, we can be more aggressive in reducing mu.
                    if (kkt_progress_metric < 0.1 * options_.barrier_update_factor)
                    {
                        // Significantly better than threshold: make reduction factor more aggressive
                        linear_reduction_target_factor = options_.barrier_update_factor * 0.5;
                    }
                    else if (kkt_progress_metric < 0.5 * options_.barrier_update_factor)
                    {
                        // Moderately better than threshold: make reduction factor slightly more aggressive
                        linear_reduction_target_factor = options_.barrier_update_factor * 0.75;
                    }
                }
                mu_ = std::max(options_.grad_tolerance / 10.0, std::min(linear_reduction_target_factor * mu_, std::pow(mu_, options_.barrier_update_power)));

                resetMSIPDDPFilter();

                if (options_.debug)
                {
                    std::cout << "MSIPDDP: Barrier update logic" << std::endl;
                }
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

        bool first_pass_flag_at_start = ms_lambda_initialization_;

        // --- Pre-computation Phase: Compute and store linearized dynamics --- TODO: Parallelize this
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
                if (ms_lambda_initialization_)
                {
                    Lambda_[t] = V_x; // V_x here is dV/dX_[t+1]
                }

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
                        Q_xx += timestep_ * lambda(i) * Fxx_[t][i];
                        Q_ux += timestep_ * lambda(i) * Fux_[t][i];
                        Q_uu += timestep_ * lambda(i) * Fuu_[t][i];
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

                // Solve for gains k_lambda, K_lambda
                k_lambda_[t] = -lambda + V_x + V_xx * d;
                K_lambda_[t] = V_xx;
                K_lambda_[t] = 0.5 * (K_lambda_[t] + K_lambda_[t].transpose()); // Symmetrize K_lambda

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

            optimality_gap_ = Qu_err;
            kkt_error_ = rf_err;

            if (options_.debug)
            {
                std::cout << "[MSIPDDP Backward Pass]\n"
                          << "    Qu_err:  " << Qu_err << "\n"
                          << "    rf_err:  " << rf_err << "\n"
                          << "    dV:      " << dV_.transpose() << std::endl;
            }
            if (first_pass_flag_at_start)
            {
                ms_lambda_initialization_ = false;
            }
            return true;
        }
        else
        {
            // Constrained case
            while (t >= 0)
            {
                if (ms_lambda_initialization_)
                {
                    Lambda_[t] = V_x; // V_x here is dV/dX_[t+1]
                }

                const Eigen::VectorXd &x = X_[t];           // Initial state of interval t
                const Eigen::VectorXd &u = U_[t];           // Control for interval t
                const Eigen::VectorXd &lambda = Lambda_[t]; // Costate for interval t
                const Eigen::VectorXd &f = F_[t];           // Dynamics at interval t
                const Eigen::VectorXd &d = f - X_[t + 1];   // Defect
                const Eigen::MatrixXd &A = A_[t];
                const Eigen::MatrixXd &B = B_[t];

                // Gather dual and slack variables, and constraint values across all constraints
                Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd s = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);
                Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);

                // Variables to store summed constraint Hessians
                Eigen::MatrixXd sum_g_xx_y = Eigen::MatrixXd::Zero(state_dim, state_dim);
                Eigen::MatrixXd sum_g_uu_y = Eigen::MatrixXd::Zero(control_dim, control_dim);
                Eigen::MatrixXd sum_g_ux_y = Eigen::MatrixXd::Zero(control_dim, state_dim);

                int offset = 0; // offset in [0..total_dual_dim)
                for (auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    auto &constraint = cKV.second;
                    int dual_dim = constraint->getDualDim();

                    // Slack, dual, and constraint at time t and constraint cname
                    const Eigen::VectorXd &y_vec = Y_[cname][t]; // dual variable
                    const Eigen::VectorXd &s_vec = S_[cname][t]; // slack variable
                    const Eigen::VectorXd &g_vec = G_[cname][t]; // constraint value
                    const Eigen::MatrixXd &g_x = constraint->getStateJacobian(x, u);
                    const Eigen::MatrixXd &g_u = constraint->getControlJacobian(x, u);

                    // Insert into big arrays
                    y.segment(offset, dual_dim) = y_vec;
                    s.segment(offset, dual_dim) = s_vec;
                    g.segment(offset, dual_dim) = g_vec;
                    Q_yx.block(offset, 0, dual_dim, state_dim) = g_x;
                    Q_yu.block(offset, 0, dual_dim, control_dim) = g_u;

                    // // Get constraint Hessians if not using iLQR
                    // if (!options_.is_ilqr) // Or a new option specific to constraint Hessians
                    // {
                    //     const auto constraint_hessians = constraint->getHessians(x, u);
                    //     const auto& g_xx_list = std::get<0>(constraint_hessians); // std::vector<Eigen::MatrixXd>
                    //     const auto& g_uu_list = std::get<1>(constraint_hessians); // std::vector<Eigen::MatrixXd>
                    //     const auto& g_ux_list = std::get<2>(constraint_hessians); // std::vector<Eigen::MatrixXd>

                    //     for (int i = 0; i < dual_dim; ++i)
                    //     {
                    //         if (g_xx_list.size() > i && !g_xx_list[i].hasNaN()) sum_g_xx_y += y_vec(i) * g_xx_list[i];
                    //         if (g_uu_list.size() > i && !g_uu_list[i].hasNaN()) sum_g_uu_y += y_vec(i) * g_uu_list[i];
                    //         if (g_ux_list.size() > i && !g_ux_list[i].hasNaN()) sum_g_ux_y += y_vec(i) * g_ux_list[i];
                    //     }
                    // }

                    offset += dual_dim;
                }

                // Cost & derivatives
                double l = objective_->running_cost(x, u, t);
                auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

                // Q expansions from cost
                Eigen::VectorXd Q_x = l_x + Q_yx.transpose() * y + A.transpose() * (V_x + V_xx * d);
                Eigen::VectorXd Q_u = l_u + Q_yu.transpose() * y + B.transpose() * (V_x + V_xx * d);
                Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
                Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
                Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

                // Add state hessian term if not using iLQR
                if (!options_.is_ilqr)
                {
                    for (int i = 0; i < state_dim; ++i)
                    {
                        Q_xx += timestep_ * lambda(i) * Fxx_[t][i];
                        Q_ux += timestep_ * lambda(i) * Fux_[t][i];
                        Q_uu += timestep_ * lambda(i) * Fuu_[t][i];
                    }
                    // Add constraint Hessian terms
                    // Q_xx += sum_g_xx_y;
                    //     Q_uu += sum_g_uu_y;
                    //     Q_ux += sum_g_ux_y;
                }

                Eigen::MatrixXd Y = y.asDiagonal();  // Diagonal matrix with y as diagonal
                Eigen::MatrixXd S = s.asDiagonal();  // Diagonal matrix with s as diagonal
                Eigen::MatrixXd G = g.asDiagonal();  // Diagonal matrix with g as diagonal
                Eigen::MatrixXd S_inv = S.inverse(); // Inverse of S
                Eigen::MatrixXd YSinv = Y * S_inv;   // Y * S_inv

                // Residuals:
                // r_p = g + s (primal feasibility)
                Eigen::VectorXd r_p = g + s;

                // r_d   = y.*s - mu (dual feasibility)
                Eigen::VectorXd r_d = y.cwiseProduct(s).array() - mu_;

                // rhat = y .* r_p - r_d = y.*(g + s) - (y.*s - mu)
                Eigen::VectorXd rhat = y.cwiseProduct(r_p) - r_d;

                // Regularization
                Eigen::MatrixXd Q_uu_reg = Q_uu;
                // Apply regularization
                Q_uu_reg.diagonal().array() += regularization_control_;
                Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

                Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg + Q_yu.transpose() * YSinv * Q_yu);
                if (llt.info() != Eigen::Success)
                {
                    if (options_.debug)
                    {
                        std::cerr << "IPDDP: Backward pass failed at time " << t << std::endl;
                    }
                    return false;
                }

                Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
                bigRHS.col(0) = Q_u + Q_yu.transpose() * S_inv * rhat;
                Eigen::MatrixXd M = //(control_dim, state_dim)
                    Q_ux + Q_yu.transpose() * YSinv * Q_yx;
                for (int col = 0; col < state_dim; col++)
                {
                    bigRHS.col(col + 1) = M.col(col);
                }

                Eigen::MatrixXd R = llt.matrixU();
                Eigen::MatrixXd z = R.triangularView<Eigen::Upper>().solve(bigRHS);
                Eigen::MatrixXd kK = -R.transpose().triangularView<Eigen::Lower>().solve(z);

                // parse out feedforward (ku) and feedback (Ku)
                Eigen::VectorXd k_u = kK.col(0); // dimension [control_dim]
                Eigen::MatrixXd K_u(control_dim, state_dim);
                for (int col = 0; col < state_dim; col++)
                {
                    K_u.col(col) = kK.col(col + 1);
                }

                // Save gains
                k_u_[t] = k_u;
                K_u_[t] = K_u;

                // Compute gains for constraints
                Eigen::VectorXd k_y = S_inv * (rhat + Y * Q_yu * k_u);
                Eigen::MatrixXd K_y = YSinv * (Q_yx + Q_yu * K_u);
                Eigen::VectorXd k_s = -r_p - Q_yu * k_u;
                Eigen::MatrixXd K_s = -Q_yx - Q_yu * K_u;

                offset = 0;
                for (auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    auto &constraint = cKV.second;
                    int dual_dim = constraint->getDualDim();

                    // Now store gains
                    k_y_[cname][t] = k_y.segment(offset, dual_dim);
                    K_y_[cname][t] = K_y.block(offset, 0, dual_dim, state_dim);
                    k_s_[cname][t] = k_s.segment(offset, dual_dim);
                    K_s_[cname][t] = K_s.block(offset, 0, dual_dim, state_dim);

                    offset += dual_dim;
                }

                // Update Q expansions
                Q_u += Q_yu.transpose() * S_inv * rhat;
                Q_x += Q_yx.transpose() * S_inv * rhat;
                Q_xx += Q_yx.transpose() * YSinv * Q_yx;
                Q_ux += Q_yx.transpose() * YSinv * Q_yu;
                Q_uu += Q_yu.transpose() * YSinv * Q_yu;

                // Update cost improvement
                dV_[0] += k_u.dot(Q_u);
                dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

                // Update value function
                V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
                V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;

                // Error tracking
                Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
                rp_err = std::max(rp_err, r_p.lpNorm<Eigen::Infinity>());
                rd_err = std::max(rd_err, r_d.lpNorm<Eigen::Infinity>());
                rf_err = std::max(rf_err, d.lpNorm<Eigen::Infinity>());

                t--;
            } // end while t

            // Compute optimality gap and print
            optimality_gap_ = Qu_err;
            kkt_error_ = std::max(rp_err, std::max(rd_err, rf_err));

            if (options_.debug)
            {
                std::cout << "[IPDDP Backward Pass]\n"
                          << "    Qu_err:  " << Qu_err << "\n"
                          << "    rp_err:  " << rp_err << "\n"
                          << "    rd_err:  " << rd_err << "\n"
                          << "    rf_err:  " << rf_err << "\n"
                          << "    dV:      " << dV_.transpose() << std::endl;
            }
            if (first_pass_flag_at_start)
            {
                ms_lambda_initialization_ = false;
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
        double tau = std::max(options_.minimum_fraction_to_boundary, 1.0 - mu_); // fraction-to-boundary parameter

        // Filter acceptance
        double filter_merit_acceptance = options_.filter_merit_acceptance;
        double filter_violation_acceptance = options_.filter_violation_acceptance;

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        std::vector<Eigen::VectorXd> F_new = F_;
        std::vector<Eigen::VectorXd> Lambda_new = Lambda_;
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
        std::map<std::string, std::vector<Eigen::VectorXd>> G_new = G_;

        // Set the initial state
        X_new[0] = initial_state_;

        // Initialize cost_new for this alpha trial
        double cost_new = 0.0;
        double log_cost_new = 0.0;
        double rp_err = 0.0;
        double rf_err = 0.0;

        // Rollout loop
        if (constraint_set_.empty())
        {
            // Unconstrained case
            for (int t = 0; t < horizon_; ++t)
            {
                const Eigen::VectorXd dx = X_new[t] - X_[t];

                // Update control
                const Eigen::VectorXd du = alpha * k_u_[t] + K_u_[t] * dx;
                U_new[t] = U_[t] + du;

                // Update lambda
                Lambda_new[t] = alpha * Lambda_[t] + K_lambda_[t] * dx;

                // --- Rollout Logic ---
                Eigen::VectorXd dynamics_eval_for_F_new_t;

                if (options_.ms_rollout_type == "nonlinear" || options_.ms_rollout_type == "hybrid")
                {
                    dynamics_eval_for_F_new_t = system_->getDiscreteDynamics(X_new[t], U_new[t]);
                }
                else // options_.ms_rollout_type == "linear"
                {
                    dynamics_eval_for_F_new_t = F_[t] + A_[t] * dx + B_[t] * du;
                }
                F_new[t] = dynamics_eval_for_F_new_t; // Store the calculated dynamics/approximation

                // Determine if the *next* step (t+1) starts a new segment boundary
                const Eigen::VectorXd defect_at_segment_end = F_[t] - X_[t + 1];
                bool is_segment_boundary = (ms_segment_length_ > 0) &&
                                           ((t + 1) % ms_segment_length_ == 0) &&
                                           (t + 1 < horizon_);
                bool defect_is_large = defect_at_segment_end.lpNorm<Eigen::Infinity>() > options_.ms_defect_tolerance_for_single_shooting;
                bool apply_gap_closing_strategy = is_segment_boundary && defect_is_large;

                if (apply_gap_closing_strategy)
                {
                    if (options_.ms_rollout_type == "nonlinear")
                    {
                        X_new[t + 1] = dynamics_eval_for_F_new_t + alpha * defect_at_segment_end;
                    }
                    else if (options_.ms_rollout_type == "linear" || options_.ms_rollout_type == "hybrid")
                    {
                        X_new[t + 1] = X_[t] + A_[t] * dx + alpha * B_[t] * k_u_[t] + alpha * defect_at_segment_end;
                    }
                }
                else // Not a segment boundary or defect is small, so no gap closing strategy applied
                {
                    if (options_.ms_rollout_type == "nonlinear" || options_.ms_rollout_type == "hybrid")
                    {
                        X_new[t + 1] = dynamics_eval_for_F_new_t; // This is the nonlinear dynamics evaluation
                    }
                    else // options_.ms_rollout_type == "linear"
                    {
                        X_new[t + 1] = X_[t] + A_[t] * dx + B_[t] * du; // Linear rollout for state
                    }
                }
                // --- End Rollout Logic ---

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
                rf_err += d.lpNorm<1>();
            } // End of cost and defect evaluation loop
            cost_new += objective_->terminal_cost(X_new.back());
            log_cost_new += cost_new;

            double constraint_violation_old = constraint_violation_;
            double constraint_violation_new = rf_err;
            double log_cost_old = L_;
            bool filter_acceptance = false;
            double expected_improvement = alpha * dV_(0);

            if (constraint_violation_new > options_.filter_maximum_violation)
            {
                if (constraint_violation_new < options_.filter_acceptance * constraint_violation_old)
                {
                    filter_acceptance = true;
                }
                else
                {
                    filter_acceptance = false;
                }
            }
            else if (std::max(constraint_violation_new, constraint_violation_old) < options_.filter_minimum_violation && expected_improvement < 0)
            {
                if (log_cost_new < log_cost_old + options_.armijo_constant * expected_improvement)
                {
                    filter_acceptance = true;
                }
            }
            else
            {
                if (log_cost_new < log_cost_old - options_.filter_merit_acceptance * constraint_violation_new || constraint_violation_new < (1 - options_.filter_violation_acceptance) * constraint_violation_old)
                {
                    filter_acceptance = true;
                }
            }

            if (filter_acceptance)
            {
                // Update the result with the new trajectories and metrics.
                result.success = true;
                result.state_sequence = X_new;
                result.control_sequence = U_new;
                result.dynamics_sequence = F_new;
                result.lambda_sequence = Lambda_new;
                result.dual_sequence = Y_new;
                result.slack_sequence = S_new;
                result.constraint_sequence = G_new;
                result.cost = cost_new;
                result.lagrangian = log_cost_new;
                result.constraint_violation = constraint_violation_new;
            }

            return result;
        }
        else
        {
            // Constrained forward pass
            double alpha_s = alpha;

            // Update S, U, X with alpha_s
            bool s_trajectory_feasible = true;
            for (int t = 0; t < horizon_; ++t)
            {
                Eigen::VectorXd delta_x_k = X_new[t] - X_[t];

                // Slack update and feasibility check for S_new
                for (auto &ckv : constraint_set_)
                {
                    const std::string &cname = ckv.first;
                    int dual_dim = ckv.second->getDualDim();
                    const Eigen::VectorXd &s_old = S_[cname][t];

                    Eigen::VectorXd s_new = s_old +
                                            alpha_s * k_s_[cname][t] + K_s_[cname][t] * delta_x_k;

                    Eigen::VectorXd s_min = (1.0 - tau) * s_old;
                    for (int i = 0; i < dual_dim; ++i)
                    {
                        if (s_new[i] < s_min[i])
                        {
                            s_trajectory_feasible = false;
                            break; // Exit i-loop
                        }
                    }
                    if (!s_trajectory_feasible)
                        break;               // Exit cname-loop
                    S_new[cname][t] = s_new; // Store if feasible for this constraint
                }
                if (!s_trajectory_feasible)
                    break; // Exit t-loop (horizon)

                // Update control
                const Eigen::VectorXd delta_u_k = alpha * k_u_[t] + K_u_[t] * delta_x_k;
                U_new[t] = U_[t] + delta_u_k;

                // --- Rollout Logic  ---
                Eigen::VectorXd dynamics_eval_for_F_new_t;

                if (options_.ms_rollout_type == "nonlinear" || options_.ms_rollout_type == "hybrid")
                {
                    dynamics_eval_for_F_new_t = system_->getDiscreteDynamics(X_new[t], U_new[t]);
                }
                else // options_.ms_rollout_type == "linear"
                {
                    dynamics_eval_for_F_new_t = F_[t] + A_[t] * delta_x_k + B_[t] * delta_u_k;
                }
                F_new[t] = dynamics_eval_for_F_new_t; // Store the calculated dynamics/approximation

                // Determine if the *next* step (t+1) starts a new segment boundary
                const Eigen::VectorXd defect_at_segment_end = F_[t] - X_[t + 1];
                bool is_segment_boundary = (ms_segment_length_ > 0) &&
                                           ((t + 1) % ms_segment_length_ == 0) &&
                                           (t + 1 < horizon_);
                bool defect_is_large = defect_at_segment_end.lpNorm<Eigen::Infinity>() > options_.ms_defect_tolerance_for_single_shooting;
                bool apply_gap_closing_strategy = is_segment_boundary && defect_is_large;

                if (apply_gap_closing_strategy)
                {
                    if (options_.ms_rollout_type == "nonlinear")
                    {
                        X_new[t + 1] = dynamics_eval_for_F_new_t + alpha * defect_at_segment_end;
                    }
                    else if (options_.ms_rollout_type == "linear" || options_.ms_rollout_type == "hybrid")
                    {
                        X_new[t + 1] = X_[t] + A_[t] * delta_x_k + alpha * B_[t] * k_u_[t] + alpha * defect_at_segment_end;
                    }
                }
                else // Not a segment boundary or defect is small, so no gap closing strategy applied
                {
                    if (options_.ms_rollout_type == "nonlinear" || options_.ms_rollout_type == "hybrid")
                    {
                        X_new[t + 1] = dynamics_eval_for_F_new_t; // This is the nonlinear dynamics evaluation
                    }
                    else // options_.ms_rollout_type == "linear"
                    {
                        X_new[t + 1] = X_[t] + A_[t] * delta_x_k + B_[t] * delta_u_k; // Linear rollout for state
                    }
                }
                // --- End Rollout Logic ---

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
            }

            if (!s_trajectory_feasible)
            {
                // alpha_s was not feasible for the S trajectory
                return result; // result.success is already false by default
            }

            // Update Y with alpha_y
            bool suitable_alpha_y_found = false;
            std::map<std::string, std::vector<Eigen::VectorXd>> Y_trial;

            for (double alpha_y_candidate : alphas_)
            {
                bool current_alpha_y_globally_feasible = true;
                Y_trial = Y_;

                for (int t = 0; t < horizon_; ++t)
                {
                    Eigen::VectorXd delta_x_k = X_new[t] - X_[t];

                    for (auto &ckv : constraint_set_)
                    {
                        const std::string &cname = ckv.first;
                        int dual_dim = ckv.second->getDualDim();
                        const Eigen::VectorXd &y_old = Y_[cname][t];

                        Eigen::VectorXd y_new = y_old +
                                                alpha_y_candidate * k_y_[cname][t] + K_y_[cname][t] * delta_x_k;

                        Eigen::VectorXd y_min = (1.0 - tau) * y_old;
                        for (int i = 0; i < dual_dim; ++i)
                        {
                            if (y_new[i] < y_min[i])
                            {
                                current_alpha_y_globally_feasible = false;
                                break; // Exit i-loop
                            }
                        }
                        if (!current_alpha_y_globally_feasible)
                            break;                 // Exit cname-loop
                        Y_trial[cname][t] = y_new; // Store trial Y for this constraint
                    }
                    if (!current_alpha_y_globally_feasible)
                    {
                        break; // Exit t-loop
                    }

                    // Update lambda
                    Lambda_new[t] = alpha_y_candidate * Lambda_[t] + K_lambda_[t] * delta_x_k;
                }

                if (current_alpha_y_globally_feasible)
                {
                    suitable_alpha_y_found = true;
                    Y_new = Y_trial; // Commit the successful trial Y to Y_new
                    break;           // Found a good alpha_y, exit the inner line search loop
                }
            }

            if (!suitable_alpha_y_found)
            {
                // No feasible alpha_y found for the current alpha_s (even though S was feasible)
                return result; // result.success is already false
            }

            // Cost Computation and filter line-search
            cost_new = 0.0;
            log_cost_new = 0.0;
            rp_err = 0.0;

            for (int t = 0; t < horizon_; ++t)
            {
                cost_new += objective_->running_cost(X_new[t], U_new[t], t);

                for (const auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    // Evaluate constraint value g
                    G_new[cname][t] = cKV.second->evaluate(X_new[t], U_new[t]) - cKV.second->getUpperBound();

                    // Log-barrier term using S_new from alpha_s pass
                    const Eigen::VectorXd &s_vec = S_new[cname][t];
                    log_cost_new -= mu_ * s_vec.array().log().sum();

                    // Primal feasibility r_p: g + s
                    Eigen::VectorXd r_p = G_new[cname][t] + s_vec;
                    rp_err += r_p.lpNorm<1>();
                }

                Eigen::VectorXd d = F_new[t] - X_new[t + 1];
                rf_err += d.lpNorm<1>();
            }

            cost_new += objective_->terminal_cost(X_new.back());
            log_cost_new += cost_new;

            double constraint_violation_old = constraint_violation_;
            double constraint_violation_new = rp_err + rf_err;
            double log_cost_old = L_;
            bool filter_acceptance = false;
            double expected_improvement = alpha * dV_(0);

            if (constraint_violation_new > options_.filter_maximum_violation)
            {
                if (constraint_violation_new < (1 - options_.filter_acceptance) * constraint_violation_old)
                {
                    filter_acceptance = true;
                }
            }
            else if (std::max(constraint_violation_new, constraint_violation_old) < options_.filter_minimum_violation && expected_improvement < 0)
            {
                if (log_cost_new < log_cost_old + options_.armijo_constant * expected_improvement)
                {
                    filter_acceptance = true;
                }
            }
            else
            {
                if (log_cost_new < log_cost_old - options_.filter_merit_acceptance * constraint_violation_old || constraint_violation_new < (1 - options_.filter_violation_acceptance) * constraint_violation_old)
                {
                    filter_acceptance = true;
                }
            }

            if (filter_acceptance)
            {
                // Update the result with the new trajectories and metrics.
                result.success = true;
                result.state_sequence = X_new;
                result.control_sequence = U_new;
                result.dynamics_sequence = F_new;
                result.lambda_sequence = Lambda_new;
                result.dual_sequence = Y_new;
                result.slack_sequence = S_new;
                result.constraint_sequence = G_new;
                result.cost = cost_new;
                result.lagrangian = log_cost_new;
                result.constraint_violation = constraint_violation_new;
            }
            return result;
        }
    } // end solveMSIPDDPForwardPass

    // TODO: Rename resetIPDDPFilter -> resetMSIPDDPFilter
    void CDDP::resetMSIPDDPFilter()
    {
        // Evaluate log-barrier cost (includes path constraints)
        L_ = J_;             // Assume J_ (total cost) is computed from a rollout
        double rp_err = 0.0; // Path constraint violation
        double rf_err = 0.0; // Gap violation

        // Calculate path constraint terms and violation
        for (int t = 0; t < horizon_; ++t)
        {
            for (const auto &cKV : constraint_set_) // Loop over path constraints
            {
                const std::string &cname = cKV.first;
                const Eigen::VectorXd &s_vec = S_[cname][t];
                const Eigen::VectorXd &g_vec = G_[cname][t]; // Assumes G_ is updated

                L_ -= mu_ * s_vec.array().log().sum();
                rp_err += (s_vec + g_vec).lpNorm<1>();
            }

            // Add defect violation penalty
            Eigen::VectorXd d = F_[t] - X_[t + 1];
            rf_err += d.lpNorm<1>();
        }

        constraint_violation_ = rp_err + rf_err;
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
                G_[cname][t] = g_vec - cKV.second->getUpperBound();
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
