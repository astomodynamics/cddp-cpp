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

namespace cddp
{

    void CDDP::initializeLogDDP()
    {
        if (!system_)
        {
            initialized_ = false;
            if (options_.verbose)
            {
                std::cerr << "LogDDP: No dynamical system provided." << std::endl;
            }
            return;
        }

        if (!objective_)
        {
            initialized_ = false;
            if (options_.verbose)
            {
                std::cerr << "LogDDP: No objective function provided." << std::endl; // Updated message
            }
            return;
        }

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();

        // Check if reference_state in objective and reference_state in IPDDP are the same
        if ((reference_state_ - objective_->getReferenceState()).norm() > 1e-6)
        {
            std::cerr << "LogDDP: Initial state and goal state in the objective function do not match" << std::endl;
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
                std::cout << "LogDDP: State regularization is not enabled for LogDDP" << std::endl;
            }
        }

        if (options_.regularization_type == "control" || options_.regularization_type == "both")
        {
            regularization_control_ = options_.regularization_control;
            regularization_control_step_ = options_.regularization_control_step;

            if (options_.debug)
            {
                std::cout << "LogDDP: Control regularization is enabled for LogDDP" << std::endl;
            }
        }
        else
        {
            regularization_control_ = 0.0;
            regularization_control_step_ = 1.0;
        }

        constraint_violation_ = 1e+7;

        ms_segment_length_ = options_.ms_segment_length;

        // Check if ms_segment_length_ is valid
        if (ms_segment_length_ < 0)
        {
            std::cerr << "LogDDP: ms_segment_length_ must be non-negative" << std::endl;
            throw std::runtime_error("LogDDP: ms_segment_length_ must be non-negative");
        }

        if (options_.ms_rollout_type != "linear" && options_.ms_rollout_type != "nonlinear" && options_.ms_rollout_type != "hybrid")
        {
            std::cerr << "LogDDP: Invalid ms_rollout_type: " << options_.ms_rollout_type << std::endl;
            throw std::runtime_error("LogDDP: Invalid ms_rollout_type");
        }

        // Initialize log barrier object
        mu_ = options_.barrier_coeff; // Initialize mu_ here as well
        relaxation_delta_ = options_.relaxation_delta;
        if (!relaxed_log_barrier_)
        { // Create if it doesn't exist
            relaxed_log_barrier_ = std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);
        }

        // Now initialized
        initialized_ = true;
        return;
    }

    CDDPSolution CDDP::solveLogDDP()
    {
        // Initialize if not done
        if (!initialized_ || !relaxed_log_barrier_)
        {
            initializeLogDDP();
        }

        if (!initialized_)
        {
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
        for (int t = 0; t <= horizon_; ++t)
        {
            solution.time_sequence.push_back(timestep_ * t);
        }
        solution.control_sequence.reserve(horizon_);
        solution.state_sequence.reserve(horizon_ + 1);
        solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency
        solution.lagrangian_sequence.reserve(options_.max_iterations);

        // Initialize trajectories and gaps
        initialLogDDPRollout(); // J_ is computed inside this function
        solution.cost_sequence.push_back(J_);

        // Reset LogDDP filter
        resetLogDDPFilter(); // L_ and constraint_violation_ are computed inside this function
        solution.lagrangian_sequence.push_back(L_);

        if (options_.verbose)
        {
            printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); // Initial iteration information
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        ForwardPassResult best_result;

        // Main loop of CDDP
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
                    if (options_.verbose)
                    {
                        std::cerr << "LogDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                    }
                    break;
                }
            }

            // 1. Backward pass: Solve Riccati recursion to compute optimal control law
            bool backward_pass_success = false;
            while (!backward_pass_success)
            {
                backward_pass_success = solveLogDDPBackwardPass();

                if (!backward_pass_success)
                {
                    if (options_.debug)
                    {
                        std::cerr << "LogDDP: Backward pass failed" << std::endl;
                    }

                    // Increase regularization
                    increaseRegularization();

                    if (isRegularizationLimitReached())
                    {
                        if (options_.verbose)
                        {
                            std::cerr << "LogDDP: Backward pass regularization limit reached!" << std::endl;
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

            // 2. Forward pass (either single-threaded or multi-threaded)
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
                    ForwardPassResult result = solveLogDDPForwardPass(alpha);

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
                                                     return solveLogDDPForwardPass(alpha);
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
                            std::cerr << "LogDDP: Forward pass thread failed: " << e.what() << std::endl;
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
                    std::cout << "[LogDDP: Forward pass] " << std::endl; // Updated message
                    std::cout << "    cost: " << best_result.cost << std::endl;
                    std::cout << "    logcost: " << best_result.lagrangian << std::endl;
                    std::cout << "    alpha: " << best_result.alpha << std::endl;
                    std::cout << "    rf_err: " << best_result.constraint_violation << std::endl; // Defect constraints
                }
                X_ = best_result.state_sequence;
                U_ = best_result.control_sequence;
                F_ = best_result.dynamics_sequence;

                dJ_ = J_ - best_result.cost;
                J_ = best_result.cost;
                dL_ = L_ - best_result.lagrangian;
                L_ = best_result.lagrangian;
                alpha_ = best_result.alpha;
                constraint_violation_ = best_result.constraint_violation; // Defect constraint violation

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
            double termination_metric = std::max(optimality_gap_, constraint_violation_);
            if (termination_metric <= options_.grad_tolerance)
            {
                if (options_.debug)
                {
                    std::cout << "LogDDP: Converged due to optimality gap and constraint violation." << std::endl;
                }
                solution.converged = true;
                break;
            }

            // TODO: This should be removed
            if (abs(dJ_) < options_.cost_tolerance && abs(dL_) < options_.cost_tolerance && termination_metric <= options_.grad_tolerance * 100.0)
            {
                if (options_.debug)
                {
                    std::cout << "LogDDP: Converged due to small change in cost and Lagrangian." << std::endl;
                }
                solution.converged = true;
                break;
            }

            // Barrier update logic
            if (forward_pass_success && termination_metric < options_.grad_tolerance)
            {
                // Dramatically decrease mu if optimization is going well
                mu_ = std::max(mu_ * 0.1, options_.barrier_tolerance);
                relaxation_delta_ = std::max(options_.grad_tolerance / 10.0, std::min(relaxation_delta_ * 0.1, std::pow(relaxation_delta_, options_.barrier_update_power)));
                resetLogDDPFilter();
            }
            else
            {
                // Normal decrease rate
                mu_ = std::max(options_.grad_tolerance / 10.0, std::min(options_.barrier_update_factor * mu_, std::pow(mu_, options_.barrier_update_power)));
                relaxation_delta_ = std::max(options_.grad_tolerance / 10.0, std::min(relaxation_delta_ * 0.1, std::pow(relaxation_delta_, options_.barrier_update_power)));
                resetLogDDPFilter();
            }

            relaxed_log_barrier_->setBarrierCoeff(mu_);
            relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Finalize solution
        solution.state_sequence = X_;
        solution.control_sequence = U_;
        solution.alpha = alpha_;
        solution.solve_time = duration.count(); // Time in microseconds

        if (options_.verbose)
        {
            printSolution(solution);
        }

        return solution;
    }

    bool CDDP::solveLogDDPBackwardPass()
    {
        // Initialize variables
        const int state_dim = system_->getStateDim();
        const int control_dim = system_->getControlDim();

        // Terminal cost and derivatives (V_x, V_xx at t=N)
        Eigen::VectorXd V_x = objective_->getFinalCostGradient(X_.back());
        Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

        dV_ = Eigen::Vector2d::Zero();
        double Qu_err = 0.0;

        // --- Pre-computation Phase: Compute and store linearized dynamics --- TODO: Parallelize this
        for (int t = 0; t < horizon_; ++t)
        {
            const Eigen::VectorXd &x = X_[t];
            const Eigen::VectorXd &u = U_[t];

            // Dynamics Jacobians
            const auto [Fx, Fu] = system_->getJacobians(x, u, t * timestep_);
            Fx_[t] = Fx;
            Fu_[t] = Fu;

            // Linearized dynamics matrices
            A_[t] = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx_[t];
            B_[t] = timestep_ * Fu_[t];

            // Dynamics Hessians
            if (!options_.is_ilqr)
            {
                const auto hessians = system_->getHessians(x, u, t * timestep_);
                Fxx_[t] = std::get<0>(hessians);
                Fuu_[t] = std::get<1>(hessians);
                Fux_[t] = std::get<2>(hessians);
            }
        }
        // --- End Pre-computation Phase ---

        // Backward Riccati recursion
        int t = horizon_ - 1;
        while (t >= 0)
        {
            const Eigen::VectorXd &x = X_[t];         // Initial state of interval t
            const Eigen::VectorXd &u = U_[t];         // Control for interval t
            const Eigen::VectorXd &f = F_[t];         // Dynamics at interval t
            const Eigen::VectorXd &d = f - X_[t + 1]; // Defect
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

            // Apply Log-barrier cost gradients and Hessians
            for (const auto &constraint_pair : constraint_set_) // Renamed to avoid conflict
            {
                auto [L_x_relaxed, L_u_relaxed] = relaxed_log_barrier_->getGradients(*constraint_pair.second, x, u);
                Q_x += L_x_relaxed;
                Q_u += L_u_relaxed;

                auto [L_xx_relaxed, L_uu_relaxed, L_ux_relaxed] = relaxed_log_barrier_->getHessians(*constraint_pair.second, x, u);
                Q_xx += L_xx_relaxed;
                Q_uu += L_uu_relaxed;
                Q_ux += L_ux_relaxed;
            }

            // Regularization
            Eigen::MatrixXd Q_uu_reg = Q_uu;
            // Apply regularization
            Q_uu_reg.diagonal().array() += regularization_control_;
            Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

            Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
            if (ldlt.info() != Eigen::Success)
            {
                if (options_.debug)
                {
                    std::cerr << "LogDDP: Backward pass failed at time " << t << std::endl;
                }
                return false;
            }

            Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
            bigRHS.col(0) = Q_u;
            Eigen::MatrixXd M = Q_ux;
            for (int col = 0; col < state_dim; col++)
            {
                bigRHS.col(col + 1) = M.col(col);
            }

            Eigen::MatrixXd kK = -ldlt.solve(bigRHS);

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

            // Compute value function approximation
            Eigen::Vector2d dV_step;
            dV_step << Q_u.dot(k_u), 0.5 * k_u.dot(Q_uu * k_u);
            dV_ = dV_ + dV_step;
            V_x = Q_x + K_u.transpose() * Q_uu * k_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_u;
            V_xx = Q_xx + K_u.transpose() * Q_uu * K_u + Q_ux.transpose() * K_u + K_u.transpose() * Q_ux;
            V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize Hessian

            // Compute optimality gap (Inf-norm) for convergence check
            Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());

            optimality_gap_ = Qu_err;

            t--;
        }

        if (options_.debug)
        {
            std::cout << "[LogDDP Backward Pass]\n"
                      << "    Qu_err:  " << Qu_err << "\n"
                      << "    rf_err:  " << constraint_violation_ << "\n"
                      << "    dV:      " << dV_.transpose() << std::endl;
        }

        return true;
    }

    ForwardPassResult CDDP::solveLogDDPForwardPass(double alpha)
    {
        // Prepare result struct
        ForwardPassResult result;
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.lagrangian = std::numeric_limits<double>::infinity();
        result.alpha = alpha;

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();

        // Filter acceptance
        double filter_merit_acceptance = options_.filter_merit_acceptance;
        double filter_violation_acceptance = options_.filter_violation_acceptance;

        // Initialize trajectories
        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        std::vector<Eigen::VectorXd> F_new = F_;

        // Set the initial state
        X_new[0] = initial_state_;

        // Initialize cost_new for this alpha trial
        double cost_new = 0.0;
        double log_cost_new = 0.0;
        double rf_err = 0.0;

        // Rollout loop
        for (int t = 0; t < horizon_; ++t)
        {
            const Eigen::VectorXd delta_x_t = X_new[t] - X_[t];

            // Determine if the *next* step (t+1) starts a new segment boundary
            bool is_segment_boundary = (ms_segment_length_ > 0) &&
                                       ((t + 1) % ms_segment_length_ == 0) &&
                                       (t + 1 < horizon_);
            bool apply_gap_closing_strategy = is_segment_boundary;

            // Update control
            Eigen::VectorXd delta_u_k = alpha * k_u_[t] + K_u_[t] * delta_x_t;
            U_new[t] = U_[t] + delta_u_k;
            if (apply_gap_closing_strategy)
            {
                if (options_.ms_rollout_type == "nonlinear")
                {
                    F_new[t] = system_->getDiscreteDynamics(X_new[t], U_new[t], t * timestep_);
                    X_new[t + 1] = X_[t + 1] + (F_new[t] - F_[t]) + alpha * (F_[t] - X_[t + 1]);
                }
                else if (options_.ms_rollout_type == "hybrid")
                {
                    F_new[t] = system_->getDiscreteDynamics(X_new[t], U_new[t], t * timestep_);

                    // Continuous dynamics
                    const auto [Fx, Fu] = system_->getJacobians(X_[t], U_[t], t * timestep_);
                    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx;
                    Eigen::MatrixXd B = timestep_ * Fu;

                    X_new[t + 1] = X_[t + 1] + (A + B * K_u_[t]) * delta_x_t + alpha * (B * k_u_[t] + F_[t] - X_[t + 1]);
                }
            }
            else
            {
                F_new[t] = system_->getDiscreteDynamics(X_new[t], U_new[t], t * timestep_);
                X_new[t + 1] = F_new[t];
            }

            // --- Robustness Check during Rollout ---
            if (!X_new[t + 1].allFinite() || !U_new[t].allFinite())
            {
                if (options_.debug)
                {
                    std::cerr << "[LogDDP Forward Pass] NaN/Inf detected during HYBRID rollout (nonlinear within, linear between) at t=" << t
                              << " for alpha=" << alpha << std::endl; // Updated debug message
                }
                result.success = false;
                cost_new = std::numeric_limits<double>::infinity();
                return result;
            }
        }

        // Cost Computation and filter line-search
        cost_new = 0.0;
        log_cost_new = 0.0;
        rf_err = 0.0;

        for (int t = 0; t < horizon_; ++t)
        {
            cost_new += objective_->running_cost(X_new[t], U_new[t], t);

            for (const auto &cKV : constraint_set_)
            {
                const std::string &cname = cKV.first;

                // Evaluate constraint value g
                log_cost_new += relaxed_log_barrier_->evaluate(*cKV.second, X_new[t], U_new[t]);
            }

            Eigen::VectorXd d = F_new[t] - X_new[t + 1];
            rf_err += d.lpNorm<1>();
        }

        cost_new += objective_->terminal_cost(X_new.back());
        log_cost_new += cost_new;

        double constraint_violation_old = constraint_violation_;
        double constraint_violation_new = rf_err;
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
            result.cost = cost_new;
            result.lagrangian = log_cost_new;
            result.constraint_violation = constraint_violation_new;
        }
        return result;
    }

    void CDDP::resetLogDDPFilter()
    {
        // Evaluate log-barrier cost (includes path constraints)
        L_ = J_; // Assume J_ (total cost) is computed from a rollout
        double defect_violation = 0.0;

        // Calculate path constraint terms and violation
        for (int t = 0; t < horizon_; ++t)
        {
            for (const auto &cKV : constraint_set_) // Loop over path constraints
            {
                const std::string &cname = cKV.first;
                L_ += relaxed_log_barrier_->evaluate(*cKV.second, X_[t], U_[t]);
            }

            // Add defect violation penalty
            Eigen::VectorXd d = F_[t] - X_[t + 1];
            defect_violation += d.lpNorm<1>();
        }
        constraint_violation_ = defect_violation;
        return;
    }

    void CDDP::initialLogDDPRollout()
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

            // Compute defect
            Eigen::VectorXd f = system_->getDiscreteDynamics(x_t, u_t, t * timestep_);
            F_[t] = f;
        }

        // Add terminal cost based on the final *guessed* state X_[N]
        cost += objective_->terminal_cost(X_.back());

        // Store the initial total cost.
        J_ = cost;

        return;
    }
} // namespace cddp