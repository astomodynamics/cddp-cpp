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

    void CDDP::initializeIPDDP()
    {
        if (!system_)
        {
            initialized_ = false;
            if (options_.verbose)
            {
                std::cerr << "IPDDP::initializeIPDDP: No dynamical system provided." << std::endl;
            }
            return;
        }

        if (!objective_)
        {
            initialized_ = false;
            if (options_.verbose)
            {
                std::cerr << "IPDDP::initializeIPDDP: No objective function provided." << std::endl;
            }
            return;
        }

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();

        // Check if reference_state in objective and reference_state in IPDDP are the same
        if ((reference_state_ - objective_->getReferenceState()).norm() > 1e-6)
        {
            std::cerr << "IPDDP: Initial state and goal state in the objective function do not match" << std::endl;
            throw std::runtime_error("Initial state and goal state in the objective function do not match");
        }

        // Initialize trajectories (X_ and U_ are std::vectors of Eigen::VectorXd)
        if (X_.size() != horizon_ + 1 && U_.size() != horizon_)
        {
            X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
            U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
        }
        else if (X_.size() != horizon_ + 1)
        {
            X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
        }
        else if (U_.size() != horizon_)
        {
            U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
        }

        k_u_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
        K_u_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));

        G_.clear(); // Constraint value
        Y_.clear(); // Dual variable
        S_.clear(); // Slack variable
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

        if (constraint_set_.empty())
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

    CDDPSolution CDDP::solveIPDDP()
    {
        // Initialize if not done or if S_ is not defined
        if (!initialized_ || S_.empty())
        {
            initializeIPDDP();
        }

        if (!initialized_)
        {
            std::cerr << "IPDDP: Initialization failed" << std::endl;
            throw std::runtime_error("IPDDP: Initialization failed");
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

        // Initialize trajectories
        initialIPDDPRollout(); // J_ is computed inside this function
        solution.cost_sequence.push_back(J_);

        // Reset IPDDP filter
        resetIPDDPFilter(); // L_ is computed inside this function
        solution.lagrangian_sequence.push_back(L_);

        // Reset regularization
        resetIPDDPRegularization();

        if (options_.verbose)
        {
            printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); // Initial iteration information
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        ForwardPassResult best_result;
        ipddp_regularization_counter_ = 0; // Reset regularization counter

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
                    if (options_.debug)
                    {
                        std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                    }
                    // TODO: Treat as convergence
                    solution.converged = true;
                    break;
                }
            }

            // 1. Backward pass: Solve Riccati recursion to compute optimal control law
            bool backward_pass_success = false;
            while (!backward_pass_success)
            {
                backward_pass_success = solveIPDDPBackwardPass();

                if (!backward_pass_success)
                {
                    if (options_.debug)
                    {
                        std::cerr << "IPDDP: Backward pass failed" << std::endl;
                    }

                    // Increase regularization
                    increaseRegularization();

                    if (isRegularizationLimitReached())
                    {
                        if (options_.verbose)
                        {
                            std::cerr << "IPDDP: Backward pass regularization limit reached!" << std::endl;
                        }
                        // TODO: Treat as convergence 
                        solution.converged = true;
                        break; // Exit if regularization limit reached
                    }
                    continue; // Continue if backward pass fails
                }
            }

            // 2. Forward pass (either single-threaded or multi-threaded)
            best_result.success = false;
            best_result.cost = std::numeric_limits<double>::infinity();
            best_result.lagrangian = std::numeric_limits<double>::infinity();
            best_result.constraint_violation = 0.0;
            bool forward_pass_success = false;

            if (!options_.use_parallel)
            {
                // Single-threaded execution with early termination
                for (double alpha : alphas_)
                {
                    ForwardPassResult result = solveIPDDPForwardPass(alpha);

                    if (result.success)
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
                                                 { return solveIPDDPForwardPass(alpha); }));
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
                            std::cerr << "CDDP: Forward pass thread failed: " << e.what() << std::endl;
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
                    std::cout << "[IPDDP: Forward pass] " << std::endl;
                    std::cout << "    cost: " << best_result.cost << std::endl;
                    std::cout << "    logcost: " << best_result.lagrangian << std::endl;
                    std::cout << "    alpha: " << best_result.alpha << std::endl;
                    std::cout << "    rp_err: " << best_result.constraint_violation << std::endl;
                }
                X_ = best_result.state_sequence;
                U_ = best_result.control_sequence;
                Y_ = best_result.dual_sequence;
                S_ = best_result.slack_sequence;
                G_ = best_result.constraint_sequence;
                dJ_ = J_ - best_result.cost;
                J_ = best_result.cost;
                dL_ = L_ - best_result.lagrangian;
                L_ = best_result.lagrangian;
                alpha_ = best_result.alpha;

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
                        std::cerr << "CDDP: Forward Pass regularization limit reached" << std::endl;
                    }

                    // TODO: Treat as convergence
                    solution.converged = true;

                    break;
                }
            }

            // Print iteration information
            if (options_.verbose)
            {
                printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_);
            }

            // Check termination
            if (std::max(optimality_gap_, mu_) <= options_.cost_tolerance)
            {
                solution.converged = true;
                break;
            }

            // From original IPDDP implementation
            if (optimality_gap_ <= 0.2 * mu_)
            {   
                if (constraint_set_.empty()) {
                }
                else {
                    mu_ = std::max(options_.cost_tolerance / 10.0, std::min(0.2 * mu_, std::pow(mu_, 1.2)));
                }
                resetIPDDPFilter();
                resetIPDDPRegularization();
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Finalize solution
        solution.state_sequence = X_;
        solution.control_sequence = U_;
        solution.alpha = alpha_;
        solution.solve_time = duration.count(); // Time in microseconds

        if (options_.header_and_footer)
        {
            printSolution(solution);
        }

        return solution;
    }

    bool CDDP::solveIPDDPBackwardPass()
    {
        // Setup
        // Initialize variables
        const int state_dim = getStateDim();
        const int control_dim = getControlDim();
        const int total_dual_dim = getTotalDualDim(); // Number of dual variables across all constraints

        // Terminal cost and derivatives
        Eigen::VectorXd V_x = objective_->getFinalCostGradient(X_.back());
        Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

        dV_ = Eigen::Vector2d::Zero();
        double Qu_err = 0.0;
        double rp_err = 0.0; // primal feasibility
        double rd_err = 0.0; // dual feasibility

        // If no constraints, use standard DDP recursion.
        if (constraint_set_.empty())
        {
            for (int t = horizon_ - 1; t >= 0; --t)
            {
                const Eigen::VectorXd &x = X_[t];
                const Eigen::VectorXd &u = U_[t];
                const auto [Fx, Fu] = system_->getJacobians(x, u);
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

                double l = objective_->running_cost(x, u, t);
                auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

                Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
                Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
                
                // Standard Q_xx calculation
                Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
                
                // Add state hessian term if not using iLQR
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_xx += timestep_ * V_x(i) * Fxx[i];
                    }
                }
                
                // Standard Q_ux calculation
                Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
                
                // Add cross hessian term if not using iLQR
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_ux += timestep_ * V_x(i) * Fux[i];
                    }
                }
                
                // Standard Q_uu calculation
                Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;
                
                // Add control hessian term if not using iLQR
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_uu += timestep_ * V_x(i) * Fuu[i];
                    }
                }

                // Regularization
                Eigen::MatrixXd Q_ux_reg = Q_ux;
                Eigen::MatrixXd Q_uu_reg = Q_uu;

                if (options_.regularization_type == "state" ||
                    options_.regularization_type == "both")
                {
                    // Apply regularization to the value function Hessian
                    Eigen::MatrixXd V_xx_reg = V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim);
                    
                    // Recompute Q_ux and Q_uu with regularized V_xx
                    Q_ux_reg = l_ux + B.transpose() * V_xx_reg * A;
                    Q_uu_reg = l_uu + B.transpose() * V_xx_reg * B;
                    
                    // Add hessian terms with regularized V_xx if not using iLQR
                    if (!options_.is_ilqr) {
                        for (int i = 0; i < state_dim; ++i) {
                            Q_ux_reg += timestep_ * V_x(i) * Fux[i];
                            Q_uu_reg += timestep_ * V_x(i) * Fuu[i];
                        }
                    }
                }
                else
                {
                    Q_ux_reg = Q_ux;
                    Q_uu_reg = Q_uu;
                }

                if (options_.regularization_type == "control" ||
                    options_.regularization_type == "both")
                {
                    Q_uu_reg.diagonal().array() += regularization_control_;
                }
                Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

                Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg);
                if (llt.info() != Eigen::Success)
                {
                    return false;
                }
                Eigen::VectorXd k_u = -llt.solve(Q_u);
                Eigen::MatrixXd K_u = -llt.solve(Q_ux);
                k_u_[t] = k_u;
                K_u_[t] = K_u;

                // Update value function
                V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
                V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;

                // Accumulate cost improvement (if desired)
                dV_[0] += k_u.dot(Q_u);
                dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

                // Error tracking
                Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
            } // end for t

            optimality_gap_ = Qu_err;
            return true;
        }
        else
        {
            // Backward Recursion
            for (int t = horizon_ - 1; t >= 0; --t)
            {
                // Expand cost around (x[t], u[t])
                const Eigen::VectorXd &x = X_[t];
                const Eigen::VectorXd &u = U_[t];

                // Continuous dynamics
                const auto [Fx, Fu] = system_->getJacobians(x, u);

                // Get dynamics hessians if not using iLQR
                std::vector<Eigen::MatrixXd> Fxx, Fuu, Fux;
                if (!options_.is_ilqr) {
                    const auto hessians = system_->getHessians(x, u);
                    Fxx = std::get<0>(hessians);
                    Fuu = std::get<1>(hessians);
                    Fux = std::get<2>(hessians);
                }

                // Discretize
                Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx;
                Eigen::MatrixXd B = timestep_ * Fu;

                // Gather dual and slack variables, and constraint values across all constraints
                Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd s = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);
                Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);

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

                    offset += dual_dim;
                }

                // Cost & derivatives
                double l = objective_->running_cost(x, u, t);
                auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

                // Q expansions from cost
                Eigen::VectorXd Q_x = l_x + Q_yx.transpose() * y + A.transpose() * V_x;
                Eigen::VectorXd Q_u = l_u + Q_yu.transpose() * y + B.transpose() * V_x;
                
                // Standard Q_xx calculation
                Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
                
                // Add state hessian term if not using iLQR
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_xx += timestep_ * V_x(i) * Fxx[i];
                    }
                }
                
                // Standard Q_ux calculation
                Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
                
                // Add cross hessian term if not using iLQR
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_ux += timestep_ * V_x(i) * Fux[i];
                    }
                }
                
                // Standard Q_uu calculation
                Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;
                
                // Add control hessian term if not using iLQR
                if (!options_.is_ilqr) {
                    for (int i = 0; i < state_dim; ++i) {
                        Q_uu += timestep_ * V_x(i) * Fuu[i];
                    }
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
                Eigen::MatrixXd Q_ux_reg = Q_ux;
                Eigen::MatrixXd Q_uu_reg = Q_uu;

                if (options_.regularization_type == "state" ||
                    options_.regularization_type == "both")
                {
                    // Apply regularization to the value function Hessian
                    Eigen::MatrixXd V_xx_reg = V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim);
                    
                    // Recompute Q_ux and Q_uu with regularized V_xx
                    Q_ux_reg = l_ux + B.transpose() * V_xx_reg * A;
                    Q_uu_reg = l_uu + B.transpose() * V_xx_reg * B;
                    
                    // Add hessian terms with regularized V_xx if not using iLQR
                    if (!options_.is_ilqr) {
                        for (int i = 0; i < state_dim; ++i) {
                            Q_ux_reg += timestep_ * V_x(i) * Fux[i];
                            Q_uu_reg += timestep_ * V_x(i) * Fuu[i];
                        }
                    }
                }
                else
                {
                    Q_ux_reg = Q_ux;
                    Q_uu_reg = Q_uu;
                }

                if (options_.regularization_type == "control" ||
                    options_.regularization_type == "both")
                {
                    Q_uu_reg.diagonal().array() += regularization_control_;
                }
                Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

                Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg + Q_yu.transpose() * YSinv * Q_yu);
                if (llt.info() != Eigen::Success)
                {
                    if (options_.debug)
                    {
                        std::cerr << "CDDP: Backward pass failed at time " << t << std::endl;
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
            } // end for t

            // Compute optimality gap and print
            optimality_gap_ = std::max(Qu_err, std::max(rp_err, rd_err));

            if (options_.debug)
            {
                std::cout << "[IPDDP Backward Pass]\n"
                          << "    Qu_err:  " << Qu_err << "\n"
                          << "    rp_err:  " << rp_err << "\n"
                          << "    rd_err:  " << rd_err << "\n"
                          << "    dV:      " << dV_.transpose() << std::endl;
            }
            return true;
        }
        return false;
    } // end solveIPDDPBackwardPass

    ForwardPassResult CDDP::solveIPDDPForwardPass(double alpha)
    {
        // Prepare result structure with default (failure) values.
        ForwardPassResult result;
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.lagrangian = std::numeric_limits<double>::infinity();
        result.alpha = alpha;

        const int state_dim = getStateDim();
        const int control_dim = getControlDim();
        const int dual_dim = getTotalDualDim();

        // Define tau as a safeguard parameter.
        double tau = std::max(0.99, 1.0 - mu_);

        // Copy the current (old) trajectories:
        //   - X_: state trajectory
        //   - U_: control trajectory
        //   - Y_: dual trajectory (indexed by constraint name)
        //   - S_: slack trajectory (indexed by constraint name)
        //   - G_: constraint value trajectory (indexed by constraint name)

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
        std::map<std::string, std::vector<Eigen::VectorXd>> G_new = G_;

        // Set the initial state explicitly.
        X_new[0] = initial_state_;

        // Initialize cost accumulators and measures.
        double cost_new = 0.0;
        double log_cost_new = 0.0;
        double rp_err = 0.0;

        // The current filter point is taken from the previous iteration.
        FilterPoint current{L_, constraint_violation_};

        // --- Forward pass loop ---
        if (constraint_set_.empty())
        {
            // Unconstrained forward pass
            for (int t = 0; t < horizon_; ++t)
            {
                // Update control using the unconstrained gains.
                U_new[t] = U_[t] + alpha * k_u_[t] + K_u_[t] * (X_new[t] - X_[t]);
                // Propagate dynamics.
                X_new[t + 1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
                // Accumulate stage cost.
                cost_new += objective_->running_cost(X_new[t], U_new[t], t);
            }
            cost_new += objective_->terminal_cost(X_new.back());
            double dJ = J_ - cost_new;
            double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
            double reduction_ratio = expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

            result.success = reduction_ratio > options_.minimum_reduction_ratio;
            result.state_sequence = X_new;
            result.control_sequence = U_new;
            result.cost = cost_new;
            result.lagrangian = cost_new;
            result.constraint_violation = 0.0;
            return result;
        }
        else
        {
            // Constrained forward pass
            for (int t = 0; t < horizon_; ++t)
            {
                // Dual and slack update.
                // For each constraint, update the dual and slack variables.
                //   y_new = y_old + alpha * k_y + K_y*(x_new - x_old)
                //   s_new = s_old + alpha * k_s + K_s*(x_new - x_old)
                for (auto &ckv : constraint_set_)
                {
                    const std::string &cname = ckv.first;
                    int dual_dim = ckv.second->getDualDim();

                    // Retrieve old dual/slack from previous solution.
                    const Eigen::VectorXd &y_old = Y_[cname][t];
                    const Eigen::VectorXd &s_old = S_[cname][t];

                    // Compute updated dual/slack variables.
                    Eigen::VectorXd y_new = y_old +
                                            alpha * k_y_[cname][t] + K_y_[cname][t] * (X_new[t] - X_[t]);

                    Eigen::VectorXd s_new = s_old +
                                            alpha * k_s_[cname][t] + K_s_[cname][t] * (X_new[t] - X_[t]);

                    // Enforce minimal feasibility:
                    Eigen::VectorXd y_min = (1.0 - tau) * y_old;
                    Eigen::VectorXd s_min = (1.0 - tau) * s_old;
                    for (int i = 0; i < dual_dim; i++)
                    {
                        if (y_new[i] < y_min[i] || s_new[i] < s_min[i])
                        {
                            // Early exit: feasibility condition violated.
                            return result;
                        }
                    }
                    // Store the updated dual and slack.
                    Y_new[cname][t] = y_new;
                    S_new[cname][t] = s_new;
                }

                // Update control.
                // The new control is computed as:
                //   u_new = u_old + alpha * k_u + K_u*(x_new - x_old)
                const Eigen::VectorXd &u_old = U_[t];
                U_new[t] = u_old + alpha * k_u_[t] + K_u_[t] * (X_new[t] - X_[t]);

                // Propagate the dynamics:
                X_new[t + 1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
            } // end for t

            // Compute the cost metrics.
            for (int t = 0; t < horizon_; ++t)
            {
                // Compute the stage cost.
                double stage_cost = objective_->running_cost(X_new[t], U_new[t], t);
                cost_new += stage_cost;

                // For each constraint, evaluate and store the constraint value.
                for (const auto &cKV : constraint_set_)
                {
                    const std::string &cname = cKV.first;
                    // Evaluate constraint: g = constraint->evaluate(x, u)
                    Eigen::VectorXd g_vec = cKV.second->evaluate(X_new[t], U_new[t]); // dimension = dual_dim

                    // Store the constraint value.
                    G_new[cname][t] = g_vec - cKV.second->getUpperBound();
                    // Compute the slack and log-barrier term.
                    const Eigen::VectorXd &s_vec = S_new[cname][t];
                    log_cost_new -= mu_ * s_vec.array().log().sum();

                    // Compute the residual for primal feasibility.
                    Eigen::VectorXd r_p = G_new[cname][t] + S_new[cname][t];
                    rp_err += r_p.lpNorm<1>();
                }
            }

            // Add terminal cost.
            cost_new += objective_->terminal_cost(X_new.back());
            log_cost_new += cost_new;

            // Compute the primal residual.
            rp_err = std::max(rp_err, options_.cost_tolerance);

            // Build a candidate filter point from the computed cost metrics.
            FilterPoint candidate{log_cost_new, rp_err};

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

                // Update the filter with the candidate point.
                double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
                double reduction_ratio = expected > 0.0
                                             ? (J_ - cost_new) / expected
                                             : std::copysign(1.0, J_ - cost_new);

                // Update the result with the new trajectories and metrics.
                result.success = true;
                result.state_sequence = X_new;
                result.control_sequence = U_new;
                result.dual_sequence = Y_new;
                result.slack_sequence = S_new;
                result.constraint_sequence = G_new;
                result.cost = cost_new;
                result.lagrangian = log_cost_new;
                result.constraint_violation = rp_err;
            }

            return result;
        }
    } // end solveIPDDPForwardPass

    void CDDP::resetIPDDPFilter()
    {
        // Evaluate log-barrier cost
        L_ = J_; // Assume J_ is already computed
        double rp_err = 0.0;
        filter_ = {};
        // # TODO: accelerate this process
        for (int t = 0; t < horizon_; ++t)
        {
            for (const auto &cKV : constraint_set_)
            {
                const std::string &cname = cKV.first;

                // Get slack vector for this constraint at time t.
                const Eigen::VectorXd &s_vec = S_[cname][t];
                const Eigen::VectorXd &g_vec = G_[cname][t];

                L_ -= mu_ * s_vec.array().log().sum();
                rp_err += (s_vec + g_vec).lpNorm<1>();
            }
        }

        // If total violation is below the tolerance, set it to zero.
        if (rp_err < options_.cost_tolerance)
        {
            rp_err = 0.0;
        }
        constraint_violation_ = rp_err;

        // Update the filter vector
        filter_.push_back(FilterPoint(L_, rp_err));
        return;
    }

    void CDDP::initialIPDDPRollout()
    {
        // Initialize the state trajectory with the initial state.
        X_[0] = initial_state_;

        double cost = 0.0;

        // Rollout the system dynamics using the current control sequence.
        for (int t = 0; t < horizon_; ++t)
        {
            // Get the current state and control.
            const Eigen::VectorXd &x = X_[t];
            const Eigen::VectorXd &u = U_[t];

            // Compute and accumulate the stage cost.
            double stage_cost = objective_->running_cost(x, u, t);
            cost += stage_cost;

            // For each constraint, evaluate and store the constraint value.
            for (const auto &cKV : constraint_set_)
            {
                const std::string &cname = cKV.first;
                // Evaluate constraint: g = constraint->evaluate(x, u)
                Eigen::VectorXd g_vec = cKV.second->evaluate(x, u); // dimension = dual_dim

                // Store the constraint value.
                G_[cname][t] = g_vec - cKV.second->getUpperBound();
            }

            // Compute the next state using the system dynamics.
            X_[t + 1] = system_->getDiscreteDynamics(x, u);
        }

        // Add terminal cost.
        cost += objective_->terminal_cost(X_.back());

        // Store the total cost.
        J_ = cost;

        return;
    }

    void CDDP::resetIPDDPRegularization()
    {
        ipddp_regularization_counter_ = 0;
        return;
    }

} // namespace cddp