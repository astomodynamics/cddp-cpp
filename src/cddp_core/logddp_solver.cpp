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

#include "cddp_core/logddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <future>
#include <execution>

namespace cddp
{

    LogDDPSolver::LogDDPSolver() 
        : mu_(1e-2), relaxation_delta_(1e-5), constraint_violation_(0.0), ms_segment_length_(0) {}

    void LogDDPSolver::initialize(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        
        int horizon = context.getHorizon();
        int control_dim = context.getControlDim();
        int state_dim = context.getStateDim();

        // For warm starts, verify that existing state is valid
        if (options.warm_start)
        {
            // Check if solver state is properly initialized and compatible
            bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                                     K_u_.size() == static_cast<size_t>(horizon) &&
                                     F_.size() == static_cast<size_t>(horizon));

            if (valid_warm_start && !k_u_.empty())
            {
                // Verify dimensions are consistent
                for (int t = 0; t < horizon; ++t)
                {
                    if (k_u_[t].size() != control_dim ||
                        K_u_[t].rows() != control_dim ||
                        K_u_[t].cols() != state_dim ||
                        F_[t].size() != state_dim)
                    {
                        valid_warm_start = false;
                        break;
                    }
                }
            }
            else
            {
                valid_warm_start = false;
            }

            if (valid_warm_start)
            {
                // Valid warm start: only update what's necessary
                if (options.verbose)
                {
                    std::cout << "LogDDP: Using warm start with existing control gains" << std::endl;
                }

                // Update barrier parameters from options
                mu_ = options.log_barrier.barrier.mu_initial;
                relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
                ms_segment_length_ = options.log_barrier.segment_length; 

                // Update relaxed log barrier
                if (relaxed_log_barrier_)
                {
                    relaxed_log_barrier_->setBarrierCoeff(mu_);
                    relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
                }

                // Evaluate current trajectory
                evaluateTrajectory(context);
                return;
            }
            else
            {
                // Invalid warm start: fall back to cold start with warning
                if (options.verbose)
                {
                    std::cout << "LogDDP: Warning - warm start requested but no valid solver state found. "
                              << "Falling back to cold start initialization." << std::endl;
                }
            }
        }

        // Cold start: full initialization
        k_u_.resize(horizon);
        K_u_.resize(horizon);
        F_.resize(horizon);
        
        for (int t = 0; t < horizon; ++t)
        {
            k_u_[t] = Eigen::VectorXd::Zero(control_dim);
            K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
            F_[t] = Eigen::VectorXd::Zero(state_dim);
        }

        dV_ = Eigen::Vector2d::Zero();

        // Initialize regularization
        context.regularization_ = options.regularization.initial_value;

        // Initialize barrier parameters from options
        mu_ = options.log_barrier.barrier.mu_initial;
        relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
        ms_segment_length_ = options.log_barrier.segment_length;

        // Create relaxed log barrier
        relaxed_log_barrier_ = std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);

        // Initialize constraint violation
        constraint_violation_ = options.filter.max_violation_threshold;

        // Evaluate initial trajectory
        evaluateTrajectory(context);
    }

    CDDPSolution LogDDPSolver::solve(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();

        // Prepare solution map
        CDDPSolution solution;
        solution["solver_name"] = getSolverName();
        solution["status_message"] = std::string("Running");
        solution["iterations_completed"] = 0;
        solution["solve_time_ms"] = 0.0;

        // Initialize history vectors only if requested
        std::vector<double> history_objective;
        std::vector<double> history_merit_function;
        std::vector<double> history_step_length_primal;
        std::vector<double> history_dual_infeasibility;
        std::vector<double> history_primal_infeasibility;
        std::vector<double> history_barrier_mu;

        if (options.return_iteration_info)
        {
            const size_t expected_size = static_cast<size_t>(options.max_iterations + 1);
            history_objective.reserve(expected_size);
            history_merit_function.reserve(expected_size);
            history_step_length_primal.reserve(expected_size);
            history_dual_infeasibility.reserve(expected_size);
            history_primal_infeasibility.reserve(expected_size);
            history_barrier_mu.reserve(expected_size);

            // Initial iteration values
            history_objective.push_back(context.cost_);
            history_merit_function.push_back(context.merit_function_);
            history_dual_infeasibility.push_back(context.inf_du_);
            history_primal_infeasibility.push_back(context.inf_pr_);
            history_barrier_mu.push_back(mu_);
        }

        if (options.verbose)
        {
            printIteration(0, context.cost_, context.inf_pr_, context.inf_du_, 
                          context.regularization_, context.alpha_pr_, mu_);
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool converged = false;
        std::string termination_reason = "MaxIterationsReached";

        // Main LogDDP loop
        while (iter < options.max_iterations)
        {
            ++iter;

            // Check maximum CPU time
            if (options.max_cpu_time > 0)
            {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                if (duration.count() > options.max_cpu_time * 1000)
                {
                    termination_reason = "MaxCpuTimeReached";
                    if (options.verbose)
                    {
                        std::cerr << "LogDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                    }
                    break;
                }
            }

            // 1. Backward pass
            bool backward_pass_success = false;
            while (!backward_pass_success)
            {
                backward_pass_success = backwardPass(context);

                if (!backward_pass_success)
                {
                    context.increaseRegularization();
                    if (context.isRegularizationLimitReached())
                    {
                        termination_reason = "RegularizationLimit_NotConverged";
                        if (options.verbose)
                        {
                            std::cerr << "LogDDP: Backward pass regularization limit reached" << std::endl;
                        }
                        break;
                    }
                }
            }

            if (!backward_pass_success)
                break;

            // Check convergence based on dual infeasibility
            double termination_metric = std::max(context.inf_du_, constraint_violation_);
            if (termination_metric <= options.tolerance)
            {
                converged = true;
                termination_reason = "OptimalSolutionFound";
                break;
            }

            // 2. Forward pass
            ForwardPassResult best_result = performForwardPass(context);

            // Update solution if forward pass succeeded
            if (best_result.success)
            {
                if (options.debug)
                {
                    std::cout << "[LogDDP: Forward pass] " << std::endl;
                    std::cout << "    cost: " << best_result.cost << std::endl;
                    std::cout << "    logcost: " << best_result.merit_function << std::endl;
                    std::cout << "    alpha: " << best_result.alpha_pr << std::endl;
                    std::cout << "    rf_err: " << best_result.constraint_violation << std::endl;
                }
                context.X_ = best_result.state_trajectory;
                context.U_ = best_result.control_trajectory;
                F_ = *best_result.dynamics_trajectory;
                
                double dJ = context.cost_ - best_result.cost;
                context.cost_ = best_result.cost;
                context.merit_function_ = best_result.merit_function;
                context.alpha_pr_ = best_result.alpha_pr;
                context.inf_pr_ = best_result.constraint_violation;

                // Store history only if requested
                if (options.return_iteration_info)
                {
                    history_objective.push_back(context.cost_);
                    history_merit_function.push_back(context.merit_function_);
                    history_step_length_primal.push_back(context.alpha_pr_);
                    history_dual_infeasibility.push_back(context.inf_du_);
                    history_primal_infeasibility.push_back(context.inf_pr_);
                    history_barrier_mu.push_back(mu_);
                }

                context.decreaseRegularization();

                // Check convergence
                if (dJ < options.acceptable_tolerance)
                {
                    converged = true;
                    termination_reason = "AcceptableSolutionFound";
                    break;
                }
            }
            else
            {
                context.increaseRegularization();

                // Check if regularization limit reached
                if (context.isRegularizationLimitReached())
                {
                    termination_reason = "RegularizationLimitReached_NotConverged";
                    converged = false;
                    if (options.verbose)
                    {
                        std::cerr << "LogDDP: Regularization limit reached. Not converged." << std::endl;
                    }
                    break;
                }
            }

            // Print iteration info
            if (options.verbose)
            {
                printIteration(iter, context.cost_, context.inf_pr_, context.inf_du_, 
                              context.regularization_, context.alpha_pr_, mu_);
            }

            // Update barrier parameters
            updateBarrierParameters(context, best_result.success, termination_metric);
        }

        // Compute final timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Populate final solution
        solution["status_message"] = termination_reason;
        solution["iterations_completed"] = iter;
        solution["solve_time_ms"] = static_cast<double>(duration.count());
        solution["final_objective"] = context.cost_;
        solution["final_step_length"] = context.alpha_pr_;

        // Add trajectories
        std::vector<double> time_points;
        time_points.reserve(static_cast<size_t>(context.getHorizon() + 1));
        for (int t = 0; t <= context.getHorizon(); ++t)
        {
            time_points.push_back(t * context.getTimestep());
        }
        solution["time_points"] = time_points;
        solution["state_trajectory"] = context.X_;
        solution["control_trajectory"] = context.U_;

        // Add iteration history if requested
        if (options.return_iteration_info)
        {
            solution["history_objective"] = history_objective;
            solution["history_merit_function"] = history_merit_function;
            solution["history_step_length_primal"] = history_step_length_primal;
            solution["history_dual_infeasibility"] = history_dual_infeasibility;
            solution["history_primal_infeasibility"] = history_primal_infeasibility;
            solution["history_barrier_mu"] = history_barrier_mu;
        }

        // Add control gains
        solution["control_feedback_gains_K"] = K_u_;

        // Final metrics
        solution["final_regularization"] = context.regularization_;
        solution["final_barrier_parameter_mu"] = mu_;
        solution["final_primal_infeasibility"] = context.inf_pr_;
        solution["final_dual_infeasibility"] = context.inf_du_;

        if (options.verbose)
        {
            printSolutionSummary(solution);
        }

        return solution;
    }

    std::string LogDDPSolver::getSolverName() const
    {
        return "LogDDP";
    }

    void LogDDPSolver::evaluateTrajectory(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        const int horizon = context.getHorizon();
        double cost = 0.0;
        constraint_violation_ = 0.0;

        // Rollout dynamics and calculate cost
        for (int t = 0; t < horizon; ++t)
        {
            const Eigen::VectorXd &x = context.X_[t];
            const Eigen::VectorXd &u = context.U_[t];

            // Compute stage cost
            cost += context.getObjective().running_cost(x, u, t);

            // Compute dynamics
            F_[t] = context.getSystem().getDiscreteDynamics(x, u, t * context.getTimestep());
            
            if (options.log_barrier.use_controlled_rollout)
            {
                context.X_[t + 1] = F_[t];
            }

            // Add defect violation (multi-shooting gap)
            Eigen::VectorXd defect = F_[t] - context.X_[t + 1];
            constraint_violation_ += defect.lpNorm<1>();
        }

        // Add terminal cost
        cost += context.getObjective().terminal_cost(context.X_.back());

        // Store the cost
        context.cost_ = cost;
        context.inf_pr_ = constraint_violation_;

        // Reset filter to compute merit function
        resetFilter(context);
    }

    void LogDDPSolver::resetFilter(CDDP &context)
    {
        // Evaluate log-barrier cost (includes path constraints)
        double merit_function = context.cost_;
        
        // Add log-barrier terms from path constraints
        const auto &constraint_set = context.getConstraintSet();
        for (int t = 0; t < context.getHorizon(); ++t)
        {
            for (const auto &constraint_pair : constraint_set)
            {
                merit_function += relaxed_log_barrier_->evaluate(*constraint_pair.second, 
                                                                context.X_[t], context.U_[t]);
            }
        }
        
        context.merit_function_ = merit_function;
    }

    bool LogDDPSolver::backwardPass(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        const int state_dim = context.getStateDim();
        const int control_dim = context.getControlDim();
        const int horizon = context.getHorizon();
        const double timestep = context.getTimestep();
        const auto &constraint_set = context.getConstraintSet();

        // Terminal cost and its derivatives
        Eigen::VectorXd V_x = context.getObjective().getFinalCostGradient(context.X_.back());
        Eigen::MatrixXd V_xx = context.getObjective().getFinalCostHessian(context.X_.back());
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

        dV_ = Eigen::Vector2d::Zero();
        double norm_Vx = V_x.lpNorm<1>();
        double Qu_error = 0.0;

        // Backward Riccati recursion
        for (int t = horizon - 1; t >= 0; --t)
        {
            const Eigen::VectorXd &x = context.X_[t];
            const Eigen::VectorXd &u = context.U_[t];
            const Eigen::VectorXd &f = F_[t];
            const Eigen::VectorXd &d = f - context.X_[t + 1]; // Defect

            // Get continuous dynamics Jacobians
            const auto [Fx, Fu] = context.getSystem().getJacobians(x, u, t * timestep);

            // Convert to discrete time
            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
            Eigen::MatrixXd B = timestep * Fu;

            // Get cost and its derivatives
            auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
            auto [l_xx, l_uu, l_ux] = context.getObjective().getRunningCostHessians(x, u, t);

            // Compute Q-function matrices
            Eigen::VectorXd Q_x = l_x + A.transpose() * (V_x + V_xx * d);
            Eigen::VectorXd Q_u = l_u + B.transpose() * (V_x + V_xx * d);
            Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
            Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
            Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

            // Add state hessian term if not using iLQR
            if (!options.use_ilqr)
            {
                const auto hessians = context.getSystem().getHessians(x, u, t * timestep);
                const auto &Fxx = std::get<0>(hessians);
                const auto &Fuu = std::get<1>(hessians);
                const auto &Fux = std::get<2>(hessians);
                
                for (int i = 0; i < state_dim; ++i)
                {
                    Q_xx += timestep * V_x(i) * Fxx[i];
                    Q_ux += timestep * V_x(i) * Fux[i];
                    Q_uu += timestep * V_x(i) * Fuu[i];
                }
            }

            // Apply Log-barrier cost gradients and Hessians for path constraints
            for (const auto &constraint_pair : constraint_set)
            {
                auto [L_x_relaxed, L_u_relaxed] = relaxed_log_barrier_->getGradients(*constraint_pair.second, x, u);
                Q_x += L_x_relaxed;
                Q_u += L_u_relaxed;

                auto [L_xx_relaxed, L_uu_relaxed, L_ux_relaxed] = relaxed_log_barrier_->getHessians(*constraint_pair.second, x, u);
                Q_xx += L_xx_relaxed;
                Q_uu += L_uu_relaxed;
                Q_ux += L_ux_relaxed;
            }

            // Apply regularization
            Eigen::MatrixXd Q_uu_reg = Q_uu;
            Q_uu_reg.diagonal().array() += context.regularization_;
            Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // Symmetrize

            // Check positive definiteness
            Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
            if (ldlt.info() != Eigen::Success)
            {
                if (options.debug)
                {
                    std::cerr << "LogDDP: Q_uu is not positive definite at time " << t << std::endl;
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

            norm_Vx += V_x.lpNorm<1>();

            // Update optimality gap
            Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());
        }

        // Normalize dual infeasibility 
        double scaling_factor = options.termination_scaling_max_factor;
        scaling_factor = std::max(scaling_factor, norm_Vx / (horizon * state_dim)) / scaling_factor;
        context.inf_du_ = Qu_error;

        if (options.debug)
        {
            std::cout << "[LogDDP Backward Pass]\n"
                      << "    Qu_err:  " << std::scientific << std::setprecision(4) << Qu_error << "\n"
                      << "    rf_err:  " << std::scientific << std::setprecision(4) <<  context.inf_pr_ << "\n"
                      << "    mu:      " << std::scientific << std::setprecision(4) << mu_ << "\n"
                      << "    relaxation_delta:      " << std::scientific << std::setprecision(4) << relaxation_delta_ << "\n"
                      << "    dV:      " << std::scientific << std::setprecision(4) << dV_.transpose() << std::endl;
        }

        return true;
    }

    ForwardPassResult LogDDPSolver::performForwardPass(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        ForwardPassResult best_result;
        best_result.cost = std::numeric_limits<double>::infinity();
        best_result.merit_function = std::numeric_limits<double>::infinity();
        best_result.success = false;

        if (!options.enable_parallel)
        {
            // Single-threaded execution with early termination
            for (double alpha_pr : context.alphas_)
            {
                ForwardPassResult result = forwardPass(context, alpha_pr);

                if (result.success && result.merit_function < best_result.merit_function)
                {
                    best_result = result;
                    if (result.success)
                    {
                        break; // Early termination
                    }
                }
            }
        }
        else
        {
            // Multi-threaded execution
            std::vector<std::future<ForwardPassResult>> futures;
            futures.reserve(context.alphas_.size());

            for (double alpha_pr : context.alphas_)
            {
                futures.push_back(std::async(std::launch::async,
                                             [this, &context, alpha_pr]()
                                             { return forwardPass(context, alpha_pr); }));
            }

            for (auto &future : futures)
            {
                try
                {
                    if (future.valid())
                    {
                        ForwardPassResult result = future.get();
                        if (result.success && result.merit_function < best_result.merit_function)
                        {
                            best_result = result;
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    if (options.verbose)
                    {
                        std::cerr << "LogDDP: Forward pass thread failed: " << e.what() << std::endl;
                    }
                }
            }
        }

        return best_result;
    }

    ForwardPassResult LogDDPSolver::forwardPass(CDDP &context, double alpha)
    {
        const CDDPOptions &options = context.getOptions();
        const auto &constraint_set = context.getConstraintSet();

        ForwardPassResult result;
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.merit_function = std::numeric_limits<double>::infinity();
        result.alpha_pr = alpha;

        const int state_dim = context.getStateDim();
        const int horizon = context.getHorizon();

        // Initialize trajectories
        result.state_trajectory = context.X_;
        result.control_trajectory = context.U_;
        result.state_trajectory[0] = context.getInitialState();

        // Initialize dynamics trajectory
        std::vector<Eigen::VectorXd> dynamics_trajectory(horizon);
        
        double cost_new = 0.0;
        double merit_function_new = 0.0;
        double constraint_violation_new = 0.0;

        // Forward simulation with multi-shooting support
        for (int t = 0; t < horizon; ++t)
        {
            const Eigen::VectorXd &x = result.state_trajectory[t];
            const Eigen::VectorXd &u = result.control_trajectory[t];
            const Eigen::VectorXd delta_x = x - context.X_[t];

            // Apply control update
            result.control_trajectory[t] = u + alpha * k_u_[t] + K_u_[t] * delta_x;

            // Compute running cost
            cost_new += context.getObjective().running_cost(x, result.control_trajectory[t], t);

            // Add log-barrier terms for path constraints
            for (const auto &constraint_pair : constraint_set)
            {
                merit_function_new += relaxed_log_barrier_->evaluate(*constraint_pair.second, 
                                                                   x, result.control_trajectory[t]);
            }

            // Determine if using multi-shooting gap-closing
            bool is_segment_boundary = (ms_segment_length_ > 0) &&
                                       ((t + 1) % ms_segment_length_ == 0) &&
                                       (t + 1 < horizon);

            // if (is_segment_boundary)
            // {
            //     // Multi-shooting: use gap-closing strategy
            //     dynamics_trajectory[t] = context.getSystem().getDiscreteDynamics(
            //         x, result.control_trajectory[t], t * context.getTimestep());
            //     result.state_trajectory[t + 1] = context.X_[t + 1] + 
            //         (dynamics_trajectory[t] - F_[t]) + alpha * (F_[t] - context.X_[t + 1]);
            // }
            // else
            // {
            //     // Regular propagation
            //     dynamics_trajectory[t] = context.getSystem().getDiscreteDynamics(
            //         x, result.control_trajectory[t], t * context.getTimestep());
            //     result.state_trajectory[t + 1] = dynamics_trajectory[t];
            // }

            // Regular propagation
            dynamics_trajectory[t] = context.getSystem().getDiscreteDynamics(
                x, result.control_trajectory[t], t * context.getTimestep());
            result.state_trajectory[t + 1] = dynamics_trajectory[t];

            // Robustness check
            if (!result.state_trajectory[t + 1].allFinite() || !result.control_trajectory[t].allFinite())
            {
                if (options.debug)
                {
                    std::cerr << "LogDDP Forward Pass: NaN/Inf detected at t=" << t 
                              << " for alpha=" << alpha << std::endl;
                }
                result.success = false;
                return result;
            }

            // Add defect violation
            Eigen::VectorXd defect = dynamics_trajectory[t] - result.state_trajectory[t + 1];
            constraint_violation_new += defect.lpNorm<1>();
        }

        // Add terminal cost
        cost_new += context.getObjective().terminal_cost(result.state_trajectory.back());
        merit_function_new += cost_new;

        // Filter-based line search acceptance
        double constraint_violation_old = constraint_violation_;
        double merit_function_old = context.merit_function_;
        double expected_improvement = alpha * dV_(0);
        
        bool filter_acceptance = false;
        const auto &filter_opts = options.filter;

        if (constraint_violation_new > filter_opts.max_violation_threshold)
        {
            if (constraint_violation_new < (1.0 - filter_opts.violation_acceptance_threshold) * constraint_violation_old)
            {
                filter_acceptance = true;
            }
        }
        else if (std::max(constraint_violation_new, constraint_violation_old) < filter_opts.min_violation_for_armijo_check && 
                 expected_improvement < 0)
        {
            if (merit_function_new < merit_function_old + filter_opts.armijo_constant * expected_improvement)
            {
                filter_acceptance = true;
            }
        }
        else
        {
            if (merit_function_new < merit_function_old - filter_opts.merit_acceptance_threshold * constraint_violation_old ||
                constraint_violation_new < (1.0 - filter_opts.violation_acceptance_threshold) * constraint_violation_old)
            {
                filter_acceptance = true;
            }
        }

        result.success = filter_acceptance;
        result.cost = cost_new;
        result.merit_function = merit_function_new;
        result.constraint_violation = constraint_violation_new;
        result.dynamics_trajectory = dynamics_trajectory;

        return result;
    }

    void LogDDPSolver::updateBarrierParameters(CDDP &context, bool forward_pass_success, double termination_metric)
    {
        const CDDPOptions &options = context.getOptions();
        const auto &barrier_opts = options.log_barrier.barrier;

        if (forward_pass_success && termination_metric < options.tolerance)
        {
            // Dramatically decrease mu if optimization is going well
            mu_ = std::max(mu_ * 0.1, barrier_opts.mu_min_value);
            
            relaxation_delta_ = std::max(options.tolerance / 10.0, 
                                        std::min(relaxation_delta_ * 0.1, 
                                                std::pow(relaxation_delta_, barrier_opts.mu_update_power)));
        }
        else
        {
            // Normal decrease rate
            mu_ = std::max(options.tolerance / 10.0, 
                          std::min(barrier_opts.mu_update_factor * mu_, 
                                  std::pow(mu_, barrier_opts.mu_update_power)));
            
            relaxation_delta_ = std::max(options.tolerance / 10.0, 
                                        std::min(relaxation_delta_ * 0.1, 
                                                std::pow(relaxation_delta_, barrier_opts.mu_update_power)));
        }

        // Update barrier object
        relaxed_log_barrier_->setBarrierCoeff(mu_);
        relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);

        // Reset filter for new barrier parameters
        resetFilter(context);
    }

    void LogDDPSolver::printIteration(int iter, double cost, double inf_pr, double inf_du,
                                      double regularization, double alpha_pr, double mu) const
    {
        if (iter == 0)
        {
            std::cout << std::setw(4) << "iter" << " "
                      << std::setw(12) << "objective" << " "
                      << std::setw(10) << "inf_pr" << " "
                      << std::setw(10) << "inf_du" << " "
                      << std::setw(8) << "lg(rg)" << " "
                      << std::setw(8) << "alpha" << " "
                      << std::setw(8) << "lg(mu)" << std::endl;
        }

        std::cout << std::setw(4) << iter << " "
                  << std::setw(12) << std::scientific << std::setprecision(4) << cost << " "
                  << std::setw(10) << std::scientific << std::setprecision(2) << inf_pr << " "
                  << std::setw(10) << std::scientific << std::setprecision(2) << inf_du << " "
                  << std::setw(8) << std::fixed << std::setprecision(1) << std::log10(regularization) << " "
                  << std::setw(8) << std::fixed << std::setprecision(4) << alpha_pr << " "
                  << std::setw(8) << std::fixed << std::setprecision(1) << std::log10(mu) << std::endl;
    }

    void LogDDPSolver::printSolutionSummary(const CDDPSolution &solution) const
    {
        std::cout << "\n========================================\n";
        std::cout << "           LogDDP Solution Summary\n";
        std::cout << "========================================\n";

        auto iterations = std::any_cast<int>(solution.at("iterations_completed"));
        auto solve_time = std::any_cast<double>(solution.at("solve_time_ms"));
        auto final_cost = std::any_cast<double>(solution.at("final_objective"));
        auto status = std::any_cast<std::string>(solution.at("status_message"));
        auto final_mu = std::any_cast<double>(solution.at("final_barrier_parameter_mu"));

        std::cout << "Status: " << status << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Solve Time: " << std::setprecision(2) << solve_time << " ms\n";
        std::cout << "Final Cost: " << std::setprecision(6) << final_cost << "\n";
        std::cout << "Final Barrier μ: " << std::setprecision(2) << std::scientific << final_mu << "\n";
        std::cout << "========================================\n\n";
    }

} // namespace cddp
