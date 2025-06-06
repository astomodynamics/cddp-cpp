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

#include "cddp_core/ipddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <future>
#include <execution>

namespace cddp
{

    IPDDPSolver::IPDDPSolver() 
        : mu_(1e-1), optimality_gap_(0.0), kkt_error_(0.0), constraint_violation_(0.0) {}

    void IPDDPSolver::initialize(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        
        int horizon = context.getHorizon();
        int control_dim = context.getControlDim();
        int state_dim = context.getStateDim();

        // For warm starts, verify that existing state is valid
        if (options.warm_start)
        {
            bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                                     K_u_.size() == static_cast<size_t>(horizon));

            if (valid_warm_start && !k_u_.empty())
            {
                for (int t = 0; t < horizon; ++t)
                {
                    if (k_u_[t].size() != control_dim ||
                        K_u_[t].rows() != control_dim ||
                        K_u_[t].cols() != state_dim)
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
                if (options.verbose)
                {
                    std::cout << "IPDDP: Using warm start with existing control gains" << std::endl;
                }
                mu_ = options.ipddp.barrier.mu_initial;
                evaluateTrajectory(context);
                return;
            }
            else if (options.verbose)
            {
                std::cout << "IPDDP: Warning - warm start requested but no valid solver state found. "
                          << "Falling back to cold start initialization." << std::endl;
            }
        }

        // Cold start: full initialization
        k_u_.resize(horizon);
        K_u_.resize(horizon);
        
        for (int t = 0; t < horizon; ++t)
        {
            k_u_[t] = Eigen::VectorXd::Zero(control_dim);
            K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
        }

        dV_ = Eigen::Vector2d::Zero();

        // Clear constraint-related variables
        G_.clear();
        Y_.clear();
        S_.clear();
        k_y_.clear();
        K_y_.clear();
        k_s_.clear();
        K_s_.clear();

        // Initialize barrier parameter
        const auto &constraint_set = context.getConstraintSet();
        if (constraint_set.empty())
        {
            mu_ = 1e-8; // Small value if no constraints
        }
        else
        {
            mu_ = options.ipddp.barrier.mu_initial;
        }

        // Initialize dual and slack variables for each constraint
        for (const auto &constraint_pair : constraint_set)
        {
            const std::string &constraint_name = constraint_pair.first;
            int dual_dim = constraint_pair.second->getDualDim();

            G_[constraint_name].resize(horizon);
            Y_[constraint_name].resize(horizon);
            S_[constraint_name].resize(horizon);
            k_y_[constraint_name].resize(horizon);
            K_y_[constraint_name].resize(horizon);
            k_s_[constraint_name].resize(horizon);
            K_s_[constraint_name].resize(horizon);

            for (int t = 0; t < horizon; ++t)
            {
                // Evaluate constraint g(x,u) = evaluate(x,u) - getUpperBound()
                Eigen::VectorXd g_val = constraint_pair.second->evaluate(context.X_[t], context.U_[t]) - 
                                       constraint_pair.second->getUpperBound();
                G_[constraint_name][t] = g_val;

                Eigen::VectorXd s_init = Eigen::VectorXd::Zero(dual_dim);
                Eigen::VectorXd y_init = Eigen::VectorXd::Zero(dual_dim);

                for (int i = 0; i < dual_dim; ++i)
                {
                    // Initialize s_i = max(slack_scale, -g_i) to ensure s_i > 0
                    s_init(i) = std::max(options.ipddp.slack_var_init_scale, -g_val(i));

                    // Initialize y_i = mu / s_i to satisfy s_i * y_i = mu
                    if (s_init(i) < 1e-12)
                    {
                        y_init(i) = mu_ / 1e-12;
                    }
                    else
                    {
                        y_init(i) = mu_ / s_init(i);
                    }
                    // Clamp dual variable
                    y_init(i) = std::max(options.ipddp.dual_var_init_scale * 0.01, 
                                        std::min(y_init(i), options.ipddp.dual_var_init_scale * 100.0));
                }
                Y_[constraint_name][t] = y_init;
                S_[constraint_name][t] = s_init;

                // Initialize gains to zero
                k_y_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
                K_y_[constraint_name][t] = Eigen::MatrixXd::Zero(dual_dim, state_dim);
                k_s_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
                K_s_[constraint_name][t] = Eigen::MatrixXd::Zero(dual_dim, state_dim);
            }
        }

        // Initialize regularization
        context.regularization_ = options.regularization.initial_value;

        // Evaluate initial trajectory
        evaluateTrajectory(context);
        resetBarrierFilter(context);
    }

    std::string IPDDPSolver::getSolverName() const
    {
        return "IPDDP";
    }

    int IPDDPSolver::getTotalDualDim(const CDDP &context) const
    {
        int total_dual_dim = 0;
        const auto &constraint_set = context.getConstraintSet();
        for (const auto &constraint_pair : constraint_set)
        {
            total_dual_dim += constraint_pair.second->getDualDim();
        }
        return total_dual_dim;
    }

    CDDPSolution IPDDPSolver::solve(CDDP &context)
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
            printIteration(0, context.cost_, context.merit_function_, optimality_gap_,
                          context.regularization_, context.alpha_pr_, mu_, constraint_violation_);
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool converged = false;
        std::string termination_reason = "MaxIterationsReached";

        // Main IPDDP loop
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
                        std::cerr << "IPDDP: Maximum CPU time reached. Returning current solution" << std::endl;
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
                        termination_reason = "RegularizationLimitReached_NotConverged";
                        if (options.verbose)
                        {
                            std::cerr << "IPDDP: Backward pass regularization limit reached" << std::endl;
                        }
                        break;
                    }
                }
            }

            if (!backward_pass_success)
                break;

            // 2. Forward pass
            ForwardPassResult best_result = performForwardPass(context);

            // Update solution if forward pass succeeded
            if (best_result.success)
            {
                if (options.debug)
                {
                    std::cout << "[IPDDP: Forward pass] " << std::endl;
                    std::cout << "    cost: " << best_result.cost << std::endl;
                    std::cout << "    merit: " << best_result.merit_function << std::endl;
                    std::cout << "    alpha: " << best_result.alpha_pr << std::endl;
                    std::cout << "    cv_err: " << best_result.constraint_violation << std::endl;
                }
                
                context.X_ = best_result.state_trajectory;
                context.U_ = best_result.control_trajectory;
                if (best_result.dual_trajectory)
                    Y_ = *best_result.dual_trajectory;
                if (best_result.slack_trajectory)
                    S_ = *best_result.slack_trajectory;
                if (best_result.constraint_eval_trajectory)
                    G_ = *best_result.constraint_eval_trajectory;
                
                double dJ = context.cost_ - best_result.cost;
                context.cost_ = best_result.cost;
                context.merit_function_ = best_result.merit_function;
                context.alpha_pr_ = best_result.alpha_pr;
                context.inf_pr_ = best_result.constraint_violation;
                constraint_violation_ = best_result.constraint_violation;

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
                const int total_dual_dim = getTotalDualDim(context);
                double l1_norm_multipliers = 0.0;
                for (const auto& s_map_entry : S_)
                {
                    for (const Eigen::VectorXd& s_t : s_map_entry.second)
                    {
                        l1_norm_multipliers += s_t.lpNorm<1>();
                    }
                }
                
                double scaling_factor = std::max(options.termination_scaling_max_factor, 
                                                l1_norm_multipliers / (total_dual_dim * context.getHorizon())) / 
                                       options.termination_scaling_max_factor;
                double termination_metric = std::max(optimality_gap_ / scaling_factor, kkt_error_ / scaling_factor);

                if (termination_metric <= options.tolerance)
                {
                    converged = true;
                    termination_reason = "OptimalSolutionFound";
                    break;
                }
                
                if (std::abs(dJ) < options.acceptable_tolerance && termination_metric <= options.tolerance * 10)
                {
                    converged = true;
                    termination_reason = "AcceptableSolutionFound";
                    break;
                }
            }
            else
            {
                context.increaseRegularization();

                if (context.isRegularizationLimitReached())
                {
                    termination_reason = "RegularizationLimitReached_NotConverged";
                    converged = false;
                    if (options.verbose)
                    {
                        std::cerr << "IPDDP: Regularization limit reached. Not converged." << std::endl;
                    }
                    break;
                }
            }

            // Print iteration info
            if (options.verbose)
            {
                printIteration(iter, context.cost_, context.merit_function_, optimality_gap_,
                              context.regularization_, context.alpha_pr_, mu_, constraint_violation_);
            }

            // Update barrier parameters
            const int total_dual_dim = getTotalDualDim(context);
            double l1_norm_multipliers = 0.0;
            for (const auto& s_map_entry : S_)
            {
                for (const Eigen::VectorXd& s_t : s_map_entry.second)
                {
                    l1_norm_multipliers += s_t.lpNorm<1>();
                }
            }
            
            double scaling_factor = std::max(options.termination_scaling_max_factor, 
                                            l1_norm_multipliers / (total_dual_dim * context.getHorizon())) / 
                                   options.termination_scaling_max_factor;
            double termination_metric = std::max(optimality_gap_ / scaling_factor, kkt_error_ / scaling_factor);

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

    void IPDDPSolver::evaluateTrajectory(CDDP &context)
    {
        const int horizon = context.getHorizon();
        double cost = 0.0;

        // Set initial state
        context.X_[0] = context.getInitialState();

        // Rollout dynamics and calculate cost
        for (int t = 0; t < horizon; ++t)
        {
            const Eigen::VectorXd &x = context.X_[t];
            const Eigen::VectorXd &u = context.U_[t];

            // Compute stage cost
            cost += context.getObjective().running_cost(x, u, t);

            // For each constraint, evaluate and store the constraint value
            const auto &constraint_set = context.getConstraintSet();
            for (const auto &constraint_pair : constraint_set)
            {
                const std::string &constraint_name = constraint_pair.first;
                Eigen::VectorXd g_val = constraint_pair.second->evaluate(x, u) - 
                                       constraint_pair.second->getUpperBound();
                G_[constraint_name][t] = g_val;
            }

            // Compute next state using dynamics
            context.X_[t + 1] = context.getSystem().getDiscreteDynamics(x, u, t * context.getTimestep());
        }

        // Add terminal cost
        cost += context.getObjective().terminal_cost(context.X_.back());

        // Store the cost
        context.cost_ = cost;
    }

    void IPDDPSolver::resetBarrierFilter(CDDP &context)
    {
        // Evaluate merit function (cost + log-barrier terms)
        double merit_function = context.cost_;
        double constraint_violation = 0.0;

        const auto &constraint_set = context.getConstraintSet();
        for (int t = 0; t < context.getHorizon(); ++t)
        {
            for (const auto &constraint_pair : constraint_set)
            {
                const std::string &constraint_name = constraint_pair.first;
                const Eigen::VectorXd &s_vec = S_[constraint_name][t];
                const Eigen::VectorXd &g_vec = G_[constraint_name][t];

                // Add log-barrier term
                merit_function -= mu_ * s_vec.array().log().sum();
                
                // Compute constraint violation: ||g + s||_1
                Eigen::VectorXd r_p = g_vec + s_vec;
                constraint_violation += r_p.lpNorm<1>();
            }
        }
        
        context.merit_function_ = merit_function;
        constraint_violation_ = constraint_violation;
        context.inf_pr_ = constraint_violation;
    }

    bool IPDDPSolver::backwardPass(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        const int state_dim = context.getStateDim();
        const int control_dim = context.getControlDim();
        const int horizon = context.getHorizon();
        const double timestep = context.getTimestep();
        const auto &constraint_set = context.getConstraintSet();
        const int total_dual_dim = getTotalDualDim(context);

        // Terminal cost and its derivatives
        Eigen::VectorXd V_x = context.getObjective().getFinalCostGradient(context.X_.back());
        Eigen::MatrixXd V_xx = context.getObjective().getFinalCostHessian(context.X_.back());
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

        dV_ = Eigen::Vector2d::Zero();
        double Qu_err = 0.0;
        double rp_err = 0.0; // primal feasibility
        double rd_err = 0.0; // dual feasibility

        // If no constraints, use standard DDP recursion
        if (constraint_set.empty())
        {
            for (int t = horizon - 1; t >= 0; --t)
            {
                const Eigen::VectorXd &x = context.X_[t];
                const Eigen::VectorXd &u = context.U_[t];
                const auto [Fx, Fu] = context.getSystem().getJacobians(x, u, t * timestep);
                Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
                Eigen::MatrixXd B = timestep * Fu;

                // Cost & derivatives
                auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = context.getObjective().getRunningCostHessians(x, u, t);

                // Q expansions from cost
                Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
                Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
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

                // Regularization
                Eigen::MatrixXd Q_uu_reg = Q_uu;
                Q_uu_reg.diagonal().array() += context.regularization_;
                Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

                Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
                if (ldlt.info() != Eigen::Success)
                {
                    if (options.debug)
                    {
                        std::cerr << "IPDDP: Backward pass failed at time " << t << std::endl;
                    }
                    return false;
                }
                
                Eigen::VectorXd k_u = -ldlt.solve(Q_u);
                Eigen::MatrixXd K_u = -ldlt.solve(Q_ux);
                k_u_[t] = k_u;
                K_u_[t] = K_u;

                // Update value function
                V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
                V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;
                V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

                // Accumulate cost improvement
                dV_[0] += k_u.dot(Q_u);
                dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

                // Error tracking
                Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
            }

            optimality_gap_ = Qu_err;
            kkt_error_ = Qu_err;
            context.inf_du_ = Qu_err;

            if (options.debug)
            {
                std::cout << "[IPDDP Backward Pass]\n"
                          << "    Qu_err:  " << Qu_err << "\n"
                          << "    dV:      " << dV_.transpose() << std::endl;
            }
            return true;
        }
        else
        {
            // Constrained backward recursion
            for (int t = horizon - 1; t >= 0; --t)
            {
                const Eigen::VectorXd &x = context.X_[t];
                const Eigen::VectorXd &u = context.U_[t];

                // Continuous dynamics
                const auto [Fx, Fu] = context.getSystem().getJacobians(x, u, t * timestep);
                Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
                Eigen::MatrixXd B = timestep * Fu;

                // Gather dual and slack variables across all constraints
                Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd s = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);
                Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);
                Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);

                int offset = 0;
                for (const auto &constraint_pair : constraint_set)
                {
                    const std::string &constraint_name = constraint_pair.first;
                    int dual_dim = constraint_pair.second->getDualDim();

                    const Eigen::VectorXd &y_vec = Y_[constraint_name][t];
                    const Eigen::VectorXd &s_vec = S_[constraint_name][t];
                    const Eigen::VectorXd &g_vec = G_[constraint_name][t];
                    const Eigen::MatrixXd &g_x = constraint_pair.second->getStateJacobian(x, u);
                    const Eigen::MatrixXd &g_u = constraint_pair.second->getControlJacobian(x, u);

                    y.segment(offset, dual_dim) = y_vec;
                    s.segment(offset, dual_dim) = s_vec;
                    g.segment(offset, dual_dim) = g_vec;
                    Q_yx.block(offset, 0, dual_dim, state_dim) = g_x;
                    Q_yu.block(offset, 0, dual_dim, control_dim) = g_u;

                    offset += dual_dim;
                }

                // Cost & derivatives
                auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
                auto [l_xx, l_uu, l_ux] = context.getObjective().getRunningCostHessians(x, u, t);

                // Q expansions from cost
                Eigen::VectorXd Q_x = l_x + Q_yx.transpose() * y + A.transpose() * V_x;
                Eigen::VectorXd Q_u = l_u + Q_yu.transpose() * y + B.transpose() * V_x;
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

                Eigen::MatrixXd Y = y.asDiagonal();
                Eigen::MatrixXd S = s.asDiagonal();
                Eigen::MatrixXd S_inv = S.inverse();
                Eigen::MatrixXd YSinv = Y * S_inv;

                // Residuals
                Eigen::VectorXd r_p = g + s; // primal feasibility
                Eigen::VectorXd r_d = y.cwiseProduct(s).array() - mu_; // dual feasibility
                Eigen::VectorXd rhat = y.cwiseProduct(r_p) - r_d;

                // Regularization
                Eigen::MatrixXd Q_uu_reg = Q_uu;
                Q_uu_reg.diagonal().array() += context.regularization_;
                Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

                Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg + Q_yu.transpose() * YSinv * Q_yu);
                if (ldlt.info() != Eigen::Success)
                {
                    if (options.debug)
                    {
                        std::cerr << "IPDDP: Backward pass failed at time " << t << std::endl;
                    }
                    return false;
                }

                Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
                bigRHS.col(0) = Q_u + Q_yu.transpose() * S_inv * rhat;
                Eigen::MatrixXd M = Q_ux + Q_yu.transpose() * YSinv * Q_yx;
                for (int col = 0; col < state_dim; col++)
                {
                    bigRHS.col(col + 1) = M.col(col);
                }

                Eigen::MatrixXd kK = -ldlt.solve(bigRHS);

                // Parse out feedforward and feedback gains
                Eigen::VectorXd k_u = kK.col(0);
                Eigen::MatrixXd K_u(control_dim, state_dim);
                for (int col = 0; col < state_dim; col++)
                {
                    K_u.col(col) = kK.col(col + 1);
                }

                k_u_[t] = k_u;
                K_u_[t] = K_u;

                // Compute gains for constraints
                Eigen::VectorXd k_y = S_inv * (rhat + Y * Q_yu * k_u);
                Eigen::MatrixXd K_y = YSinv * (Q_yx + Q_yu * K_u);
                Eigen::VectorXd k_s = -r_p - Q_yu * k_u;
                Eigen::MatrixXd K_s = -Q_yx - Q_yu * K_u;

                offset = 0;
                for (const auto &constraint_pair : constraint_set)
                {
                    const std::string &constraint_name = constraint_pair.first;
                    int dual_dim = constraint_pair.second->getDualDim();

                    k_y_[constraint_name][t] = k_y.segment(offset, dual_dim);
                    K_y_[constraint_name][t] = K_y.block(offset, 0, dual_dim, state_dim);
                    k_s_[constraint_name][t] = k_s.segment(offset, dual_dim);
                    K_s_[constraint_name][t] = K_s.block(offset, 0, dual_dim, state_dim);

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
                V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

                // Error tracking
                Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
                rp_err = std::max(rp_err, r_p.lpNorm<Eigen::Infinity>());
                rd_err = std::max(rd_err, r_d.lpNorm<Eigen::Infinity>());
            }

            // Compute optimality gap and KKT error
            optimality_gap_ = Qu_err;
            kkt_error_ = std::max(rp_err, rd_err);
            context.inf_du_ = optimality_gap_;

            if (options.debug)
            {
                std::cout << "[IPDDP Backward Pass]\n"
                          << "    Qu_err:  " << std::scientific << std::setprecision(4) << Qu_err << "\n"
                          << "    rp_err:  " << std::scientific << std::setprecision(4) << rp_err << "\n"
                          << "    rd_err:  " << std::scientific << std::setprecision(4) << rd_err << "\n"
                          << "    dV:      " << std::scientific << std::setprecision(4) << dV_.transpose() << std::endl;
            }
            return true;
        }
    }

    ForwardPassResult IPDDPSolver::performForwardPass(CDDP &context)
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
                        std::cerr << "IPDDP: Forward pass thread failed: " << e.what() << std::endl;
                    }
                }
            }
        }

        return best_result;
    }

    ForwardPassResult IPDDPSolver::forwardPass(CDDP &context, double alpha)
    {
        const CDDPOptions &options = context.getOptions();
        const auto &constraint_set = context.getConstraintSet();

        ForwardPassResult result;
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.merit_function = std::numeric_limits<double>::infinity();
        result.alpha_pr = alpha;

        const int horizon = context.getHorizon();
        const double tau = std::max(options.ipddp.barrier.min_fraction_to_boundary, 1.0 - mu_);

        // Initialize trajectories
        result.state_trajectory = context.X_;
        result.control_trajectory = context.U_;
        result.state_trajectory[0] = context.getInitialState();

        std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
        std::map<std::string, std::vector<Eigen::VectorXd>> G_new = G_;

        double cost_new = 0.0;
        double merit_function_new = 0.0;
        double constraint_violation_new = 0.0;

        // Handle unconstrained case
        if (constraint_set.empty())
        {
            for (int t = 0; t < horizon; ++t)
            {
                const Eigen::VectorXd delta_x = result.state_trajectory[t] - context.X_[t];
                result.control_trajectory[t] = context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x;

                // Propagate dynamics
                result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
                    result.state_trajectory[t], result.control_trajectory[t], t * context.getTimestep());

                // Accumulate stage cost
                cost_new += context.getObjective().running_cost(result.state_trajectory[t], result.control_trajectory[t], t);
            }
            cost_new += context.getObjective().terminal_cost(result.state_trajectory.back());

            double dJ = context.cost_ - cost_new;
            double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
            double reduction_ratio = expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

            result.success = reduction_ratio > options.line_search.step_reduction_factor;
            result.cost = cost_new;
            result.merit_function = cost_new;
            result.constraint_violation = 0.0;
            return result;
        }

        // Constrained forward pass
        bool s_trajectory_feasible = true;
        for (int t = 0; t < horizon; ++t)
        {
            const Eigen::VectorXd delta_x = result.state_trajectory[t] - context.X_[t];

            // Slack update and feasibility check
            for (const auto &constraint_pair : constraint_set)
            {
                const std::string &constraint_name = constraint_pair.first;
                int dual_dim = constraint_pair.second->getDualDim();
                const Eigen::VectorXd &s_old = S_[constraint_name][t];

                Eigen::VectorXd s_new = s_old + alpha * k_s_[constraint_name][t] + K_s_[constraint_name][t] * delta_x;
                Eigen::VectorXd s_min = (1.0 - tau) * s_old;
                
                for (int i = 0; i < dual_dim; ++i)
                {
                    if (s_new[i] < s_min[i])
                    {
                        s_trajectory_feasible = false;
                        break;
                    }
                }
                if (!s_trajectory_feasible) break;
                S_new[constraint_name][t] = s_new;
            }
            if (!s_trajectory_feasible) break;

            // Update control
            result.control_trajectory[t] = context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x;

            // Propagate dynamics
            result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
                result.state_trajectory[t], result.control_trajectory[t], t * context.getTimestep());
        }

        if (!s_trajectory_feasible)
        {
            return result; // Failed
        }

        // Update dual variables
        bool suitable_alpha_y_found = false;
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_trial;

        for (double alpha_y_candidate : context.alphas_)
        {
            bool current_alpha_y_globally_feasible = true;
            Y_trial = Y_;

            for (int t = 0; t < horizon; ++t)
            {
                const Eigen::VectorXd delta_x = result.state_trajectory[t] - context.X_[t];

                for (const auto &constraint_pair : constraint_set)
                {
                    const std::string &constraint_name = constraint_pair.first;
                    int dual_dim = constraint_pair.second->getDualDim();
                    const Eigen::VectorXd &y_old = Y_[constraint_name][t];

                    Eigen::VectorXd y_new = y_old + alpha_y_candidate * k_y_[constraint_name][t] + 
                                           K_y_[constraint_name][t] * delta_x;
                    Eigen::VectorXd y_min = (1.0 - tau) * y_old;
                    
                    for (int i = 0; i < dual_dim; ++i)
                    {
                        if (y_new[i] < y_min[i])
                        {
                            current_alpha_y_globally_feasible = false;
                            break;
                        }
                    }
                    if (!current_alpha_y_globally_feasible) break;
                    Y_trial[constraint_name][t] = y_new;
                }
                if (!current_alpha_y_globally_feasible) break;
            }

            if (current_alpha_y_globally_feasible)
            {
                suitable_alpha_y_found = true;
                Y_new = Y_trial;
                break;
            }
        }

        if (!suitable_alpha_y_found)
        {
            return result; // Failed
        }

        // Cost computation and filter line-search
        for (int t = 0; t < horizon; ++t)
        {
            cost_new += context.getObjective().running_cost(result.state_trajectory[t], result.control_trajectory[t], t);

            for (const auto &constraint_pair : constraint_set)
            {
                const std::string &constraint_name = constraint_pair.first;
                G_new[constraint_name][t] = constraint_pair.second->evaluate(result.state_trajectory[t], 
                                                                           result.control_trajectory[t]) - 
                                           constraint_pair.second->getUpperBound();

                const Eigen::VectorXd &s_vec = S_new[constraint_name][t];
                merit_function_new -= mu_ * s_vec.array().log().sum();

                Eigen::VectorXd r_p = G_new[constraint_name][t] + s_vec;
                constraint_violation_new += r_p.lpNorm<1>();
            }
        }

        cost_new += context.getObjective().terminal_cost(result.state_trajectory.back());
        merit_function_new += cost_new;

        // Filter-based acceptance
        const auto &filter_opts = options.filter;
        double constraint_violation_old = constraint_violation_;
        double merit_function_old = context.merit_function_;
        double expected_improvement = alpha * dV_(0);
        
        bool filter_acceptance = false;

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

        if (filter_acceptance)
        {
            result.success = true;
            result.cost = cost_new;
            result.merit_function = merit_function_new;
            result.constraint_violation = constraint_violation_new;
            result.dual_trajectory = Y_new;
            result.slack_trajectory = S_new;
            result.constraint_eval_trajectory = G_new;
        }

        return result;
    }

    void IPDDPSolver::updateBarrierParameters(CDDP &context, bool forward_pass_success, double termination_metric)
    {
        const CDDPOptions &options = context.getOptions();
        const auto &barrier_opts = options.ipddp.barrier;

        if (termination_metric <= barrier_opts.mu_update_factor * mu_)
        {
            mu_ = std::max(options.tolerance / 10.0, 
                          std::min(barrier_opts.mu_update_factor * mu_, 
                                  std::pow(mu_, barrier_opts.mu_update_power)));
            resetBarrierFilter(context);
        }
    }

    void IPDDPSolver::printIteration(int iter, double cost, double merit_function, double optimality_gap,
                                     double regularization, double alpha_pr, double mu, double constraint_violation) const
    {
        if (iter == 0)
        {
            std::cout << std::setw(4) << "iter" << " "
                      << std::setw(12) << "objective" << " "
                      << std::setw(12) << "merit_func" << " "
                      << std::setw(10) << "opt_gap" << " "
                      << std::setw(8) << "lg(rg)" << " "
                      << std::setw(8) << "alpha" << " "
                      << std::setw(8) << "lg(mu)" << " "
                      << std::setw(10) << "cv_viol" << std::endl;
        }

        std::cout << std::setw(4) << iter << " "
                  << std::setw(12) << std::scientific << std::setprecision(4) << cost << " "
                  << std::setw(12) << std::scientific << std::setprecision(4) << merit_function << " "
                  << std::setw(10) << std::scientific << std::setprecision(2) << optimality_gap << " "
                  << std::setw(8) << std::fixed << std::setprecision(1) << std::log10(regularization) << " "
                  << std::setw(8) << std::fixed << std::setprecision(4) << alpha_pr << " "
                  << std::setw(8) << std::fixed << std::setprecision(1) << std::log10(mu) << " "
                  << std::setw(10) << std::scientific << std::setprecision(2) << constraint_violation << std::endl;
    }

    void IPDDPSolver::printSolutionSummary(const CDDPSolution &solution) const
    {
        std::cout << "\n========================================\n";
        std::cout << "           IPDDP Solution Summary\n";
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
