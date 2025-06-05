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

#include "cddp_core/asddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include "osqp++.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <future>
#include <execution>

namespace cddp
{

    ASDDPSolver::ASDDPSolver() {}

    void ASDDPSolver::initialize(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();

        int horizon = context.getHorizon();
        int control_dim = context.getControlDim();
        int state_dim = context.getStateDim();

        // For warm starts, verify that existing state is valid
        if (options.warm_start)
        {
            bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                                     K_u_.size() == static_cast<size_t>(horizon) &&
                                     Q_UU_.size() == static_cast<size_t>(horizon) &&
                                     Q_UX_.size() == static_cast<size_t>(horizon) &&
                                     Q_U_.size() == static_cast<size_t>(horizon));

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
                    std::cout << "ASDDP: Using warm start with existing control gains" << std::endl;
                }
                if (!context.X_.empty() && !context.U_.empty())
                {
                    computeCost(context);
                }
                return;
            }
            else
            {
                if (options.verbose)
                {
                    std::cout << "ASDDP: Warning - warm start requested but no valid solver state found. "
                              << "Falling back to cold start initialization." << std::endl;
                }
            }
        }

        // Cold start: full initialization
        k_u_.resize(horizon);
        K_u_.resize(horizon);
        Q_UU_.resize(horizon);
        Q_UX_.resize(horizon);
        Q_U_.resize(horizon);

        for (int t = 0; t < horizon; ++t)
        {
            k_u_[t] = Eigen::VectorXd::Zero(control_dim);
            K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
        }

        dV_ = Eigen::Vector2d::Zero();

        // Compute initial cost if trajectories exist
        if (!context.X_.empty() && !context.U_.empty())
        {
            computeCost(context);
        }
    }

    CDDPSolution ASDDPSolver::solve(CDDP &context)
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
        std::vector<double> history_regularization;

        if (options.return_iteration_info)
        {
            const size_t expected_size = static_cast<size_t>(options.max_iterations + 1);
            history_objective.reserve(expected_size);
            history_merit_function.reserve(expected_size);
            history_step_length_primal.reserve(expected_size);
            history_dual_infeasibility.reserve(expected_size);
            history_regularization.reserve(expected_size);

            // Initial iteration values
            history_objective.push_back(context.cost_);
            history_merit_function.push_back(context.merit_function_);
            history_dual_infeasibility.push_back(context.inf_du_);
            history_regularization.push_back(context.regularization_);
        }

        if (options.verbose)
        {
            printIteration(0, context.cost_, context.merit_function_, context.inf_du_,
                           context.regularization_, context.alpha_);
        }

        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool converged = false;
        std::string termination_reason = "MaxIterationsReached";

        // Main ASDDP loop
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
                        std::cerr << "ASDDP: Maximum CPU time reached. Returning current solution" << std::endl;
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
                            std::cerr << "ASDDP: Backward pass regularization limit reached" << std::endl;
                        }
                        break;
                    }
                }
            }

            if (!backward_pass_success)
                break;

            // Check convergence
            double scaling_factor = 100.0;
            scaling_factor = std::max(scaling_factor, norm_Vx_ / (context.getHorizon() * context.getStateDim())) / scaling_factor;

            if (context.inf_du_ / scaling_factor < options.tolerance)
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
                context.X_ = best_result.state_trajectory;
                context.U_ = best_result.control_trajectory;
                double dJ = context.cost_ - best_result.cost;
                context.cost_ = best_result.cost;
                context.merit_function_ = best_result.cost; // For ASDDP, merit function equals cost
                context.alpha_ = best_result.alpha;

                // Store history only if requested
                if (options.return_iteration_info)
                {
                    history_objective.push_back(context.cost_);
                    history_merit_function.push_back(context.merit_function_);
                    history_step_length_primal.push_back(context.alpha_);
                    history_dual_infeasibility.push_back(context.inf_du_);
                    history_regularization.push_back(context.regularization_);
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
                        std::cerr << "ASDDP: Regularization limit reached. Not converged." << std::endl;
                    }
                    break;
                }
            }

            // Print iteration info
            if (options.verbose)
            {
                printIteration(iter, context.cost_, context.merit_function_, context.inf_du_,
                               context.regularization_, context.alpha_);
            }
        }

        // Compute final timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Populate final solution
        solution["status_message"] = termination_reason;
        solution["iterations_completed"] = iter;
        solution["solve_time_ms"] = static_cast<double>(duration.count());
        solution["final_objective"] = context.cost_;
        solution["final_step_length"] = context.alpha_;
        solution["final_dual_infeasibility"] = context.inf_du_;
        solution["final_primal_infeasibility"] = context.inf_pr_;

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
            solution["history_regularization"] = history_regularization;
        }

        // Add control gains
        solution["control_feedback_gains_K"] = K_u_;

        // Final metrics
        solution["final_regularization"] = context.regularization_;

        if (options.verbose)
        {
            printSolutionSummary(solution);
        }

        return solution;
    }

    std::string ASDDPSolver::getSolverName() const
    {
        return "ASDDP";
    }

    bool ASDDPSolver::backwardPass(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        const int state_dim = context.getStateDim();
        const int control_dim = context.getControlDim();
        const int horizon = context.getHorizon();
        const int dual_dim = context.getTotalDualDim() - control_dim;
        const double timestep = context.getTimestep();
        const auto active_set_tol = 1e-6;

        // Extract control box constraint
        auto control_box_constraint = context.getConstraint<ControlBoxConstraint>("ControlBoxConstraint");

        // Terminal cost and its derivatives
        Eigen::VectorXd V_x = context.getObjective().getFinalCostGradient(context.X_.back());
        Eigen::MatrixXd V_xx = context.getObjective().getFinalCostHessian(context.X_.back());

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
        Eigen::VectorXd k(control_dim);
        Eigen::MatrixXd K(control_dim, state_dim);

        dV_ = Eigen::Vector2d::Zero();
        double Qu_error = 0.0;
        double norm_Vx = V_x.lpNorm<1>();

        // Backward Riccati recursion
        for (int t = horizon - 1; t >= 0; --t)
        {
            const Eigen::VectorXd &x = context.X_[t];
            const Eigen::VectorXd &u = context.U_[t];

            // Get continuous dynamics Jacobians
            const auto [Fx, Fu] = context.getSystem().getJacobians(x, u, t * timestep);

            // Convert to discrete time
            A = timestep * Fx;
            A.diagonal().array() += 1.0;
            B = timestep * Fu;

            // Get cost and its derivatives
            auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
            auto [l_xx, l_uu, l_ux] = context.getObjective().getRunningCostHessians(x, u, t);

            // Compute Q-function matrices
            Q_x = l_x + A.transpose() * V_x;
            Q_u = l_u + B.transpose() * V_x;
            Q_xx = l_xx + A.transpose() * V_xx * A;
            Q_ux = l_ux + B.transpose() * V_xx * A;
            Q_uu = l_uu + B.transpose() * V_xx * B;

            // Apply regularization
            Q_uu_reg = Q_uu;
            Q_uu_reg.diagonal().array() += context.regularization_;

            // Check positive definiteness
            Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
            if (es.eigenvalues().real().minCoeff() <= 0)
            {
                if (options.debug)
                {
                    std::cerr << "ASDDP: Q_uu is not positive definite at time " << t << std::endl;
                }
                return false;
            }

            /*  --- Identify Active Constraint --- */
            int active_constraint_index = 0;
            Eigen::MatrixXd C(dual_dim, control_dim); // Control constraint matrix
            Eigen::MatrixXd D(dual_dim, state_dim);   // State constraint matrix

            // Identify control constraints
            if (control_box_constraint != nullptr)
            {
                for (int j = 0; j < control_dim; j++)
                {
                    if (u(j) <= control_box_constraint->getLowerBound()(j) + active_set_tol)
                    {
                        Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
                        e(j) = 1.0;
                        C.row(active_constraint_index) = -e; // Note the negative sign
                        D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
                        active_constraint_index += 1;
                    }
                    else if (u(j) >= control_box_constraint->getUpperBound()(j) - active_set_tol)
                    {
                        Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
                        e(j) = 1.0; // No negative here
                        C.row(active_constraint_index) = e;
                        D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
                        active_constraint_index += 1;
                    }
                }
            }

            // Identify state constraints
            const auto &constraint_set = context.getConstraintSet();
            if (t < horizon - 1)
            {
                for (const auto &[name, constraint] : constraint_set)
                {
                    if (name == "ControlBoxConstraint")
                    {
                        continue;
                    }

                    Eigen::VectorXd constraint_vals = constraint->evaluate(context.X_[t + 1], context.U_[t + 1]);
                    auto [cons_jac_x, cons_jac_u] = constraint->getJacobians(context.X_[t + 1], context.U_[t + 1]);

                    for (int j = 0; j < constraint_vals.size(); j++)
                    {
                        if (std::abs(constraint_vals(j)) <= active_set_tol)
                        {
                            C.row(active_constraint_index) = cons_jac_x * B;
                            D.row(active_constraint_index) = cons_jac_x * A;
                            active_constraint_index++;
                        }
                    }
                }
            }

            if (active_constraint_index == 0)
            { // No active constraints
                const Eigen::MatrixXd &H = Q_uu_reg.inverse();
                k = -H * Q_u;
                K = -H * Q_ux;
            }
            else
            {
                // Extract identified active constraints
                Eigen::MatrixXd grad_x_g = D.topRows(active_constraint_index);
                Eigen::MatrixXd grad_u_g = C.topRows(active_constraint_index);

                // Calculate Lagrange multipliers
                Eigen::MatrixXd Q_uu_inv = Q_uu_reg.inverse();
                Eigen::MatrixXd lambda = -(grad_u_g * Q_uu_inv * grad_u_g.transpose()).inverse() * (grad_u_g * Q_uu_inv * Q_u);

                // Find indices where lambda is non-negative
                std::vector<int> active_indices;
                for (int i = 0; i < lambda.rows(); ++i)
                {
                    if (lambda(i) >= 0)
                    {
                        active_indices.push_back(i);
                    }
                }
                int active_count_new = active_indices.size();

                // Create new constraint matrices
                Eigen::MatrixXd C_new = Eigen::MatrixXd::Zero(active_count_new, control_dim);
                Eigen::MatrixXd D_new = Eigen::MatrixXd::Zero(active_count_new, state_dim);

                if (active_count_new > 0)
                {
                    // Fill new constraint matrices with active constraints
                    for (int i = 0; i < active_count_new; ++i)
                    {
                        C_new.row(i) = grad_u_g.row(active_indices[i]);
                        D_new.row(i) = grad_x_g.row(active_indices[i]);
                    }

                    // Calculate feedback gains
                    Eigen::MatrixXd W = -(C_new * Q_uu_inv * C_new.transpose()).inverse() * (C_new * Q_uu_inv);
                    Eigen::MatrixXd H = Q_uu_inv * (Eigen::MatrixXd::Identity(control_dim, control_dim) - C_new.transpose() * W);
                    k = -H * Q_u;
                    K = -H * Q_ux + W.transpose() * D_new;
                }
                else
                {
                    // If no active constraints remain, revert to unconstrained solution
                    Eigen::MatrixXd H = Q_uu_reg.inverse();
                    K = -H * Q_ux;
                    k = -H * Q_u;
                }
            }

            // Store Q-function matrices and gains
            Q_UU_[t] = Q_uu_reg;
            Q_UX_[t] = Q_ux;
            Q_U_[t] = Q_u;
            k_u_[t] = k;
            K_u_[t] = K;

            // Update value function
            Eigen::Vector2d dV_step;
            dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
            dV_ += dV_step;

            V_x = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
            V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
            V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

            // 1-norm of the value function gradient
            norm_Vx += V_x.lpNorm<1>();

            // Update optimality gap
            Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());
        }

        // Normalize dual infeasibility 
        double scaling_factor = options.termination_scaling_max_factor;
        scaling_factor = std::max(scaling_factor, norm_Vx / (horizon * state_dim)) / scaling_factor;
        context.inf_du_ = Qu_error / scaling_factor;

        if (options.debug)
        {
            std::cout << "Qu_error: " << Qu_error << std::endl;
            std::cout << "dV: " << dV_.transpose() << std::endl;
        }

        return true;
    }

    ForwardPassResult ASDDPSolver::performForwardPass(CDDP &context)
    {
        const CDDPOptions &options = context.getOptions();
        ForwardPassResult best_result;
        best_result.cost = std::numeric_limits<double>::infinity();
        best_result.success = false;

        if (!options.enable_parallel)
        {
            // Single-threaded execution with early termination
            for (double alpha : context.alphas_)
            {
                ForwardPassResult result = forwardPass(context, alpha);

                if (result.success && result.cost < best_result.cost)
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

            for (double alpha : context.alphas_)
            {
                futures.push_back(std::async(std::launch::async,
                                             [this, &context, alpha]()
                                             { return forwardPass(context, alpha); }));
            }

            for (auto &future : futures)
            {
                try
                {
                    if (future.valid())
                    {
                        ForwardPassResult result = future.get();
                        if (result.success && result.cost < best_result.cost)
                        {
                            best_result = result;
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    if (options.verbose)
                    {
                        std::cerr << "ASDDP: Forward pass thread failed: " << e.what() << std::endl;
                    }
                }
            }
        }

        return best_result;
    }

    ForwardPassResult ASDDPSolver::forwardPass(CDDP &context, double alpha)
    {
        const CDDPOptions &options = context.getOptions();

        ForwardPassResult result;
        result.success = false;
        result.cost = std::numeric_limits<double>::infinity();
        result.merit_function = std::numeric_limits<double>::infinity();
        result.alpha = alpha;

        const int state_dim = context.getStateDim();
        const int control_dim = context.getControlDim();
        const int dual_dim = context.getTotalDualDim() - control_dim;
        const double timestep = context.getTimestep();

        // Extract control box constraint
        auto control_box_constraint = context.getConstraint<ControlBoxConstraint>("ControlBoxConstraint");

        // Initialize trajectories
        result.state_trajectory = context.X_;
        result.control_trajectory = context.U_;
        result.state_trajectory[0] = context.getInitialState();

        double J_new = 0.0;

        // Forward simulation with OSQP
        for (int t = 0; t < context.getHorizon(); ++t)
        {
            const Eigen::VectorXd &x = result.state_trajectory[t];
            const Eigen::VectorXd &u = result.control_trajectory[t];
            const Eigen::VectorXd delta_x = x - context.X_[t];

            // Extract Q-function matrices computed in the backward pass
            const Eigen::VectorXd &Q_u = Q_U_[t];
            const Eigen::MatrixXd &Q_uu = Q_UU_[t];
            const Eigen::MatrixXd &Q_ux = Q_UX_[t];

            // Create QP problem
            Eigen::SparseMatrix<double> P = Q_uu.sparseView();
            P.makeCompressed();

            // Form the gradient of the QP objective: q = alpha * Q_u + Q_ux * delta_x
            Eigen::VectorXd q = alpha * Q_u + Q_ux * delta_x;

            // Create QP constraints
            Eigen::MatrixXd A_dense = Eigen::MatrixXd::Identity(control_dim, control_dim);
            Eigen::VectorXd lb_dense = control_box_constraint->getLowerBound() - u;
            Eigen::VectorXd ub_dense = control_box_constraint->getUpperBound() - u;

            Eigen::MatrixXd A_aug = Eigen::MatrixXd::Zero(dual_dim, control_dim);
            Eigen::VectorXd lb_aug = Eigen::VectorXd::Zero(dual_dim);
            Eigen::VectorXd ub_aug = Eigen::VectorXd::Zero(dual_dim);

            // First block: control constraints
            A_aug.topRows(control_dim) = A_dense;
            lb_aug.head(control_dim) = lb_dense;
            ub_aug.head(control_dim) = ub_dense;

            // Second block: state constraints
            int row_index = control_dim;
            if (t < context.getHorizon() - 1)
            {
                auto [fx, fu] = context.getSystem().getJacobians(x, u, t * timestep);
                Eigen::MatrixXd Fu = timestep * fu;

                // Predicted next state
                Eigen::VectorXd x_next = context.getSystem().getDiscreteDynamics(x, u, t * timestep);

                const auto &constraint_set = context.getConstraintSet();
                for (const auto &[name, constraint] : constraint_set)
                {
                    if (name == "ControlBoxConstraint")
                    {
                        continue;
                    }
                    Eigen::VectorXd cons_vals = constraint->evaluate(x_next, u);
                    auto [cons_jac_x, cons_jac_u] = constraint->getJacobians(x_next, u);

                    int m = cons_vals.size();
                    A_aug.block(row_index, 0, m, control_dim) = cons_jac_x * Fu;
                    lb_aug.segment(row_index, m).setConstant(-std::numeric_limits<double>::infinity());
                    ub_aug.segment(row_index, m) = -cons_vals;
                    row_index += m;
                }
            }

            // Convert augmented constraint matrix to sparse format
            Eigen::SparseMatrix<double> A_sparse = A_aug.sparseView();

            // Initialize QP solver
            osqp::OsqpInstance instance;
            instance.objective_matrix = P;
            instance.objective_vector = q;
            instance.constraint_matrix = A_sparse;
            instance.lower_bounds = lb_aug;
            instance.upper_bounds = ub_aug;

            // Solve the QP problem
            osqp::OsqpSolver osqp_solver;
            osqp::OsqpSettings osqp_settings;
            osqp_settings.warm_start = true;
            osqp_settings.verbose = false;

            try
            {
                osqp_solver.Init(instance, osqp_settings);
                osqp::OsqpExitCode exit_code = osqp_solver.Solve();

                if (exit_code != osqp::OsqpExitCode::kOptimal)
                {
                    if (options.debug)
                    {
                        std::cerr << "ASDDP: QP solver failed at time step " << t << std::endl;
                    }
                    result.success = false;
                    return result;
                }

                // Update control using the QP solution delta_u
                Eigen::VectorXd delta_u = osqp_solver.primal_solution();
                result.control_trajectory[t] += delta_u;
            }
            catch (const std::exception &e)
            {
                if (options.debug)
                {
                    std::cerr << "ASDDP: OSQP exception at time step " << t << ": " << e.what() << std::endl;
                }
                result.success = false;
                return result;
            }

            // Compute running cost and propagate state
            J_new += context.getObjective().running_cost(x, result.control_trajectory[t], t);
            result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(x, result.control_trajectory[t], t * context.getTimestep());
        }

        // Add terminal cost
        J_new += context.getObjective().terminal_cost(result.state_trajectory.back());

        // Compute actual cost reduction and the predicted improvement
        double dJ = context.cost_ - J_new;
        double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
        double reduction_ratio = (expected > 0.0) ? dJ / expected : std::copysign(1.0, dJ);

        // Acceptance criterion
        if (dJ <= 0)
        {
            if (options.debug)
            {
                std::cerr << "ASDDP: Forward pass did not yield sufficient decrease (dJ: "
                          << dJ << ", reduction_ratio: " << reduction_ratio << ")" << std::endl;
            }
            result.success = false;
        }
        else
        {
            result.success = true;
            result.cost = J_new;
        }

        return result;
    }

    void ASDDPSolver::computeCost(CDDP &context)
    {
        context.cost_ = 0.0;

        // Running costs
        for (int t = 0; t < context.getHorizon(); ++t)
        {
            context.cost_ += context.getObjective().running_cost(context.X_[t], context.U_[t], t);
        }

        // Terminal cost
        context.cost_ += context.getObjective().terminal_cost(context.X_.back());
        context.merit_function_ = context.cost_; // For ASDDP, merit function equals cost
    }

    void ASDDPSolver::printIteration(int iter, double cost, double merit, double inf_du,
                                     double regularization, double alpha) const
    {
        if (iter == 0)
        {
            std::cout << std::setw(4) << "iter" << " "
                      << std::setw(12) << "objective" << " "
                      << std::setw(12) << "merit" << " "
                      << std::setw(10) << "inf_du" << " "
                      << std::setw(8) << "lg(rg)" << " "
                      << std::setw(8) << "alpha" << std::endl;
        }

        std::cout << std::setw(4) << iter << " "
                  << std::setw(12) << std::scientific << std::setprecision(4) << cost << " "
                  << std::setw(12) << std::scientific << std::setprecision(4) << merit << " "
                  << std::setw(10) << std::scientific << std::setprecision(2) << inf_du << " "
                  << std::setw(8) << std::fixed << std::setprecision(1) << std::log10(regularization) << " "
                  << std::setw(8) << std::fixed << std::setprecision(4) << alpha << std::endl;
    }

    void ASDDPSolver::printSolutionSummary(const CDDPSolution &solution) const
    {
        std::cout << "\n========================================\n";
        std::cout << "           ASDDP Solution Summary\n";
        std::cout << "========================================\n";

        auto iterations = std::any_cast<int>(solution.at("iterations_completed"));
        auto solve_time = std::any_cast<double>(solution.at("solve_time_ms"));
        auto final_cost = std::any_cast<double>(solution.at("final_objective"));
        auto status = std::any_cast<std::string>(solution.at("status_message"));
        auto final_inf_pr = std::any_cast<double>(solution.at("final_primal_infeasibility"));

        std::cout << "Status: " << status << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Solve Time: " << std::setprecision(2) << solve_time << " ms\n";
        std::cout << "Final Cost: " << std::setprecision(6) << final_cost << "\n";
        std::cout << "Final Constraint Violation: " << std::setprecision(4) << final_inf_pr << "\n";
        std::cout << "========================================\n\n";
    }

} // namespace cddp
