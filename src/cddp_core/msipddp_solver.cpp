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

#include "cddp_core/msipddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include <chrono>
#include <cmath>
#include <execution>
#include <future>
#include <iomanip>
#include <iostream>
#include <thread>

namespace cddp
{

  MSIPDDPSolver::MSIPDDPSolver() : mu_(1e-1), ms_segment_length_(5) {}

  void MSIPDDPSolver::initialize(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const auto &constraint_set = context.getConstraintSet();

    int horizon = context.getHorizon();
    int control_dim = context.getControlDim();
    int state_dim = context.getStateDim();

    // Initialize workspace if not already done
    if (!workspace_.initialized) {
      // Allocate backward pass workspace
      workspace_.A_matrices.resize(horizon);
      workspace_.B_matrices.resize(horizon);
      workspace_.Q_xx_matrices.resize(horizon);
      workspace_.Q_ux_matrices.resize(horizon);
      workspace_.Q_uu_matrices.resize(horizon);
      workspace_.Q_x_vectors.resize(horizon);
      workspace_.Q_u_vectors.resize(horizon);
      
      // Allocate LDLT solver cache
      workspace_.ldlt_solvers.resize(horizon);
      workspace_.ldlt_valid.resize(horizon, false);
      
      // Allocate forward pass workspace
      workspace_.delta_x_vectors.resize(horizon + 1);
      
      // MSIPDDP-specific: defect vectors
      workspace_.d_vectors.resize(horizon);
      
      for (int t = 0; t < horizon; ++t) {
        workspace_.A_matrices[t] = Eigen::MatrixXd::Zero(state_dim, state_dim);
        workspace_.B_matrices[t] = Eigen::MatrixXd::Zero(state_dim, control_dim);
        workspace_.Q_xx_matrices[t] = Eigen::MatrixXd::Zero(state_dim, state_dim);
        workspace_.Q_ux_matrices[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
        workspace_.Q_uu_matrices[t] = Eigen::MatrixXd::Zero(control_dim, control_dim);
        workspace_.Q_x_vectors[t] = Eigen::VectorXd::Zero(state_dim);
        workspace_.Q_u_vectors[t] = Eigen::VectorXd::Zero(control_dim);
        workspace_.d_vectors[t] = Eigen::VectorXd::Zero(state_dim);
      }
      
      for (int t = 0; t <= horizon; ++t) {
        workspace_.delta_x_vectors[t] = Eigen::VectorXd::Zero(state_dim);
      }
      
      // Allocate constraint workspace if needed
      if (!constraint_set.empty()) {
        int total_dual_dim = getTotalDualDim(context);
        workspace_.y_combined = Eigen::VectorXd::Zero(total_dual_dim);
        workspace_.s_combined = Eigen::VectorXd::Zero(total_dual_dim);
        workspace_.g_combined = Eigen::VectorXd::Zero(total_dual_dim);
        workspace_.Q_yu_combined = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);
        workspace_.Q_yx_combined = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);
        workspace_.YSinv = Eigen::MatrixXd::Zero(total_dual_dim, total_dual_dim);
        workspace_.bigRHS = Eigen::MatrixXd::Zero(control_dim, 1 + state_dim);
      }
      
      workspace_.initialized = true;
    }

    // Validate reference state consistency
    if ((context.getReferenceState() - context.getObjective().getReferenceState()).norm() > 1e-6)
    {
      throw std::runtime_error("MSIPDDP: Reference state mismatch between context and objective");
    }

    // For warm starts, verify that existing state is valid
    if (options.warm_start)
    {
      bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                               K_u_.size() == static_cast<size_t>(horizon));

      if (valid_warm_start && !k_u_.empty())
      {
        for (int t = 0; t < horizon; ++t)
        {
          if (k_u_[t].size() != control_dim || K_u_[t].rows() != control_dim ||
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

      // For constrained problems, we don't require pre-existing dual/slack
      // variables They will be re-initialized properly during warm start

      if (valid_warm_start)
      {
        if (options.verbose)
        {
          std::cout << "MSIPDDP: Using warm start with existing control gains"
                    << std::endl;
        }
        mu_ = options.msipddp.barrier.mu_initial * 0.1;
        context.step_norm_ = 0.0;
        evaluateTrajectoryWarmStart(context);
        initializeDualSlackCostateVariablesWarmStart(context);
        resetFilter(context);
        return;
      }
      else
      {
        // Warm start with provided trajectory (no existing solver state)
        if (options.verbose)
        {
          std::cout << "MSIPDDP: Warm start with provided trajectory" << std::endl;
        }

        // Initialize gains and constraints
        k_u_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
        K_u_.assign(horizon, Eigen::MatrixXd::Zero(control_dim, state_dim));
        dV_ = Eigen::Vector2d::Zero();

        // Initialize MSIPDDP-specific costate variables and gains
        Lambda_.assign(horizon, Eigen::VectorXd::Zero(state_dim));
        k_lambda_.assign(horizon, Eigen::VectorXd::Zero(state_dim));
        K_lambda_.assign(horizon, Eigen::MatrixXd::Zero(state_dim, state_dim));

        // Initialize dynamics storage for multi-shooting
        F_.assign(horizon, Eigen::VectorXd::Zero(state_dim));

        // Initialize costate variables with scaling
        for (int t = 0; t < horizon; ++t)
        {
          Lambda_[t] = options.msipddp.costate_var_init_scale * Eigen::VectorXd::Ones(state_dim);
        }

        // Set multi-shooting segment length from options
        ms_segment_length_ = options.msipddp.segment_length;
        if (ms_segment_length_ < 0)
        {
          throw std::runtime_error("MSIPDDP: ms_segment_length must be non-negative");
        }

        initializeConstraintStorage(context);

        // Set barrier parameter based on constraint evaluation
        if (constraint_set.empty())
        {
          mu_ = 1e-8; // Small value if no constraints
        }
        else
        {
          // Evaluate constraints and set barrier parameter
          evaluateTrajectoryWarmStart(context);
          double max_constraint_violation = computeMaxConstraintViolation(context);
          if (max_constraint_violation <= options.tolerance)
          {
            mu_ = options.tolerance * 0.01; // Feasible trajectory
          }
          else if (max_constraint_violation <= 0.1)
          {
            mu_ = options.tolerance; // Slightly infeasible
          }
          else
          {
            mu_ = options.msipddp.barrier.mu_initial * 0.1; // Significantly infeasible
          }
        }

        // Initialize regularization
        context.regularization_ = options.regularization.initial_value;

        // Initialize step norm
        context.step_norm_ = 0.0;

        // Initialize dual/slack variables
        initializeDualSlackCostateVariablesWarmStart(context);
        resetFilter(context);
        return;
      }
    }

    // Cold start: check if trajectory is provided
    bool trajectory_provided = (context.X_.size() == static_cast<size_t>(horizon + 1) &&
                                context.U_.size() == static_cast<size_t>(horizon) &&
                                context.X_[0].size() == state_dim &&
                                context.U_[0].size() == control_dim);

    if (!trajectory_provided)
    {
      // Create interpolated initial trajectory
      context.X_.resize(horizon + 1);
      context.U_.resize(horizon);

      for (int t = 0; t <= horizon; ++t)
      {
        context.X_[t] = context.getInitialState() +
                        t * (context.getReferenceState() - context.getInitialState()) / horizon;
      }

      for (int t = 0; t < horizon; ++t)
      {
        context.U_[t] = Eigen::VectorXd::Zero(control_dim);
      }

      if (options.verbose)
      {
        std::cout << "MSIPDDP: Using interpolated initial trajectory" << std::endl;
      }
    }
    else if (options.verbose)
    {
      std::cout << "MSIPDDP: Using provided initial trajectory" << std::endl;
    }

    // Initialize gains, constraints, and parameters
    k_u_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
    K_u_.assign(horizon, Eigen::MatrixXd::Zero(control_dim, state_dim));
    dV_ = Eigen::Vector2d::Zero();

    // Initialize MSIPDDP-specific costate variables and gains
    Lambda_.assign(horizon, Eigen::VectorXd::Zero(state_dim));
    k_lambda_.assign(horizon, Eigen::VectorXd::Zero(state_dim));
    K_lambda_.assign(horizon, Eigen::MatrixXd::Zero(state_dim, state_dim));

    // Initialize dynamics storage for multi-shooting
    F_.assign(horizon, Eigen::VectorXd::Zero(state_dim));

    // Initialize costate variables with scaling
    for (int t = 0; t < horizon; ++t)
    {
      Lambda_[t] = options.msipddp.costate_var_init_scale * Eigen::VectorXd::Ones(state_dim);
    }

    // Set multi-shooting segment length from options
    ms_segment_length_ = options.msipddp.segment_length;
    if (ms_segment_length_ < 0)
    {
      throw std::runtime_error("MSIPDDP: ms_segment_length must be non-negative");
    }

    initializeConstraintStorage(context);

    // Set barrier parameter
    if (constraint_set.empty())
    {
      mu_ = 1e-8;
    }
    else
    {
      mu_ = options.msipddp.barrier.mu_initial;
    }

    initializeDualSlackCostateVariables(context);
    context.regularization_ = options.regularization.initial_value;
    context.step_norm_ = 0.0;
    evaluateTrajectory(context);
    resetFilter(context);
  }

  std::string MSIPDDPSolver::getSolverName() const { return "MSIPDDP"; }

  int MSIPDDPSolver::getTotalDualDim(const CDDP &context) const
  {
    int total_dual_dim = 0;
    const auto &constraint_set = context.getConstraintSet();
    for (const auto &constraint_pair : constraint_set)
    {
      total_dual_dim += constraint_pair.second->getDualDim();
    }
    return total_dual_dim;
  }

  CDDPSolution MSIPDDPSolver::solve(CDDP &context)
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
    std::vector<double> history_step_length_dual;
    std::vector<double> history_dual_infeasibility;
    std::vector<double> history_primal_infeasibility;
    std::vector<double> history_complementary_infeasibility;
    std::vector<double> history_barrier_mu;

    if (options.return_iteration_info)
    {
      const size_t expected_size =
          static_cast<size_t>(options.max_iterations + 1);
      history_objective.reserve(expected_size);
      history_merit_function.reserve(expected_size);
      history_step_length_primal.reserve(expected_size);
      history_step_length_dual.reserve(expected_size);
      history_dual_infeasibility.reserve(expected_size);
      history_primal_infeasibility.reserve(expected_size);
      history_complementary_infeasibility.reserve(expected_size);
      history_barrier_mu.reserve(expected_size);

      // Initial iteration values
      history_objective.push_back(context.cost_);
      history_merit_function.push_back(context.merit_function_);
      history_step_length_primal.push_back(1.0); // Initial step length
      history_step_length_dual.push_back(1.0);   // Initial dual step length
      history_dual_infeasibility.push_back(context.inf_du_);
      history_primal_infeasibility.push_back(context.inf_pr_);
      history_complementary_infeasibility.push_back(context.inf_comp_);
      history_barrier_mu.push_back(mu_);
    }

    if (options.verbose)
    {
      printIteration(0, context.cost_, context.inf_pr_, context.inf_du_,
                     context.inf_comp_, mu_, context.step_norm_,
                     context.regularization_, context.alpha_du_,
                     context.alpha_pr_);
    }

    // Main solve loop
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;
    bool converged = false;
    std::string termination_reason = "MaxIterationsReached";
    double dJ = 0.0;

    while (iter < options.max_iterations)
    {
      ++iter;

      // Check CPU time limit
      if (options.max_cpu_time > 0)
      {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        if (duration.count() > options.max_cpu_time * 1000)
        {
          termination_reason = "MaxCpuTimeReached";
          if (options.verbose)
          {
            std::cerr << "MSIPDDP: Maximum CPU time reached" << std::endl;
          }
          break;
        }
      }

      // Backward pass with regularization
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
              std::cerr << "MSIPDDP: Regularization limit reached" << std::endl;
            }
            break;
          }
        }
      }
      if (!backward_pass_success)
        break;

      // Forward pass
      ForwardPassResult best_result = performForwardPass(context);

      // Update trajectories if forward pass succeeded
      if (best_result.success)
      {
        if (options.debug)
        {
          std::cout << "[MSIPDDP Forward] cost: " << std::scientific << std::setprecision(4)
                    << best_result.cost << " α: " << best_result.alpha_pr
                    << " cv: " << best_result.constraint_violation << std::endl;
        }

        // Update trajectories and variables
        context.X_ = best_result.state_trajectory;
        context.U_ = best_result.control_trajectory;
        if (best_result.dual_trajectory)
          Y_ = *best_result.dual_trajectory;
        if (best_result.slack_trajectory)
          S_ = *best_result.slack_trajectory;
        if (best_result.constraint_eval_trajectory)
          G_ = *best_result.constraint_eval_trajectory;
        if (best_result.dynamics_trajectory)
          F_ = *best_result.dynamics_trajectory;
        if (best_result.costate_trajectory)
          Lambda_ = *best_result.costate_trajectory;

        // Update costs and step lengths
        dJ = context.cost_ - best_result.cost;
        context.cost_ = best_result.cost;
        context.merit_function_ = best_result.merit_function;
        context.alpha_pr_ = best_result.alpha_pr;

        // Update filter with accepted point
        acceptFilterEntry(best_result.merit_function, best_result.constraint_violation);

        updateIterationHistory(options, context, history_objective, history_merit_function,
                               history_step_length_primal, history_step_length_dual,
                               history_dual_infeasibility, history_primal_infeasibility,
                               history_complementary_infeasibility, history_barrier_mu,
                               best_result.alpha_du);

        context.decreaseRegularization();
      }
      else
      {
        // Try filter restoration before increasing regularization
        bool restoration_performed = checkAndPerformFilterRestoration(context);
        
        if (!restoration_performed)
        {
          context.increaseRegularization();
          if (context.isRegularizationLimitReached())
          {
            termination_reason = "RegularizationLimitReached_NotConverged";
            converged = false;
            if (options.verbose)
            {
              std::cerr << "MSIPDDP: Regularization limit reached" << std::endl;
            }
            break;
          }
        }
        else if (options.debug)
        {
          std::cout << "MSIPDDP: Filter restoration performed, retrying forward pass" << std::endl;
        }
      }

      // Check convergence
      converged = checkConvergence(options, context, dJ, iter, termination_reason);
      if (converged)
        break;

      // Print iteration info
      if (options.verbose)
      {
        printIteration(iter, context.cost_, context.inf_pr_, context.inf_du_,
                       context.inf_comp_, mu_, context.step_norm_,
                       context.regularization_, best_result.alpha_du,
                       context.alpha_pr_);
      }

      // Update barrier parameters using the extracted method
      updateBarrierParameters(context, best_result.success);
    }

    // Compute final timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

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
      solution["history_step_length_dual"] = history_step_length_dual;
      solution["history_dual_infeasibility"] = history_dual_infeasibility;
      solution["history_primal_infeasibility"] = history_primal_infeasibility;
      solution["history_complementary_infeasibility"] = history_complementary_infeasibility;
      solution["history_barrier_mu"] = history_barrier_mu;
    }

    // Add control gains
    solution["control_feedback_gains_K"] = K_u_;

    // Final metrics
    solution["final_regularization"] = context.regularization_;
    solution["final_barrier_parameter_mu"] = mu_;
    solution["final_primal_infeasibility"] = context.inf_pr_;
    solution["final_dual_infeasibility"] = context.inf_du_;
    solution["final_complementary_infeasibility"] = context.inf_comp_;

    if (options.verbose)
    {
      printSolutionSummary(solution);
    }

    return solution;
  }

  void MSIPDDPSolver::evaluateTrajectory(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
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

      // Evaluate and store dynamics for multi-shooting
      F_[t] = context.getSystem().getDiscreteDynamics(x, u, t * context.getTimestep());

      // Compute next state using dynamics
      // For initial trajectory evaluation, we always propagate the dynamics
      context.X_[t + 1] = F_[t];
    }

    // Add terminal cost
    cost += context.getObjective().terminal_cost(context.X_.back());

    // Store the cost
    context.cost_ = cost;
  }

  void MSIPDDPSolver::evaluateTrajectoryWarmStart(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    double cost = 0.0;

    // For warm start, the trajectory (X_, U_) is already provided
    // We just need to evaluate the cost and constraints

    // Initialize constraint storage first
    const auto &constraint_set = context.getConstraintSet();
    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      G_[constraint_name].resize(horizon);
    }

    // Rollout dynamics and calculate cost
    for (int t = 0; t < horizon; ++t)
    {
      const Eigen::VectorXd &x = context.X_[t];
      const Eigen::VectorXd &u = context.U_[t];

      // Compute stage cost
      cost += context.getObjective().running_cost(x, u, t);

      // For each constraint, evaluate and store the constraint value
      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &constraint_name = constraint_pair.first;
        Eigen::VectorXd g_val = constraint_pair.second->evaluate(x, u) -
                                constraint_pair.second->getUpperBound();
        G_[constraint_name][t] = g_val;
      }

      // Evaluate dynamics for multi-shooting
      F_[t] = context.getSystem().getDiscreteDynamics(x, u, t * context.getTimestep());

      if (options.msipddp.use_controlled_rollout)
      {
        context.X_[t + 1] = F_[t];
      }
    }

    // Add terminal cost
    cost += context.getObjective().terminal_cost(context.X_.back());

    // Store the cost
    context.cost_ = cost;
  }

  void MSIPDDPSolver::initializeDualSlackCostateVariablesWarmStart(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const auto &constraint_set = context.getConstraintSet();

    // Check if we have existing dual/slack variables from previous solve
    bool has_existing_dual_slack = true;
    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      if (Y_.find(constraint_name) == Y_.end() ||
          S_.find(constraint_name) == S_.end() ||
          Y_[constraint_name].size() != static_cast<size_t>(horizon) ||
          S_[constraint_name].size() != static_cast<size_t>(horizon))
      {
        has_existing_dual_slack = false;
        break;
      }
    }

    // Initialize/resize gains storage for all constraints
    k_y_.clear();
    K_y_.clear();
    k_s_.clear();
    K_s_.clear();

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      int dual_dim = constraint_pair.second->getDualDim();

      // Ensure proper sizing
      if (!has_existing_dual_slack)
      {
        Y_[constraint_name].resize(horizon);
        S_[constraint_name].resize(horizon);
      }

      k_y_[constraint_name].resize(horizon);
      K_y_[constraint_name].resize(horizon);
      k_s_[constraint_name].resize(horizon);
      K_s_[constraint_name].resize(horizon);

      for (int t = 0; t < horizon; ++t)
      {
        // Use the already evaluated constraint values from
        // evaluateTrajectoryWarmStart
        const Eigen::VectorXd &g_val = G_[constraint_name][t];

        bool need_reinit = false;
        Eigen::VectorXd y_current, s_current;

        if (has_existing_dual_slack)
        {
          y_current = Y_[constraint_name][t];
          s_current = S_[constraint_name][t];

          // Check if existing dual/slack variables are feasible and compatible
          if (y_current.size() != dual_dim || s_current.size() != dual_dim)
          {
            need_reinit = true;
          }
          else
          {
            // Check feasibility conditions
            for (int i = 0; i < dual_dim; ++i)
            {
              // Check positivity: y_i > 0 and s_i > 0
              if (y_current(i) <= 1e-12 || s_current(i) <= 1e-12)
              {
                need_reinit = true;
                break;
              }

              // Check if constraint is severely violated (slack should be
              // reasonable)
              double required_slack =
                  std::max(options.msipddp.slack_var_init_scale, -g_val(i));
              if (s_current(i) < 0.1 * required_slack)
              {
                need_reinit = true;
                break;
              }
            }
          }
        }
        else
        {
          need_reinit = true;
        }

        if (need_reinit)
        {
          // Use the same initialization as cold start for consistency
          Eigen::VectorXd s_init = Eigen::VectorXd::Zero(dual_dim);
          Eigen::VectorXd y_init = Eigen::VectorXd::Zero(dual_dim);

          for (int i = 0; i < dual_dim; ++i)
          {
            // Initialize s_i = max(slack_scale, -g_i) to ensure s_i > 0 (same as
            // cold start)
            s_init(i) = std::max(options.msipddp.slack_var_init_scale, -g_val(i));

            // Initialize y_i = mu / s_i to satisfy s_i * y_i = mu (same as cold
            // start)
            if (s_init(i) < 1e-12)
            {
              y_init(i) = mu_ / 1e-12;
            }
            else
            {
              y_init(i) = mu_ / s_init(i);
            }
            // Clamp dual variable (same as cold start)
            y_init(i) = std::max(
                options.msipddp.dual_var_init_scale * 0.01,
                std::min(y_init(i), options.msipddp.dual_var_init_scale * 100.0));
          }
          Y_[constraint_name][t] = y_init;
          S_[constraint_name][t] = s_init;
        }

        // Always initialize gains to zero
        k_y_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        K_y_[constraint_name][t] =
            Eigen::MatrixXd::Zero(dual_dim, context.getStateDim());
        k_s_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        K_s_[constraint_name][t] =
            Eigen::MatrixXd::Zero(dual_dim, context.getStateDim());
      }
    }

    // Initialize or preserve costate variables for MSIPDDP
    bool has_existing_costate = (Lambda_.size() == static_cast<size_t>(horizon));

    if (!has_existing_costate)
    {
      // Initialize costate variables for first time
      for (int t = 0; t < horizon; ++t)
      {
        Lambda_[t] = options.msipddp.costate_var_init_scale * Eigen::VectorXd::Ones(context.getStateDim());
        k_lambda_[t] = Eigen::VectorXd::Zero(context.getStateDim());
        K_lambda_[t] = Eigen::MatrixXd::Zero(context.getStateDim(), context.getStateDim());
      }
    }
    else
    {
      // Preserve existing costate variables, just initialize gains
      for (int t = 0; t < horizon; ++t)
      {
        k_lambda_[t] = Eigen::VectorXd::Zero(context.getStateDim());
        K_lambda_[t] = Eigen::MatrixXd::Zero(context.getStateDim(), context.getStateDim());
      }
    }

    if (options.verbose)
    {
      std::cout << "MSIPDDP: " << (has_existing_dual_slack ? "Preserved" : "Initialized")
                << " dual/slack variables, " << (has_existing_costate ? "preserved" : "initialized")
                << " costate variables, μ = " << std::scientific << std::setprecision(2)
                << mu_ << ", max violation = " << computeMaxConstraintViolation(context) << std::endl;
    }
  }

  void MSIPDDPSolver::initializeDualSlackCostateVariables(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const auto &constraint_set = context.getConstraintSet();

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
        Eigen::VectorXd g_val =
            constraint_pair.second->evaluate(context.X_[t], context.U_[t]) -
            constraint_pair.second->getUpperBound();
        G_[constraint_name][t] = g_val;

        Eigen::VectorXd s_init = Eigen::VectorXd::Zero(dual_dim);
        Eigen::VectorXd y_init = Eigen::VectorXd::Zero(dual_dim);

        for (int i = 0; i < dual_dim; ++i)
        {
          // Initialize s_i = max(slack_scale, -g_i) to ensure s_i > 0
          s_init(i) = std::max(options.msipddp.slack_var_init_scale, -g_val(i));

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
          y_init(i) = std::max(
              options.msipddp.dual_var_init_scale * 0.01,
              std::min(y_init(i), options.msipddp.dual_var_init_scale * 100.0));
        }
        Y_[constraint_name][t] = y_init;
        S_[constraint_name][t] = s_init;

        // Initialize gains to zero
        k_y_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        K_y_[constraint_name][t] =
            Eigen::MatrixXd::Zero(dual_dim, context.getStateDim());
        k_s_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        K_s_[constraint_name][t] =
            Eigen::MatrixXd::Zero(dual_dim, context.getStateDim());
      }
    }

    // Initialize costate variables for MSIPDDP
    for (int t = 0; t < horizon; ++t)
    {
      // Initialize costate variables with proper scaling
      Lambda_[t] = options.msipddp.costate_var_init_scale * Eigen::VectorXd::Ones(context.getStateDim());

      // Initialize costate gains to zero
      k_lambda_[t] = Eigen::VectorXd::Zero(context.getStateDim());
      K_lambda_[t] = Eigen::MatrixXd::Zero(context.getStateDim(), context.getStateDim());
    }

    // Initialize cost using objective evaluation
    context.cost_ = context.getObjective().evaluate(context.X_, context.U_);
  }

  void MSIPDDPSolver::resetBarrierFilter(CDDP &context)
  {
    // Evaluate merit function (cost + log-barrier terms)
    double merit_function = context.cost_;
    double inf_pr = 0.0;                      // inf_pr: infinity norm (largest absolute residual)
    double filter_constraint_violation = 0.0; // l1 norm for filter (sum of residuals)
    double inf_du = 0.0;                      // dual infeasibility (computed separately in backward pass)
    double inf_comp = 0.0;                    // complementary infeasibility
    double inf_defect = 0.0;                  // defect infeasibility for multi-shooting

    const auto &constraint_set = context.getConstraintSet();

    if (!constraint_set.empty())
    {
      for (int t = 0; t < context.getHorizon(); ++t)
      {
        for (const auto &constraint_pair : constraint_set)
        {
          const std::string &constraint_name = constraint_pair.first;
          const Eigen::VectorXd &s_vec = S_[constraint_name][t];
          const Eigen::VectorXd &g_vec = G_[constraint_name][t];
          const Eigen::VectorXd &y_vec = Y_[constraint_name][t];

          // Add log-barrier term
          merit_function -= mu_ * s_vec.array().log().sum();

          // Compute primal residual vector
          Eigen::VectorXd primal_residual = g_vec + s_vec;

          // inf_pr: infinity norm (largest absolute residual)
          inf_pr = std::max(inf_pr, primal_residual.lpNorm<Eigen::Infinity>());

          // Filter constraint violation: l1 norm (sum of residuals)
          filter_constraint_violation += primal_residual.lpNorm<1>();

          // Compute complementary infeasibility: ||y .* s - mu||_inf
          Eigen::VectorXd complementary_residual = y_vec.cwiseProduct(s_vec).array() - mu_;
          inf_comp = std::max(inf_comp, complementary_residual.lpNorm<Eigen::Infinity>());
        }

        // Add defect residual calculation 
        if (t < static_cast<int>(F_.size()) && (t + 1) < static_cast<int>(context.X_.size()))
        {
          Eigen::VectorXd defect_residual = F_[t] - context.X_[t + 1];
          inf_defect = std::max(inf_defect, defect_residual.lpNorm<Eigen::Infinity>());
          filter_constraint_violation += defect_residual.lpNorm<1>();
        }
      }
    }
    else
    {
      // No constraints: set infeasibility metrics to zero
      inf_pr = 0.0;
      filter_constraint_violation = 0.0;
      inf_du = 0.0;
      inf_comp = 0.0;
    }

    // Update primal infeasibility to include defect violations
    context.inf_pr_ = std::max(inf_pr, inf_defect);
    context.merit_function_ = merit_function;
    // Note: inf_du_ (dual infeasibility/optimality gap) is computed in backward pass for constrained case
    context.inf_comp_ = inf_comp;

    // Simple filter initialization with current point
    filter_.clear();
    filter_.push_back(FilterPoint(merit_function, filter_constraint_violation));
  }

  void MSIPDDPSolver::resetFilter(CDDP &context) { resetBarrierFilter(context); }

  bool MSIPDDPSolver::acceptFilterEntry(double merit_function, double constraint_violation)
  {
    FilterPoint candidate(merit_function, constraint_violation);
    
    // Check if candidate is dominated by any existing filter point
    for (const auto &filter_point : filter_)
    {
      if (filter_point.dominates(candidate))
      {
        return false; // Candidate is dominated, reject
      }
    }
    
    // Remove any filter points that are dominated by the candidate
    filter_.erase(
        std::remove_if(filter_.begin(), filter_.end(),
                      [&candidate](const FilterPoint &point) {
                        return candidate.dominates(point);
                      }),
        filter_.end());
    
    // Add the candidate to the filter
    filter_.push_back(candidate);
    
    return true;
  }

  bool MSIPDDPSolver::isFilterAcceptable(double merit_function, double constraint_violation,
                                        const SolverSpecificFilterOptions &options,
                                        double expected_improvement) const
  {
    // If filter is empty, any point is acceptable
    if (filter_.empty())
    {
      return true;
    }
    
    // Quick domination check: reject if any filter point dominates candidate
    FilterPoint candidate(merit_function, constraint_violation);
    for (const auto &filter_point : filter_)
    {
      if (filter_point.dominates(candidate))
      {
        return false;
      }
    }
    
    // Find best filter point for comparison
    double best_violation = std::numeric_limits<double>::infinity();
    double best_merit = std::numeric_limits<double>::infinity();
    
    for (const auto &filter_point : filter_)
    {
      if (filter_point.constraint_violation < best_violation)
      {
        best_violation = filter_point.constraint_violation;
        best_merit = filter_point.merit_function;
      }
    }
    
    // Simple acceptance test: either improve constraint violation or merit function
    bool violation_improvement = constraint_violation < best_violation * (1.0 - options.violation_acceptance_threshold);
    bool merit_improvement = merit_function < best_merit - options.merit_acceptance_threshold * constraint_violation;
    
    // For very small violations, use Armijo condition
    if (constraint_violation < options.min_violation_for_armijo_check && expected_improvement < 0)
    {
      return merit_function < best_merit + options.armijo_constant * expected_improvement;
    }
    
    // Additional acceptance criterion: if we're very close to feasibility and making any improvement
    if (constraint_violation < 1e-6 && merit_function <= best_merit * (1.0 + 1e-8))
    {
      return true;  // Accept small improvements when nearly feasible
    }
    
    return violation_improvement || merit_improvement;
  }

  bool MSIPDDPSolver::checkAndPerformFilterRestoration(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    
    // Simple restoration criteria: too many points or invalid values
    bool needs_restoration = (filter_.size() > 5);
    
    // Check for numerical issues
    if (!needs_restoration)
    {
      for (const auto &point : filter_)
      {
        if (std::isnan(point.merit_function) || std::isnan(point.constraint_violation) ||
            std::isinf(point.merit_function) || std::isinf(point.constraint_violation))
        {
          needs_restoration = true;
          break;
        }
      }
    }
    
    if (needs_restoration && !filter_.empty())
    {
      if (options.debug)
      {
        std::cout << "MSIPDDP: Filter restoration: " << filter_.size() << " -> ";
      }
      
      // Keep only the best 2 points: best violation and best merit function
      auto best_violation = *std::min_element(filter_.begin(), filter_.end(),
        [](const FilterPoint &a, const FilterPoint &b) {
          return a.constraint_violation < b.constraint_violation;
        });
      
      auto best_merit = *std::min_element(filter_.begin(), filter_.end(),
        [](const FilterPoint &a, const FilterPoint &b) {
          return a.merit_function < b.merit_function;
        });
      
      filter_.clear();
      filter_.push_back(best_violation);
      if (std::abs(best_merit.constraint_violation - best_violation.constraint_violation) > 1e-12 ||
          std::abs(best_merit.merit_function - best_violation.merit_function) > 1e-12)
      {
        filter_.push_back(best_merit);
      }
      
      if (options.debug)
      {
        std::cout << filter_.size() << " points" << std::endl;
      }
      
      return true;
    }
    
    return false;
  }

  void MSIPDDPSolver::precomputeDynamicsDerivatives(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const int state_dim = context.getStateDim();
    const double timestep = context.getTimestep();

    // Resize storage
    F_x_.resize(horizon);
    F_u_.resize(horizon);
    F_xx_.resize(horizon);
    F_uu_.resize(horizon);
    F_ux_.resize(horizon);

    // Use parallel computation for larger horizons
    const int MIN_HORIZON_FOR_PARALLEL = 50;
    const bool use_parallel = options.enable_parallel && horizon >= MIN_HORIZON_FOR_PARALLEL;

    if (!use_parallel)
    {
      // Single-threaded computation
      for (int t = 0; t < horizon; ++t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        // Compute jacobians
        const auto [Fx, Fu] =
            context.getSystem().getJacobians(x, u, t * timestep);
        F_x_[t] = Fx;
        F_u_[t] = Fu;

        // Compute hessians if not using iLQR
        if (!options.use_ilqr)
        {
          const auto hessians =
              context.getSystem().getHessians(x, u, t * timestep);
          F_xx_[t] = std::get<0>(hessians);
          F_uu_[t] = std::get<1>(hessians);
          F_ux_[t] = std::get<2>(hessians);
        }
        else
        {
          // Initialize empty hessians for iLQR
          F_xx_[t] = std::vector<Eigen::MatrixXd>();
          F_uu_[t] = std::vector<Eigen::MatrixXd>();
          F_ux_[t] = std::vector<Eigen::MatrixXd>();
        }
      }
    }
    else
    {
      // Parallel computation
      const int num_threads = std::min(options.num_threads,
                                       static_cast<int>(std::thread::hardware_concurrency()));
      const int chunk_size = std::max(1, horizon / num_threads);

      std::vector<std::future<void>> futures;
      futures.reserve(num_threads);

      for (int thread_id = 0; thread_id < num_threads; ++thread_id)
      {
        int start_t = thread_id * chunk_size;
        int end_t = (thread_id == num_threads - 1) ? horizon : (thread_id + 1) * chunk_size;

        if (start_t >= horizon)
          break;

        futures.push_back(std::async(std::launch::async,
                                     [this, &context, &options, start_t, end_t, timestep]()
                                     {
            // Process chunk of time steps
            for (int t = start_t; t < end_t; ++t) {
              const Eigen::VectorXd &x = context.X_[t];
              const Eigen::VectorXd &u = context.U_[t];

              // Compute jacobians
              const auto [Fx, Fu] =
                  context.getSystem().getJacobians(x, u, t * timestep);
              F_x_[t] = Fx;
              F_u_[t] = Fu;

              // Compute hessians if not using iLQR
              if (!options.use_ilqr) {
                const auto hessians =
                    context.getSystem().getHessians(x, u, t * timestep);
                F_xx_[t] = std::get<0>(hessians);
                F_uu_[t] = std::get<1>(hessians);
                F_ux_[t] = std::get<2>(hessians);
              } else {
                // Initialize empty hessians for iLQR
                F_xx_[t] = std::vector<Eigen::MatrixXd>();
                F_uu_[t] = std::vector<Eigen::MatrixXd>();
                F_ux_[t] = std::vector<Eigen::MatrixXd>();
              }
            } }));
      }

      // Wait for all computations to complete
      for (auto &future : futures)
      {
        try
        {
          if (future.valid())
          {
            future.get();
          }
        }
        catch (const std::exception &e)
        {
          if (options.verbose)
          {
            std::cerr << "MSIPDDP: Dynamics derivatives computation thread failed: "
                      << e.what() << std::endl;
          }
          throw;
        }
      }
    }
  }

  void MSIPDDPSolver::precomputeConstraintGradients(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const auto &constraint_set = context.getConstraintSet();

    // Clear and resize storage
    G_x_.clear();
    G_u_.clear();
    G_xx_.clear();
    G_uu_.clear();
    G_ux_.clear();

    // If no constraints, return early
    if (constraint_set.empty())
    {
      return;
    }

    // Initialize storage for each constraint only if not already allocated
    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      if (G_x_.find(constraint_name) == G_x_.end() || G_x_[constraint_name].size() != horizon) {
        G_x_[constraint_name].resize(horizon);
        G_u_[constraint_name].resize(horizon);
        // Pre-allocate matrices with correct dimensions
        int state_dim = context.getStateDim();
        int control_dim = context.getControlDim();
        int constraint_dim = constraint_pair.second->getDualDim();
        for (int t = 0; t < horizon; ++t) {
          G_x_[constraint_name][t] = Eigen::MatrixXd::Zero(constraint_dim, state_dim);
          G_u_[constraint_name][t] = Eigen::MatrixXd::Zero(constraint_dim, control_dim);
        }
      }
      G_xx_[constraint_name].resize(horizon);
      G_uu_[constraint_name].resize(horizon);
      G_ux_[constraint_name].resize(horizon);
    }

    // Threshold for when parallelization is worth it - increased for better
    // performance
    const int MIN_HORIZON_FOR_PARALLEL = 50;
    const bool use_parallel =
        options.enable_parallel && horizon >= MIN_HORIZON_FOR_PARALLEL;

    if (!use_parallel)
    {
      // Single-threaded computation
      for (int t = 0; t < horizon; ++t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        for (const auto &constraint_pair : constraint_set)
        {
          const std::string &constraint_name = constraint_pair.first;
          G_x_[constraint_name][t] =
              constraint_pair.second->getStateJacobian(x, u);
          G_u_[constraint_name][t] =
              constraint_pair.second->getControlJacobian(x, u);

          // Compute constraint hessians if not using iLQR
          if (!options.use_ilqr)
          {
            G_xx_[constraint_name][t] = constraint_pair.second->getStateHessian(x, u);
            G_uu_[constraint_name][t] = constraint_pair.second->getControlHessian(x, u);
            G_ux_[constraint_name][t] = constraint_pair.second->getCrossHessian(x, u);
          }
        }
      }
    }
    else
    {
      // Chunked parallel computation
      const int num_threads =
          std::min(options.num_threads,
                   static_cast<int>(std::thread::hardware_concurrency()));
      const int chunk_size = std::max(1, horizon / num_threads);

      std::vector<std::future<void>> futures;
      futures.reserve(num_threads);

      for (int thread_id = 0; thread_id < num_threads; ++thread_id)
      {
        int start_t = thread_id * chunk_size;
        int end_t = (thread_id == num_threads - 1) ? horizon
                                                   : (thread_id + 1) * chunk_size;

        if (start_t >= horizon)
          break;

        futures.push_back(
            std::async(std::launch::async, [this, &context, &constraint_set,
                                            start_t, end_t]()
                       {
            // Process a chunk of time steps
            const CDDPOptions &options = context.getOptions();
            for (int t = start_t; t < end_t; ++t) {
              const Eigen::VectorXd &x = context.X_[t];
              const Eigen::VectorXd &u = context.U_[t];

              for (const auto &constraint_pair : constraint_set) {
                const std::string &constraint_name = constraint_pair.first;
                G_x_[constraint_name][t] =
                    constraint_pair.second->getStateJacobian(x, u);
                G_u_[constraint_name][t] =
                    constraint_pair.second->getControlJacobian(x, u);
                
                // Compute constraint hessians if not using iLQR
                if (!options.use_ilqr)
                {
                  G_xx_[constraint_name][t] = constraint_pair.second->getStateHessian(x, u);
                  G_uu_[constraint_name][t] = constraint_pair.second->getControlHessian(x, u);
                  G_ux_[constraint_name][t] = constraint_pair.second->getCrossHessian(x, u);
                }
              }
            } }));
      }

      // Wait for all computations to complete
      for (auto &future : futures)
      {
        try
        {
          if (future.valid())
          {
            future.get();
          }
        }
        catch (const std::exception &e)
        {
          if (options.verbose)
          {
            std::cerr << "MSIPDDP: Constraint gradients computation thread failed: "
                      << e.what() << std::endl;
          }
          throw;
        }
      }
    }
  }

  bool MSIPDDPSolver::backwardPass(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int state_dim = context.getStateDim();
    const int control_dim = context.getControlDim();
    const int horizon = context.getHorizon();
    const double timestep = context.getTimestep();
    const auto &constraint_set = context.getConstraintSet();
    const int total_dual_dim = getTotalDualDim(context);

    // Pre-compute dynamics jacobians and hessians for all time steps
    precomputeDynamicsDerivatives(context);

    // Pre-compute constraint gradients for all time steps and constraints
    precomputeConstraintGradients(context);

    // Terminal cost and its derivatives
    Eigen::VectorXd V_x =
        context.getObjective().getFinalCostGradient(context.X_.back());
    Eigen::MatrixXd V_xx =
        context.getObjective().getFinalCostHessian(context.X_.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

    dV_ = Eigen::Vector2d::Zero();
    double inf_du = 0.0;     // dual infeasibility (optimality gap; Qu_err)
    double inf_pr = 0.0;     // primal infeasibility (constraint violation)
    double inf_comp = 0.0;   // complementary infeasibility
    double inf_defect = 0.0; // defect norm for multi-shooting
    double step_norm = 0.0;

    // If no constraints, use standard DDP recursion
    if (constraint_set.empty())
    {
      for (int t = horizon - 1; t >= 0; --t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        // MSIPDDP: Access costate variables and compute defect
        const Eigen::VectorXd &lambda = Lambda_[t];
        const Eigen::VectorXd &f = F_[t];
        Eigen::VectorXd &d = workspace_.d_vectors[t];
        d.setZero();
        if ((t + 1) < static_cast<int>(context.X_.size()))
        {
          d = f - context.X_[t + 1]; // defect: dynamics mismatch
        }

        // Use pre-computed dynamics Jacobians
        const Eigen::MatrixXd &Fx = F_x_[t];
        const Eigen::MatrixXd &Fu = F_u_[t];
        
        // Use pre-allocated workspace matrices
        Eigen::MatrixXd &A = workspace_.A_matrices[t];
        Eigen::MatrixXd &B = workspace_.B_matrices[t];
        A.noalias() = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
        B.noalias() = timestep * Fu;

        // Cost & derivatives
        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] =
            context.getObjective().getRunningCostHessians(x, u, t);

        // Q expansions from cost - use pre-allocated workspace
        Eigen::VectorXd &Q_x = workspace_.Q_x_vectors[t];
        Eigen::VectorXd &Q_u = workspace_.Q_u_vectors[t];
        Eigen::MatrixXd &Q_xx = workspace_.Q_xx_matrices[t];
        Eigen::MatrixXd &Q_ux = workspace_.Q_ux_matrices[t];
        Eigen::MatrixXd &Q_uu = workspace_.Q_uu_matrices[t];
        
        Q_x.noalias() = l_x + A.transpose() * (V_x + V_xx * d);
        Q_u.noalias() = l_u + B.transpose() * (V_x + V_xx * d);
        Q_xx.noalias() = l_xx + A.transpose() * V_xx * A;
        Q_ux.noalias() = l_ux + B.transpose() * V_xx * A;
        Q_uu.noalias() = l_uu + B.transpose() * V_xx * B;

        // Add state hessian term if not using iLQR
        if (!options.use_ilqr)
        {
          // Use pre-computed hessians
          const auto &Fxx = F_xx_[t];
          const auto &Fuu = F_uu_[t];
          const auto &Fux = F_ux_[t];

          for (int i = 0; i < state_dim; ++i)
          {
            Q_xx += timestep * lambda(i) * Fxx[i];
            Q_ux += timestep * lambda(i) * Fux[i];
            Q_uu += timestep * lambda(i) * Fuu[i];
          }
        }

        // Apply standard DDP regularization
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose()); // symmetrize NOTE: This is critical
        Q_uu.diagonal().array() += context.regularization_;

        // Use cached LDLT solver or compute new factorization
        if (!workspace_.ldlt_valid[t] || workspace_.ldlt_solvers[t].matrixLDLT().rows() != control_dim) {
          workspace_.ldlt_solvers[t].compute(Q_uu);
          workspace_.ldlt_valid[t] = true;
        } else {
          // Reuse existing factorization structure
          workspace_.ldlt_solvers[t].compute(Q_uu);
        }
        
        if (workspace_.ldlt_solvers[t].info() != Eigen::Success)
        {
          if (options.debug)
          {
            std::cerr << "MSIPDDP: Backward pass failed at time " << t << " (Q_uu not positive definite)" << std::endl;
          }
          workspace_.ldlt_valid[t] = false;
          return false;
        }

        Eigen::VectorXd k_u = -workspace_.ldlt_solvers[t].solve(Q_u);
        Eigen::MatrixXd K_u = -workspace_.ldlt_solvers[t].solve(Q_ux);
        k_u_[t] = k_u;
        K_u_[t] = K_u;

        // MSIPDDP: Compute costate gains for multi-shooting
        k_lambda_[t] = -lambda + V_x + V_xx * d;
        K_lambda_[t] = V_xx;
        K_lambda_[t] = 0.5 * (K_lambda_[t] + K_lambda_[t].transpose()); // Symmetrize

        // Update value function
        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u +
              K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u +
               K_u.transpose() * Q_uu * K_u;
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

        // Accumulate cost improvement
        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        // Error tracking
        inf_du = std::max(inf_du, Q_u.lpNorm<Eigen::Infinity>());
        step_norm = std::max(step_norm, k_u.lpNorm<Eigen::Infinity>());
        inf_defect = std::max(inf_defect, d.lpNorm<Eigen::Infinity>());
      }

      // Update termination metrics
      context.inf_du_ = inf_du;
      context.step_norm_ = step_norm;
      context.inf_pr_ = inf_defect; // Include defect violations as primal infeasibility
      context.inf_comp_ = 0.0;      // No complementary constraints

      if (options.debug)
      {
        std::cout << "[MSIPDDP Backward] inf_du: " << std::scientific << std::setprecision(2)
                  << inf_du << " inf_defect: " << inf_defect << " ||d||: " << context.step_norm_ << " dV: " << dV_.transpose() << std::endl;
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

        // MSIPDDP: Access costate variables and compute defect
        const Eigen::VectorXd &lambda = Lambda_[t];
        const Eigen::VectorXd &f = F_[t];
        Eigen::VectorXd &d = workspace_.d_vectors[t];
        d.setZero();
        if ((t + 1) < static_cast<int>(context.X_.size()))
        {
          d = f - context.X_[t + 1]; // defect: dynamics mismatch
        }

        // Use pre-computed dynamics Jacobians
        const Eigen::MatrixXd &Fx = F_x_[t];
        const Eigen::MatrixXd &Fu = F_u_[t];
        
        // Use pre-allocated workspace matrices
        Eigen::MatrixXd &A = workspace_.A_matrices[t];
        Eigen::MatrixXd &B = workspace_.B_matrices[t];
        A.noalias() = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
        B.noalias() = timestep * Fu;

        // Use pre-allocated workspace for constraint variables
        Eigen::VectorXd &y = workspace_.y_combined;
        Eigen::VectorXd &s = workspace_.s_combined;
        Eigen::VectorXd &g = workspace_.g_combined;
        Eigen::MatrixXd &Q_yu = workspace_.Q_yu_combined;
        Eigen::MatrixXd &Q_yx = workspace_.Q_yx_combined;

        int offset = 0;
        for (const auto &constraint_pair : constraint_set)
        {
          const std::string &constraint_name = constraint_pair.first;
          int dual_dim = constraint_pair.second->getDualDim();

          const Eigen::VectorXd &y_vec = Y_[constraint_name][t];
          const Eigen::VectorXd &s_vec = S_[constraint_name][t];
          const Eigen::VectorXd &g_vec = G_[constraint_name][t];
          const Eigen::MatrixXd &g_x = G_x_[constraint_name][t];
          const Eigen::MatrixXd &g_u = G_u_[constraint_name][t];

          y.segment(offset, dual_dim) = y_vec;
          s.segment(offset, dual_dim) = s_vec;
          g.segment(offset, dual_dim) = g_vec;
          Q_yx.block(offset, 0, dual_dim, state_dim) = g_x;
          Q_yu.block(offset, 0, dual_dim, control_dim) = g_u;

          offset += dual_dim;
        }

        // Cost & derivatives
        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] =
            context.getObjective().getRunningCostHessians(x, u, t);

        // Q expansions from cost - use pre-allocated workspace
        Eigen::VectorXd &Q_x = workspace_.Q_x_vectors[t];
        Eigen::VectorXd &Q_u = workspace_.Q_u_vectors[t];
        Eigen::MatrixXd &Q_xx = workspace_.Q_xx_matrices[t];
        Eigen::MatrixXd &Q_ux = workspace_.Q_ux_matrices[t];
        Eigen::MatrixXd &Q_uu = workspace_.Q_uu_matrices[t];
        
        Q_x.noalias() = l_x + Q_yx.transpose() * y + A.transpose() * (V_x + V_xx * d);
        Q_u.noalias() = l_u + Q_yu.transpose() * y + B.transpose() * (V_x + V_xx * d);
        Q_xx.noalias() = l_xx + A.transpose() * V_xx * A;
        Q_ux.noalias() = l_ux + B.transpose() * V_xx * A;
        Q_uu.noalias() = l_uu + B.transpose() * V_xx * B;

        // Add state hessian term if not using iLQR
        if (!options.use_ilqr)
        {
          // Use pre-computed hessians
          const auto &Fxx = F_xx_[t];
          const auto &Fuu = F_uu_[t];
          const auto &Fux = F_ux_[t];

          for (int i = 0; i < state_dim; ++i)
          {
            Q_xx += timestep * lambda(i) * Fxx[i];
            Q_ux += timestep * lambda(i) * Fux[i];
            Q_uu += timestep * lambda(i) * Fuu[i];
          }

          // Add constraint hessian terms
          int offset = 0;
          for (const auto &constraint_pair : constraint_set)
          {
            const std::string &constraint_name = constraint_pair.first;
            int dual_dim = constraint_pair.second->getDualDim();

            const auto &G_xx = G_xx_[constraint_name][t];
            const auto &G_uu = G_uu_[constraint_name][t];
            const auto &G_ux = G_ux_[constraint_name][t];

            for (int i = 0; i < dual_dim; ++i)
            {
              Q_xx += y(offset + i) * G_xx[i];
              Q_ux += y(offset + i) * G_ux[i];
              Q_uu += y(offset + i) * G_uu[i];
            }
            offset += dual_dim;
          }
        }

        // Optimize diagonal matrix operations
        Eigen::MatrixXd &YSinv = workspace_.YSinv;
        YSinv.setZero();
        for (int i = 0; i < total_dual_dim; ++i) {
          YSinv(i, i) = y(i) / s(i);
        }

        // Residuals
        Eigen::VectorXd primal_residual = g + s;                                  // primal infeasibility
        Eigen::VectorXd complementary_residual = y.cwiseProduct(s).array() - mu_; // complementary infeasibility
        Eigen::VectorXd rhat = y.cwiseProduct(primal_residual) - complementary_residual;

        // Apply standard DDP regularization
        Eigen::MatrixXd Q_uu_reg = Q_uu;
        Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize
        
        // Add constraint contribution
        Q_uu_reg.noalias() += Q_yu.transpose() * YSinv * Q_yu;
        
        // Apply standard DDP regularization
        Q_uu_reg.diagonal().array() += context.regularization_;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
        if (ldlt.info() != Eigen::Success)
        {
          if (options.debug)
          {
            std::cerr << "MSIPDDP: Backward pass failed at time " << t << " (Q_uu not positive definite)" << std::endl;
          }
          return false;
        }

        // Use pre-allocated workspace
        Eigen::MatrixXd &bigRHS = workspace_.bigRHS;
        // Compute S_inv * rhat efficiently
        Eigen::VectorXd S_inv_rhat(total_dual_dim);
        for (int i = 0; i < total_dual_dim; ++i) {
          S_inv_rhat(i) = rhat(i) / s(i);
        }
        bigRHS.col(0).noalias() = Q_u + Q_yu.transpose() * S_inv_rhat;
        // Compute M = Q_ux + Q_yu.transpose() * YSinv * Q_yx efficiently
        bigRHS.rightCols(state_dim).noalias() = Q_ux + Q_yu.transpose() * YSinv * Q_yx;

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

        // Compute gains for constraints efficiently
        Eigen::VectorXd k_y(total_dual_dim);
        Eigen::VectorXd temp = Q_yu * k_u;
        for (int i = 0; i < total_dual_dim; ++i) {
          k_y(i) = (rhat(i) + y(i) * temp(i)) / s(i);
        }
        Eigen::MatrixXd K_y = YSinv * (Q_yx + Q_yu * K_u);
        Eigen::VectorXd k_s = -primal_residual - temp;
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

        // MSIPDDP: Compute costate gains for multi-shooting
        k_lambda_[t] = -lambda + V_x + V_xx * d;
        K_lambda_[t] = V_xx;
        K_lambda_[t] = 0.5 * (K_lambda_[t] + K_lambda_[t].transpose()); // Symmetrize

        // Update Q expansions efficiently
        Q_u.noalias() += Q_yu.transpose() * S_inv_rhat;
        Q_x.noalias() += Q_yx.transpose() * S_inv_rhat;
        Q_xx.noalias() += Q_yx.transpose() * YSinv * Q_yx;
        Q_ux.noalias() += Q_yx.transpose() * YSinv * Q_yu;
        Q_uu.noalias() += Q_yu.transpose() * YSinv * Q_yu;

        // Update cost improvement
        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        // Update value function
        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u +
              K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u +
               K_u.transpose() * Q_uu * K_u;
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize NOTE: This is critical

        // Error tracking
        inf_du = std::max(inf_du, Q_u.lpNorm<Eigen::Infinity>());
        inf_pr = std::max(inf_pr, primal_residual.lpNorm<Eigen::Infinity>());
        inf_comp = std::max(inf_comp, complementary_residual.lpNorm<Eigen::Infinity>());
        inf_defect = std::max(inf_defect, d.lpNorm<Eigen::Infinity>());
        step_norm = std::max(step_norm, k_u.lpNorm<Eigen::Infinity>());
      }

      // Update termination metrics (properly separated)
      context.inf_pr_ = std::max(inf_pr, inf_defect); // Primal infeasibility (constraint + defect violations)
      context.inf_du_ = inf_du;                       // Dual infeasibility (optimality gap)
      context.inf_comp_ = inf_comp;                   // Complementary infeasibility
      context.step_norm_ = step_norm;

      if (options.debug)
      {
        std::cout << "[MSIPDDP Backward] inf_du: " << std::scientific << std::setprecision(2)
                  << inf_du << " inf_pr: " << inf_pr << " inf_defect: " << inf_defect << " inf_comp: " << inf_comp
                  << " ||d||: " << context.step_norm_ << " dV: " << dV_.transpose() << std::endl;
      }
      return true;
    }
  }

  ForwardPassResult MSIPDDPSolver::performForwardPass(CDDP &context)
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

        if (result.success &&
            result.merit_function < best_result.merit_function)
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
        futures.push_back(
            std::async(std::launch::async, [this, &context, alpha_pr]()
                       { return forwardPass(context, alpha_pr); }));
      }

      for (auto &future : futures)
      {
        try
        {
          if (future.valid())
          {
            ForwardPassResult result = future.get();
            if (result.success &&
                result.merit_function < best_result.merit_function)
            {
              best_result = result;
            }
          }
        }
        catch (const std::exception &e)
        {
          if (options.verbose)
          {
            std::cerr << "MSIPDDP: Forward pass thread failed: " << e.what()
                      << std::endl;
          }
        }
      }
    }

    return best_result;
  }

  ForwardPassResult MSIPDDPSolver::forwardPass(CDDP &context, double alpha)
  {
    const CDDPOptions &options = context.getOptions();
    const auto &constraint_set = context.getConstraintSet();

    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.merit_function = std::numeric_limits<double>::infinity();
    result.alpha_pr = alpha;

    const int horizon = context.getHorizon();
    const double tau =
        std::max(options.msipddp.barrier.min_fraction_to_boundary, 1.0 - mu_);

    // Initialize trajectories
    result.state_trajectory = context.X_;
    result.control_trajectory = context.U_;
    result.state_trajectory[0] = context.getInitialState();

    // Initialize trajectories for forward pass
    std::vector<Eigen::VectorXd> F_new = F_;
    std::vector<Eigen::VectorXd> Lambda_new = Lambda_;
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
        const Eigen::VectorXd &delta_x = workspace_.delta_x_vectors[t];
        workspace_.delta_x_vectors[t] = result.state_trajectory[t] - context.X_[t];
        result.control_trajectory[t] =
            context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x;

        // Update costate variables for multi-shooting
        if (t < static_cast<int>(Lambda_new.size()))
        {
          Lambda_new[t] = Lambda_[t] + alpha * k_lambda_[t] + K_lambda_[t] * delta_x;
        }

        // Determine if we're at a segment boundary for gap-closing
        bool is_segment_boundary = (ms_segment_length_ > 1) &&
                                   ((t + 1) % ms_segment_length_ == 0) &&
                                   (t + 1 < horizon);
        bool apply_gap_closing = is_segment_boundary;

        // Evaluate dynamics at current point
        F_new[t] = context.getSystem().getDiscreteDynamics(
            result.state_trajectory[t], result.control_trajectory[t],
            t * context.getTimestep());

        // Apply multi-shooting strategy based on options
        if (apply_gap_closing)
        {
          // Multi-shooting gap-closing at segment boundaries
          if (options.msipddp.rollout_type == "nonlinear")
          {
            // Nonlinear rollout: Gap-closing with defect correction
            result.state_trajectory[t + 1] = context.X_[t + 1] +
                                             (F_new[t] - F_[t]) +
                                             alpha * (F_[t] - context.X_[t + 1]);
          }
          else if (options.msipddp.rollout_type == "hybrid")
          {
            // Hybrid rollout: Linear approximation + defect correction
            const auto [Fx, Fu] = context.getSystem().getJacobians(
                context.X_[t], context.U_[t], t * context.getTimestep());
            const double timestep = context.getTimestep();
            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(context.getStateDim(), context.getStateDim()) + timestep * Fx;
            Eigen::MatrixXd B = timestep * Fu;

            result.state_trajectory[t + 1] = context.X_[t + 1] +
                                             (A + B * K_u_[t]) * delta_x +
                                             alpha * (B * k_u_[t] + F_[t] - context.X_[t + 1]);
          }
          else
          {
            // Default: standard dynamics propagation
            result.state_trajectory[t + 1] = F_new[t];
          }
        }
        else
        {
          // Normal propagation (not at segment boundary)
          result.state_trajectory[t + 1] = F_new[t];
        }

        // Accumulate stage cost
        cost_new += context.getObjective().running_cost(
            result.state_trajectory[t], result.control_trajectory[t], t);
      }
      cost_new +=
          context.getObjective().terminal_cost(result.state_trajectory.back());

      double dJ = context.cost_ - cost_new;
      double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
      double reduction_ratio =
          expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

      result.success = reduction_ratio > 1e-6;
      result.cost = cost_new;
      result.merit_function = cost_new;
      result.constraint_violation = 0.0;
      result.alpha_du = 1.0; // No dual variables for unconstrained case
      result.dynamics_trajectory = F_new;
      result.costate_trajectory = Lambda_new;

      return result;
    }

    // Constrained forward pass
    double alpha_s = alpha;

    // Step 1: Update slack variables and state/control with alpha_s
    bool s_trajectory_feasible = true;
    for (int t = 0; t < horizon; ++t)
    {
      const Eigen::VectorXd &delta_x = workspace_.delta_x_vectors[t];
      workspace_.delta_x_vectors[t] = result.state_trajectory[t] - context.X_[t];

      // Update slack variables first
      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &constraint_name = constraint_pair.first;
        int dual_dim = constraint_pair.second->getDualDim();
        const Eigen::VectorXd &s_old = S_[constraint_name][t];

        Eigen::VectorXd s_new = s_old + alpha_s * k_s_[constraint_name][t] +
                                K_s_[constraint_name][t] * delta_x;
        Eigen::VectorXd s_min = (1.0 - tau) * s_old;

        for (int i = 0; i < dual_dim; ++i)
        {
          if (s_new[i] < s_min[i])
          {
            s_trajectory_feasible = false;
            break;
          }
        }
        if (!s_trajectory_feasible)
          break;

        S_new[constraint_name][t] = s_new;
      }
      if (!s_trajectory_feasible)
        break;

      // Update control
      result.control_trajectory[t] =
          context.U_[t] + alpha_s * k_u_[t] + K_u_[t] * delta_x;

      // Determine if we're at a segment boundary for gap-closing
      bool is_segment_boundary = (ms_segment_length_ > 1) &&
                                 ((t + 1) % ms_segment_length_ == 0) &&
                                 (t + 1 < horizon);
      bool apply_gap_closing = is_segment_boundary;

      // Evaluate dynamics at current point
      F_new[t] = context.getSystem().getDiscreteDynamics(
          result.state_trajectory[t], result.control_trajectory[t],
          t * context.getTimestep());

      // Apply multi-shooting strategy based on options
      if (apply_gap_closing)
      {
        // Multi-shooting gap-closing at segment boundaries
        if (options.msipddp.rollout_type == "nonlinear")
        {
          // Nonlinear rollout: Gap-closing with defect correction
          result.state_trajectory[t + 1] = context.X_[t + 1] +
                                           (F_new[t] - F_[t]) +
                                           alpha_s * (F_[t] - context.X_[t + 1]);
        }
        else if (options.msipddp.rollout_type == "hybrid")
        {
          // Hybrid rollout: Linear approximation + defect correction
          const auto [Fx, Fu] = context.getSystem().getJacobians(
              context.X_[t], context.U_[t], t * context.getTimestep());
          const double timestep = context.getTimestep();
          Eigen::MatrixXd A = Eigen::MatrixXd::Identity(context.getStateDim(), context.getStateDim()) + timestep * Fx;
          Eigen::MatrixXd B = timestep * Fu;

          result.state_trajectory[t + 1] = context.X_[t + 1] +
                                           (A + B * K_u_[t]) * delta_x +
                                           alpha_s * (B * k_u_[t] + F_[t] - context.X_[t + 1]);
        }
        else
        {
          // Default: standard dynamics propagation
          result.state_trajectory[t + 1] = F_new[t];
        }
      }
      else
      {
        // Normal propagation (not at segment boundary)
        result.state_trajectory[t + 1] = F_new[t];
      }
    }

    if (!s_trajectory_feasible)
    {
      return result; // Failed slack update
    }

    // Step 2: Separate line search for dual variables
    bool suitable_alpha_y_found = false;
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_trial;

    for (double alpha_y_candidate : context.alphas_)
    {
      bool current_alpha_y_globally_feasible = true;
      Y_trial = Y_;

      for (int t = 0; t < horizon; ++t)
      {
        const Eigen::VectorXd &delta_x = workspace_.delta_x_vectors[t];

        for (const auto &constraint_pair : constraint_set)
        {
          const std::string &constraint_name = constraint_pair.first;
          int dual_dim = constraint_pair.second->getDualDim();
          const Eigen::VectorXd &y_old = Y_[constraint_name][t];

          Eigen::VectorXd y_new = y_old +
                                  alpha_y_candidate * k_y_[constraint_name][t] +
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
          if (!current_alpha_y_globally_feasible)
            break;

          Y_trial[constraint_name][t] = y_new;
        }

        // Update costate variables for multi-shooting
        if (t < static_cast<int>(Lambda_new.size()))
        {
          Lambda_new[t] = Lambda_[t] + alpha_s * k_lambda_[t] + K_lambda_[t] * delta_x;
        }

        if (!current_alpha_y_globally_feasible)
          break;
      }

      if (current_alpha_y_globally_feasible)
      {
        suitable_alpha_y_found = true;
        Y_new = Y_trial;
        result.alpha_du = alpha_y_candidate; // Store the dual step size
        break;
      }
    }

    if (!suitable_alpha_y_found)
    {

      return result; // Failed dual variable update
    }

    // Cost computation and filter line-search
    for (int t = 0; t < horizon; ++t)
    {
      cost_new += context.getObjective().running_cost(
          result.state_trajectory[t], result.control_trajectory[t], t);

      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &constraint_name = constraint_pair.first;
        G_new[constraint_name][t] =
            constraint_pair.second->evaluate(result.state_trajectory[t],
                                             result.control_trajectory[t]) -
            constraint_pair.second->getUpperBound();

        const Eigen::VectorXd &s_vec = S_new[constraint_name][t];
        merit_function_new -= mu_ * s_vec.array().log().sum();

        // Primal infeasibility: g + s
        Eigen::VectorXd primal_residual = G_new[constraint_name][t] + s_vec;
        constraint_violation_new += primal_residual.lpNorm<1>();
      }

      // Defect infeasibility: d = f - x_{k+1}
      Eigen::VectorXd defect_residual = F_new[t] - result.state_trajectory[t + 1];
      constraint_violation_new += defect_residual.lpNorm<1>();
    }

    cost_new +=
        context.getObjective().terminal_cost(result.state_trajectory.back());
    merit_function_new += cost_new;

    // Enhanced filter acceptance logic using new methods
    double expected_improvement = alpha * dV_(0);
    bool filter_acceptance = isFilterAcceptable(merit_function_new, constraint_violation_new,
                                               options.filter, expected_improvement);

    if (filter_acceptance)
    {
      result.success = true;
      result.cost = cost_new;
      result.merit_function = merit_function_new;
      result.constraint_violation = constraint_violation_new;
      result.dual_trajectory = Y_new;
      result.slack_trajectory = S_new;
      result.constraint_eval_trajectory = G_new;
      result.dynamics_trajectory = F_new;
      result.costate_trajectory = Lambda_new;
    }

    return result;
  }

  void MSIPDDPSolver::printIteration(int iter, double objective, double inf_pr,
                                     double inf_du, double inf_comp, double mu,
                                     double step_norm, double regularization,
                                     double alpha_du, double alpha_pr) const
  {
    if (iter == 0)
    {
      std::cout << std::setw(4) << "iter" << " " << std::setw(12) << "objective"
                << " " << std::setw(9) << "inf_pr" << " " << std::setw(9)
                << "inf_du" << " " << std::setw(9) << "inf_comp" << " "
                << std::setw(7) << "lg(mu)" << " " << std::setw(9) << "||d||"
                << " " << std::setw(7) << "lg(rg)"
                << " " << std::setw(9) << "alpha_du" << " " << std::setw(9)
                << "alpha_pr" << std::endl;
    }

    // Format numbers with appropriate precision
    std::cout << std::setw(4) << iter << " ";

    // Objective value
    std::cout << std::setw(12) << std::scientific << std::setprecision(6)
              << objective << " ";

    // Primal infeasibility (constraint violation)
    std::cout << std::setw(9) << std::scientific << std::setprecision(2) << inf_pr
              << " ";

    // Dual infeasibility (optimality gap)
    std::cout << std::setw(9) << std::scientific << std::setprecision(2) << inf_du
              << " ";

    // Complementary infeasibility
    std::cout << std::setw(9) << std::scientific << std::setprecision(2)
              << inf_comp << " ";

    // Log of barrier parameter
    if (mu > 0.0)
    {
      std::cout << std::setw(7) << std::fixed << std::setprecision(1)
                << std::log10(mu) << " ";
    }
    else
    {
      std::cout << std::setw(7) << "-inf" << " ";
    }

    // Step norm
    std::cout << std::setw(9) << std::scientific << std::setprecision(2)
              << step_norm << " ";

    // Log of regularization
    if (regularization > 0.0)
    {
      std::cout << std::setw(7) << std::fixed << std::setprecision(1)
                << std::log10(regularization) << " ";
    }
    else
    {
      std::cout << std::setw(7) << "-" << " ";
    }

    // Dual step length
    std::cout << std::setw(9) << std::fixed << std::setprecision(6) << alpha_du
              << " ";

    // Primal step length
    std::cout << std::setw(9) << std::fixed << std::setprecision(6) << alpha_pr;

    std::cout << std::endl;
  }

  void MSIPDDPSolver::printSolutionSummary(const CDDPSolution &solution) const
  {
    std::cout << "\n========================================\n";
    std::cout << "           MSIPDDP Solution Summary\n";
    std::cout << "========================================\n";

    auto iterations = std::any_cast<int>(solution.at("iterations_completed"));
    auto solve_time = std::any_cast<double>(solution.at("solve_time_ms"));
    auto final_cost = std::any_cast<double>(solution.at("final_objective"));
    auto status = std::any_cast<std::string>(solution.at("status_message"));
    auto final_mu =
        std::any_cast<double>(solution.at("final_barrier_parameter_mu"));

    std::cout << "Status: " << status << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Solve Time: " << std::setprecision(2) << solve_time << " ms\n";
    std::cout << "Final Cost: " << std::setprecision(6) << final_cost << "\n";
    std::cout << "Final Barrier μ: " << std::setprecision(2) << std::scientific
              << final_mu << "\n";
    std::cout << "========================================\n\n";
  }

  void MSIPDDPSolver::updateBarrierParameters(CDDP &context, bool forward_pass_success)
  {
    const CDDPOptions &options = context.getOptions();
    const auto &barrier_opts = options.msipddp.barrier;
    const auto &constraint_set = context.getConstraintSet();

    // Only update barrier parameters if we have constraints
    if (constraint_set.empty())
    {
      return; // No constraints case - no barrier update needed
    }

    // Compute termination metric from current infeasibility metrics with IPOPT-style scaling
    double scaled_inf_du = computeScaledDualInfeasibility(context);
    double termination_metric = std::max({scaled_inf_du, context.inf_pr_, context.inf_comp_});

    // Adaptive barrier parameter update strategy
    // Use different thresholds based on current progress
    double barrier_update_threshold;
    
    // If mu is already small, use a more relaxed threshold
    if (mu_ < 1e-5)
    {
      // For small mu, update if we've made any reasonable progress
      barrier_update_threshold = std::max(termination_metric * 10.0, mu_ * 100.0);
    }
    else
    {
      // Standard threshold for larger mu values
      barrier_update_threshold = std::max(barrier_opts.mu_update_factor * mu_, mu_ * 2.0);
    }

    // Also consider updating if forward pass succeeded but with very small cost change NOTE: Hand tuning
    bool slow_progress = forward_pass_success && context.alpha_pr_ > 0 &&
                        (termination_metric < 1e-3);  // Only when reasonably feasible

    if (termination_metric <= barrier_update_threshold || slow_progress)
    {
      // Adaptive barrier reduction strategy
      double reduction_factor = barrier_opts.mu_update_factor;

      if (mu_ > 1e-12)
      {
        double kkt_progress_ratio = termination_metric / mu_;

        // Very aggressive reduction for good KKT satisfaction
        if (kkt_progress_ratio < 0.01)
        {
          reduction_factor = barrier_opts.mu_update_factor * 0.1;
        }
        // Aggressive reduction if we're significantly satisfying KKT conditions
        else if (kkt_progress_ratio < 0.1)
        {
          reduction_factor = barrier_opts.mu_update_factor * 0.3;
        }
        // Moderate reduction if we're moderately satisfying KKT conditions
        else if (kkt_progress_ratio < 0.5)
        {
          reduction_factor = barrier_opts.mu_update_factor * 0.6;
        }
        // Standard reduction otherwise
      }

      // Update barrier parameter with bounds
      double new_mu_linear = reduction_factor * mu_;
      double new_mu_superlinear = std::pow(mu_, barrier_opts.mu_update_power);

      // Choose the more aggressive reduction when progress is slow
      if (slow_progress && mu_ > options.tolerance)
      {
        mu_ = std::min(new_mu_linear, new_mu_superlinear);
      }
      else
      {
        mu_ = std::max(options.tolerance / 100.0,
                       std::min(new_mu_linear, new_mu_superlinear));
      }

      // Reset filter when barrier parameter changes
      resetFilter(context);

      if (options.debug)
      {
        std::cout << "[MSIPDDP Barrier] Termination metric: " << std::scientific << std::setprecision(2)
                  << termination_metric << " (scaled inf_du: " << scaled_inf_du
                  << ", inf_pr: " << context.inf_pr_ << ", inf_comp: " << context.inf_comp_
                  << ") → μ: " << mu_ << std::endl;
      }
    }
  }

  void MSIPDDPSolver::updateIterationHistory(
      const CDDPOptions &options,
      const CDDP &context,
      std::vector<double> &history_objective,
      std::vector<double> &history_merit_function,
      std::vector<double> &history_step_length_primal,
      std::vector<double> &history_step_length_dual,
      std::vector<double> &history_dual_infeasibility,
      std::vector<double> &history_primal_infeasibility,
      std::vector<double> &history_complementary_infeasibility,
      std::vector<double> &history_barrier_mu,
      double alpha_du) const
  {
    if (options.return_iteration_info)
    {
      history_objective.push_back(context.cost_);
      history_merit_function.push_back(context.merit_function_);
      history_step_length_primal.push_back(context.alpha_pr_);
      history_step_length_dual.push_back(alpha_du);
      history_dual_infeasibility.push_back(context.inf_du_);
      history_primal_infeasibility.push_back(context.inf_pr_);
      history_complementary_infeasibility.push_back(context.inf_comp_);
      history_barrier_mu.push_back(mu_);
    }
  }

  bool MSIPDDPSolver::checkConvergence(
      const CDDPOptions &options,
      const CDDP &context,
      double dJ,
      int iter,
      std::string &termination_reason) const
  {
    // Compute IPOPT-style scaling factors
    double scaled_inf_du = computeScaledDualInfeasibility(context);
    double termination_metric = std::max({scaled_inf_du, context.inf_pr_, context.inf_comp_});

    if (termination_metric <= options.tolerance)
    {
      termination_reason = "OptimalSolutionFound";
      if (options.verbose)
      {
        std::cout << "MSIPDDP: Converged due to scaled optimality gap and constraint "
                     "violation (metric: "
                  << std::scientific << std::setprecision(2)
                  << termination_metric << ", scaled inf_du: " << scaled_inf_du << ")" << std::endl;
      }
      return true;
    }

    // For acceptable tolerance, also check if we're making minimal progress over several iterations
    if (std::abs(dJ) < options.acceptable_tolerance && iter > 10)
    {
      // Check if all infeasibility measures are reasonably small
      bool acceptable_infeasibility = (context.inf_pr_ < std::sqrt(options.acceptable_tolerance) &&
                                       context.inf_comp_ < std::sqrt(options.acceptable_tolerance));
      
      if (acceptable_infeasibility)
      {
        termination_reason = "AcceptableSolutionFound";
        if (options.verbose)
        {
          std::cout << "MSIPDDP: Converged due to small change in cost (dJ: "
                    << std::scientific << std::setprecision(2) << std::abs(dJ)
                    << ") with acceptable infeasibility" << std::endl;
        }
        return true;
      }
    }

    // Check step norm for early termination
    if (iter >= 1 &&
        context.step_norm_ < options.tolerance * 10.0 && // Small step norm
        context.inf_pr_ < 1e-4)                          // Reasonably feasible
    {
      termination_reason = "AcceptableSolutionFound";
      if (options.verbose)
      {
        std::cout << "MSIPDDP: Converged based on small step norm and feasibility"
                  << std::endl;
      }
      return true;
    }

    return false;
  }

  void MSIPDDPSolver::initializeConstraintStorage(CDDP &context)
  {
    const auto &constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();

    // Clear and initialize constraint storage
    G_.clear();
    G_x_.clear();
    G_u_.clear();
    Y_.clear();
    S_.clear();
    k_y_.clear();
    K_y_.clear();
    k_s_.clear();
    K_s_.clear();

    // Initialize storage for each constraint
    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      G_[constraint_name].resize(horizon);
      Y_[constraint_name].resize(horizon);
      S_[constraint_name].resize(horizon);
      k_y_[constraint_name].resize(horizon);
      K_y_[constraint_name].resize(horizon);
      k_s_[constraint_name].resize(horizon);
      K_s_[constraint_name].resize(horizon);
    }
  }

  double MSIPDDPSolver::computeMaxConstraintViolation(const CDDP &context) const
  {
    const auto &constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();
    double max_violation = 0.0;

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      auto g_it = G_.find(constraint_name);
      if (g_it != G_.end())
      {
        for (int t = 0; t < horizon; ++t)
        {
          const Eigen::VectorXd &g_vec = g_it->second[t];
          max_violation = std::max(max_violation, g_vec.maxCoeff());
        }
      }
    }
    return max_violation;
  }

  double MSIPDDPSolver::computeScaledDualInfeasibility(const CDDP &context) const
  {
    const auto &constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();
    const int control_dim = context.getControlDim();

    // If no constraints, return the unscaled dual infeasibility
    if (constraint_set.empty())
    {
      return context.inf_du_;
    }

    // IPOPT-style scaling: sd = max{smax, (||y||₁ + ||s||₁)/(m+n)}/smax
    const double smax = 100.0; // Standard IPOPT value

    // Compute total L1 norms of dual and slack variables
    double y_norm_l1 = 0.0;
    double s_norm_l1 = 0.0;
    int total_dual_dim = 0; // m: total number of constraints

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      auto y_it = Y_.find(constraint_name);
      auto s_it = S_.find(constraint_name);

      if (y_it != Y_.end() && s_it != S_.end())
      {
        for (int t = 0; t < horizon; ++t)
        {
          const Eigen::VectorXd &y_vec = y_it->second[t];
          const Eigen::VectorXd &s_vec = s_it->second[t];

          y_norm_l1 += y_vec.lpNorm<1>();
          s_norm_l1 += s_vec.lpNorm<1>();
          total_dual_dim += y_vec.size();
        }
      }
    }

    // m = total_dual_dim (number of constraints)
    // n = control_dim * horizon (number of control variables)
    int m = total_dual_dim;
    int n = control_dim * horizon;
    int m_plus_n = m + n;

    // Compute scaling factor: sd = max{smax, (||y||₁ + ||s||₁)/(m+n)}/smax
    double scaling_numerator = (m_plus_n > 0) ? (y_norm_l1 + s_norm_l1) / static_cast<double>(m_plus_n) : 0.0;
    double sd = std::max(smax, scaling_numerator) / smax;

    // Return scaled dual infeasibility
    return context.inf_du_ / sd;
  }

} // namespace cddp
