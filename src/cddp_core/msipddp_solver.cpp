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
      workspace_.A_matrices.resize(horizon);
      workspace_.B_matrices.resize(horizon);
      workspace_.Q_xx_matrices.resize(horizon);
      workspace_.Q_ux_matrices.resize(horizon);
      workspace_.Q_uu_matrices.resize(horizon);
      workspace_.Q_x_vectors.resize(horizon);
      workspace_.Q_u_vectors.resize(horizon);

      workspace_.ldlt_solvers.resize(horizon);
      workspace_.ldlt_valid.resize(horizon, false);

      workspace_.delta_x_vectors.resize(horizon + 1);
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
        if (options.verbose)
        {
          std::cout << "MSIPDDP: Warm start with provided trajectory" << std::endl;
        }

        initializeGains(horizon, control_dim, state_dim);

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

        if (constraint_set.empty())
        {
          mu_ = 1e-8;
        }
        else
        {
          evaluateTrajectoryWarmStart(context);
          double max_constraint_violation = computeMaxConstraintViolation(context);
          if (max_constraint_violation <= options.tolerance)
          {
            mu_ = options.tolerance * 0.01;
          }
          else if (max_constraint_violation <= 0.1)
          {
            mu_ = options.tolerance;
          }
          else
          {
            mu_ = options.msipddp.barrier.mu_initial * 0.1;
          }
        }

        context.regularization_ = options.regularization.initial_value;
        context.step_norm_ = 0.0;
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
    initializeGains(horizon, control_dim, state_dim);

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

  // === CDDPSolverBase hook implementations ===

  void MSIPDDPSolver::preIterationSetup(CDDP &context)
  {
    // evaluateTrajectory and resetFilter are already called in initialize().
    // No additional pre-iteration setup needed for MSIPDDP.
  }

  void MSIPDDPSolver::applyForwardPassResult(CDDP &context,
                                              const ForwardPassResult &result)
  {
    // Call base to update X_, U_, cost_, merit_function_, alpha_pr_, alpha_du_
    CDDPSolverBase::applyForwardPassResult(context, result);

    // Update IP-specific variables
    if (result.dual_trajectory)
      Y_ = *result.dual_trajectory;
    if (result.slack_trajectory)
      S_ = *result.slack_trajectory;
    if (result.constraint_eval_trajectory)
      G_ = *result.constraint_eval_trajectory;

    // Update MS-specific variables
    if (result.dynamics_trajectory)
      F_ = *result.dynamics_trajectory;
    if (result.costate_trajectory)
      Lambda_ = *result.costate_trajectory;

    // Update filter with accepted point
    acceptFilterEntry(result.merit_function, result.constraint_violation);
  }

  bool MSIPDDPSolver::checkConvergence(CDDP &context, double dJ, double dL,
                                        int iter, std::string &termination_reason)
  {
    const CDDPOptions &options = context.getOptions();

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

    if (std::abs(dJ) < options.acceptable_tolerance && iter > 10)
    {
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

    if (iter >= 1 &&
        context.step_norm_ < options.tolerance * 10.0 &&
        context.inf_pr_ < 1e-4)
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

  void MSIPDDPSolver::postIterationUpdate(CDDP &context, bool forward_pass_success)
  {
    updateBarrierParameters(context, forward_pass_success);
  }

  bool MSIPDDPSolver::handleForwardPassFailure(CDDP &context,
                                                std::string &termination_reason)
  {
    const CDDPOptions &options = context.getOptions();

    // Try filter restoration before increasing regularization
    bool restoration_performed = checkAndPerformFilterRestoration(context);

    if (!restoration_performed)
    {
      context.increaseRegularization();
      if (context.isRegularizationLimitReached())
      {
        termination_reason = "RegularizationLimitReached_NotConverged";
        if (options.verbose)
        {
          std::cerr << "MSIPDDP: Regularization limit reached" << std::endl;
        }
        return true; // break
      }
    }
    else if (options.debug)
    {
      std::cout << "MSIPDDP: Filter restoration performed, retrying forward pass" << std::endl;
    }

    return false; // continue
  }

  void MSIPDDPSolver::recordIterationHistory(const CDDP &context)
  {
    CDDPSolverBase::recordIterationHistory(context);
    history_.barrier_mu.push_back(mu_);
  }

  void MSIPDDPSolver::populateSolverSpecificSolution(CDDPSolution &solution,
                                                      const CDDP &context)
  {
    solution["final_barrier_parameter_mu"] = mu_;
    solution["final_primal_infeasibility"] = context.inf_pr_;
    solution["final_dual_infeasibility"] = context.inf_du_;
    solution["final_complementary_infeasibility"] = context.inf_comp_;
  }

  void MSIPDDPSolver::printIteration(int iter, const CDDP &context) const
  {
    printIterationLegacy(iter, context.cost_, context.inf_pr_, context.inf_du_,
                         context.inf_comp_, mu_, context.step_norm_,
                         context.regularization_, context.alpha_du_,
                         context.alpha_pr_);
  }

  // === Private methods ===

  void MSIPDDPSolver::evaluateTrajectory(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    double cost = 0.0;

    context.X_[0] = context.getInitialState();

    for (int t = 0; t < horizon; ++t)
    {
      const Eigen::VectorXd &x = context.X_[t];
      const Eigen::VectorXd &u = context.U_[t];

      cost += context.getObjective().running_cost(x, u, t);

      const auto &constraint_set = context.getConstraintSet();
      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &constraint_name = constraint_pair.first;
        Eigen::VectorXd g_val = constraint_pair.second->evaluate(x, u) -
                                constraint_pair.second->getUpperBound();
        G_[constraint_name][t] = g_val;
      }

      F_[t] = context.getSystem().getDiscreteDynamics(x, u, t * context.getTimestep());
      context.X_[t + 1] = F_[t];
    }

    cost += context.getObjective().terminal_cost(context.X_.back());
    context.cost_ = cost;
  }

  void MSIPDDPSolver::evaluateTrajectoryWarmStart(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    double cost = 0.0;

    const auto &constraint_set = context.getConstraintSet();
    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      G_[constraint_name].resize(horizon);
    }

    for (int t = 0; t < horizon; ++t)
    {
      const Eigen::VectorXd &x = context.X_[t];
      const Eigen::VectorXd &u = context.U_[t];

      cost += context.getObjective().running_cost(x, u, t);

      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &constraint_name = constraint_pair.first;
        Eigen::VectorXd g_val = constraint_pair.second->evaluate(x, u) -
                                constraint_pair.second->getUpperBound();
        G_[constraint_name][t] = g_val;
      }

      F_[t] = context.getSystem().getDiscreteDynamics(x, u, t * context.getTimestep());

      if (options.msipddp.use_controlled_rollout)
      {
        context.X_[t + 1] = F_[t];
      }
    }

    cost += context.getObjective().terminal_cost(context.X_.back());
    context.cost_ = cost;
  }

  void MSIPDDPSolver::initializeDualSlackCostateVariablesWarmStart(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const auto &constraint_set = context.getConstraintSet();

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

    k_y_.clear();
    K_y_.clear();
    k_s_.clear();
    K_s_.clear();

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      int dual_dim = constraint_pair.second->getDualDim();

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
        const Eigen::VectorXd &g_val = G_[constraint_name][t];

        bool need_reinit = false;
        Eigen::VectorXd y_current, s_current;

        if (has_existing_dual_slack)
        {
          y_current = Y_[constraint_name][t];
          s_current = S_[constraint_name][t];

          if (y_current.size() != dual_dim || s_current.size() != dual_dim)
          {
            need_reinit = true;
          }
          else
          {
            for (int i = 0; i < dual_dim; ++i)
            {
              if (y_current(i) <= 1e-12 || s_current(i) <= 1e-12)
              {
                need_reinit = true;
                break;
              }

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
          Eigen::VectorXd s_init = Eigen::VectorXd::Zero(dual_dim);
          Eigen::VectorXd y_init = Eigen::VectorXd::Zero(dual_dim);

          for (int i = 0; i < dual_dim; ++i)
          {
            s_init(i) = std::max(options.msipddp.slack_var_init_scale, -g_val(i));

            if (s_init(i) < 1e-12)
            {
              y_init(i) = mu_ / 1e-12;
            }
            else
            {
              y_init(i) = mu_ / s_init(i);
            }
            y_init(i) = std::max(
                options.msipddp.dual_var_init_scale * 0.01,
                std::min(y_init(i), options.msipddp.dual_var_init_scale * 100.0));
          }
          Y_[constraint_name][t] = y_init;
          S_[constraint_name][t] = s_init;
        }

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
      for (int t = 0; t < horizon; ++t)
      {
        Lambda_[t] = options.msipddp.costate_var_init_scale * Eigen::VectorXd::Ones(context.getStateDim());
        k_lambda_[t] = Eigen::VectorXd::Zero(context.getStateDim());
        K_lambda_[t] = Eigen::MatrixXd::Zero(context.getStateDim(), context.getStateDim());
      }
    }
    else
    {
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
        Eigen::VectorXd g_val =
            constraint_pair.second->evaluate(context.X_[t], context.U_[t]) -
            constraint_pair.second->getUpperBound();
        G_[constraint_name][t] = g_val;

        Eigen::VectorXd s_init = Eigen::VectorXd::Zero(dual_dim);
        Eigen::VectorXd y_init = Eigen::VectorXd::Zero(dual_dim);

        for (int i = 0; i < dual_dim; ++i)
        {
          s_init(i) = std::max(options.msipddp.slack_var_init_scale, -g_val(i));

          if (s_init(i) < 1e-12)
          {
            y_init(i) = mu_ / 1e-12;
          }
          else
          {
            y_init(i) = mu_ / s_init(i);
          }
          y_init(i) = std::max(
              options.msipddp.dual_var_init_scale * 0.01,
              std::min(y_init(i), options.msipddp.dual_var_init_scale * 100.0));
        }
        Y_[constraint_name][t] = y_init;
        S_[constraint_name][t] = s_init;

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
      Lambda_[t] = options.msipddp.costate_var_init_scale * Eigen::VectorXd::Ones(context.getStateDim());
      k_lambda_[t] = Eigen::VectorXd::Zero(context.getStateDim());
      K_lambda_[t] = Eigen::MatrixXd::Zero(context.getStateDim(), context.getStateDim());
    }

    context.cost_ = context.getObjective().evaluate(context.X_, context.U_);
  }

  void MSIPDDPSolver::resetBarrierFilter(CDDP &context)
  {
    double merit_function = context.cost_;
    double inf_pr = 0.0;
    double filter_constraint_violation = 0.0;
    double inf_du = 0.0;
    double inf_comp = 0.0;
    double inf_defect = 0.0;

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

          merit_function -= mu_ * s_vec.array().log().sum();

          Eigen::VectorXd primal_residual = g_vec + s_vec;
          inf_pr = std::max(inf_pr, primal_residual.lpNorm<Eigen::Infinity>());
          filter_constraint_violation += primal_residual.lpNorm<1>();

          Eigen::VectorXd complementary_residual = y_vec.cwiseProduct(s_vec).array() - mu_;
          inf_comp = std::max(inf_comp, complementary_residual.lpNorm<Eigen::Infinity>());
        }

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
      inf_pr = 0.0;
      filter_constraint_violation = 0.0;
      inf_du = 0.0;
      inf_comp = 0.0;
    }

    context.inf_pr_ = std::max(inf_pr, inf_defect);
    context.merit_function_ = merit_function;
    context.inf_comp_ = inf_comp;

    filter_.clear();
    filter_.push_back(FilterPoint(merit_function, filter_constraint_violation));
  }

  void MSIPDDPSolver::resetFilter(CDDP &context) { resetBarrierFilter(context); }

  bool MSIPDDPSolver::acceptFilterEntry(double merit_function, double constraint_violation)
  {
    FilterPoint candidate(merit_function, constraint_violation);

    for (const auto &filter_point : filter_)
    {
      if (filter_point.dominates(candidate))
      {
        return false;
      }
    }

    filter_.erase(
        std::remove_if(filter_.begin(), filter_.end(),
                      [&candidate](const FilterPoint &point) {
                        return candidate.dominates(point);
                      }),
        filter_.end());

    filter_.push_back(candidate);

    return true;
  }

  bool MSIPDDPSolver::isFilterAcceptable(double merit_function, double constraint_violation,
                                        const SolverSpecificFilterOptions &options,
                                        double expected_improvement) const
  {
    if (filter_.empty())
    {
      return true;
    }

    FilterPoint candidate(merit_function, constraint_violation);
    for (const auto &filter_point : filter_)
    {
      if (filter_point.dominates(candidate))
      {
        return false;
      }
    }

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

    bool violation_improvement = constraint_violation < best_violation * (1.0 - options.violation_acceptance_threshold);
    bool merit_improvement = merit_function < best_merit - options.merit_acceptance_threshold * constraint_violation;

    if (constraint_violation < options.min_violation_for_armijo_check && expected_improvement < 0)
    {
      return merit_function < best_merit + options.armijo_constant * expected_improvement;
    }

    if (constraint_violation < 1e-6 && merit_function <= best_merit * (1.0 + 1e-8))
    {
      return true;
    }

    return violation_improvement || merit_improvement;
  }

  bool MSIPDDPSolver::checkAndPerformFilterRestoration(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();

    bool needs_restoration = (filter_.size() > 5);

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

    F_x_.resize(horizon);
    F_u_.resize(horizon);
    F_xx_.resize(horizon);
    F_uu_.resize(horizon);
    F_ux_.resize(horizon);

    const int MIN_HORIZON_FOR_PARALLEL = 50;
    const bool use_parallel = options.enable_parallel && horizon >= MIN_HORIZON_FOR_PARALLEL;

    if (!use_parallel)
    {
      for (int t = 0; t < horizon; ++t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        const auto [Fx, Fu] =
            context.getSystem().getJacobians(x, u, t * timestep);
        F_x_[t] = Fx;
        F_u_[t] = Fu;

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
          F_xx_[t] = std::vector<Eigen::MatrixXd>();
          F_uu_[t] = std::vector<Eigen::MatrixXd>();
          F_ux_[t] = std::vector<Eigen::MatrixXd>();
        }
      }
    }
    else
    {
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
            for (int t = start_t; t < end_t; ++t) {
              const Eigen::VectorXd &x = context.X_[t];
              const Eigen::VectorXd &u = context.U_[t];

              const auto [Fx, Fu] =
                  context.getSystem().getJacobians(x, u, t * timestep);
              F_x_[t] = Fx;
              F_u_[t] = Fu;

              if (!options.use_ilqr) {
                const auto hessians =
                    context.getSystem().getHessians(x, u, t * timestep);
                F_xx_[t] = std::get<0>(hessians);
                F_uu_[t] = std::get<1>(hessians);
                F_ux_[t] = std::get<2>(hessians);
              } else {
                F_xx_[t] = std::vector<Eigen::MatrixXd>();
                F_uu_[t] = std::vector<Eigen::MatrixXd>();
                F_ux_[t] = std::vector<Eigen::MatrixXd>();
              }
            } }));
      }

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

    G_x_.clear();
    G_u_.clear();
    G_xx_.clear();
    G_uu_.clear();
    G_ux_.clear();

    if (constraint_set.empty())
    {
      return;
    }

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      if (G_x_.find(constraint_name) == G_x_.end() || G_x_[constraint_name].size() != horizon) {
        G_x_[constraint_name].resize(horizon);
        G_u_[constraint_name].resize(horizon);
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

    const int MIN_HORIZON_FOR_PARALLEL = 50;
    const bool use_parallel =
        options.enable_parallel && horizon >= MIN_HORIZON_FOR_PARALLEL;

    if (!use_parallel)
    {
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

                if (!options.use_ilqr)
                {
                  G_xx_[constraint_name][t] = constraint_pair.second->getStateHessian(x, u);
                  G_uu_[constraint_name][t] = constraint_pair.second->getControlHessian(x, u);
                  G_ux_[constraint_name][t] = constraint_pair.second->getCrossHessian(x, u);
                }
              }
            } }));
      }

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

    precomputeDynamicsDerivatives(context);
    precomputeConstraintGradients(context);

    Eigen::VectorXd V_x =
        context.getObjective().getFinalCostGradient(context.X_.back());
    Eigen::MatrixXd V_xx =
        context.getObjective().getFinalCostHessian(context.X_.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose());

    dV_ = Eigen::Vector2d::Zero();
    double inf_du = 0.0;
    double inf_pr = 0.0;
    double inf_comp = 0.0;
    double inf_defect = 0.0;
    double step_norm = 0.0;

    if (constraint_set.empty())
    {
      for (int t = horizon - 1; t >= 0; --t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        const Eigen::VectorXd &lambda = Lambda_[t];
        const Eigen::VectorXd &f = F_[t];
        Eigen::VectorXd &d = workspace_.d_vectors[t];
        d.setZero();
        if ((t + 1) < static_cast<int>(context.X_.size()))
        {
          d = f - context.X_[t + 1];
        }

        const Eigen::MatrixXd &Fx = F_x_[t];
        const Eigen::MatrixXd &Fu = F_u_[t];

        Eigen::MatrixXd &A = workspace_.A_matrices[t];
        Eigen::MatrixXd &B = workspace_.B_matrices[t];
        A.noalias() = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
        B.noalias() = timestep * Fu;

        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] =
            context.getObjective().getRunningCostHessians(x, u, t);

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

        if (!options.use_ilqr)
        {
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

        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());
        Q_uu.diagonal().array() += context.regularization_;

        bool need_recompute = !workspace_.ldlt_valid[t] ||
                             (workspace_.ldlt_valid[t] && workspace_.ldlt_solvers[t].matrixLDLT().rows() != control_dim);

        if (need_recompute) {
          workspace_.ldlt_solvers[t].compute(Q_uu);
          workspace_.ldlt_valid[t] = true;
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

        k_lambda_[t] = -lambda + V_x + V_xx * d;
        K_lambda_[t] = V_xx;
        K_lambda_[t] = 0.5 * (K_lambda_[t] + K_lambda_[t].transpose());

        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u +
              K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u +
               K_u.transpose() * Q_uu * K_u;
        V_xx = 0.5 * (V_xx + V_xx.transpose());

        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        inf_du = std::max(inf_du, Q_u.lpNorm<Eigen::Infinity>());
        step_norm = std::max(step_norm, k_u.lpNorm<Eigen::Infinity>());
        inf_defect = std::max(inf_defect, d.lpNorm<Eigen::Infinity>());
      }

      context.inf_du_ = inf_du;
      context.step_norm_ = step_norm;
      context.inf_pr_ = inf_defect;
      context.inf_comp_ = 0.0;

      if (options.debug)
      {
        std::cout << "[MSIPDDP Backward] inf_du: " << std::scientific << std::setprecision(2)
                  << inf_du << " inf_defect: " << inf_defect << " ||d||: " << context.step_norm_ << " dV: " << dV_.transpose() << std::endl;
      }
      return true;
    }
    else
    {
      for (int t = horizon - 1; t >= 0; --t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        const Eigen::VectorXd &lambda = Lambda_[t];
        const Eigen::VectorXd &f = F_[t];
        Eigen::VectorXd &d = workspace_.d_vectors[t];
        d.setZero();
        if ((t + 1) < static_cast<int>(context.X_.size()))
        {
          d = f - context.X_[t + 1];
        }

        const Eigen::MatrixXd &Fx = F_x_[t];
        const Eigen::MatrixXd &Fu = F_u_[t];

        Eigen::MatrixXd &A = workspace_.A_matrices[t];
        Eigen::MatrixXd &B = workspace_.B_matrices[t];
        A.noalias() = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
        B.noalias() = timestep * Fu;

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

        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] =
            context.getObjective().getRunningCostHessians(x, u, t);

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

        if (!options.use_ilqr)
        {
          const auto &Fxx = F_xx_[t];
          const auto &Fuu = F_uu_[t];
          const auto &Fux = F_ux_[t];

          for (int i = 0; i < state_dim; ++i)
          {
            Q_xx += timestep * lambda(i) * Fxx[i];
            Q_ux += timestep * lambda(i) * Fux[i];
            Q_uu += timestep * lambda(i) * Fuu[i];
          }

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

        Eigen::MatrixXd &YSinv = workspace_.YSinv;
        YSinv.setZero();
        for (int i = 0; i < total_dual_dim; ++i) {
          YSinv(i, i) = y(i) / s(i);
        }

        Eigen::VectorXd primal_residual = g + s;
        Eigen::VectorXd complementary_residual = y.cwiseProduct(s).array() - mu_;
        Eigen::VectorXd rhat = y.cwiseProduct(primal_residual) - complementary_residual;

        Eigen::MatrixXd Q_uu_reg = Q_uu;
        Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose());

        Q_uu_reg.noalias() += Q_yu.transpose() * YSinv * Q_yu;
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

        Eigen::MatrixXd &bigRHS = workspace_.bigRHS;
        Eigen::VectorXd S_inv_rhat(total_dual_dim);
        for (int i = 0; i < total_dual_dim; ++i) {
          S_inv_rhat(i) = rhat(i) / s(i);
        }
        bigRHS.col(0).noalias() = Q_u + Q_yu.transpose() * S_inv_rhat;
        bigRHS.rightCols(state_dim).noalias() = Q_ux + Q_yu.transpose() * YSinv * Q_yx;

        Eigen::MatrixXd kK = -ldlt.solve(bigRHS);

        Eigen::VectorXd k_u = kK.col(0);
        Eigen::MatrixXd K_u(control_dim, state_dim);
        for (int col = 0; col < state_dim; col++)
        {
          K_u.col(col) = kK.col(col + 1);
        }

        k_u_[t] = k_u;
        K_u_[t] = K_u;

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

        k_lambda_[t] = -lambda + V_x + V_xx * d;
        K_lambda_[t] = V_xx;
        K_lambda_[t] = 0.5 * (K_lambda_[t] + K_lambda_[t].transpose());

        Q_u.noalias() += Q_yu.transpose() * S_inv_rhat;
        Q_x.noalias() += Q_yx.transpose() * S_inv_rhat;
        Q_xx.noalias() += Q_yx.transpose() * YSinv * Q_yx;
        Q_ux.noalias() += Q_yx.transpose() * YSinv * Q_yu;
        Q_uu.noalias() += Q_yu.transpose() * YSinv * Q_yu;

        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u +
              K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u +
               K_u.transpose() * Q_uu * K_u;
        V_xx = 0.5 * (V_xx + V_xx.transpose());

        inf_du = std::max(inf_du, Q_u.lpNorm<Eigen::Infinity>());
        inf_pr = std::max(inf_pr, primal_residual.lpNorm<Eigen::Infinity>());
        inf_comp = std::max(inf_comp, complementary_residual.lpNorm<Eigen::Infinity>());
        inf_defect = std::max(inf_defect, d.lpNorm<Eigen::Infinity>());
        step_norm = std::max(step_norm, k_u.lpNorm<Eigen::Infinity>());
      }

      context.inf_pr_ = std::max(inf_pr, inf_defect);
      context.inf_du_ = inf_du;
      context.inf_comp_ = inf_comp;
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

    result.state_trajectory = context.X_;
    result.control_trajectory = context.U_;
    result.state_trajectory[0] = context.getInitialState();

    std::vector<Eigen::VectorXd> F_new = F_;
    std::vector<Eigen::VectorXd> Lambda_new = Lambda_;
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
    std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
    std::map<std::string, std::vector<Eigen::VectorXd>> G_new = G_;

    double cost_new = 0.0;
    double merit_function_new = 0.0;
    double constraint_violation_new = 0.0;

    if (constraint_set.empty())
    {
      for (int t = 0; t < horizon; ++t)
      {
        Eigen::VectorXd delta_x = result.state_trajectory[t] - context.X_[t];
        workspace_.delta_x_vectors[t] = delta_x;
        result.control_trajectory[t] =
            context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x;

        if (t < static_cast<int>(Lambda_new.size()))
        {
          Lambda_new[t] = Lambda_[t] + alpha * k_lambda_[t] + K_lambda_[t] * delta_x;
        }

        bool is_segment_boundary = (ms_segment_length_ > 1) &&
                                   ((t + 1) % ms_segment_length_ == 0) &&
                                   (t + 1 < horizon);
        bool apply_gap_closing = is_segment_boundary;

        F_new[t] = context.getSystem().getDiscreteDynamics(
            result.state_trajectory[t], result.control_trajectory[t],
            t * context.getTimestep());

        if (apply_gap_closing)
        {
          if (options.msipddp.rollout_type == "nonlinear")
          {
            result.state_trajectory[t + 1] = context.X_[t + 1] +
                                             (F_new[t] - F_[t]) +
                                             alpha * (F_[t] - context.X_[t + 1]);
          }
          else if (options.msipddp.rollout_type == "hybrid")
          {
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
            result.state_trajectory[t + 1] = F_new[t];
          }
        }
        else
        {
          result.state_trajectory[t + 1] = F_new[t];
        }

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
      result.alpha_du = 1.0;
      result.dynamics_trajectory = F_new;
      result.costate_trajectory = Lambda_new;

      return result;
    }

    double alpha_s = alpha;

    bool s_trajectory_feasible = true;
    for (int t = 0; t < horizon; ++t)
    {
      Eigen::VectorXd delta_x = result.state_trajectory[t] - context.X_[t];
      workspace_.delta_x_vectors[t] = delta_x;

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

      result.control_trajectory[t] =
          context.U_[t] + alpha_s * k_u_[t] + K_u_[t] * delta_x;

      bool is_segment_boundary = (ms_segment_length_ > 1) &&
                                 ((t + 1) % ms_segment_length_ == 0) &&
                                 (t + 1 < horizon);
      bool apply_gap_closing = is_segment_boundary;

      F_new[t] = context.getSystem().getDiscreteDynamics(
          result.state_trajectory[t], result.control_trajectory[t],
          t * context.getTimestep());

      if (apply_gap_closing)
      {
        if (options.msipddp.rollout_type == "nonlinear")
        {
          result.state_trajectory[t + 1] = context.X_[t + 1] +
                                           (F_new[t] - F_[t]) +
                                           alpha_s * (F_[t] - context.X_[t + 1]);
        }
        else if (options.msipddp.rollout_type == "hybrid")
        {
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
          result.state_trajectory[t + 1] = F_new[t];
        }
      }
      else
      {
        result.state_trajectory[t + 1] = F_new[t];
      }
    }

    if (!s_trajectory_feasible)
    {
      return result;
    }

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
        result.alpha_du = alpha_y_candidate;
        break;
      }
    }

    if (!suitable_alpha_y_found)
    {
      return result;
    }

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

        Eigen::VectorXd primal_residual = G_new[constraint_name][t] + s_vec;
        constraint_violation_new += primal_residual.lpNorm<1>();
      }

      Eigen::VectorXd defect_residual = F_new[t] - result.state_trajectory[t + 1];
      constraint_violation_new += defect_residual.lpNorm<1>();
    }

    cost_new +=
        context.getObjective().terminal_cost(result.state_trajectory.back());
    merit_function_new += cost_new;

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

  void MSIPDDPSolver::printIterationLegacy(int iter, double objective, double inf_pr,
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

    std::cout << std::setw(4) << iter << " ";

    std::cout << std::setw(12) << std::scientific << std::setprecision(6)
              << objective << " ";

    std::cout << std::setw(9) << std::scientific << std::setprecision(2) << inf_pr
              << " ";

    std::cout << std::setw(9) << std::scientific << std::setprecision(2) << inf_du
              << " ";

    std::cout << std::setw(9) << std::scientific << std::setprecision(2)
              << inf_comp << " ";

    if (mu > 0.0)
    {
      std::cout << std::setw(7) << std::fixed << std::setprecision(1)
                << std::log10(mu) << " ";
    }
    else
    {
      std::cout << std::setw(7) << "-inf" << " ";
    }

    std::cout << std::setw(9) << std::scientific << std::setprecision(2)
              << step_norm << " ";

    if (regularization > 0.0)
    {
      std::cout << std::setw(7) << std::fixed << std::setprecision(1)
                << std::log10(regularization) << " ";
    }
    else
    {
      std::cout << std::setw(7) << "-" << " ";
    }

    std::cout << std::setw(9) << std::fixed << std::setprecision(6) << alpha_du
              << " ";

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

    if (constraint_set.empty())
    {
      return;
    }

    switch (barrier_opts.strategy)
    {
      case BarrierStrategy::MONOTONIC:
      {
        mu_ = std::max(barrier_opts.mu_min_value,
                       barrier_opts.mu_update_factor * mu_);
        resetFilter(context);
        break;
      }

      case BarrierStrategy::IPOPT:
      {
        double scaled_inf_du = computeScaledDualInfeasibility(context);
        double error_k = std::max({scaled_inf_du, context.inf_pr_, context.inf_comp_});

        double kappa_epsilon = 10.0;

        if (error_k <= kappa_epsilon * mu_)
        {
          double new_mu_linear = barrier_opts.mu_update_factor * mu_;
          double new_mu_superlinear = std::pow(mu_, barrier_opts.mu_update_power);
          mu_ = std::max(options.tolerance / 10.0,
                         std::min(new_mu_linear, new_mu_superlinear));
          resetFilter(context);
        }
        break;
      }

      case BarrierStrategy::ADAPTIVE:
      default:
      {
        double scaled_inf_du = computeScaledDualInfeasibility(context);
        double termination_metric = std::max({scaled_inf_du, context.inf_pr_, context.inf_comp_});

        double barrier_update_threshold;

        if (mu_ < 1e-5)
        {
          barrier_update_threshold = std::max(termination_metric * 10.0, mu_ * 100.0);
        }
        else
        {
          barrier_update_threshold = std::max(barrier_opts.mu_update_factor * mu_, mu_ * 2.0);
        }

        bool slow_progress = forward_pass_success && context.alpha_pr_ > 0 &&
                            (termination_metric < 1e-3);

        if (termination_metric <= barrier_update_threshold || slow_progress)
        {
          double reduction_factor = barrier_opts.mu_update_factor;

          if (mu_ > 1e-12)
          {
            double kkt_progress_ratio = termination_metric / mu_;

            if (kkt_progress_ratio < 0.01)
            {
              reduction_factor = barrier_opts.mu_update_factor * 0.1;
            }
            else if (kkt_progress_ratio < 0.1)
            {
              reduction_factor = barrier_opts.mu_update_factor * 0.3;
            }
            else if (kkt_progress_ratio < 0.5)
            {
              reduction_factor = barrier_opts.mu_update_factor * 0.6;
            }
          }

          double new_mu_linear = reduction_factor * mu_;
          double new_mu_superlinear = std::pow(mu_, barrier_opts.mu_update_power);

          if (slow_progress && mu_ > options.tolerance)
          {
            mu_ = std::min(new_mu_linear, new_mu_superlinear);
          }
          else
          {
            mu_ = std::max(options.tolerance / 100.0,
                           std::min(new_mu_linear, new_mu_superlinear));
          }

          resetFilter(context);
        }
        break;
      }
    }
  }

  void MSIPDDPSolver::initializeConstraintStorage(CDDP &context)
  {
    const auto &constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();

    G_.clear();
    G_x_.clear();
    G_u_.clear();
    Y_.clear();
    S_.clear();
    k_y_.clear();
    K_y_.clear();
    k_s_.clear();
    K_s_.clear();

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

    if (constraint_set.empty())
    {
      return context.inf_du_;
    }

    const double smax = 100.0;

    double y_norm_l1 = 0.0;
    double s_norm_l1 = 0.0;
    int total_dual_dim = 0;

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

    int m = total_dual_dim;
    int n = control_dim * horizon;
    int m_plus_n = m + n;

    double scaling_numerator = (m_plus_n > 0) ? (y_norm_l1 + s_norm_l1) / static_cast<double>(m_plus_n) : 0.0;
    double sd = std::max(smax, scaling_numerator) / smax;

    return context.inf_du_ / sd;
  }

} // namespace cddp
