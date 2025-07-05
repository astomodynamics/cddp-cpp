/*
 * Copyright 2024 Tomo Sasaki
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cddp_core/alddp_solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace cddp {

AlddpSolver::AlddpSolver()
    : cost_(0.0), constraint_violation_(0.0), lagrangian_value_(0.0),
      optimality_gap_(0.0) {}

void AlddpSolver::initialize(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();

  // For warm starts, verify that existing state is valid
  if (options.warm_start) {
    bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                             K_u_.size() == static_cast<size_t>(horizon) &&
                             Lambda_.size() == static_cast<size_t>(horizon));

    if (valid_warm_start && !k_u_.empty()) {
      for (int t = 0; t < horizon; ++t) {
        if (k_u_[t].size() != control_dim || K_u_[t].rows() != control_dim ||
            K_u_[t].cols() != state_dim || Lambda_[t].size() != state_dim) {
          valid_warm_start = false;
          break;
        }
      }
    } else {
      valid_warm_start = false;
    }

    // Check dual variables validity for warm start
    if (valid_warm_start) {
      const auto &constraint_set = context.getConstraintSet();
      for (const auto &constraint_pair : constraint_set) {
        const std::string &constraint_name = constraint_pair.first;
        int dual_dim = constraint_pair.second->getDualDim();

        if (Y_.find(constraint_name) == Y_.end() ||
            Y_[constraint_name].size() != static_cast<size_t>(horizon)) {
          valid_warm_start = false;
          break;
        }

        for (int t = 0; t < horizon; ++t) {
          if (Y_[constraint_name][t].size() != dual_dim) {
            valid_warm_start = false;
            break;
          }
        }
        if (!valid_warm_start)
          break;
      }
    }

    if (valid_warm_start) {
      if (options.verbose) {
        std::cout << "ALDDP: Using warm start with existing control gains, "
                     "dual variables, and defect multipliers"
                  << std::endl;
      }
      // Initialize dynamics storage for warm start
      F_.resize(horizon, Eigen::VectorXd::Zero(state_dim));
      evaluateTrajectory(context);
      return;
    } else if (options.verbose) {
      std::cout << "ALDDP: Warning - warm start requested but no valid solver "
                   "state found. "
                << "Falling back to cold start initialization." << std::endl;
    }
  }

  // Cold start: full initialization
  k_u_.resize(horizon);
  K_u_.resize(horizon);

  for (int t = 0; t < horizon; ++t) {
    k_u_[t] = Eigen::VectorXd::Zero(control_dim);
    K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
  }

  // Initialize dynamics storage
  F_.resize(horizon, Eigen::VectorXd::Zero(state_dim));

  // Initialize Lagrange multipliers for defect constraints
  Lambda_.resize(horizon);
  for (int t = 0; t < horizon; ++t) {
    Lambda_[t] = Eigen::VectorXd::Constant(
        state_dim, options.altro.defect_dual_init_scale);
  }

  // Initialize dual variables for constraints
  Y_.clear();
  const auto &constraint_set = context.getConstraintSet();

  for (const auto &constraint_pair : constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    const auto &constraint = constraint_pair.second;
    int dual_dim = constraint->getDualDim();

    Y_[constraint_name].resize(horizon, Eigen::VectorXd::Zero(dual_dim));

    // Initialize dual variables to small positive values
    for (int t = 0; t < horizon; ++t) {
      Y_[constraint_name][t] = Eigen::VectorXd::Constant(
          dual_dim, options.altro.dual_var_init_scale);
    }

    // Initialize path constraint penalty parameter (scalar for ALDDP)
    rho_path_[constraint_name] = options.altro.penalty_scaling;
  }

  // Initialize defect constraint penalty parameter (scalar for ALDDP)
  rho_defect_ = options.altro.defect_penalty_scaling;

  // Initialize regularization
  context.regularization_ = options.regularization.initial_value;

  // ALDDP uses single-shooting only (simplified approach)
  if (options.verbose) {
    std::cout << "ALDDP: Single-shooting mode (standard dynamics propagation)"
              << std::endl;
  }
}

std::string AlddpSolver::getSolverName() const { return "ALDDP"; }

CDDPSolution AlddpSolver::solve(CDDP &context) {
  const CDDPOptions &options = context.getOptions();

  // Prepare solution map
  CDDPSolution solution;
  solution["solver_name"] = getSolverName();
  solution["status_message"] = std::string("Running");
  solution["iterations_completed"] = 0;
  solution["solve_time_ms"] = 0.0;

  // Initialize history vectors only if requested
  std::vector<double> history_objective;
  std::vector<double> history_lagrangian;
  std::vector<double> history_step_length_primal;
  std::vector<double> history_dual_infeasibility;
  std::vector<double> history_primal_infeasibility;
  std::vector<double> history_penalty_parameter;

  // Initial trajectory evaluation
  evaluateTrajectory(context);

  if (options.return_iteration_info) {
    const size_t expected_size =
        static_cast<size_t>(options.max_iterations + 1);
    history_objective.reserve(expected_size);
    history_lagrangian.reserve(expected_size);
    history_step_length_primal.reserve(expected_size);
    history_dual_infeasibility.reserve(expected_size);
    history_primal_infeasibility.reserve(expected_size);
    history_penalty_parameter.reserve(expected_size);

    // Initial iteration values
    history_objective.push_back(cost_);
    history_lagrangian.push_back(lagrangian_value_);
    history_step_length_primal.push_back(1.0);
    history_dual_infeasibility.push_back(optimality_gap_);
    history_primal_infeasibility.push_back(constraint_violation_);
    history_penalty_parameter.push_back(options.altro.penalty_scaling);
  }

  // Start timer
  auto start_time = std::chrono::high_resolution_clock::now();
  int iter = 0;
  bool converged = false;
  std::string termination_reason = "MaxIterationsReached";

  if (options.verbose) {
    printIteration(0, cost_, lagrangian_value_, 0.0, context.regularization_,
                   1.0, options.altro.penalty_scaling, constraint_violation_);
  }

  // Main ALTRO loop
  while (iter < options.max_iterations) {
    ++iter;

    // Check maximum CPU time
    if (options.max_cpu_time > 0) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          current_time - start_time);
      if (duration.count() > options.max_cpu_time * 1000) {
        termination_reason = "MaxCpuTimeReached";
        if (options.verbose) {
          std::cerr
              << "ALDDP: Maximum CPU time reached. Returning current solution"
              << std::endl;
        }
        break;
      }
    }

    // 1. Backward pass
    bool backward_pass_success = false;
    while (!backward_pass_success) {
      backward_pass_success = backwardPass(context);

      if (!backward_pass_success) {
        context.increaseRegularization();
        if (context.isRegularizationLimitReached()) {
          termination_reason = "RegularizationLimitReached_NotConverged";
          if (options.verbose) {
            std::cerr << "ALDDP: Backward pass regularization limit reached"
                      << std::endl;
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
    if (best_result.success) {
      if (options.debug) {
        std::cout << "[ALDDP: Forward pass] " << std::endl;
        std::cout << "    cost: " << best_result.cost << std::endl;
        std::cout << "    merit_function: " << best_result.merit_function
                  << std::endl;
        std::cout << "    alpha: " << best_result.alpha_pr << std::endl;
        std::cout << "    cv_err: " << best_result.constraint_violation
                  << std::endl;
      }

      context.X_ = best_result.state_trajectory;
      context.U_ = best_result.control_trajectory;
      if (best_result.dynamics_trajectory) {
        F_ = *best_result.dynamics_trajectory;
      }

      double dJ = cost_ - best_result.cost;
      double dL = lagrangian_value_ - best_result.merit_function;
      cost_ = best_result.cost;
      lagrangian_value_ = best_result.merit_function;
      context.alpha_pr_ = best_result.alpha_pr;
      constraint_violation_ = best_result.constraint_violation;

      // Store history only if requested
      if (options.return_iteration_info) {
        history_objective.push_back(cost_);
        history_lagrangian.push_back(lagrangian_value_);
        history_step_length_primal.push_back(context.alpha_pr_);
        history_dual_infeasibility.push_back(optimality_gap_);
        history_primal_infeasibility.push_back(constraint_violation_);
        history_penalty_parameter.push_back(options.altro.penalty_scaling);
      }

      context.decreaseRegularization();

      // Check convergence using proper gradient norm (optimality_gap_ is set by
      // backwardPass) Note: optimality_gap_ contains the gradient norm (Qu_err)
      // from backward pass
      double cost_change = std::abs(dJ);
      double lagrangian_change = std::abs(dL);

      if (optimality_gap_ <= options.tolerance &&
          constraint_violation_ <= options.altro.constraint_tolerance) {
        converged = true;
        termination_reason = "OptimalSolutionFound";
        break;
      }

      if (cost_change < options.acceptable_tolerance &&
          constraint_violation_ <= options.altro.constraint_tolerance) {
        converged = true;
        termination_reason = "AcceptableSolutionFound";
        break;
      }
    } else {
      context.increaseRegularization();

      if (context.isRegularizationLimitReached()) {
        termination_reason = "RegularizationLimitReached_NotConverged";
        converged = false;
        if (options.verbose) {
          std::cerr << "ALDDP: Regularization limit reached. Not converged."
                    << std::endl;
        }
        break;
      }
    }

    // Print iteration info
    if (options.verbose) {
      printIteration(iter, cost_, lagrangian_value_, optimality_gap_,
                     context.regularization_, context.alpha_pr_,
                     options.altro.penalty_scaling, constraint_violation_);
    }

    // Update augmented Lagrangian parameters
    updateAugmentedLagrangian(context);
  }

  // Compute final timing
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // Populate final solution
  solution["status_message"] = termination_reason;
  solution["iterations_completed"] = iter;
  solution["solve_time_ms"] = static_cast<double>(duration.count());
  solution["final_objective"] = cost_;
  solution["final_step_length"] = context.alpha_pr_;

  // Add trajectories
  std::vector<double> time_points;
  time_points.reserve(static_cast<size_t>(context.getHorizon() + 1));
  for (int t = 0; t <= context.getHorizon(); ++t) {
    time_points.push_back(t * context.getTimestep());
  }
  solution["time_points"] = time_points;
  solution["state_trajectory"] = context.X_;
  solution["control_trajectory"] = context.U_;

  // Add iteration history if requested
  if (options.return_iteration_info) {
    solution["history_objective"] = history_objective;
    solution["history_lagrangian"] = history_lagrangian;
    solution["history_step_length_primal"] = history_step_length_primal;
    solution["history_dual_infeasibility"] = history_dual_infeasibility;
    solution["history_primal_infeasibility"] = history_primal_infeasibility;
    solution["history_penalty_parameter"] = history_penalty_parameter;
  }

  // Add control gains
  solution["control_feedback_gains_K"] = K_u_;

  // Final metrics
  solution["final_regularization"] = context.regularization_;
  solution["final_penalty_parameter"] = options.altro.penalty_scaling;
  solution["final_primal_infeasibility"] = constraint_violation_;
  solution["final_dual_infeasibility"] = optimality_gap_;
  solution["final_lagrangian"] = lagrangian_value_;

  if (options.verbose) {
    printSolutionSummary(solution);
  }

  return solution;
}

void AlddpSolver::evaluateTrajectory(CDDP &context) {
  const auto &X = context.X_;
  const auto &U = context.U_;
  const auto &objective = context.getObjective();
  const auto &system = context.getSystem();
  const auto &constraint_set = context.getConstraintSet();
  const int horizon = context.getHorizon();
  const double timestep = context.getTimestep();
  const double penalty_scaling = context.getOptions().altro.penalty_scaling;
  const double defect_penalty_scaling =
      context.getOptions().altro.defect_penalty_scaling;

  // Compute cost
  cost_ = 0.0;
  for (int t = 0; t < horizon; ++t) {
    cost_ += objective.running_cost(X[t], U[t], t);

    // Store dynamics
    F_[t] = system.getDiscreteDynamics(X[t], U[t], t * timestep);
  }
  cost_ += objective.terminal_cost(X.back());

  // Compute constraint violation and augmented Lagrangian terms
  constraint_violation_ = 0.0;
  double penalty_cost = 0.0;

  // Add defect constraint Lagrangian terms
  for (int t = 0; t < horizon; ++t) {
    // Compute defect: d_t = x_{t+1} - f(x_t, u_t)
    Eigen::VectorXd defect = X[t + 1] - F_[t];

    // Add Lagrangian term: λ_t^T * d_t
    penalty_cost += Lambda_[t].dot(defect);

    // Add quadratic penalty term: 0.5 * ρ_defect * ||d_t||^2 (simplified for
    // ALDDP)
    penalty_cost += 0.5 * rho_defect_ * defect.squaredNorm();

    // Update constraint violation with defect norm
    constraint_violation_ += defect.norm();
  }

  // Add path constraint terms
  for (const auto &constraint_pair : constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    const auto &constraint = constraint_pair.second;

    for (int t = 0; t < horizon; ++t) {
      // Evaluate constraint
      Eigen::VectorXd g =
          constraint->evaluate(X[t], U[t]) - constraint->getUpperBound();
      const Eigen::VectorXd &y = Y_[constraint_name][t];
      const double rho_path = rho_path_[constraint_name];

      // Update constraint violation
      constraint_violation_ += std::max(0.0, g.maxCoeff());

      // Simplified augmented Lagrangian terms for ALDDP
      for (int i = 0; i < g.size(); ++i) {
        const double constraint_tolerance =
            1e-12; // Small tolerance for numerical stability

        if (g(i) > constraint_tolerance) {
          // Active constraint: add linear and quadratic penalty terms
          penalty_cost += y(i) * g(i) + 0.5 * rho_path * g(i) * g(i);
        } else {
          // Inactive constraint: only quadratic penalty if multiplier is large
          double projected_multiplier = std::max(0.0, y(i) + rho_path * g(i));
          if (projected_multiplier > constraint_tolerance) {
            penalty_cost += 0.5 * rho_path * g(i) * g(i);
          }
        }
      }
    }
  }

  lagrangian_value_ = cost_ + penalty_cost;
}

bool AlddpSolver::backwardPass(CDDP &context) {
  const auto &options = context.getOptions();

  const auto &X = context.X_;
  const auto &U = context.U_;
  const auto &objective = context.getObjective();
  const auto &system = context.getSystem();
  const auto &constraint_set = context.getConstraintSet();
  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const double timestep = context.getTimestep();
  const double penalty_scaling = context.getOptions().altro.penalty_scaling;
  const double defect_penalty_scaling =
      context.getOptions().altro.defect_penalty_scaling;
  const bool is_ilqr = context.getOptions().use_ilqr;

  double Qu_err = 0.0;

  // Terminal cost derivatives
  Eigen::VectorXd V_x = objective.getFinalCostGradient(X.back());
  Eigen::MatrixXd V_xx = objective.getFinalCostHessian(X.back());
  V_xx = 0.5 * (V_xx + V_xx.transpose());

  // Backward recursion
  for (int t = horizon - 1; t >= 0; --t) {
    const Eigen::VectorXd &x = X[t];
    const Eigen::VectorXd &u = U[t];
    const Eigen::VectorXd &f = F_[t];
    const Eigen::VectorXd &d = f - context.X_[t + 1]; // Defect
    const Eigen::VectorXd &lambda = Lambda_[t];

    // Get dynamics derivatives
    const auto [Fx, Fu] = system.getJacobians(x, u, t * timestep);
    Eigen::MatrixXd A =
        Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
    Eigen::MatrixXd B = timestep * Fu;

    // Cost derivatives at (x_t, u_t)
    auto [l_x, l_u] = objective.getRunningCostGradients(x, u, t);
    auto [l_xx, l_uu, l_ux] = objective.getRunningCostHessians(x, u, t);

    // Initialize Q-function with cost and dynamics terms (simplified for ALDDP)
    Eigen::VectorXd Q_x =
        l_x + A.transpose() * V_x + A.transpose() * (lambda + rho_defect_ * d);
    Eigen::VectorXd Q_u =
        l_u + B.transpose() * V_x + B.transpose() * (lambda + rho_defect_ * d);
    Eigen::MatrixXd Q_xx =
        l_xx + A.transpose() * V_xx * A + rho_defect_ * A.transpose() * A;
    Eigen::MatrixXd Q_ux =
        l_ux + B.transpose() * V_xx * A + rho_defect_ * B.transpose() * A;
    Eigen::MatrixXd Q_uu =
        l_uu + B.transpose() * V_xx * B + rho_defect_ * B.transpose() * B;

    // Add path constraint terms to Q-function
    for (const auto &constraint_pair : constraint_set) {
      const std::string &constraint_name = constraint_pair.first;
      const auto &constraint = constraint_pair.second;
      const Eigen::VectorXd &y = Y_[constraint_name][t];
      const double rho_path = rho_path_[constraint_name];

      // Evaluate constraint and its derivatives
      Eigen::VectorXd g =
          constraint->evaluate(x, u) - constraint->getUpperBound();
      Eigen::MatrixXd g_x = constraint->getStateJacobian(x, u);
      Eigen::MatrixXd g_u = constraint->getControlJacobian(x, u);

      for (int i = 0; i < g.size(); ++i) {
        const double constraint_tolerance =
            1e-12; // Small tolerance for numerical stability

        if (g(i) > constraint_tolerance ||
            (g(i) <= constraint_tolerance &&
             y(i) + rho_path * g(i) > constraint_tolerance)) {
          // Add first-order terms to Q-function (simplified for ALDDP)
          double weight = y(i) + rho_path * g(i);
          Q_x += weight * g_x.row(i).transpose();
          Q_u += weight * g_u.row(i).transpose();

          // Add second-order terms (Gauss-Newton approximation)
          Q_xx += rho_path * g_x.row(i).transpose() * g_x.row(i);
          Q_ux += rho_path * g_u.row(i).transpose() * g_x.row(i);
          Q_uu += rho_path * g_u.row(i).transpose() * g_u.row(i);
        }
      }
    }

    // Regularization
    double reg = context.regularization_;
    Eigen::MatrixXd Q_uu_reg = Q_uu;
    Q_uu_reg.diagonal().array() += reg;
    Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose());

    // Solve for control law
    Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
    if (ldlt.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "ALDDP: Backward pass failed at time " << t << std::endl;
      }
      return false;
    }

    Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
    bigRHS.col(0) = Q_u;
    Eigen::MatrixXd M = Q_ux;
    for (int col = 0; col < state_dim; col++) {
      bigRHS.col(col + 1) = M.col(col);
    }

    Eigen::MatrixXd kK = -ldlt.solve(bigRHS);

    // parse out feedforward (ku) and feedback (Ku)
    Eigen::VectorXd k_u = kK.col(0);
    Eigen::MatrixXd K_u(control_dim, state_dim);
    for (int col = 0; col < state_dim; col++) {
      K_u.col(col) = kK.col(col + 1);
    }

    // Save gains
    k_u_[t] = k_u;
    K_u_[t] = K_u;

    // Update value function
    V_x = Q_x + K_u_[t].transpose() * Q_u + Q_ux.transpose() * k_u_[t] +
          K_u_[t].transpose() * Q_uu * k_u_[t];
    V_xx = Q_xx + K_u_[t].transpose() * Q_ux + Q_ux.transpose() * K_u_[t] +
           K_u_[t].transpose() * Q_uu * K_u_[t];
    V_xx = 0.5 * (V_xx + V_xx.transpose());

    // Compute optimality gap (Inf-norm) for convergence check
    Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
  }

  optimality_gap_ = Qu_err;
  context.inf_du_ = optimality_gap_;

  if (options.debug) {
    std::cout << "[ALDDP Backward Pass]\n"
              << "    Qu_err:  " << std::scientific << std::setprecision(4)
              << Qu_err << std::endl;
  }

  return true;
}

ForwardPassResult AlddpSolver::performForwardPass(CDDP &context) {
  const auto &options = context.getOptions();

  ForwardPassResult best_result;
  best_result.success = false;
  best_result.merit_function = std::numeric_limits<double>::infinity();

  // Try different step sizes from the context
  for (double alpha : context.alphas_) {
    ForwardPassResult result = forwardPass(context, alpha);

    if (result.success && result.merit_function < best_result.merit_function) {
      best_result = result;
    }

    // Early termination if we found a good step
    if (result.success && result.merit_function < lagrangian_value_) {
      break;
    }
  }

  return best_result;
}

ForwardPassResult AlddpSolver::forwardPass(CDDP &context, double alpha) {
  const auto &options = context.getOptions();

  ForwardPassResult result;
  result.success = false;
  result.alpha_pr = alpha;
  result.cost = std::numeric_limits<double>::infinity();
  result.merit_function = std::numeric_limits<double>::infinity();

  const auto &X = context.X_;
  const auto &U = context.U_;
  const auto &system = context.getSystem();
  const auto &objective = context.getObjective();
  const auto &constraint_set = context.getConstraintSet();
  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();
  const double timestep = context.getTimestep();

  // Initialize new trajectories
  std::vector<Eigen::VectorXd> X_new = X;
  std::vector<Eigen::VectorXd> U_new = U;
  std::vector<Eigen::VectorXd> F_new(horizon);

  // Set initial state
  X_new[0] = context.getInitialState();

  // Forward rollout with control law (with optional multiple-shooting)
  for (int t = 0; t < horizon; ++t) {
    Eigen::VectorXd dx = X_new[t] - X[t];

    // Apply control law: u_new = u + α*k + K*dx
    U_new[t] = U[t] + alpha * k_u_[t] + K_u_[t] * dx;

    // Check for numerical issues
    if (!U_new[t].allFinite()) {
      if (options.debug) {
        std::cerr << "ALDDP: Forward pass - control NaN/Inf at time " << t
                  << std::endl;
      }
      return result;
    }

    // Integrate dynamics
    F_new[t] = system.getDiscreteDynamics(X_new[t], U_new[t], t * timestep);

    if (!F_new[t].allFinite()) {
      if (options.debug) {
        std::cerr << "ALDDP: Forward pass - dynamics NaN/Inf at time " << t
                  << std::endl;
      }
      return result;
    }

    // ALDDP uses single-shooting only: direct dynamics propagation
    X_new[t + 1] = F_new[t];

    // Check for numerical issues
    if (!X_new[t + 1].allFinite()) {
      if (options.debug) {
        std::cerr << "ALDDP: Forward pass - state NaN/Inf at time " << t
                  << std::endl;
      }
      return result;
    }
  }

  // Evaluate new trajectory
  double cost_new = 0.0;
  double constraint_violation_new = 0.0;
  double penalty_cost = 0.0;

  // 1. Compute cost
  for (int t = 0; t < horizon; ++t) {
    cost_new += objective.running_cost(X_new[t], U_new[t], t);
  }
  cost_new += objective.terminal_cost(X_new.back());

  // 2. Add defect constraint Lagrangian terms
  for (int t = 0; t < horizon; ++t) {
    // Compute defect for new trajectory: d_t = x_{t+1} - f(x_t, u_t)
    Eigen::VectorXd defect_new = X_new[t + 1] - F_new[t];
    const Eigen::VectorXd &lambda = Lambda_[t];

    // Lagrangian term: λ_t^T * defect_new
    penalty_cost += lambda.dot(defect_new);

    // Quadratic penalty: 0.5 * ρ_defect * ||defect_new||^2 (simplified for
    // ALDDP)
    penalty_cost += 0.5 * rho_defect_ * defect_new.squaredNorm();

    // Update constraint violation
    constraint_violation_new += defect_new.norm();
  }

  // 3. Add path constraint Lagrangian terms
  for (const auto &constraint_pair : constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    const auto &constraint = constraint_pair.second;

    for (int t = 0; t < horizon; ++t) {
      Eigen::VectorXd g = constraint->evaluate(X_new[t], U_new[t]) -
                          constraint->getUpperBound();
      const Eigen::VectorXd &y = Y_[constraint_name][t];
      const double rho_path = rho_path_[constraint_name];

      constraint_violation_new += std::max(0.0, g.maxCoeff());

      // Apply simplified augmented Lagrangian terms for ALDDP
      for (int i = 0; i < g.size(); ++i) {
        const double constraint_tolerance =
            1e-12; // Small tolerance for numerical stability

        if (g(i) > constraint_tolerance) {
          // Active constraint: linear + quadratic penalty terms
          penalty_cost += y(i) * g(i) + 0.5 * rho_path * g(i) * g(i);
        } else {
          // Inactive constraint: only quadratic penalty if projected multiplier
          // is positive
          double projected_multiplier = std::max(0.0, y(i) + rho_path * g(i));
          if (projected_multiplier > constraint_tolerance) {
            penalty_cost += 0.5 * rho_path * g(i) * g(i);
          }
        }
      }
    }
  }

  double merit_function_new = cost_new + penalty_cost;

  // Acceptance test: accept if merit function improves or constraint violation
  // decreases
  if (merit_function_new < lagrangian_value_ ||
      constraint_violation_new < constraint_violation_) {
    result.success = true;
    result.state_trajectory = X_new;
    result.control_trajectory = U_new;
    result.dynamics_trajectory = F_new;
    result.cost = cost_new;
    result.merit_function = merit_function_new;
    result.constraint_violation = constraint_violation_new;

    if (options.debug) {
      std::cout << "[ALDDP Forward Pass]\n"
                << "    alpha: " << std::fixed << std::setprecision(4) << alpha
                << "\n"
                << "    cost: " << std::scientific << std::setprecision(4)
                << cost_new << "\n"
                << "    merit: " << std::scientific << std::setprecision(4)
                << merit_function_new << "\n"
                << "    cv_err: " << std::scientific << std::setprecision(4)
                << constraint_violation_new << std::endl;
    }
  }

  return result;
}

void AlddpSolver::updateAugmentedLagrangian(CDDP &context) {
  const auto &options = context.getOptions();
  const auto &X = context.X_;
  const auto &U = context.U_;
  const auto &constraint_set = context.getConstraintSet();
  const int horizon = context.getHorizon();

  double max_defect_violation = 0.0;
  double max_path_violation = 0.0;

  // 1. Update defect constraint Lagrange multipliers (simplified for ALDDP)
  for (int t = 0; t < horizon; ++t) {
    // Compute defect: d_t = x_{t+1} - f(x_t, u_t)
    Eigen::VectorXd defect = X[t + 1] - F_[t];

    // Update multipliers: λ_new = λ_old + ρ_defect * defect (scalar penalty)
    Lambda_[t] += rho_defect_ * defect;

    // Track maximum defect violation for debugging
    max_defect_violation = std::max(max_defect_violation, defect.norm());
  }

  // 2. Update path constraint dual variables (Lagrange multipliers)
  for (const auto &constraint_pair : constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    const auto &constraint = constraint_pair.second;

    for (int t = 0; t < horizon; ++t) {
      Eigen::VectorXd g =
          constraint->evaluate(X[t], U[t]) - constraint->getUpperBound();
      Eigen::VectorXd &y = Y_[constraint_name][t];
      const double rho_path = rho_path_[constraint_name];

      // Update multipliers: y_new = max(0, y_old + rho_path * g) (simplified
      // for ALDDP)
      for (int i = 0; i < g.size(); ++i) {
        y(i) = std::max(0.0, y(i) + rho_path * g(i));

        // Track maximum path constraint violation
        if (g(i) > 0.0) {
          max_path_violation = std::max(max_path_violation, g(i));
        }
      }
    }
  }

  if (options.debug) {
    std::cout << "[ALDDP Multiplier Update]\n"
              << "    max_defect_viol: " << std::scientific
              << std::setprecision(4) << max_defect_violation << "\n"
              << "    max_path_viol:   " << std::scientific
              << std::setprecision(4) << max_path_violation << std::endl;
  }
}

void AlddpSolver::printIteration(int iter, double cost, double lagrangian,
                                 double grad_norm, double regularization,
                                 double alpha, double mu,
                                 double constraint_violation) const {
  if (iter == 0) {
    std::cout << std::setw(4) << "iter" << std::setw(12) << "cost"
              << std::setw(12) << "lagrangian" << std::setw(12) << "grad_norm"
              << std::setw(8) << "alpha" << std::setw(10) << "penalty"
              << std::setw(12) << "viol" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
  }

  std::cout << std::setw(4) << iter << std::setw(12) << std::scientific
            << std::setprecision(3) << cost << std::setw(12) << std::scientific
            << std::setprecision(3) << lagrangian << std::setw(12)
            << std::scientific << std::setprecision(3) << grad_norm
            << std::setw(8) << std::fixed << std::setprecision(2) << alpha
            << std::setw(10) << std::scientific << std::setprecision(2) << mu
            << std::setw(12) << std::scientific << std::setprecision(3)
            << constraint_violation << std::endl;
}

void AlddpSolver::printSolutionSummary(const CDDPSolution &solution) const {
  std::cout << "\n=== ALDDP Solution Summary ===" << std::endl;
  auto status_it = solution.find("status_message");
  auto iterations_it = solution.find("iterations_completed");
  auto solve_time_it = solution.find("solve_time_ms");
  auto final_cost_it = solution.find("final_objective");
  auto final_lagrangian_it = solution.find("final_lagrangian");
  auto final_alpha_it = solution.find("final_step_length");

  std::cout << "Status: "
            << (status_it != solution.end()
                    ? std::any_cast<std::string>(status_it->second)
                    : "N/A")
            << std::endl;
  std::cout << "Iterations: "
            << (iterations_it != solution.end()
                    ? std::any_cast<int>(iterations_it->second)
                    : -1)
            << std::endl;
  std::cout << "Solve time: "
            << (solve_time_it != solution.end()
                    ? std::any_cast<double>(solve_time_it->second) * 1e-3
                    : -1.0)
            << " seconds" << std::endl;
  std::cout << "Final cost: "
            << (final_cost_it != solution.end()
                    ? std::any_cast<double>(final_cost_it->second)
                    : -1.0)
            << std::endl;
  std::cout << "Final lagrangian: "
            << (final_lagrangian_it != solution.end()
                    ? std::any_cast<double>(final_lagrangian_it->second)
                    : -1.0)
            << std::endl;
  std::cout << "Final step size: "
            << (final_alpha_it != solution.end()
                    ? std::any_cast<double>(final_alpha_it->second)
                    : -1.0)
            << std::endl;
  std::cout << "================================\n" << std::endl;
}

} // namespace cddp
