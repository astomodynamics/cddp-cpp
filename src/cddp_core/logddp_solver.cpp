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
#include <chrono>
#include <cmath>
#include <execution>
#include <future>
#include <iomanip>
#include <iostream>

namespace cddp {

namespace {

void rollOutNominalTrajectory(CDDP &context) {
  const int horizon = context.getHorizon();

  context.X_[0] = context.getInitialState();
  for (int t = 0; t < horizon; ++t) {
    context.X_[t + 1] = context.getSystem().getDiscreteDynamics(
        context.X_[t], context.U_[t], t * context.getTimestep());
  }
}

} // namespace

LogDDPSolver::LogDDPSolver()
    : mu_(1e-1), relaxation_delta_(1e-5), constraint_violation_(1e+7) {}

void LogDDPSolver::initialize(CDDP &context) {
  const CDDPOptions &options = context.getOptions();

  int horizon = context.getHorizon();
  int control_dim = context.getControlDim();
  int state_dim = context.getStateDim();

  // Check if reference_state in objective and reference_state in context are
  // the same
  if ((context.getReferenceState() - context.getObjective().getReferenceState())
          .norm() > 1e-6) {
    std::cerr << "LogDDP: Initial state and goal state in the objective "
                 "function do not match"
              << std::endl;
    throw std::runtime_error(
        "Initial state and goal state in the objective function do not match");
  }

  // For warm starts, verify that existing state is valid
  if (options.warm_start) {
    bool valid_warm_start =
        (k_u_.size() == static_cast<size_t>(horizon) &&
         K_u_.size() == static_cast<size_t>(horizon) &&
         context.X_.size() == static_cast<size_t>(horizon + 1) &&
         context.U_.size() == static_cast<size_t>(horizon));

    if (valid_warm_start && !k_u_.empty()) {
      for (int t = 0; t < horizon; ++t) {
        if (k_u_[t].size() != control_dim || K_u_[t].rows() != control_dim ||
            K_u_[t].cols() != state_dim) {
          valid_warm_start = false;
          break;
        }
      }
    } else {
      valid_warm_start = false;
    }

    if (valid_warm_start) {
      if (options.verbose) {
        std::cout << "LogDDP: Using warm start with existing control gains"
                  << std::endl;
      }

      // Reset barrier parameters for warm start
      mu_ = options.log_barrier.barrier.mu_initial;
      relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
      if (!relaxed_log_barrier_) {
        relaxed_log_barrier_ =
            std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);
      } else {
        relaxed_log_barrier_->setBarrierCoeff(mu_);
        relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
      }

      // Initialize regularization
      context.regularization_ = options.regularization.initial_value;

      // Initialize line search parameters
      context.alphas_.clear();
      double alpha = options.line_search.initial_step_size;
      for (int i = 0; i < options.line_search.max_iterations; ++i) {
        context.alphas_.push_back(alpha);
        alpha *= options.line_search.step_reduction_factor;
      }
      context.alpha_pr_ = options.line_search.initial_step_size;
      dV_ = Eigen::Vector2d::Zero();

      constraint_violation_ = std::numeric_limits<double>::infinity();

      // Warm starts may reuse gains with a user-modified state trajectory.
      // Re-roll the nominal state sequence so the linearization point stays
      // dynamically consistent.
      rollOutNominalTrajectory(context);

      // Evaluate current trajectory and reset filter
      evaluateTrajectory(context);
      resetFilter(context);
      return;
    } else if (options.verbose) {
      std::cout << "LogDDP: Warning - warm start requested but no valid solver "
                   "state found. "
                << "Falling back to cold start initialization." << std::endl;
    }
  }

  // Cold start: initialize a nominal trajectory and always roll it out from
  // the current control guess so the linearization point is dynamically
  // consistent, even when the user provided a same-sized state guess.
  if (context.X_.size() != static_cast<size_t>(horizon + 1) ||
      context.U_.size() != static_cast<size_t>(horizon)) {
    context.X_.resize(horizon + 1);
    context.U_.resize(horizon);

    // Create X_ initial guess using initial_state and reference_state by
    // interpolating between them
    for (int t = 0; t <= horizon; ++t) {
      context.X_[t] =
          context.getInitialState() +
          t * (context.getReferenceState() - context.getInitialState()) /
              horizon;
    }

    for (int t = 0; t < horizon; ++t) {
      context.U_[t] = Eigen::VectorXd::Zero(control_dim);
    }
  }

  // Roll out the nominal trajectory for all cold starts, including user-
  // supplied trajectories whose state sequence is only an initial guess.
  rollOutNominalTrajectory(context);

  // Use base class helper for gains
  initializeGains(horizon, control_dim, state_dim);

  G_.clear();

  const auto &constraint_set = context.getConstraintSet();
  for (const auto &constraint_pair : constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    int dual_dim = constraint_pair.second->getDualDim();

    G_[constraint_name].resize(horizon);
    for (int t = 0; t < horizon; ++t) {
      G_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
    }
  }

  // Initialize cost using objective evaluation
  context.cost_ = context.getObjective().evaluate(context.X_, context.U_);

  // Initialize line search parameters
  context.alphas_.clear();
  double alpha = options.line_search.initial_step_size;
  for (int i = 0; i < options.line_search.max_iterations; ++i) {
    context.alphas_.push_back(alpha);
    alpha *= options.line_search.step_reduction_factor;
  }
  context.alpha_pr_ = options.line_search.initial_step_size;
  dV_ = Eigen::Vector2d::Zero();

  // Initialize regularization
  context.regularization_ = options.regularization.initial_value;

  constraint_violation_ = std::numeric_limits<double>::infinity();

  // Initialize log barrier object
  mu_ = options.log_barrier.barrier.mu_initial;
  relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
  if (!relaxed_log_barrier_) {
    relaxed_log_barrier_ =
        std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);
  } else {
    relaxed_log_barrier_->setBarrierCoeff(mu_);
    relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
  }

  // Evaluate initial trajectory
  evaluateTrajectory(context);
  resetFilter(context);
}

std::string LogDDPSolver::getSolverName() const { return "LogDDP"; }

// === Hook implementations ===

void LogDDPSolver::preIterationSetup(CDDP &context) {
  evaluateTrajectory(context);
  resetFilter(context);
}

bool LogDDPSolver::handleBackwardPassRegularizationLimit(
    CDDP & /*context*/, std::string &termination_reason) {
  // LogDDP treats regularization exhaustion as converged — the current
  // barrier-penalized trajectory is typically still usable.
  termination_reason = "RegularizationLimitReached_Converged";
  return true; // converged
}

void LogDDPSolver::applyForwardPassResult(CDDP &context,
                                          const ForwardPassResult &result) {
  // Call base to update X_, U_, cost_, merit_function_, alpha_pr_, alpha_du_
  CDDPSolverBase::applyForwardPassResult(context, result);

  // Also update LogDDP-specific state
  constraint_violation_ = result.constraint_violation;
}

bool LogDDPSolver::checkConvergence(CDDP &context, double dJ, double dL,
                                    int /*iter*/, std::string &reason) {
  const CDDPOptions &options = context.getOptions();

  double termination_metric = std::max(context.inf_du_, context.inf_pr_);
  if (termination_metric <= options.tolerance) {
    if (options.debug) {
      std::cout << "LogDDP: Converged due to optimality gap and constraint "
                   "violation."
                << std::endl;
    }
    reason = "OptimalSolutionFound";
    return true;
  }

  if (std::abs(dJ) < options.acceptable_tolerance &&
      std::abs(dL) < options.acceptable_tolerance) {
    if (options.debug) {
      std::cout
          << "LogDDP: Converged due to small change in cost and Lagrangian."
          << std::endl;
    }
    reason = "AcceptableSolutionFound";
    return true;
  }

  return false;
}

void LogDDPSolver::postIterationUpdate(CDDP &context,
                                       bool forward_pass_success) {
  const CDDPOptions &options = context.getOptions();

  // Barrier update logic
  if (forward_pass_success) {
    mu_ = std::max(options.log_barrier.barrier.mu_min_value,
                   mu_ * options.log_barrier.barrier.mu_update_factor);
  } else {
    mu_ = std::min(options.log_barrier.barrier.mu_initial, mu_ * 5.0);
  }
  relaxed_log_barrier_->setBarrierCoeff(mu_);
  relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
  resetFilter(context);
}

void LogDDPSolver::recordIterationHistory(const CDDP &context) {
  // Call base to record common metrics
  CDDPSolverBase::recordIterationHistory(context);

  // Add LogDDP-specific metrics
  history_.barrier_mu.push_back(mu_);
}

void LogDDPSolver::populateSolverSpecificSolution(CDDPSolution &solution,
                                                  const CDDP &context) {
  solution.final_barrier_mu = mu_;
  solution.final_primal_infeasibility = constraint_violation_;
  solution.final_dual_infeasibility = context.inf_du_;
}

void LogDDPSolver::printIteration(int iter, const CDDP &context) const {
  if (iter == 0) {
    std::cout << std::setw(4) << "iter" << " " << std::setw(12) << "objective"
              << " " << std::setw(12) << "lagrangian" << " " << std::setw(10)
              << "opt_gap" << " " << std::setw(8) << "lg(rg)" << " "
              << std::setw(8) << "alpha" << " " << std::setw(8) << "lg(mu)"
              << " " << std::setw(10) << "cv_viol" << std::endl;
  }

  std::cout << std::setw(4) << iter << " " << std::setw(12) << std::scientific
            << std::setprecision(4) << context.cost_ << " " << std::setw(12)
            << std::scientific << std::setprecision(4) << context.merit_function_
            << " " << std::setw(10) << std::scientific << std::setprecision(2)
            << context.inf_du_ << " " << std::setw(8) << std::fixed
            << std::setprecision(1) << std::log10(context.regularization_)
            << " " << std::setw(8) << std::fixed << std::setprecision(4)
            << context.alpha_pr_ << " " << std::setw(8) << std::fixed
            << std::setprecision(1) << std::log10(mu_) << " " << std::setw(10)
            << std::scientific << std::setprecision(2) << constraint_violation_
            << std::endl;
}

// === Private helper methods ===

void LogDDPSolver::evaluateTrajectory(CDDP &context) {
  const int horizon = context.getHorizon();
  double cost = 0.0;

  // Calculate cost over the (already consistent) trajectory
  for (int t = 0; t < horizon; ++t) {
    cost += context.getObjective().running_cost(context.X_[t], context.U_[t], t);
  }

  // Add terminal cost based on the final guessed state
  cost += context.getObjective().terminal_cost(context.X_.back());

  // Store the initial total cost
  context.cost_ = cost;
}

void LogDDPSolver::resetFilter(CDDP &context) {
  // Evaluate log-barrier cost (includes path constraints)
  context.merit_function_ = context.cost_;
  constraint_violation_ = 0.0;

  const auto &constraint_set = context.getConstraintSet();

  // Calculate path constraint terms and violation
  for (int t = 0; t < context.getHorizon(); ++t) {
    for (const auto &constraint_pair : constraint_set) {
      const std::string &constraint_name = constraint_pair.first;
      Eigen::VectorXd g_t =
          constraint_pair.second->evaluate(context.X_[t], context.U_[t]) -
          constraint_pair.second->getUpperBound();
      G_[constraint_name][t] = g_t;
      context.merit_function_ += relaxed_log_barrier_->evaluate(
          *constraint_pair.second, context.X_[t], context.U_[t]);

      for (int i = 0; i < g_t.size(); ++i) {
        if (g_t(i) > 0.0) {
          constraint_violation_ += g_t(i);
        }
      }
    }
  }
  context.inf_pr_ = constraint_violation_;
}

// === Backward pass ===

bool LogDDPSolver::backwardPass(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();
  const double timestep = context.getTimestep();
  const auto &constraint_set = context.getConstraintSet();

  const int MIN_HORIZON_FOR_PARALLEL = 20;
  const bool use_parallel =
      options.enable_parallel && horizon >= MIN_HORIZON_FOR_PARALLEL;

  // Resize storage
  F_x_.resize(horizon);
  F_u_.resize(horizon);
  F_xx_.resize(horizon);
  F_uu_.resize(horizon);
  F_ux_.resize(horizon);

  if (!use_parallel) {
    for (int t = 0; t < horizon; ++t) {
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
    }
  } else {
    const int num_threads =
        std::min(options.num_threads,
                 static_cast<int>(std::thread::hardware_concurrency()));
    const int chunk_size = std::max(1, horizon / num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      int start_t = thread_id * chunk_size;
      int end_t = (thread_id == num_threads - 1) ? horizon
                                                 : (thread_id + 1) * chunk_size;

      if (start_t >= horizon)
        break;

      futures.push_back(
          std::async(std::launch::async, [this, &context, &options, start_t,
                                          end_t, timestep]() {
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
            }
          }));
    }

    for (auto &future : futures) {
      try {
        if (future.valid()) {
          future.get();
        }
      } catch (const std::exception &e) {
        if (options.verbose) {
          std::cerr
              << "LogDDP: Dynamics derivatives computation thread failed: "
              << e.what() << std::endl;
        }
        throw;
      }
    }
  }

  if (options.debug) {
    std::cout << "[LogDDP] Pre-computed dynamics derivatives for " << horizon
              << " time steps using "
              << (use_parallel ? "parallel" : "sequential") << " computation"
              << std::endl;
  }

  // Terminal cost and derivatives (V_x, V_xx at t=N)
  Eigen::VectorXd V_x =
      context.getObjective().getFinalCostGradient(context.X_.back());
  Eigen::MatrixXd V_xx =
      context.getObjective().getFinalCostHessian(context.X_.back());
  V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

  dV_ = Eigen::Vector2d::Zero();
  double Qu_err = 0.0;

  // Backward Riccati recursion
  for (int t = horizon - 1; t >= 0; --t) {
    const Eigen::VectorXd &x = context.X_[t];
    const Eigen::VectorXd &u = context.U_[t];

    // Use pre-computed continuous-time dynamics Jacobians
    const Eigen::MatrixXd &Fx = F_x_[t];
    const Eigen::MatrixXd &Fu = F_u_[t];
    const Eigen::MatrixXd &A =
        timestep * Fx + Eigen::MatrixXd::Identity(state_dim, state_dim);
    const Eigen::MatrixXd &B = timestep * Fu;

    // Cost derivatives at (x_t, u_t)
    auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
    auto [l_xx, l_uu, l_ux] =
        context.getObjective().getRunningCostHessians(x, u, t);

    Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
    Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
    Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
    Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
    Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

    // Add state hessian term if not using iLQR
    if (!options.use_ilqr) {
      const auto &Fxx = F_xx_[t];
      const auto &Fuu = F_uu_[t];
      const auto &Fux = F_ux_[t];

      for (int i = 0; i < state_dim; ++i) {
        Q_xx += timestep * V_x(i) * Fxx[i];
        Q_ux += timestep * V_x(i) * Fux[i];
        Q_uu += timestep * V_x(i) * Fuu[i];
      }
    }

    // Apply Log-barrier cost gradients and Hessians
    for (const auto &constraint_pair : constraint_set) {
      auto [L_x_relaxed, L_u_relaxed] =
          relaxed_log_barrier_->getGradients(*constraint_pair.second, x, u);
      Q_x += L_x_relaxed;
      Q_u += L_u_relaxed;

      auto [L_xx_relaxed, L_uu_relaxed, L_ux_relaxed] =
          relaxed_log_barrier_->getHessians(*constraint_pair.second, x, u);
      Q_xx += L_xx_relaxed;
      Q_uu += L_uu_relaxed;
      Q_ux += L_ux_relaxed;
    }

    // Regularization
    Eigen::MatrixXd Q_uu_reg = Q_uu;
    Q_uu_reg.diagonal().array() += context.regularization_;
    Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

    Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
    if (ldlt.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "LogDDP: Backward pass failed at time " << t << std::endl;
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

    // Compute value function approximation
    Eigen::Vector2d dV_step;
    dV_step << Q_u.dot(k_u), 0.5 * k_u.dot(Q_uu * k_u);
    dV_ = dV_ + dV_step;
    V_x = Q_x + K_u.transpose() * Q_uu * k_u + Q_ux.transpose() * k_u +
          K_u.transpose() * Q_u;
    V_xx = Q_xx + K_u.transpose() * Q_uu * K_u + Q_ux.transpose() * K_u +
           K_u.transpose() * Q_ux;
    V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize Hessian

    // Compute optimality gap (Inf-norm) for convergence check
    Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
  }
  context.inf_du_ = Qu_err;

  if (options.debug) {
    std::cout << "[LogDDP Backward Pass]\n"
              << "    Qu_err:  " << std::scientific << std::setprecision(4)
              << Qu_err << "\n"
              << "    rf_err:  " << std::scientific << std::setprecision(4)
              << constraint_violation_ << "\n"
              << "    dV:      " << std::scientific << std::setprecision(4)
              << dV_.transpose() << std::endl;
  }

  return true;
}

// === Forward pass ===

ForwardPassResult LogDDPSolver::forwardPass(CDDP &context, double alpha) {
  const CDDPOptions &options = context.getOptions();
  const auto &constraint_set = context.getConstraintSet();

  ForwardPassResult result;
  result.success = false;
  result.cost = std::numeric_limits<double>::infinity();
  result.merit_function = std::numeric_limits<double>::infinity();
  result.alpha_pr = alpha;

  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();

  // Initialize trajectories
  result.state_trajectory = context.X_;
  result.control_trajectory = context.U_;
  result.state_trajectory[0] = context.getInitialState();

  double cost_new = 0.0;
  double merit_function_new = 0.0;
  double rp_err = 0.0;

  // Single-shooting rollout
  for (int t = 0; t < horizon; ++t) {
    const Eigen::VectorXd delta_x_t =
        result.state_trajectory[t] - context.X_[t];

    result.control_trajectory[t] =
        context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x_t;

    result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
        result.state_trajectory[t], result.control_trajectory[t],
        t * context.getTimestep());

    if (!result.state_trajectory[t + 1].allFinite() ||
        !result.control_trajectory[t].allFinite()) {
      if (options.debug) {
        std::cerr
            << "[LogDDP Forward Pass] NaN/Inf detected during rollout at t="
            << t << " for alpha=" << alpha << std::endl;
      }
      result.success = false;
      return result;
    }
  }

  // Cost computation and constraint evaluation
  for (int t = 0; t < horizon; ++t) {
    cost_new += context.getObjective().running_cost(
        result.state_trajectory[t], result.control_trajectory[t], t);

    for (const auto &constraint_pair : constraint_set) {
      Eigen::VectorXd g_t =
          constraint_pair.second->evaluate(result.state_trajectory[t],
                                           result.control_trajectory[t]) -
          constraint_pair.second->getUpperBound();
      merit_function_new += relaxed_log_barrier_->evaluate(
          *constraint_pair.second, result.state_trajectory[t],
          result.control_trajectory[t]);

      for (int i = 0; i < g_t.size(); ++i) {
        if (g_t(i) > 0.0) {
          rp_err += g_t(i);
        }
      }
    }
  }

  cost_new +=
      context.getObjective().terminal_cost(result.state_trajectory.back());
  merit_function_new += cost_new;

  // Filter-based acceptance
  double constraint_violation_old = constraint_violation_;
  double constraint_violation_new = rp_err;
  double merit_function_old = context.merit_function_;
  bool filter_acceptance = false;
  double expected_improvement = alpha * dV_(0);

  const auto &filter_opts = options.filter;

  if (constraint_violation_new > filter_opts.max_violation_threshold) {
    if (constraint_violation_new <
        (1.0 - filter_opts.violation_acceptance_threshold) *
            constraint_violation_old) {
      filter_acceptance = true;
    }
  } else if (std::max(constraint_violation_new, constraint_violation_old) <
                 filter_opts.min_violation_for_armijo_check &&
             expected_improvement < 0) {
    if (merit_function_new < merit_function_old + filter_opts.armijo_constant *
                                                      expected_improvement) {
      filter_acceptance = true;
    }
  } else {
    if (merit_function_new <
            merit_function_old - filter_opts.merit_acceptance_threshold *
                                     constraint_violation_old ||
        constraint_violation_new <
            (1.0 - filter_opts.violation_acceptance_threshold) *
                constraint_violation_old) {
      filter_acceptance = true;
    }
  }

  if (filter_acceptance) {
    result.success = true;
    result.cost = cost_new;
    result.merit_function = merit_function_new;
    result.constraint_violation = constraint_violation_new;
  }

  return result;
}

} // namespace cddp
