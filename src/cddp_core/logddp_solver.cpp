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
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace cddp {

LogDDPSolver::LogDDPSolver()
    : mu_(1e-1), relaxation_delta_(1e-5), constraint_violation_(0.0) {}

void LogDDPSolver::initialize(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int horizon = context.getHorizon();
  const int control_dim = context.getControlDim();
  const int state_dim = context.getStateDim();

  if ((context.getReferenceState() - context.getObjective().getReferenceState())
          .norm() > 1e-6) {
    std::cerr << "LogDDP: Initial state and goal state in the objective "
                 "function do not match"
              << std::endl;
    throw std::runtime_error(
        "Initial state and goal state in the objective function do not match");
  }

  if (options.log_barrier.segment_length != horizon) {
    throw std::runtime_error(
        "LogDDP: segment_length must equal horizon in single-shooting mode");
  }
  if (options.log_barrier.rollout_type != "nonlinear") {
    throw std::runtime_error(
        "LogDDP: rollout_type must be 'nonlinear' in single-shooting mode");
  }

  auto initialize_line_search = [&]() {
    context.alphas_.clear();
    double alpha = options.line_search.initial_step_size;
    for (int i = 0; i < options.line_search.max_iterations; ++i) {
      context.alphas_.push_back(alpha);
      alpha *= options.line_search.step_reduction_factor;
    }
    context.alpha_pr_ = options.line_search.initial_step_size;
    dV_ = Eigen::Vector2d::Zero();
  };

  auto initialize_barrier = [&]() {
    mu_ = options.log_barrier.barrier.mu_initial;
    relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
    if (!relaxed_log_barrier_) {
      relaxed_log_barrier_ =
          std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);
    } else {
      relaxed_log_barrier_->setBarrierCoeff(mu_);
      relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
    }
  };

  if (options.warm_start) {
    bool valid_warm_start =
        (k_u_.size() == static_cast<size_t>(horizon) &&
         K_u_.size() == static_cast<size_t>(horizon) &&
         context.U_.size() == static_cast<size_t>(horizon));

    if (valid_warm_start && !k_u_.empty()) {
      for (int t = 0; t < horizon; ++t) {
        if (k_u_[t].size() != control_dim || K_u_[t].rows() != control_dim ||
            K_u_[t].cols() != state_dim || context.U_[t].size() != control_dim) {
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
      context.regularization_ = options.regularization.initial_value;
      initialize_line_search();
      initialize_barrier();
      evaluateTrajectory(context);
      resetFilter(context);
      return;
    }

    if (options.verbose) {
      std::cout << "LogDDP: Warning - warm start requested but no valid solver "
                   "state found. Falling back to cold start initialization."
                << std::endl;
    }
  }

  initializeGains(horizon, control_dim, state_dim);

  context.U_.resize(horizon);
  for (int t = 0; t < horizon; ++t) {
    if (context.U_[t].size() != control_dim) {
      context.U_[t] = Eigen::VectorXd::Zero(control_dim);
    }
  }

  context.regularization_ = options.regularization.initial_value;
  initialize_line_search();
  initialize_barrier();
  evaluateTrajectory(context);
  resetFilter(context);
}

std::string LogDDPSolver::getSolverName() const { return "LogDDP"; }

void LogDDPSolver::preIterationSetup(CDDP &context) {
  evaluateTrajectory(context);
  resetFilter(context);
}

bool LogDDPSolver::handleBackwardPassRegularizationLimit(
    CDDP & /*context*/, std::string &termination_reason) {
  termination_reason = "RegularizationLimitReached_Converged";
  return true;
}

void LogDDPSolver::applyForwardPassResult(CDDP &context,
                                          const ForwardPassResult &result) {
  CDDPSolverBase::applyForwardPassResult(context, result);
  constraint_violation_ = result.constraint_violation;
  context.inf_pr_ = result.constraint_violation;
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
  CDDPSolverBase::recordIterationHistory(context);
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
            << std::scientific << std::setprecision(4)
            << context.merit_function_ << " " << std::setw(10)
            << std::scientific << std::setprecision(2) << context.inf_du_
            << " " << std::setw(8) << std::fixed << std::setprecision(1)
            << std::log10(context.regularization_) << " " << std::setw(8)
            << std::fixed << std::setprecision(4) << context.alpha_pr_ << " "
            << std::setw(8) << std::fixed << std::setprecision(1)
            << std::log10(mu_) << " " << std::setw(10) << std::scientific
            << std::setprecision(2) << constraint_violation_ << std::endl;
}

void LogDDPSolver::evaluateTrajectory(CDDP &context) {
  const int horizon = context.getHorizon();

  context.X_.resize(horizon + 1);
  context.X_[0] = context.getInitialState();
  context.cost_ = 0.0;

  for (int t = 0; t < horizon; ++t) {
    const Eigen::VectorXd &x_t = context.X_[t];
    const Eigen::VectorXd &u_t = context.U_[t];

    context.cost_ += context.getObjective().running_cost(x_t, u_t, t);
    context.X_[t + 1] = context.getSystem().getDiscreteDynamics(
        x_t, u_t, t * context.getTimestep());
  }

  context.cost_ += context.getObjective().terminal_cost(context.X_.back());
}

void LogDDPSolver::resetFilter(CDDP &context) {
  context.merit_function_ =
      computeBarrierMerit(context, context.X_, context.U_, &constraint_violation_);
  context.inf_pr_ = constraint_violation_;
}

void LogDDPSolver::augmentRunningCostDerivatives(
    const CDDP &context, int t, const Eigen::VectorXd &x,
    const Eigen::VectorXd &u, Eigen::VectorXd &l_x, Eigen::VectorXd &l_u,
    Eigen::MatrixXd &l_xx, Eigen::MatrixXd &l_uu,
    Eigen::MatrixXd &l_ux) const {
  for (const auto &[name, constraint] : context.getConstraintSet()) {
    auto [L_x_relaxed, L_u_relaxed] =
        relaxed_log_barrier_->getGradients(*constraint, x, u, t);
    l_x += L_x_relaxed;
    l_u += L_u_relaxed;

    auto [L_xx_relaxed, L_uu_relaxed, L_ux_relaxed] =
        relaxed_log_barrier_->getHessians(*constraint, x, u, t);
    l_xx += L_xx_relaxed;
    l_uu += L_uu_relaxed;
    l_ux += L_ux_relaxed;
  }
}

void LogDDPSolver::augmentTerminalCostDerivatives(const CDDP &context,
                                                  const Eigen::VectorXd &x_N,
                                                  Eigen::VectorXd &V_x,
                                                  Eigen::MatrixXd &V_xx) const {
  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());
  const int horizon = context.getHorizon();

  for (const auto &[name, constraint] : context.getTerminalConstraintSet()) {
    auto [L_x_relaxed, L_u_relaxed] =
        relaxed_log_barrier_->getGradients(*constraint, x_N, dummy_u, horizon);
    (void)L_u_relaxed;
    V_x += L_x_relaxed;

    auto [L_xx_relaxed, L_uu_relaxed, L_ux_relaxed] =
        relaxed_log_barrier_->getHessians(*constraint, x_N, dummy_u, horizon);
    (void)L_uu_relaxed;
    (void)L_ux_relaxed;
    V_xx += L_xx_relaxed;
  }
}

double LogDDPSolver::computeBarrierMerit(
    const CDDP &context, const std::vector<Eigen::VectorXd> &X,
    const std::vector<Eigen::VectorXd> &U, double *max_constraint_violation) const {
  double merit = 0.0;
  double max_violation = 0.0;

  for (int t = 0; t < context.getHorizon(); ++t) {
    merit += context.getObjective().running_cost(X[t], U[t], t);

    for (const auto &[name, constraint] : context.getConstraintSet()) {
      merit += relaxed_log_barrier_->evaluate(*constraint, X[t], U[t], t);
      max_violation = std::max(
          max_violation, constraint->computeViolation(X[t], U[t], t));
    }
  }

  merit += context.getObjective().terminal_cost(X.back());

  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());
  for (const auto &[name, constraint] : context.getTerminalConstraintSet()) {
    merit +=
        relaxed_log_barrier_->evaluate(*constraint, X.back(), dummy_u, context.getHorizon());
    max_violation = std::max(max_violation, constraint->computeViolation(
                                                X.back(), dummy_u,
                                                context.getHorizon()));
  }

  if (max_constraint_violation != nullptr) {
    *max_constraint_violation = max_violation;
  }

  return merit;
}

bool LogDDPSolver::backwardPass(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();

  precomputeDynamicsDerivatives(context, 20);

  Eigen::VectorXd V_x =
      context.getObjective().getFinalCostGradient(context.X_.back());
  Eigen::MatrixXd V_xx =
      context.getObjective().getFinalCostHessian(context.X_.back());
  augmentTerminalCostDerivatives(context, context.X_.back(), V_x, V_xx);
  V_xx = 0.5 * (V_xx + V_xx.transpose());

  dV_ = Eigen::Vector2d::Zero();
  double Qu_err = 0.0;
  double norm_Vx = V_x.lpNorm<1>();

  for (int t = horizon - 1; t >= 0; --t) {
    const Eigen::VectorXd &x = context.X_[t];
    const Eigen::VectorXd &u = context.U_[t];
    const Eigen::MatrixXd &A = F_x_[t];
    const Eigen::MatrixXd &B = F_u_[t];

    auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
    auto [l_xx, l_uu, l_ux] =
        context.getObjective().getRunningCostHessians(x, u, t);
    augmentRunningCostDerivatives(context, t, x, u, l_x, l_u, l_xx, l_uu, l_ux);

    Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
    Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
    Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
    Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
    Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

    if (!options.use_ilqr) {
      const auto &Fxx = F_xx_[t];
      const auto &Fuu = F_uu_[t];
      const auto &Fux = F_ux_[t];
      for (int i = 0; i < state_dim; ++i) {
        Q_xx += V_x(i) * Fxx[i];
        Q_ux += V_x(i) * Fux[i];
        Q_uu += V_x(i) * Fuu[i];
      }
    }

    Eigen::MatrixXd Q_uu_reg = 0.5 * (Q_uu + Q_uu.transpose());
    Q_uu_reg.diagonal().array() += context.regularization_;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
    if (ldlt.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "LogDDP: Backward pass failed at time " << t << std::endl;
      }
      return false;
    }

    Eigen::MatrixXd rhs(control_dim, 1 + state_dim);
    rhs.col(0) = Q_u;
    rhs.rightCols(state_dim) = Q_ux;

    Eigen::MatrixXd gains = -ldlt.solve(rhs);
    const Eigen::VectorXd k_u = gains.col(0);
    const Eigen::MatrixXd K_u = gains.rightCols(state_dim);

    k_u_[t] = k_u;
    K_u_[t] = K_u;

    dV_(0) += Q_u.dot(k_u);
    dV_(1) += 0.5 * k_u.dot(Q_uu * k_u);

    V_x = Q_x + K_u.transpose() * Q_uu * k_u + Q_ux.transpose() * k_u +
          K_u.transpose() * Q_u;
    V_xx = Q_xx + K_u.transpose() * Q_uu * K_u + Q_ux.transpose() * K_u +
           K_u.transpose() * Q_ux;
    V_xx = 0.5 * (V_xx + V_xx.transpose());

    Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
    norm_Vx += V_x.lpNorm<1>();
  }

  double scaling_factor = options.termination_scaling_max_factor;
  scaling_factor = std::max(scaling_factor, norm_Vx / (horizon * state_dim)) /
                   scaling_factor;
  context.inf_du_ = Qu_err / scaling_factor;

  if (options.debug) {
    std::cout << "[LogDDP Backward Pass]\n"
              << "    Qu_err:  " << std::scientific << std::setprecision(4)
              << context.inf_du_ << "\n"
              << "    rf_err:  " << std::scientific << std::setprecision(4)
              << constraint_violation_ << "\n"
              << "    dV:      " << std::scientific << std::setprecision(4)
              << dV_.transpose() << std::endl;
  }

  return true;
}

ForwardPassResult LogDDPSolver::forwardPass(CDDP &context, double alpha) {
  const CDDPOptions &options = context.getOptions();

  ForwardPassResult result;
  result.success = false;
  result.cost = std::numeric_limits<double>::infinity();
  result.merit_function = std::numeric_limits<double>::infinity();
  result.alpha_pr = alpha;

  result.state_trajectory = context.X_;
  result.control_trajectory = context.U_;
  result.state_trajectory[0] = context.getInitialState();

  double cost_new = 0.0;

  for (int t = 0; t < context.getHorizon(); ++t) {
    const Eigen::VectorXd delta_x_t = result.state_trajectory[t] - context.X_[t];
    result.control_trajectory[t] =
        context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x_t;

    if (!result.control_trajectory[t].allFinite()) {
      return result;
    }

    cost_new += context.getObjective().running_cost(result.state_trajectory[t],
                                                    result.control_trajectory[t], t);
    result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
        result.state_trajectory[t], result.control_trajectory[t],
        t * context.getTimestep());

    if (!result.state_trajectory[t + 1].allFinite()) {
      if (options.debug) {
        std::cerr
            << "[LogDDP Forward Pass] NaN/Inf detected during rollout at t="
            << t << " for alpha=" << alpha << std::endl;
      }
      return result;
    }
  }

  cost_new += context.getObjective().terminal_cost(result.state_trajectory.back());

  double constraint_violation_new = 0.0;
  double merit_function_new = computeBarrierMerit(
      context, result.state_trajectory, result.control_trajectory,
      &constraint_violation_new);

  const double constraint_violation_old = constraint_violation_;
  const double merit_function_old = context.merit_function_;
  const double expected_improvement = alpha * dV_(0);
  const auto &filter_opts = options.filter;

  bool filter_acceptance = false;
  if (constraint_violation_new > filter_opts.max_violation_threshold) {
    if (constraint_violation_new <
        (1.0 - filter_opts.violation_acceptance_threshold) *
            constraint_violation_old) {
      filter_acceptance = true;
    }
  } else if (std::max(constraint_violation_new, constraint_violation_old) <
                 filter_opts.min_violation_for_armijo_check &&
             expected_improvement < 0.0) {
    if (merit_function_new < merit_function_old +
                                 filter_opts.armijo_constant *
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
