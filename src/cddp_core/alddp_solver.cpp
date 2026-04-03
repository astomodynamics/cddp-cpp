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

#include "cddp_core/alddp_solver.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/cddp_core.hpp"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace cddp {

// ============================================================================
// Construction / Initialization
// ============================================================================

ALDDPSolver::ALDDPSolver() : boxqp_solver_(BoxQPOptions()) {}

std::string ALDDPSolver::getSolverName() const { return "ALDDP"; }

void ALDDPSolver::initialize(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const auto &al_opts = options.alddp;
  const int horizon = context.getHorizon();
  const int control_dim = context.getControlDim();
  const int state_dim = context.getStateDim();

  // Warm start validation
  if (options.warm_start) {
    bool valid = (k_u_.size() == static_cast<size_t>(horizon) &&
                  K_u_.size() == static_cast<size_t>(horizon));
    if (valid && !k_u_.empty()) {
      for (int t = 0; t < horizon; ++t) {
        if (k_u_[t].size() != control_dim || K_u_[t].rows() != control_dim ||
            K_u_[t].cols() != state_dim) {
          valid = false;
          break;
        }
      }
    } else {
      valid = false;
    }

    if (valid) {
      if (options.verbose) {
        std::cout << "ALDDP: Using warm start with existing solver state"
                  << std::endl;
      }
      boxqp_solver_.setOptions(options.box_qp);
      // The solver state (gains, multipliers, penalties) is reused, but the
      // current seeded trajectory may have changed since the last solve.
      // Refresh slack controls so warm starts remain consistent with the
      // current X/U pair, including dynamically infeasible reseeds.
      initializeSlackControls(context);
      if (!context.X_.empty() && !context.U_.empty()) {
        computeCost(context);
      }
      return;
    } else if (options.verbose) {
      std::cout << "ALDDP: Warning - warm start requested but no valid "
                   "solver state found. Falling back to cold start."
                << std::endl;
    }
  }

  // Cold start
  initializeGains(horizon, control_dim, state_dim);
  boxqp_solver_ = BoxQPSolver(options.box_qp);

  // Initialize AL state
  al_outer_iter_ = 0;
  inner_tolerance_ = al_opts.inner_tolerance_init;
  max_constraint_violation_ = 0.0;
  prev_max_constraint_violation_ = std::numeric_limits<double>::infinity();
  prev_outer_cost_ = std::numeric_limits<double>::infinity();
  inner_converged_ = false;
  inner_iter_count_ = 0;
  current_slack_penalty_ = al_opts.slack_penalty;

  // Initialize multipliers and penalties
  initializeMultipliersAndPenalties(context);

  // Initialize slack controls for infeasible start
  initializeSlackControls(context);

  // Initialize sqrt backward pass workspace
  if (al_opts.use_sqrt_backward_pass) {
    S_chol_.resize(horizon + 1);
    p_.resize(horizon + 1);
    for (int t = 0; t <= horizon; ++t) {
      S_chol_[t] = Eigen::MatrixXd::Zero(state_dim, state_dim);
      p_[t] = Eigen::VectorXd::Zero(state_dim);
    }
  }

  // Warn if max_iterations may be too low
  int effective_budget = al_opts.max_inner_iterations * al_opts.max_outer_iterations;
  if (options.max_iterations < effective_budget && options.verbose) {
    std::cerr << "ALDDP: Warning - max_iterations (" << options.max_iterations
              << ") may be too low for " << al_opts.max_outer_iterations
              << " outer x " << al_opts.max_inner_iterations
              << " inner iterations (= " << effective_budget << ")"
              << std::endl;
  }

  if (!context.X_.empty() && !context.U_.empty()) {
    computeCost(context);
  }
}

void ALDDPSolver::initializeMultipliersAndPenalties(CDDP &context) {
  const auto &al_opts = context.getOptions().alddp;
  const int horizon = context.getHorizon();

  lambda_.clear();
  penalty_.clear();
  G_.clear();
  G_x_.clear();
  G_u_.clear();

  for (const auto &[name, constraint] : context.getConstraintSet()) {
    // Skip ControlConstraint if handled by BoxQP
    if (al_opts.use_boxqp_for_controls && name == "ControlConstraint") {
      continue;
    }
    int dual_dim = constraint->getDualDim();
    if (dual_dim == 0) {
      dual_dim = constraint->getUpperBound().size();
    }

    lambda_[name].resize(horizon);
    penalty_[name].resize(horizon);
    G_[name].resize(horizon);
    G_x_[name].resize(horizon);
    G_u_[name].resize(horizon);

    for (int t = 0; t < horizon; ++t) {
      lambda_[name][t] = Eigen::VectorXd::Constant(dual_dim, al_opts.lambda_init);
      penalty_[name][t] = Eigen::VectorXd::Constant(dual_dim, al_opts.penalty_init);
    }
  }

  // Terminal constraints
  terminal_lambda_.clear();
  terminal_penalty_.clear();
  G_terminal_.clear();

  for (const auto &[name, constraint] : context.getTerminalConstraintSet()) {
    int dual_dim = constraint->getDualDim();
    if (dual_dim == 0) {
      dual_dim = constraint->getUpperBound().size();
    }
    terminal_lambda_[name] = Eigen::VectorXd::Constant(dual_dim, al_opts.lambda_init);
    terminal_penalty_[name] = Eigen::VectorXd::Constant(dual_dim, al_opts.penalty_init);
  }
}

void ALDDPSolver::initializeSlackControls(CDDP &context) {
  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();

  S_.resize(horizon, Eigen::VectorXd::Zero(state_dim));
  k_s_.resize(horizon, Eigen::VectorXd::Zero(state_dim));
  K_s_.resize(horizon, Eigen::MatrixXd::Zero(state_dim, state_dim));

  // Check if the initial trajectory is dynamically infeasible
  double max_slack = 0.0;
  for (int t = 0; t < horizon; ++t) {
    Eigen::VectorXd x_next = context.getSystem().getDiscreteDynamics(
        context.X_[t], context.U_[t], t * context.getTimestep());
    S_[t] = context.X_[t + 1] - x_next;
    max_slack = std::max(max_slack, S_[t].lpNorm<Eigen::Infinity>());
  }

  infeasible_start_ = (max_slack > 1e-8);
  if (infeasible_start_ && context.getOptions().verbose) {
    std::cout << "ALDDP: Infeasible start detected (max slack = "
              << max_slack << "). Using slack controls." << std::endl;
  }
}

// ============================================================================
// Constraint Evaluation
// ============================================================================

void ALDDPSolver::evaluateConstraints(CDDP &context) {
  const int horizon = context.getHorizon();

  // Reset max violation tracker
  max_constraint_violation_ = 0.0;

  for (auto &[name, G_vec] : G_) {
    const auto &constraint = context.getConstraintSet().at(name);
    for (int t = 0; t < horizon; ++t) {
      G_vec[t] = constraint->evaluate(context.X_[t], context.U_[t], t);
      G_x_[name][t] = constraint->getStateJacobian(context.X_[t], context.U_[t], t);
      G_u_[name][t] = constraint->getControlJacobian(context.X_[t], context.U_[t], t);

      // Track max violation using the constraint's own method
      double viol = constraint->computeViolation(context.X_[t], context.U_[t], t);
      max_constraint_violation_ = std::max(max_constraint_violation_, viol);
    }
  }

  // Terminal constraints
  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());
  for (auto &[name, G_term] : G_terminal_) {
    const auto &constraint = context.getTerminalConstraintSet().at(name);
    G_term = constraint->evaluate(context.X_.back(), dummy_u, horizon);

    double viol = constraint->computeViolation(context.X_.back(), dummy_u, horizon);
    max_constraint_violation_ = std::max(max_constraint_violation_, viol);
  }
}

double ALDDPSolver::computeMaxConstraintViolation() const {
  // max_constraint_violation_ is computed during evaluateConstraints()
  return max_constraint_violation_;
}

double ALDDPSolver::computeMaxSlackNorm() const {
  double max_norm = 0.0;
  for (const auto &s : S_) {
    max_norm = std::max(max_norm, s.lpNorm<Eigen::Infinity>());
  }
  return max_norm;
}

// ============================================================================
// AL Cost Derivative Augmentation
// ============================================================================

void ALDDPSolver::augmentRunningCostDerivatives(
    const CDDP &context, int t,
    const Eigen::VectorXd &x, const Eigen::VectorXd &u,
    Eigen::VectorXd &l_x, Eigen::VectorXd &l_u,
    Eigen::MatrixXd &l_xx, Eigen::MatrixXd &l_uu,
    Eigen::MatrixXd &l_ux) const {

  for (const auto &[name, G_vec] : G_) {
    const auto &constraint = context.getConstraintSet().at(name);
    const Eigen::VectorXd &g = G_vec[t];
    const Eigen::VectorXd &upper = constraint->getUpperBound();
    const Eigen::VectorXd &lower = constraint->getLowerBound();
    const Eigen::VectorXd &lam = lambda_.at(name)[t];
    const Eigen::VectorXd &pen = penalty_.at(name)[t];
    const Eigen::MatrixXd &C_x = G_x_.at(name)[t];
    const Eigen::MatrixXd &C_u = G_u_.at(name)[t];

    const int dim = g.size();
    for (int i = 0; i < dim; ++i) {
      // Violation: c_i = g_i - upper_i (positive means violated)
      double c_i = g(i) - upper(i);

      // Determine if this component is active (I_mu masking, Eq. 3)
      bool is_equality = (std::abs(lower(i) - upper(i)) < 1e-12);
      bool is_active = is_equality || (c_i >= 0.0) || (lam(i) > 0.0);

      if (!is_active) continue;

      double effective_lambda = lam(i) + pen(i) * c_i;

      // First-order augmentation
      l_x += effective_lambda * C_x.row(i).transpose();
      l_u += effective_lambda * C_u.row(i).transpose();

      // Second-order augmentation (Gauss-Newton approximation)
      l_xx += pen(i) * C_x.row(i).transpose() * C_x.row(i);
      l_uu += pen(i) * C_u.row(i).transpose() * C_u.row(i);
      l_ux += pen(i) * C_u.row(i).transpose() * C_x.row(i);
    }
  }
}

void ALDDPSolver::augmentTerminalCostDerivatives(
    const CDDP &context, const Eigen::VectorXd &x_N,
    Eigen::VectorXd &V_x, Eigen::MatrixXd &V_xx) const {

  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());

  for (const auto &[name, g_term] : G_terminal_) {
    const auto &constraint = context.getTerminalConstraintSet().at(name);
    const Eigen::VectorXd &upper = constraint->getUpperBound();
    const Eigen::VectorXd &lower = constraint->getLowerBound();
    const Eigen::VectorXd &lam = terminal_lambda_.at(name);
    const Eigen::VectorXd &pen = terminal_penalty_.at(name);
    const Eigen::MatrixXd C_x =
        constraint->getStateJacobian(x_N, dummy_u, context.getHorizon());

    const int dim = g_term.size();
    for (int i = 0; i < dim; ++i) {
      double c_i = g_term(i) - upper(i);
      bool is_equality = (std::abs(lower(i) - upper(i)) < 1e-12);
      bool is_active = is_equality || (c_i >= 0.0) || (lam(i) > 0.0);

      if (!is_active) continue;

      double effective_lambda = lam(i) + pen(i) * c_i;
      V_x += effective_lambda * C_x.row(i).transpose();
      V_xx += pen(i) * C_x.row(i).transpose() * C_x.row(i);
    }
  }
}

// ============================================================================
// AL Merit Function (thread-safe, no member writes)
// ============================================================================

double ALDDPSolver::computeALMerit(
    const CDDP &context,
    const std::vector<Eigen::VectorXd> &X,
    const std::vector<Eigen::VectorXd> &U,
    const std::vector<Eigen::VectorXd> *S_trial) const {

  const auto &al_opts = context.getOptions().alddp;

  // Base cost
  double merit = 0.0;
  for (int t = 0; t < context.getHorizon(); ++t) {
    merit += context.getObjective().running_cost(X[t], U[t], t);
  }
  merit += context.getObjective().terminal_cost(X.back());

  // Slack penalty
  if (S_trial != nullptr) {
    for (int t = 0; t < context.getHorizon(); ++t) {
      merit += 0.5 * current_slack_penalty_ * (*S_trial)[t].squaredNorm();
    }
  }

  // AL penalty terms for path constraints
  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());
  for (const auto &[name, lam_vec] : lambda_) {
    const auto &constraint = context.getConstraintSet().at(name);
    const Eigen::VectorXd &upper = constraint->getUpperBound();
    const Eigen::VectorXd &lower = constraint->getLowerBound();

    for (int t = 0; t < context.getHorizon(); ++t) {
      Eigen::VectorXd g = constraint->evaluate(X[t], U[t], t);
      const Eigen::VectorXd &lam = lam_vec[t];
      const Eigen::VectorXd &pen = penalty_.at(name)[t];

      for (int i = 0; i < g.size(); ++i) {
        double c_i = g(i) - upper(i);
        bool is_equality = (std::abs(lower(i) - upper(i)) < 1e-12);
        bool is_active = is_equality || (c_i >= 0.0) || (lam(i) > 0.0);
        if (is_active) {
          merit += lam(i) * c_i + 0.5 * pen(i) * c_i * c_i;
        }
      }
    }
  }

  // AL penalty terms for terminal constraints
  for (const auto &[name, lam] : terminal_lambda_) {
    const auto &constraint = context.getTerminalConstraintSet().at(name);
    const Eigen::VectorXd &upper = constraint->getUpperBound();
    const Eigen::VectorXd &lower = constraint->getLowerBound();
    Eigen::VectorXd g = constraint->evaluate(X.back(), dummy_u, context.getHorizon());
    const Eigen::VectorXd &pen = terminal_penalty_.at(name);

    for (int i = 0; i < g.size(); ++i) {
      double c_i = g(i) - upper(i);
      bool is_equality = (std::abs(lower(i) - upper(i)) < 1e-12);
      bool is_active = is_equality || (c_i >= 0.0) || (lam(i) > 0.0);
      if (is_active) {
        merit += lam(i) * c_i + 0.5 * pen(i) * c_i * c_i;
      }
    }
  }

  return merit;
}

// ============================================================================
// Backward Pass
// ============================================================================

bool ALDDPSolver::backwardPass(CDDP &context) {
  if (context.getOptions().alddp.use_sqrt_backward_pass) {
    return backwardPassSqrt(context);
  }
  return backwardPassStandard(context);
}

bool ALDDPSolver::backwardPassStandard(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const auto &al_opts = options.alddp;
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();

  auto control_constraint =
      al_opts.use_boxqp_for_controls
          ? context.getConstraint<ControlConstraint>("ControlConstraint")
          : nullptr;

  // Terminal cost derivatives with AL augmentation
  Eigen::VectorXd V_x =
      context.getObjective().getFinalCostGradient(context.X_.back());
  Eigen::MatrixXd V_xx =
      context.getObjective().getFinalCostHessian(context.X_.back());

  augmentTerminalCostDerivatives(context, context.X_.back(), V_x, V_xx);

  // Pre-allocate
  Eigen::MatrixXd A(state_dim, state_dim);
  Eigen::MatrixXd B(state_dim, control_dim);
  Eigen::VectorXd Q_x(state_dim), Q_u(control_dim);
  Eigen::MatrixXd Q_xx(state_dim, state_dim);
  Eigen::MatrixXd Q_uu(control_dim, control_dim);
  Eigen::MatrixXd Q_uu_reg(control_dim, control_dim);
  Eigen::MatrixXd Q_ux(control_dim, state_dim);
  Eigen::VectorXd k(control_dim);
  Eigen::MatrixXd K(control_dim, state_dim);

  dV_ = Eigen::Vector2d::Zero();
  double norm_Vx = V_x.lpNorm<1>();
  double Qu_error = 0.0;

  for (int t = horizon - 1; t >= 0; --t) {
    const Eigen::VectorXd &x = context.X_[t];
    const Eigen::VectorXd &u = context.U_[t];

    const auto [Fx, Fu] =
        context.getSystem().getJacobians(x, u, t * context.getTimestep());

    A = context.getTimestep() * Fx;
    A.diagonal().array() += 1.0;
    B = context.getTimestep() * Fu;

    auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
    auto [l_xx, l_uu, l_ux] =
        context.getObjective().getRunningCostHessians(x, u, t);

    // Augment with AL constraint penalty terms
    augmentRunningCostDerivatives(context, t, x, u, l_x, l_u, l_xx, l_uu, l_ux);

    // Slack contribution to V_x through dynamics: x_{t+1} = f(x,u) + s
    // The slack penalty 0.5 * R_s * ||s||^2 contributes to cost-to-go
    if (infeasible_start_) {
      // Slack penalty gradient: R_s * s_t contributes via the identity Jacobian
      // Q_s = V_x' + R_s * s_t  (handled below after V_x' is available)
      // For the standard backward pass, we incorporate slacks after computing V_x, V_xx
    }

    Q_x = l_x + A.transpose() * V_x;
    Q_u = l_u + B.transpose() * V_x;
    Q_xx = l_xx + A.transpose() * V_xx * A;
    Q_ux = l_ux + B.transpose() * V_xx * A;
    Q_uu = l_uu + B.transpose() * V_xx * B;

    Q_uu_reg = Q_uu;
    Q_uu_reg.diagonal().array() += context.regularization_;

    // Check positive definiteness
    Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
    if (es.eigenvalues().real().minCoeff() <= 0) {
      if (options.debug) {
        std::cerr << "ALDDP: Q_uu is not positive definite at time " << t
                  << std::endl;
      }
      return false;
    }

    // Solve for control gains
    if (control_constraint == nullptr) {
      const Eigen::MatrixXd H = Q_uu_reg.inverse();
      k = -H * Q_u;
      K = -H * Q_ux;
    } else {
      const Eigen::VectorXd lb = control_constraint->rawLowerBound() - u;
      const Eigen::VectorXd ub = control_constraint->rawUpperBound() - u;
      const Eigen::VectorXd x0 = k_u_[t];

      BoxQPResult qp_result = boxqp_solver_.solve(Q_uu_reg, Q_u, lb, ub, x0);

      if (qp_result.status == BoxQPStatus::HESSIAN_NOT_PD ||
          qp_result.status == BoxQPStatus::NO_DESCENT) {
        if (options.debug) {
          std::cerr << "ALDDP: BoxQP failed at time step " << t << std::endl;
        }
        return false;
      }

      k = qp_result.x;
      K = Eigen::MatrixXd::Zero(control_dim, state_dim);
      if (qp_result.free.sum() > 0) {
        std::vector<int> free_idx;
        for (int i = 0; i < control_dim; i++) {
          if (qp_result.free(i)) {
            free_idx.push_back(i);
          }
        }
        Eigen::MatrixXd Q_ux_free(free_idx.size(), state_dim);
        for (size_t i = 0; i < free_idx.size(); i++) {
          Q_ux_free.row(i) = Q_ux.row(free_idx[i]);
        }
        Eigen::MatrixXd K_free = -qp_result.Hfree.solve(Q_ux_free);
        for (size_t i = 0; i < free_idx.size(); i++) {
          K.row(free_idx[i]) = K_free.row(i);
        }
      }
    }

    k_u_[t] = k;
    K_u_[t] = K;

    // Solve for slack gains (decoupled, ignoring Q_us/Q_su coupling)
    if (infeasible_start_) {
      // Q_ss = R_s*I + V_xx (from next step, V_xx is current since we're going backward)
      Eigen::MatrixXd Q_ss = current_slack_penalty_ * Eigen::MatrixXd::Identity(state_dim, state_dim) + V_xx;
      Eigen::VectorXd Q_s = V_x + current_slack_penalty_ * S_[t];
      Eigen::MatrixXd Q_sx = V_xx * A;

      // Regularize Q_ss
      Q_ss.diagonal().array() += context.regularization_;

      Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_ss);
      k_s_[t] = -ldlt.solve(Q_s);
      K_s_[t] = -ldlt.solve(Q_sx);
    }

    // Expected cost change
    Eigen::Vector2d dV_step;
    dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
    dV_ += dV_step;

    // Add slack contribution to expected cost change
    if (infeasible_start_) {
      Eigen::MatrixXd Q_ss_unreg = current_slack_penalty_ * Eigen::MatrixXd::Identity(state_dim, state_dim) + V_xx;
      Eigen::VectorXd Q_s = V_x + current_slack_penalty_ * S_[t];
      dV_(0) += Q_s.dot(k_s_[t]);
      dV_(1) += 0.5 * k_s_[t].dot(Q_ss_unreg * k_s_[t]);
    }

    // Update cost-to-go
    V_x = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k +
          K.transpose() * Q_u;
    V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K +
           K.transpose() * Q_ux;
    V_xx = 0.5 * (V_xx + V_xx.transpose());

    norm_Vx += V_x.lpNorm<1>();
    Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());
  }

  double scaling_factor = options.termination_scaling_max_factor;
  scaling_factor = std::max(scaling_factor, norm_Vx / (horizon * state_dim)) /
                   scaling_factor;
  context.inf_du_ = Qu_error / scaling_factor;

  return true;
}

bool ALDDPSolver::backwardPassSqrt(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const auto &al_opts = options.alddp;
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();

  auto control_constraint =
      al_opts.use_boxqp_for_controls
          ? context.getConstraint<ControlConstraint>("ControlConstraint")
          : nullptr;

  // Terminal cost derivatives
  Eigen::VectorXd V_x =
      context.getObjective().getFinalCostGradient(context.X_.back());
  Eigen::MatrixXd V_xx =
      context.getObjective().getFinalCostHessian(context.X_.back());

  augmentTerminalCostDerivatives(context, context.X_.back(), V_x, V_xx);

  // Regularize terminal Hessian before Cholesky
  V_xx.diagonal().array() += 1e-8;

  // Compute Cholesky factor of terminal cost-to-go Hessian: S_N = √P_N
  Eigen::LLT<Eigen::MatrixXd> llt_terminal(V_xx);
  if (llt_terminal.info() != Eigen::Success) {
    if (options.debug) {
      std::cerr << "ALDDP sqrt: Terminal Hessian Cholesky failed" << std::endl;
    }
    return false;
  }
  S_chol_[horizon] = llt_terminal.matrixU(); // upper triangular
  p_[horizon] = V_x;

  dV_ = Eigen::Vector2d::Zero();
  double norm_Vx = V_x.lpNorm<1>();
  double Qu_error = 0.0;

  for (int t = horizon - 1; t >= 0; --t) {
    const Eigen::VectorXd &x = context.X_[t];
    const Eigen::VectorXd &u = context.U_[t];

    const auto [Fx, Fu] =
        context.getSystem().getJacobians(x, u, t * context.getTimestep());

    Eigen::MatrixXd A = context.getTimestep() * Fx;
    A.diagonal().array() += 1.0;
    Eigen::MatrixXd B = context.getTimestep() * Fu;

    auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
    auto [l_xx, l_uu, l_ux] =
        context.getObjective().getRunningCostHessians(x, u, t);

    augmentRunningCostDerivatives(context, t, x, u, l_x, l_u, l_xx, l_uu, l_ux);

    const Eigen::MatrixXd &S_next = S_chol_[t + 1]; // upper triangular √P'
    const Eigen::VectorXd &p_next = p_[t + 1];

    // Q-function first-order terms
    Eigen::VectorXd Q_x = l_x + A.transpose() * p_next;
    Eigen::VectorXd Q_u = l_u + B.transpose() * p_next;

    // Compute Cholesky factors of running cost Hessians
    // Add regularization to ensure PD
    l_xx.diagonal().array() += context.regularization_;
    l_uu.diagonal().array() += context.regularization_;

    Eigen::LLT<Eigen::MatrixXd> llt_lxx(l_xx);
    Eigen::LLT<Eigen::MatrixXd> llt_luu(l_uu);

    if (llt_lxx.info() != Eigen::Success || llt_luu.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "ALDDP sqrt: Running cost Hessian Cholesky failed at t="
                  << t << std::endl;
      }
      return false;
    }

    Eigen::MatrixXd sqrt_lxx = llt_lxx.matrixU(); // upper triangular
    Eigen::MatrixXd sqrt_luu = llt_luu.matrixU();

    // Reconstruct V_xx' = S_next^T * S_next for explicit Q-function computation
    Eigen::MatrixXd V_xx_next = S_next.transpose() * S_next;

    // Build Q-function Hessians explicitly (use sqrt factors only for propagation)
    Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx_next * A;
    Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx_next * A;
    Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx_next * B;

    // Regularize Q_uu
    Eigen::MatrixXd Q_uu_reg = Q_uu;
    Q_uu_reg.diagonal().array() += context.regularization_;

    // Check positive definiteness
    Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
    if (es.eigenvalues().real().minCoeff() <= 0) {
      if (options.debug) {
        std::cerr << "ALDDP sqrt: Q_uu is not positive definite at time " << t
                  << std::endl;
      }
      return false;
    }

    // Compute Z_uu = cholesky(Q_uu_reg) for gain computation
    Eigen::LLT<Eigen::MatrixXd> llt_Quu(Q_uu_reg);
    if (llt_Quu.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "ALDDP sqrt: Q_uu Cholesky failed at t=" << t << std::endl;
      }
      return false;
    }
    Eigen::MatrixXd Z_uu = llt_Quu.matrixU(); // upper triangular

    // Gains via triangular solves (Eqs. 22-23): K = -Z_uu^{-T} Z_uu^{-1} Q_ux
    Eigen::VectorXd k_vec(control_dim);
    Eigen::MatrixXd K_mat(control_dim, state_dim);

    if (control_constraint == nullptr) {
      // d = -solve(Z_uu, solve(Z_uu^T, Q_u))
      Eigen::VectorXd tmp1 = Z_uu.transpose().triangularView<Eigen::Lower>().solve(Q_u);
      k_vec = -Z_uu.triangularView<Eigen::Upper>().solve(tmp1);

      Eigen::MatrixXd tmp2 = Z_uu.transpose().triangularView<Eigen::Lower>().solve(Q_ux);
      K_mat = -Z_uu.triangularView<Eigen::Upper>().solve(tmp2);
    } else {
      // Reconstruct Q_uu for BoxQP
      Eigen::MatrixXd Q_uu_full = Z_uu.transpose() * Z_uu;

      const Eigen::VectorXd lb = control_constraint->rawLowerBound() - u;
      const Eigen::VectorXd ub = control_constraint->rawUpperBound() - u;
      const Eigen::VectorXd x0 = k_u_[t];

      BoxQPResult qp_result = boxqp_solver_.solve(Q_uu_full, Q_u, lb, ub, x0);

      if (qp_result.status == BoxQPStatus::HESSIAN_NOT_PD ||
          qp_result.status == BoxQPStatus::NO_DESCENT) {
        if (options.debug) {
          std::cerr << "ALDDP sqrt: BoxQP failed at time step " << t << std::endl;
        }
        return false;
      }

      k_vec = qp_result.x;
      K_mat = Eigen::MatrixXd::Zero(control_dim, state_dim);
      if (qp_result.free.sum() > 0) {
        std::vector<int> free_idx;
        for (int i = 0; i < control_dim; i++) {
          if (qp_result.free(i)) free_idx.push_back(i);
        }
        Eigen::MatrixXd Q_ux_free(free_idx.size(), state_dim);
        for (size_t i = 0; i < free_idx.size(); i++) {
          Q_ux_free.row(i) = Q_ux.row(free_idx[i]);
        }
        Eigen::MatrixXd K_free = -qp_result.Hfree.solve(Q_ux_free);
        for (size_t i = 0; i < free_idx.size(); i++) {
          K_mat.row(free_idx[i]) = K_free.row(i);
        }
      }
    }

    k_u_[t] = k_vec;
    K_u_[t] = K_mat;

    // Slack gains (same as standard path)
    if (infeasible_start_) {
      Eigen::MatrixXd Q_ss = current_slack_penalty_ * Eigen::MatrixXd::Identity(state_dim, state_dim)
                            + S_next.transpose() * S_next;
      Eigen::VectorXd Q_s = p_next + current_slack_penalty_ * S_[t];
      Eigen::MatrixXd Q_sx = S_next.transpose() * S_next * A;
      Q_ss.diagonal().array() += context.regularization_;

      Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_ss);
      k_s_[t] = -ldlt.solve(Q_s);
      K_s_[t] = -ldlt.solve(Q_sx);
    }

    // Expected cost change (Eq. 25)
    Eigen::VectorXd Zuu_d = Z_uu * k_vec;
    dV_(0) += Q_u.dot(k_vec);
    dV_(1) += 0.5 * Zuu_d.dot(Zuu_d);

    if (infeasible_start_) {
      Eigen::MatrixXd Q_ss_unreg = current_slack_penalty_ * Eigen::MatrixXd::Identity(state_dim, state_dim)
                                  + S_next.transpose() * S_next;
      Eigen::VectorXd Q_s = p_next + current_slack_penalty_ * S_[t];
      dV_(0) += Q_s.dot(k_s_[t]);
      dV_(1) += 0.5 * k_s_[t].dot(Q_ss_unreg * k_s_[t]);
    }

    // Cost-to-go gradient update (standard Riccati)
    p_[t] = Q_x + K_mat.transpose() * Q_uu * k_vec + Q_ux.transpose() * k_vec +
            K_mat.transpose() * Q_u;

    // Cost-to-go Hessian update (standard Riccati, then take Cholesky for sqrt storage)
    Eigen::MatrixXd V_xx_new = Q_xx + K_mat.transpose() * Q_uu * K_mat +
                               Q_ux.transpose() * K_mat + K_mat.transpose() * Q_ux;
    V_xx_new = 0.5 * (V_xx_new + V_xx_new.transpose());
    V_xx_new.diagonal().array() += 1e-8;

    Eigen::LLT<Eigen::MatrixXd> llt_Vxx(V_xx_new);
    if (llt_Vxx.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "ALDDP sqrt: V_xx Cholesky failed at t=" << t << std::endl;
      }
      return false;
    }
    S_chol_[t] = llt_Vxx.matrixU();

    norm_Vx += p_[t].lpNorm<1>();
    Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());
  }

  double scaling_factor = options.termination_scaling_max_factor;
  scaling_factor = std::max(scaling_factor, norm_Vx / (horizon * state_dim)) /
                   scaling_factor;
  context.inf_du_ = Qu_error / scaling_factor;

  return true;
}

// ============================================================================
// Forward Pass
// ============================================================================

ForwardPassResult ALDDPSolver::forwardPass(CDDP &context, double alpha_pr) {
  ForwardPassResult result;
  result.success = false;
  result.cost = std::numeric_limits<double>::infinity();
  result.merit_function = std::numeric_limits<double>::infinity();
  result.alpha_pr = alpha_pr;

  result.state_trajectory = context.X_;
  result.control_trajectory = context.U_;
  result.state_trajectory[0] = context.getInitialState();

  const auto &al_opts = context.getOptions().alddp;
  auto control_constraint =
      al_opts.use_boxqp_for_controls
          ? context.getConstraint<ControlConstraint>("ControlConstraint")
          : nullptr;

  // Trial slack controls (local copy for thread safety)
  std::vector<Eigen::VectorXd> S_trial;
  if (infeasible_start_) {
    S_trial = S_;
  }

  double J_new = 0.0;

  for (int t = 0; t < context.getHorizon(); ++t) {
    const Eigen::VectorXd &x = result.state_trajectory[t];
    const Eigen::VectorXd delta_x = x - context.X_[t];

    result.control_trajectory[t] =
        result.control_trajectory[t] + alpha_pr * k_u_[t] + K_u_[t] * delta_x;

    if (control_constraint != nullptr) {
      result.control_trajectory[t] =
          control_constraint->clamp(result.control_trajectory[t]);
    }

    J_new += context.getObjective().running_cost(x, result.control_trajectory[t], t);

    // Dynamics with slack
    result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
        x, result.control_trajectory[t], t * context.getTimestep());

    if (infeasible_start_) {
      S_trial[t] = S_[t] + alpha_pr * k_s_[t] + K_s_[t] * delta_x;
      result.state_trajectory[t + 1] += S_trial[t];
    }
  }

  J_new += context.getObjective().terminal_cost(result.state_trajectory.back());

  // Compute AL merit function (thread-safe, uses local S_trial)
  double merit_new = computeALMerit(
      context, result.state_trajectory, result.control_trajectory,
      infeasible_start_ ? &S_trial : nullptr);

  // Armijo condition on the AL merit function
  double current_merit = context.merit_function_;
  double dJ = current_merit - merit_new;
  double expected = -alpha_pr * (dV_(0) + 0.5 * alpha_pr * dV_(1));
  double reduction_ratio =
      expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

  result.success = reduction_ratio > context.getOptions().filter.armijo_constant;
  result.cost = J_new;
  result.merit_function = merit_new;

  // Compute constraint violation for this trial (using local evaluation)
  double max_viol = 0.0;
  for (const auto &[name, lam_vec] : lambda_) {
    const auto &constraint = context.getConstraintSet().at(name);
    for (int t = 0; t < context.getHorizon(); ++t) {
      double viol = constraint->computeViolation(
          result.state_trajectory[t], result.control_trajectory[t], t);
      max_viol = std::max(max_viol, viol);
    }
  }
  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());
  for (const auto &[name, lam] : terminal_lambda_) {
    const auto &constraint = context.getTerminalConstraintSet().at(name);
    double viol = constraint->computeViolation(
        result.state_trajectory.back(), dummy_u, context.getHorizon());
    max_viol = std::max(max_viol, viol);
  }
  result.constraint_violation = max_viol;

  // Stash slack trajectory for applyForwardPassResult
  if (infeasible_start_ && result.success) {
    // Store in the optional dynamics_trajectory field
    result.dynamics_trajectory = S_trial;
  }

  return result;
}

void ALDDPSolver::applyForwardPassResult(CDDP &context,
                                          const ForwardPassResult &result) {
  // Apply base class updates (X_, U_, cost_, merit_function_, alpha_pr_)
  CDDPSolverBase::applyForwardPassResult(context, result);

  // Update merit to AL merit (not bare cost)
  context.merit_function_ = result.merit_function;

  // Update slack controls if infeasible start
  if (infeasible_start_ && result.dynamics_trajectory.has_value()) {
    S_ = result.dynamics_trajectory.value();
  }
}

// ============================================================================
// Iteration Setup and Convergence
// ============================================================================

void ALDDPSolver::preIterationSetup(CDDP &context) {
  evaluateConstraints(context);
  // max_constraint_violation_ is set by evaluateConstraints()

  // Set initial merit function to AL merit
  context.merit_function_ = computeALMerit(
      context, context.X_, context.U_,
      infeasible_start_ ? &S_ : nullptr);
}

bool ALDDPSolver::checkEarlyConvergence(CDDP &context, int /*iter*/,
                                         std::string &reason) {
  // Check if already fully converged after backward pass
  const auto &al_opts = context.getOptions().alddp;

  if (context.inf_du_ < al_opts.constraint_tolerance &&
      max_constraint_violation_ < al_opts.constraint_tolerance &&
      (!infeasible_start_ || computeMaxSlackNorm() < al_opts.constraint_tolerance)) {
    reason = "OptimalSolutionFound";
    return true;
  }
  return false;
}

bool ALDDPSolver::checkConvergence(CDDP &context, double dJ, double /*dL*/,
                                    int /*iter*/, std::string &reason) {
  const auto &al_opts = context.getOptions().alddp;

  // Check for AL outer iteration limit
  if (al_outer_iter_ >= al_opts.max_outer_iterations) {
    reason = "MaxOuterIterationsReached";
    return true;
  }

  // Check full convergence (outer)
  bool constraints_satisfied = max_constraint_violation_ < al_opts.constraint_tolerance;
  bool slacks_zero = !infeasible_start_ || computeMaxSlackNorm() < al_opts.constraint_tolerance;
  bool dual_converged = context.inf_du_ < al_opts.constraint_tolerance;

  if (constraints_satisfied && slacks_zero && dual_converged) {
    reason = "OptimalSolutionFound";
    return true;
  }

  // Check inner convergence
  inner_iter_count_++;
  bool inner_cost_converged = (dJ > 0.0 && dJ < context.getOptions().acceptable_tolerance);
  bool inner_du_converged = context.inf_du_ < inner_tolerance_;
  bool inner_iter_limit = inner_iter_count_ >= al_opts.max_inner_iterations;

  if (inner_cost_converged || inner_du_converged || inner_iter_limit) {
    inner_converged_ = true;
    // Return false so postIterationUpdate can handle the AL outer update
  }

  return false;
}

// ============================================================================
// AL Outer Loop (Multiplier/Penalty Updates)
// ============================================================================

void ALDDPSolver::postIterationUpdate(CDDP &context, bool /*forward_pass_success*/) {
  if (!inner_converged_) return;

  const auto &al_opts = context.getOptions().alddp;

  // Re-evaluate constraints at current trajectory
  evaluateConstraints(context);
  // max_constraint_violation_ is set by evaluateConstraints()
  double current_violation = max_constraint_violation_;

  // Update multipliers (Eq. 17)
  updateMultipliers(context);

  // Check if violation improved sufficiently
  if (current_violation > 0.5 * prev_max_constraint_violation_ || al_outer_iter_ == 0) {
    // Insufficient progress -- increase penalties
    updatePenalties(context);
  }

  // Increase slack penalty
  if (infeasible_start_ && computeMaxSlackNorm() > al_opts.constraint_tolerance) {
    current_slack_penalty_ *= al_opts.penalty_update_factor;
    current_slack_penalty_ = std::min(current_slack_penalty_, al_opts.penalty_max);
  }

  // Cost divergence check
  if (context.cost_ > 100.0 * prev_outer_cost_ && al_outer_iter_ > 0) {
    if (context.getOptions().verbose) {
      std::cerr << "ALDDP: Cost divergence detected (cost = " << context.cost_
                << ", prev = " << prev_outer_cost_ << ")" << std::endl;
    }
  }

  // Update state
  prev_max_constraint_violation_ = current_violation;
  prev_outer_cost_ = context.cost_;
  al_outer_iter_++;
  inner_converged_ = false;
  inner_iter_count_ = 0;

  // Tighten inner tolerance
  inner_tolerance_ = std::max(
      al_opts.inner_tolerance_min,
      inner_tolerance_ * al_opts.inner_tolerance_factor);

  // Reset regularization for new inner loop
  context.regularization_ = context.getOptions().regularization.initial_value;

  if (context.getOptions().verbose) {
    std::cout << "  ALDDP outer iter " << al_outer_iter_
              << ": violation=" << std::scientific << std::setprecision(2)
              << current_violation
              << " slack=" << computeMaxSlackNorm()
              << " inner_tol=" << inner_tolerance_
              << std::endl;
  }
}

void ALDDPSolver::updateMultipliers(CDDP &context) {
  // Path constraints (Eq. 17)
  for (auto &[name, lam_vec] : lambda_) {
    const auto &constraint = context.getConstraintSet().at(name);
    const Eigen::VectorXd &upper = constraint->getUpperBound();
    const Eigen::VectorXd &lower = constraint->getLowerBound();

    for (int t = 0; t < context.getHorizon(); ++t) {
      const Eigen::VectorXd &g = G_[name][t];
      for (int i = 0; i < g.size(); ++i) {
        double c_i = g(i) - upper(i);
        bool is_equality = (std::abs(lower(i) - upper(i)) < 1e-12);
        if (is_equality) {
          lam_vec[t](i) += penalty_[name][t](i) * c_i;
        } else {
          lam_vec[t](i) = std::max(0.0, lam_vec[t](i) + penalty_[name][t](i) * c_i);
        }
      }
    }
  }

  // Terminal constraints
  Eigen::VectorXd dummy_u = Eigen::VectorXd::Zero(context.getControlDim());
  for (auto &[name, lam] : terminal_lambda_) {
    const auto &constraint = context.getTerminalConstraintSet().at(name);
    const Eigen::VectorXd &upper = constraint->getUpperBound();
    const Eigen::VectorXd &lower = constraint->getLowerBound();
    const Eigen::VectorXd &g = G_terminal_[name];

    for (int i = 0; i < g.size(); ++i) {
      double c_i = g(i) - upper(i);
      bool is_equality = (std::abs(lower(i) - upper(i)) < 1e-12);
      if (is_equality) {
        lam(i) += terminal_penalty_[name](i) * c_i;
      } else {
        lam(i) = std::max(0.0, lam(i) + terminal_penalty_[name](i) * c_i);
      }
    }
  }
}

void ALDDPSolver::updatePenalties(CDDP &context) {
  const auto &al_opts = context.getOptions().alddp;

  for (auto &[name, pen_vec] : penalty_) {
    for (int t = 0; t < context.getHorizon(); ++t) {
      pen_vec[t] *= al_opts.penalty_update_factor;
      pen_vec[t] = pen_vec[t].cwiseMin(al_opts.penalty_max);
    }
  }

  for (auto &[name, pen] : terminal_penalty_) {
    pen *= al_opts.penalty_update_factor;
    pen = pen.cwiseMin(al_opts.penalty_max);
  }
}

// ============================================================================
// Regularization Limit Handler
// ============================================================================

bool ALDDPSolver::handleBackwardPassRegularizationLimit(
    CDDP &context, std::string &termination_reason) {
  const auto &al_opts = context.getOptions().alddp;

  // Perform AL outer update in-place since postIterationUpdate won't fire
  evaluateConstraints(context);
  double current_violation = computeMaxConstraintViolation();

  // Check if truly converged
  if (current_violation < al_opts.constraint_tolerance &&
      (!infeasible_start_ || computeMaxSlackNorm() < al_opts.constraint_tolerance)) {
    termination_reason = "OptimalSolutionFound";
    return true;
  }

  // Update multipliers and penalties
  updateMultipliers(context);
  if (current_violation > 0.5 * prev_max_constraint_violation_ || al_outer_iter_ == 0) {
    updatePenalties(context);
  }

  prev_max_constraint_violation_ = current_violation;
  max_constraint_violation_ = current_violation;
  al_outer_iter_++;

  if (al_outer_iter_ < al_opts.max_outer_iterations) {
    // Reset regularization and return true (early exit with current best)
    context.regularization_ = context.getOptions().regularization.initial_value;
    termination_reason = "RegularizationLimitReached";
    return true;
  }

  termination_reason = "MaxOuterIterationsReached";
  return false;
}

// ============================================================================
// History and Solution
// ============================================================================

void ALDDPSolver::recordIterationHistory(const CDDP &context) {
  CDDPSolverBase::recordIterationHistory(context);
  // AL-specific history could be added here
  history_.primal_infeasibility.push_back(max_constraint_violation_);
}

void ALDDPSolver::populateSolverSpecificSolution(CDDPSolution &solution,
                                                   const CDDP &context) {
  solution.final_primal_infeasibility = max_constraint_violation_;
  solution.final_dual_infeasibility = context.inf_du_;

  // Append AL info to status message
  std::string al_info = " (" + std::to_string(al_outer_iter_) + " AL outer iters)";
  solution.status_message += al_info;
}

// ============================================================================
// Printing
// ============================================================================

void ALDDPSolver::printIteration(int iter, const CDDP &context) const {
  if (iter == 0) {
    std::cout << std::setw(4) << "iter" << " " << std::setw(12) << "objective"
              << " " << std::setw(10) << "inf_du" << " " << std::setw(10)
              << "violation" << " " << std::setw(8) << "lg(rg)" << " "
              << std::setw(8) << "alpha" << " " << std::setw(5) << "outer"
              << std::endl;
  }

  std::cout << std::setw(4) << iter << " " << std::setw(12) << std::scientific
            << std::setprecision(4) << context.cost_ << " " << std::setw(10)
            << std::scientific << std::setprecision(2) << context.inf_du_
            << " " << std::setw(10) << std::scientific << std::setprecision(2)
            << max_constraint_violation_ << " " << std::setw(8) << std::fixed
            << std::setprecision(1) << std::log10(context.regularization_)
            << " " << std::setw(8) << std::fixed << std::setprecision(4)
            << context.alpha_pr_ << " " << std::setw(5) << al_outer_iter_
            << std::endl;
}

void ALDDPSolver::printSolutionSummary(const CDDPSolution &solution) const {
  std::cout << "\n=== ALDDP Solution Summary ===" << std::endl;
  std::cout << "Status: " << solution.status_message << std::endl;
  std::cout << "Iterations: " << solution.iterations_completed << std::endl;
  std::cout << "Solve time: " << std::fixed << std::setprecision(2)
            << solution.solve_time_ms << " ms" << std::endl;
  std::cout << "Final objective: " << std::scientific << std::setprecision(6)
            << solution.final_objective << std::endl;
  std::cout << "Primal infeasibility: " << std::scientific << std::setprecision(2)
            << solution.final_primal_infeasibility << std::endl;
  std::cout << "Dual infeasibility: " << std::scientific << std::setprecision(2)
            << solution.final_dual_infeasibility << std::endl;
  std::cout << "==============================\n" << std::endl;
}

} // namespace cddp
