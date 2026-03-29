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

#include "cddp_core/clddp_solver.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/cddp_core.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

namespace cddp {

CLDDPSolver::CLDDPSolver() : boxqp_solver_(BoxQPOptions()) {}

void CLDDPSolver::initialize(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  int horizon = context.getHorizon();
  int control_dim = context.getControlDim();
  int state_dim = context.getStateDim();

  // Warm start validation
  if (options.warm_start) {
    bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                             K_u_.size() == static_cast<size_t>(horizon));

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
        std::cout << "CLDDP: Using warm start with existing control gains"
                  << std::endl;
      }
      boxqp_solver_.setOptions(options.box_qp);
      if (!context.X_.empty() && !context.U_.empty()) {
        computeCost(context);
      }
      return;
    } else if (options.verbose) {
      std::cout << "CLDDP: Warning - warm start requested but no valid "
                   "solver state found. Falling back to cold start."
                << std::endl;
    }
  }

  // Cold start
  initializeGains(horizon, control_dim, state_dim);
  boxqp_solver_ = BoxQPSolver(options.box_qp);

  if (!context.X_.empty() && !context.U_.empty()) {
    computeCost(context);
  }
}

std::string CLDDPSolver::getSolverName() const { return "CLDDP"; }

bool CLDDPSolver::backwardPass(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();

  auto control_constraint =
      context.getConstraint<ControlConstraint>("ControlConstraint");

  // Terminal cost derivatives
  Eigen::VectorXd V_x =
      context.getObjective().getFinalCostGradient(context.X_.back());
  Eigen::MatrixXd V_xx =
      context.getObjective().getFinalCostHessian(context.X_.back());

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

    Q_x = l_x + A.transpose() * V_x;
    Q_u = l_u + B.transpose() * V_x;
    Q_xx = l_xx + A.transpose() * V_xx * A;
    Q_ux = l_ux + B.transpose() * V_xx * A;
    Q_uu = l_uu + B.transpose() * V_xx * B;

    Q_uu_reg = Q_uu;
    Q_uu_reg.diagonal().array() += context.regularization_;

    Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
    if (es.eigenvalues().real().minCoeff() <= 0) {
      if (options.debug) {
        std::cerr << "CLDDP: Q_uu is not positive definite at time " << t
                  << std::endl;
      }
      return false;
    }

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
          std::cerr << "CLDDP: BoxQP failed at time step " << t << std::endl;
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

    Eigen::Vector2d dV_step;
    dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
    dV_ += dV_step;

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

bool CLDDPSolver::checkEarlyConvergence(CDDP &context, int /*iter*/,
                                        std::string &reason) {
  if (context.inf_du_ < context.getOptions().tolerance) {
    reason = "OptimalSolutionFound";
    return true;
  }
  return false;
}

ForwardPassResult CLDDPSolver::forwardPass(CDDP &context, double alpha_pr) {
  ForwardPassResult result;
  result.success = false;
  result.cost = std::numeric_limits<double>::infinity();
  result.merit_function = std::numeric_limits<double>::infinity();
  result.alpha_pr = alpha_pr;

  result.state_trajectory = context.X_;
  result.control_trajectory = context.U_;
  result.state_trajectory[0] = context.getInitialState();

  double J_new = 0.0;
  auto control_constraint =
      context.getConstraint<ControlConstraint>("ControlConstraint");

  for (int t = 0; t < context.getHorizon(); ++t) {
    const Eigen::VectorXd &x = result.state_trajectory[t];
    const Eigen::VectorXd delta_x = x - context.X_[t];

    result.control_trajectory[t] =
        result.control_trajectory[t] + alpha_pr * k_u_[t] + K_u_[t] * delta_x;

    if (control_constraint != nullptr) {
      result.control_trajectory[t] =
          control_constraint->clamp(result.control_trajectory[t]);
    }

    J_new +=
        context.getObjective().running_cost(x, result.control_trajectory[t], t);

    result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
        x, result.control_trajectory[t], t * context.getTimestep());
  }

  J_new += context.getObjective().terminal_cost(result.state_trajectory.back());

  double dJ = context.cost_ - J_new;
  double expected = -alpha_pr * (dV_(0) + 0.5 * alpha_pr * dV_(1));
  double reduction_ratio =
      expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

  const CDDPOptions &options = context.getOptions();
  result.success = reduction_ratio > options.filter.armijo_constant;
  result.cost = J_new;
  result.merit_function = J_new;

  return result;
}

bool CLDDPSolver::checkConvergence(CDDP &context, double dJ, double /*dL*/,
                                   int /*iter*/, std::string &reason) {
  const CDDPOptions &options = context.getOptions();

  if (context.inf_du_ < options.tolerance) {
    reason = "OptimalSolutionFound";
    return true;
  }
  if (dJ > 0.0 && dJ < options.acceptable_tolerance) {
    reason = "AcceptableSolutionFound";
    return true;
  }
  return false;
}

void CLDDPSolver::printIteration(int iter, const CDDP &context) const {
  if (iter == 0) {
    std::cout << std::setw(4) << "iter" << " " << std::setw(12) << "objective"
              << " " << std::setw(10) << "inf_du" << " " << std::setw(8)
              << "lg(rg)" << " " << std::setw(8) << "alpha" << std::endl;
  }

  std::cout << std::setw(4) << iter << " " << std::setw(12) << std::scientific
            << std::setprecision(4) << context.cost_ << " " << std::setw(10)
            << std::scientific << std::setprecision(2) << context.inf_du_
            << " " << std::setw(8) << std::fixed << std::setprecision(1)
            << std::log10(context.regularization_) << " " << std::setw(8)
            << std::fixed << std::setprecision(4) << context.alpha_pr_
            << std::endl;
}

} // namespace cddp
