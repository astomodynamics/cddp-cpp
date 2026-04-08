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

#include "interior_point_utils.hpp"
#include "cddp_core/ipddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include "cddp_core/terminal_constraint.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <execution>
#include <future>
#include <iomanip>
#include <iostream>
#include <thread>

namespace cddp
{
  namespace
  {
    constexpr double kSlackInteriorOffset = 1e-4;
    constexpr double kWarmstartStateMatchTolerance = 1e-6;
    constexpr double EPS_SLACK = 1e-10;
    constexpr double EPS_DUAL = 1e-10;
    constexpr double MAX_BARRIER_RATIO = 1e6;

    struct TerminalEqualityLayoutEntry
    {
      std::string name;
      int dim = 0;
    };

    struct TerminalInequalityLayoutEntry
    {
      std::string name;
      int dim = 0;
    };

    std::vector<TerminalEqualityLayoutEntry> getTerminalEqualityLayout(
        const CDDP &context)
    {
      std::vector<TerminalEqualityLayoutEntry> layout;
      for (const auto &constraint_pair : context.getTerminalConstraintSet())
      {
        if (dynamic_cast<const TerminalEqualityConstraint *>(
                constraint_pair.second.get()) == nullptr)
        {
          continue;
        }
        layout.push_back(
            TerminalEqualityLayoutEntry{constraint_pair.first,
                                        constraint_pair.second->getDualDim()});
      }
      return layout;
    }

    std::vector<TerminalInequalityLayoutEntry> getTerminalInequalityLayout(
        const CDDP &context)
    {
      std::vector<TerminalInequalityLayoutEntry> layout;
      for (const auto &constraint_pair : context.getTerminalConstraintSet())
      {
        if (dynamic_cast<const TerminalInequalityConstraint *>(
                constraint_pair.second.get()) == nullptr)
        {
          continue;
        }
        layout.push_back(TerminalInequalityLayoutEntry{
            constraint_pair.first, constraint_pair.second->getDualDim()});
      }
      return layout;
    }

    int getTerminalEqualityDim(const CDDP &context)
    {
      int dim = 0;
      for (const auto &entry : getTerminalEqualityLayout(context))
      {
        dim += entry.dim;
      }
      return dim;
    }

    std::map<std::string, Eigen::VectorXd> evaluateTerminalInequalityResidualMap(
        const CDDP &context, const Eigen::VectorXd &x_terminal)
    {
      std::map<std::string, Eigen::VectorXd> residuals;
      for (const auto &entry : getTerminalInequalityLayout(context))
      {
        const auto *constraint =
            dynamic_cast<const TerminalInequalityConstraint *>(
                context.getTerminalConstraintSet().at(entry.name).get());
        if (!constraint) {
          throw std::runtime_error(
              "IPDDP: terminal constraint '" + entry.name +
              "' is not a TerminalInequalityConstraint");
        }
        residuals[entry.name] = constraint->evaluate(x_terminal);
      }
      return residuals;
    }

    std::map<std::string, Eigen::MatrixXd> evaluateTerminalInequalityJacobianMap(
        const CDDP &context, const Eigen::VectorXd &x_terminal)
    {
      std::map<std::string, Eigen::MatrixXd> jacobians;
      for (const auto &entry : getTerminalInequalityLayout(context))
      {
        const auto *constraint =
            dynamic_cast<const TerminalInequalityConstraint *>(
                context.getTerminalConstraintSet().at(entry.name).get());
        if (!constraint) {
          throw std::runtime_error(
              "IPDDP: terminal constraint '" + entry.name +
              "' is not a TerminalInequalityConstraint");
        }
        jacobians[entry.name] = constraint->getStateJacobian(x_terminal);
      }
      return jacobians;
    }

    Eigen::VectorXd evaluateTerminalEqualityResidual(
        const CDDP &context, const Eigen::VectorXd &x_terminal)
    {
      const auto layout = getTerminalEqualityLayout(context);
      const int total_dim = getTerminalEqualityDim(context);
      Eigen::VectorXd residual = Eigen::VectorXd::Zero(total_dim);
      int offset = 0;
      for (const auto &entry : layout)
      {
        const auto *constraint =
            dynamic_cast<const TerminalEqualityConstraint *>(
                context.getTerminalConstraintSet().at(entry.name).get());
        if (!constraint) {
          throw std::runtime_error(
              "IPDDP: terminal constraint '" + entry.name +
              "' is not a TerminalEqualityConstraint");
        }
        residual.segment(offset, entry.dim) = constraint->evaluate(x_terminal);
        offset += entry.dim;
      }
      return residual;
    }

    Eigen::MatrixXd evaluateTerminalEqualityJacobian(
        const CDDP &context, const Eigen::VectorXd &x_terminal)
    {
      const auto layout = getTerminalEqualityLayout(context);
      const int total_dim = getTerminalEqualityDim(context);
      Eigen::MatrixXd jacobian =
          Eigen::MatrixXd::Zero(total_dim, x_terminal.size());
      int offset = 0;
      for (const auto &entry : layout)
      {
        const auto *constraint =
            dynamic_cast<const TerminalEqualityConstraint *>(
                context.getTerminalConstraintSet().at(entry.name).get());
        if (!constraint) {
          throw std::runtime_error(
              "IPDDP: terminal constraint '" + entry.name +
              "' is not a TerminalEqualityConstraint");
        }
        jacobian.block(offset, 0, entry.dim, x_terminal.size()) =
            constraint->getStateJacobian(x_terminal);
        offset += entry.dim;
      }
      return jacobian;
    }

    std::map<std::string, Eigen::VectorXd> splitTerminalEqualityVector(
        const CDDP &context, const Eigen::VectorXd &stacked)
    {
      std::map<std::string, Eigen::VectorXd> split;
      const auto layout = getTerminalEqualityLayout(context);
      int offset = 0;
      for (const auto &entry : layout)
      {
        split[entry.name] = stacked.segment(offset, entry.dim);
        offset += entry.dim;
      }
      return split;
    }

    Eigen::MatrixXd symmetrizeMatrix(const Eigen::MatrixXd &matrix)
    {
      return 0.5 * (matrix + matrix.transpose());
    }

    double clipPositiveBarrierRatio(double numerator, double denominator)
    {
      return std::clamp(numerator / denominator, 0.0, MAX_BARRIER_RATIO);
    }

    double clipSignedBarrierRatio(double numerator, double denominator)
    {
      return std::clamp(numerator / denominator, -MAX_BARRIER_RATIO,
                        MAX_BARRIER_RATIO);
    }

    void repairWarmstartInterior(Eigen::VectorXd &s, Eigen::VectorXd &y,
                                 const CDDPOptions &options)
    {
      if (!options.ipddp.warmstart_repair)
      {
        return;
      }
      if (s.size() > 0)
      {
        s = s.cwiseMax(options.ipddp.warmstart_s_min);
        const double min_s = s.minCoeff();
        if (min_s <
            options.ipddp.warmstart_s_min *
                options.ipddp.warmstart_interior_factor)
        {
          s *= options.ipddp.warmstart_interior_factor;
        }
      }
      if (y.size() > 0)
      {
        y = y.cwiseMax(options.ipddp.warmstart_y_min);
        const double min_y = y.minCoeff();
        if (min_y <
            options.ipddp.warmstart_y_min *
                options.ipddp.warmstart_interior_factor)
        {
          y *= options.ipddp.warmstart_interior_factor;
        }
      }
    }

    void rolloutLinearPolicy(const std::vector<Eigen::MatrixXd> &A,
                             const std::vector<Eigen::MatrixXd> &B,
                             const std::vector<Eigen::VectorXd> &d,
                             const std::vector<Eigen::MatrixXd> &K,
                             const std::vector<Eigen::VectorXd> &k,
                             const Eigen::VectorXd &dx0,
                             std::vector<Eigen::VectorXd> &dX,
                             std::vector<Eigen::VectorXd> &dU)
    {
      const int T = static_cast<int>(K.size());
      if (T == 0 || A.empty() || B.empty())
      {
        dX.assign(1, dx0);
        dU.clear();
        return;
      }
      dX.assign(T + 1, Eigen::VectorXd::Zero(A.front().rows()));
      dU.assign(T, Eigen::VectorXd::Zero(B.front().cols()));
      dX[0] = dx0;
      for (int t = 0; t < T; ++t)
      {
        dU[t] = k[t] + K[t] * dX[t];
        dX[t + 1] = A[t] * dX[t] + B[t] * dU[t] + d[t];
      }
    }

    void rolloutLinearPolicy(const std::vector<Eigen::MatrixXd> &A,
                             const std::vector<Eigen::MatrixXd> &B,
                             const std::vector<Eigen::VectorXd> &d,
                             const std::vector<Eigen::MatrixXd> &K,
                             const std::vector<Eigen::VectorXd> &k,
                             std::vector<Eigen::VectorXd> &dX,
                             std::vector<Eigen::VectorXd> &dU)
    {
      if (A.empty() || B.empty())
      {
        const int state_dim = !K.empty() ? K.front().cols() : 0;
        dX.assign(1, Eigen::VectorXd::Zero(state_dim));
        dU.clear();
        return;
      }
      rolloutLinearPolicy(A, B, d, K, k,
                          Eigen::VectorXd::Zero(A.front().rows()), dX, dU);
    }

    bool solveSequentialLQR(
        const std::vector<Eigen::MatrixXd> &Q,
        const std::vector<Eigen::VectorXd> &q,
        const std::vector<Eigen::MatrixXd> &R,
        const std::vector<Eigen::VectorXd> &r,
        const std::vector<Eigen::MatrixXd> &M,
        const std::vector<Eigen::MatrixXd> &A,
        const std::vector<Eigen::MatrixXd> &B,
        const std::vector<Eigen::VectorXd> &d,
        std::vector<Eigen::MatrixXd> &K,
        std::vector<Eigen::VectorXd> &k,
        std::vector<Eigen::MatrixXd> &P,
        std::vector<Eigen::VectorXd> &p)
    {
      const int T = static_cast<int>(R.size());
      if (T == 0)
      {
        return true;
      }

      const int n = Q.front().rows();
      const int m = R.front().rows();
      K.assign(T, Eigen::MatrixXd::Zero(m, n));
      k.assign(T, Eigen::VectorXd::Zero(m));
      P.assign(T + 1, Eigen::MatrixXd::Zero(n, n));
      p.assign(T + 1, Eigen::VectorXd::Zero(n));
      P[T] = 0.5 * (Q[T] + Q[T].transpose());
      p[T] = q[T];

      for (int t = T - 1; t >= 0; --t)
      {
        const Eigen::MatrixXd &P_next = P[t + 1];
        const Eigen::VectorXd &p_next = p[t + 1];
        const Eigen::MatrixXd BtP = B[t].transpose() * P_next;
        const Eigen::MatrixXd Q_uu =
            0.5 * (R[t] + BtP * B[t] + R[t].transpose() +
                   B[t].transpose() * P_next.transpose() * B[t]);
        const Eigen::MatrixXd Q_ux = BtP * A[t] + M[t].transpose();
        const Eigen::MatrixXd Q_xu = Q_ux.transpose();
        const Eigen::VectorXd drift = p_next + P_next * d[t];
        const Eigen::VectorXd Q_x = q[t] + A[t].transpose() * drift;
        const Eigen::VectorXd Q_u = r[t] + B[t].transpose() * drift;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu);
        if (ldlt.info() != Eigen::Success)
        {
          return false;
        }

        K[t] = -ldlt.solve(Q_ux);
        k[t] = -ldlt.solve(Q_u);
        P[t] = Q[t] + A[t].transpose() * P_next * A[t] + Q_xu * K[t] +
               K[t].transpose() * Q_ux + K[t].transpose() * Q_uu * K[t];
        P[t] = 0.5 * (P[t] + P[t].transpose());
        p[t] = Q_x + Q_xu * k[t] + K[t].transpose() * Q_u +
               K[t].transpose() * Q_uu * k[t];
        if (!P[t].allFinite() || !p[t].allFinite() || !K[t].allFinite() ||
            !k[t].allFinite())
        {
          return false;
        }
      }
      return true;
    }

    bool solveTerminalEqualityLQR(
        const std::vector<Eigen::MatrixXd> &Q,
        const std::vector<Eigen::VectorXd> &q,
        const std::vector<Eigen::MatrixXd> &R,
        const std::vector<Eigen::VectorXd> &r,
        const std::vector<Eigen::MatrixXd> &M,
        const std::vector<Eigen::MatrixXd> &A,
        const std::vector<Eigen::MatrixXd> &B,
        const std::vector<Eigen::VectorXd> &d,
        const Eigen::VectorXd &dx0,
        const Eigen::MatrixXd &H_T,
        const Eigen::VectorXd &b_T,
        double mu,
        double reg_scale,
        double reg_exponent,
        const Eigen::VectorXd *lambda_prev,
        std::vector<Eigen::MatrixXd> &K_out,
        std::vector<Eigen::VectorXd> &k_out,
        std::vector<Eigen::MatrixXd> &P_out,
        std::vector<Eigen::VectorXd> &p_out,
        Eigen::VectorXd &lambda_total,
        Eigen::VectorXd &lambda_delta)
    {
      const int p_dim = H_T.rows();
      if (p_dim == 0)
      {
        lambda_total = Eigen::VectorXd::Zero(0);
        lambda_delta = Eigen::VectorXd::Zero(0);
        return solveSequentialLQR(Q, q, R, r, M, A, B, d, K_out, k_out, P_out,
                                  p_out);
      }

      std::vector<Eigen::VectorXd> q_base = q;
      Eigen::VectorXd lambda_prev_vec = Eigen::VectorXd::Zero(p_dim);
      if (lambda_prev != nullptr && lambda_prev->size() == p_dim)
      {
        lambda_prev_vec = *lambda_prev;
        q_base.back() += H_T.transpose() * lambda_prev_vec;
      }

      const int T = static_cast<int>(R.size());
      const int n = Q.front().rows();
      std::vector<std::vector<Eigen::MatrixXd>> K_variants(
          p_dim + 1, std::vector<Eigen::MatrixXd>(T));
      std::vector<std::vector<Eigen::VectorXd>> k_variants(
          p_dim + 1, std::vector<Eigen::VectorXd>(T));
      std::vector<std::vector<Eigen::MatrixXd>> P_variants(
          p_dim + 1, std::vector<Eigen::MatrixXd>(T + 1));
      std::vector<std::vector<Eigen::VectorXd>> p_variants(
          p_dim + 1, std::vector<Eigen::VectorXd>(T + 1));
      std::vector<Eigen::VectorXd> xT_variants(
          p_dim + 1, Eigen::VectorXd::Zero(n));

      for (int i = 0; i < p_dim + 1; ++i)
      {
        std::vector<Eigen::VectorXd> q_variant = q_base;
        if (i > 0)
        {
          q_variant.back() += H_T.row(i - 1).transpose();
        }
        if (!solveSequentialLQR(Q, q_variant, R, r, M, A, B, d, K_variants[i],
                                k_variants[i], P_variants[i], p_variants[i]))
        {
          return false;
        }
        std::vector<Eigen::VectorXd> dX_variant;
        std::vector<Eigen::VectorXd> dU_variant;
        rolloutLinearPolicy(A, B, d, K_variants[i], k_variants[i], dx0,
                            dX_variant, dU_variant);
        xT_variants[i] = dX_variant.back();
      }

      Eigen::MatrixXd S_mat(n, p_dim);
      for (int i = 0; i < p_dim; ++i)
      {
        S_mat.col(i) = xT_variants[i + 1] - xT_variants[0];
      }

      const Eigen::MatrixXd A_small = H_T * S_mat;
      const Eigen::VectorXd rhs = b_T - H_T * xT_variants[0];
      const Eigen::MatrixXd AtA = A_small.transpose() * A_small;
      const Eigen::VectorXd Atb = A_small.transpose() * rhs;

      const double trace_term =
          (AtA.trace() > 1.0 ? AtA.trace() / std::max(p_dim, 1) : 1.0);
      const double base_floor =
          std::max(1e-10, reg_scale * std::pow(std::max(mu, 0.0), reg_exponent));
      const double reg = std::max(base_floor, 1e-6 * trace_term);
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_small);
      const auto &singular = svd.singularValues();
      const double sigma_max = singular.size() > 0 ? singular.maxCoeff() : 0.0;
      const double sigma_min = singular.size() > 0 ? singular.minCoeff() : 0.0;
      const double svd_reg = std::max(1e-8 * sigma_max - sigma_min, 0.0);
      const double reg_base = std::max(reg, svd_reg);
      const double lambda_norm_cap = 100.0 * (1.0 + rhs.norm());

      const std::array<double, 5> reg_scales{{1.0, 10.0, 100.0, 1e3, 1e4}};
      Eigen::VectorXd best_lambda = Eigen::VectorXd::Zero(p_dim);
      double best_residual = std::numeric_limits<double>::infinity();
      bool found_finite = false;
      for (double scale : reg_scales)
      {
        const double reg_i = std::max(reg_base * scale, 1e-12);
        Eigen::MatrixXd shifted =
            AtA + reg_i * Eigen::MatrixXd::Identity(p_dim, p_dim);
        Eigen::LDLT<Eigen::MatrixXd> ldlt(shifted);
        if (ldlt.info() != Eigen::Success)
        {
          continue;
        }
        Eigen::VectorXd lambda_i = ldlt.solve(Atb);
        if (!lambda_i.allFinite())
        {
          continue;
        }
        const double lambda_norm = lambda_i.norm();
        if (lambda_norm > lambda_norm_cap)
        {
          lambda_i *= lambda_norm_cap / std::max(lambda_norm, 1e-12);
        }
        const double residual = (A_small * lambda_i - rhs).norm();
        if (!std::isfinite(residual))
        {
          continue;
        }
        if (!found_finite || residual < best_residual)
        {
          best_lambda = lambda_i;
          best_residual = residual;
          found_finite = true;
        }
      }

      if (!found_finite)
      {
        std::cerr << "IPDDP: solveTerminalEqualityLQR failed for all "
                  << reg_scales.size() << " regularization scales; "
                  << "using zero terminal multipliers" << std::endl;
        best_lambda = Eigen::VectorXd::Zero(p_dim);
      }

      K_out = K_variants[0];
      k_out = k_variants[0];
      P_out = P_variants[0];
      p_out = p_variants[0];
      for (int i = 0; i < p_dim; ++i)
      {
        const double coeff = best_lambda(i);
        for (int t = 0; t < T; ++t)
        {
          k_out[t] += coeff * (k_variants[i + 1][t] - k_variants[0][t]);
        }
        for (int t = 0; t <= T; ++t)
        {
          p_out[t] += coeff * (p_variants[i + 1][t] - p_variants[0][t]);
        }
      }

      lambda_delta = best_lambda;
      lambda_total = lambda_prev_vec + best_lambda;
      return true;
    }
  } // namespace

  IPDDPSolver::IPDDPSolver() : mu_(1e-1) {}

  void IPDDPSolver::initialize(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const auto &constraint_set = context.getConstraintSet();
    int horizon = context.getHorizon();
    int control_dim = context.getControlDim();
    int state_dim = context.getStateDim();

    // Check for valid warm start with existing solver state
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
          std::cout << "IPDDP: Using warm start with existing control gains"
                    << std::endl;
        }
        mu_ = options.ipddp.barrier.mu_initial * 0.1;
        context.step_norm_ = 0.0;

        // Re-rollout trajectory with existing controls
        context.X_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        context.X_[0] = context.getInitialState();
        for (int t = 0; t < horizon; ++t)
        {
          context.X_[t + 1] = context.getSystem().getDiscreteDynamics(
              context.X_[t], context.U_[t], t * context.getTimestep());
        }

        // Resize costate arrays if needed
        if (Lambda_.size() != static_cast<size_t>(horizon + 1))
        {
          Lambda_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        }
        if (k_lambda_.size() != static_cast<size_t>(horizon + 1))
        {
          k_lambda_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
          K_lambda_.assign(horizon + 1, Eigen::MatrixXd::Zero(state_dim, state_dim));
        }
        if (dX_.size() != static_cast<size_t>(horizon + 1))
        {
          dX_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
          dU_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
        }
        dV_ = Eigen::Vector2d::Zero();
        Lambda_T_eq_ = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));
        dLambda_T_eq_ = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));

        initializeConstraintStorage(context);
        evaluateTrajectoryWarmStart(context);
        initializeDualSlackVariablesWarmStart(context);
        G_ = G_raw_;
        resetFilter(context);
        return;
      }
      else
      {
        // Warm start with provided trajectory (no existing solver state)
        if (options.verbose)
        {
          std::cout << "IPDDP: Warm start with provided trajectory" << std::endl;
        }

        // Initialize gains and constraints
        k_u_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
        K_u_.assign(horizon, Eigen::MatrixXd::Zero(control_dim, state_dim));
        dX_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        dU_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
        k_lambda_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        K_lambda_.assign(horizon + 1, Eigen::MatrixXd::Zero(state_dim, state_dim));
        Lambda_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        dV_ = Eigen::Vector2d::Zero();
        Lambda_T_eq_ = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));
        dLambda_T_eq_ = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));

        initializeConstraintStorage(context);

        // Re-rollout trajectory with provided controls
        if (static_cast<int>(context.U_.size()) != horizon)
        {
          context.U_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
        }
        context.X_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        context.X_[0] = context.getInitialState();
        for (int t = 0; t < horizon; ++t)
        {
          context.X_[t + 1] = context.getSystem().getDiscreteDynamics(
              context.X_[t], context.U_[t], t * context.getTimestep());
        }

        // Set barrier parameter based on constraint evaluation
        if (constraint_set.empty() && context.getTerminalConstraintSet().empty())
        {
          mu_ = std::max(options.tolerance / 10.0,
                         options.ipddp.barrier.mu_min_value);
        }
        else
        {
          evaluateTrajectoryWarmStart(context);
          double max_constraint_violation = computeMaxConstraintViolation(context);
          if (max_constraint_violation <= options.tolerance)
          {
            mu_ = std::max(options.tolerance,
                           options.ipddp.barrier.mu_min_value);
          }
          else if (max_constraint_violation <= 0.1)
          {
            mu_ = std::max(options.tolerance * 10.0,
                           options.ipddp.barrier.mu_initial * 0.01);
          }
          else
          {
            mu_ = options.ipddp.barrier.mu_initial * 0.1;
          }
        }

        context.regularization_ = options.regularization.initial_value;
        context.step_norm_ = 0.0;
        context.alpha_pr_ = 1.0;
        context.alpha_du_ = 1.0;
        initializeDualSlackVariablesWarmStart(context);
        G_ = G_raw_;
        resetFilter(context);
        context.inf_du_ = 0.0;
        return;
      }
    }

    initializeConstraintStorage(context);

    k_u_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
    K_u_.assign(horizon, Eigen::MatrixXd::Zero(control_dim, state_dim));
    dX_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    dU_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
    k_lambda_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    K_lambda_.assign(horizon + 1, Eigen::MatrixXd::Zero(state_dim, state_dim));
    Lambda_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    dV_ = Eigen::Vector2d::Zero();
    Lambda_T_eq_ = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));
    dLambda_T_eq_ = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));

    if (static_cast<int>(context.U_.size()) != horizon)
    {
      context.U_.assign(horizon, Eigen::VectorXd::Zero(control_dim));
    }
    for (auto &u : context.U_)
    {
      if (u.size() != control_dim)
      {
        u = Eigen::VectorXd::Zero(control_dim);
      }
    }

    bool reset_warmstart = false;
    if (options.warm_start && options.ipddp.warmstart_reset_x0_threshold > 0.0 &&
        !context.X_.empty())
    {
      const double dx0 =
          (context.getInitialState() - context.X_.front()).norm();
      reset_warmstart = dx0 > options.ipddp.warmstart_reset_x0_threshold;
    }
    if (reset_warmstart)
    {
      for (auto &u : context.U_)
      {
        u.setZero();
      }
      Y_.clear();
      S_.clear();
      G_.clear();
      Y_T_.clear();
      S_T_.clear();
      G_T_.clear();
      Lambda_T_eq_.setZero();
      dLambda_T_eq_.setZero();
    }

    context.X_.assign(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    context.X_[0] = context.getInitialState();
    for (int t = 0; t < horizon; ++t)
    {
      context.X_[t + 1] = context.getSystem().getDiscreteDynamics(
          context.X_[t], context.U_[t], t * context.getTimestep());
    }

    mu_ = constraint_set.empty() && context.getTerminalConstraintSet().empty()
              ? std::max(options.tolerance / 10.0,
                         options.ipddp.barrier.mu_min_value)
              : options.ipddp.barrier.mu_initial;

    context.regularization_ = options.regularization.initial_value;
    context.step_norm_ = 0.0;
    context.alpha_pr_ = 1.0;
    context.alpha_du_ = 1.0;

    evaluateTrajectory(context);
    initializeDualSlackVariables(context);

    for (const auto &entry : getTerminalInequalityLayout(context))
    {
      const Eigen::VectorXd &g_T = G_T_[entry.name];
      Eigen::VectorXd s_init =
          g_T.unaryExpr([&](double g) {
            return std::max(options.ipddp.slack_var_init_scale,
                            -g + kSlackInteriorOffset);
          });
      Eigen::VectorXd y_init(entry.dim);
      for (int i = 0; i < entry.dim; ++i)
      {
        y_init(i) =
            (mu_ * options.ipddp.dual_var_init_scale) / std::max(s_init(i), EPS_SLACK);
      }
      repairWarmstartInterior(s_init, y_init, options);
      S_T_[entry.name] = s_init;
      Y_T_[entry.name] = y_init;
      dS_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
      dY_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
    }

    G_ = G_raw_;

    resetFilter(context);
    context.inf_du_ = 0.0;
  }

  std::string IPDDPSolver::getSolverName() const { return "IPDDP"; }

  // === CDDPSolverBase hook overrides ===

  void IPDDPSolver::preIterationSetup(CDDP &context)
  {
    // Initialization already done in initialize()
  }

  bool IPDDPSolver::checkEarlyConvergence(CDDP &context, int iter,
                                           std::string &reason)
  {
    // Check convergence after backward pass (before forward pass)
    const CDDPOptions &options = context.getOptions();
    const bool no_barrier_needed =
        context.getConstraintSet().empty() &&
        getTerminalInequalityLayout(context).empty();
    const double scaled_inf_du = computeScaledDualInfeasibility(context);
    const double scaled_inf_comp = context.inf_comp_;

    if (no_barrier_needed)
    {
      if (context.inf_pr_ < options.tolerance &&
          scaled_inf_du < options.tolerance)
      {
        reason = "OptimalSolutionFound";
        return true;
      }
      return false;
    }

    const double tol = std::max(options.tolerance,
                                options.ipddp.barrier_tol_mult * mu_);
    const double accepted_step_norm =
        std::abs(context.alpha_pr_) * context.step_norm_;
    if (context.inf_pr_ < tol && scaled_inf_du < tol && scaled_inf_comp < tol &&
        accepted_step_norm < options.tolerance * 10.0)
    {
      reason = "OptimalSolutionFound";
      return true;
    }
    return false;
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
    const bool has_terminal_ineq =
        !getTerminalInequalityLayout(context).empty();
    const bool has_terminal_eq = getTerminalEqualityDim(context) > 0;
    const bool has_path_constraints = !constraint_set.empty();

    CDDPSolverBase::precomputeDynamicsDerivatives(context);
    precomputeConstraintGradients(context);
    G_ = G_raw_;

    Eigen::VectorXd V_x =
        context.getObjective().getFinalCostGradient(context.X_.back());
    Eigen::MatrixXd V_xx =
        context.getObjective().getFinalCostHessian(context.X_.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose());

    dV_ = Eigen::Vector2d::Zero();
    double inf_du = 0.0;
    double inf_pr = 0.0;
    double inf_comp = 0.0;
    double step_norm = 0.0;

    if (has_terminal_ineq)
    {
      G_T_ = evaluateTerminalInequalityResidualMap(context, context.X_.back());
      const auto terminal_jacobians =
          evaluateTerminalInequalityJacobianMap(context, context.X_.back());
      for (const auto &entry : getTerminalInequalityLayout(context))
      {
        const Eigen::VectorXd &g_T = G_T_.at(entry.name);
        const Eigen::MatrixXd &G_T_x = terminal_jacobians.at(entry.name);
        const Eigen::VectorXd &S_T = S_T_.at(entry.name);
        const Eigen::VectorXd &Y_T = Y_T_.at(entry.name);
        Eigen::VectorXd sigma_T(entry.dim);
        Eigen::VectorXd barrier_grad_T(entry.dim);
        for (int i = 0; i < entry.dim; ++i)
        {
          const double s_safe =
              std::max(S_T(i), std::max(mu_ * 1e-3, EPS_SLACK));
          const double y_safe = std::max(Y_T(i), EPS_DUAL);
          sigma_T(i) = clipPositiveBarrierRatio(y_safe, s_safe);
          const double barrier_grad_correction =
              clipSignedBarrierRatio(y_safe * g_T(i) + mu_, s_safe);
          barrier_grad_T(i) = y_safe + barrier_grad_correction;
        }
        V_x.noalias() += G_T_x.transpose() * barrier_grad_T;
        V_xx.noalias() += G_T_x.transpose() * sigma_T.asDiagonal() * G_T_x;
        V_xx = symmetrizeMatrix(V_xx);
        inf_pr = std::max(inf_pr, (g_T + S_T).lpNorm<Eigen::Infinity>());
        inf_comp = std::max(
            inf_comp,
            (Y_T.cwiseProduct(S_T).array() - mu_).matrix().lpNorm<Eigen::Infinity>());
      }
    }

    if (has_terminal_eq)
    {
      const Eigen::VectorXd h_T =
          evaluateTerminalEqualityResidual(context, context.X_.back());
      const Eigen::MatrixXd H_T =
          evaluateTerminalEqualityJacobian(context, context.X_.back());
      V_x.noalias() += H_T.transpose() * Lambda_T_eq_;
      inf_pr = std::max(inf_pr, h_T.lpNorm<Eigen::Infinity>());
      dLambda_T_eq_ = -h_T;
    }
    else
    {
      dLambda_T_eq_ = Eigen::VectorXd::Zero(0);
    }

    k_lambda_.back() = V_x;
    K_lambda_.back() = V_xx;

    if (!has_path_constraints && !has_terminal_ineq && !has_terminal_eq)
    {
      for (int t = horizon - 1; t >= 0; --t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        // Use pre-computed discrete Jacobians from base class
        const Eigen::MatrixXd &A = F_x_[t];
        const Eigen::MatrixXd &B = F_u_[t];

        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] =
            context.getObjective().getRunningCostHessians(x, u, t);

        Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
        Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

        if (!options.use_ilqr)
        {
          const auto &Fxx = F_xx_[t];
          const auto &Fuu = F_uu_[t];
          const auto &Fux = F_ux_[t];

          for (int i = 0; i < state_dim; ++i)
          {
            Q_xx += V_x(i) * Fxx[i];
            Q_ux += V_x(i) * Fux[i];
            Q_uu += V_x(i) * Fuu[i];
          }
        }

        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());
        Q_uu.diagonal().array() += context.regularization_;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu);
        if (ldlt.info() != Eigen::Success)
        {
          return false;
        }

        Eigen::VectorXd k_u = -ldlt.solve(Q_u);
        Eigen::MatrixXd K_u = -ldlt.solve(Q_ux);
        k_u_[t] = k_u;
        K_u_[t] = K_u;

        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u +
              K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u +
               K_u.transpose() * Q_uu * K_u;
        V_xx = 0.5 * (V_xx + V_xx.transpose());
        k_lambda_[t] = V_x;
        K_lambda_[t] = V_xx;

        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        inf_du = std::max(inf_du, Q_u.lpNorm<Eigen::Infinity>());
        step_norm = std::max(step_norm, k_u.lpNorm<Eigen::Infinity>());
      }

      context.inf_du_ = inf_du;
      context.step_norm_ = step_norm;
      context.inf_pr_ = 0.0;
      context.inf_comp_ = 0.0;
      return true;
    }
    else
    {
      for (int t = horizon - 1; t >= 0; --t)
      {
        const Eigen::VectorXd &x = context.X_[t];
        const Eigen::VectorXd &u = context.U_[t];

        const Eigen::MatrixXd &A = F_x_[t];
        const Eigen::MatrixXd &B = F_u_[t];

        Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::VectorXd s = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);
        Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);

        int offset = 0;
        for (const auto &constraint_pair : constraint_set)
        {
          const std::string &constraint_name = constraint_pair.first;
          int dual_dim = constraint_pair.second->getDualDim();

          y.segment(offset, dual_dim) = Y_[constraint_name][t];
          s.segment(offset, dual_dim) = S_[constraint_name][t];
          g.segment(offset, dual_dim) = G_[constraint_name][t];
          Q_yx.block(offset, 0, dual_dim, state_dim) = G_x_[constraint_name][t];
          Q_yu.block(offset, 0, dual_dim, control_dim) = G_u_[constraint_name][t];

          offset += dual_dim;
        }

        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] =
            context.getObjective().getRunningCostHessians(x, u, t);

        Eigen::VectorXd Q_x = l_x + Q_yx.transpose() * y + A.transpose() * V_x;
        Eigen::VectorXd Q_u = l_u + Q_yu.transpose() * y + B.transpose() * V_x;
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

        if (!options.use_ilqr)
        {
          const auto &Fxx = F_xx_[t];
          const auto &Fuu = F_uu_[t];
          const auto &Fux = F_ux_[t];

          for (int i = 0; i < state_dim; ++i)
          {
            Q_xx += V_x(i) * Fxx[i];
            Q_ux += V_x(i) * Fux[i];
            Q_uu += V_x(i) * Fuu[i];
          }
        }

        Eigen::MatrixXd YSinv = Eigen::MatrixXd::Zero(total_dual_dim, total_dual_dim);
        for (int i = 0; i < total_dual_dim; ++i) {
          const double s_safe =
              std::max(s(i), std::max(mu_ * 1e-3, EPS_SLACK));
          YSinv(i, i) = clipPositiveBarrierRatio(y(i), s_safe);
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
          return false;
        }

        Eigen::VectorXd S_inv_rhat(total_dual_dim);
        for (int i = 0; i < total_dual_dim; ++i) {
          const double s_safe =
              std::max(s(i), std::max(mu_ * 1e-3, EPS_SLACK));
          S_inv_rhat(i) = clipSignedBarrierRatio(rhat(i), s_safe);
        }
        Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
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
          const double s_safe =
              std::max(s(i), std::max(mu_ * 1e-3, EPS_SLACK));
          k_y(i) =
              clipSignedBarrierRatio(rhat(i) + y(i) * temp(i), s_safe);
        }
        Eigen::MatrixXd K_y =
            (YSinv * (Q_yx + Q_yu * K_u))
                .cwiseMax(-MAX_BARRIER_RATIO)
                .cwiseMin(MAX_BARRIER_RATIO);
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

        Q_u.noalias() += Q_yu.transpose() * S_inv_rhat;
        Q_x.noalias() += Q_yx.transpose() * S_inv_rhat;
        Q_xx.noalias() += Q_yx.transpose() * YSinv * Q_yx;
        Q_ux.noalias() += Q_yu.transpose() * YSinv * Q_yx;
        Q_uu.noalias() += Q_yu.transpose() * YSinv * Q_yu;

        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u +
              K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u +
               K_u.transpose() * Q_uu * K_u;
        V_xx = 0.5 * (V_xx + V_xx.transpose());
        k_lambda_[t] = V_x;
        K_lambda_[t] = V_xx;

        inf_du = std::max(inf_du, Q_u.lpNorm<Eigen::Infinity>());
        inf_pr = std::max(inf_pr, primal_residual.lpNorm<Eigen::Infinity>());
        inf_comp = std::max(inf_comp, complementary_residual.lpNorm<Eigen::Infinity>());
        step_norm = std::max(step_norm, k_u.lpNorm<Eigen::Infinity>());
      }

      // Rollout linear policy to compute search directions for fraction-to-boundary
      {
        std::vector<Eigen::MatrixXd> A_vec(horizon);
        std::vector<Eigen::MatrixXd> B_vec(horizon);
        std::vector<Eigen::VectorXd> d_vec(horizon, Eigen::VectorXd::Zero(state_dim));
        for (int t = 0; t < horizon; ++t)
        {
          A_vec[t] = F_x_[t];
          B_vec[t] = F_u_[t];
        }
        rolloutLinearPolicy(A_vec, B_vec, d_vec, K_u_, k_u_, dX_, dU_);

        // Compute dual/slack search directions from gains and dX_
        for (const auto &constraint_pair : constraint_set)
        {
          const std::string &name = constraint_pair.first;
          for (int t = 0; t < horizon; ++t)
          {
            dS_[name][t] = k_s_[name][t] + K_s_[name][t] * dX_[t];
            dY_[name][t] = (k_y_[name][t] + K_y_[name][t] * dX_[t])
                               .cwiseMax(-MAX_BARRIER_RATIO)
                               .cwiseMin(MAX_BARRIER_RATIO);
          }
        }

        // Terminal inequality search directions
        if (has_terminal_ineq)
        {
          auto G_T_x = evaluateTerminalInequalityJacobianMap(context, context.X_.back());
          auto G_T_eval = evaluateTerminalInequalityResidualMap(context, context.X_.back());
          for (const auto &entry : getTerminalInequalityLayout(context))
          {
            const Eigen::VectorXd &g_T = G_T_eval.at(entry.name);
            const Eigen::MatrixXd &Gtx = G_T_x.at(entry.name);
            const Eigen::VectorXd &S_T = S_T_.at(entry.name);
            const Eigen::VectorXd &Y_T = Y_T_.at(entry.name);
            const Eigen::VectorXd r_p_T = g_T + S_T;
            const Eigen::VectorXd r_d_T = S_T.cwiseProduct(Y_T).array() - mu_;
            dS_T_[entry.name] = -r_p_T - Gtx * dX_.back();
            Eigen::VectorXd dY_T = Eigen::VectorXd::Zero(entry.dim);
            for (int i = 0; i < entry.dim; ++i)
            {
              const double s_safe = std::max(S_T(i), std::max(mu_ * 1e-3, EPS_SLACK));
              const double dual_ratio =
                  std::clamp(Y_T(i) / s_safe, 0.0, MAX_BARRIER_RATIO);
              const double affine =
                  std::clamp(-r_d_T(i) / s_safe, -MAX_BARRIER_RATIO, MAX_BARRIER_RATIO);
              dY_T(i) = std::clamp(
                  affine - dual_ratio * dS_T_[entry.name](i),
                  -MAX_BARRIER_RATIO, MAX_BARRIER_RATIO);
            }
            dY_T_[entry.name] = dY_T;
          }
        }
      }

      context.inf_pr_ = inf_pr;
      context.inf_du_ = inf_du;
      context.inf_comp_ = inf_comp;
      context.step_norm_ = step_norm;
      return true;
    }
  }

  ForwardPassResult IPDDPSolver::forwardPass(CDDP &context, double alpha)
  {
    const CDDPOptions &options = context.getOptions();
    const auto &constraint_set = context.getConstraintSet();
    const bool has_terminal_ineq = !getTerminalInequalityLayout(context).empty();
    const bool has_terminal_eq = getTerminalEqualityDim(context) > 0;
    const auto [alpha_pr_max, alpha_du_max] = computeMaxStepSizes(context);

    ForwardPassResult result;
    result.success = false;
    result.cost = context.cost_;
    result.merit_function = phi_;
    result.theta = theta_;
    const int horizon = context.getHorizon();
    const double tau = (constraint_set.empty() && !has_terminal_ineq)
                           ? 1.0
                           : std::max(options.ipddp.barrier.min_fraction_to_boundary,
                                      1.0 - mu_);
    const double alpha_pr = std::min(alpha, alpha_pr_max);
    const double alpha_du = std::min(alpha, alpha_du_max);
    result.alpha_pr = alpha_pr;
    result.alpha_du = alpha_du;

    result.state_trajectory.assign(horizon + 1, Eigen::VectorXd::Zero(context.getStateDim()));
    result.control_trajectory.assign(horizon, Eigen::VectorXd::Zero(context.getControlDim()));
    result.state_trajectory[0] = context.getInitialState();

    std::vector<Eigen::VectorXd> dx_real(horizon + 1,
                                         Eigen::VectorXd::Zero(context.getStateDim()));
    std::vector<Eigen::VectorXd> Lambda_new = Lambda_;
    std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;
    std::map<std::string, std::vector<Eigen::VectorXd>> G_new;
    std::map<std::string, Eigen::VectorXd> S_T_new = S_T_;
    std::map<std::string, Eigen::VectorXd> Y_T_new = Y_T_;
    std::map<std::string, Eigen::VectorXd> G_T_new = G_T_;
    Eigen::VectorXd Lambda_T_eq_new = Lambda_T_eq_;
    bool step_feasible = true;

    for (int t = 0; t < horizon; ++t)
    {
      dx_real[t] = result.state_trajectory[t] - context.X_[t];
      Lambda_new[t] =
          Lambda_[t] + alpha_pr * k_lambda_[t] + K_lambda_[t] * dx_real[t];
      if (!Lambda_new[t].allFinite())
      {
        return result;
      }

      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &name = constraint_pair.first;
        Eigen::VectorXd s_new =
            S_[name][t] + alpha_pr * k_s_[name][t] + K_s_[name][t] * dx_real[t];
        Eigen::VectorXd s_min = (1.0 - tau) * S_[name][t];
        Eigen::VectorXd y_new =
            Y_[name][t] + alpha_du * k_y_[name][t] + K_y_[name][t] * dx_real[t];
        Eigen::VectorXd y_min = (1.0 - tau) * Y_[name][t];
        for (int i = 0; i < constraint_pair.second->getDualDim(); ++i)
        {
          if (s_new(i) < s_min(i) || y_new(i) < y_min(i))
          {
            step_feasible = false;
            break;
          }
        }
        if (!step_feasible) break;
        if (!s_new.allFinite() || !y_new.allFinite())
        {
          return result;
        }
        S_new[name][t] = s_new;
        Y_new[name][t] = y_new;
      }
      if (!step_feasible) return result;

      result.control_trajectory[t] =
          context.U_[t] + alpha_pr * k_u_[t] + K_u_[t] * dx_real[t];
      result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
          result.state_trajectory[t], result.control_trajectory[t],
          t * context.getTimestep());
      if (!result.state_trajectory[t + 1].allFinite() ||
          !result.control_trajectory[t].allFinite())
      {
        return result;
      }
    }

    dx_real.back() = result.state_trajectory.back() - context.X_.back();
    Lambda_new.back() =
        Lambda_.back() + alpha_pr * k_lambda_.back() + K_lambda_.back() * dx_real.back();
    if (!Lambda_new.back().allFinite())
    {
      return result;
    }

    if (has_terminal_ineq)
    {
      auto terminal_residuals =
          evaluateTerminalInequalityResidualMap(context, context.X_.back());
      auto terminal_jacobians =
          evaluateTerminalInequalityJacobianMap(context, context.X_.back());
      for (const auto &entry : getTerminalInequalityLayout(context))
      {
        const Eigen::VectorXd &g_T0 = terminal_residuals.at(entry.name);
        const Eigen::MatrixXd &G_T_x0 = terminal_jacobians.at(entry.name);
        const Eigen::VectorXd k_s_T = -(g_T0 + S_T_.at(entry.name));
        const Eigen::MatrixXd K_s_T = -G_T_x0;
        S_T_new[entry.name] =
            S_T_.at(entry.name) + alpha_pr * k_s_T + K_s_T * dx_real.back();

        Eigen::VectorXd Y_trial = Y_T_.at(entry.name);
        for (int i = 0; i < entry.dim; ++i)
        {
          const double s_safe =
              std::max(S_T_.at(entry.name)(i), std::max(mu_ * 1e-3, EPS_SLACK));
          const double r_d =
              Y_T_.at(entry.name)(i) * S_T_.at(entry.name)(i) - mu_;
          const double dual_ratio =
              clipPositiveBarrierRatio(Y_T_.at(entry.name)(i), s_safe);
          Eigen::RowVectorXd K_y_row =
              -(dual_ratio * K_s_T.row(i));
          const double k_y =
              clipSignedBarrierRatio(-r_d - Y_T_.at(entry.name)(i) * k_s_T(i),
                                     s_safe);
          const double y_new =
              Y_T_.at(entry.name)(i) + alpha_du * k_y + K_y_row.dot(dx_real.back());
          Y_trial(i) = y_new;
        }
        Y_T_new[entry.name] = Y_trial;

        const Eigen::ArrayXd s_floor =
            ((1.0 - tau) * S_T_.at(entry.name).array())
                .max(Eigen::ArrayXd::Constant(
                    entry.dim, std::max(mu_ * 1e-3, EPS_SLACK)));
        if ((S_T_new[entry.name].array() < s_floor).any() ||
            (Y_T_new[entry.name].array() <
             (1.0 - tau) * Y_T_.at(entry.name).array()).any() ||
            !S_T_new[entry.name].allFinite() || !Y_T_new[entry.name].allFinite())
        {
          return result;
        }
      }
    }

    if (has_terminal_eq)
    {
      Lambda_T_eq_new = Lambda_T_eq_ + alpha_pr * dLambda_T_eq_;
      if (!Lambda_T_eq_new.allFinite())
      {
        return result;
      }
    }

    double cost_new = 0.0;
    std::map<std::string, std::vector<Eigen::VectorXd>> G_raw_new;
    for (const auto &constraint_pair : constraint_set)
    {
      G_raw_new[constraint_pair.first].resize(horizon);
      G_new[constraint_pair.first].resize(horizon);
    }

    for (int t = 0; t < horizon; ++t)
    {
      const Eigen::VectorXd &x_trial = result.state_trajectory[t];
      const Eigen::VectorXd &u_trial = result.control_trajectory[t];
      cost_new += context.getObjective().running_cost(x_trial, u_trial, t);
      for (const auto &constraint_pair : constraint_set)
      {
        const std::string &name = constraint_pair.first;
        G_raw_new[name][t] =
            constraint_pair.second->evaluate(x_trial, u_trial) -
            constraint_pair.second->getUpperBound();
      }
    }
    cost_new += context.getObjective().terminal_cost(result.state_trajectory.back());

    G_new = G_raw_new;

    Eigen::VectorXd h_T_new = Eigen::VectorXd::Zero(getTerminalEqualityDim(context));
    if (has_terminal_ineq)
    {
      G_T_new = evaluateTerminalInequalityResidualMap(
          context, result.state_trajectory.back());
    }
    if (has_terminal_eq)
    {
      h_T_new = evaluateTerminalEqualityResidual(
          context, result.state_trajectory.back());
    }

    const double phi_new = computeBarrierMerit(
        context, S_new, cost_new, has_terminal_ineq ? &S_T_new : nullptr,
        has_terminal_eq ? &Lambda_T_eq_new : nullptr,
        has_terminal_eq ? &h_T_new : nullptr);
    const double theta_new =
        computeTheta(options, G_new, S_new, has_terminal_ineq ? &G_T_new : nullptr,
                     has_terminal_ineq ? &S_T_new : nullptr,
                     has_terminal_eq ? &h_T_new : nullptr);
    const auto [inf_pr_new, inf_comp_new] =
        computePrimalAndComplementarity(
            context, G_new, S_new, Y_new, mu_,
            has_terminal_ineq ? &G_T_new : nullptr,
            has_terminal_ineq ? &S_T_new : nullptr,
            has_terminal_ineq ? &Y_T_new : nullptr,
            has_terminal_eq ? &h_T_new : nullptr);

    if (!std::isfinite(phi_new) || !std::isfinite(theta_new) ||
        !std::isfinite(inf_pr_new) || !std::isfinite(inf_comp_new))
    {
      return result;
    }

    bool filter_acceptance = false;
    if (constraint_set.empty() && !has_terminal_ineq)
    {
      double dJ = context.cost_ - cost_new;
      double expected = -alpha_pr * (dV_(0) + 0.5 * alpha_pr * dV_(1));
      double reduction_ratio =
          expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);
      filter_acceptance = reduction_ratio > 1e-6;
    }
    else
    {
      double expected_improvement = alpha_pr * dV_(0);
      double constraint_violation_old =
          filter_.empty() ? 0.0 : filter_.back().constraint_violation;
      double merit_function_old = context.merit_function_;

      if (theta_new > options.filter.max_violation_threshold)
      {
        if (theta_new <
            (1 - options.filter.violation_acceptance_threshold) *
                constraint_violation_old)
        {
          filter_acceptance = true;
        }
      }
      else if (std::max(theta_new, constraint_violation_old) <
                   options.filter.min_violation_for_armijo_check &&
               expected_improvement < 0)
      {
        if (phi_new <
            merit_function_old +
                options.filter.armijo_constant * expected_improvement)
        {
          filter_acceptance = true;
        }
      }
      else
      {
        if (phi_new <
                merit_function_old - options.filter.merit_acceptance_threshold *
                                         theta_new ||
            theta_new <
                (1 - options.filter.violation_acceptance_threshold) *
                    constraint_violation_old)
        {
          filter_acceptance = true;
        }
      }
    }

    if (!filter_acceptance)
    {
      return result;
    }

    result.success = true;
    result.cost = cost_new;
    result.merit_function = phi_new;
    result.theta = theta_new;
    result.constraint_violation = theta_new;
    result.inf_pr = inf_pr_new;
    result.inf_comp = inf_comp_new;
    result.dual_trajectory = Y_new;
    result.slack_trajectory = S_new;
    result.constraint_eval_trajectory = G_new;
    result.costate_trajectory = Lambda_new;
    if (has_terminal_ineq)
    {
      result.terminal_slack = S_T_new;
      result.terminal_constraint_dual = Y_T_new;
      result.terminal_constraint_value = G_T_new;
    }
    if (has_terminal_eq)
    {
      if (!result.terminal_constraint_dual.has_value())
      {
        result.terminal_constraint_dual =
            std::map<std::string, Eigen::VectorXd>{};
      }
      auto equality_duals = splitTerminalEqualityVector(context, Lambda_T_eq_new);
      result.terminal_constraint_dual->insert(equality_duals.begin(), equality_duals.end());
      if (!result.terminal_constraint_value.has_value())
      {
        result.terminal_constraint_value =
            std::map<std::string, Eigen::VectorXd>{};
      }
      auto equality_values = splitTerminalEqualityVector(context, h_T_new);
      result.terminal_constraint_value->insert(equality_values.begin(), equality_values.end());
    }
    return result;
  }

  void IPDDPSolver::applyForwardPassResult(CDDP &context, const ForwardPassResult &result)
  {
    // Call base class to update X_, U_, cost_, merit_function_, alpha_pr_, alpha_du_
    CDDPSolverBase::applyForwardPassResult(context, result);

    if (result.dual_trajectory)
    {
      Y_ = *result.dual_trajectory;
    }
    if (result.slack_trajectory)
    {
      S_ = *result.slack_trajectory;
    }
    if (result.constraint_eval_trajectory)
    {
      G_ = *result.constraint_eval_trajectory;
      G_raw_ = *result.constraint_eval_trajectory;
    }
    if (result.costate_trajectory)
    {
      Lambda_ = *result.costate_trajectory;
    }
    if (result.terminal_slack)
    {
      S_T_ = *result.terminal_slack;
    }
    if (result.terminal_constraint_dual)
    {
      for (const auto &entry : getTerminalInequalityLayout(context))
      {
        auto it = result.terminal_constraint_dual->find(entry.name);
        if (it != result.terminal_constraint_dual->end())
        {
          Y_T_[entry.name] = it->second;
        }
      }
      const auto eq_layout = getTerminalEqualityLayout(context);
      if (!eq_layout.empty())
      {
        int total_dim = getTerminalEqualityDim(context);
        Lambda_T_eq_ = Eigen::VectorXd::Zero(total_dim);
        int offset = 0;
        for (const auto &entry : eq_layout)
        {
          auto it = result.terminal_constraint_dual->find(entry.name);
          if (it != result.terminal_constraint_dual->end())
          {
            Lambda_T_eq_.segment(offset, entry.dim) = it->second;
          }
          offset += entry.dim;
        }
      }
    }
    if (result.terminal_constraint_value)
    {
      for (const auto &entry : getTerminalInequalityLayout(context))
      {
        auto it = result.terminal_constraint_value->find(entry.name);
        if (it != result.terminal_constraint_value->end())
        {
          G_T_[entry.name] = it->second;
        }
      }
    }

    context.inf_pr_ = result.inf_pr;
    context.inf_comp_ = result.inf_comp;
    phi_ = result.merit_function;
    theta_ = result.theta;

    // Update barrier parameter before convergence check (matching old solve loop timing)
    updateBarrierParameters(context, true);
  }

  bool IPDDPSolver::checkConvergence(CDDP &context, double dJ, double dL,
                                      int iter, std::string &reason)
  {
    const CDDPOptions &options = context.getOptions();
    const bool no_barrier_needed =
        context.getConstraintSet().empty() &&
        getTerminalInequalityLayout(context).empty();
    const double scaled_inf_du = computeScaledDualInfeasibility(context);
    const double scaled_inf_comp = context.inf_comp_;

    if (no_barrier_needed)
    {
      if (context.inf_pr_ < options.tolerance &&
          scaled_inf_du < options.tolerance)
      {
        reason = "OptimalSolutionFound";
        return true;
      }
      if (options.acceptable_tolerance > 0.0)
      {
        const double sqrt_atol = std::sqrt(options.acceptable_tolerance);
        bool acceptable = (context.inf_pr_ < sqrt_atol &&
                           scaled_inf_du < sqrt_atol && iter > 50);
        if (dJ > 0.0)
        {
          acceptable = acceptable ||
                       (dJ < options.acceptable_tolerance && iter > 50 &&
                        context.inf_pr_ < sqrt_atol &&
                        scaled_inf_du < sqrt_atol);
        }
        if (acceptable)
        {
          reason = "AcceptableSolutionFound";
          return true;
        }
      }
      return false;
    }

    const double tol = std::max(options.tolerance,
                                options.ipddp.barrier_tol_mult * mu_);
    if (context.inf_pr_ < tol && scaled_inf_du < tol && scaled_inf_comp < tol &&
        context.step_norm_ < options.tolerance * 10.0)
    {
      reason = "OptimalSolutionFound";
      return true;
    }

    if (options.acceptable_tolerance > 0.0)
    {
      const double accept_tol = std::sqrt(options.acceptable_tolerance);
      const double barrier_accept_tol =
          std::max(options.ipddp.barrier.mu_min_value * 100.0,
                   options.tolerance / 10.0);
      const bool acceptable_kkt =
          context.inf_pr_ < accept_tol && scaled_inf_du < accept_tol &&
          scaled_inf_comp < accept_tol;
      const bool barrier_phase_complete = mu_ <= barrier_accept_tol;
      bool acceptable =
          acceptable_kkt && barrier_phase_complete && iter > 10 &&
          std::abs(dJ) < options.acceptable_tolerance;
      acceptable = acceptable ||
                   (acceptable_kkt && barrier_phase_complete && iter >= 1 &&
                    context.step_norm_ < options.tolerance * 10.0 &&
                    context.inf_pr_ < 1e-4);
      if (acceptable)
      {
        reason = "AcceptableSolutionFound";
        return true;
      }
    }
    return false;
  }

  void IPDDPSolver::postIterationUpdate(CDDP &context, bool forward_pass_success)
  {
    // Barrier update on success is done in applyForwardPassResult (before convergence check).
    // Only update on failure here.
    if (!forward_pass_success)
    {
      updateBarrierParameters(context, false);
    }
  }

  bool IPDDPSolver::handleForwardPassFailure(CDDP &context,
                                               std::string &termination_reason)
  {
    const CDDPOptions &options = context.getOptions();
    context.increaseRegularization();

    // Extra regularization bump for terminal equality problems
    const bool no_barrier_needed =
        context.getConstraintSet().empty() &&
        getTerminalInequalityLayout(context).empty();
    if (!no_barrier_needed && getTerminalEqualityDim(context) > 0)
    {
      context.increaseRegularization();
    }

    if (context.isRegularizationLimitReached())
    {
      const double scaled_inf_du = computeScaledDualInfeasibility(context);
      const double scaled_inf_comp = context.inf_comp_;
      const double accept_tol =
          no_barrier_needed
              ? std::sqrt(std::max(options.acceptable_tolerance,
                                   options.tolerance))
              : std::max(std::sqrt(std::max(options.acceptable_tolerance,
                                            options.tolerance)),
                         options.ipddp.barrier_tol_mult * mu_);
      const bool acceptable =
          options.acceptable_tolerance > 0.0 &&
          context.inf_pr_ < accept_tol &&
          scaled_inf_du < accept_tol &&
          (no_barrier_needed || scaled_inf_comp < accept_tol);
      if (acceptable)
      {
        termination_reason = "AcceptableSolutionFound";
        return true;
      }
      termination_reason = "RegularizationLimitReached_NotConverged";
      if (options.verbose)
      {
        std::cerr << getSolverName()
                  << ": Regularization limit reached. Not converged." << std::endl;
      }
      return true; // break
    }
    return false; // continue
  }

  void IPDDPSolver::recordIterationHistory(const CDDP &context)
  {
    CDDPSolverBase::recordIterationHistory(context);
    history_.barrier_mu.push_back(mu_);
  }

  void IPDDPSolver::populateSolverSpecificSolution(CDDPSolution &solution,
                                                    const CDDP &context)
  {
    solution.final_barrier_mu = mu_;
    solution.final_primal_infeasibility = context.inf_pr_;
    solution.final_dual_infeasibility = context.inf_du_;
    solution.final_complementary_infeasibility = context.inf_comp_;
  }

  void IPDDPSolver::printIteration(int iter, const CDDP &context) const
  {
    printIterationLegacy(iter, context.cost_, context.inf_pr_, context.inf_du_,
                         context.inf_comp_, mu_, context.step_norm_,
                         context.regularization_, context.alpha_du_,
                         context.alpha_pr_);
  }

  void IPDDPSolver::printIterationLegacy(int iter, double objective, double inf_pr,
                                          double inf_du, double inf_comp, double mu,
                                          double step_norm, double regularization,
                                          double alpha_du, double alpha_pr) const
  {
    detail::printInteriorPointIteration(iter, objective, inf_pr, inf_du,
                                        inf_comp, mu, step_norm,
                                        regularization, alpha_du, alpha_pr);
  }

  void IPDDPSolver::printSolutionSummary(const CDDPSolution &solution) const
  {
    std::cout << "\n========================================\n";
    std::cout << "           IPDDP Solution Summary\n";
    std::cout << "========================================\n";

    std::cout << "Status: " << solution.status_message << "\n";
    std::cout << "Iterations: " << solution.iterations_completed << "\n";
    std::cout << "Solve Time: " << std::setprecision(2) << solution.solve_time_ms << " ms\n";
    std::cout << "Final Cost: " << std::setprecision(6) << solution.final_objective << "\n";
    std::cout << "Final Barrier mu: " << std::setprecision(2) << std::scientific
              << solution.final_barrier_mu << "\n";
    std::cout << "========================================\n\n";
  }

  // === Private helper methods ===

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

  void IPDDPSolver::precomputeConstraintGradients(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const auto &constraint_set = context.getConstraintSet();

    if (constraint_set.empty())
    {
      return;
    }

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      if (G_x_.find(constraint_name) == G_x_.end() || static_cast<int>(G_x_[constraint_name].size()) != horizon) {
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
            for (int t = start_t; t < end_t; ++t) {
              const Eigen::VectorXd &x = context.X_[t];
              const Eigen::VectorXd &u = context.U_[t];

              for (const auto &constraint_pair : constraint_set) {
                const std::string &constraint_name = constraint_pair.first;
                G_x_[constraint_name][t] =
                    constraint_pair.second->getStateJacobian(x, u);
                G_u_[constraint_name][t] =
                    constraint_pair.second->getControlJacobian(x, u);
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
            std::cerr << "IPDDP: Constraint gradients computation thread failed: "
                      << e.what() << std::endl;
          }
          throw;
        }
      }
    }
  }

  void IPDDPSolver::evaluateTrajectory(CDDP &context)
  {
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
        G_raw_[constraint_name][t] = g_val;
        G_[constraint_name][t] = g_val;
      }

      context.X_[t + 1] = context.getSystem().getDiscreteDynamics(
          x, u, t * context.getTimestep());
    }

    cost += context.getObjective().terminal_cost(context.X_.back());

    for (const auto &entry : getTerminalInequalityLayout(context))
    {
      const auto *constraint =
          dynamic_cast<const TerminalInequalityConstraint *>(
              context.getTerminalConstraintSet().at(entry.name).get());
      if (!constraint) {
        throw std::runtime_error(
            "IPDDP: terminal constraint '" + entry.name +
            "' is not a TerminalInequalityConstraint");
      }
      G_T_[entry.name] = constraint->evaluate(context.X_.back());
    }

    context.cost_ = cost;
  }

  void IPDDPSolver::evaluateTrajectoryWarmStart(CDDP &context)
  {
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
        G_raw_[constraint_name][t] = g_val;
        G_[constraint_name][t] = g_val;
      }
    }

    cost += context.getObjective().terminal_cost(context.X_.back());

    for (const auto &entry : getTerminalInequalityLayout(context))
    {
      const auto *constraint =
          dynamic_cast<const TerminalInequalityConstraint *>(
              context.getTerminalConstraintSet().at(entry.name).get());
      if (!constraint) {
        throw std::runtime_error(
            "IPDDP: terminal constraint '" + entry.name +
            "' is not a TerminalInequalityConstraint");
      }
      G_T_[entry.name] = constraint->evaluate(context.X_.back());
    }

    context.cost_ = cost;
  }

  void IPDDPSolver::initializeDualSlackVariablesWarmStart(CDDP &context)
  {
    initializeDualSlackVariables(context);
  }

  void IPDDPSolver::initializeDualSlackVariables(CDDP &context)
  {
    const CDDPOptions &options = context.getOptions();
    const int horizon = context.getHorizon();
    const auto &constraint_set = context.getConstraintSet();

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      int dual_dim = constraint_pair.second->getDualDim();

      G_raw_[constraint_name].resize(horizon);
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
        G_raw_[constraint_name][t] = g_val;
        G_[constraint_name][t] = g_val;

        Eigen::VectorXd s_init = Eigen::VectorXd::Zero(dual_dim);
        Eigen::VectorXd y_init = Eigen::VectorXd::Zero(dual_dim);

        for (int i = 0; i < dual_dim; ++i)
        {
          s_init(i) = std::max(options.ipddp.slack_var_init_scale,
                               -g_val(i) + kSlackInteriorOffset);
          y_init(i) = (mu_ * options.ipddp.dual_var_init_scale) /
                      std::max(s_init(i), EPS_SLACK);
        }
        repairWarmstartInterior(s_init, y_init, options);
        Y_[constraint_name][t] = y_init;
        S_[constraint_name][t] = s_init;
        dY_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        dS_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);

        k_y_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        K_y_[constraint_name][t] =
            Eigen::MatrixXd::Zero(dual_dim, context.getStateDim());
        k_s_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
        K_s_[constraint_name][t] =
            Eigen::MatrixXd::Zero(dual_dim, context.getStateDim());
      }
    }

    context.cost_ = context.getObjective().evaluate(context.X_, context.U_);
  }

  void IPDDPSolver::resetBarrierFilter(CDDP &context)
  {
    const bool has_terminal_ineq = !getTerminalInequalityLayout(context).empty();
    const bool has_terminal_eq = getTerminalEqualityDim(context) > 0;
    Eigen::VectorXd h_T =
        has_terminal_eq
            ? evaluateTerminalEqualityResidual(context, context.X_.back())
            : Eigen::VectorXd::Zero(0);
    const auto [inf_pr, inf_comp] = computePrimalAndComplementarity(
        context, G_, S_, Y_, mu_, has_terminal_ineq ? &G_T_ : nullptr,
        has_terminal_ineq ? &S_T_ : nullptr, has_terminal_ineq ? &Y_T_ : nullptr,
        has_terminal_eq ? &h_T : nullptr);

    context.merit_function_ = computeBarrierMerit(
        context, S_, context.cost_, has_terminal_ineq ? &S_T_ : nullptr,
        has_terminal_eq ? &Lambda_T_eq_ : nullptr,
        has_terminal_eq ? &h_T : nullptr);
    context.inf_pr_ = inf_pr;
    context.inf_comp_ = inf_comp;
    phi_ = context.merit_function_;
    theta_ =
        std::max(computeTheta(context.getOptions(), G_, S_,
                              has_terminal_ineq ? &G_T_ : nullptr,
                              has_terminal_ineq ? &S_T_ : nullptr,
                              has_terminal_eq ? &h_T : nullptr),
                 std::max(context.getOptions().ipddp.theta_0_floor, 1e-8));
    filter_.clear();
  }

  void IPDDPSolver::resetFilter(CDDP &context) { resetBarrierFilter(context); }

  bool IPDDPSolver::acceptFilterEntry(double merit_function,
                                      double constraint_violation)
  {
    return detail::acceptFilterEntry(filter_, merit_function,
                                     constraint_violation);
  }

  bool IPDDPSolver::isFilterAcceptable(double merit_function,
                                       double constraint_violation) const
  {
    if (!std::isfinite(merit_function) || !std::isfinite(constraint_violation))
    {
      return false;
    }
    constexpr double eps = 1e-12;
    if (filter_.empty())
    {
      const bool ties_current =
          std::abs(merit_function - phi_) <= eps &&
          std::abs(constraint_violation - theta_) <= eps;
      return merit_function < phi_ || constraint_violation < theta_ ||
             ties_current;
    }
    return !detail::isFilterCandidateDominated(filter_, merit_function,
                                               constraint_violation);
  }

  void IPDDPSolver::updateBarrierParameters(CDDP &context, bool forward_pass_success)
  {
    const CDDPOptions &options = context.getOptions();
    const auto &barrier_opts = options.ipddp.barrier;
    const bool no_barrier_needed =
        context.getConstraintSet().empty() &&
        getTerminalInequalityLayout(context).empty();

    if (!forward_pass_success)
    {
      return;
    }

    const double scaled_inf_du = computeScaledDualInfeasibility(context);
    const double scaled_inf_comp = context.inf_comp_;
    const double mu_old = mu_;

    if (no_barrier_needed)
    {
      mu_ = mu_old;
    }
    else if (barrier_opts.strategy == BarrierStrategy::ADAPTIVE)
    {
      const double kkt_error =
          std::max({context.inf_pr_, scaled_inf_du, scaled_inf_comp});
      const double threshold =
          std::max(barrier_opts.mu_update_factor * mu_, 2.0 * mu_);
      if (kkt_error <= threshold)
      {
        double factor = barrier_opts.mu_update_factor;
        if (mu_ > 1e-20)
        {
          const double ratio = kkt_error / std::max(mu_, 1e-20);
          if (ratio < 0.01)
          {
            factor = 0.1 * barrier_opts.mu_update_factor;
          }
          else if (ratio < 0.1)
          {
            factor = 0.3 * barrier_opts.mu_update_factor;
          }
          else if (ratio < 0.5)
          {
            factor = 0.6 * barrier_opts.mu_update_factor;
          }
        }
        const double linear = factor * mu_;
        const double superlinear = std::pow(mu_, barrier_opts.mu_update_power);
        mu_ = std::max(
            std::min(linear, superlinear),
            std::max(barrier_opts.mu_min_value, options.tolerance / 100.0));
      }
    }
    else
    {
      const double weighted_inf_du =
          scaled_inf_du * options.ipddp.barrier_update_dual_weight;
      const double kkt_error =
          std::max({context.inf_pr_, weighted_inf_du, scaled_inf_comp});
      if (kkt_error <= options.ipddp.mu_kappa_epsilon * mu_)
      {
        const double linear = barrier_opts.mu_update_factor * mu_;
        const double superlinear = std::pow(mu_, barrier_opts.mu_update_power);
        mu_ = std::max(barrier_opts.mu_min_value,
                       std::min(linear, superlinear));
      }
    }

    const bool reset_filter = (mu_ < mu_old) && (mu_ > 0.0);
    if (reset_filter)
    {
      filter_.clear();
    }
    else
    {
      acceptFilterEntry(phi_, theta_);
      if (static_cast<int>(filter_.size()) > 5)
      {
        detail::pruneFilterToBestPoints(filter_);
      }
    }

    const bool has_terminal_ineq = !getTerminalInequalityLayout(context).empty();
    const bool has_terminal_eq = getTerminalEqualityDim(context) > 0;
    Eigen::VectorXd h_T =
        has_terminal_eq
            ? evaluateTerminalEqualityResidual(context, context.X_.back())
            : Eigen::VectorXd::Zero(0);
    const auto [inf_pr, inf_comp] = computePrimalAndComplementarity(
        context, G_, S_, Y_, mu_, has_terminal_ineq ? &G_T_ : nullptr,
        has_terminal_ineq ? &S_T_ : nullptr, has_terminal_ineq ? &Y_T_ : nullptr,
        has_terminal_eq ? &h_T : nullptr);
    context.inf_pr_ = inf_pr;
    context.inf_comp_ = inf_comp;
    context.merit_function_ = computeBarrierMerit(
        context, S_, context.cost_, has_terminal_ineq ? &S_T_ : nullptr,
        has_terminal_eq ? &Lambda_T_eq_ : nullptr,
        has_terminal_eq ? &h_T : nullptr);
    phi_ = context.merit_function_;
    theta_ = std::max(computeTheta(options, G_, S_,
                                   has_terminal_ineq ? &G_T_ : nullptr,
                                   has_terminal_ineq ? &S_T_ : nullptr,
                                   has_terminal_eq ? &h_T : nullptr),
                      std::max(options.ipddp.theta_0_floor, 1e-8));
  }

  void IPDDPSolver::initializeConstraintStorage(CDDP &context)
  {
    const auto &constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();

    G_raw_.clear();
    G_.clear();
    G_x_.clear();
    G_u_.clear();
    Y_.clear();
    S_.clear();
    dY_.clear();
    dS_.clear();
    k_y_.clear();
    K_y_.clear();
    k_s_.clear();
    K_s_.clear();
    G_T_.clear();
    Y_T_.clear();
    S_T_.clear();
    dY_T_.clear();
    dS_T_.clear();

    for (const auto &constraint_pair : constraint_set)
    {
      const std::string &constraint_name = constraint_pair.first;
      G_raw_[constraint_name].resize(horizon);
      G_[constraint_name].resize(horizon);
      Y_[constraint_name].resize(horizon);
      S_[constraint_name].resize(horizon);
      dY_[constraint_name].resize(horizon);
      dS_[constraint_name].resize(horizon);
      k_y_[constraint_name].resize(horizon);
      K_y_[constraint_name].resize(horizon);
      k_s_[constraint_name].resize(horizon);
      K_s_[constraint_name].resize(horizon);
    }

    for (const auto &entry : getTerminalInequalityLayout(context))
    {
      G_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
      Y_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
      S_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
      dY_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
      dS_T_[entry.name] = Eigen::VectorXd::Zero(entry.dim);
    }
  }

  double IPDDPSolver::computeMaxConstraintViolation(const CDDP &context) const
  {
    (void)context;
    return detail::computeMaxConstraintViolation(G_);
  }

  double IPDDPSolver::computeScaledDualInfeasibility(const CDDP &context) const
  {
    return context.inf_du_;
  }

  double IPDDPSolver::computeTheta(
      const CDDPOptions &options,
      const std::map<std::string, std::vector<Eigen::VectorXd>> &constraints,
      const std::map<std::string, std::vector<Eigen::VectorXd>> &slacks,
      const std::map<std::string, Eigen::VectorXd> *terminal_constraints,
      const std::map<std::string, Eigen::VectorXd> *terminal_slacks,
      const Eigen::VectorXd *terminal_equality_residual) const
  {
    const bool use_l2 = options.ipddp.theta_norm == "l2";
    double total = 0.0;
    double max_entry = 0.0;
    for (const auto &constraint_pair : constraints)
    {
      const auto slack_it = slacks.find(constraint_pair.first);
      if (slack_it == slacks.end())
      {
        continue;
      }
      const auto &g_traj = constraint_pair.second;
      const auto &s_traj = slack_it->second;
      for (size_t t = 0; t < g_traj.size(); ++t)
      {
        const Eigen::VectorXd residual = g_traj[t] + s_traj[t];
        if (use_l2)
        {
          total += residual.squaredNorm();
        }
        else
        {
          total += residual.lpNorm<1>();
        }
        max_entry = std::max(max_entry, residual.lpNorm<Eigen::Infinity>());
      }
    }
    if (terminal_constraints != nullptr && terminal_slacks != nullptr)
    {
      for (const auto &constraint_pair : *terminal_constraints)
      {
        const auto slack_it = terminal_slacks->find(constraint_pair.first);
        if (slack_it == terminal_slacks->end())
        {
          continue;
        }
        const Eigen::VectorXd residual = constraint_pair.second + slack_it->second;
        if (use_l2)
        {
          total += residual.squaredNorm();
        }
        else
        {
          total += residual.lpNorm<1>();
        }
        max_entry = std::max(max_entry, residual.lpNorm<Eigen::Infinity>());
      }
    }
    if (terminal_equality_residual != nullptr && terminal_equality_residual->size() > 0)
    {
      if (use_l2)
      {
        total += terminal_equality_residual->squaredNorm();
      }
      else
      {
        total += terminal_equality_residual->lpNorm<1>();
      }
      max_entry =
          std::max(max_entry, terminal_equality_residual->lpNorm<Eigen::Infinity>());
    }
    const double theta = use_l2 ? std::sqrt(total) : total;
    return std::max(theta, max_entry);
  }

  double IPDDPSolver::computeBarrierMerit(
      const CDDP &context,
      const std::map<std::string, std::vector<Eigen::VectorXd>> &slacks,
      double cost,
      const std::map<std::string, Eigen::VectorXd> *terminal_slacks,
      const Eigen::VectorXd *terminal_equality_multipliers,
      const Eigen::VectorXd *terminal_equality_residual) const
  {
    double merit = cost;
    for (const auto &entry : slacks)
    {
      for (const auto &s_vec : entry.second)
      {
        merit -= mu_ * s_vec.array().max(EPS_SLACK).log().sum();
      }
    }
    if (terminal_slacks != nullptr)
    {
      for (const auto &entry : *terminal_slacks)
      {
        merit -= mu_ * entry.second.array().max(EPS_SLACK).log().sum();
      }
    }
    if (terminal_equality_multipliers != nullptr &&
        terminal_equality_residual != nullptr &&
        terminal_equality_multipliers->size() == terminal_equality_residual->size())
    {
      merit += terminal_equality_multipliers->dot(*terminal_equality_residual);
    }
    return merit;
  }

  std::pair<double, double> IPDDPSolver::computePrimalAndComplementarity(
      const CDDP &context,
      const std::map<std::string, std::vector<Eigen::VectorXd>> &constraints,
      const std::map<std::string, std::vector<Eigen::VectorXd>> &slacks,
      const std::map<std::string, std::vector<Eigen::VectorXd>> &duals,
      double mu,
      const std::map<std::string, Eigen::VectorXd> *terminal_constraints,
      const std::map<std::string, Eigen::VectorXd> *terminal_slacks,
      const std::map<std::string, Eigen::VectorXd> *terminal_duals,
      const Eigen::VectorXd *terminal_equality_residual) const
  {
    double inf_pr = 0.0;
    double inf_comp = 0.0;
    for (const auto &constraint_pair : constraints)
    {
      const auto slack_it = slacks.find(constraint_pair.first);
      const auto dual_it = duals.find(constraint_pair.first);
      if (slack_it == slacks.end() || dual_it == duals.end())
      {
        continue;
      }
      for (size_t t = 0; t < constraint_pair.second.size(); ++t)
      {
        const Eigen::VectorXd r_p = constraint_pair.second[t] + slack_it->second[t];
        const Eigen::VectorXd r_d =
            dual_it->second[t].cwiseProduct(slack_it->second[t]).array() - mu;
        inf_pr = std::max(inf_pr, r_p.lpNorm<Eigen::Infinity>());
        inf_comp = std::max(inf_comp, r_d.lpNorm<Eigen::Infinity>());
      }
    }
    if (terminal_constraints != nullptr && terminal_slacks != nullptr &&
        terminal_duals != nullptr)
    {
      for (const auto &entry : *terminal_constraints)
      {
        const auto slack_it = terminal_slacks->find(entry.first);
        const auto dual_it = terminal_duals->find(entry.first);
        if (slack_it == terminal_slacks->end() ||
            dual_it == terminal_duals->end())
        {
          continue;
        }
        const Eigen::VectorXd r_p = entry.second + slack_it->second;
        const Eigen::VectorXd r_d =
            dual_it->second.cwiseProduct(slack_it->second).array() - mu;
        inf_pr = std::max(inf_pr, r_p.lpNorm<Eigen::Infinity>());
        inf_comp = std::max(inf_comp, r_d.lpNorm<Eigen::Infinity>());
      }
    }
    if (terminal_equality_residual != nullptr && terminal_equality_residual->size() > 0)
    {
      inf_pr = std::max(
          inf_pr, terminal_equality_residual->lpNorm<Eigen::Infinity>());
    }
    return {inf_pr, inf_comp};
  }

  std::pair<double, double> IPDDPSolver::computeMaxStepSizes(
      const CDDP &context) const
  {
    const double tau =
        std::max(context.getOptions().ipddp.barrier.min_fraction_to_boundary,
                 1.0 - mu_);
    double alpha_pr = 1.0;
    double alpha_du = 1.0;
    for (const auto &constraint_pair : context.getConstraintSet())
    {
      const auto &name = constraint_pair.first;
      for (size_t t = 0; t < dS_.at(name).size(); ++t)
      {
        const Eigen::VectorXd &s = S_.at(name)[t];
        const Eigen::VectorXd &y = Y_.at(name)[t];
        const Eigen::VectorXd &ds = dS_.at(name)[t];
        const Eigen::VectorXd &dy = dY_.at(name)[t];
        for (int i = 0; i < s.size(); ++i)
        {
          if (ds(i) < 0.0)
          {
            alpha_pr = std::min(alpha_pr, -tau * s(i) / ds(i));
          }
          if (dy(i) < 0.0)
          {
            alpha_du = std::min(alpha_du, -tau * y(i) / dy(i));
          }
        }
      }
    }
    for (const auto &entry : getTerminalInequalityLayout(context))
    {
      const Eigen::VectorXd &s = S_T_.at(entry.name);
      const Eigen::VectorXd &y = Y_T_.at(entry.name);
      const Eigen::VectorXd &ds = dS_T_.at(entry.name);
      const Eigen::VectorXd &dy = dY_T_.at(entry.name);
      for (int i = 0; i < entry.dim; ++i)
      {
        if (ds(i) < 0.0)
        {
          alpha_pr = std::min(alpha_pr, -tau * s(i) / ds(i));
        }
        if (dy(i) < 0.0)
        {
          alpha_du = std::min(alpha_du, -tau * y(i) / dy(i));
        }
      }
    }
    return {std::clamp(alpha_pr, 0.0, 1.0), std::clamp(alpha_du, 0.0, 1.0)};
  }

} // namespace cddp
