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
#ifndef CDDP_BARRIER_HPP
#define CDDP_BARRIER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include "constraint.hpp"

namespace cddp {

/**
 * @class RelaxedLogBarrier
 * @brief Implements the relaxed log-barrier function for inequality constraints
 *        of the form lower_bound <= g(x,u) <= upper_bound (element-wise),
 *        based on the formulation:
 *        beta_delta(z) = -log(z)                               if z > delta
 *                        0.5 * [((z - 2*delta)/delta)^2 - 1] - log(delta) if z
 * <= delta
 */
class RelaxedLogBarrier {
public:
  /**
   * @brief Construct a relaxed log barrier with given parameters.
   *
   * @param barrier_coeff     Coefficient multiplying the barrier penalty
   * (mu_penalty).
   * @param relaxation_delta  The relaxation parameter delta.
   */
  RelaxedLogBarrier(double barrier_coeff = 1e-2, double relaxation_delta = 1e-1)
      : barrier_coeff_(barrier_coeff), relaxation_delta_(relaxation_delta) {
    if (relaxation_delta_ <= 0) {
      throw std::invalid_argument("Relaxation delta must be positive.");
    }
  }

  /**
   * @brief Evaluate the relaxed log-barrier function for a given constraint.
   *
   * @param constraint A reference to the constraint object.
   * @param state      Current state vector.
   * @param control    Current control vector.
   * @return Barrier function value.
   */
  double evaluate(const Constraint &constraint, const Eigen::VectorXd &state,
                  const Eigen::VectorXd &control) const {
    Eigen::VectorXd g_val = constraint.evaluate(state, control);
    Eigen::VectorXd L = constraint.getLowerBound();
    Eigen::VectorXd U = constraint.getUpperBound();
    int constraint_dim = g_val.size();
    double total_barrier_cost = 0.0;

    for (int i = 0; i < constraint_dim; ++i) {
      double beta_val_L = 0.0, beta_prime_L = 0.0, beta_double_prime_L = 0.0;
      double beta_val_U = 0.0, beta_prime_U = 0.0, beta_double_prime_U = 0.0;

      if (L(i) != -std::numeric_limits<double>::infinity()) {
        double s_L = g_val(i) - L(i);
        if (s_L < 0 && std::abs(s_L) > 1e-9) {
        }
        calculate_beta_derivatives(s_L, relaxation_delta_, beta_val_L,
                                   beta_prime_L, beta_double_prime_L);
        total_barrier_cost += beta_val_L;
      }

      if (U(i) != std::numeric_limits<double>::infinity()) {
        double s_U = U(i) - g_val(i);
        calculate_beta_derivatives(s_U, relaxation_delta_, beta_val_U,
                                   beta_prime_U, beta_double_prime_U);
        total_barrier_cost += beta_val_U;
      }
    }
    return barrier_coeff_ * total_barrier_cost;
  }

  /**
   * @brief Compute the gradient (first derivative) w.r.t. state and control.
   *
   * @param constraint    Constraint being enforced.
   * @param state         Current state vector.
   * @param control       Current control vector.
   * @return A tuple of two vectors: `(dBarrier/dx, dBarrier/du)`.
   */
  std::tuple<Eigen::VectorXd, Eigen::VectorXd>
  getGradients(const Constraint &constraint, const Eigen::VectorXd &state,
               const Eigen::VectorXd &control) const {

    Eigen::VectorXd g_val = constraint.evaluate(state, control);
    Eigen::VectorXd L = constraint.getLowerBound();
    Eigen::VectorXd U = constraint.getUpperBound();
    Eigen::MatrixXd Gx = constraint.getStateJacobian(state, control);
    Eigen::MatrixXd Gu = constraint.getControlJacobian(state, control);

    int state_dim = state.size();
    int control_dim = control.size();
    int constraint_dim = g_val.size();

    Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(state_dim);
    Eigen::VectorXd grad_u = Eigen::VectorXd::Zero(control_dim);

    for (int i = 0; i < constraint_dim; ++i) {
      double beta_val_L, beta_prime_L, beta_double_prime_L;
      double beta_val_U, beta_prime_U, beta_double_prime_U;

      double dCost_dg_i = 0.0;

      if (L(i) != -std::numeric_limits<double>::infinity()) {
        double s_L = g_val(i) - L(i);
        calculate_beta_derivatives(s_L, relaxation_delta_, beta_val_L,
                                   beta_prime_L, beta_double_prime_L);
        dCost_dg_i += beta_prime_L;
      }

      if (U(i) != std::numeric_limits<double>::infinity()) {
        double s_U = U(i) - g_val(i);
        calculate_beta_derivatives(s_U, relaxation_delta_, beta_val_U,
                                   beta_prime_U, beta_double_prime_U);
        dCost_dg_i -= beta_prime_U;
      }

      grad_x += dCost_dg_i * Gx.row(i).transpose();
      grad_u += dCost_dg_i * Gu.row(i).transpose();
    }

    return std::make_tuple(barrier_coeff_ * grad_x, barrier_coeff_ * grad_u);
  }

  /**
   * @brief Compute second derivatives (Hessians) of the barrier function.
   *
   * @param constraint    Constraint being enforced.
   * @param state         Current state vector.
   * @param control       Current control vector.
   * @return A tuple of matrices `(Hxx, Huu, Hxu)` in that order.
   */
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
  getHessians(const Constraint &constraint, const Eigen::VectorXd &state,
              const Eigen::VectorXd &control) const {

    Eigen::VectorXd g_val = constraint.evaluate(state, control);
    Eigen::VectorXd L = constraint.getLowerBound();
    Eigen::VectorXd U = constraint.getUpperBound();
    Eigen::MatrixXd Gx = constraint.getStateJacobian(state, control);
    Eigen::MatrixXd Gu = constraint.getControlJacobian(state, control);

    int state_dim = state.size();
    int control_dim = control.size();
    int constraint_dim = g_val.size();

    Eigen::MatrixXd Hxx = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd Huu = Eigen::MatrixXd::Zero(control_dim, control_dim);
    Eigen::MatrixXd Hux = Eigen::MatrixXd::Zero(control_dim, state_dim);

    std::vector<Eigen::MatrixXd> Gxx_constraint_vec, Guu_constraint_vec,
        Gux_constraint_vec;
    bool constraint_provides_hessians = true;
    try {
      auto hess_tuple = constraint.getHessians(state, control);
      Gxx_constraint_vec = std::get<0>(hess_tuple);
      Guu_constraint_vec = std::get<1>(hess_tuple);
      Gux_constraint_vec = std::get<2>(hess_tuple);
      if (Gxx_constraint_vec.size() != constraint_dim ||
          Guu_constraint_vec.size() != constraint_dim ||
          Gux_constraint_vec.size() != constraint_dim) {
        constraint_provides_hessians = false;
      }
    } catch (const std::logic_error &e) {
      constraint_provides_hessians = false;
    }

    for (int i = 0; i < constraint_dim; ++i) {
      double beta_val_L, beta_prime_L, beta_double_prime_L;
      double beta_val_U, beta_prime_U, beta_double_prime_U;

      double term1_coeff_i = 0.0;
      double term2_coeff_i = 0.0;

      if (L(i) != -std::numeric_limits<double>::infinity()) {
        double s_L = g_val(i) - L(i);
        calculate_beta_derivatives(s_L, relaxation_delta_, beta_val_L,
                                   beta_prime_L, beta_double_prime_L);
        term1_coeff_i += beta_double_prime_L;
        term2_coeff_i += beta_prime_L;
      }

      if (U(i) != std::numeric_limits<double>::infinity()) {
        double s_U = U(i) - g_val(i);
        calculate_beta_derivatives(s_U, relaxation_delta_, beta_val_U,
                                   beta_prime_U, beta_double_prime_U);
        term1_coeff_i += beta_double_prime_U;
        term2_coeff_i -= beta_prime_U;
      }

      Hxx += term1_coeff_i * Gx.row(i).transpose() * Gx.row(i);
      Huu += term1_coeff_i * Gu.row(i).transpose() * Gu.row(i);
      Hux += term1_coeff_i * Gu.row(i).transpose() * Gx.row(i);

      if (constraint_provides_hessians) {
        if (i < Gxx_constraint_vec.size() && Gxx_constraint_vec[i].size() > 0 &&
            Gxx_constraint_vec[i].rows() == state_dim &&
            Gxx_constraint_vec[i].cols() == state_dim) {
          Hxx += term2_coeff_i * Gxx_constraint_vec[i];
        }
        if (i < Guu_constraint_vec.size() && Guu_constraint_vec[i].size() > 0 &&
            Guu_constraint_vec[i].rows() == control_dim &&
            Guu_constraint_vec[i].cols() == control_dim) {
          Huu += term2_coeff_i * Guu_constraint_vec[i];
        }
        if (i < Gux_constraint_vec.size() && Gux_constraint_vec[i].size() > 0 &&
            Gux_constraint_vec[i].rows() == control_dim &&
            Gux_constraint_vec[i].cols() == state_dim) {
          Hux += term2_coeff_i * Gux_constraint_vec[i];
        }
      }
    }

    return std::make_tuple(barrier_coeff_ * Hxx, barrier_coeff_ * Huu,
                           barrier_coeff_ * Hux);
  }

  /**
   * @brief Get the barrier coefficient (mu_penalty).
   * @return The coefficient multiplying the barrier penalty.
   */
  double getBarrierCoeff() const { return barrier_coeff_; }

  /**
   * @brief Set the barrier coefficient (mu_penalty).
   * @param barrier_coeff New barrier penalty coefficient.
   */
  void setBarrierCoeff(double barrier_coeff) { barrier_coeff_ = barrier_coeff; }

  /**
   * @brief Get the current relaxation parameter delta.
   * @return relaxation_delta_
   */
  double getRelaxationDelta() const { return relaxation_delta_; }

  /**
   * @brief Set the relaxation parameter delta.
   * @param relaxation_delta New relaxation delta.
   */
  void setRelaxationDelta(double relaxation_delta) {
    if (relaxation_delta <= 0) {
      throw std::invalid_argument("Relaxation delta must be positive.");
    }
    relaxation_delta_ = relaxation_delta;
  }

private:
  /**
   * @brief Calculates beta_delta(z) and its first two derivatives.
   * beta_delta(z) = -log(z)                               if z > delta
   *                 0.5 * [((z - 2*delta)/delta)^2 - 1] - log(delta) if z <=
   * delta Assumes delta > 0. Handles z approaching 0 for the -log(z) case with
   * care.
   */
  void calculate_beta_derivatives(double z, double delta, double &beta_val,
                                  double &beta_prime,
                                  double &beta_double_prime) const {
    if (z > delta) {
      if (z <= 1e-12) {
        beta_val = -std::log(1e-12);
        beta_prime = -1.0 / 1e-12;
        beta_double_prime = 1.0 / (1e-12 * 1e-12);
      } else {
        beta_val = -std::log(z);
        beta_prime = -1.0 / z;
        beta_double_prime = 1.0 / (z * z);
      }
    } else { // z <= delta
      double term_val = (z - 2.0 * delta);
      double term_div_delta = term_val / delta;

      beta_val =
          0.5 * (term_div_delta * term_div_delta - 1.0) - std::log(delta);
      beta_prime = term_div_delta / delta;
      beta_double_prime = 1.0 / (delta * delta);
    }
  }

  double barrier_coeff_;    ///< Coefficient multiplying the barrier penalty
                            ///< (mu_penalty)
  double relaxation_delta_; ///< The relaxation parameter delta
};

/**
 * @class DiscreteBarrierState
 * @brief Manages discrete barrier states for DBAS-DDP implementation.
 *
 * This class handles the evolution and management of discrete barrier states
 * that are explicitly integrated as part of the augmented state in DBAS-DDP.
 */
class DiscreteBarrierState {
public:
  /**
   * @brief Construct discrete barrier state manager.
   *
   * @param constraint_dim Dimension of the constraint being handled.
   * @param init_value Initial value for barrier states.
   * @param decay_rate Decay rate for barrier state evolution.
   * @param weight Weight for barrier state dynamics penalty.
   */
  DiscreteBarrierState(int constraint_dim, double init_value = 1.0,
                       double decay_rate = 0.1, double weight = 1.0)
      : constraint_dim_(constraint_dim), init_value_(init_value),
        decay_rate_(decay_rate), weight_(weight) {
    if (constraint_dim <= 0) {
      throw std::invalid_argument("Constraint dimension must be positive.");
    }
    if (init_value <= 0.0) {
      throw std::invalid_argument("Initial value must be positive.");
    }
    if (decay_rate < 0.0) {
      throw std::invalid_argument("Decay rate must be non-negative.");
    }
    if (weight <= 0.0) {
      throw std::invalid_argument("Weight must be positive.");
    }
  }

  /**
   * @brief Initialize barrier states for a given horizon.
   *
   * @param horizon Time horizon.
   * @return Vector of initialized barrier state vectors.
   */
  std::vector<Eigen::VectorXd> initializeBarrierStates(int horizon) const {
    std::vector<Eigen::VectorXd> barrier_states(horizon);
    for (int t = 0; t < horizon; ++t) {
      barrier_states[t] =
          Eigen::VectorXd::Constant(constraint_dim_, init_value_);
    }
    return barrier_states;
  }

  /**
   * @brief Compute constraint violations from constraint evaluation and bounds.
   *
   * @param g_val Constraint evaluation g(x,u).
   * @param lower_bound Lower bounds for constraints.
   * @param upper_bound Upper bounds for constraints.
   * @return Constraint violations (positive values indicate violations).
   */
  Eigen::VectorXd
  computeConstraintViolations(const Eigen::VectorXd &g_val,
                              const Eigen::VectorXd &lower_bound,
                              const Eigen::VectorXd &upper_bound) const {
    if (g_val.size() != constraint_dim_ ||
        lower_bound.size() != constraint_dim_ ||
        upper_bound.size() != constraint_dim_) {
      throw std::invalid_argument(
          "Dimension mismatch in constraint violation computation.");
    }

    Eigen::VectorXd violations = Eigen::VectorXd::Zero(constraint_dim_);

    for (int i = 0; i < constraint_dim_; ++i) {
      // Lower bound violation
      if (g_val(i) < lower_bound(i)) {
        violations(i) = lower_bound(i) - g_val(i);
      }
      // Upper bound violation
      else if (g_val(i) > upper_bound(i)) {
        violations(i) = g_val(i) - upper_bound(i);
      }
      // No violation
      else {
        violations(i) = 0.0;
      }
    }

    return violations;
  }

  /**
   * @brief Update barrier state based on constraint violation.
   *
   * @param current_barrier_state Current barrier state.
   * @param constraint_violation Current constraint violation.
   * @param timestep Integration timestep.
   * @return Updated barrier state.
   */
  Eigen::VectorXd
  updateBarrierState(const Eigen::VectorXd &current_barrier_state,
                     const Eigen::VectorXd &constraint_violation,
                     double timestep) const {
    if (current_barrier_state.size() != constraint_dim_ ||
        constraint_violation.size() != constraint_dim_) {
      throw std::invalid_argument(
          "Dimension mismatch in barrier state update.");
    }
    if (timestep < 0.0) {
      throw std::invalid_argument("Timestep must be non-negative.");
    }

    Eigen::VectorXd next_state = current_barrier_state;

    // Discrete barrier state dynamics: exponential decay with constraint-based
    // updates
    for (int i = 0; i < constraint_dim_; ++i) {
      // Exponential decay with numerical stability
      double decay_factor = std::exp(-decay_rate_ * timestep);
      decay_factor = std::max(decay_factor, 1e-10); // Prevent complete decay
      next_state(i) *= decay_factor;

      // Add constraint violation feedback
      if (constraint_violation(i) > 0.0) {
        double violation_term = weight_ * constraint_violation(i) * timestep;
        next_state(i) += violation_term;
      }

      // Ensure barrier state remains positive and bounded
      next_state(i) = std::clamp(next_state(i), 1e-8, 1e8);
    }

    return next_state;
  }

  /**
   * @brief Compute barrier state dynamics jacobian w.r.t. constraint violation.
   *
   * @param current_barrier_state Current barrier state.
   * @param timestep Integration timestep.
   * @return Jacobian matrix.
   */
  Eigen::MatrixXd
  getBarrierStateDynamicsJacobian(const Eigen::VectorXd &current_barrier_state,
                                  double timestep) const {
    Eigen::MatrixXd jacobian =
        Eigen::MatrixXd::Zero(constraint_dim_, constraint_dim_);

    for (int i = 0; i < constraint_dim_; ++i) {
      jacobian(i, i) =
          weight_ * timestep; // Derivative w.r.t. constraint violation
    }

    return jacobian;
  }

  /**
   * @brief Compute barrier state penalty cost.
   *
   * @param barrier_state Current barrier state.
   * @param reference_barrier_state Reference barrier state (typically zero).
   * @return Barrier state penalty cost.
   */
  double
  evaluateBarrierStateCost(const Eigen::VectorXd &barrier_state,
                           const Eigen::VectorXd &reference_barrier_state =
                               Eigen::VectorXd()) const {
    Eigen::VectorXd ref = reference_barrier_state;
    if (ref.size() == 0) {
      ref = Eigen::VectorXd::Zero(constraint_dim_);
    }

    if (barrier_state.size() != constraint_dim_ ||
        ref.size() != constraint_dim_) {
      throw std::invalid_argument(
          "Dimension mismatch in barrier state cost evaluation.");
    }

    Eigen::VectorXd diff = barrier_state - ref;
    return 0.5 * weight_ * diff.squaredNorm();
  }

  /**
   * @brief Compute gradient of barrier state cost w.r.t. barrier state.
   *
   * @param barrier_state Current barrier state.
   * @param reference_barrier_state Reference barrier state.
   * @return Gradient vector.
   */
  Eigen::VectorXd
  getBarrierStateCostGradient(const Eigen::VectorXd &barrier_state,
                              const Eigen::VectorXd &reference_barrier_state =
                                  Eigen::VectorXd()) const {
    Eigen::VectorXd ref = reference_barrier_state;
    if (ref.size() == 0) {
      ref = Eigen::VectorXd::Zero(constraint_dim_);
    }

    if (barrier_state.size() != constraint_dim_ ||
        ref.size() != constraint_dim_) {
      throw std::invalid_argument(
          "Dimension mismatch in barrier state gradient computation.");
    }

    return weight_ * (barrier_state - ref);
  }

  /**
   * @brief Compute Hessian of barrier state cost w.r.t. barrier state.
   *
   * @param barrier_state Current barrier state (unused for quadratic cost).
   * @return Hessian matrix.
   */
  Eigen::MatrixXd
  getBarrierStateCostHessian(const Eigen::VectorXd &barrier_state) const {
    return weight_ *
           Eigen::MatrixXd::Identity(constraint_dim_, constraint_dim_);
  }

  // Getters and setters
  int getConstraintDim() const { return constraint_dim_; }
  double getInitValue() const { return init_value_; }
  double getDecayRate() const { return decay_rate_; }
  double getWeight() const { return weight_; }

  void setInitValue(double init_value) { init_value_ = init_value; }
  void setDecayRate(double decay_rate) { decay_rate_ = decay_rate; }
  void setWeight(double weight) { weight_ = weight; }

private:
  int constraint_dim_; ///< Dimension of the constraint
  double init_value_;  ///< Initial value for barrier states
  double decay_rate_;  ///< Decay rate for barrier state evolution
  double weight_;      ///< Weight for barrier state dynamics penalty
};

} // namespace cddp

#endif // CDDP_BARRIER_HPP