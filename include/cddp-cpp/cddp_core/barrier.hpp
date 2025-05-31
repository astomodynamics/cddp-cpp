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
#include <tuple>

namespace cddp {

/**
 * @class LogBarrier
 * @brief Implements a log-barrier function for inequality constraints of the form
 *        lower_bound <= g(x,u) <= upper_bound (element-wise).
 */
class LogBarrier {
public:
    /**
     * @brief Construct a log barrier with given parameters.
     *
     * @param barrier_coeff     Coefficient controlling barrier steepness.
     * @param relaxation_coeff  Relaxation factor for numerical stability
     * @param barrier_order    Order of barrier polynomial 
     */
    LogBarrier(double barrier_coeff = 1e-2, 
               double relaxation_coeff = 1.0,
               int barrier_order = 2,
               bool is_relaxed_log_barrier = false);

    /**
     * @brief Evaluate the barrier function for a given constraint.
     * 
     * @param constraint A reference to the constraint object whose output is bounded.
     * @param state      Current state vector.
     * @param control    Current control vector.
     * @return Barrier function value, or \f$+\infty\f$ if infeasible.
     */
    double evaluate(const Constraint& constraint, 
                    const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) const;

    /**
     * @brief Compute the gradient (first derivative) w.r.t. state and control.
     *
     * @param constraint    Constraint being enforced.
     * @param state         Current state vector.
     * @param control       Current control vector.
     * @param barrier_value (Optional) The last evaluated barrier cost 
     * @return A tuple of two vectors: `(dBarrier/dx, dBarrier/du)`.
     */
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> getGradients(
        const Constraint& constraint, 
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control,
        double barrier_value) const;

    /**
     * @brief Compute second derivatives (Hessians) of the barrier function.
     *
     * @param constraint    Constraint being enforced.
     * @param state         Current state vector.
     * @param control       Current control vector.
     * @param barrier_value (Optional) The last evaluated barrier cost 
     * @return A tuple of matrices `(Hxx, Huu, Hxu)` in that order.
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getHessians(
        const Constraint& constraint,
        const Eigen::VectorXd& state, 
        const Eigen::VectorXd& control,
        double barrier_value) const;
    /**
     * @brief Get the barrier coefficient.
     * @return The coefficient controlling the steepness of the log penalty.
     */
    double getBarrierCoeff() const {
        return barrier_coeff_;
    }

    /**
     * @brief Set the barrier coefficient.
     * @param barrier_coeff New log-barrier penalty coefficient.
     */
    void setBarrierCoeff(double barrier_coeff) {
        barrier_coeff_ = barrier_coeff;
    }

    /**
     * @brief Get the current relaxation coefficient.
     * @return relaxation_coeff
     */
    double getRelaxationCoeff() const {
        return relaxation_coeff_;
    }

    /**
     * @brief Set the relaxation coefficient.
     * @param relaxation_coeff 
     */
    void setRelaxationCoeff(double relaxation_coeff) {
        relaxation_coeff_ = relaxation_coeff;
    }

private:
    double barrier_coeff_;        ///< Coefficient controlling barrier steepness
    double relaxation_coeff_;     ///< Relaxation factor for numerical stability  
    int barrier_order_;           ///< Order of barrier polynomial
    bool is_relaxed_log_barrier_; ///< Use relaxed log-barrier method
};

/**
 * @class RelaxedLogBarrier
 * @brief Implements the relaxed log-barrier function for inequality constraints
 *        of the form lower_bound <= g(x,u) <= upper_bound (element-wise),
 *        based on the formulation:
 *        beta_delta(z) = -log(z)                               if z > delta
 *                        0.5 * [((z - 2*delta)/delta)^2 - 1] - log(delta) if z <= delta
 */
class RelaxedLogBarrier {
public:
    /**
     * @brief Construct a relaxed log barrier with given parameters.
     *
     * @param barrier_coeff     Coefficient multiplying the barrier penalty (mu_penalty).
     * @param relaxation_delta  The relaxation parameter delta.
     */
    RelaxedLogBarrier(double barrier_coeff = 1e-2, 
                      double relaxation_delta = 1e-1)
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
    double evaluate(const Constraint& constraint, 
                    const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) const {
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
                if (s_L < 0 && std::abs(s_L) > 1e-9) { // Constraint violated significantly for lower bound
                     // Apply a large penalty or handle as per specific strategy for infeasible points.
                     // For now, let it pass to calculate_beta_derivatives which might produce large values.
                }
                calculate_beta_derivatives(s_L, relaxation_delta_, beta_val_L, beta_prime_L, beta_double_prime_L);
                total_barrier_cost += beta_val_L;
            }

            if (U(i) != std::numeric_limits<double>::infinity()) {
                double s_U = U(i) - g_val(i);
                 if (s_U < 0 && std::abs(s_U) > 1e-9) { // Constraint violated significantly for upper bound
                     // Similar handling for infeasible points.
                 }
                calculate_beta_derivatives(s_U, relaxation_delta_, beta_val_U, beta_prime_U, beta_double_prime_U);
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
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> getGradients(
        const Constraint& constraint, 
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control) const {
        
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
                calculate_beta_derivatives(s_L, relaxation_delta_, beta_val_L, beta_prime_L, beta_double_prime_L);
                dCost_dg_i += beta_prime_L;
            }

            if (U(i) != std::numeric_limits<double>::infinity()) {
                double s_U = U(i) - g_val(i);
                calculate_beta_derivatives(s_U, relaxation_delta_, beta_val_U, beta_prime_U, beta_double_prime_U);
                dCost_dg_i -= beta_prime_U; // ds_U/dg_i is -1
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
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getHessians(
        const Constraint& constraint,
        const Eigen::VectorXd& state, 
        const Eigen::VectorXd& control) const {

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
        Eigen::MatrixXd Hux = Eigen::MatrixXd::Zero(control_dim, state_dim); // Will store d^2C / (du dx)

        std::vector<Eigen::MatrixXd> Gxx_constraint_vec, Guu_constraint_vec, Gux_constraint_vec;
        bool constraint_provides_hessians = true;
        try {
            auto hess_tuple = constraint.getHessians(state, control);
            Gxx_constraint_vec = std::get<0>(hess_tuple);
            Guu_constraint_vec = std::get<1>(hess_tuple);
            Gux_constraint_vec = std::get<2>(hess_tuple); // This is d^2g_i/dudx
            if (Gxx_constraint_vec.size() != constraint_dim || 
                Guu_constraint_vec.size() != constraint_dim || 
                Gux_constraint_vec.size() != constraint_dim) {
                // Mismatch in expected number of Hessian matrices from constraint
                constraint_provides_hessians = false; 
            }
        } catch (const std::logic_error& e) {
            constraint_provides_hessians = false;
        }

        for (int i = 0; i < constraint_dim; ++i) {
            double beta_val_L, beta_prime_L, beta_double_prime_L;
            double beta_val_U, beta_prime_U, beta_double_prime_U;

            double term1_coeff_i = 0.0; // For Gx^T * D * Gx term
            double term2_coeff_i = 0.0; // For coeff * Gxx_constraint term

            if (L(i) != -std::numeric_limits<double>::infinity()) {
                double s_L = g_val(i) - L(i);
                calculate_beta_derivatives(s_L, relaxation_delta_, beta_val_L, beta_prime_L, beta_double_prime_L);
                term1_coeff_i += beta_double_prime_L;
                term2_coeff_i += beta_prime_L;
            }

            if (U(i) != std::numeric_limits<double>::infinity()) {
                double s_U = U(i) - g_val(i);
                calculate_beta_derivatives(s_U, relaxation_delta_, beta_val_U, beta_prime_U, beta_double_prime_U);
                term1_coeff_i += beta_double_prime_U; // (ds_U/dg_i)^2 = (-1)^2 = 1
                term2_coeff_i -= beta_prime_U;       // ds_U/dg_i = -1
            }
            
            // Term 1: Gx.row(i).transpose() * D_i * Gx.row(i)
            Hxx += term1_coeff_i * Gx.row(i).transpose() * Gx.row(i);
            Huu += term1_coeff_i * Gu.row(i).transpose() * Gu.row(i);
            Hux += term1_coeff_i * Gu.row(i).transpose() * Gx.row(i);


            // Term 2: coeff_i * Gxx_constraint[i]
            if (constraint_provides_hessians) {
                if (i < Gxx_constraint_vec.size() && Gxx_constraint_vec[i].size() > 0 &&
                    Gxx_constraint_vec[i].rows() == state_dim && Gxx_constraint_vec[i].cols() == state_dim) {
                    Hxx += term2_coeff_i * Gxx_constraint_vec[i];
                }
                if (i < Guu_constraint_vec.size() && Guu_constraint_vec[i].size() > 0 &&
                    Guu_constraint_vec[i].rows() == control_dim && Guu_constraint_vec[i].cols() == control_dim) {
                    Huu += term2_coeff_i * Guu_constraint_vec[i];
                }
                if (i < Gux_constraint_vec.size() && Gux_constraint_vec[i].size() > 0 &&
                    Gux_constraint_vec[i].rows() == control_dim && Gux_constraint_vec[i].cols() == state_dim) {
                    Hux += term2_coeff_i * Gux_constraint_vec[i];
                }
            }
        }
        
        return std::make_tuple(barrier_coeff_ * Hxx, barrier_coeff_ * Huu, barrier_coeff_ * Hux);
    }

    /**
     * @brief Get the barrier coefficient (mu_penalty).
     * @return The coefficient multiplying the barrier penalty.
     */
    double getBarrierCoeff() const {
        return barrier_coeff_;
    }

    /**
     * @brief Set the barrier coefficient (mu_penalty).
     * @param barrier_coeff New barrier penalty coefficient.
     */
    void setBarrierCoeff(double barrier_coeff) {
        barrier_coeff_ = barrier_coeff;
    }

    /**
     * @brief Get the current relaxation parameter delta.
     * @return relaxation_delta_
     */
    double getRelaxationDelta() const {
        return relaxation_delta_;
    }

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
     *                 0.5 * [((z - 2*delta)/delta)^2 - 1] - log(delta) if z <= delta
     * Assumes delta > 0. Handles z approaching 0 for the -log(z) case with care.
     */
    void calculate_beta_derivatives(double z, double delta,
                                    double& beta_val, 
                                    double& beta_prime, 
                                    double& beta_double_prime) const {
        // Assertion: delta is already checked to be > 0 in constructor/setter.
        
        if (z > delta) {
            if (z <= 1e-12) { // z is very close to or at zero (or negative, though less likely for a valid slack)
                // This region implies a highly violated constraint if delta is also small,
                // or an extremely satisfied constraint if z is positive and extremely small.
                // Standard log barrier -> infinity.
                // We use a large finite value to maintain numerical stability.
                // The exact behavior might need tuning or a more sophisticated regularization
                // if z can validly be very close to 0 and > delta.
                // This typically occurs if delta itself is extremely small.
                beta_val = -std::log(1e-12); // Large positive
                beta_prime = -1.0 / 1e-12;    // Large negative
                beta_double_prime = 1.0 / (1e-12 * 1e-12); // Very large positive
            } else {
                beta_val = -std::log(z);
                beta_prime = -1.0 / z;
                beta_double_prime = 1.0 / (z * z);
            }
        } else { // z <= delta
            double term_val = (z - 2.0 * delta); // Numerator for term
            // If delta is extremely small, term_val / delta can be large.
            // (z - 2*delta)/delta = z/delta - 2
            double term_div_delta = term_val / delta;

            beta_val = 0.5 * (term_div_delta * term_div_delta - 1.0) - std::log(delta);
            beta_prime = term_div_delta / delta; // (z - 2*delta) / (delta*delta)
            beta_double_prime = 1.0 / (delta * delta);
        }
    }

    double barrier_coeff_;    ///< Coefficient multiplying the barrier penalty (mu_penalty)
    double relaxation_delta_; ///< The relaxation parameter delta
};

} // namespace cddp

#endif // CDDP_BARRIER_HPP