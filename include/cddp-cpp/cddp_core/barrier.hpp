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

} // namespace cddp

#endif // CDDP_BARRIER_HPP