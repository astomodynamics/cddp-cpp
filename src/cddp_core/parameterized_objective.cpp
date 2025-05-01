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

#include <iostream> // For error messages
#include <stdexcept> // For exceptions
#include "cddp_core/parameterized_objective.hpp"

namespace cddp {
ParameterizedQuadraticObjective::ParameterizedQuadraticObjective(
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& Qf,
    int parameter_dim,
    const Eigen::VectorXd& reference_state, // Default running target
    const std::vector<Eigen::VectorXd>& reference_states,
    double timestep)
    : ParameterizedObjective(parameter_dim), Q_(Q), R_(R), Qf_(Qf), reference_state_(reference_state), reference_states_(reference_states), timestep_(timestep) {

    // Scale Q and R by timestep
    Q_ = Q * timestep_;
    R_ = R * timestep_;

    // Dimension checks
    if (Q.rows() != Q.cols()) {
        throw std::invalid_argument("ParameterizedQuadraticObjective: Q matrix must be square");
    }
    if (R.rows() != R.cols()) {
        throw std::invalid_argument("ParameterizedQuadraticObjective: R matrix must be square");
    }
    if (Qf.rows() != Qf.cols()) {
        throw std::invalid_argument("ParameterizedQuadraticObjective: Qf matrix must be square");
    }
     if (Q.rows() != Qf.rows()) {
        throw std::invalid_argument("ParameterizedQuadraticObjective: Q and Qf must have the same dimensions (state dimension)");
    }
    if (Qf.rows() != parameter_dim) {
        throw std::invalid_argument("ParameterizedQuadraticObjective: Qf dimension (state dimension) must match parameter_dim, as parameter defines final target state.");
    }
    if (reference_state_.size() != Qf.rows()) {
        throw std::invalid_argument("ParameterizedQuadraticObjective: reference_state dimension must match Qf dimension (state dimension).");
    }
    // Check reference states dimensions if provided
    if (!reference_states_.empty()) {
        for(const auto& ref_state : reference_states_) {
            if (ref_state.size() != Q.rows()) {
                 throw std::invalid_argument("ParameterizedQuadraticObjective: Reference states must have dimension matching Q.");
            }
        }
    }
}

// Parameterized running cost: l(x, u, p, t)
double ParameterizedQuadraticObjective::running_cost(const Eigen::VectorXd& state,
                                                      const Eigen::VectorXd& control,
                                                      const Eigen::VectorXd& parameter,
                                                      int index) const {
    // Parameter argument is ignored here. Reference is determined by reference_states_ or reference_state_.
    Eigen::VectorXd state_error;
    if (reference_states_.size() > index) {
        state_error = state - reference_states_[index];
    } else {
        // Otherwise, use the default reference state
        state_error = state - reference_state_;
    }
    double control_cost = (R_.rows() > 0 && R_.cols() > 0) ? (control.transpose() * R_ * control).value() : 0.0;
    return ((state_error.transpose() * Q_ * state_error).value() + control_cost);
}

// Parameterized final/terminal cost: lf(x_T, p)
// Parameter argument is ignored here, uses nominal reference_state_
double ParameterizedQuadraticObjective::terminal_cost(const Eigen::VectorXd& final_state,
                                                        const Eigen::VectorXd& parameter) const {
     // Parameter argument is ignored.
     if (final_state.size() != reference_state_.size()) {
         throw std::runtime_error("terminal_cost final_state dim does not match reference_state_ dim.");
     }
    Eigen::VectorXd state_error = final_state - reference_state_;
    return (state_error.transpose() * Qf_ * state_error).value();
}

// Gradient of running cost w.r.t state: dl/dx (p)
Eigen::VectorXd ParameterizedQuadraticObjective::getRunningCostStateGradient(const Eigen::VectorXd& state,
                                                                             const Eigen::VectorXd& control,
                                                                             const Eigen::VectorXd& parameter,
                                                                             int index) const {
    // Parameter argument is ignored here.
    Eigen::VectorXd state_error;
    if (reference_states_.size() > index) {
        state_error = state - reference_states_[index];
    } else {
        state_error = state - reference_state_;
    }
    return 2.0 * Q_ * state_error;
}

// Gradient of running cost w.r.t control: dl/du (p)
Eigen::VectorXd ParameterizedQuadraticObjective::getRunningCostControlGradient(const Eigen::VectorXd& state,
                                                                                const Eigen::VectorXd& control,
                                                                                const Eigen::VectorXd& parameter,
                                                                                int index) const {
    // Parameter argument is ignored here. Control cost doesn't depend on parameter p or reference state.
    // Handle 0 control dimension case where R might be 0x0
    if (R_.rows() > 0 && R_.cols() > 0) {
        return 2.0 * R_ * control;
    } else {
        return Eigen::VectorXd::Zero(0); 
    }
}

// Gradient of running cost w.r.t parameter: dl/dp
Eigen::VectorXd ParameterizedQuadraticObjective::getRunningCostParameterGradient(const Eigen::VectorXd& state,
                                                                                  const Eigen::VectorXd& control,
                                                                                  const Eigen::VectorXd& parameter,
                                                                                  int index) const {
    return Eigen::VectorXd::Zero(parameter_dim_);
}

// Gradient of final cost w.r.t state: dlf/dx (p). Parameter p is ignored.
Eigen::VectorXd ParameterizedQuadraticObjective::getFinalCostGradient(const Eigen::VectorXd& final_state,
                                                                      const Eigen::VectorXd& parameter) const {
     // Parameter argument is ignored.
     if (final_state.size() != reference_state_.size()) {
         throw std::runtime_error("getFinalCostGradient final_state dim does not match reference_state_ dim.");
     }
    Eigen::VectorXd state_error = final_state - reference_state_;
    return 2.0 * Qf_ * state_error;
}

// Gradient of final cost w.r.t parameter: dlf/dp. Parameter p is ignored.
Eigen::VectorXd ParameterizedQuadraticObjective::getFinalCostParameterGradient(const Eigen::VectorXd& final_state,
                                                                                const Eigen::VectorXd& parameter) const {
    // Final cost does not depend on the parameter argument in this design.
    return Eigen::VectorXd::Zero(parameter_dim_);
}

// --- Hessian Implementations ---

Eigen::MatrixXd ParameterizedQuadraticObjective::getRunningCostStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const {
    // l_xx = 2 * Q (scaled by timestep)
    return 2.0 * Q_;
}

Eigen::MatrixXd ParameterizedQuadraticObjective::getRunningCostControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const {
    // l_uu = 2 * R (scaled by timestep)
    // Handle 0 control dimension
    if (R_.rows() > 0 && R_.cols() > 0) {
        return 2.0 * R_;
    } else {
        return Eigen::MatrixXd::Zero(0, 0);
    }
}

Eigen::MatrixXd ParameterizedQuadraticObjective::getRunningCostCrossHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const {
    // l_ux = 0 for standard quadratic cost
    int state_dim = Q_.rows(); // Assuming Q defines state dim
    int control_dim = R_.rows(); // Assuming R defines control dim
    if (control_dim == 0 || state_dim == 0) {
        return Eigen::MatrixXd::Zero(0, 0);
    }
    return Eigen::MatrixXd::Zero(control_dim, state_dim);
}

Eigen::MatrixXd ParameterizedQuadraticObjective::getFinalCostHessian(
    const Eigen::VectorXd& final_state) const {
    // lf_xx = 2 * Qf
    return 2.0 * Qf_;
}

// --- Implementations for base Objective pure virtual function overrides ---

double ParameterizedQuadraticObjective::evaluate(const std::vector<Eigen::VectorXd>& states,
                                                 const std::vector<Eigen::VectorXd>& controls) const {
    throw std::logic_error("Use parameterized objective evaluation or provide parameter explicitly.");
    // Or, implement evaluation using a default/zero parameter? Less safe.
}

double ParameterizedQuadraticObjective::running_cost(const Eigen::VectorXd& state, 
                                                     const Eigen::VectorXd& control, 
                                                     int index) const {
    // Call parameterized version with zero parameter? Or throw?
    // Throwing is safer to enforce explicit parameter handling.
    // return running_cost(state, control, Eigen::VectorXd::Zero(parameter_dim_), index);
    throw std::logic_error("Use parameterized running_cost version for ParameterizedQuadraticObjective.");
}

double ParameterizedQuadraticObjective::terminal_cost(const Eigen::VectorXd& final_state) const {
    // Call parameterized version with zero parameter? Or throw?
    // return terminal_cost(final_state, Eigen::VectorXd::Zero(parameter_dim_));
    throw std::logic_error("Use parameterized terminal_cost version for ParameterizedQuadraticObjective.");
}

Eigen::VectorXd ParameterizedQuadraticObjective::getRunningCostStateGradient(const Eigen::VectorXd& state, 
                                                                           const Eigen::VectorXd& control, 
                                                                           int index) const {
    // return getRunningCostStateGradient(state, control, Eigen::VectorXd::Zero(parameter_dim_), index);
    throw std::logic_error("Use parameterized getRunningCostStateGradient version.");
}

Eigen::VectorXd ParameterizedQuadraticObjective::getRunningCostControlGradient(const Eigen::VectorXd& state, 
                                                                             const Eigen::VectorXd& control, 
                                                                             int index) const {
    // return getRunningCostControlGradient(state, control, Eigen::VectorXd::Zero(parameter_dim_), index);
    throw std::logic_error("Use parameterized getRunningCostControlGradient version.");
}

Eigen::VectorXd ParameterizedQuadraticObjective::getFinalCostGradient(const Eigen::VectorXd& final_state) const {
    // return getFinalCostGradient(final_state, Eigen::VectorXd::Zero(parameter_dim_));
     throw std::logic_error("Use parameterized getFinalCostGradient version.");
}

} // namespace cddp 