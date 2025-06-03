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

#include "dynamics_model/dreyfus_rocket.hpp"
#include <cmath>
#include "cddp_core/helper.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

DreyfusRocket::DreyfusRocket(double timestep, std::string integration_type,
                           double thrust_acceleration, double gravity_acceleration)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      thrust_acceleration_(thrust_acceleration),
      gravity_acceleration_(gravity_acceleration) {
}

Eigen::VectorXd DreyfusRocket::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    const double x_dot = state(STATE_X_DOT);
    const double theta = control(CONTROL_THETA);
    
    state_dot(STATE_X) = x_dot;
    state_dot(STATE_X_DOT) = thrust_acceleration_ * std::cos(theta) - gravity_acceleration_;
    
    return state_dot;
}

Eigen::MatrixXd DreyfusRocket::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    
    A(STATE_X, STATE_X_DOT) = 1.0;
    return A;
}

Eigen::MatrixXd DreyfusRocket::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    
    const double theta = control(CONTROL_THETA);
    B(STATE_X_DOT, CONTROL_THETA) = -thrust_acceleration_ * std::sin(theta);
    
    return B;
}

std::vector<Eigen::MatrixXd> DreyfusRocket::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> DreyfusRocket::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    
    // The only non-zero element is the second derivative of x_dot with respect to theta
    const double theta = control(CONTROL_THETA);
    hessians[STATE_X_DOT](CONTROL_THETA, CONTROL_THETA) = -thrust_acceleration_ * std::cos(theta);
    
    return hessians;
}

VectorXdual2nd DreyfusRocket::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {

    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);
    const autodiff::dual2nd x_dot = state(STATE_X_DOT);
    const autodiff::dual2nd theta = control(CONTROL_THETA);

    state_dot(STATE_X) = x_dot;
    state_dot(STATE_X_DOT) = thrust_acceleration_ * cos(theta) - gravity_acceleration_;

    return state_dot;
}

} // namespace cddp