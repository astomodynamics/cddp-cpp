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

#include "dynamics_model/pendulum.hpp"
#include <cmath>

namespace cddp {

Pendulum::Pendulum(double timestep, double length, double mass, double damping,
                   std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
        length_(length), mass_(mass), damping_(damping) {}

Eigen::VectorXd Pendulum::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);
    const double theta_dot = state(STATE_THETA_DOT);
    
    // Extract control variable
    const double torque = control(CONTROL_TORQUE);
    
    // Precompute constants
    const double intertia = mass_ * length_ * length_;

    // Pendulum dynamics equations
    state_dot(STATE_THETA) = theta_dot;
    state_dot(STATE_THETA_DOT) = (torque - damping_ * theta_dot + mass_ * gravity_ * length_ * std::sin(theta)) / intertia;

    return state_dot;
}

Eigen::MatrixXd Pendulum::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);
    
    // Compute partial derivatives with respect to state variables
    A(STATE_THETA, STATE_THETA_DOT) = 1.0;
    
    // d(dtheta_dot/dt)/dtheta
    A(STATE_THETA_DOT, STATE_THETA) = (gravity_ / length_) * std::cos(theta);
    
    // d(dtheta_dot/dt)/dtheta_dot
    A(STATE_THETA_DOT, STATE_THETA_DOT) = -damping_ / (mass_ * length_ * length_);
    
    return A;
}

Eigen::MatrixXd Pendulum::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    
    // Compute partial derivatives with respect to control variable
    // d(dtheta_dot/dt)/dtorque
    B(STATE_THETA_DOT, CONTROL_TORQUE) = 1.0 / (mass_ * length_ * length_);
    
    return B;
}

Eigen::MatrixXd Pendulum::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(STATE_DIM * STATE_DIM, STATE_DIM);
    
    return H;
}

Eigen::MatrixXd Pendulum::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(STATE_DIM * CONTROL_DIM, CONTROL_DIM);
}

} // namespace cddp