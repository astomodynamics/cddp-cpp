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

#include "dynamics_model/dubins_car.hpp"  // Adjust include path as needed
#include <cmath>

namespace cddp {

DubinsCar::DubinsCar(double speed,
                     double timestep,
                     std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      speed_(speed)
{
}

Eigen::VectorXd DubinsCar::getContinuousDynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const
{
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

    // Extract state components
    const double theta = state(STATE_THETA);

    // Extract control (turn rate)
    const double omega = control(CONTROL_OMEGA);

    // Dubins car dynamics with constant forward speed
    state_dot(STATE_X)     = speed_ * std::cos(theta);
    state_dot(STATE_Y)     = speed_ * std::sin(theta);
    state_dot(STATE_THETA) = omega;

    return state_dot;
}

Eigen::MatrixXd DubinsCar::getStateJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const
{
    (void)control; // Not used in partials wrt. state for linear terms
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    const double theta = state(STATE_THETA);
    A(STATE_X, STATE_THETA) = -speed_ * std::sin(theta);
    A(STATE_Y, STATE_THETA) = speed_ * std::cos(theta);

    return A;
}

Eigen::MatrixXd DubinsCar::getControlJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const
{
    (void)state;  
    (void)control; 

    // 3x1 matrix: (X, Y, THETA) x (OMEGA)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);

    // dx/domega = 0, dy/domega = 0, dtheta/domega = 1
    B(STATE_THETA, CONTROL_OMEGA) = 1.0;

    return B;
}

Eigen::MatrixXd DubinsCar::getStateHessian(
    const Eigen::VectorXd& /*state*/,
    const Eigen::VectorXd& /*control*/) const
{
    return Eigen::MatrixXd::Zero(STATE_DIM * STATE_DIM, CONTROL_DIM);
}

Eigen::MatrixXd DubinsCar::getControlHessian(
    const Eigen::VectorXd& /*state*/,
    const Eigen::VectorXd& /*control*/) const
{
    return Eigen::MatrixXd::Zero(STATE_DIM * CONTROL_DIM, CONTROL_DIM);
}

} // namespace cddp
