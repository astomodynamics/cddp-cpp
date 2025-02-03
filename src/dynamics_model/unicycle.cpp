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

#include "dynamics_model/unicycle.hpp"
#include <cmath>

namespace cddp {

Unicycle::Unicycle(double timestep, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type) {
}

Eigen::VectorXd Unicycle::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);  // heading angle
    
    // Extract control variables
    const double v = control(CONTROL_V);      // velocity
    const double omega = control(CONTROL_OMEGA);  // angular velocity
    
    // Unicycle dynamics equations
    state_dot(STATE_X) = v * std::cos(theta);     // dx/dt
    state_dot(STATE_Y) = v * std::sin(theta);     // dy/dt
    state_dot(STATE_THETA) = omega;               // dtheta/dt
    
    return state_dot;
}

Eigen::MatrixXd Unicycle::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);  // heading angle
    
    // Extract control variables
    const double v = control(CONTROL_V);  // velocity
    
    // Compute partial derivatives with respect to state variables
    // df1/dtheta = d(dx/dt)/dtheta
    A(STATE_X, STATE_THETA) = -v * std::sin(theta);
    
    // df2/dtheta = d(dy/dt)/dtheta
    A(STATE_Y, STATE_THETA) = v * std::cos(theta);
    
    return A;
}

Eigen::MatrixXd Unicycle::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);  // Note: Using 2 for control dim as per original
    
    // Extract state variables
    const double theta = state(STATE_THETA);  // heading angle
    
    // Compute partial derivatives with respect to control variables
    // df1/dv = d(dx/dt)/dv
    B(STATE_X, CONTROL_V) = std::cos(theta);
    
    // df2/dv = d(dy/dt)/dv
    B(STATE_Y, CONTROL_V) = std::sin(theta);
    
    // df3/domega = d(dtheta/dt)/domega
    B(STATE_THETA, CONTROL_OMEGA) = 1.0;
    
    return B;
}

Eigen::MatrixXd Unicycle::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    return Eigen::MatrixXd::Zero(STATE_DIM * STATE_DIM, 2);
}

Eigen::MatrixXd Unicycle::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    // TODO: Compute and return the Hessian tensor d^2f/du^2 (represented as a matrix)
    return Eigen::MatrixXd::Zero(STATE_DIM * 2, 2);
}

} // namespace cddp