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

#include "dynamics_model/bicycle.hpp"
#include <cmath>

namespace cddp {

Bicycle::Bicycle(double timestep, double wheelbase, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      wheelbase_(wheelbase) {
}

Eigen::VectorXd Bicycle::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);  // heading angle
    const double v = state(STATE_V);          // velocity
    
    // Extract control variables
    const double a = control(CONTROL_ACC);      // acceleration
    const double delta = control(CONTROL_DELTA); // steering angle
    
    // Kinematic bicycle model equations
    state_dot(STATE_X) = v * std::cos(theta);                  // dx/dt
    state_dot(STATE_Y) = v * std::sin(theta);                  // dy/dt
    state_dot(STATE_THETA) = (v / wheelbase_) * std::tan(delta); // dtheta/dt
    state_dot(STATE_V) = a;                                    // dv/dt
    
    return state_dot;
}

Eigen::MatrixXd Bicycle::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);  // heading angle
    const double v = state(STATE_V);          // velocity
    
    // Extract control variables
    const double delta = control(CONTROL_DELTA); // steering angle
    
    // Compute partial derivatives with respect to state variables
    // df1/dtheta = d(dx/dt)/dtheta
    A(STATE_X, STATE_THETA) = -v * std::sin(theta);
    // df1/dv = d(dx/dt)/dv
    A(STATE_X, STATE_V) = std::cos(theta);
    
    // df2/dtheta = d(dy/dt)/dtheta
    A(STATE_Y, STATE_THETA) = v * std::cos(theta);
    // df2/dv = d(dy/dt)/dv
    A(STATE_Y, STATE_V) = std::sin(theta);
    
    // df3/dv = d(dtheta/dt)/dv
    A(STATE_THETA, STATE_V) = std::tan(delta) / wheelbase_;
    
    return A;
}

Eigen::MatrixXd Bicycle::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    
    // Extract state variables
    const double v = state(STATE_V);          // velocity
    
    // Extract control variables
    const double delta = control(CONTROL_DELTA); // steering angle
    
    // Compute partial derivatives with respect to control variables
    // df4/da = d(dv/dt)/da
    B(STATE_V, CONTROL_ACC) = 1.0;
    
    // df3/ddelta = d(dtheta/dt)/ddelta
    B(STATE_THETA, CONTROL_DELTA) = v / (wheelbase_ * std::pow(std::cos(delta), 2));
    
    return B;
}

Eigen::MatrixXd Bicycle::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(STATE_DIM * STATE_DIM, STATE_DIM);
    
    // Extract state variables
    const double theta = state(STATE_THETA);  // heading angle
    const double v = state(STATE_V);          // velocity
    
    // Second derivatives with respect to states
    // d²(dx/dt)/dtheta²
    H(STATE_THETA * STATE_DIM + STATE_X, STATE_THETA) = -v * std::cos(theta);
    
    // d²(dy/dt)/dtheta²
    H(STATE_THETA * STATE_DIM + STATE_Y, STATE_THETA) = -v * std::sin(theta);
    
    return H;
}

Eigen::MatrixXd Bicycle::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(STATE_DIM * CONTROL_DIM, CONTROL_DIM);
    
    // Extract state variables
    const double v = state(STATE_V);          // velocity
    
    // Extract control variables
    const double delta = control(CONTROL_DELTA); // steering angle
    
    // Second derivatives with respect to controls
    // d²(dtheta/dt)/ddelta²
    H(CONTROL_DELTA * STATE_DIM + STATE_THETA, CONTROL_DELTA) = 
        2.0 * v * std::sin(delta) / (wheelbase_ * std::pow(std::cos(delta), 3));
    
    return H;
}

} // namespace cddp