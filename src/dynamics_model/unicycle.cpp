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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

Unicycle::Unicycle(double timestep, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type) {
}

Eigen::VectorXd Unicycle::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    const double theta = state(STATE_THETA);  // heading angle
    const double v = control(CONTROL_V);      // velocity
    const double omega = control(CONTROL_OMEGA);  // angular velocity
    state_dot(STATE_X) = v * std::cos(theta);     // dx/dt
    state_dot(STATE_Y) = v * std::sin(theta);     // dy/dt
    state_dot(STATE_THETA) = omega;               // dtheta/dt
    
    return state_dot;
}

Eigen::MatrixXd Unicycle::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    
    const double theta = state(STATE_THETA);  // heading angle
    const double v = control(CONTROL_V);  // velocity
    A(STATE_X, STATE_THETA) = -v * std::sin(theta);
    A(STATE_Y, STATE_THETA) = v * std::cos(theta);
    
    return A;
}

Eigen::MatrixXd Unicycle::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);  // Note: Using 2 for control dim as per original
    const double theta = state(STATE_THETA);  // heading angle
    B(STATE_X, CONTROL_V) = std::cos(theta);
    B(STATE_Y, CONTROL_V) = std::sin(theta);
    B(STATE_THETA, CONTROL_OMEGA) = 1.0;
    
    return B;
}

std::vector<Eigen::MatrixXd> Unicycle::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    auto hessians = makeZeroTensor(STATE_DIM, STATE_DIM, STATE_DIM);
    
    const double v = control(CONTROL_V);
    const double theta = state(STATE_THETA);
    hessians[STATE_X](STATE_THETA, STATE_THETA) = -v * std::cos(theta);
    hessians[STATE_Y](STATE_THETA, STATE_THETA) = -v * std::sin(theta);
    
    return hessians;
}

std::vector<Eigen::MatrixXd> Unicycle::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    auto hessians = makeZeroTensor(STATE_DIM, CONTROL_DIM, CONTROL_DIM);
    
    return hessians;
}

VectorXdual2nd Unicycle::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {

    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);

    const autodiff::dual2nd theta = state(STATE_THETA);

    const autodiff::dual2nd v = control(CONTROL_V);
    const autodiff::dual2nd omega = control(CONTROL_OMEGA);

    state_dot(STATE_X) = v * cos(theta);
    state_dot(STATE_Y) = v * sin(theta);
    state_dot(STATE_THETA) = omega;

    return state_dot;
}

} // namespace cddp
