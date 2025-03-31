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

#include "dynamics_model/spacecraft_linear.hpp"
#include <cmath>

namespace cddp {

HCW::HCW(double timestep, double mean_motion, double mass,
         std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mean_motion_(mean_motion),
      mass_(mass) {
}

Eigen::VectorXd HCW::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract state variables
    const double x = state(STATE_X);
    const double y = state(STATE_Y);
    const double z = state(STATE_Z);
    
    const double vx = state(STATE_VX);
    const double vy = state(STATE_VY);
    const double vz = state(STATE_VZ);
    
    // Extract control forces
    const double Fx = control(CONTROL_FX);
    const double Fy = control(CONTROL_FY);
    const double Fz = control(CONTROL_FZ);
    
    // Position derivatives (velocity)
    state_dot(STATE_X) = vx;
    state_dot(STATE_Y) = vy;
    state_dot(STATE_Z) = vz;
    
    // Velocity derivatives (HCW equations)
    // ẍ = 2nẏ + 3n²x + Fx/m
    // ÿ = -2nẋ + Fy/m
    // z̈ = -n²z + Fz/m
    const double n = mean_motion_;
    const double n2 = n * n;
    
    state_dot(STATE_VX) = 2.0 * n * vy + 3.0 * n2 * x + Fx/mass_;
    state_dot(STATE_VY) = -2.0 * n * vx + Fy/mass_;
    state_dot(STATE_VZ) = -n2 * z + Fz/mass_;
    
    return state_dot;
}

Eigen::MatrixXd HCW::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For HCW equations, we can compute the analytical Jacobian
    const double n = mean_motion_;
    const double n2 = n * n;
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    
    // Position derivatives
    A(STATE_X, STATE_VX) = 1.0;
    A(STATE_Y, STATE_VY) = 1.0;
    A(STATE_Z, STATE_VZ) = 1.0;
    
    // Velocity derivatives
    A(STATE_VX, STATE_X) = 3.0 * n2;
    A(STATE_VX, STATE_VY) = 2.0 * n;
    
    A(STATE_VY, STATE_VX) = -2.0 * n;
    
    A(STATE_VZ, STATE_Z) = -n2;
    
    return A;
}

Eigen::MatrixXd HCW::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For HCW equations, we can compute the analytical control Jacobian
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    
    // Control only affects velocity states
    B(STATE_VX, CONTROL_FX) = 1.0/mass_;
    B(STATE_VY, CONTROL_FY) = 1.0/mass_;
    B(STATE_VZ, CONTROL_FZ) = 1.0/mass_;
    
    return B;
}

std::vector<Eigen::MatrixXd> HCW::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    // HCW equations are linear, so state Hessian is zero
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> HCW::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    // HCW equations are linear in control, so control Hessian is zero
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

} // namespace cddp