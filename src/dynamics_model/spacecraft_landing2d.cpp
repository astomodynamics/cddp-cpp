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

#include "dynamics_model/spacecraft_landing2d.hpp"
#include <cmath>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <algorithm>

namespace cddp {

SpacecraftLanding2D::SpacecraftLanding2D(double timestep, 
                                        std::string integration_type,
                                        double mass,
                                        double length,
                                        double width,
                                        double min_thrust,
                                        double max_thrust,
                                        double max_gimble)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mass_(mass),
      length_(length),
      width_(width),
      min_thrust_(min_thrust),
      max_thrust_(max_thrust),
      max_gimble_(max_gimble) {
    // Calculate moment of inertia for uniform density rod
    inertia_ = (1.0/12.0) * mass_ * length_ * length_;
}

Eigen::VectorXd SpacecraftLanding2D::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

    // Extract state variables
    const double theta = state(STATE_THETA);
    const double theta_dot = state(STATE_THETA_DOT);

    // Extract control variables
    const double thrust_percent = control(CONTROL_THRUST);
    const double thrust_angle = control(CONTROL_ANGLE);

    // Calculate total thrust angle (thrust angle + spacecraft angle)
    const double total_angle = thrust_angle + theta;

    // Calculate forces
    const double thrust = max_thrust_ * thrust_percent;
    const double F_x = thrust * std::sin(total_angle);
    const double F_y = thrust * std::cos(total_angle);

    // Calculate torque
    const double T = -length_ / 2.0 * thrust * std::sin(thrust_angle);

    // Position derivatives (velocities)
    state_dot(STATE_X) = state(STATE_X_DOT);
    state_dot(STATE_Y) = state(STATE_Y_DOT);
    state_dot(STATE_THETA) = theta_dot;

    // Velocity derivatives (accelerations)
    state_dot(STATE_X_DOT) = F_x / mass_;
    state_dot(STATE_Y_DOT) = F_y / mass_ - gravity_;
    state_dot(STATE_THETA_DOT) = T / inertia_;

    return state_dot;
}

Eigen::MatrixXd SpacecraftLanding2D::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    auto f = [&](const Eigen::VectorXd& x) {
        return getContinuousDynamics(x, control, time);
    };

    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd SpacecraftLanding2D::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    auto f = [&](const Eigen::VectorXd& u) {
        return getContinuousDynamics(state, u, time);
    };
    return finite_difference_jacobian(f, control);
}

std::vector<Eigen::MatrixXd> SpacecraftLanding2D::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftLanding2D::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

VectorXdual2nd SpacecraftLanding2D::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {    

    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);

    // Extract state variables (dual2nd)
    const autodiff::dual2nd theta = state(STATE_THETA);
    const autodiff::dual2nd x_dot = state(STATE_X_DOT);
    const autodiff::dual2nd y_dot = state(STATE_Y_DOT);
    const autodiff::dual2nd theta_dot = state(STATE_THETA_DOT);

    // Extract control variables (dual2nd) - Assume valid inputs (no clamping here)
    autodiff::dual2nd thrust_cmd = control(CONTROL_THRUST);
    autodiff::dual2nd gimble_angle_cmd = control(CONTROL_ANGLE);

    // Total angle of thrust vector in world frame
    const autodiff::dual2nd total_angle = gimble_angle_cmd + theta;

    // Calculate forces in world frame using ADL for math
    const autodiff::dual2nd F_x = thrust_cmd * sin(total_angle);
    const autodiff::dual2nd F_y = thrust_cmd * cos(total_angle);

    // Calculate torque using ADL for math
    const autodiff::dual2nd T = -thrust_cmd * (length_ / 2.0) * sin(gimble_angle_cmd);

    // Position derivatives
    state_dot(STATE_X) = x_dot;
    state_dot(STATE_Y) = y_dot;
    state_dot(STATE_THETA) = theta_dot;

    // Velocity derivatives
    state_dot(STATE_X_DOT) = F_x / mass_;
    state_dot(STATE_Y_DOT) = F_y / mass_ - gravity_;
    state_dot(STATE_THETA_DOT) = T / inertia_;

    return state_dot;
}

} // namespace cddp