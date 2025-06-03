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


#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>
#include <string>
#include <memory>
#include "cddp.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

SpacecraftLinearFuel::SpacecraftLinearFuel(double timestep, double mean_motion, double isp, double g0,
                                           std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mean_motion_(mean_motion),
      isp_(isp),
      g0_(g0),
      epsilon_(1e-8) {
}

Eigen::VectorXd SpacecraftLinearFuel::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract state variables
    const double x = state(STATE_X);
    const double y = state(STATE_Y);
    const double z = state(STATE_Z);
    
    const double vx = state(STATE_VX);
    const double vy = state(STATE_VY);
    const double vz = state(STATE_VZ);
    const double mass = state(STATE_MASS);
    
    // Extract control forces
    const double Fx = control(CONTROL_FX);
    const double Fy = control(CONTROL_FY);
    const double Fz = control(CONTROL_FZ);
    
    // Position derivatives (velocity)
    state_dot(STATE_X) = vx;
    state_dot(STATE_Y) = vy;
    state_dot(STATE_Z) = vz;
    
    // Velocity derivatives (HCW equations, modified for mass)
    // ẍ = 2nẏ + 3n²x + Fx/m
    // ÿ = -2nẋ + Fy/m
    // z̈ = -n²z + Fz/m
    const double n = mean_motion_;
    const double n2 = n * n;
    
    state_dot(STATE_VX) = 2.0 * n * vy + 3.0 * n2 * x + Fx/mass;
    state_dot(STATE_VY) = -2.0 * n * vx + Fy/mass;
    state_dot(STATE_VZ) = -n2 * z + Fz/mass;
    const double thrust_squared = Fx*Fx + Fy*Fy + Fz*Fz;
    const double thrust_norm = std::sqrt(thrust_squared + epsilon_);
    state_dot(STATE_MASS) = -thrust_norm / (isp_ * g0_);
    state_dot(STATE_ACCUMULATED_CONTROL_EFFORT) = 0.5 * thrust_squared;
    
    return state_dot;
}

VectorXdual2nd SpacecraftLinearFuel::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {
    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);

    const autodiff::dual2nd x = state(STATE_X);
    const autodiff::dual2nd y = state(STATE_Y);
    const autodiff::dual2nd z = state(STATE_Z);

    const autodiff::dual2nd vx = state(STATE_VX);
    const autodiff::dual2nd vy = state(STATE_VY);
    const autodiff::dual2nd vz = state(STATE_VZ);
    const autodiff::dual2nd mass = state(STATE_MASS);
    const autodiff::dual2nd accumulated_control_effort = state(STATE_ACCUMULATED_CONTROL_EFFORT);

    const autodiff::dual2nd Fx = control(CONTROL_FX);
    const autodiff::dual2nd Fy = control(CONTROL_FY);
    const autodiff::dual2nd Fz = control(CONTROL_FZ);

    const autodiff::dual2nd n = mean_motion_;
    const autodiff::dual2nd n2 = n * n;

    state_dot(STATE_X) = vx;
    state_dot(STATE_Y) = vy;
    state_dot(STATE_Z) = vz;

    state_dot(STATE_VX) = 2.0 * n * vy + 3.0 * n2 * x + Fx/mass;
    state_dot(STATE_VY) = -2.0 * n * vx + Fy/mass;
    state_dot(STATE_VZ) = -n2 * z + Fz/mass;

    const autodiff::dual2nd thrust_squared = Fx*Fx + Fy*Fy + Fz*Fz;
    const autodiff::dual2nd thrust_norm = sqrt(thrust_squared + epsilon_);

    state_dot(STATE_MASS) = -thrust_norm / (isp_ * g0_);
    state_dot(STATE_ACCUMULATED_CONTROL_EFFORT) = 0.5 * thrust_squared;


    return state_dot;
}

Eigen::MatrixXd SpacecraftLinearFuel::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {

    auto f = [&](const Eigen::VectorXd& x) {
        return getContinuousDynamics(x, control, time);
    };

    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd SpacecraftLinearFuel::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    auto f = [&](const Eigen::VectorXd& u) {
        return getContinuousDynamics(state, u, time);
    };
    return finite_difference_jacobian(f, control);
}


std::vector<Eigen::MatrixXd> SpacecraftLinearFuel::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftLinearFuel::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftLinearFuel::getCrossHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    }
    return hessians;
}


} // namespace cddp