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

namespace cddp {

SpacecraftLinearFuel::SpacecraftLinearFuel(double timestep, double mean_motion, double isp, double g0,
                                           std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mean_motion_(mean_motion),
      isp_(isp),
      g0_(g0) {
}

Eigen::VectorXd SpacecraftLinearFuel::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
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
    
    if (mass <= 0) {
        state_dot(STATE_VX) = 0;
        state_dot(STATE_VY) = 0;
        state_dot(STATE_VZ) = 0;
        state_dot(STATE_MASS) = 0;
    } else {
        state_dot(STATE_VX) = 2.0 * n * vy + 3.0 * n2 * x + Fx/mass;
        state_dot(STATE_VY) = -2.0 * n * vx + Fy/mass;
        state_dot(STATE_VZ) = -n2 * z + Fz/mass;

        const double thrust_norm = std::sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
        if (isp_ * g0_ > 1e-9) {
             state_dot(STATE_MASS) = -thrust_norm / (isp_ * g0_);
        } else {
             state_dot(STATE_MASS) = 0;
        }
    }
    
    state_dot(STATE_ACCUMULATED_CONTROL_EFFORT) = 0.5 * (Fx*Fx + Fy*Fy + Fz*Fz);
    
    return state_dot;
}

Eigen::MatrixXd SpacecraftLinearFuel::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    const double n = mean_motion_;
    const double n2 = n * n;
    const double mass = state(STATE_MASS);
    const double Fx = control(CONTROL_FX);
    const double Fy = control(CONTROL_FY);
    const double Fz = control(CONTROL_FZ);

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

    if (mass > 1e-6) {
        A(STATE_VX, STATE_MASS) = -Fx / (mass * mass);
        A(STATE_VY, STATE_MASS) = -Fy / (mass * mass);
        A(STATE_VZ, STATE_MASS) = -Fz / (mass * mass);
    }
    
    return A;
}

Eigen::MatrixXd SpacecraftLinearFuel::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    const double mass = state(STATE_MASS);
    const double Fx = control(CONTROL_FX);
    const double Fy = control(CONTROL_FY);
    const double Fz = control(CONTROL_FZ);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    
    if (mass > 1e-6) {
        B(STATE_VX, CONTROL_FX) = 1.0/mass;
        B(STATE_VY, CONTROL_FY) = 1.0/mass;
        B(STATE_VZ, CONTROL_FZ) = 1.0/mass;
    }

    const double thrust_norm = std::sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
    if (isp_ * g0_ > 1e-9 && thrust_norm > 1e-9) {
        B(STATE_MASS, CONTROL_FX) = -Fx / (isp_ * g0_ * thrust_norm);
        B(STATE_MASS, CONTROL_FY) = -Fy / (isp_ * g0_ * thrust_norm);
        B(STATE_MASS, CONTROL_FZ) = -Fz / (isp_ * g0_ * thrust_norm);
    }
    
    B(STATE_ACCUMULATED_CONTROL_EFFORT, CONTROL_FX) = Fx;
    B(STATE_ACCUMULATED_CONTROL_EFFORT, CONTROL_FY) = Fy;
    B(STATE_ACCUMULATED_CONTROL_EFFORT, CONTROL_FZ) = Fz;
    
    return B;
}

std::vector<Eigen::MatrixXd> SpacecraftLinearFuel::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftLinearFuel::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }

    const double Fx = control(CONTROL_FX);
    const double Fy = control(CONTROL_FY);
    const double Fz = control(CONTROL_FZ);

    // Hessian for STATE_ACCUMULATED_CONTROL_EFFORT
    // d(0.5 * (Fx^2 + Fy^2 + Fz^2)) / dU^2
    hessians[STATE_ACCUMULATED_CONTROL_EFFORT](CONTROL_FX, CONTROL_FX) = 1.0;
    hessians[STATE_ACCUMULATED_CONTROL_EFFORT](CONTROL_FY, CONTROL_FY) = 1.0;
    hessians[STATE_ACCUMULATED_CONTROL_EFFORT](CONTROL_FZ, CONTROL_FZ) = 1.0;

    // Hessian for STATE_MASS
    // d(-thrust_norm / (isp_ * g0_)) / dU^2
    if (isp_ * g0_ > 1e-9) {
        const double thrust_norm_sq = Fx*Fx + Fy*Fy + Fz*Fz;
        if (thrust_norm_sq > 1e-9) { // Avoid division by zero if thrust is zero
            const double thrust_norm = std::sqrt(thrust_norm_sq);
            const double thrust_norm_cubed = thrust_norm_sq * thrust_norm;
            const double C = isp_ * g0_;
            const double common_factor = -1.0 / (C * thrust_norm_cubed);

            hessians[STATE_MASS](CONTROL_FX, CONTROL_FX) = common_factor * (Fy*Fy + Fz*Fz);
            hessians[STATE_MASS](CONTROL_FY, CONTROL_FY) = common_factor * (Fx*Fx + Fz*Fz);
            hessians[STATE_MASS](CONTROL_FZ, CONTROL_FZ) = common_factor * (Fx*Fx + Fy*Fy);

            hessians[STATE_MASS](CONTROL_FX, CONTROL_FY) = common_factor * (-Fx * Fy);
            hessians[STATE_MASS](CONTROL_FY, CONTROL_FX) = common_factor * (-Fx * Fy);

            hessians[STATE_MASS](CONTROL_FX, CONTROL_FZ) = common_factor * (-Fx * Fz);
            hessians[STATE_MASS](CONTROL_FZ, CONTROL_FX) = common_factor * (-Fx * Fz);

            hessians[STATE_MASS](CONTROL_FY, CONTROL_FZ) = common_factor * (-Fy * Fz);
            hessians[STATE_MASS](CONTROL_FZ, CONTROL_FY) = common_factor * (-Fy * Fz);
        }
    }

    return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftLinearFuel::getCrossHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);
    }
    return hessians;
}

VectorXdual2nd SpacecraftLinearFuel::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control) const {
    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);

    return state_dot;
}

} // namespace cddp