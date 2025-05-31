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

#include "dynamics_model/cartpole.hpp"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "cddp_core/helper.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

CartPole::CartPole(double timestep, std::string integration_type,
                   double cart_mass, double pole_mass, double pole_length,
                   double gravity, double damping)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      cart_mass_(cart_mass),
      pole_mass_(pole_mass),
      pole_length_(pole_length),
      gravity_(gravity),
      damping_(damping) {
}

Eigen::VectorXd CartPole::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    const double x = state(STATE_X);
    const double theta = state(STATE_THETA);
    const double x_dot = state(STATE_X_DOT);
    const double theta_dot = state(STATE_THETA_DOT);
    const double force = control(CONTROL_FORCE);
    
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    const double total_mass = cart_mass_ + pole_mass_;
    
    const double den = cart_mass_ + pole_mass_ * sin_theta * sin_theta;
    
    state_dot(STATE_X) = x_dot;
    
    state_dot(STATE_THETA) = theta_dot;
    
    state_dot(STATE_X_DOT) = (force + pole_mass_ * sin_theta * (pole_length_ * theta_dot * theta_dot + gravity_ * cos_theta)) / den;

    state_dot(STATE_THETA_DOT) = (-force * cos_theta - pole_mass_ * pole_length_ * theta_dot * theta_dot * cos_theta * sin_theta - total_mass * gravity_ * sin_theta) / (pole_length_ * den);
    
    return state_dot;
}

cddp::VectorXdual2nd CartPole::getContinuousDynamicsAutodiff(
    const cddp::VectorXdual2nd& state, const cddp::VectorXdual2nd& control) const {

    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);

    const autodiff::dual2nd theta = state(STATE_THETA);
    const autodiff::dual2nd x_dot = state(STATE_X_DOT);
    const autodiff::dual2nd theta_dot = state(STATE_THETA_DOT);

    const autodiff::dual2nd force = control(CONTROL_FORCE);

    const double total_mass = cart_mass_ + pole_mass_;

    const autodiff::dual2nd sin_theta = sin(theta);
    const autodiff::dual2nd cos_theta = cos(theta);

    const autodiff::dual2nd mp_sin2_theta = pole_mass_ * sin_theta * sin_theta;
    const autodiff::dual2nd den = cart_mass_ + mp_sin2_theta;

    state_dot(STATE_X) = x_dot;

    state_dot(STATE_THETA) = theta_dot;

    state_dot(STATE_X_DOT) = (force + pole_mass_ * sin_theta * (pole_length_ * theta_dot * theta_dot + gravity_ * cos_theta)) / den;

    state_dot(STATE_THETA_DOT) = (-force * cos_theta - pole_mass_ * pole_length_ * theta_dot * theta_dot * cos_theta * sin_theta - total_mass * gravity_ * sin_theta - damping_ * theta_dot) / (pole_length_ * den);

    return state_dot;
}

Eigen::MatrixXd CartPole::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {

    const double theta = state(STATE_THETA);
    const double theta_dot_val = state(STATE_THETA_DOT);
    const double force_val = control(CONTROL_FORCE);

    const double mc = cart_mass_;
    const double mp = pole_mass_;
    const double l = pole_length_;
    const double g = gravity_;
    const double d = damping_;
    const double total_mass_val = mc + mp;

    const double s_th = std::sin(theta);
    const double c_th = std::cos(theta);
    const double s_th_sq = s_th * s_th;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    A(STATE_X, STATE_X_DOT) = 1.0;
    A(STATE_THETA, STATE_THETA_DOT) = 1.0;

    // Common denominator term for f3_dot and f4_dot
    const double den = mc + mp * s_th_sq;
    const double den_sq = den * den;

    // Derivatives for f3_dot (x_ddot)
    const double num3 = force_val + mp * s_th * (l * theta_dot_val * theta_dot_val + g * c_th);

    // df3_dtheta
    const double d_num3_d_theta = mp * c_th * (l * theta_dot_val * theta_dot_val + g * c_th) + mp * s_th * (-g * s_th);
    const double d_den_d_theta = mp * 2.0 * s_th * c_th;
    A(STATE_X_DOT, STATE_THETA) = (d_num3_d_theta * den - num3 * d_den_d_theta) / den_sq;

    // df3_dtheta_dot
    const double d_num3_d_theta_dot = mp * s_th * (l * 2.0 * theta_dot_val);
    A(STATE_X_DOT, STATE_THETA_DOT) = d_num3_d_theta_dot / den;

    // Derivatives for f4_dot (theta_ddot)
    const double den4 = l * den;
    const double den4_sq = den4 * den4;
    const double num4_damped = -force_val * c_th - mp * l * theta_dot_val * theta_dot_val * c_th * s_th - total_mass_val * g * s_th - d * theta_dot_val;

    // df4_dtheta
    const double c_2th = c_th * c_th - s_th * s_th; // cos(2*theta)
    const double d_num4_damped_d_theta = force_val * s_th - mp * l * theta_dot_val * theta_dot_val * c_2th - total_mass_val * g * c_th;
    // d_den4_d_theta is l * d_den_d_theta
    const double d_den4_d_theta = l * d_den_d_theta;
    A(STATE_THETA_DOT, STATE_THETA) = (d_num4_damped_d_theta * den4 - num4_damped * d_den4_d_theta) / den4_sq;

    // df4_dtheta_dot
    const double s_2th = 2.0 * s_th * c_th; // sin(2*theta)
    const double d_num4_damped_d_theta_dot = -mp * l * theta_dot_val * s_2th - d; // -mp * l * 2 * theta_dot_val * c_th * s_th - d
    A(STATE_THETA_DOT, STATE_THETA_DOT) = d_num4_damped_d_theta_dot / den4;

    return A;
}

Eigen::MatrixXd CartPole::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {

    const double theta = state(STATE_THETA);

    const double mc = cart_mass_;
    const double mp = pole_mass_;
    const double l = pole_length_;

    const double s_th = std::sin(theta);
    const double c_th = std::cos(theta);
    const double s_th_sq = s_th * s_th;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);

    const double den = mc + mp * s_th_sq;

    // df3_dforce
    B(STATE_X_DOT, CONTROL_FORCE) = 1.0 / den;

    // df4_dforce
    const double den4 = l * den;
    B(STATE_THETA_DOT, CONTROL_FORCE) = -c_th / den4;
    
    return B;
}

std::vector<Eigen::MatrixXd> CartPole::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return DynamicalSystem::getStateHessian(state, control); // Use autodiff
}

std::vector<Eigen::MatrixXd> CartPole::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return DynamicalSystem::getControlHessian(state, control); // Use autodiff
}

} // namespace cddp