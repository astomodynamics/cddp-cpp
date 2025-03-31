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
    // TODO: Implement damping term
    
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

Eigen::MatrixXd CartPole::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {

    auto f = [&](const Eigen::VectorXd& x) {
        return getContinuousDynamics(x, control);
    };

    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd CartPole::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    auto f = [&](const Eigen::VectorXd& u) {
        return getContinuousDynamics(state, u);
    };
    return finite_difference_jacobian(f, control);
}


std::vector<Eigen::MatrixXd> CartPole::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> CartPole::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

} // namespace cddp