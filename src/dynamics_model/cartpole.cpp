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
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    // const double p = state(STATE_X); // x0, not directly in f2, f3
    const double theta = state(STATE_THETA);         // x1
    // const double p_dot = state(STATE_X_DOT);    // x2
    const double theta_dot = state(STATE_THETA_DOT); // x3
    const double force = control(CONTROL_FORCE);   // u0

    const double s1 = std::sin(theta);
    const double c1 = std::cos(theta);
    const double s1_sq = s1 * s1;
    // const double c1_sq = c1 * c1; // if needed
    const double sin2theta = 2.0 * s1 * c1; // sin(2*theta)
    const double cos2theta = c1 * c1 - s1 * s1; // cos(2*theta)

    const double mc = cart_mass_;
    const double mp = pole_mass_;
    const double l = pole_length_;
    const double g = gravity_;
    const double d = damping_; 

    const double den = mc + mp * s1_sq;
    const double den_sq = den * den;

    // Row 0: df0/dx = [0, 0, 1, 0]
    A(STATE_X, STATE_X_DOT) = 1.0;

    // Row 1: df1/dx = [0, 0, 0, 1]
    A(STATE_THETA, STATE_THETA_DOT) = 1.0;

    // Row 2: df2/dx (derivatives of p_ddot)
    // f2 = (force + mp * l * theta_dot^2 * s1 + mp * g * s1 * c1) / den
    double N2 = force + mp * l * theta_dot * theta_dot * s1 + mp * g * s1 * c1;
    // d(N2)/d(theta)
    double dN2_dtheta = mp * l * theta_dot * theta_dot * c1 + mp * g * cos2theta;
    // d(den)/d(theta)
    double dDen_dtheta = mp * sin2theta;

    A(STATE_X_DOT, STATE_X) = 0.0; // df2/dx0
    A(STATE_X_DOT, STATE_THETA) = (dN2_dtheta * den - N2 * dDen_dtheta) / den_sq; // df2/dx1
    A(STATE_X_DOT, STATE_X_DOT) = 0.0; // df2/dx2
    A(STATE_X_DOT, STATE_THETA_DOT) = (2.0 * mp * l * theta_dot * s1) / den; // df2/dx3

    // Row 3: df3/dx (derivatives of theta_ddot)
    // f3 = (-force * c1 - mp * l * theta_dot^2 * s1 * c1 - (mc + mp) * g * s1 - d * theta_dot) / (l * den)
    double N3 = -force * c1 - mp * l * theta_dot * theta_dot * s1 * c1 - (mc + mp) * g * s1 - d * theta_dot;
    // d(N3)/d(theta)
    double dN3_dtheta = force * s1 - mp * l * theta_dot * theta_dot * cos2theta - (mc + mp) * g * c1;

    A(STATE_THETA_DOT, STATE_X) = 0.0; // df3/dx0
    A(STATE_THETA_DOT, STATE_THETA) = (dN3_dtheta * (l * den) - N3 * (l * dDen_dtheta)) / (l * l * den_sq); // df3/dx1
    A(STATE_THETA_DOT, STATE_X_DOT) = 0.0; // df3/dx2
    A(STATE_THETA_DOT, STATE_THETA_DOT) = (-mp * l * theta_dot * sin2theta - d) / (l * den); // df3/dx3
    
    return A;
}

Eigen::MatrixXd CartPole::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);

    const double theta = state(STATE_THETA); // x1
    // const double force = control(CONTROL_FORCE); // u0 (not needed for derivatives)

    const double s1 = std::sin(theta);
    const double c1 = std::cos(theta);
    const double s1_sq = s1 * s1;

    const double mc = cart_mass_;
    const double mp = pole_mass_;
    const double l = pole_length_;
    // const double g = gravity_;

    const double den = mc + mp * s1_sq;

    // df0/du0 = 0
    B(STATE_X, CONTROL_FORCE) = 0.0;
    // df1/du0 = 0
    B(STATE_THETA, CONTROL_FORCE) = 0.0;

    // df2/du0 (derivative of p_ddot w.r.t. force)
    B(STATE_X_DOT, CONTROL_FORCE) = 1.0 / den;

    // df3/du0 (derivative of theta_ddot w.r.t. force)
    B(STATE_THETA_DOT, CONTROL_FORCE) = -c1 / (l * den);

    return B;
}


std::vector<Eigen::MatrixXd> CartPole::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians_f_xx(STATE_DIM);

    const double theta = state(STATE_THETA);         // x1
    const double theta_dot = state(STATE_THETA_DOT); // x3
    const double force = control(CONTROL_FORCE);   // u0

    const double s1 = std::sin(theta);
    const double c1 = std::cos(theta);
    const double s1_sq = s1 * s1;
    const double sin2theta = 2.0 * s1 * c1;
    const double cos2theta = c1 * c1 - s1 * s1;

    const double mc = cart_mass_;
    const double mp = pole_mass_;
    const double l = pole_length_;
    const double g = gravity_;
    const double d = damping_;

    const double den = mc + mp * s1_sq;
    const double den_sq = den * den;
    // const double den_cub = den_sq * den; // if needed for d^2f/dx1^2

    // Hessian for f0 (STATE_X dynamics) = 0
    hessians_f_xx[STATE_X] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    // Hessian for f1 (STATE_THETA dynamics) = 0
    hessians_f_xx[STATE_THETA] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    // Hessian for f2 (STATE_X_DOT dynamics)
    Eigen::MatrixXd H2_xx = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    // d^2f2 / dx3^2
    H2_xx(STATE_THETA_DOT, STATE_THETA_DOT) = (2.0 * mp * l * s1) / den;
    // d^2f2 / dx1 dx3 (and dx3 dx1)
    double d2f2_dx1dx3 = ( (2.0 * mp * l * theta_dot * c1) * den - (2.0 * mp * l * theta_dot * s1) * (mp * sin2theta) ) / den_sq;
    H2_xx(STATE_THETA, STATE_THETA_DOT) = d2f2_dx1dx3;
    H2_xx(STATE_THETA_DOT, STATE_THETA) = d2f2_dx1dx3;
    // d^2f2 / dx1^2 (This is the most complex term for f2)
    // Let N2_dx1 = (mp * l * theta_dot^2 * c1 + mp * g * cos2theta)
    // Let Den_dx1 = mp * sin2theta
    // Let N2 = force + mp * l * theta_dot^2 * s1 + mp * g * s1 * c1
    // df2/dx1 = (N2_dx1 * Den - N2 * Den_dx1) / Den^2
    // Differentiating this again w.r.t. x1 is lengthy.
    // For now, as an example of structure:
    // H2_xx(STATE_THETA, STATE_THETA) = ... complicated expression ...;
    hessians_f_xx[STATE_X_DOT] = H2_xx;


    // Hessian for f3 (STATE_THETA_DOT dynamics)
    Eigen::MatrixXd H3_xx = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    // d^2f3 / dx3^2
    H3_xx(STATE_THETA_DOT, STATE_THETA_DOT) = (-mp * l * sin2theta) / (l * den); // = -mp * sin2theta / den
    // d^2f3 / dx1 dx3 (and dx3 dx1)
    // df3/dx3 = (-mp * l * theta_dot * sin2theta - d) / (l * den)
    double d2f3_dx1dx3_num_term1 = (-2.0 * mp * l * theta_dot * cos2theta) * (l*den);
    double d2f3_dx1dx3_num_term2 = (-mp * l * theta_dot * sin2theta - d) * (l * mp * sin2theta);
    H3_xx(STATE_THETA, STATE_THETA_DOT) = (d2f3_dx1dx3_num_term1 - d2f3_dx1dx3_num_term2) / (l*l*den_sq);
    H3_xx(STATE_THETA_DOT, STATE_THETA) = H3_xx(STATE_THETA, STATE_THETA_DOT);
    // d^2f3 / dx1^2 (This is the most complex term for f3)
    // Similar to H2_xx(STATE_THETA, STATE_THETA), this is very involved.
    // H3_xx(STATE_THETA, STATE_THETA) = ... complicated expression ...;
    hessians_f_xx[STATE_THETA_DOT] = H3_xx;

    return hessians_f_xx;
}

// This would be a new function, e.g., in CartPole.hpp
// std::vector<Eigen::MatrixXd> getMixedControlStateHessian(
//    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const;

// Implementation for getMixedControlStateHessian
std::vector<Eigen::MatrixXd> CartPole::getCrossHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians_f_ux(STATE_DIM); // Each is CONTROL_DIM x STATE_DIM

    const double theta = state(STATE_THETA); // x1

    const double s1 = std::sin(theta);
    const double c1 = std::cos(theta);
    const double s1_sq = s1 * s1;
    const double sin2theta = 2.0 * s1 * c1;

    const double mc = cart_mass_;
    const double mp = pole_mass_;
    const double l = pole_length_;

    const double den = mc + mp * s1_sq;
    const double den_sq = den * den;

    // Hessian for f0 (STATE_X dynamics)
    hessians_f_ux[STATE_X] = Eigen::MatrixXd::Zero(CONTROL_DIM, STATE_DIM);
    // Hessian for f1 (STATE_THETA dynamics)
    hessians_f_ux[STATE_THETA] = Eigen::MatrixXd::Zero(CONTROL_DIM, STATE_DIM);

    // Hessian for f2 (STATE_X_DOT dynamics)
    hessians_f_ux[STATE_X_DOT] = Eigen::MatrixXd::Zero(CONTROL_DIM, STATE_DIM);
    hessians_f_ux[STATE_X_DOT](CONTROL_FORCE, STATE_THETA) = -mp * sin2theta / den_sq; // d^2f2 / du0 dx1

    // Hessian for f3 (STATE_THETA_DOT dynamics)
    hessians_f_ux[STATE_THETA_DOT] = Eigen::MatrixXd::Zero(CONTROL_DIM, STATE_DIM);
    hessians_f_ux[STATE_THETA_DOT](CONTROL_FORCE, STATE_THETA) = (s1 * den + c1 * mp * sin2theta) / (l * den_sq); // d^2f3 / du0 dx1
    
    return hessians_f_ux;
}

std::vector<Eigen::MatrixXd> CartPole::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        // Since f_k is linear in control (force), f_uu is zero.
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

} // namespace cddp