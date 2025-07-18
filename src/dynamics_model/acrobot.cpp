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

#include "dynamics_model/acrobot.hpp"
#include <cmath>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

Acrobot::Acrobot(double timestep,
                 double l1, double l2,
                 double m1, double m2,
                 double J1, double J2,
                 std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      l1_(l1), l2_(l2), m1_(m1), m2_(m2), J1_(J1), J2_(J2) {}

Eigen::VectorXd Acrobot::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract state variables
    const double theta1 = state(STATE_THETA1);
    const double theta2 = state(STATE_THETA2);
    const double theta1_dot = state(STATE_THETA1_DOT);
    const double theta2_dot = state(STATE_THETA2_DOT);
    
    // Extract control variable (torque on second joint)
    const double u = control(CONTROL_TORQUE);
    
    // Compute trigonometric functions
    double s1, c1, s2, c2;
    s1 = std::sin(theta1);
    c1 = std::cos(theta1);
    s2 = std::sin(theta2);
    c2 = std::cos(theta2);
    double c12 = std::cos(theta1 + theta2);
    
    // Mass matrix M
    double m11 = m1_*l1_*l1_ + J1_ + m2_*(l1_*l1_ + l2_*l2_ + 2*l1_*l2_*c2) + J2_;
    double m12 = m2_*(l2_*l2_ + l1_*l2_*c2 + J2_);
    double m22 = l2_*l2_*m2_ + J2_;
    
    Eigen::Matrix2d M;
    M << m11, m12,
         m12, m22;
    
    // Bias term B (Coriolis forces)
    double tmp = l1_*l2_*m2_*s2;
    double b1 = -(2 * theta1_dot * theta2_dot + theta2_dot*theta2_dot)*tmp;
    double b2 = tmp * theta1_dot*theta1_dot;
    
    Eigen::Vector2d B;
    B << b1, b2;
    
    // Friction term C
    Eigen::Vector2d C;
    C << friction_*theta1_dot, friction_*theta2_dot;
    
    // Gravity term G
    double g1 = ((m1_ + m2_)*l1_*c1 + m2_*l2_*c12) * gravity_;
    double g2 = m2_*l2_*c12*gravity_;
    
    Eigen::Vector2d G;
    G << g1, g2;
    
    // Control torque vector
    Eigen::Vector2d tau;
    tau << 0, u;
    
    // Equations of motion: M*q_ddot = tau - B - G - C
    Eigen::Vector2d q_ddot = M.inverse() * (tau - B - G - C);
    
    // Assemble state derivative
    state_dot(STATE_THETA1) = theta1_dot;
    state_dot(STATE_THETA2) = theta2_dot;
    state_dot(STATE_THETA1_DOT) = q_ddot(0);
    state_dot(STATE_THETA2_DOT) = q_ddot(1);
    
    return state_dot;
}

VectorXdual2nd Acrobot::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {
    
    using namespace autodiff;
    
    VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);
    
    // Extract state variables
    const dual2nd theta1 = state(STATE_THETA1);
    const dual2nd theta2 = state(STATE_THETA2);
    const dual2nd theta1_dot = state(STATE_THETA1_DOT);
    const dual2nd theta2_dot = state(STATE_THETA2_DOT);
    
    // Extract control variable
    const dual2nd u = control(CONTROL_TORQUE);
    
    // Compute trigonometric functions
    dual2nd s1 = sin(theta1);
    dual2nd c1 = cos(theta1);
    dual2nd s2 = sin(theta2);
    dual2nd c2 = cos(theta2);
    dual2nd c12 = cos(theta1 + theta2);
    
    // Mass matrix M
    dual2nd m11 = m1_*l1_*l1_ + J1_ + m2_*(l1_*l1_ + l2_*l2_ + 2*l1_*l2_*c2) + J2_;
    dual2nd m12 = m2_*(l2_*l2_ + l1_*l2_*c2 + J2_);
    dual2nd m22 = dual2nd(l2_*l2_*m2_ + J2_);
    
    Eigen::Matrix<dual2nd, 2, 2> M;
    M << m11, m12,
         m12, m22;
    
    // Bias term B (Coriolis forces)
    dual2nd tmp = l1_*l2_*m2_*s2;
    dual2nd b1 = -(2 * theta1_dot * theta2_dot + theta2_dot*theta2_dot)*tmp;
    dual2nd b2 = tmp * theta1_dot*theta1_dot;
    
    Eigen::Matrix<dual2nd, 2, 1> B;
    B << b1, b2;
    
    // Friction term C
    Eigen::Matrix<dual2nd, 2, 1> C;
    C << friction_*theta1_dot, friction_*theta2_dot;
    
    // Gravity term G
    dual2nd g1 = ((m1_ + m2_)*l1_*c1 + m2_*l2_*c12) * gravity_;
    dual2nd g2 = m2_*l2_*c12*gravity_;
    
    Eigen::Matrix<dual2nd, 2, 1> G;
    G << g1, g2;
    
    // Control torque vector
    Eigen::Matrix<dual2nd, 2, 1> tau;
    tau << dual2nd(0.0), u;
    
    // Equations of motion: M*q_ddot = tau - B - G - C
    Eigen::Matrix<dual2nd, 2, 1> q_ddot = M.inverse() * (tau - B - G - C);
    
    // Assemble state derivative
    state_dot(STATE_THETA1) = theta1_dot;
    state_dot(STATE_THETA2) = theta2_dot;
    state_dot(STATE_THETA1_DOT) = q_ddot(0);
    state_dot(STATE_THETA2_DOT) = q_ddot(1);
    
    return state_dot;
}

} // namespace cddp