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

#include "dynamics_model/manipulator.hpp"
#include <cmath>
#include "cddp_core/helper.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

Manipulator::Manipulator(double timestep, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type) {
}

Eigen::VectorXd Manipulator::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);
    
    // Extract joint positions and velocities
    Eigen::VectorXd q = state.segment(0, NUM_JOINTS);
    Eigen::VectorXd dq = state.segment(NUM_JOINTS, NUM_JOINTS);
    
    // Get dynamic matrices
    Eigen::MatrixXd M = getMassMatrix(q);
    Eigen::VectorXd G = getGravityVector(q);
    
    // Compute accelerations using simplified dynamics:
    // M(q)ddq + G(q) = tau
    Eigen::VectorXd ddq = M.inverse() * (control - G);
    
    // State derivative
    state_dot.segment(0, NUM_JOINTS) = dq;
    state_dot.segment(NUM_JOINTS, NUM_JOINTS) = ddq;
    
    return state_dot;
}

Eigen::MatrixXd Manipulator::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {

    auto f = [&](const Eigen::VectorXd& x) {
        return getContinuousDynamics(x, control, time);
    };

    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd Manipulator::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    auto f = [&](const Eigen::VectorXd& u) {
        return getContinuousDynamics(state, u, time);
    };
    return finite_difference_jacobian(f, control);
}

std::vector<Eigen::MatrixXd> Manipulator::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    // For the simplified model, return zero Hessian
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> Manipulator::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    
    // For the simplified model, return zero Hessian
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

Eigen::Matrix4d Manipulator::rotX(double alpha) const {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    double c = cos(alpha);
    double s = sin(alpha);
    mat << 1,  0,  0, 0,
           0,  c, -s, 0,
           0,  s,  c, 0,
           0,  0,  0, 1;
    return mat;
}

Eigen::Matrix4d Manipulator::rotY(double beta) const {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    double c = cos(beta);
    double s = sin(beta);
    mat <<  c, 0, s, 0,
            0, 1, 0, 0,
           -s, 0, c, 0,
            0, 0, 0, 1;
    return mat;
}

Eigen::Matrix4d Manipulator::rotZ(double theta) const {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    double c = cos(theta);
    double s = sin(theta);
    mat << c, -s, 0, 0,
           s,  c, 0, 0,
           0,  0, 1, 0,
           0,  0, 0, 1;
    return mat;
}

std::vector<Eigen::Matrix4d> Manipulator::getTransformationMatrices(
    double theta1, double theta2, double theta3) const {
    
    std::vector<Eigen::Matrix4d> transforms;
    transforms.reserve(4);

    // T01: Based on MATLAB implementation
    Eigen::Matrix4d T01 = rotZ(theta1);
    transforms.push_back(T01);

    // T12: First joint to second joint
    Eigen::Matrix4d T12 = rotX(alpha1_) * rotZ(theta2);
    transforms.push_back(T12);

    // T23: Second joint to third joint
    Eigen::Matrix4d T23 = rotX(alpha2_) * rotZ(theta3);
    T23.block<3,1>(0,3) = Eigen::Vector3d(la_, 0, 0);  // Translation part
    transforms.push_back(T23);

    // T34: Third joint to end effector
    Eigen::Matrix4d T34 = rotX(alpha3_) * rotZ(0);
    T34.block<3,1>(0,3) = Eigen::Vector3d(lc_, 0, lb_);  // Translation part
    transforms.push_back(T34);

    return transforms;
}

Eigen::Matrix4d Manipulator::getForwardKinematics(const Eigen::VectorXd& state) const {
    // Extract joint angles
    double theta1 = state(0);
    double theta2 = state(1);
    double theta3 = state(2);
    
    // Get transformation matrices
    auto transforms = getTransformationMatrices(theta1, theta2, theta3);
    
    // Compute full transformation T04
    Eigen::Matrix4d T = transforms[0];
    for (size_t i = 1; i < transforms.size(); ++i) {
        T = T * transforms[i];
    }
    
    return T;
}

Eigen::Vector3d Manipulator::getEndEffectorPosition(const Eigen::VectorXd& state) const {
    // Get the full transformation matrix
    Eigen::Matrix4d T = getForwardKinematics(state);
    
    // Return the position part (last column, first 3 elements)
    return T.block<3,1>(0,3);
}

Eigen::MatrixXd Manipulator::getMassMatrix(const Eigen::VectorXd& q) const {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(NUM_JOINTS, NUM_JOINTS);
    
    // Simplified mass matrix - assuming point masses at the end of each link
    double m1 = 1.0;  // Mass of first link
    double m2 = 1.0;  // Mass of second link
    double m3 = 0.5;  // Mass of third link (end effector)
    
    // Diagonal terms (simplified)
    M(0,0) = (m1 + m2 + m3) * (la_ * la_);
    M(1,1) = (m2 + m3) * (lb_ * lb_);
    M(2,2) = m3 * (lc_ * lc_);
    
    // Off-diagonal terms (coupling between joints - simplified)
    M(0,1) = M(1,0) = (m2 + m3) * la_ * lb_ * cos(q(1));
    M(1,2) = M(2,1) = m3 * lb_ * lc_ * cos(q(2));
    M(0,2) = M(2,0) = m3 * la_ * lc_ * cos(q(1) + q(2));
    
    return M;
}

Eigen::VectorXd Manipulator::getGravityVector(const Eigen::VectorXd& q) const {
    Eigen::VectorXd G = Eigen::VectorXd::Zero(NUM_JOINTS);
    
    double m1 = 1.0;
    double m2 = 1.0;
    double m3 = 0.5;
    
    G(0) = 0;  // Base joint not affected by gravity
    G(1) = -(m2 + m3) * gravity_ * lb_ * cos(q(1)) 
           - m3 * gravity_ * lc_ * cos(q(1) + q(2));
    G(2) = -m3 * gravity_ * lc_ * cos(q(1) + q(2));
    
    return G;
}

Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, Eigen::Dynamic> Manipulator::getMassMatrixAutodiff(const autodiff::VectorXdual2nd& q) const {
    Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, Eigen::Dynamic> M = Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, Eigen::Dynamic>::Zero(NUM_JOINTS, NUM_JOINTS);
    
    // Simplified mass matrix - assuming point masses at the end of each link
    double m1 = 1.0;  // Mass of first link
    double m2 = 1.0;  // Mass of second link
    double m3 = 0.5;  // Mass of third link (end effector)
    
    // Use ADL for math functions
    autodiff::dual2nd cos_q1 = cos(q(1));
    autodiff::dual2nd cos_q2 = cos(q(2));
    autodiff::dual2nd cos_q1_plus_q2 = cos(q(1) + q(2));
    
    // Diagonal terms (simplified)
    M(0,0) = (m1 + m2 + m3) * (la_ * la_);
    M(1,1) = (m2 + m3) * (lb_ * lb_);
    M(2,2) = m3 * (lc_ * lc_);
    
    // Off-diagonal terms (coupling between joints - simplified)
    M(0,1) = M(1,0) = (m2 + m3) * la_ * lb_ * cos_q1;
    M(1,2) = M(2,1) = m3 * lb_ * lc_ * cos_q2;
    M(0,2) = M(2,0) = m3 * la_ * lc_ * cos_q1_plus_q2;
    
    return M;
}

autodiff::VectorXdual2nd Manipulator::getGravityVectorAutodiff(const autodiff::VectorXdual2nd& q) const {
    autodiff::VectorXdual2nd G = autodiff::VectorXdual2nd::Zero(NUM_JOINTS);
    
    double m1 = 1.0;
    double m2 = 1.0;
    double m3 = 0.5;
    
    G(0) = 0;  // Base joint not affected by gravity
    G(1) = -(m2 + m3) * gravity_ * lb_ * cos(q(1)) 
           - m3 * gravity_ * lc_ * cos(q(1) + q(2));
    G(2) = -m3 * gravity_ * lc_ * cos(q(1) + q(2));
    
    return G;
}

autodiff::VectorXdual2nd Manipulator::getContinuousDynamicsAutodiff(
    const autodiff::VectorXdual2nd& state, const autodiff::VectorXdual2nd& control, double time) const {
    autodiff::VectorXdual2nd state_dot = autodiff::VectorXdual2nd::Zero(STATE_DIM);
    
    // Extract joint positions and velocities
    autodiff::VectorXdual2nd q = state.segment(0, NUM_JOINTS);
    autodiff::VectorXdual2nd dq = state.segment(NUM_JOINTS, NUM_JOINTS);
    
    // Get dynamic matrices
    Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, Eigen::Dynamic> M = getMassMatrixAutodiff(q);
    autodiff::VectorXdual2nd G = getGravityVectorAutodiff(q);
    
    // Compute accelerations using simplified dynamics:
    // M(q)ddq + G(q) = tau
    autodiff::VectorXdual2nd ddq = M.inverse() * (control - G);
    
    // State derivative
    state_dot.segment(0, NUM_JOINTS) = dq;
    state_dot.segment(NUM_JOINTS, NUM_JOINTS) = ddq;
    
    return state_dot;
}

} // namespace cddp