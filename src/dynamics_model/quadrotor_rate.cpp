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

#include "cddp-cpp/dynamics_model/quadrotor_rate.hpp"
#include <iostream>

namespace cddp {

QuadrotorRate::QuadrotorRate(double timestep, double mass, double max_thrust, double max_rate,
                           std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mass_(mass), max_thrust_(max_thrust), max_rate_(max_rate) {
    
    // Validate parameters
    if (mass_ <= 0.0) {
        throw std::invalid_argument("Mass must be positive");
    }
    if (max_thrust_ <= 0.0) {
        throw std::invalid_argument("Maximum thrust must be positive");
    }
    if (max_rate_ <= 0.0) {
        throw std::invalid_argument("Maximum angular rate must be positive");
    }
}

Eigen::VectorXd QuadrotorRate::getContinuousDynamics(const Eigen::VectorXd& state, 
                                                     const Eigen::VectorXd& control, 
                                                     double /*time*/) const {
    // Validate input dimensions
    if (state.size() != STATE_DIM) {
        throw std::invalid_argument("State dimension mismatch. Expected " + 
                                  std::to_string(STATE_DIM) + ", got " + 
                                  std::to_string(state.size()));
    }
    if (control.size() != CONTROL_DIM) {
        throw std::invalid_argument("Control dimension mismatch. Expected " + 
                                  std::to_string(CONTROL_DIM) + ", got " + 
                                  std::to_string(control.size()));
    }

    // Extract state components
    const Eigen::Vector3d position = state.segment<3>(STATE_PX);
    const Eigen::Vector3d velocity = state.segment<3>(STATE_VX);
    const double qw = state(STATE_QW);
    const double qx = state(STATE_QX);
    const double qy = state(STATE_QY);
    const double qz = state(STATE_QZ);

    // Normalize quaternion
    const double q_norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    const double qw_n = qw / q_norm;
    const double qx_n = qx / q_norm;
    const double qy_n = qy / q_norm;
    const double qz_n = qz / q_norm;

    // Extract control inputs
    const double thrust = control(CONTROL_THRUST);
    const double wx = control(CONTROL_WX);
    const double wy = control(CONTROL_WY);
    const double wz = control(CONTROL_WZ);

    // Initialize state derivative
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

    // Position derivatives (velocity)
    state_dot.segment<3>(STATE_PX) = velocity;

    // Velocity derivatives (acceleration)
    // Compute rotation matrix from quaternion
    const Eigen::Matrix3d R = getRotationMatrix(qw_n, qx_n, qy_n, qz_n);
    
    // Thrust acts along body z-axis (upward in body frame)
    const Eigen::Vector3d thrust_body(0.0, 0.0, thrust);
    const Eigen::Vector3d thrust_world = R * thrust_body;
    
    // Acceleration = thrust/mass + gravity
    const Eigen::Vector3d gravity_vec(0.0, 0.0, -gravity_);
    state_dot.segment<3>(STATE_VX) = thrust_world / mass_ + gravity_vec;

    // Quaternion derivatives
    // q_dot = 0.5 * Omega(w) * q
    const Eigen::Matrix4d Omega = getOmegaMatrix(wx, wy, wz);
    const Eigen::Vector4d q_current(qw_n, qx_n, qy_n, qz_n);
    const Eigen::Vector4d q_dot = 0.5 * Omega * q_current;
    
    state_dot(STATE_QW) = q_dot(0);
    state_dot(STATE_QX) = q_dot(1);
    state_dot(STATE_QY) = q_dot(2);
    state_dot(STATE_QZ) = q_dot(3);

    return state_dot;
}

Eigen::MatrixXd QuadrotorRate::getStateJacobian(const Eigen::VectorXd& state, 
                                               const Eigen::VectorXd& control, 
                                               double time) const {
    // Use base class autodiff implementation
    return DynamicalSystem::getStateJacobian(state, control, time);
}

Eigen::MatrixXd QuadrotorRate::getControlJacobian(const Eigen::VectorXd& state, 
                                                 const Eigen::VectorXd& control, 
                                                 double time) const {
    // Use base class autodiff implementation
    return DynamicalSystem::getControlJacobian(state, control, time);
}

std::vector<Eigen::MatrixXd> QuadrotorRate::getStateHessian(const Eigen::VectorXd& state, 
                                                           const Eigen::VectorXd& control, 
                                                           double time) const {
    // Use base class autodiff implementation
    return DynamicalSystem::getStateHessian(state, control, time);
}

std::vector<Eigen::MatrixXd> QuadrotorRate::getControlHessian(const Eigen::VectorXd& state, 
                                                             const Eigen::VectorXd& control, 
                                                             double time) const {
    // Use base class autodiff implementation
    return DynamicalSystem::getControlHessian(state, control, time);
}

std::vector<Eigen::MatrixXd> QuadrotorRate::getCrossHessian(const Eigen::VectorXd& state, 
                                                           const Eigen::VectorXd& control, 
                                                           double time) const {
    // Use base class autodiff implementation
    return DynamicalSystem::getCrossHessian(state, control, time);
}

VectorXdual2nd QuadrotorRate::getContinuousDynamicsAutodiff(const VectorXdual2nd& state, 
                                                           const VectorXdual2nd& control, 
                                                           double /*time*/) const {
    // Extract state components
    const auto& px = state(STATE_PX);
    const auto& py = state(STATE_PY);
    const auto& pz = state(STATE_PZ);
    const auto& vx = state(STATE_VX);
    const auto& vy = state(STATE_VY);
    const auto& vz = state(STATE_VZ);
    const auto& qw = state(STATE_QW);
    const auto& qx = state(STATE_QX);
    const auto& qy = state(STATE_QY);
    const auto& qz = state(STATE_QZ);

    // Normalize quaternion
    const auto q_norm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    const auto qw_n = qw / q_norm;
    const auto qx_n = qx / q_norm;
    const auto qy_n = qy / q_norm;
    const auto qz_n = qz / q_norm;

    // Extract control inputs
    const auto& thrust = control(CONTROL_THRUST);
    const auto& wx = control(CONTROL_WX);
    const auto& wy = control(CONTROL_WY);
    const auto& wz = control(CONTROL_WZ);

    // Initialize state derivative
    VectorXdual2nd state_dot(STATE_DIM);

    // Position derivatives (velocity)
    state_dot(STATE_PX) = vx;
    state_dot(STATE_PY) = vy;
    state_dot(STATE_PZ) = vz;

    // Velocity derivatives (acceleration)
    // Compute rotation matrix from quaternion
    const auto R = getRotationMatrixAutodiff(qw_n, qx_n, qy_n, qz_n);
    
    // Thrust acts along body z-axis
    const auto thrust_world_x = R(0, 2) * thrust;
    const auto thrust_world_y = R(1, 2) * thrust;
    const auto thrust_world_z = R(2, 2) * thrust;
    
    // Acceleration = thrust/mass + gravity
    state_dot(STATE_VX) = thrust_world_x / mass_;
    state_dot(STATE_VY) = thrust_world_y / mass_;
    state_dot(STATE_VZ) = thrust_world_z / mass_ - gravity_;

    // Quaternion derivatives
    // q_dot = 0.5 * Omega(w) * q
    const auto Omega = getOmegaMatrixAutodiff(wx, wy, wz);
    
    state_dot(STATE_QW) = 0.5 * (-qx_n * wx - qy_n * wy - qz_n * wz);
    state_dot(STATE_QX) = 0.5 * (qw_n * wx + qy_n * wz - qz_n * wy);
    state_dot(STATE_QY) = 0.5 * (qw_n * wy - qx_n * wz + qz_n * wx);
    state_dot(STATE_QZ) = 0.5 * (qw_n * wz + qx_n * wy - qy_n * wx);

    return state_dot;
}

Eigen::Matrix3d QuadrotorRate::getRotationMatrix(double qw, double qx, double qy, double qz) const {
    Eigen::Matrix3d R;
    
    R(0, 0) = 1.0 - 2.0 * (qy * qy + qz * qz);
    R(0, 1) = 2.0 * (qx * qy - qz * qw);
    R(0, 2) = 2.0 * (qx * qz + qy * qw);
    
    R(1, 0) = 2.0 * (qx * qy + qz * qw);
    R(1, 1) = 1.0 - 2.0 * (qx * qx + qz * qz);
    R(1, 2) = 2.0 * (qy * qz - qx * qw);
    
    R(2, 0) = 2.0 * (qx * qz - qy * qw);
    R(2, 1) = 2.0 * (qy * qz + qx * qw);
    R(2, 2) = 1.0 - 2.0 * (qx * qx + qy * qy);
    
    return R;
}

Eigen::Matrix<autodiff::dual2nd, 3, 3> QuadrotorRate::getRotationMatrixAutodiff(
    const autodiff::dual2nd& qw, const autodiff::dual2nd& qx,
    const autodiff::dual2nd& qy, const autodiff::dual2nd& qz) const {
    
    Eigen::Matrix<autodiff::dual2nd, 3, 3> R;
    
    R(0, 0) = 1.0 - 2.0 * (qy * qy + qz * qz);
    R(0, 1) = 2.0 * (qx * qy - qz * qw);
    R(0, 2) = 2.0 * (qx * qz + qy * qw);
    
    R(1, 0) = 2.0 * (qx * qy + qz * qw);
    R(1, 1) = 1.0 - 2.0 * (qx * qx + qz * qz);
    R(1, 2) = 2.0 * (qy * qz - qx * qw);
    
    R(2, 0) = 2.0 * (qx * qz - qy * qw);
    R(2, 1) = 2.0 * (qy * qz + qx * qw);
    R(2, 2) = 1.0 - 2.0 * (qx * qx + qy * qy);
    
    return R;
}

Eigen::Matrix4d QuadrotorRate::getOmegaMatrix(double wx, double wy, double wz) const {
    Eigen::Matrix4d Omega;
    
    Omega << 0.0, -wx, -wy, -wz,
             wx,  0.0,  wz, -wy,
             wy, -wz,  0.0,  wx,
             wz,  wy, -wx,  0.0;
    
    return Omega;
}

Eigen::Matrix<autodiff::dual2nd, 4, 4> QuadrotorRate::getOmegaMatrixAutodiff(
    const autodiff::dual2nd& wx, const autodiff::dual2nd& wy, 
    const autodiff::dual2nd& wz) const {
    
    Eigen::Matrix<autodiff::dual2nd, 4, 4> Omega;
    
    Omega(0, 0) = 0.0; Omega(0, 1) = -wx; Omega(0, 2) = -wy; Omega(0, 3) = -wz;
    Omega(1, 0) = wx;  Omega(1, 1) = 0.0; Omega(1, 2) = wz;  Omega(1, 3) = -wy;
    Omega(2, 0) = wy;  Omega(2, 1) = -wz; Omega(2, 2) = 0.0; Omega(2, 3) = wx;
    Omega(3, 0) = wz;  Omega(3, 1) = wy;  Omega(3, 2) = -wx; Omega(3, 3) = 0.0;
    
    return Omega;
}

} // namespace cddp