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

#include "dynamics_model/quadrotor.hpp"
#include <cmath>
#include "cddp_core/helper.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace cddp {

Quadrotor::Quadrotor(double timestep, double mass, const Eigen::Matrix3d& inertia_matrix,
                     double arm_length, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mass_(mass),
      inertia_(inertia_matrix),
      arm_length_(arm_length) {
}

Eigen::VectorXd Quadrotor::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

    // --- Position Derivative ---
    // The derivative of the position is the linear velocity.
    state_dot.segment<3>(STATE_X) = state.segment<3>(STATE_VX);

    // --- Quaternion Derivative ---
    // Extract the quaternion (assumed to be [qw, qx, qy, qz])
    double qw = state(STATE_QW);
    double qx = state(STATE_QX);
    double qy = state(STATE_QY);
    double qz = state(STATE_QZ);

    // Normalize the quaternion to enforce unit norm before further use.
    double norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (norm > 1e-6) {
        qw /= norm;
        qx /= norm;
        qy /= norm;
        qz /= norm;
    } else {
        // In degenerate cases, default to the identity quaternion.
        qw = 1.0; qx = 0.0; qy = 0.0; qz = 0.0;
    }

    // Extract body angular velocity components
    double omega_x = state(STATE_OMEGA_X);
    double omega_y = state(STATE_OMEGA_Y);
    double omega_z = state(STATE_OMEGA_Z);

    // Compute quaternion derivative using: q_dot = 0.5 * q ⊗ [0, omega]
    // Quaternion multiplication yields:
    // q_dot[0] = -0.5 * (qx*omega_x + qy*omega_y + qz*omega_z)
    // q_dot[1] =  0.5 * (qw*omega_x + qy*omega_z - qz*omega_y)
    // q_dot[2] =  0.5 * (qw*omega_y - qx*omega_z + qz*omega_x)
    // q_dot[3] =  0.5 * (qw*omega_z + qx*omega_y - qy*omega_x)
    state_dot(STATE_QW) = -0.5 * (qx * omega_x + qy * omega_y + qz * omega_z);
    state_dot(STATE_QX) =  0.5 * (qw * omega_x + qy * omega_z - qz * omega_y);
    state_dot(STATE_QY) =  0.5 * (qw * omega_y - qx * omega_z + qz * omega_x);
    state_dot(STATE_QZ) =  0.5 * (qw * omega_z + qx * omega_y - qy * omega_x);

    // --- Velocity Derivative ---
    // Extract control variables (motor forces)
    const double f1 = control(CONTROL_F1);
    const double f2 = control(CONTROL_F2);
    const double f3 = control(CONTROL_F3);
    const double f4 = control(CONTROL_F4);
    
    // Compute total thrust and moments 
    const double thrust = f1 + f2 + f3 + f4;
    const double tau_x = arm_length_ * (f1 - f3);
    const double tau_y = arm_length_ * (f2 - f4);
    const double tau_z = 0.1 * (f1 - f2 + f3 - f4);

    // Compute rotation matrix from the normalized quaternion
    Eigen::Matrix3d R = getRotationMatrix(qw, qx, qy, qz);

    // Thrust is applied along the body z-axis. 
    Eigen::Vector3d F_thrust(0, 0, thrust);
    Eigen::Vector3d acceleration = (1.0/mass_) * (R * F_thrust) - Eigen::Vector3d(0, 0, gravity_);
    state_dot.segment<3>(STATE_VX) = acceleration;

    // --- Angular Velocity Derivative ---
    Eigen::Vector3d omega(omega_x, omega_y, omega_z);
    Eigen::Vector3d tau(tau_x, tau_y, tau_z);
    Eigen::Vector3d angular_acc = inertia_.inverse() * (tau - omega.cross(inertia_ * omega));
    state_dot.segment<3>(STATE_OMEGA_X) = angular_acc;

    return state_dot;
}

Eigen::Matrix3d Quadrotor::getRotationMatrix(double qw, double qx, double qy, double qz) const {
    // Compute the rotation matrix from a unit quaternion.
    // The quaternion is assumed to be normalized and in the form [qw, qx, qy, qz]
    Eigen::Matrix3d R;
    R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
    R(0, 1) = 2 * (qx * qy - qz * qw);
    R(0, 2) = 2 * (qx * qz + qy * qw);
    
    R(1, 0) = 2 * (qx * qy + qz * qw);
    R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
    R(1, 2) = 2 * (qy * qz - qx * qw);
    
    R(2, 0) = 2 * (qx * qz - qy * qw);
    R(2, 1) = 2 * (qy * qz + qx * qw);
    R(2, 2) = 1 - 2 * (qx * qx + qy * qy);
    
    return R;
}

Eigen::MatrixXd Quadrotor::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {

    auto f = [&](const Eigen::VectorXd& x) {
        return getContinuousDynamics(x, control);
    };

    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd Quadrotor::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    auto f = [&](const Eigen::VectorXd& u) {
        return getContinuousDynamics(state, u);
    };
    return finite_difference_jacobian(f, control);
}


// TODO: Implement a more accurate version if needed
std::vector<Eigen::MatrixXd> Quadrotor::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

// TODO: Implement a more accurate version if needed
std::vector<Eigen::MatrixXd> Quadrotor::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

Eigen::Matrix<autodiff::dual2nd, 3, 3> Quadrotor::getRotationMatrixAutodiff(
    const autodiff::dual2nd& qw, const autodiff::dual2nd& qx,
    const autodiff::dual2nd& qy, const autodiff::dual2nd& qz) const {
    // No trig functions here, just multiplication
}

VectorXdual2nd Quadrotor::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control) const {
    // ... lots of state/control extraction ...
    // No direct trig calls here, done in getRotationMatrixAutodiff (which doesn't use trig)
    // Relies on Eigen operations (.cross, *, /) on dual types.
}

} // namespace cddp
