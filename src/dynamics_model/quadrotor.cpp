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
    
    // Extract state variables
    const double phi = state(STATE_PHI);      // roll
    const double theta = state(STATE_THETA);  // pitch
    const double psi = state(STATE_PSI);      // yaw
    
    const double vx = state(STATE_VX);
    const double vy = state(STATE_VY);
    const double vz = state(STATE_VZ);
    
    const double omega_x = state(STATE_OMEGA_X);
    const double omega_y = state(STATE_OMEGA_Y);
    const double omega_z = state(STATE_OMEGA_Z);
    
    // Extract control variables (motor forces)
    const double f1 = control(CONTROL_F1);
    const double f2 = control(CONTROL_F2);
    const double f3 = control(CONTROL_F3);
    const double f4 = control(CONTROL_F4);
    
    // Compute total thrust and moments
    const double thrust = f1 + f2 + f3 + f4;
    const double tau_x = arm_length_ * (f1 - f3);  // roll moment
    const double tau_y = arm_length_ * (f2 - f4);  // pitch moment
    const double tau_z = 0.1 * (f1 - f2 + f3 - f4); // yaw moment (assumed drag coefficient)
    
    // Get rotation matrix
    Eigen::Matrix3d R = getRotationMatrix(phi, theta, psi);
    
    // Position derivatives (velocity)
    state_dot.segment<3>(STATE_X) = state.segment<3>(STATE_VX);
    
    // Velocity derivatives (acceleration)
    Eigen::Vector3d F_thrust(0, 0, thrust);
    Eigen::Vector3d acceleration = (1.0/mass_) * (R * F_thrust) - 
                                 Eigen::Vector3d(0, 0, gravity_);
    state_dot.segment<3>(STATE_VX) = acceleration;
    
    // Angular velocity to Euler rates transformation
    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double c_theta = std::cos(theta);
    double t_theta = std::tan(theta);
    
    Eigen::Matrix3d W;
    W << 1, s_phi*t_theta, c_phi*t_theta,
         0, c_phi, -s_phi,
         0, s_phi/c_theta, c_phi/c_theta;
    
    // Euler angle derivatives
    state_dot.segment<3>(STATE_PHI) = W * state.segment<3>(STATE_OMEGA_X);
    
    // Angular acceleration
    Eigen::Vector3d omega(omega_x, omega_y, omega_z);
    Eigen::Vector3d tau(tau_x, tau_y, tau_z);
    Eigen::Vector3d angular_acc = inertia_.inverse() * 
        (tau - omega.cross(inertia_ * omega));
    
    state_dot.segment<3>(STATE_OMEGA_X) = angular_acc;
    
    return state_dot;
}

Eigen::Matrix3d Quadrotor::getRotationMatrix(double phi, double theta, double psi) const {
    // Compute rotation matrix from Euler angles (ZYX convention)
    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double c_theta = std::cos(theta);
    double s_theta = std::sin(theta);
    double c_psi = std::cos(psi);
    double s_psi = std::sin(psi);
    
    Eigen::Matrix3d R;
    R << c_psi*c_theta, c_psi*s_theta*s_phi - s_psi*c_phi, c_psi*s_theta*c_phi + s_psi*s_phi,
         s_psi*c_theta, s_psi*s_theta*s_phi + c_psi*c_phi, s_psi*s_theta*c_phi - c_psi*s_phi,
         -s_theta, c_theta*s_phi, c_theta*c_phi;
    
    return R;
}

Eigen::MatrixXd Quadrotor::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd A = getFiniteDifferenceStateJacobian(state, control);
    
    return A;
}

Eigen::MatrixXd Quadrotor::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    Eigen::MatrixXd B = getFiniteDifferenceControlJacobian(state, control);

    return B;
}

Eigen::MatrixXd Quadrotor::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(STATE_DIM * STATE_DIM, STATE_DIM);
}

Eigen::MatrixXd Quadrotor::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(STATE_DIM * CONTROL_DIM, CONTROL_DIM);
}

} // namespace cddp