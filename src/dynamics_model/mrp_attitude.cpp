/*
 Copyright 2025 Tomo Sasaki

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

#include "cddp-cpp/dynamics_model/mrp_attitude.hpp"
#include <autodiff/forward/dual.hpp>       // Include autodiff
#include <autodiff/forward/dual/eigen.hpp> // Include autodiff Eigen support
#include <cmath>                           // For tanh

namespace cddp {

MrpAttitude::MrpAttitude(double timestep, const Eigen::Matrix3d& inertia_matrix,
                         std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      inertia_(inertia_matrix),
      inertia_inv_(inertia_matrix.inverse()) {}

Eigen::VectorXd MrpAttitude::getContinuousDynamics(const Eigen::VectorXd& state,
                                                 const Eigen::VectorXd& control) const {
    Eigen::Vector3d mrp = state.segment<3>(STATE_MRP_X);
    Eigen::Vector3d omega = state.segment<3>(STATE_OMEGA_X);
    Eigen::Vector3d tau = control.segment<3>(CONTROL_TAU_X);

    Eigen::VectorXd state_dot(STATE_DIM);

    // Check for MRP switching condition
    double mrp_norm_sq = mrp.squaredNorm();
    Eigen::Vector3d mrp_for_kinematics = mrp;
    if (mrp_norm_sq > 1.0) {
        // Switch to shadow set for kinematics calculation
        mrp_for_kinematics = -mrp / mrp_norm_sq;
    }

    // MRP Kinematics: dmrp/dt = 0.25 * B(mrp) * omega
    // Use the potentially switched MRP for the B matrix
    state_dot.segment<3>(STATE_MRP_X) = 0.25 * mrpKinematicsMatrix<double>(mrp_for_kinematics) * omega;

    // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
    // This part is independent of the MRP representation
    state_dot.segment<3>(STATE_OMEGA_X) = inertia_inv_ * (-skew<double>(omega) * (inertia_ * omega) + tau);

    return state_dot;
}

Eigen::MatrixXd MrpAttitude::getStateJacobian(const Eigen::VectorXd& state,
                                            const Eigen::VectorXd& control) const {
    return DynamicalSystem::getStateJacobian(state, control); // Use autodiff
}

Eigen::MatrixXd MrpAttitude::getControlJacobian(const Eigen::VectorXd& state,
                                              const Eigen::VectorXd& control) const {
    return DynamicalSystem::getControlJacobian(state, control); // Use autodiff
}

std::vector<Eigen::MatrixXd> MrpAttitude::getStateHessian(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& control) const {
    return DynamicalSystem::getStateHessian(state, control); // Use autodiff
}

std::vector<Eigen::MatrixXd> MrpAttitude::getControlHessian(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control) const {
    return DynamicalSystem::getControlHessian(state, control); // Use autodiff
}

std::vector<Eigen::MatrixXd> MrpAttitude::getCrossHessian(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control) const {
    return DynamicalSystem::getCrossHessian(state, control); // Use autodiff
}

VectorXdual2nd MrpAttitude::getContinuousDynamicsAutodiff(const VectorXdual2nd& state,
                                                        const VectorXdual2nd& control) const {
    autodiff::Matrix3dual2nd inertia_ad = inertia_.cast<autodiff::dual2nd>();
    autodiff::Matrix3dual2nd inertia_inv_ad = inertia_inv_.cast<autodiff::dual2nd>();

    autodiff::Vector3dual2nd mrp = state.segment<3>(STATE_MRP_X);
    autodiff::Vector3dual2nd omega = state.segment<3>(STATE_OMEGA_X);
    autodiff::Vector3dual2nd tau = control.segment<3>(CONTROL_TAU_X);

    VectorXdual2nd state_dot(STATE_DIM);

    // Smooth MRP switching using tanh
    autodiff::dual2nd mrp_norm_sq_ad = mrp.squaredNorm();
    double k = 100.0; // Steepness factor for smooth transition
    autodiff::dual2nd weight = 0.5 * (1.0 - tanh(k * (mrp_norm_sq_ad - 1.0)));
    autodiff::Vector3dual2nd mrp_shadow = -mrp / mrp_norm_sq_ad;
    autodiff::Vector3dual2nd mrp_for_kinematics_ad = weight * mrp + (1.0 - weight) * mrp_shadow;

    // MRP Kinematics: dmrp/dt = 0.25 * B(mrp) * omega
    state_dot.segment<3>(STATE_MRP_X) = 0.25 * mrpKinematicsMatrix<autodiff::dual2nd>(mrp_for_kinematics_ad) * omega;

    // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
    state_dot.segment<3>(STATE_OMEGA_X) = inertia_inv_ad * (-skew<autodiff::dual2nd>(omega) * (inertia_ad * omega) + tau);

    return state_dot;
}

} // namespace cddp 