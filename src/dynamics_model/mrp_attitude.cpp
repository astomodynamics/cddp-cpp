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

namespace cddp
{

    MrpAttitude::MrpAttitude(double timestep, const Eigen::Matrix3d &inertia_matrix,
                             std::string integration_type)
        : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
          inertia_(inertia_matrix),
          inertia_inv_(inertia_matrix.inverse()) {}

    Eigen::VectorXd MrpAttitude::getContinuousDynamics(const Eigen::VectorXd &state,
                                                       const Eigen::VectorXd &control) const
    {
        Eigen::Vector3d mrp = state.segment<3>(STATE_MRP_X);
        Eigen::Vector3d omega = state.segment<3>(STATE_OMEGA_X);
        Eigen::Vector3d tau = control.segment<3>(CONTROL_TAU_X);

        Eigen::VectorXd state_dot(STATE_DIM);

        // Correct MRP Kinematics: dmrp/dt = 0.25 * B(mrp) * omega
        // Use the *current* mrp state for the kinematics matrix B.
        state_dot.segment<3>(STATE_MRP_X) = 0.25 * this->mrpKinematicsMatrix<double>(mrp) * omega;

        // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
        state_dot.segment<3>(STATE_OMEGA_X) = this->inertia_inv_ * (-this->skew<double>(omega) * (this->inertia_ * omega) + tau);

        return state_dot;
    }

    Eigen::MatrixXd MrpAttitude::getStateJacobian(const Eigen::VectorXd &state,
                                                  const Eigen::VectorXd &control) const
    {
        return DynamicalSystem::getStateJacobian(state, control); // Use autodiff
    }

    Eigen::MatrixXd MrpAttitude::getControlJacobian(const Eigen::VectorXd &state,
                                                    const Eigen::VectorXd &control) const
    {
        return DynamicalSystem::getControlJacobian(state, control); // Use autodiff
    }

    std::vector<Eigen::MatrixXd> MrpAttitude::getStateHessian(const Eigen::VectorXd &state,
                                                              const Eigen::VectorXd &control) const
    {
        return DynamicalSystem::getStateHessian(state, control); // Use autodiff
    }

    std::vector<Eigen::MatrixXd> MrpAttitude::getControlHessian(const Eigen::VectorXd &state,
                                                                const Eigen::VectorXd &control) const
    {
        return DynamicalSystem::getControlHessian(state, control); // Use autodiff
    }

    std::vector<Eigen::MatrixXd> MrpAttitude::getCrossHessian(const Eigen::VectorXd &state,
                                                              const Eigen::VectorXd &control) const
    {
        return DynamicalSystem::getCrossHessian(state, control); // Use autodiff
    }

    VectorXdual2nd MrpAttitude::getContinuousDynamicsAutodiff(const VectorXdual2nd &state,
                                                              const VectorXdual2nd &control) const
    {
        autodiff::Matrix3dual2nd inertia_ad = this->inertia_.cast<autodiff::dual2nd>();
        autodiff::Matrix3dual2nd inertia_inv_ad = this->inertia_inv_.cast<autodiff::dual2nd>();

        autodiff::Vector3dual2nd mrp = state.segment<3>(STATE_MRP_X);
        autodiff::Vector3dual2nd omega = state.segment<3>(STATE_OMEGA_X);
        autodiff::Vector3dual2nd tau = control.segment<3>(CONTROL_TAU_X);

        VectorXdual2nd state_dot(STATE_DIM);

        // MRP Kinematics: dmrp/dt = 0.25 * B(mrp) * omega
        state_dot.segment<3>(STATE_MRP_X) = 0.25 * this->mrpKinematicsMatrix<autodiff::dual2nd>(mrp) * omega;

        // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
        state_dot.segment<3>(STATE_OMEGA_X) = inertia_inv_ad * (-this->skew<autodiff::dual2nd>(omega) * (inertia_ad * omega) + tau);

        return state_dot;
    }

} // namespace cddp