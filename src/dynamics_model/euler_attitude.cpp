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

#include "cddp-cpp/dynamics_model/euler_attitude.hpp"
#include <cmath>                           // For trigonometric functions
#include <autodiff/forward/dual.hpp>       // Include autodiff
#include <autodiff/forward/dual/eigen.hpp> // Include autodiff Eigen support
#include <autodiff/forward/real.hpp>       // For autodiff::val()
#include <autodiff/forward/real/eigen.hpp> // For autodiff::val()

namespace cddp
{

    EulerAttitude::EulerAttitude(double timestep, const Eigen::Matrix3d &inertia_matrix,
                                 std::string integration_type)
        : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
          inertia_(inertia_matrix),
          inertia_inv_(inertia_matrix.inverse()) {}

    Eigen::VectorXd EulerAttitude::getContinuousDynamics(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
    {
        Eigen::VectorXd state_dot(STATE_DIM);

        // Extract states
        Eigen::Vector3d euler_angles = state.segment<3>(STATE_EULER_Z); // [psi, theta, phi]
        Eigen::Vector3d omega = state.segment<3>(STATE_OMEGA_X);

        // Extract control
        Eigen::Vector3d tau = control.segment<3>(CONTROL_TAU_X);

        // Euler Angle Kinematics: d(angles)/dt = E(angles) * omega
        state_dot.segment<3>(STATE_EULER_Z) = this->eulerKinematicsMatrix<double>(euler_angles) * omega;

        // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
        state_dot.segment<3>(STATE_OMEGA_X) = this->inertia_inv_ * (-this->skew<double>(omega) * (this->inertia_ * omega) + tau);

        return state_dot;
    }

    Eigen::MatrixXd EulerAttitude::getStateJacobian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute state Jacobian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        auto dynamics_wrt_x = [&](const VectorXdual2nd &x_ad) -> VectorXdual2nd
        {
            return this->getContinuousDynamicsAutodiff(x_ad, u);
        };

        return autodiff::jacobian(dynamics_wrt_x, wrt(x), at(x));
    }

    Eigen::MatrixXd EulerAttitude::getControlJacobian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute control Jacobian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        auto dynamics_wrt_u = [&](const VectorXdual2nd &u_ad) -> VectorXdual2nd
        {
            return this->getContinuousDynamicsAutodiff(x, u_ad);
        };

        return autodiff::jacobian(dynamics_wrt_u, wrt(u), at(u));
    }

    std::vector<Eigen::MatrixXd> EulerAttitude::getStateHessian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute state Hessian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            auto fi_x = [&, i](const VectorXdual2nd &x_ad) -> autodiff::dual2nd
            {
                return this->getContinuousDynamicsAutodiff(x_ad, u)(i);
            };
            hessians[i] = autodiff::hessian(fi_x, wrt(x), at(x));
        }

        return hessians;
    }

    std::vector<Eigen::MatrixXd> EulerAttitude::getControlHessian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute control Hessian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            auto fi_u = [&, i](const VectorXdual2nd &u_ad) -> autodiff::dual2nd
            {
                return this->getContinuousDynamicsAutodiff(x, u_ad)(i);
            };
            hessians[i] = autodiff::hessian(fi_u, wrt(u), at(u));
        }

        return hessians;
    }

    std::vector<Eigen::MatrixXd> EulerAttitude::getCrossHessian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute cross Hessian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        std::vector<Eigen::MatrixXd> cross_hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            auto gradient_fi_x = [&, i](const VectorXdual2nd &u_ad) -> VectorXdual2nd
            {
                auto fi_x = [&, u_ad, i](const VectorXdual2nd &x_ad) -> autodiff::dual2nd
                {
                    return this->getContinuousDynamicsAutodiff(x_ad, u_ad)(i);
                };
                return autodiff::gradient(fi_x, wrt(x), at(x));
            };
            cross_hessians[i] = autodiff::jacobian(gradient_fi_x, wrt(u), at(u));
        }

        return cross_hessians;
    }

    // Autodiff version of the continuous dynamics
    VectorXdual2nd EulerAttitude::getContinuousDynamicsAutodiff(
        const VectorXdual2nd &state,
        const VectorXdual2nd &control) const
    {

        // Cast member variables to autodiff types
        autodiff::Matrix3dual2nd inertia_ad = this->inertia_.cast<autodiff::dual2nd>();
        autodiff::Matrix3dual2nd inertia_inv_ad = this->inertia_inv_.cast<autodiff::dual2nd>();

        // Extract states
        autodiff::Vector3dual2nd euler_angles = state.segment<3>(STATE_EULER_Z);
        autodiff::Vector3dual2nd omega = state.segment<3>(STATE_OMEGA_X);

        // Extract control
        autodiff::Vector3dual2nd tau = control.segment<3>(CONTROL_TAU_X);

        // Initialize state derivative vector
        VectorXdual2nd state_dot(STATE_DIM);

        // Euler Angle Kinematics: d(angles)/dt = E(angles) * omega
        state_dot.segment<3>(STATE_EULER_Z) = this->eulerKinematicsMatrix<autodiff::dual2nd>(euler_angles) * omega;

        // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
        state_dot.segment<3>(STATE_OMEGA_X) = inertia_inv_ad * (-this->skew<autodiff::dual2nd>(omega) * (inertia_ad * omega) + tau);

        return state_dot;
    }

} // namespace cddp