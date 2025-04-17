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

#include "cddp-cpp/dynamics_model/quaternion_attitude.hpp"
#include <cmath>                           // For std::sqrt
#include <autodiff/forward/dual.hpp>       // Include autodiff
#include <autodiff/forward/dual/eigen.hpp> // Include autodiff Eigen support
#include <autodiff/forward/real.hpp>       // For autodiff::val()
#include <autodiff/forward/real/eigen.hpp> // For autodiff::val()

namespace cddp
{

    QuaternionAttitude::QuaternionAttitude(double timestep, const Eigen::Matrix3d &inertia_matrix,
                                           std::string integration_type)
        : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
          inertia_(inertia_matrix),
          inertia_inv_(inertia_matrix.inverse()) {}

    Eigen::VectorXd QuaternionAttitude::getContinuousDynamics(const Eigen::VectorXd &state,
                                                              const Eigen::VectorXd &control) const
    {
        Eigen::VectorXd state_dot(STATE_DIM);

        // Extract states
        Eigen::Vector4d quat = state.segment<4>(STATE_QUAT_W);
        Eigen::Vector3d omega = state.segment<3>(STATE_OMEGA_X);

        // Extract control
        Eigen::Vector3d tau = control.segment<3>(CONTROL_TAU_X);

        // Normalize quaternion to prevent drift
        double quat_norm = quat.norm();
        if (quat_norm > 1e-9)
        { // Avoid division by zero
            quat /= quat_norm;
        }
        else
        {
            // Handle potential singularity, e.g., reinitialize to identity
            quat << 1.0, 0.0, 0.0, 0.0;
        }

        // Quaternion Kinematics: dq/dt = 0.5 * Omega(omega) * q
        state_dot.segment<4>(STATE_QUAT_W) = 0.5 * this->quatKinematicsMatrix<double>(omega) * quat;

        // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
        state_dot.segment<3>(STATE_OMEGA_X) = this->inertia_inv_ * (-this->skew<double>(omega) * (this->inertia_ * omega) + tau);

        return state_dot;
    }

    Eigen::MatrixXd QuaternionAttitude::getStateJacobian(const Eigen::VectorXd &state,
                                                         const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute state Jacobian
        VectorXdual2nd x = state;   // Cast state to autodiff type
        VectorXdual2nd u = control; // Cast control to autodiff type

        // Define lambda for dynamics w.r.t. state
        auto dynamics_wrt_x = [&](const VectorXdual2nd &x_ad) -> VectorXdual2nd
        {
            return this->getContinuousDynamicsAutodiff(x_ad, u);
        };

        // Compute Jacobian
        return autodiff::jacobian(dynamics_wrt_x, wrt(x), at(x));
    }

    Eigen::MatrixXd QuaternionAttitude::getControlJacobian(const Eigen::VectorXd &state,
                                                           const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute control Jacobian
        VectorXdual2nd x = state;   // Cast state to autodiff type
        VectorXdual2nd u = control; // Cast control to autodiff type

        // Define lambda for dynamics w.r.t. control
        auto dynamics_wrt_u = [&](const VectorXdual2nd &u_ad) -> VectorXdual2nd
        {
            return this->getContinuousDynamicsAutodiff(x, u_ad);
        };

        // Compute Jacobian
        return autodiff::jacobian(dynamics_wrt_u, wrt(u), at(u));
    }

    std::vector<Eigen::MatrixXd> QuaternionAttitude::getStateHessian(const Eigen::VectorXd &state,
                                                                     const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute state Hessian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            // Define lambda for the i-th component of dynamics w.r.t. state
            auto fi_x = [&, i](const VectorXdual2nd &x_ad) -> autodiff::dual2nd
            {
                return this->getContinuousDynamicsAutodiff(x_ad, u)(i);
            };

            // Compute Hessian for the i-th component
            hessians[i] = autodiff::hessian(fi_x, wrt(x), at(x));
        }

        return hessians;
    }

    std::vector<Eigen::MatrixXd> QuaternionAttitude::getControlHessian(const Eigen::VectorXd &state,
                                                                       const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute control Hessian
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            // Define lambda for the i-th component of dynamics w.r.t. control
            auto fi_u = [&, i](const VectorXdual2nd &u_ad) -> autodiff::dual2nd
            {
                return this->getContinuousDynamicsAutodiff(x, u_ad)(i);
            };

            // Compute Hessian for the i-th component
            hessians[i] = autodiff::hessian(fi_u, wrt(u), at(u));
        }

        return hessians;
    }

    std::vector<Eigen::MatrixXd> QuaternionAttitude::getCrossHessian(const Eigen::VectorXd &state,
                                                                     const Eigen::VectorXd &control) const
    {
        // Use autodiff to compute cross Hessian (Jacobian of gradient)
        VectorXdual2nd x = state;
        VectorXdual2nd u = control;

        std::vector<Eigen::MatrixXd> cross_hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            // Define lambda that computes the gradient of the i-th component w.r.t. state
            auto gradient_fi_x = [&, i](const VectorXdual2nd &u_ad) -> VectorXdual2nd
            {
                // Inner lambda: i-th component of dynamics w.r.t state (holding u_ad constant)
                auto fi_x = [&, u_ad, i](const VectorXdual2nd &x_ad) -> autodiff::dual2nd
                {
                    return this->getContinuousDynamicsAutodiff(x_ad, u_ad)(i);
                };
                // Return the gradient w.r.t. x
                return autodiff::gradient(fi_x, wrt(x), at(x));
            };

            // Compute the Jacobian of this gradient function w.r.t. control
            cross_hessians[i] = autodiff::jacobian(gradient_fi_x, wrt(u), at(u));
        }

        return cross_hessians;
    }

    // Autodiff version of the continuous dynamics
    VectorXdual2nd QuaternionAttitude::getContinuousDynamicsAutodiff(
        const VectorXdual2nd &state,
        const VectorXdual2nd &control) const
    {

        // Cast member variables to autodiff types
        autodiff::Matrix3dual2nd inertia_ad = this->inertia_.cast<autodiff::dual2nd>();
        autodiff::Matrix3dual2nd inertia_inv_ad = this->inertia_inv_.cast<autodiff::dual2nd>();

        // Extract states
        autodiff::Vector4dual2nd quat = state.segment<4>(STATE_QUAT_W);
        autodiff::Vector3dual2nd omega = state.segment<3>(STATE_OMEGA_X);

        // Extract control
        autodiff::Vector3dual2nd tau = control.segment<3>(CONTROL_TAU_X);

        // Initialize state derivative vector
        VectorXdual2nd state_dot(STATE_DIM);

        // Quaternion Kinematics: dq/dt = 0.5 * Omega(omega) * q
        state_dot.segment<4>(STATE_QUAT_W) = 0.5 * this->quatKinematicsMatrix<autodiff::dual2nd>(omega) * quat;

        // Euler's Rotational Dynamics: I * d(omega)/dt = -omega x (I * omega) + tau
        state_dot.segment<3>(STATE_OMEGA_X) = inertia_inv_ad * (-this->skew<autodiff::dual2nd>(omega) * (inertia_ad * omega) + tau);

        return state_dot;
    }

} // namespace cddp