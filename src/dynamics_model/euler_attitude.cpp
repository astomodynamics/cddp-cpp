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
        const Eigen::VectorXd &state, const Eigen::VectorXd &control, double time) const
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
        const Eigen::VectorXd &state, const Eigen::VectorXd &control, double time) const
    {
        VectorXdual2nd x_ad = state;
        VectorXdual2nd u_ad = control;

        auto dynamics_eval = [&](const VectorXdual2nd &x_in) -> VectorXdual2nd
        {
            return this->getContinuousDynamicsAutodiff(x_in, u_ad, time);
        };
        return autodiff::jacobian(dynamics_eval, wrt(x_ad), at(x_ad));
    }

    Eigen::MatrixXd EulerAttitude::getControlJacobian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control, double time) const
    {
        VectorXdual2nd x_ad = state;
        VectorXdual2nd u_ad = control;

        auto dynamics_eval = [&](const VectorXdual2nd &u_in) -> VectorXdual2nd
        {
            return this->getContinuousDynamicsAutodiff(x_ad, u_in, time);
        };
        return autodiff::jacobian(dynamics_eval, wrt(u_ad), at(u_ad));
    }

    std::vector<Eigen::MatrixXd> EulerAttitude::getStateHessian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control, double time) const
    {
        VectorXdual2nd x_ad = state;
        VectorXdual2nd u_ad = control;

        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
        for (int i = 0; i < STATE_DIM; ++i)
        {
            auto dynamics_component_i = [&](const VectorXdual2nd &x_in) -> autodiff::dual2nd
            {
                return this->getContinuousDynamicsAutodiff(x_in, u_ad, time)(i);
            };
            hessians[i] = autodiff::hessian(dynamics_component_i, wrt(x_ad), at(x_ad));
        }
        return hessians;
    }

    std::vector<Eigen::MatrixXd> EulerAttitude::getControlHessian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control, double time) const
    {
        VectorXdual2nd x_ad = state;
        VectorXdual2nd u_ad = control;

        std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
        for (int i = 0; i < STATE_DIM; ++i)
        {
            auto dynamics_component_i = [&](const VectorXdual2nd &u_in) -> autodiff::dual2nd
            {
                return this->getContinuousDynamicsAutodiff(x_ad, u_in, time)(i);
            };
            hessians[i] = autodiff::hessian(dynamics_component_i, wrt(u_ad), at(u_ad));
        }
        return hessians;
    }

    std::vector<Eigen::MatrixXd> EulerAttitude::getCrossHessian(
        const Eigen::VectorXd &state, const Eigen::VectorXd &control, double time) const
    {
        VectorXdual2nd x_target = state; // Point at which to evaluate the gradient w.r.t. x
        VectorXdual2nd u_wrt = control; // Variable of differentiation for the outer Jacobian

        std::vector<Eigen::MatrixXd> cross_hessians(STATE_DIM);

        for (int i = 0; i < STATE_DIM; ++i)
        {
            // Define a lambda that computes the gradient of the i-th component of dynamics w.r.t. state (x)
            // This gradient will be a function of control (u_inner)
            auto gradient_fi_x_of_u = [&](const VectorXdual2nd &u_inner) -> VectorXdual2nd
            {
                // Inner lambda: f_i(x, u_inner) - evaluates the i-th component of dynamics
                auto fi_of_x = [&](const VectorXdual2nd &x_inner) -> autodiff::dual2nd
                {
                    return this->getContinuousDynamicsAutodiff(x_inner, u_inner, time)(i);
                };
                // Compute gradient of fi_of_x w.r.t x, evaluated at x_target.
                // For autodiff::gradient, wrt(variable_to_diff) and at(point_of_evaluation)
                return autodiff::gradient(fi_of_x, wrt(x_target), at(x_target)); 
            };
            
            // Compute the Jacobian of gradient_fi_x_of_u w.r.t. u_wrt
            cross_hessians[i] = autodiff::jacobian(gradient_fi_x_of_u, wrt(u_wrt), at(u_wrt));
        }
        return cross_hessians;
    }

    // Autodiff version of the continuous dynamics
    VectorXdual2nd EulerAttitude::getContinuousDynamicsAutodiff(
        const VectorXdual2nd &state,
        const VectorXdual2nd &control,
        double time) const
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