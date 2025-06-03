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

#ifndef CDDP_EULER_ATTITUDE_HPP
#define CDDP_EULER_ATTITUDE_HPP

#include "cddp_core/dynamical_system.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>       
#include <autodiff/forward/dual/eigen.hpp> 

namespace cddp
{

     class EulerAttitude : public DynamicalSystem
     {
     public:
          // State indices (ZYX Euler angles: Yaw, Pitch, Roll)
          static constexpr int STATE_EULER_Z = 0; // Yaw (psi)
          static constexpr int STATE_EULER_Y = 1; // Pitch (theta)
          static constexpr int STATE_EULER_X = 2; // Roll (phi)
          static constexpr int STATE_OMEGA_X = 3; // Body angular velocity x
          static constexpr int STATE_OMEGA_Y = 4; // Body angular velocity y
          static constexpr int STATE_OMEGA_Z = 5; // Body angular velocity z
          static constexpr int STATE_DIM = 6;

          // Control indices (torques)
          static constexpr int CONTROL_TAU_X = 0;
          static constexpr int CONTROL_TAU_Y = 1;
          static constexpr int CONTROL_TAU_Z = 2;
          static constexpr int CONTROL_DIM = 3;

          /**
           * Constructor for the Euler Angle-based Attitude Dynamics model (ZYX convention)
           * @param timestep Time step for discretization
           * @param inertia_matrix Inertia matrix of the rigid body
           * @param integration_type Integration method ("euler" by default)
           */
          EulerAttitude(double timestep, const Eigen::Matrix3d &inertia_matrix,
                        std::string integration_type = "euler");

          /**
           * Computes the continuous-time dynamics of the Euler angle attitude model
           * State vector: [psi, theta, phi, omega_x, omega_y, omega_z]
           * Control vector: [tau_x, tau_y, tau_z] (applied torques)
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return State derivative vector
           */
          Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd &state,
                                                const Eigen::VectorXd &control, double time) const override;

          /**
           * Computes the discrete-time dynamics using the specified integration method
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return Next state vector
           */
          Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                              const Eigen::VectorXd &control, double time) const override
          {
               return DynamicalSystem::getDiscreteDynamics(state, control, time);
          }

          /**
           * Computes the Jacobian of the dynamics with respect to the state using Autodiff.
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return State Jacobian matrix
           */
          Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                           const Eigen::VectorXd &control, double time) const override;

          /**
           * Computes the Jacobian of the dynamics with respect to the control input using Autodiff.
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return Control Jacobian matrix
           */
          Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                             const Eigen::VectorXd &control, double time) const override;

          /**
           * Computes the Hessian of the dynamics with respect to the state using Autodiff.
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return State Hessian matrix
           */
          std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state,
                                                      const Eigen::VectorXd &control, double time) const override;

          /**
           * Computes the Hessian of the dynamics with respect to the control using Autodiff.
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return Control Hessian matrix
           */
          std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd &state,
                                                         const Eigen::VectorXd &control, double time) const override;

          /**
           * Computes the cross Hessian of the dynamics w.r.t. state and control using Autodiff.
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return Cross Hessian matrix
           */
          std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd &state,
                                                       const Eigen::VectorXd &control, double time) const override;

          /**
           * Computes the continuous-time dynamics using Autodiff types.
           * Used internally for calculating Jacobians and Hessians.
           * @param state Current state vector
           * @param control Current control input
           * @param time Current time
           * @return State derivative vector
           */
          VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd &state,
                                                       const VectorXdual2nd &control, double time) const override;

     private:
          Eigen::Matrix3d inertia_;     // Inertia matrix
          Eigen::Matrix3d inertia_inv_; // Inverse of the inertia matrix

          // Helper function for skew-symmetric matrix
          template <typename T>
          Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1> &v) const
          {
               Eigen::Matrix<T, 3, 3> S;
               S << T(0), -v(2), v(1),
                   v(2), T(0), -v(0),
                   -v(1), v(0), T(0);
               return S;
          }

          // Helper function for ZYX Euler angle kinematics matrix E(angles)
          template <typename T>
          Eigen::Matrix<T, 3, 3> eulerKinematicsMatrix(const Eigen::Matrix<T, 3, 1> &angles) const
          {
               T psi = angles(0);   // Yaw
               T theta = angles(1); // Pitch
               T phi = angles(2);   // Roll

               T c_phi = cos(phi), s_phi = sin(phi);
               T c_theta = cos(theta), s_theta = sin(theta);
               T tan_theta = tan(theta);

               T cos_theta_safe = c_theta;
               if (abs(autodiff::val(c_theta)) < 1e-9)
               { 
                    cos_theta_safe = T(1e-9 * (autodiff::val(c_theta) >= 0 ? 1 : -1)); 
               }

               Eigen::Matrix<T, 3, 3> E;
               E << T(0), s_phi / cos_theta_safe, c_phi / cos_theta_safe, // dpsi/dt
                   T(0), c_phi, -s_phi,                                   // dtheta/dt
                   T(1), s_phi * tan_theta, c_phi * tan_theta;            // dphi/dt

               return E;
          }
     };

} // namespace cddp

#endif // CDDP_EULER_ATTITUDE_HPP