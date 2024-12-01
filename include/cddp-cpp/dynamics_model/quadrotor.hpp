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

#ifndef CDDP_QUADROTOR_HPP
#define CDDP_QUADROTOR_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class Quadrotor : public DynamicalSystem {
public:
    /**
     * Constructor for the Quadrotor model
     * @param timestep Time step for discretization
     * @param mass Mass of the quadrotor
     * @param inertia_matrix Inertia matrix of the quadrotor
     * @param arm_length Length of the quadrotor arm
     * @param integration_type Integration method ("euler" by default)
     */
    Quadrotor(double timestep, double mass, const Eigen::Matrix3d& inertia_matrix,
              double arm_length, std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the quadrotor model
     * State vector: [x, y, z, phi, theta, psi, vx, vy, vz, omega_x, omega_y, omega_z]
     * Control vector: [f1, f2, f3, f4] (motor forces)
     * @param state Current state vector
     * @param control Current control input
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control) const override;

    /**
     * Computes the discrete-time dynamics using the specified integration method
     * @param state Current state vector
     * @param control Current control input
     * @return Next state vector
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control);
    }

    /**
     * Computes the Jacobian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @return State Jacobian matrix (A matrix)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector
     * @param control Current control input
     * @return Control Jacobian matrix (B matrix)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @return State Hessian matrix
     */
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @return Control Hessian matrix
     */
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

private:
    double mass_;              // Mass of the quadrotor
    Eigen::Matrix3d inertia_; // Inertia matrix
    double arm_length_;       // Length of the quadrotor arm
    double gravity_{9.81};    // Gravitational acceleration

    // State indices
    static constexpr int STATE_X = 0;       // x position
    static constexpr int STATE_Y = 1;       // y position
    static constexpr int STATE_Z = 2;       // z position
    static constexpr int STATE_PHI = 3;     // roll angle
    static constexpr int STATE_THETA = 4;   // pitch angle
    static constexpr int STATE_PSI = 5;     // yaw angle
    static constexpr int STATE_VX = 6;      // x velocity
    static constexpr int STATE_VY = 7;      // y velocity
    static constexpr int STATE_VZ = 8;      // z velocity
    static constexpr int STATE_OMEGA_X = 9; // angular velocity around x
    static constexpr int STATE_OMEGA_Y = 10;// angular velocity around y
    static constexpr int STATE_OMEGA_Z = 11;// angular velocity around z
    static constexpr int STATE_DIM = 12;    // state dimension

    // Control indices (motor forces)
    static constexpr int CONTROL_F1 = 0;    // front motor
    static constexpr int CONTROL_F2 = 1;    // right motor
    static constexpr int CONTROL_F3 = 2;    // back motor
    static constexpr int CONTROL_F4 = 3;    // left motor
    static constexpr int CONTROL_DIM = 4;   // control dimension

    /**
     * Helper function to compute rotation matrix from Euler angles
     * @param phi Roll angle
     * @param theta Pitch angle
     * @param psi Yaw angle
     * @return 3x3 rotation matrix
     */
    Eigen::Matrix3d getRotationMatrix(double phi, double theta, double psi) const;
};

} // namespace cddp

#endif // CDDP_QUADROTOR_HPP