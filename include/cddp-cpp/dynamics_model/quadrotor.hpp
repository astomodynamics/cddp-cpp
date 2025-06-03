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
     * Constructor for the Quaternion-based Quadrotor model
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
     * State vector: [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
     * Control vector: [f1, f2, f3, f4] (motor forces)
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the discrete-time dynamics using the specified integration method
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Next state vector
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control, time);
    }

    /**
     * Computes the Jacobian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return State Jacobian matrix (A matrix)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Control Jacobian matrix (B matrix)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Vector of control Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the cross Hessian of the dynamics with respect to both state and control
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Vector of cross Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control, double time) const override;

    /**
     * Computes the continuous-time dynamics of the quadrotor model using autodiff
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return State derivative vector
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd& state, 
                                                const VectorXdual2nd& control, double time) const override;

private:
    double mass_;              // Mass of the quadrotor
    Eigen::Matrix3d inertia_;  // Inertia matrix
    double arm_length_;        // Length of the quadrotor arm
    double gravity_{9.81};     // Gravitational acceleration

    // State indices for quaternion-based representation
    static constexpr int STATE_X = 0;       // x position
    static constexpr int STATE_Y = 1;       // y position
    static constexpr int STATE_Z = 2;       // z position
    static constexpr int STATE_QW = 3;      // quaternion w (scalar part)
    static constexpr int STATE_QX = 4;      // quaternion x
    static constexpr int STATE_QY = 5;      // quaternion y
    static constexpr int STATE_QZ = 6;      // quaternion z
    static constexpr int STATE_VX = 7;      // x velocity
    static constexpr int STATE_VY = 8;      // y velocity
    static constexpr int STATE_VZ = 9;      // z velocity
    static constexpr int STATE_OMEGA_X = 10; // angular velocity about x
    static constexpr int STATE_OMEGA_Y = 11; // angular velocity about y
    static constexpr int STATE_OMEGA_Z = 12; // angular velocity about z
    static constexpr int STATE_DIM = 13;    // state dimension

    // Control indices (motor forces)
    static constexpr int CONTROL_F1 = 0;    // front motor
    static constexpr int CONTROL_F2 = 1;    // right motor
    static constexpr int CONTROL_F3 = 2;    // back motor
    static constexpr int CONTROL_F4 = 3;    // left motor
    static constexpr int CONTROL_DIM = 4;   // control dimension

    /**
     * Helper function to compute rotation matrix from a quaternion
     * @param qw Scalar part of quaternion
     * @param qx x component
     * @param qy y component
     * @param qz z component
     * @return 3x3 rotation matrix
     */
    Eigen::Matrix3d getRotationMatrix(double qw, double qx, double qy, double qz) const;

    /**
     * Helper function to compute rotation matrix from a quaternion using autodiff
     * @param qw Scalar part of quaternion
     * @param qx x component
     * @param qy y component
     * @param qz z component
     * @return 3x3 rotation matrix
     */
    Eigen::Matrix<autodiff::dual2nd, 3, 3> getRotationMatrixAutodiff(
        const autodiff::dual2nd& qw, const autodiff::dual2nd& qx,
        const autodiff::dual2nd& qy, const autodiff::dual2nd& qz) const;
};

} // namespace cddp

#endif // CDDP_QUADROTOR_HPP
