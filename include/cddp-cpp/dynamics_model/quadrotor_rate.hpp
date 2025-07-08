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

#ifndef CDDP_QUADROTOR_RATE_HPP
#define CDDP_QUADROTOR_RATE_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * QuadrotorRate: Rate-based quadrotor dynamics model for trajectory optimization
 * 
 * State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz] (10 states)
 * Control: [thrust, wx, wy, wz] (4 inputs)
 * 
 * This model uses thrust and angular rates as control inputs, providing a
 * computationally efficient yet accurate representation of quadrotor dynamics.
 * Suitable for real-time applications and hierarchical control architectures.
 */
class QuadrotorRate : public DynamicalSystem {
public:
    /**
     * Constructor for the Rate-based Quadrotor model
     * @param timestep Time step for discretization
     * @param mass Mass of the quadrotor
     * @param max_thrust Maximum thrust capability (in Newtons)
     * @param max_rate Maximum angular rate (in rad/s)
     * @param integration_type Integration method ("euler", "rk4", etc.)
     */
    QuadrotorRate(double timestep, double mass, double max_thrust, double max_rate,
                  std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the rate-based quadrotor model
     * @param state Current state vector [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
     * @param control Current control input [thrust, wx, wy, wz]
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

    // Getters for model parameters
    double getMass() const { return mass_; }
    double getMaxThrust() const { return max_thrust_; }
    double getMaxRate() const { return max_rate_; }

private:
    double mass_;              // Mass of the quadrotor
    double max_thrust_;        // Maximum thrust capability
    double max_rate_;          // Maximum angular rate
    double gravity_{9.81};     // Gravitational acceleration

    // State indices for rate-based representation
    static constexpr int STATE_PX = 0;      // x position
    static constexpr int STATE_PY = 1;      // y position
    static constexpr int STATE_PZ = 2;      // z position
    static constexpr int STATE_VX = 3;      // x velocity
    static constexpr int STATE_VY = 4;      // y velocity
    static constexpr int STATE_VZ = 5;      // z velocity
    static constexpr int STATE_QW = 6;      // quaternion w (scalar part)
    static constexpr int STATE_QX = 7;      // quaternion x
    static constexpr int STATE_QY = 8;      // quaternion y
    static constexpr int STATE_QZ = 9;      // quaternion z
    static constexpr int STATE_DIM = 10;    // state dimension

    // Control indices
    static constexpr int CONTROL_THRUST = 0;  // total thrust
    static constexpr int CONTROL_WX = 1;      // angular rate x
    static constexpr int CONTROL_WY = 2;      // angular rate y
    static constexpr int CONTROL_WZ = 3;      // angular rate z
    static constexpr int CONTROL_DIM = 4;     // control dimension

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

    /**
     * Helper function to create skew-symmetric matrix for quaternion dynamics
     * @param wx angular rate x
     * @param wy angular rate y
     * @param wz angular rate z
     * @return 4x4 skew-symmetric matrix Omega
     */
    Eigen::Matrix4d getOmegaMatrix(double wx, double wy, double wz) const;

    /**
     * Helper function to create skew-symmetric matrix for quaternion dynamics using autodiff
     * @param wx angular rate x
     * @param wy angular rate y
     * @param wz angular rate z
     * @return 4x4 skew-symmetric matrix Omega
     */
    Eigen::Matrix<autodiff::dual2nd, 4, 4> getOmegaMatrixAutodiff(
        const autodiff::dual2nd& wx, const autodiff::dual2nd& wy, 
        const autodiff::dual2nd& wz) const;
};

} // namespace cddp

#endif // CDDP_QUADROTOR_RATE_HPP