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

#ifndef CDDP_MRP_ATTITUDE_HPP
#define CDDP_MRP_ATTITUDE_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class MrpAttitude : public DynamicalSystem {
public:
    // State indices
    static constexpr int STATE_MRP_X = 0;
    static constexpr int STATE_MRP_Y = 1;
    static constexpr int STATE_MRP_Z = 2;
    static constexpr int STATE_OMEGA_X = 3;
    static constexpr int STATE_OMEGA_Y = 4;
    static constexpr int STATE_OMEGA_Z = 5;
    static constexpr int STATE_DIM = 6;

    // Control indices (torques)
    static constexpr int CONTROL_TAU_X = 0;
    static constexpr int CONTROL_TAU_Y = 1;
    static constexpr int CONTROL_TAU_Z = 2;
    static constexpr int CONTROL_DIM = 3;

    /**
     * Constructor for the MRP-based Attitude Dynamics model
     * @param timestep Time step for discretization
     * @param inertia_matrix Inertia matrix of the rigid body
     * @param integration_type Integration method ("euler" by default)
     */
    MrpAttitude(double timestep, const Eigen::Matrix3d& inertia_matrix,
                std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the MRP attitude model
     * State vector: [mrp_x, mrp_y, mrp_z, omega_x, omega_y, omega_z]
     * Control vector: [tau_x, tau_y, tau_z] (applied torques)
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
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of control Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control) const override;

    /**
     * Computes the cross Hessian of the dynamics with respect to both state and control
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of cross Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control) const override;

    /**
     * Computes the continuous-time dynamics of the MRP attitude model using autodiff
     * @param state Current state vector
     * @param control Current control input
     * @return State derivative vector
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd& state,
                                                const VectorXdual2nd& control) const override;

private:
    Eigen::Matrix3d inertia_;       // Inertia matrix
    Eigen::Matrix3d inertia_inv_;   // Inverse of the inertia matrix

    // Helper function for skew-symmetric matrix
    template <typename T>
    Eigen::Matrix<T, 3, 3> skew(const Eigen::Matrix<T, 3, 1>& v) const {
        Eigen::Matrix<T, 3, 3> S;
        S << T(0), -v(2), v(1),
             v(2), T(0), -v(0),
             -v(1), v(0), T(0);
        return S;
    }

    // Helper function for MRP kinematics matrix B(mrp)
    template <typename T>
    Eigen::Matrix<T, 3, 3> mrpKinematicsMatrix(const Eigen::Matrix<T, 3, 1>& mrp) const {
        T mrp_norm_sq = mrp.squaredNorm();
        Eigen::Matrix<T, 3, 3> I = Eigen::Matrix<T, 3, 3>::Identity();
        return (T(1.0) - mrp_norm_sq) * I + T(2.0) * skew(mrp) + T(2.0) * mrp * mrp.transpose();
    }

};

} // namespace cddp

#endif // CDDP_MRP_ATTITUDE_HPP 