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

#ifndef CDDP_MANIPULATOR_HPP
#define CDDP_MANIPULATOR_HPP

#include "cddp_core/dynamical_system.hpp"
#include <Eigen/Dense>

namespace cddp {
/**
 * @brief Manipulator model implementation
 * 
 * This class implements a simplified PUMA manipulator model with state vector
 * [q1, q2, q3, dq1, dq2, dq3] where q1, q2, q3 are joint angles and dq1, dq2, dq3
 * are joint velocities. The control input is the joint torques [tau1, tau2, tau3].
 */
class Manipulator : public DynamicalSystem {
public:
    /**
     * Constructor for simplified PUMA manipulator model
     * @param timestep Time step for discretization
     * @param integration_type Integration method ("euler" or "rk4")
     */
    Manipulator(double timestep, std::string integration_type = "rk4");

    /**
     * Computes the continuous-time dynamics of the manipulator
     * State vector: [q1, q2, q3, dq1, dq2, dq3]
     * Control vector: [tau1, tau2, tau3]
     * @param state Current state vector
     * @param control Current control input (joint torques)
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                         const Eigen::VectorXd& control) const override;

    /**
     * Computes the discrete-time dynamics
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

    /**
     * Computes the forward kinematics for the end-effector
     * @param state Current state vector (only joint angles are used)
     * @return 4x4 homogeneous transformation matrix of the end-effector
     */
    Eigen::Matrix4d getForwardKinematics(const Eigen::VectorXd& state) const;

    /**
     * Gets the end-effector position in world coordinates
     * @param state Current state vector
     * @return 3D position vector of the end-effector
     */
    Eigen::Vector3d getEndEffectorPosition(const Eigen::VectorXd& state) const;

    /**
     * Helper function to compute transformation matrices
     * @param q1,q2,q3 Joint angles
     * @return Vector of 4x4 transformation matrices [T01, T12, T23, T34]
     */
    std::vector<Eigen::Matrix4d> getTransformationMatrices(
        double q1, double q2, double q3) const;

    /**
     * Get link length by identifier
     * @param link_id 'a' for la_, 'b' for lb_, 'c' for lc_
     * @return Link length value
     */
    double getLinkLength(char link_id) const {
        switch(link_id) {
            case 'a': return la_;
            case 'b': return lb_;
            case 'c': return lc_;
            default: return 0.0;
        }
    }

private:
    // Link lengths (match MATLAB example)
    const double la_{1.0};    // Link a length
    const double lb_{0.2};    // Link b length
    const double lc_{1.0};    // Link c length
    const double gravity_{9.81};

    // DH parameters for PUMA-like configuration
    const double alpha1_{-M_PI/2};  // rotation about x-axis between z0 and z1
    const double alpha2_{0.0};   // rotation about x-axis between z1 and z2
    const double alpha3_{0.0};      // rotation about x-axis between z2 and z3

    // State dimensions
    static constexpr int NUM_JOINTS = 3;
    static constexpr int STATE_DIM = 2 * NUM_JOINTS;  // positions and velocities
    static constexpr int CONTROL_DIM = NUM_JOINTS;    // joint torques

    /**
     * Computes the rotation matrix about the x-axis
     * @param alpha Angle of rotation
     * @return 4x4 rotation matrix
     */
    Eigen::Matrix4d rotX(double alpha) const;

    /**
     * Computes the rotation matrix about the y-axis
     * @param beta Angle of rotation
     * @return 4x4 rotation matrix
     */
    Eigen::Matrix4d rotY(double beta) const;

    /**
     * Computes the rotation matrix about the z-axis
     * @param theta Angle of rotation
     * @return 4x4 rotation matrix
     */
    Eigen::Matrix4d rotZ(double theta) const;

    /**
     * Computes the mass matrix M(q) of the manipulator
     * @param q Joint positions
     * @return Mass matrix
     */
    Eigen::MatrixXd getMassMatrix(const Eigen::VectorXd& q) const;

    /**
     * Computes the gravity compensation terms
     * @param q Joint positions
     * @return Gravity torque vector
     */
    Eigen::VectorXd getGravityVector(const Eigen::VectorXd& q) const;
};

} // namespace cddp

#endif // CDDP_MANIPULATOR_HPP