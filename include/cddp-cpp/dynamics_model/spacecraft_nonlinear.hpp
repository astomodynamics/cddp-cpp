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

#ifndef CDDP_SPACECRAFT_NONLINEAR_HPP
#define CDDP_SPACECRAFT_NONLINEAR_HPP

#include "cddp_core/dynamical_system.hpp"
#include <cmath>

namespace cddp {

/**
 * @brief Spacecraft Nonlinear Equations of Relative Motion
 * 
 * This class implements the nonlinear equations of relative motion for a spacecraft
 * based on the model from:
 * Alfriend et al., "Spacecraft Formation Flying", Chapter 4 - Nonlinear Models of Relative Dynamics
 * 
 * State vector (10 dimensions):
 * x = [px, py, pz, vx, vy, vz, r0, theta, dr0, dtheta]
 * where:
 * - px, py, pz: relative position components
 * - vx, vy, vz: relative velocity components
 * - r0: reference orbit radius
 * - theta: reference orbit angle
 * - dr0: rate of change of r0
 * - dtheta: angular velocity
 * 
 * Control vector (3 dimensions):
 * u = [ux, uy, uz]
 * where:
 * - ux, uy, uz: control accelerations in each direction
 */
class SpacecraftNonlinear : public DynamicalSystem {
public:
    /**
     * @brief Constructor
     * @param timestep Integration timestep
     * @param integration_type Integration method
     * @param mass Spacecraft mass [kg]
     * @param r_scale Position scaling factor
     * @param v_scale Velocity scaling factor
     * @param mu Gravitational parameter [m^3/s^2]
     */
    SpacecraftNonlinear(
        double timestep,
        std::string integration_type = "rk4",
        double mass = 1.0,
        double r_scale = 1.0,
        double v_scale = 1.0,
        double mu = 1.0
    );

    // State indices
    static constexpr int STATE_PX = 0;       ///< Relative x-position
    static constexpr int STATE_PY = 1;       ///< Relative y-position
    static constexpr int STATE_PZ = 2;       ///< Relative z-position
    static constexpr int STATE_VX = 3;       ///< Relative x-velocity
    static constexpr int STATE_VY = 4;       ///< Relative y-velocity
    static constexpr int STATE_VZ = 5;       ///< Relative z-velocity
    static constexpr int STATE_R0 = 6;       ///< Reference orbit radius
    static constexpr int STATE_THETA = 7;    ///< Reference orbit angle
    static constexpr int STATE_DR0 = 8;      ///< Rate of change of r0
    static constexpr int STATE_DTHETA = 9;   ///< Angular velocity
    static constexpr int STATE_DIM = 10;     ///< State dimension

    // Control indices
    static constexpr int CONTROL_UX = 0;     ///< X-axis control acceleration
    static constexpr int CONTROL_UY = 1;     ///< Y-axis control acceleration
    static constexpr int CONTROL_UZ = 2;     ///< Z-axis control acceleration
    static constexpr int CONTROL_DIM = 3;    ///< Control dimension

    /**
     * @brief Compute continuous-time dynamics
     * @param state Current state vector
     * @param control Current control vector
     * @param time Current time
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Compute state Jacobian matrix
     * @param state Current state vector
     * @param control Current control vector
     * @param time Current time
     * @return State Jacobian matrix
     */
    Eigen::MatrixXd getStateJacobian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Compute control Jacobian matrix
     * @param state Current state vector
     * @param control Current control vector
     * @param time Current time
     * @return Control Jacobian matrix
     */
    Eigen::MatrixXd getControlJacobian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Compute state Hessian tensor
     * @param state Current state vector
     * @param control Current control vector
     * @param time Current time
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getStateHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Compute control Hessian tensor
     * @param state Current state vector
     * @param control Current control vector
     * @param time Current time
     * @return Vector of control Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getControlHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control, double time) const override;

private:
    double mass_;      ///< Spacecraft mass [kg]
    double r_scale_;   ///< Position scaling factor
    double v_scale_;   ///< Velocity scaling factor
    double mu_;        ///< Gravitational parameter [m^3/s^2]
};

} // namespace cddp

#endif // CDDP_SPACECRAFT_NONLINEAR_HPP