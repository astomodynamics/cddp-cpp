/*
 Copyright 2024

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

#ifndef CDDP_SPACECRAFT_ROE_HPP
#define CDDP_SPACECRAFT_ROE_HPP

#include "cddp_core/dynamical_system.hpp"
#include <cmath>

namespace cddp
{

/**
 * @brief Spacecraft Relative Orbital Elements (ROE) Dynamics
 * 
 * State vector (6 dimensions):
 *   x = [ da, dlambda, dex, dey, dix, diy ]
 * where:
 *   - da       : Relative semi-major axis
 *   - dlambda  : Relative mean longitude (or some variant of relative angle)
 *   - dex      : Relative x-component of eccentricity vector
 *   - dey      : Relative y-component of eccentricity vector
 *   - dix      : Relative x-component of inclination vector
 *   - diy      : Relative y-component of inclination vector
 * 
 * Control vector (3 dimensions):
 *   u = [ ur, ut, un ]
 * where:
 *   - ur : radial acceleration [m/s^2]
 *   - ut : tangential acceleration [m/s^2]
 *   - un : normal acceleration [m/s^2]
 */
class SpacecraftROE : public DynamicalSystem
{
public:
    /**
     * @brief Constructor
     * @param timestep           Integration timestep [s]
     * @param integration_type   Integration method (e.g. "rk4")
     * @param mass               Spacecraft mass [kg]
     * @param mu                 Gravitational parameter [m^3/s^2]
     * @param n_ref              Mean motion of the reference orbit [rad/s]
     */
    SpacecraftROE(
        double timestep,
        const std::string& integration_type,
        double mass,
        double mu,
        double n_ref);

    // State indices
    static constexpr int STATE_DA       = 0;  ///< Relative semi-major axis
    static constexpr int STATE_DLAMBDA  = 1;  ///< Relative mean longitude
    static constexpr int STATE_DEX      = 2;  ///< Relative eccentricity x-component
    static constexpr int STATE_DEY      = 3;  ///< Relative eccentricity y-component
    static constexpr int STATE_DIX      = 4;  ///< Relative inclination x-component
    static constexpr int STATE_DIY      = 5;  ///< Relative inclination y-component
    static constexpr int STATE_DIM      = 6;  ///< State dimension

    // Control indices
    static constexpr int CONTROL_UR = 0;      ///< Radial acceleration
    static constexpr int CONTROL_UT = 1;      ///< Tangential acceleration
    static constexpr int CONTROL_UN = 2;      ///< Normal acceleration
    static constexpr int CONTROL_DIM = 3;     ///< Control dimension

    /**
     * @brief Compute continuous-time dynamics in ROE coordinates
     * @param state   Current ROE state vector
     * @param control Current control vector [ur, ut, un]
     * @return        Derivative of the ROE state vector
     */
    Eigen::VectorXd getContinuousDynamics(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control) const override;

    /**
     * @brief Compute state Jacobian matrix via numerical finite difference
     * @param state   Current ROE state vector
     * @param control Current control vector
     * @return        State Jacobian matrix (d f / d state)
     */
    Eigen::MatrixXd getStateJacobian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control) const override;

    /**
     * @brief Compute control Jacobian matrix via numerical finite difference
     * @param state   Current ROE state vector
     * @param control Current control vector
     * @return        Control Jacobian matrix (d f / d control)
     */
    Eigen::MatrixXd getControlJacobian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control) const override;

    /**
     * @brief Compute state Hessian matrix
     * @param state   Current state vector
     * @param control Current control vector
     * @return        State Hessian matrix
     */
    Eigen::MatrixXd getStateHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control) const override;

    /**
     * @brief Compute control Hessian matrix
     * @param state   Current state vector
     * @param control Current control vector
     * @return        Control Hessian matrix
     */
    Eigen::MatrixXd getControlHessian(
        const Eigen::VectorXd& state,
        const Eigen::VectorXd& control) const override;

private:
    double mass_;   ///< Spacecraft mass [kg]
    double mu_;     ///< Gravitational parameter [m^3/s^2]
    double n_ref_;  ///< Mean motion of reference orbit [rad/s]
};

} // namespace cddp

#endif // CDDP_SPACECRAFT_ROE_HPP
