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
     *   x = [ da, dlambda, dex, dey, dix, diy]
     * where:
     *   - da       : Relative semi-major axis
     *   - dlambda  : Relative mean longitude (or some variant of relative angle)
     *   - dex      : Relative x-component of eccentricity vector
     *   - dey      : Relative y-component of eccentricity vector
     *   - dix      : Relative x-component of inclination vector
     *   - diy      : Relative y-component of inclination vector
     *   - M        : Mean anomaly [rad]
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
         * @param a                  Semi-major axis of the reference orbit [m]
         * @param u0                Initial mean argument of latitude [rad]
         */
        SpacecraftROE(
            double timestep,
            const std::string &integration_type,
            double a,
            double u0 = 0.0,
            double mass_kg = 1.0);

        // State indices
        static constexpr int STATE_DA = 0;      ///< Relative semi-major axis
        static constexpr int STATE_DLAMBDA = 1; ///< Relative mean longitude
        static constexpr int STATE_DEX = 2;     ///< Relative eccentricity x-component
        static constexpr int STATE_DEY = 3;     ///< Relative eccentricity y-component
        static constexpr int STATE_DIX = 4;     ///< Relative inclination x-component
        static constexpr int STATE_DIY = 5;     ///< Relative inclination y-component
        static constexpr int STATE_NU = 6;      ///< Mean argument of latitude [rad]
        static constexpr int STATE_DIM = 7;     ///< State dimension

        // Control indices
        static constexpr int CONTROL_UR = 0;  ///< Radial acceleration
        static constexpr int CONTROL_UT = 1;  ///< Tangential acceleration
        static constexpr int CONTROL_UN = 2;  ///< Normal acceleration
        static constexpr int CONTROL_DIM = 3; ///< Control dimension

        /**
         * @brief Compute continuous-time dynamics in ROE coordinates
         * @param state   Current ROE state vector
         * @param control Current control vector [ur, ut, un]
         * @param time Current time
         * @return        Derivative of the ROE state vector
         */
        Eigen::VectorXd getContinuousDynamics(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &control, double time) const override;

        /**
         * @brief Compute state Jacobian matrix via numerical finite difference
         * @param state   Current ROE state vector
         * @param control Current control vector
         * @param time Current time
         * @return        State Jacobian matrix (d f / d state)
         */
        Eigen::MatrixXd getStateJacobian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &control, double time) const override;

        /**
         * @brief Compute control Jacobian matrix via numerical finite difference
         * @param state   Current ROE state vector
         * @param control Current control vector
         * @param time Current time
         * @return        Control Jacobian matrix (d f / d control)
         */
        Eigen::MatrixXd getControlJacobian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &control, double time) const override;

        /**
         * @brief Compute state Hessian tensor
         * @param state   Current state vector
         * @param control Current control vector
         * @param time Current time
         * @return        Vector of state Hessian matrices, one per state dimension
         */
        std::vector<Eigen::MatrixXd> getStateHessian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &control, double time) const override;

        /**
         * @brief Compute control Hessian tensor
         * @param state   Current state vector
         * @param control Current control vector
         * @param time Current time
         * @return        Vector of control Hessian matrices, one per state dimension
         */
        std::vector<Eigen::MatrixXd> getControlHessian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &control, double time) const override;

        /**
         * Computes the continuous-time dynamics of the quadrotor model using autodiff
         * @param state Current state vector
         * @param control Current control input
         * @return State derivative vector
         */
        VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd &state,
                                                     const VectorXdual2nd &control) const override;

        /**
         * @brief Transform the QNS-ROE state to the local Hill/Clohessy-Wiltshire frame.
         *
         * The returned vector is [x, y, z, xdot, ydot, zdot], representing the relative
         * position and velocity in the Hill frame.
         *
         * @param roe  A 6D QNS-ROE state vector
         * @param t    Time since epoch [s]
         * @return     A 6D vector [x, y, z, xdot, ydot, zdot] in the HCW frame
         */
        Eigen::VectorXd transformROEToHCW(const Eigen::VectorXd &roe, double time) const;

        /**
         * @brief Transform a HCW state to the QNS-ROE state.
         *
         * Expects a 6D input [x, y, z, xdot, ydot, zdot], representing the relative
         * position and velocity in the Hill/Clohessy‐Wiltshire frame.
         *
         * @param hcw A 6D Hill/CWH state vector
         * @param t   Time since epoch [s] (used for the orbit phase)
         * @return    A 6D QNS-ROE vector [da, dlambda, dex, dey, dix, diy]
         */
        Eigen::VectorXd transformHCWToROE(const Eigen::VectorXd &hcw, double t) const;

        double getSemiMajorAxis() const { return a_; }
        double getMeanMotion() const { return n_ref_; }
        double getInitialMeanArgumentOfLatitude() const { return u0_; }
        double getGravitationalParameter() const { return mu_; }
        void setGravitationalParameter(double mu) { mu_ = mu; }
        void setMeanMotion(double n) { n_ref_ = n; }
        void setSemiMajorAxis(double a) { a_ = a; }
        void setInitialMeanArgumentOfLatitude(double u0) { u0_ = u0; }

    private:
        double a_;                 ///< Semi-major axis of the reference orbit [m]
        double n_ref_;             ///< Mean motion of the reference orbit [rad/s]
        double u0_;                ///< Initial mean argu- ment of latitude [rad]
        double mu_ = 3.9860044e14; ///< Gravitational parameter [m^3/s^2]
        double mass_kg_;           ///< Mass in kg
    };

} // namespace cddp

#endif // CDDP_SPACECRAFT_ROE_HPP
