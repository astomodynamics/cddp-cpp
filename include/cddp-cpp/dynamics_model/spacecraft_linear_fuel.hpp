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

#ifndef CDDP_SPACECRAFT_LINEAR_FUEL_HPP
#define CDDP_SPACECRAFT_LINEAR_FUEL_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class SpacecraftLinearFuel : public DynamicalSystem {
public:
    /**
     * Constructor for the Linear Spacecraft model with fuel consumption (based on HCW equations)
     * @param timestep Time step for discretization
     * @param mean_motion Mean motion of the reference orbit (sqrt(mu/a^3))
     * @param isp Specific impulse of the thruster (in seconds)
     * @param g0 Standard gravity (m/s^2, defaults to 9.80665)
     * @param integration_type Integration method ("euler" by default)
     */
    SpacecraftLinearFuel(double timestep, double mean_motion, double isp, 
                         double g0 = 9.80665, std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the model
     * State vector: [x, y, z, vx, vy, vz, mass, accumulated_control_effort]
     * where (x,y,z) is the relative position in LVLH frame
     * and (vx,vy,vz) is the relative velocity
     * and mass is the current mass of the spacecraft
     * and accumulated_control_effort is the integral of norm(control)^2 / 2
     * Control vector: [Fx, Fy, Fz] (applied forces in LVLH frame)
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
     * @param state Current state vector [x, y, z, vx, vy, vz, mass, accumulated_control_effort]
     * @param control Current control input
     * @return State Jacobian matrix (A matrix)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector [x, y, z, vx, vy, vz, mass, accumulated_control_effort]
     * @param control Current control input
     * @return Control Jacobian matrix (B matrix)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the state and control input
     * @param state Current state vector [x, y, z, vx, vy, vz, mass, accumulated_control_effort]
     * @param control Current control input
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd& state, 
                                                const Eigen::VectorXd& control) const override;
    /**
     * Computes the Hessian of the dynamics with respect to the state
     * Note: For this non-linear system, Hessians are generally non-zero.
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the control
     * Note: For this non-linear system, Hessians are generally non-zero.
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of control Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

    /**
     * Computes the continuous-time dynamics of the model using autodiff
     * @param state Current state vector [x, y, z, vx, vy, vz, mass, accumulated_control_effort]
     * @param control Current control input
     * @return State derivative vector
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd& state, 
                                                const VectorXdual2nd& control) const override;

private:
    double mean_motion_;  // Mean motion of reference orbit
    double isp_;          // Specific impulse (s)
    double g0_;           // Standard gravity (m/s^2)
    double epsilon_;      // Small value to avoid division by zero

    // State indices
    static constexpr int STATE_X = 0;   // x position (radial)
    static constexpr int STATE_Y = 1;   // y position (along-track)
    static constexpr int STATE_Z = 2;   // z position (cross-track)
    static constexpr int STATE_VX = 3;  // x velocity
    static constexpr int STATE_VY = 4;  // y velocity
    static constexpr int STATE_VZ = 5;  // z velocity
    static constexpr int STATE_MASS = 6; // spacecraft mass
    static constexpr int STATE_ACCUMULATED_CONTROL_EFFORT = 7; // accumulated control effort (integral of norm(u)^2/2)
    static constexpr int STATE_DIM = 8; // state dimension

    // Control indices (forces in LVLH frame)
    static constexpr int CONTROL_FX = 0;  // force in x direction
    static constexpr int CONTROL_FY = 1;  // force in y direction
    static constexpr int CONTROL_FZ = 2;  // force in z direction
    static constexpr int CONTROL_DIM = 3; // control dimension
};

} // namespace cddp

#endif // CDDP_SPACECRAFT_LINEAR_FUEL_HPP