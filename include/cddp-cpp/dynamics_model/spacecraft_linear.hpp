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

#ifndef CDDP_HCW_HPP
#define CDDP_HCW_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class HCW : public DynamicalSystem {
public:
    /**
     * Constructor for the Hill-Clohessy-Wiltshire model
     * @param timestep Time step for discretization
     * @param mean_motion Mean motion of the reference orbit (sqrt(mu/a^3))
     * @param mass Mass of the chaser spacecraft
     * @param integration_type Integration method ("euler" by default)
     */
    HCW(double timestep, double mean_motion, double mass, 
        std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the HCW model
     * State vector: [x, y, z, vx, vy, vz]
     * where (x,y,z) is the relative position in LVLH frame
     * and (vx,vy,vz) is the relative velocity
     * Control vector: [Fx, Fy, Fz] (applied forces in LVLH frame)
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
     * Computes the continuous-time dynamics of the HCW model using autodiff
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return State derivative vector
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(const VectorXdual2nd& state, 
                                                const VectorXdual2nd& control, double time) const override;

private:
    double mean_motion_;  // Mean motion of reference orbit
    double mass_;         // Mass of the chaser spacecraft

    // State indices
    static constexpr int STATE_X = 0;   // x position (radial)
    static constexpr int STATE_Y = 1;   // y position (along-track)
    static constexpr int STATE_Z = 2;   // z position (cross-track)
    static constexpr int STATE_VX = 3;  // x velocity
    static constexpr int STATE_VY = 4;  // y velocity
    static constexpr int STATE_VZ = 5;  // z velocity
    static constexpr int STATE_DIM = 6; // state dimension

    // Control indices (forces in LVLH frame)
    static constexpr int CONTROL_FX = 0;  // force in x direction
    static constexpr int CONTROL_FY = 1;  // force in y direction
    static constexpr int CONTROL_FZ = 2;  // force in z direction
    static constexpr int CONTROL_DIM = 3; // control dimension
};

} // namespace cddp

#endif // CDDP_HCW_HPP