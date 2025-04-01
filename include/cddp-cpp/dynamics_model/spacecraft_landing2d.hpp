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

#ifndef CDDP_SPACECRAFT_LANDING2D_HPP
#define CDDP_SPACECRAFT_LANDING2D_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class SpacecraftLanding2D : public DynamicalSystem {
public:
    /**
     * Constructor for the 2D Spacecraft Landing model
     * @param timestep Time step for discretization
     * @param integration_type Integration method ("euler" by default)
     * @param mass Mass of the spacecraft [kg]
     * @param length Length of the spacecraft [m]
     * @param width Width of the spacecraft [m]
     * @param min_thrust Minimum thrust [N]
     * @param max_thrust Maximum thrust [N]
     * @param max_gimble Maximum gimble angle [rad]
     * @ref https://thomas-godden.medium.com/how-spacex-lands-starship-sort-of-ee96cdde650b 
     */
    SpacecraftLanding2D(double timestep = 0.1,
                        std::string integration_type = "rk4",
                        double mass = 100000.0,
                        double length = 50.0,
                        double width = 10.0,
                        double min_thrust = 880000.0,
                        double max_thrust = 2210000.0,
                        double max_gimble = 0.349066); // 20 degrees in radians

    /**
     * Computes continuous-time dynamics
     * State vector: [x, x_dot, y, y_dot, theta, theta_dot]
     * Control vector: [thrust_percent, thrust_angle]
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control) const override;

    VectorXdual2nd getContinuousDynamicsAutodiff( // Use alias
        const VectorXdual2nd& state, const VectorXdual2nd& control) const override;

    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control);
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

    // Accessors
    int getStateDim() const { return STATE_DIM; }
    int getControlDim() const { return CONTROL_DIM; }

    double getMass() const { return mass_; }
    double getLength() const { return length_; }
    double getWidth() const { return width_; }
    double getInertia() const { return inertia_; }
    double getMinThrust() const { return min_thrust_; }
    double getMaxThrust() const { return max_thrust_; }
    double getMaxGimble() const { return max_gimble_; }
    double getGravity() const { return gravity_; }
    
private:
    // System parameters
    double mass_;        // Mass of spacecraft [kg]
    double length_;      // Length of spacecraft [m]
    double width_;       // Width of spacecraft [m]
    double inertia_;     // Moment of inertia [kg*m^2]
    double min_thrust_;  // Minimum thrust [N]
    double max_thrust_;  // Maximum thrust [N]
    double max_gimble_;  // Maximum gimble angle [rad]
    double gravity_{9.81};  // Gravitational acceleration [m/s^2]

    // State indices
    static constexpr int STATE_X = 0;           // x position
    static constexpr int STATE_X_DOT = 1;       // x velocity
    static constexpr int STATE_Y = 2;           // y position
    static constexpr int STATE_Y_DOT = 3;       // y velocity
    static constexpr int STATE_THETA = 4;       // angle
    static constexpr int STATE_THETA_DOT = 5;   // angular velocity
    static constexpr int STATE_DIM = 6;         // state dimension

    // Control indices
    static constexpr int CONTROL_THRUST = 0;     // thrust percentage
    static constexpr int CONTROL_ANGLE = 1;      // thrust angle
    static constexpr int CONTROL_DIM = 2;        // control dimension
};

} // namespace cddp

#endif // CDDP_SPACECRAFT_LANDING2D_HPP