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

#ifndef CDDP_CARTPOLE_HPP
#define CDDP_CARTPOLE_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief CartPole (inverted pendulum on cart) dynamics model
 * 
 * A classical control problem consisting of a pole attached to a cart moving along a
 * frictionless track. The system is controlled by applying a force to the cart.
 * State: [x, x_dot, theta, theta_dot]
 * - x: Cart position
 * - x_dot: Cart velocity
 * - theta: Pole angle (0 is upright)
 * - theta_dot: Pole angular velocity
 * Control: [force]
 * - force: Force applied to cart
 */
class CartPole : public DynamicalSystem {
public:
    /**
     * @brief Constructs a CartPole system with configurable parameters
     * @param timestep Time step for discretization [s]
     * @param integration_type Integration method ("euler" by default)
     * @param cart_mass Mass of the cart [kg]
     * @param pole_mass Mass of the pole [kg]
     * @param pole_length Length of the pole [m]
     * @param gravity Gravitational acceleration [m/s^2]
     * @param damping Damping coefficient for cart motion
     */
    CartPole(double timestep, 
             std::string integration_type = "rk4",
             double cart_mass = 1.0,
             double pole_mass = 0.2,
             double pole_length = 0.5,
             double gravity = 9.81,
             double damping = 0.0);

    /**
     * @brief Computes continuous-time system dynamics
     * @param state Current state vector
     * @param control Current control input
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes discrete-time system dynamics
     * @param state Current state vector
     * @param control Current control input
     * @return Next state vector
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control);
    }

    /**
     * @brief Computes state Jacobian matrix (∂f/∂x)
     * @param state Current state vector
     * @param control Current control input
     * @return State Jacobian matrix
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes control Jacobian matrix (∂f/∂u)
     * @param state Current state vector
     * @param control Current control input
     * @return Control Jacobian matrix
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes state Hessian matrix (∂²f/∂x²)
     * @param state Current state vector
     * @param control Current control input
     * @return State Hessian matrix
     */
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes control Hessian matrix (∂²f/∂u²)
     * @param state Current state vector
     * @param control Current control input
     * @return Control Hessian matrix
     */
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

    // Getters
    double getCartMass() const { return cart_mass_; }
    double getPoleMass() const { return pole_mass_; }
    double getPoleLength() const { return pole_length_; }
    double getGravity() const { return gravity_; }
    double getDamping() const { return damping_; }

private:
    // State indices
    static constexpr int STATE_X = 0;          ///< Cart position index
    static constexpr int STATE_THETA = 1;      ///< Pole angle index
    static constexpr int STATE_X_DOT = 2;      ///< Cart velocity index
    static constexpr int STATE_THETA_DOT = 3;  ///< Pole angular velocity index
    static constexpr int STATE_DIM = 4;        ///< State dimension

    // Control indices
    static constexpr int CONTROL_FORCE = 0;    ///< Force control index
    static constexpr int CONTROL_DIM = 1;      ///< Control dimension

    // System parameters
    double cart_mass_;    ///< Mass of the cart [kg]
    double pole_mass_;    ///< Mass of the pole [kg]
    double pole_length_;  ///< Length of the pole [m]
    double gravity_;      ///< Gravitational acceleration [m/s^2]
    double damping_;      ///< Damping coefficient
};

} // namespace cddp

#endif // CDDP_CARTPOLE_HPP