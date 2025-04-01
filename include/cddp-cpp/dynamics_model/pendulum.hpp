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

#ifndef CDDP_PENDULUM_HPP
#define CDDP_PENDULUM_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Pendulum model implementation
 * 
 * This class implements a simple pendulum model with state vector [theta, theta_dot]
 * where theta is the angle and theta_dot is the angular velocity.
 * The control input is the torque applied to the pendulum.
 */
class Pendulum : public DynamicalSystem {
public:
    /**
     * @brief Constructor for the pendulum model
     * @param timestep Time step for discretization
     * @param integration_type Integration method ("euler" by default)
     */
    Pendulum(double timestep, 
             double length = 1.0, 
             double mass = 1.0, 
             double damping = 0.0,
             std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics of the pendulum
     * State vector: [theta, theta_dot]
     * Control vector: [torque]
     * @param state Current state vector
     * @param control Current control input
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the discrete-time dynamics using the specified integration method
     * @param state Current state vector
     * @param control Current control input
     * @return Next state vector
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control);
    }

    /**
     * @brief Computes the Jacobian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @return State Jacobian matrix (A matrix)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector
     * @param control Current control input
     * @return Control Jacobian matrix (B matrix)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of state Hessian matrices (one matrix per state dimension)
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of control Hessian matrices (one matrix per state dimension)
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

    // Getters
    int getStateDim() const { return STATE_DIM; }
    int getControlDim() const { return CONTROL_DIM; }

    // Pendulum parameters
    double getLength() const { return length_; }
    double getMass() const { return mass_; }
    double getDamping() const { return damping_; }
    double getGravity() const { return gravity_; }


    VectorXdual2nd getContinuousDynamicsAutodiff(
        const VectorXdual2nd& state, const VectorXdual2nd& control) const override;

private:
    // Pendulum parameters
    double length_;  // length [m]
    double mass_;    // mass [kg]
    double damping_; // damping coefficient
    double gravity_ = 9.81; // gravity [m/s^2]

    // State indices
    static constexpr int STATE_THETA = 0;      // angle
    static constexpr int STATE_THETA_DOT = 1;  // angular velocity
    static constexpr int STATE_DIM = 2;        // state dimension

    // Control indices
    static constexpr int CONTROL_TORQUE = 0;   // applied torque
    static constexpr int CONTROL_DIM = 1;      // control dimension
};

} // namespace cddp

#endif // CDDP_PENDULUM_HPP