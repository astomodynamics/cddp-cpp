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

#ifndef CDDP_DREYFUS_ROCKET_HPP
#define CDDP_DREYFUS_ROCKET_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Dreyfus Rocket dynamics model
 * 
 * A simple rocket model with vertical motion under thrust and gravity.
 */
class DreyfusRocket : public DynamicalSystem {
public:
    /**
     * @brief Constructs a DreyfusRocket system with configurable parameters
     * @param timestep Time step for discretization [s]
     * @param integration_type Integration method ("euler" by default)
     * @param thrust_acceleration Thrust acceleration [m/s^2]
     * @param gravity_acceleration Gravitational acceleration [m/s^2]
     */
    DreyfusRocket(double timestep, 
                  std::string integration_type = "rk4",
                  double thrust_acceleration = 64.0,
                  double gravity_acceleration = 32.0);

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
     * @brief Computes state Hessian tensor (∂²f/∂x²)
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes control Hessian tensor (∂²f/∂u²)
     * @param state Current state vector
     * @param control Current control input
     * @return Vector of control Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

    // Getters
    double getThrustAcceleration() const { return thrust_acceleration_; }
    double getGravityAcceleration() const { return gravity_acceleration_; }

private:
    // State indices
    static constexpr int STATE_X = 0;        ///< Position index
    static constexpr int STATE_X_DOT = 1;    ///< Velocity index
    static constexpr int STATE_DIM = 2;      ///< State dimension

    // Control indices
    static constexpr int CONTROL_THETA = 0;  ///< Thrust angle index
    static constexpr int CONTROL_DIM = 1;    ///< Control dimension

    // System parameters
    double thrust_acceleration_;     ///< Thrust acceleration [m/s^2]
    double gravity_acceleration_;    ///< Gravitational acceleration [m/s^2]
};

} // namespace cddp

#endif // CDDP_DREYFUS_ROCKET_HPP