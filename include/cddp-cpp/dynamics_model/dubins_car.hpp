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

#ifndef CDDP_DUBINS_CAR_HPP
#define CDDP_DUBINS_CAR_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Dubins Car model implementation
 * 
 * This class implements the Dubins car model, which is a simple car-like robot
 * that can only move forward at a constant speed and can change its heading angle.
 * The state vector consists of [x, y, theta], where (x,y) is the position and
 * theta is the heading angle. The control input is the steering angle rate.
 */
class DubinsCar : public DynamicalSystem {
public:
    /**
     * @brief Constructor for the Dubins car model
     * @param timestep Time step for discretization
     * @param integration_type Integration method ("euler" by default)
     */
    DubinsCar(double timestep, 
              std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics of the Dubins car
     * State vector: [x, y, theta]
     * Control vector: [omega] (steering angle rate)
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
     * @return State Hessian matrix
     */
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @return Control Hessian matrix
     */
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

private:
    // State indices
    static constexpr int STATE_X = 0;      // x position
    static constexpr int STATE_Y = 1;      // y position
    static constexpr int STATE_THETA = 2;  // heading angle
    static constexpr int STATE_DIM = 3;    // state dimension

    // Control indices
    static constexpr int CONTROL_V = 0;      // velocity
    static constexpr int CONTROL_OMEGA = 1;  // steering angle rate
    static constexpr int CONTROL_DIM = 2;    // control dimension
};

} // namespace cddp

#endif // CDDP_DUBINS_CAR_HPP