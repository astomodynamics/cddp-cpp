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

#ifndef CDDP_CAR_HPP
#define CDDP_CAR_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Car dynamics model
 * 
 * Implements a kinematic car model with the following state and control:
 * State vector: [x, y, θ, v]
 * - x: x position
 * - y: y position
 * - θ: heading angle
 * - v: velocity
 * 
 * Control vector: [δ, a]
 * - δ: steering angle
 * - a: acceleration
 */
class Car : public DynamicalSystem {
public:
    /**
     * @brief Constructor for the Car model
     * @param timestep Time step for discretization
     * @param wheelbase Wheelbase length (distance between front and rear axles)
     * @param integration_type Integration method ("euler", "heun", "rk3", "rk4")
     */
    Car(double timestep = 0.03, 
        double wheelbase = 2.0, 
        std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics of the car model
     * @param state Current state vector [x, y, θ, v]
     * @param control Current control input [δ, a]
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
     * @return State Jacobian matrix (df/dx)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector
     * @param control Current control input
     * @return Control Jacobian matrix (df/du)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @return State Hessian matrix (d²f/dx²)
     */
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @return Control Hessian matrix (d²f/du²)
     */
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

private:
    double wheelbase_;       ///< Distance between front and rear axles

    // State indices
    static constexpr int STATE_X = 0;      ///< x position index
    static constexpr int STATE_Y = 1;      ///< y position index
    static constexpr int STATE_THETA = 2;  ///< heading angle index
    static constexpr int STATE_V = 3;      ///< velocity index
    static constexpr int STATE_DIM = 4;    ///< total state dimension

    // Control indices
    static constexpr int CONTROL_DELTA = 0; ///< steering angle index
    static constexpr int CONTROL_A = 1;     ///< acceleration index
    static constexpr int CONTROL_DIM = 2;   ///< total control dimension
};

} // namespace cddp

#endif // CDDP_CAR_HPP