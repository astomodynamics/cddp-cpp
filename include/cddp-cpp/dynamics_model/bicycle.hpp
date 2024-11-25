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

#ifndef CDDP_BICYCLE_HPP
#define CDDP_BICYCLE_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class Bicycle : public DynamicalSystem {
public:
    /**
     * Constructor for the Bicycle model
     * @param timestep Time step for discretization
     * @param wheelbase Distance between front and rear axles
     * @param integration_type Integration method ("euler" by default)
     */
    Bicycle(double timestep, double wheelbase, std::string integration_type = "euler");

    /**
     * Computes the continuous-time dynamics of the bicycle model
     * State vector: [x, y, theta, v]
     * Control vector: [acceleration, steering_angle]
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
     * @param state Current state vector
     * @param control Current control input
     * @return State Jacobian matrix (A matrix)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector
     * @param control Current control input
     * @return Control Jacobian matrix (B matrix)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @return State Hessian matrix
     */
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @return Control Hessian matrix
     */
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

private:
    double wheelbase_;  // Distance between front and rear axles

    // State indices
    static constexpr int STATE_X = 0;      // x position
    static constexpr int STATE_Y = 1;      // y position
    static constexpr int STATE_THETA = 2;  // heading angle
    static constexpr int STATE_V = 3;      // velocity
    static constexpr int STATE_DIM = 4;    // state dimension

    // Control indices
    static constexpr int CONTROL_ACC = 0;    // acceleration
    static constexpr int CONTROL_DELTA = 1;  // steering angle
    static constexpr int CONTROL_DIM = 2;    // control dimension
};

} // namespace cddp

#endif // CDDP_BICYCLE_HPP