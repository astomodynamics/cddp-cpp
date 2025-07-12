/*
 Copyright 2025 Tomo Sasaki

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

#ifndef CDDP_FORKLIFT_HPP
#define CDDP_FORKLIFT_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Forklift dynamics model with rear-wheel steering (derived from Car class)
 * 
 * Implements a kinematic forklift model with first-order steering actuator dynamics:
 * State vector: [x, y, θ, v, δ]
 * - x: x position
 * - y: y position
 * - θ: heading angle
 * - v: velocity
 * - δ: steering angle (state variable for actuator dynamics)
 * 
 * Control vector: [a, δ̇]
 * - a: acceleration
 * - δ̇: steering angle rate
 * 
 * This model uses Ackermann steering geometry with optional rear-steer configuration,
 * typical for warehouse forklifts. The steering angle is modeled as a state to
 * represent first-order actuator dynamics.
 */
class Forklift : public DynamicalSystem {
public:

    /**
     * @brief Constructor for the Forklift model
     * @param timestep Time step for discretization
     * @param wheelbase Wheelbase length (distance between front and rear axles)
     * @param integration_type Integration method ("euler", "heun", "rk3", "rk4")
     * @param rear_steer True for rear-wheel steering (typical for forklifts)
     * @param max_steering_angle Maximum steering angle in radians
     */
    Forklift(double timestep = 0.01,
             double wheelbase = 2.0,
             std::string integration_type = "euler",
             bool rear_steer = true,
             double max_steering_angle = 0.785398);  // 45 degrees default

    /**
     * @brief Computes the continuous-time dynamics of the forklift model
     * @param state Current state vector [x, y, θ, v, δ]
     * @param control Current control input [a, δ̇]
     * @param time Current time
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                         const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getContinuousDynamics(state, control, time);
    }

    /**
     * @brief Computes the discrete-time dynamics using the specified integration method
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Next state vector
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Computes the Jacobian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return State Jacobian matrix (df/dx)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Computes the Jacobian of the dynamics with respect to the control input
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Control Jacobian matrix (df/du)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the state
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Vector of state Hessian matrices, one per state dimension (d²f/dx²)
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Computes the Hessian of the dynamics with respect to the control
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time
     * @return Vector of control Hessian matrices, one per state dimension (d²f/du²)
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control, double time) const override;


    cddp::VectorXdual2nd getContinuousDynamicsAutodiff(
        const cddp::VectorXdual2nd& state, const cddp::VectorXdual2nd& control, double time) const override;

private:
    double wheelbase_;           ///< Distance between front and rear axles
    bool rear_steer_;           ///< True for rear-steer forklift
    double max_steering_angle_; ///< Maximum steering angle in radians

    // State indices
    static constexpr int STATE_X = 0;      ///< x position index
    static constexpr int STATE_Y = 1;      ///< y position index
    static constexpr int STATE_THETA = 2;  ///< heading angle index
    static constexpr int STATE_V = 3;      ///< velocity index
    static constexpr int STATE_DELTA = 4;  ///< steering angle index
    static constexpr int STATE_DIM = 5;    ///< total state dimension

    // Control indices
    static constexpr int CONTROL_A = 0;       ///< acceleration index
    static constexpr int CONTROL_DDELTA = 1;  ///< steering rate index
    static constexpr int CONTROL_DIM = 2;     ///< total control dimension

    // Helper function for autodiff discrete dynamics
    cddp::VectorXdual2nd getDiscreteDynamicsAutodiff(
        const cddp::VectorXdual2nd& state, const cddp::VectorXdual2nd& control, double time) const;
};

} // namespace cddp

#endif // CDDP_FORKLIFT_HPP