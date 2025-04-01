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
#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Dubins car model with a constant forward speed and steering rate control
 *
 * The state vector is [x, y, theta], where (x,y) is position and theta is heading.
 * The control input has dimension 1: [omega], the steering (turn) rate.
 */
class DubinsCar : public DynamicalSystem {
public:
    /**
     * @brief Constructor for the DubinsCar
     * @param speed Constant forward speed
     * @param timestep Time step for discretization
     * @param integration_type Integration method ("euler" by default)
     */
    DubinsCar(double speed,
              double timestep,
              std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics of the Dubins car
     *
     * State: [x, y, theta]
     * Control: [omega]
     *
     * @param state Current state vector
     * @param control Current control input (turn rate)
     * @return State derivative [dx/dt, dy/dt, dtheta/dt]
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) const override;

    /**
     * @brief Discrete dynamics via the base class method (e.g., Euler or RK)
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                        const Eigen::VectorXd& control) const override {
        // Uses the base class integrator approach
        return DynamicalSystem::getDiscreteDynamics(state, control);
    }

    /**
     * @brief Jacobian of the dynamics wrt. the state
     *
     * @param state Current state
     * @param control Current control
     * @return A matrix (3x3)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control) const override;

    /**
     * @brief Jacobian of the dynamics wrt. the control
     *
     * @param state Current state
     * @param control Current control
     * @return B matrix (3x1)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control) const override;

    /**
     * @brief Hessian of the dynamics wrt. the state
     *
     * @param state Current state
     * @param control Current control
     * @return Vector of state Hessian matrices (one per state dimension)
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Hessian of the dynamics wrt. the control
     *
     * @param state Current state
     * @param control Current control
     * @return Vector of control Hessian matrices (one per state dimension)
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control) const override;

    // Add the required autodiff version declaration
    VectorXdual2nd getContinuousDynamicsAutodiff(
        const VectorXdual2nd& state, const VectorXdual2nd& control) const override;

private:
    // State indices
    static constexpr int STATE_X = 0;      // x position
    static constexpr int STATE_Y = 1;      // y position
    static constexpr int STATE_THETA = 2;  // heading angle
    static constexpr int STATE_DIM = 3;    // dimension

    // Control index
    static constexpr int CONTROL_OMEGA = 0;  // steering rate
    static constexpr int CONTROL_DIM = 1;    // dimension

    double speed_; // Constant forward speed
};

} // namespace cddp

#endif // CDDP_DUBINS_CAR_HPP
