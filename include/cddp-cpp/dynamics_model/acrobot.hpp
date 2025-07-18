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

#ifndef CDDP_ACROBOT_HPP
#define CDDP_ACROBOT_HPP

#include "cddp_core/dynamical_system.hpp"

namespace cddp {

/**
 * @brief Acrobot model implementation
 * 
 * A double-pendulum with actuation only at the elbow joint.
 * State vector: [theta1, theta2, theta1_dot, theta2_dot]
 * Control vector: [torque] (applied to second joint only)
 * 
 * This implementation follows the Julia RobotZoo.jl Acrobot model.
 */
class Acrobot : public DynamicalSystem {
public:
    /**
     * @brief Constructor for the acrobot model
     * @param timestep Time step for discretization
     * @param l1 Length of first link
     * @param l2 Length of second link
     * @param m1 Mass of first link
     * @param m2 Mass of second link
     * @param J1 Inertia of first link
     * @param J2 Inertia of second link
     * @param integration_type Integration method
     */
    Acrobot(double timestep,
            double l1 = 1.0, double l2 = 1.0,
            double m1 = 1.0, double m2 = 1.0,
            double J1 = 1.0, double J2 = 1.0,
            std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics of the acrobot
     * @param state Current state vector
     * @param control Current control input
     * @param time Current time (unused in this model)
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control, double time) const override;

    /**
     * @brief Computes the discrete-time dynamics using the specified integration method
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
     * @brief Computes the Jacobian of the dynamics with respect to the state
     * Uses base class autodiff implementation
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getStateJacobian(state, control, time);
    }

    /**
     * @brief Computes the Jacobian of the dynamics with respect to the control input
     * Uses base class autodiff implementation
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getControlJacobian(state, control, time);
    }

    /**
     * @brief Computes the Hessian of the dynamics with respect to the state
     * Uses base class autodiff implementation
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                                const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getStateHessian(state, control, time);
    }

    /**
     * @brief Computes the Hessian of the dynamics with respect to the control
     * Uses base class autodiff implementation
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                                  const Eigen::VectorXd& control, double time) const override {
        return DynamicalSystem::getControlHessian(state, control, time);
    }

    // Getters
    int getStateDim() const { return STATE_DIM; }
    int getControlDim() const { return CONTROL_DIM; }

    // Acrobot parameters
    double getL1() const { return l1_; }
    double getL2() const { return l2_; }
    double getM1() const { return m1_; }
    double getM2() const { return m2_; }
    double getJ1() const { return J1_; }
    double getJ2() const { return J2_; }
    double getGravity() const { return gravity_; }
    double getFriction() const { return friction_; }

    /**
     * @brief Autodiff implementation for automatic differentiation
     */
    VectorXdual2nd getContinuousDynamicsAutodiff(
        const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const override;

private:
    // Acrobot parameters
    double l1_;   // length of link 1 [m]
    double l2_;   // length of link 2 [m]
    double m1_;   // mass of link 1 [kg]
    double m2_;   // mass of link 2 [kg]
    double J1_;   // inertia of link 1 [kg*m^2]
    double J2_;   // inertia of link 2 [kg*m^2]
    double gravity_ = 9.81; // gravity [m/s^2]
    double friction_ = 1.0; // friction coefficient

    // State indices
    static constexpr int STATE_THETA1 = 0;      // angle of first link
    static constexpr int STATE_THETA2 = 1;      // angle of second link
    static constexpr int STATE_THETA1_DOT = 2;  // angular velocity of first link
    static constexpr int STATE_THETA2_DOT = 3;  // angular velocity of second link
    static constexpr int STATE_DIM = 4;         // state dimension

    // Control indices
    static constexpr int CONTROL_TORQUE = 0;    // torque applied to second joint
    static constexpr int CONTROL_DIM = 1;       // control dimension
};

} // namespace cddp

#endif // CDDP_ACROBOT_HPP