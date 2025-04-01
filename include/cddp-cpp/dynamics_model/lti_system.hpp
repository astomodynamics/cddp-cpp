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
#ifndef CDDP_LTI_SYSTEM_HPP
#define CDDP_LTI_SYSTEM_HPP

#include "cddp_core/dynamical_system.hpp"
#include <random>

namespace cddp {

/**
 * @brief Linear Time-Invariant System dynamics model
 * 
 * Implements a control-limited LTI system with dynamics:
 * x_{k+1} = Ax_{k} + Bu_{k}
 * where A and B are constant matrices
 */
class LTISystem : public DynamicalSystem {
public:
    /**
     * @brief Constructor for random LTI system
     * @param state_dim State dimension
     * @param control_dim Control dimension
     * @param timestep Time step for discretization
     * @param integration_type Integration method ("euler" by default)
     */
    LTISystem(int state_dim, 
              int control_dim, 
              double timestep,
              std::string integration_type = "euler");

    /**
     * @brief Constructor for specified LTI system
     * @param A System matrix
     * @param B Input matrix
     * @param timestep Time step for discretization
     * @param integration_type Integration method
     */
    LTISystem(const Eigen::MatrixXd& A,
              const Eigen::MatrixXd& B,
              double timestep,
              std::string integration_type = "euler");

    /**
     * @brief Computes the continuous-time dynamics
     * @param state Current state vector
     * @param control Current control input
     * @return State derivative vector
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getContinuousDynamics(state, control);
    }

    /**
     * @brief Computes the discrete-time dynamics using the specified integration method
     * @param state Current state vector
     * @param control Current control input
     * @return Next state vector
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Jacobian w.r.t state (A matrix)
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes the Jacobian w.r.t control (B matrix)
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes state Hessian (zero for linear system)
     * @return Vector of state Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                   const Eigen::VectorXd& control) const override;

    /**
     * @brief Computes control Hessian (zero for linear system)
     * @return Vector of control Hessian matrices, one per state dimension
     */
    std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override;

    // Getters
    const Eigen::MatrixXd& getA() const { return A_; }
    const Eigen::MatrixXd& getB() const { return B_; }


    VectorXdual2nd getContinuousDynamicsAutodiff(
        const VectorXdual2nd& state, const VectorXdual2nd& control) const override;

private:
    Eigen::MatrixXd A_;  ///< System matrix
    Eigen::MatrixXd B_;  ///< Input matrix

    /**
     * @brief Initialize random stable system matrices
     */
    void initializeRandomSystem();

    // Ensure helper declaration exists
    VectorXdual2nd getDiscreteDynamicsAutodiff( 
        const VectorXdual2nd& state, const VectorXdual2nd& control) const;
};

} // namespace cddp

#endif // CDDP_LTI_SYSTEM_HPP