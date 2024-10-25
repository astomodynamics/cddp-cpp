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

#include "cddp-cpp/cddp_core/dynamical_system.hpp"

namespace cddp {

class DubinsCar : public DynamicalSystem {
public:
    DubinsCar(double timestep, std::string integration_type = "euler");

    // Dynamics: Computes the next state given the current state and control input
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Discrete dynamics (declare it even if using base class implementation)
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control); 
    }
    
    // Jacobians: Computes the Jacobian matrices of the dynamics with respect to state and control
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Hessians: Computes the Hessian matrices of the dynamics (optional for now)
    // You might implement these later if needed for your DDP algorithm
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
};

} // namespace cddp

#endif // CDDP_DUBINS_CAR_HPP