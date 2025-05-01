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
#ifndef CDDP_PARAMETERIZED_DYNAMICAL_SYSTEM_HPP
#define CDDP_PARAMETERIZED_DYNAMICAL_SYSTEM_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept> // For std::logic_error
#include "cddp_core/dynamical_system.hpp"

namespace cddp {

class ParameterizedDynamicalSystem : public DynamicalSystem {
public:
    // Constructor
    ParameterizedDynamicalSystem(int state_dim, int control_dim, int parameter_dim, double timestep, std::string integration_type)
        : DynamicalSystem(state_dim, control_dim, timestep, integration_type), parameter_dim_(parameter_dim) {}

    virtual ~ParameterizedDynamicalSystem() = default; // Virtual destructor

    // --- Parameterized Dynamics Functions ---
    // These functions now accept an additional parameter vector.
    // Derived classes MUST override these to implement parameter-dependent dynamics.

    // Parameterized continuous dynamics: xdot = f(x_t, u_t, p)
    virtual Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control,
                                                  const Eigen::VectorXd& parameter) const {
        throw std::logic_error("getContinuousDynamics with parameter must be overridden in the derived parameterized class.");
    }

    // Parameterized discrete dynamics: x_{t+1} = f(x_t, u_t, p)
    virtual Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& control,
                                                const Eigen::VectorXd& parameter) const {
        throw std::logic_error("getDiscreteDynamics with parameter must be overridden in the derived parameterized class.");
    }

    // --- Parameterized Derivative Functions ---
    // These functions also accept the parameter vector.
    // Derived classes should override these for efficiency, though default implementations
    // (e.g., using finite differences or autodiff on the parameterized dynamics) could be provided here later.

    // Jacobian of dynamics w.r.t state: df/dx (p)
    virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                             const Eigen::VectorXd& control,
                                             const Eigen::VectorXd& parameter) const {
         throw std::logic_error("getStateJacobian with parameter must be overridden in the derived parameterized class.");
    }

    // Jacobian of dynamics w.r.t control: df/du (p)
    virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control,
                                               const Eigen::VectorXd& parameter) const {
        throw std::logic_error("getControlJacobian with parameter must be overridden in the derived parameterized class.");
    }

    // Jacobian of dynamics w.r.t parameter: df/dp
    virtual Eigen::MatrixXd getParameterJacobian(const Eigen::VectorXd& state,
                                                 const Eigen::VectorXd& control,
                                                 const Eigen::VectorXd& parameter) const {
        throw std::logic_error("getParameterJacobian must be overridden in the derived parameterized class.");
    }

    // Combined Jacobians: df/dx, df/du, df/dp
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getJacobians(const Eigen::VectorXd& state,
                                                                                     const Eigen::VectorXd& control,
                                                                                     const Eigen::VectorXd& parameter) const {
        return {getStateJacobian(state, control, parameter),
                getControlJacobian(state, control, parameter),
                getParameterJacobian(state, control, parameter)};
    }

    // --- Hessians (Optional - Add later if needed for specific algorithms) ---
    // Hessians involving parameters (d^2f/dxdp, d^2f/dudp, d^2f/dpdp, etc.)
    // could be added here following the same pattern.

    // Accessor methods
    int getParameterDim() const { return parameter_dim_; }

    // --- Hide or Override Non-Parameterized Base Class Functions ---
    // It's often good practice to explicitly hide or override the base class
    // functions that don't take parameters to avoid ambiguity.
    // We can decide on the best strategy (override and throw, or make private/protected).
    // For now, let's assume the user will call the parameterized versions.
    // Example (if we wanted to force using parameterized versions):
    /*
    virtual Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control) const override {
        throw std::logic_error("Use parameterized version of getContinuousDynamics.");
    }
    // ... similar overrides for other non-parameterized functions ...
    */


protected:
    int parameter_dim_;
};

} // namespace cddp

#endif // CDDP_PARAMETERIZED_DYNAMICAL_SYSTEM_HPP 