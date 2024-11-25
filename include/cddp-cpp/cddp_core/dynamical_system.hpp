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
#ifndef CDDP_DYNAMICAL_SYSTEM_HPP
#define CDDP_DYNAMICAL_SYSTEM_HPP

#include <Eigen/Dense>

namespace cddp {
class DynamicalSystem {
public:

    // Constructor
    DynamicalSystem(int state_dim, int control_dim, double timestep, std::string integration_type) 
        : state_dim_(state_dim), control_dim_(control_dim), timestep_(timestep), integration_type_(integration_type) {}

    virtual ~DynamicalSystem() {} // Virtual destructor

    // Core dynamics function: xdot = f(x_t, u_t)
    virtual Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                  const Eigen::VectorXd& control) const = 0;
    
    // Discrete dynamics function: x_{t+1} = f(x_t, u_t)
    virtual Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, 
                                  const Eigen::VectorXd& control) const;

    // Jacobian of dynamics w.r.t state: df/dx
    virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                        const Eigen::VectorXd& control) const = 0;

    // Jacobian of dynamics w.r.t control: df/du
    virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                          const Eigen::VectorXd& control) const = 0;

    // Jacobians of dynamics w.r.t state and control: df/dx, df/du
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getJacobians(const Eigen::VectorXd& state, 
                                                            const Eigen::VectorXd& control) const {
        return {getStateJacobian(state, control), getControlJacobian(state, control)};
    }

    Eigen::MatrixXd getFiniteDifferenceStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const;
    
    Eigen::MatrixXd getFiniteDifferenceControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const;

    // TODO: Add methods for Hessian calculations
    // Hessian of dynamics w.r.t state: d^2f/dx^2
    // Note: This is a tensor, but we represent it as a matrix for now.
    // Each row corresponds to the Hessian for one state dimension
    virtual Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const = 0;

    // Hessian of dynamics w.r.t control: d^2f/du^2
    // Similar representation as state Hessian
    virtual Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, 
                                        const Eigen::VectorXd& control) const = 0;

    // Hessian of dynamics w.r.t state and control: d^2f/dudx
    // Similar representation
    virtual Eigen::MatrixXd getCrossHessian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const {
        return Eigen::MatrixXd::Zero(state.size() * control.size(), state.size()); 
    }

    // Hessian of dynamics w.r.t state and control: d^2f/dx^2, d^2f/du^2, d^2f/dudx
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getHessians(const             Eigen::VectorXd& state, 
                                                                            const Eigen::VectorXd& control) const {
        return {getStateHessian(state, control), getControlHessian(state, control), getCrossHessian(state, control)};
    }

    // Accessor methods
    int getStateDim() const { return state_dim_; }
    int getControlDim() const { return control_dim_; }
    double getTimestep() const { return timestep_; }
    std::string getIntegrationType() const { return integration_type_; }

protected:
    int state_dim_;
    int control_dim_;
    double timestep_;
    std::string integration_type_; // Integration type: Euler, Heun, RK3, RK4

    // Integration step functions (declare as protected or private)
    Eigen::VectorXd euler_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
    Eigen::VectorXd heun_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
    Eigen::VectorXd rk3_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
    Eigen::VectorXd rk4_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
};
}
#endif // CDDP_DYNAMICAL_SYSTEM_HPP