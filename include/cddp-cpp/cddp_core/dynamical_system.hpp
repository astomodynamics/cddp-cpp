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
#include <vector>
#include "cddp_core/helper.hpp"

namespace cddp {
class DynamicalSystem {
public:

    // Constructor
    DynamicalSystem(int state_dim, int control_dim, double timestep, std::string integration_type) 
        : state_dim_(state_dim), control_dim_(control_dim), timestep_(timestep), integration_type_(integration_type) {}

    virtual ~DynamicalSystem() {} // Virtual destructor

    // Core dynamics function: xdot = f(x_t, u_t)
    virtual Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                  const Eigen::VectorXd& control) const;
    
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

    // Hessian of dynamics w.r.t state: d^2f/dx^2
    // This is a tensor (state_dim x state_dim x state_dim), represented as a vector of matrices
    // Each matrix is state_dim x state_dim, and corresponds to the Hessian of one state dimension
    virtual std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const = 0;

    // Hessian of dynamics w.r.t control: d^2f/du^2
    // This is a tensor (state_dim x control_dim x control_dim), represented as a vector of matrices
    // Each matrix is control_dim x control_dim, and corresponds to the Hessian of one state dimension
    virtual std::vector<Eigen::MatrixXd> getControlHessian(const Eigen::VectorXd& state, 
                                        const Eigen::VectorXd& control) const = 0;

    // Hessian of dynamics w.r.t state and control: d^2f/dudx
    // This is a tensor (state_dim x control_dim x state_dim), represented as a vector of matrices
    // Each matrix is control_dim x state_dim, and corresponds to the Hessian of one state dimension
    virtual std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const {
        std::vector<Eigen::MatrixXd> cross_hessian(state_dim_);
        for (int i = 0; i < state_dim_; ++i) {
            cross_hessian[i] = Eigen::MatrixXd::Zero(control_dim_, state_dim_);
        }
        return cross_hessian;
    }

    // Hessian of dynamics w.r.t state and control: d^2f/dx^2, d^2f/du^2, d^2f/dudx
    virtual std::tuple<std::vector<Eigen::MatrixXd>, 
                      std::vector<Eigen::MatrixXd>, 
                      std::vector<Eigen::MatrixXd>> getHessians(const Eigen::VectorXd& state, 
                                                               const Eigen::VectorXd& control) const {
        return {getStateHessian(state, control), 
                getControlHessian(state, control), 
                getCrossHessian(state, control)};
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

    // Integration step functions
    Eigen::VectorXd euler_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
    Eigen::VectorXd heun_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
    Eigen::VectorXd rk3_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
    Eigen::VectorXd rk4_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const;
};
}
#endif // CDDP_DYNAMICAL_SYSTEM_HPP