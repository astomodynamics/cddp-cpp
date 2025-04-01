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

#include <iostream>
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>       // Include autodiff
#include <autodiff/forward/dual/eigen.hpp> // Include autodiff Eigen support

#include "cddp_core/dynamical_system.hpp"

using namespace cddp;
using namespace autodiff; // Use autodiff namespace

// Implement integration methods
Eigen::VectorXd DynamicalSystem::euler_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const {
    return state + dt * getContinuousDynamics(state, control);
}

Eigen::VectorXd DynamicalSystem::heun_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const {
    Eigen::VectorXd k1 = getContinuousDynamics(state, control);
    Eigen::VectorXd k2 = getContinuousDynamics(state + dt * k1, control);
    return state + 0.5 * dt * (k1 + k2);
}

Eigen::VectorXd DynamicalSystem::rk3_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const {
    Eigen::VectorXd k1 = getContinuousDynamics(state, control);
    Eigen::VectorXd k2 = getContinuousDynamics(state + 0.5 * dt * k1, control);
    Eigen::VectorXd k3 = getContinuousDynamics(state - dt * k1 + 2 * dt * k2, control);
    return state + (dt / 6) * (k1 + 4 * k2 + k3);
}

Eigen::VectorXd DynamicalSystem::rk4_step(const Eigen::VectorXd& state, const Eigen::VectorXd& control, double dt) const {
    Eigen::VectorXd k1 = getContinuousDynamics(state, control);
    Eigen::VectorXd k2 = getContinuousDynamics(state + 0.5 * dt * k1, control);
    Eigen::VectorXd k3 = getContinuousDynamics(state + 0.5 * dt * k2, control);
    Eigen::VectorXd k4 = getContinuousDynamics(state + dt * k3, control);
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}


Eigen::VectorXd DynamicalSystem::getDiscreteDynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    if (integration_type_ == "euler") {
        return euler_step(state, control, timestep_);
    } else if (integration_type_ == "heun") {
        return heun_step(state, control, timestep_);
    } else if (integration_type_ == "rk3") {
        return rk3_step(state, control, timestep_);
    } else if (integration_type_ == "rk4") {
        return rk4_step(state, control, timestep_);
    } else {
        std::cerr << "Integration type not supported!" << std::endl;
        return Eigen::VectorXd::Zero(state.size()); 
    }
}

Eigen::VectorXd DynamicalSystem::getContinuousDynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {

    // Get next state using discrete dynamics
    Eigen::VectorXd next_state = getDiscreteDynamics(state, control);
    
    // Compute continuous dynamics using finite difference
    // dx/dt ≈ (x_{k+1} - x_k) / dt
    Eigen::VectorXd continuous_dynamics = (next_state - state) / timestep_;
    
    return continuous_dynamics;
}

// --- Autodiff Default Implementations for Jacobians ---

Eigen::MatrixXd DynamicalSystem::getStateJacobian(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control) const {
    // Use second-order duals for consistency, jacobian works fine
    VectorXdual2nd x = state;
    VectorXdual2nd u = control;

    // Need to capture 'this' pointer for member function access
    auto dynamics_wrt_x = [&](const VectorXdual2nd& x_ad) -> VectorXdual2nd {
        return this->getContinuousDynamicsAutodiff(x_ad, u);
    };

    // Compute Jacobian w.r.t. state
    Eigen::MatrixXd Jx = jacobian(dynamics_wrt_x, wrt(x), at(x));
    return Jx;
}

Eigen::MatrixXd DynamicalSystem::getControlJacobian(const Eigen::VectorXd& state,
                                                    const Eigen::VectorXd& control) const {
    VectorXdual2nd x = state;
    VectorXdual2nd u = control;

    auto dynamics_wrt_u = [&](const VectorXdual2nd& u_ad) -> VectorXdual2nd {
        return this->getContinuousDynamicsAutodiff(x, u_ad);
    };

    Eigen::MatrixXd Ju = jacobian(dynamics_wrt_u, wrt(u), at(u));
    return Ju;
}

// --- Autodiff Default Implementations for Hessians ---

std::vector<Eigen::MatrixXd> DynamicalSystem::getStateHessian(const Eigen::VectorXd& state,
                                                             const Eigen::VectorXd& control) const {
    int n = state_dim_;
    int m = control_dim_;
    std::vector<Eigen::MatrixXd> state_hessian_tensor(state_dim_);

    // Create the combined state-control vector using second-order duals
    VectorXdual2nd z(n + m);
    z.head(n) = state;
    z.tail(m) = control;

    // Compute Hessian for each output dimension
    for (int i = 0; i < state_dim_; ++i) {
        // Define a scalar function for the i-th output dimension
        auto f_i = [&](const VectorXdual2nd& z_ad) -> autodiff::dual2nd {
            VectorXdual2nd x_ad = z_ad.head(n);
            VectorXdual2nd u_ad = z_ad.tail(m);
            // Return the i-th component of the dynamics vector
            return this->getContinuousDynamicsAutodiff(x_ad, u_ad)(i);
        };

        // Compute the full Hessian matrix for the i-th output w.r.t z = [x, u]
        Eigen::MatrixXd H_i = hessian(f_i, wrt(z), at(z));

        // Extract the top-left (n x n) block (d^2 f_i / dx^2)
        state_hessian_tensor[i] = H_i.topLeftCorner(n, n);
    }
    return state_hessian_tensor;
}

std::vector<Eigen::MatrixXd> DynamicalSystem::getControlHessian(const Eigen::VectorXd& state,
                                                               const Eigen::VectorXd& control) const {
    int n = state_dim_;
    int m = control_dim_;
    std::vector<Eigen::MatrixXd> control_hessian_tensor(state_dim_);

    VectorXdual2nd z(n + m);
    z.head(n) = state;
    z.tail(m) = control;

    for (int i = 0; i < state_dim_; ++i) {
        auto f_i = [&](const VectorXdual2nd& z_ad) -> autodiff::dual2nd {
            VectorXdual2nd x_ad = z_ad.head(n);
            VectorXdual2nd u_ad = z_ad.tail(m);
            return this->getContinuousDynamicsAutodiff(x_ad, u_ad)(i);
        };
        Eigen::MatrixXd H_i = hessian(f_i, wrt(z), at(z));
        // Extract the bottom-right (m x m) block (d^2 f_i / du^2)
        control_hessian_tensor[i] = H_i.bottomRightCorner(m, m);
    }
    return control_hessian_tensor;
}

std::vector<Eigen::MatrixXd> DynamicalSystem::getCrossHessian(const Eigen::VectorXd& state,
                                                             const Eigen::VectorXd& control) const {
    int n = state_dim_;
    int m = control_dim_;
    std::vector<Eigen::MatrixXd> cross_hessian_tensor(state_dim_);

    VectorXdual2nd z(n + m);
    z.head(n) = state;
    z.tail(m) = control;

    for (int i = 0; i < state_dim_; ++i) {
        auto f_i = [&](const VectorXdual2nd& z_ad) -> autodiff::dual2nd {
            VectorXdual2nd x_ad = z_ad.head(n);
            VectorXdual2nd u_ad = z_ad.tail(m);
            return this->getContinuousDynamicsAutodiff(x_ad, u_ad)(i);
        };
        Eigen::MatrixXd H_i = hessian(f_i, wrt(z), at(z));
        // Extract the bottom-left (m x n) block (d^2 f_i / dudx)
        cross_hessian_tensor[i] = H_i.bottomLeftCorner(m, n);
    }
    return cross_hessian_tensor;
}
