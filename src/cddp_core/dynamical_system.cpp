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

#include "cddp_core/dynamical_system.hpp" 

using namespace cddp;

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

// TODO: Create a finite difference method for not only dynamics but also objective and constraint
Eigen::MatrixXd DynamicalSystem::getFiniteDifferenceStateJacobian(const Eigen::VectorXd& x, 
                                                            const Eigen::VectorXd& u) const {
    const int n = x.size();
    Eigen::MatrixXd J(n, n);
    Eigen::VectorXd f0 = getContinuousDynamics(x, u);
    
    const double h = 1e-6; // Small perturbation value
    Eigen::VectorXd x_perturbed = x;
    for (int i = 0; i < n; ++i) {
        x_perturbed(i) += h;
        J.col(i) = (getContinuousDynamics(x_perturbed, u) - f0) / h;
        x_perturbed(i) = x(i);
    }
    return J;
}

Eigen::MatrixXd DynamicalSystem::getFiniteDifferenceControlJacobian(const Eigen::VectorXd& x, 
                                                            const Eigen::VectorXd& u) const {
    const int n = x.size();
    const int m = u.size();
    Eigen::MatrixXd J(n, m);
    Eigen::VectorXd f0 = getContinuousDynamics(x, u);
    
    const double h = 1e-6; // Small perturbation value
    Eigen::VectorXd u_perturbed = u;
    for (int i = 0; i < m; ++i) {
        u_perturbed(i) += h;
        J.col(i) = (getContinuousDynamics(x, u_perturbed) - f0) / h;
        u_perturbed(i) = u(i);
    }
    return J;
}