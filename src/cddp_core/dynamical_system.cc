#include <iostream>
#include <Eigen/Dense>

#include "cddp-cpp/cddp_core/dynamical_system.h" 

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

// Implement the core dynamics function
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