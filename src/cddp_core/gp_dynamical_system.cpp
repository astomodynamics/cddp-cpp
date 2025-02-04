/*
 * gp_dynamical_system.cpp
 *
 * Copyright 2024 Tomo Sasaki
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * This file implements a Gaussian Process based dynamical system.
 */

#include "cddp_core/gp_dynamical_system.hpp"
#include "cddp_core/helper.hpp"  // For finite_difference_jacobian(…) and other helpers.
#include <iostream>
#include <cmath>

namespace cddp {

GaussianProcessDynamics::GaussianProcessDynamics(int state_dim, int control_dim, double timestep,
                                                 std::string integration_type,
                                                 bool is_continuous,
                                                 double length_scale,
                                                 double signal_variance,
                                                 double noise_variance)
    : DynamicalSystem(state_dim, control_dim, timestep, integration_type),
      is_continuous_(is_continuous),
      length_scale_(length_scale),
      signal_variance_(signal_variance),
      noise_variance_(noise_variance),
      trained_(false)
{
}

GaussianProcessDynamics::~GaussianProcessDynamics() {
    // Nothing to clean up
}

void GaussianProcessDynamics::train(const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& Y_train) {
    // Store training data
    X_train_ = X_train;
    Y_train_ = Y_train;
    const int n = static_cast<int>(X_train_.rows());
    
    // Compute the kernel matrix K (n x n)
    Eigen::MatrixXd K(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            K(i,j) = kernel(X_train_.row(i), X_train_.row(j));
        }
    }
    // Add noise variance on the diagonal
    K += noise_variance_ * Eigen::MatrixXd::Identity(n, n);
    
    // Compute the inverse (for large n, consider using Cholesky factorization)
    K_inv_ = K.inverse();
    trained_ = true;
    std::cout << "GP training complete. Inverted kernel matrix computed." << std::endl;
}

double GaussianProcessDynamics::kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const {
    // Squared Euclidean distance
    double sqdist = (x1 - x2).squaredNorm();
    return signal_variance_ * std::exp(-0.5 * sqdist / (length_scale_ * length_scale_));
}

Eigen::VectorXd GaussianProcessDynamics::computeKernelVector(const Eigen::VectorXd& x) const {
    const int n = static_cast<int>(X_train_.rows());
    Eigen::VectorXd k(n);
    for (int i = 0; i < n; ++i) {
        k(i) = kernel(x, X_train_.row(i));
    }
    return k;
}

Eigen::VectorXd GaussianProcessDynamics::predict(const Eigen::VectorXd& x) const {
    if (!trained_) {
        throw std::runtime_error("GP model has not been trained (no training data available).");
    }
    // Compute the kernel vector k(x) (n x 1)
    Eigen::VectorXd k = computeKernelVector(x);
    // Predictive mean: μ(x) = k^T * K_inv * Y_train.
    // (Y_train_ is assumed to be of size (n x output_dim))
    Eigen::VectorXd prediction = (k.transpose() * K_inv_ * Y_train_).transpose();
    return prediction;
}

Eigen::VectorXd GaussianProcessDynamics::getContinuousDynamics(const Eigen::VectorXd& state,
                                                                 const Eigen::VectorXd& control) const {
    Eigen::VectorXd input(state.size() + control.size());
    input << state, control;
    
    if (is_continuous_) {
        return predict(input);
    } else {
        Eigen::VectorXd x_next = predict(input);
        return (x_next - state) / timestep_;
    }
}

Eigen::VectorXd GaussianProcessDynamics::getDiscreteDynamics(const Eigen::VectorXd& state,
                                                               const Eigen::VectorXd& control) const {

    if (integration_type_ == "euler") {
        return euler_step(state, control, timestep_);
    } else if (integration_type_ == "heun") {
        return heun_step(state, control, timestep_);
    } else if (integration_type_ == "rk3") {
        return rk3_step(state, control, timestep_);
    } else if (integration_type_ == "rk4") {
        return rk4_step(state, control, timestep_);
    } else {
        std::cerr << "Integration type '" << integration_type_ << "' not supported in GP dynamics!" << std::endl;
        return Eigen::VectorXd::Zero(state.size());
    }
}

//
// Jacobian and Hessian approximations via finite differences
//
Eigen::MatrixXd GaussianProcessDynamics::getStateJacobian(const Eigen::VectorXd& state,
                                                            const Eigen::VectorXd& control) const {
    auto f = [this, &control](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        return this->getDiscreteDynamics(x, control);
    };
    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd GaussianProcessDynamics::getControlJacobian(const Eigen::VectorXd& state,
                                                              const Eigen::VectorXd& control) const {
    auto f = [this, &state](const Eigen::VectorXd& u) -> Eigen::VectorXd {
        return this->getDiscreteDynamics(state, u);
    };
    return finite_difference_jacobian(f, control);
}

Eigen::MatrixXd GaussianProcessDynamics::getStateHessian(const Eigen::VectorXd& state,
                                                         const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(state.size() * state.size(), state.size());
}

Eigen::MatrixXd GaussianProcessDynamics::getControlHessian(const Eigen::VectorXd& state,
                                                           const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(state.size() * control.size(), control.size());
}

Eigen::MatrixXd GaussianProcessDynamics::getCrossHessian(const Eigen::VectorXd& state,
                                                         const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(state.size() * control.size(), state.size());
}

} // namespace cddp
