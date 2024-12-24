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

#include <stdexcept>
#include <iostream>
#include "cddp_core/neural_dynamical_system.hpp"

namespace cddp {

NeuralDynamicalSystem::NeuralDynamicalSystem(int state_dim,
                                             int control_dim,
                                             double timestep,
                                             const std::string& integration_type,
                                             std::shared_ptr<DynamicsModelInterface> model,
                                             torch::Device device)
    : DynamicalSystem(state_dim, control_dim, timestep, integration_type)
    , model_(std::move(model))
    , device_(device)
{
    if (!model_) {
        throw std::runtime_error("NeuralDynamicalSystem: Received null model pointer.");
    }

    // Move model to desired device
    model_->to(device_);

    // Check model dimension
    auto dummy_state   = torch::zeros({1, state_dim_},  torch::kDouble).to(device_);
    auto dummy_control = torch::zeros({1, control_dim_}, torch::kDouble).to(device_);
    auto output        = model_->forward({dummy_state, dummy_control});

    // Expected output to be shape [1, state_dim_]
    if (output.dim() != 2 || output.size(0) != 1 || output.size(1) != state_dim_) {
        throw std::runtime_error(
            "NeuralDynamicalSystem: Model output shape mismatch. "
            "Expected [1, " + std::to_string(state_dim_) + "] but got " +
            std::to_string(output.sizes()[0]) + " x " + std::to_string(output.sizes()[1]));
    }
}

// ----------------------------------------------------------------------------
//                    getContinuousDynamics
// ----------------------------------------------------------------------------
Eigen::VectorXd NeuralDynamicalSystem::getContinuousDynamics(const Eigen::VectorXd& state,
                                                             const Eigen::VectorXd& control) const
{
    // We'll assume the model produces x_dot = f(x,u).
    // Or if it produces next state, adapt accordingly.
    torch::NoGradGuard no_grad;  // We don't need gradient for forward pass

    auto state_tensor   = eigenToTorch(state);     // shape: [1, state_dim_]
    auto control_tensor = eigenToTorch(control);   // shape: [1, control_dim_]
    
    auto output = model_->forward({state_tensor, control_tensor});
    // Convert to Eigen
    return torchToEigen(output);
}

// ----------------------------------------------------------------------------
//                    getDiscreteDynamics
// ----------------------------------------------------------------------------
Eigen::VectorXd NeuralDynamicalSystem::getDiscreteDynamics(const Eigen::VectorXd& state,
                                                           const Eigen::VectorXd& control) const
{
    // A typical approach: x_next = x + x_dot*dt if the model outputs x_dot
    // If your model directly outputs next-state, just return the model result.
    // For demonstration: x_{t+1} = x + dt * f(x, u)
    // Adjust as needed if your model is purely discrete.

    Eigen::VectorXd x_dot = getContinuousDynamics(state, control);
    return state + x_dot * timestep_;
}

// ----------------------------------------------------------------------------
//                           getStateJacobian
// ----------------------------------------------------------------------------
Eigen::MatrixXd NeuralDynamicalSystem::getStateJacobian(const Eigen::VectorXd& state,
                                                        const Eigen::VectorXd& control) const
{
    // Placeholder approach #1: Identity, as a quick stub
    // return Eigen::MatrixXd::Identity(state_dim_, state_dim_);

    // Placeholder approach #2: zero
    // return Eigen::MatrixXd::Zero(state_dim_, state_dim_);

    // Real approach: use finite difference or PyTorch autograd.
    // For illustration, let's do a naive finite-difference:
    const double eps = 1e-6;
    Eigen::MatrixXd A(state_dim_, state_dim_);

    // Baseline
    Eigen::VectorXd f0 = getContinuousDynamics(state, control);

    for (int i = 0; i < state_dim_; ++i) {
        Eigen::VectorXd perturbed = state;
        perturbed(i) += eps;

        Eigen::VectorXd f_pert = getContinuousDynamics(perturbed, control);
        A.col(i) = (f_pert - f0) / eps;
    }
    return A;
}

// ----------------------------------------------------------------------------
//                           getControlJacobian
// ----------------------------------------------------------------------------
Eigen::MatrixXd NeuralDynamicalSystem::getControlJacobian(const Eigen::VectorXd& state,
                                                          const Eigen::VectorXd& control) const
{
    // Similar naive finite-difference:
    const double eps = 1e-6;
    Eigen::MatrixXd B(state_dim_, control_dim_);

    // Baseline
    Eigen::VectorXd f0 = getContinuousDynamics(state, control);

    for (int j = 0; j < control_dim_; ++j) {
        Eigen::VectorXd ctrl_pert = control;
        ctrl_pert(j) += eps;

        Eigen::VectorXd f_pert = getContinuousDynamics(state, ctrl_pert);
        B.col(j) = (f_pert - f0) / eps;
    }
    return B;
}

// ----------------------------------------------------------------------------
//                         Hessians (placeholders)
// ----------------------------------------------------------------------------
Eigen::MatrixXd NeuralDynamicalSystem::getStateHessian(const Eigen::VectorXd& /*state*/,
                                                       const Eigen::VectorXd& /*control*/) const
{
    // Typically a 3D object. Return zero or implement if needed.
    return Eigen::MatrixXd::Zero(state_dim_ * state_dim_, state_dim_);
}

Eigen::MatrixXd NeuralDynamicalSystem::getControlHessian(const Eigen::VectorXd& /*state*/,
                                                         const Eigen::VectorXd& /*control*/) const
{
    return Eigen::MatrixXd::Zero(control_dim_ * control_dim_, state_dim_);
}

Eigen::MatrixXd NeuralDynamicalSystem::getCrossHessian(const Eigen::VectorXd& /*state*/,
                                                       const Eigen::VectorXd& /*control*/) const
{
    return Eigen::MatrixXd::Zero(state_dim_ * control_dim_, state_dim_);
}

// ----------------------------------------------------------------------------
//                          eigenToTorch / torchToEigen
// ----------------------------------------------------------------------------
torch::Tensor NeuralDynamicalSystem::eigenToTorch(const Eigen::VectorXd& eigen_vec,
                                                  bool requires_grad) const
{
    // Shape [1, size]
    // If you prefer shape [size] without batch-dim, adjust accordingly.
    auto tensor = torch::from_blob(
        const_cast<double*>(eigen_vec.data()),
        {1, static_cast<long>(eigen_vec.size())},
        torch::TensorOptions().dtype(torch::kDouble)
    ).clone(); // .clone() to own memory

    // Move to device
    tensor = tensor.to(device_);

    // optionally set requires_grad
    if (requires_grad) {
        tensor.set_requires_grad(true);
    }
    return tensor;
}

Eigen::VectorXd NeuralDynamicalSystem::torchToEigen(const torch::Tensor& tensor) const
{
    // Expect shape [1, state_dim_] or [1, control_dim_]. We'll read out the second dimension.
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    if (cpu_tensor.dim() != 2) {
        throw std::runtime_error("torchToEigen: expected a 2D tensor with batch-dim");
    }
    auto rows = cpu_tensor.size(0);
    auto cols = cpu_tensor.size(1);

    // For a single sample, rows == 1. We'll produce an Eigen vector of length = cols.
    Eigen::VectorXd eigen_vec(cols);
    std::memcpy(eigen_vec.data(), cpu_tensor.data_ptr<double>(), sizeof(double)*cols);
    return eigen_vec;
}

} // namespace cddp
