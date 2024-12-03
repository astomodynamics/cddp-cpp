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

#include "cddp_core/torch_dynamical_system.hpp"

namespace cddp {

TorchDynamicalSystem::TorchDynamicalSystem(
    int state_dim, 
    int control_dim, 
    double timestep,
    std::string integration_type,
    std::shared_ptr<DynamicsModelInterface> model,
    bool use_gpu)
    : DynamicalSystem(state_dim, control_dim, timestep, integration_type),
      model_(model),
      use_gpu_(use_gpu && torch::cuda::is_available()),
      device_(use_gpu_ ? torch::kCUDA : torch::kCPU) {
    
    if (use_gpu_ != use_gpu) {
        std::cout << "Warning: GPU requested but not available. Using CPU instead." << std::endl;
    }
    
    if (use_gpu_) {
        model_->to(device_);
    }
}

torch::Tensor TorchDynamicalSystem::eigenToTorch(
    const Eigen::VectorXd& eigen_vec, 
    bool requires_grad) const {
    
    auto tensor = torch::from_blob(
        const_cast<double*>(eigen_vec.data()),
        {1, eigen_vec.size()},
        torch::kFloat64
    ).clone().to(device_);  // Move to correct device immediately after creation
    
    tensor.requires_grad_(requires_grad);
    return tensor;
}

Eigen::VectorXd TorchDynamicalSystem::torchToEigen(
    const torch::Tensor& tensor) const {
    
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();
    Eigen::VectorXd eigen_vec(cpu_tensor.size(1));
    std::memcpy(eigen_vec.data(), 
                cpu_tensor.data_ptr<double>(), 
                sizeof(double) * cpu_tensor.numel());
    return eigen_vec;
}

Eigen::VectorXd TorchDynamicalSystem::getContinuousDynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
    
    torch::NoGradGuard no_grad;
    
    auto state_tensor = eigenToTorch(state);
    auto control_tensor = eigenToTorch(control);
    
    auto output = model_->forward({state_tensor, control_tensor});
    return torchToEigen(output);
}

Eigen::MatrixXd TorchDynamicalSystem::getStateJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
    
    auto state_tensor = eigenToTorch(state, true);  // requires_grad = true
    auto control_tensor = eigenToTorch(control);
    
    auto output = model_->forward({state_tensor, control_tensor});
    
    Eigen::MatrixXd jacobian(state_dim_, state_dim_);
    for (int i = 0; i < output.size(1); i++) {
        auto grad = torch::autograd::grad(
            {output[0][i]},
            {state_tensor},
            {},
            true,
            true)[0];
        
        Eigen::VectorXd grad_vec = torchToEigen(grad);
        jacobian.row(i) = grad_vec;
    }
    
    return jacobian;
}

Eigen::MatrixXd TorchDynamicalSystem::getControlJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
    
    auto state_tensor = eigenToTorch(state);
    auto control_tensor = eigenToTorch(control, true);  // requires_grad = true
    
    auto output = model_->forward({state_tensor, control_tensor});
    
    Eigen::MatrixXd jacobian(state_dim_, control_dim_);
    for (int i = 0; i < output.size(1); i++) {
        auto grad = torch::autograd::grad(
            {output[0][i]},
            {control_tensor},
            {},
            true,
            true)[0];
        
        Eigen::VectorXd grad_vec = torchToEigen(grad);
        jacobian.row(i) = grad_vec;
    }
    
    return jacobian;
}

std::vector<Eigen::VectorXd> TorchDynamicalSystem::getBatchDynamics(
    const std::vector<Eigen::VectorXd>& states,
    const std::vector<Eigen::VectorXd>& controls) const {
    
    torch::NoGradGuard no_grad;
    
    // Create batched tensors
    auto state_batch = torch::zeros(
        {static_cast<long>(states.size()), state_dim_},
        torch::kFloat64
    ).to(device_);
    
    auto control_batch = torch::zeros(
        {static_cast<long>(controls.size()), control_dim_},
        torch::kFloat64
    ).to(device_);
    
    // Copy data to tensors
    for (size_t i = 0; i < states.size(); ++i) {
        state_batch[i] = eigenToTorch(states[i])[0];
        control_batch[i] = eigenToTorch(controls[i])[0];
    }
    
    // Forward pass
    auto output_batch = model_->forward({state_batch, control_batch});
    
    // Convert back to vector of Eigen vectors
    std::vector<Eigen::VectorXd> results;
    results.reserve(states.size());
    
    for (int i = 0; i < output_batch.size(0); ++i) {
        results.push_back(torchToEigen(output_batch[i].unsqueeze(0)));
    }
    
    return results;
}

// Optional: Implement Hessian computations if needed
Eigen::MatrixXd TorchDynamicalSystem::getStateHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
    // TODO: Implement second-order derivatives if needed
    return Eigen::MatrixXd::Zero(state_dim_ * state_dim_, state_dim_);
}

Eigen::MatrixXd TorchDynamicalSystem::getControlHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
    // TODO: Implement second-order derivatives if needed
    return Eigen::MatrixXd::Zero(state_dim_ * control_dim_, control_dim_);
}

} // namespace cddp