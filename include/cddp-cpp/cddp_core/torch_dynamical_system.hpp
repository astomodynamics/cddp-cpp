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
#ifndef CDDP_TORCH_DYNAMICAL_SYSTEM_HPP
#define CDDP_TORCH_DYNAMICAL_SYSTEM_HPP

#include "cddp_core/dynamical_system.hpp" 
#include <torch/torch.h>
#include <Eigen/Dense>

namespace cddp {

class DynamicsModelInterface : public torch::nn::Module {
public:
    virtual torch::Tensor forward(std::vector<torch::Tensor> inputs) = 0;
    virtual ~DynamicsModelInterface() = default;
};

class TorchDynamicalSystem : public DynamicalSystem {
public:
    TorchDynamicalSystem(int state_dim, 
                        int control_dim, 
                        double timestep,
                        std::string integration_type,
                        std::shared_ptr<DynamicsModelInterface> model,
                        bool use_gpu = false);

    // Override core dynamics methods
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, 
                                         const Eigen::VectorXd& control) const override;

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const override;

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                      const Eigen::VectorXd& control) const override;
                                      
    // Add batch processing capability
    std::vector<Eigen::VectorXd> getBatchDynamics(
        const std::vector<Eigen::VectorXd>& states,
        const std::vector<Eigen::VectorXd>& controls) const;

    // Optional: Override Hessian computations if needed
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control) const override;
    
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control) const override;

private:
    // Helper methods for tensor conversions
    torch::Tensor eigenToTorch(const Eigen::VectorXd& eigen_vec, bool requires_grad = false) const;
    Eigen::VectorXd torchToEigen(const torch::Tensor& tensor) const;

    std::shared_ptr<DynamicsModelInterface> model_;
    bool use_gpu_;
    torch::Device device_;
};

} // namespace cddp

#endif // CDDP_TORCH_DYNAMICAL_SYSTEM_HPP