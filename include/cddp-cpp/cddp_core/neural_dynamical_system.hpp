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

#ifndef CDDP_NEURAL_DYNAMICAL_SYSTEM_HPP
#define CDDP_NEURAL_DYNAMICAL_SYSTEM_HPP

#include <torch/torch.h>
#include <Eigen/Dense>
#include <memory>
#include <string>
#include "cddp_core/dynamical_system.hpp"

namespace cddp {
/**
 * @brief Interface for a neural network model representing system dynamics.
 */

class DynamicsModelInterface : public torch::nn::Module {
public:
    virtual torch::Tensor forward(std::vector<torch::Tensor> inputs) = 0;
    virtual ~DynamicsModelInterface() = default;
};

/**
 * @brief A NeuralDynamicalSystem that uses a PyTorch model to represent system dynamics.
 */
class NeuralDynamicalSystem : public DynamicalSystem {
public:
    /**
     * @brief Construct a new NeuralDynamicalSystem object
     * 
     * @param state_dim Dimension of the system state
     * @param control_dim Dimension of the system control
     * @param timestep Integration timestep
     * @param integration_type Type of numerical integration (Euler, Heun, RK3, RK4)
     * @param model A torch::nn::Module (e.g. an MLP) representing the learned dynamics,
     * @param device Device to run the model on (CPU or CUDA)
     */
    NeuralDynamicalSystem(int state_dim,
                          int control_dim,
                          double timestep,
                          const std::string& integration_type,
                          std::shared_ptr<DynamicsModelInterface> model,
                          torch::Device device = torch::kCPU);

    /**
     * @brief Compute continuous-time dynamics: x_dot = f(x, u).
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::VectorXd Continuous-time derivative of state
     */
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) const override;

    /**
     * @brief Compute discrete-time dynamics: x_{t+1} = f(x_t, u_t).
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::VectorXd Discrete next state
     */
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) const override;

    /**
     * @brief Jacobian of the dynamics w.r.t. state: df/dx
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::MatrixXd Jacobian df/dx
     */
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control) const override;

    /**
     * @brief Jacobian of the dynamics w.r.t. control: df/du
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::MatrixXd Jacobian df/du
     */
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control) const override;

    /**
     * @brief Hessian of the dynamics w.r.t. state (flattened or block representation).
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::MatrixXd Hessian d^2f/dx^2
     */
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control) const override;

    /**
     * @brief Hessian of the dynamics w.r.t. control (flattened or block representation).
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::MatrixXd Hessian d^2f/du^2
     */
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control) const override;

    /**
     * @brief Hessian of the dynamics w.r.t. state and control (flattened or block representation).
     *
     * @param state Current state (Eigen vector)
     * @param control Current control (Eigen vector)
     * @return Eigen::MatrixXd Hessian d^2f/dudx
     */
    Eigen::MatrixXd getCrossHessian(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control) const override;

private:
    std::shared_ptr<DynamicsModelInterface> model_;
    torch::Device device_;
    
    // Helper methods for tensor conversions
    torch::Tensor eigenToTorch(const Eigen::VectorXd& eigen_vec, bool requires_grad = false) const;
    Eigen::VectorXd torchToEigen(const torch::Tensor& tensor) const;
};
} // namespace cddp

#endif // CDDP_NEURAL_DYNAMICAL_SYSTEM_HPP
