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

#ifndef CDDP_TORCH_OBJECTIVE_HPP
#define CDDP_TORCH_OBJECTIVE_HPP

#include <torch/torch.h>
#include <Eigen/Dense>
#include "objective.hpp"

namespace cddp {

/**
 * @brief TorchObjective class for integrating PyTorch-based computations into the CDDP framework.
 */
class TorchObjective : public Objective {
public:
    /**
     * @brief Constructor for TorchObjective.
     *
     * @param model A user-defined PyTorch module that computes costs given state/control.
     *        This model should implement a forward method that can compute running and terminal costs.
     * @param state_dim Dimension of the state vector.
     * @param control_dim Dimension of the control vector.
     * @param timestep The time step used for cost scaling.
     * @param device The torch device (CPU/GPU) to perform computations on.
     */
    TorchObjective(torch::nn::Module& model,
                   int state_dim,
                   int control_dim,
                   double timestep = 0.1,
                   torch::Device device = torch::kCPU)
    : model_(model),
      state_dim_(state_dim),
      control_dim_(control_dim),
      timestep_(timestep),
      device_(device)
    {
        // Ensure the model is in eval mode
        model_->eval();
        // Optionally, we could move the model to the specified device here:
        model_->to(device_);
    }

    /**
     * @brief Evaluate the total cost over a given trajectory.
     *
     * @param states Vector of state vectors over the trajectory.
     * @param controls Vector of control vectors over the trajectory.
     * @return Total cost as a double.
     */
    double evaluate(const std::vector<Eigen::VectorXd>& states,
                    const std::vector<Eigen::VectorXd>& controls) const override {
        double total_cost = 0.0;
        for (int i = 0; i < static_cast<int>(controls.size()); ++i) {
            total_cost += running_cost(states[i], controls[i], i);
        }
        // Terminal cost
        total_cost += terminal_cost(states.back());
        return total_cost;
    }

    /**
     * @brief Compute the running cost at a single timestep.
     *
     * @param state Current state vector.
     * @param control Current control vector.
     * @param index Time index in the trajectory.
     * @return Running cost value as a double.
     */
    double running_cost(const Eigen::VectorXd& state,
                        const Eigen::VectorXd& control,
                        int index) const override {
        auto state_tensor = eigenToTensor(state);
        auto control_tensor = eigenToTensor(control);
        torch::NoGradGuard no_grad;
        auto cost = callRunningCostModel(state_tensor, control_tensor, index);
        return cost.item<double>() * timestep_;
    }

    /**
     * @brief Compute the terminal cost for the final state.
     *
     * @param final_state Final state vector.
     * @return Terminal cost value as a double.
     */
    double terminal_cost(const Eigen::VectorXd& final_state) const override {
        auto state_tensor = eigenToTensor(final_state);
        torch::NoGradGuard no_grad;
        auto cost = callTerminalCostModel(state_tensor);
        return cost.item<double>();
    }

    /**
     * @brief Compute the gradient of the running cost w.r.t state.
     *
     * @param state Current state vector.
     * @param control Current control vector.
     * @param index Time index.
     * @return Gradient vector w.r.t state.
     */
    Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& control,
                                                int index) const override {
        return computeGradient(state, control, index, /*wrt_state=*/true, /*terminal=*/false);
    }

    /**
     * @brief Compute the gradient of the running cost w.r.t control.
     *
     * @param state Current state vector.
     * @param control Current control vector.
     * @param index Time index.
     * @return Gradient vector w.r.t control.
     */
    Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control,
                                                  int index) const override {
        return computeGradient(state, control, index, /*wrt_state=*/false, /*terminal=*/false);
    }

    /**
     * @brief Compute the gradient of the final cost w.r.t state.
     *
     * @param final_state Final state vector.
     * @return Gradient vector w.r.t final state.
     */
    Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const override {
        Eigen::VectorXd dummy_control; // not needed for terminal cost
        return computeGradient(final_state, dummy_control, /*index=*/0, /*wrt_state=*/true, /*terminal=*/true);
    }

    /**
     * @brief Compute the Hessian of the running cost w.r.t the state.
     *
     * @param state Current state vector.
     * @param control Current control vector.
     * @param index Time index.
     * @return Hessian matrix w.r.t state.
     */
    Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control,
                                               int index) const override {
        return computeHessian(state, control, index, /*wrt_state_state=*/true, /*wrt_control_control=*/false, /*wrt_state_control=*/false, /*terminal=*/false);
    }

    /**
     * @brief Compute the Hessian of the running cost w.r.t the control.
     *
     * @param state Current state vector.
     * @param control Current control vector.
     * @param index Time index.
     * @return Hessian matrix w.r.t control.
     */
    Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state,
                                                 const Eigen::VectorXd& control,
                                                 int index) const override {
        return computeHessian(state, control, index, /*wrt_state_state=*/false, /*wrt_control_control=*/true, /*wrt_state_control=*/false, /*terminal=*/false);
    }

    /**
     * @brief Compute the cross Hessian of the running cost (d^2l/dxdu).
     *
     * @param state Current state vector.
     * @param control Current control vector.
     * @param index Time index.
     * @return Hessian matrix w.r.t x and u.
     */
    Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control,
                                               int index) const override {
        return computeHessian(state, control, index, /*wrt_state_state=*/false, /*wrt_control_control=*/false, /*wrt_state_control=*/true, /*terminal=*/false);
    }

    /**
     * @brief Compute the Hessian of the final cost w.r.t the state.
     *
     * @param final_state Final state vector.
     * @return Hessian matrix w.r.t final state.
     */
    Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const override {
        Eigen::VectorXd dummy_control; // not used for terminal cost
        return computeHessian(final_state, dummy_control, /*index=*/0, /*wrt_state_state=*/true, /*wrt_control_control=*/false, /*wrt_state_control=*/false, /*terminal=*/true);
    }

    // Accessors & mutators for reference states if needed
    const Eigen::VectorXd& getReferenceState() const override { return reference_state_; }
    const std::vector<Eigen::VectorXd>& getReferenceStates() const override { return reference_states_; }

    void setReferenceState(const Eigen::VectorXd& reference_state) override {
        reference_state_ = reference_state;
    }

    void setReferenceStates(const std::vector<Eigen::VectorXd>& reference_states) override {
        reference_states_ = reference_states;
    }

private:
    // Convert Eigen vector to torch tensor and move to device
    torch::Tensor eigenToTensor(const Eigen::VectorXd& vec) const {
        if (vec.size() == 0) {
            return torch::empty({0}, torch::dtype(torch::kDouble).device(device_));
        }
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble).device(device_);
        auto tensor = torch::from_blob((double*)vec.data(), {vec.size()}, options).clone();
        return tensor;
    }

    // Call model for running cost
    torch::Tensor callRunningCostModel(const torch::Tensor& state,
                                       const torch::Tensor& control,
                                       int index) const {
        // Assumes model has forward method: forward(state, control, index, terminal=false)
        return model_->forward({state, control, torch::tensor(index, torch::kInt64).to(device_), torch::tensor(false, torch::kBool).to(device_)})[0];
    }

    // Call model for terminal cost
    torch::Tensor callTerminalCostModel(const torch::Tensor& state) const {
        auto dummy_control = torch::zeros({control_dim_}, torch::dtype(torch::kDouble).device(device_));
        return model_->forward({state, dummy_control, torch::tensor(0, torch::kInt64).to(device_), torch::tensor(true, torch::kBool).to(device_)})[0];
    }

    /**
     * @brief Compute gradient w.r.t state or control using PyTorch autograd.
     *
     * @param state Eigen state vector.
     * @param control Eigen control vector (ignored if terminal is true).
     * @param index Time index.
     * @param wrt_state If true, compute gradient w.r.t state; otherwise w.r.t control.
     * @param terminal If true, compute gradient of terminal cost; otherwise running cost.
     * @return Gradient as an Eigen vector.
     */
    Eigen::VectorXd computeGradient(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control,
                                    int index,
                                    bool wrt_state,
                                    bool terminal) const {
        // Prepare inputs
        auto state_tensor = eigenToTensor(state).clone().detach().requires_grad_(wrt_state);
        torch::Tensor control_tensor;
        if (!terminal) {
            control_tensor = eigenToTensor(control).clone().detach().requires_grad_(!wrt_state);
        } else {
            control_tensor = torch::zeros({control_dim_}, torch::dtype(torch::kDouble).device(device_));
        }

        // Compute cost
        auto out = terminal ? 
            model_->forward({state_tensor, control_tensor, torch::tensor(0, torch::kInt64).to(device_), torch::tensor(true, torch::kBool).to(device_)}) :
            model_->forward({state_tensor, control_tensor, torch::tensor(index, torch::kInt64).to(device_), torch::tensor(false, torch::kBool).to(device_)});

        torch::Tensor cost = out[0];
        cost.backward();

        if (wrt_state) {
            auto grad = state_tensor.grad();
            return tensorToEigen(grad);
        } else {
            auto grad = control_tensor.grad();
            return tensorToEigen(grad);
        }
    }

    /**
     * @brief Compute Hessian using second-order derivatives.
     *
     * Currently a placeholder. A production version should use finite-differences on `computeGradient`
     * or implement second-order differentiation. For now, returns zero matrices.
     */
    Eigen::MatrixXd computeHessian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control,
                                   int index,
                                   bool wrt_state_state,
                                   bool wrt_control_control,
                                   bool wrt_state_control,
                                   bool terminal) const {
        int nx = state_dim_;
        int nu = control_dim_;

        if (wrt_state_state) {
            return Eigen::MatrixXd::Zero(nx, nx);
        } else if (wrt_control_control) {
            return Eigen::MatrixXd::Zero(nu, nu);
        } else if (wrt_state_control) {
            return Eigen::MatrixXd::Zero(nx, nu);
        } else {
            return Eigen::MatrixXd(0,0);
        }
    }

    // Convert torch tensor to Eigen vector
    Eigen::VectorXd tensorToEigen(const torch::Tensor& tensor) const {
        // Assumes tensor is 1D
        Eigen::VectorXd vec(tensor.size(0));
        auto accessor = tensor.accessor<double,1>();
        for (int i = 0; i < vec.size(); ++i) {
            vec(i) = accessor[i];
        }
        return vec;
    }

private:
    torch::nn::Module& model_;              ///< Reference to a user-defined PyTorch model/module
    int state_dim_;                         ///< State dimension
    int control_dim_;                       ///< Control dimension
    double timestep_;                       ///< Time step for cost scaling
    torch::Device device_;                  ///< Torch device to run computations on (CPU/GPU)

    // Optional reference states
    Eigen::VectorXd reference_state_;
    std::vector<Eigen::VectorXd> reference_states_;
};

} // namespace cddp

#endif // CDDP_TORCH_OBJECTIVE_HPP
