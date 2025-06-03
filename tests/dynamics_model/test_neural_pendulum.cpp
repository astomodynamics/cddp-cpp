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
#include <vector>
#include <filesystem>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp" 
#include "cddp_core/neural_dynamical_system.hpp"  

using namespace cddp;

class PendulumModel : public DynamicsModelInterface {
public:
    PendulumModel(double length = 1.0, double mass = 1.0, double damping = 0.0);

    torch::Tensor forward(std::vector<torch::Tensor> inputs) override;

private:
    void initialize_weights();

    double length_, mass_, damping_;
    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr};
    torch::Device device_;
};

PendulumModel::PendulumModel(double length, double mass, double damping)
    : length_(length), mass_(mass), damping_(damping),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    // Create linear layers
    linear1 = register_module("linear1", torch::nn::Linear(3, 32));
    linear2 = register_module("linear2", torch::nn::Linear(32, 32));
    linear3 = register_module("linear3", torch::nn::Linear(32, 2));

    // Move to device and set dtype
    this->to(device_);
    if (device_.is_cpu()) {
        this->to(torch::kFloat64);
    } else {
        this->to(torch::kFloat32);
    }
    initialize_weights();
}

void PendulumModel::initialize_weights()
{
    torch::NoGradGuard no_grad;

    double angle_scale = 0.1;
    double velocity_scale = 0.1;

    // Example: manually set a few weights in linear3
    auto w = linear3->weight.data();
    w[0][0] = angle_scale;
    w[1][1] = velocity_scale;
    linear3->weight.data() = w;

    auto b = linear3->bias.data();
    b[0] = 0.0;
    b[1] = -9.81 / length_ * angle_scale; // a quick guess
    linear3->bias.data() = b;
}

torch::Tensor PendulumModel::forward(std::vector<torch::Tensor> inputs)
{
    // Expect 2 inputs: [state, control]
    auto state = inputs[0].to(device_);
    auto control = inputs[1].to(device_);

    if (device_.is_cuda()) {
        // If on GPU, we use float32
        state = state.to(torch::kFloat32);
        control = control.to(torch::kFloat32);
    }

    // Concatenate along dim=1 if shapes are [batch_size, 2] and [batch_size, 1]
    auto x = torch::cat({state, control}, /*dim=*/1);
    x = torch::tanh(linear1(x));
    x = torch::tanh(linear2(x));
    x = linear3(x);

    if (device_.is_cuda()) {
        x = x.to(torch::kFloat64); // move back to double if desired
    }
    return x;
}

void printTensorInfo(const torch::Tensor& tensor, const std::string& name)
{
    std::cout << name << ":\n"
              << " - shape: [" << tensor.sizes() << "]\n"
              << " - dtype: " << tensor.dtype() << "\n"
              << " - device: " << tensor.device() << "\n"
              << " - values: " << tensor << "\n" << std::endl;
}

void printVectorInfo(const Eigen::VectorXd& vec, const std::string& name)
{
    std::cout << name << ":\n"
              << " - size: " << vec.size() << "\n"
              << " - values: " << vec.transpose() << "\n" << std::endl;
}

//-----------------------------------------//
//        TEST SUITE: TorchPendulumTest
//-----------------------------------------//
TEST(TorchPendulumTest, DiscreteDynamics)
{
    // Parameters
    double timestep = 0.01;
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.0;
    std::string integration_type = "rk4";

    cddp::Pendulum analytical_pendulum(timestep, length, mass, damping, integration_type);

    auto model = std::make_shared<PendulumModel>(length, mass, damping);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    NeuralDynamicalSystem torch_pendulum(/*state_dim=*/2, /*control_dim=*/1,
                                        timestep, integration_type, model, device);

    // Create a test state/control
    Eigen::VectorXd test_state(2);
    test_state << M_PI / 4, 0.0; // 45 deg, no velocity
    Eigen::VectorXd test_control(1);
    test_control << 0.0;

    // Analytical next state
    auto analytical_next = analytical_pendulum.getDiscreteDynamics(test_state, test_control, 0.0);
    printVectorInfo(analytical_next, "Analytical Next State");

    // Torch next state
    auto torch_next = torch_pendulum.getDiscreteDynamics(test_state, test_control, 0.0);
    printVectorInfo(torch_next, "Torch Next State");

    // Compare errors
    double error = (analytical_next - torch_next).norm();
    std::cout << "L2 error: " << error << std::endl;

    // Basic tests
    ASSERT_EQ(torch_pendulum.getStateDim(), 2);
    ASSERT_EQ(torch_pendulum.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(torch_pendulum.getTimestep(), timestep);
    ASSERT_EQ(torch_pendulum.getIntegrationType(), integration_type);

    // Check if the error is within a tolerance
    // (This is arbitrary; you may need a looser or tighter tolerance)
    EXPECT_NEAR(error, 0.0, 0.1) 
        << "Discrete dynamics: model deviance too large";

    // Test Jacobians
    auto A = torch_pendulum.getStateJacobian(test_state, test_control, 0.0);
    auto B = torch_pendulum.getControlJacobian(test_state, test_control, 0.0);

    std::cout << "State Jacobian A:\n" << A << std::endl;
    std::cout << "Control Jacobian B:\n" << B << std::endl;

    // Verify shapes
    ASSERT_EQ(A.rows(), 2);
    ASSERT_EQ(A.cols(), 2);
    ASSERT_EQ(B.rows(), 2);
    ASSERT_EQ(B.cols(), 1);
}

/**
 * @brief Demonstrates a small training loop that tries to fit the Torch model
 * to data from an analytical pendulum's discrete dynamics.
 */
TEST(NeuralPendulumTest, Training)
{
    // Parameters
    double timestep = 0.01;
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.1;

    // Create an analytical pendulum and a Torch model
    cddp::Pendulum analytical_pendulum(timestep, length, mass, damping, "rk4");
    auto model = std::make_shared<PendulumModel>(length, mass, damping);
    bool use_gpu = torch::cuda::is_available();

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    int num_epochs = 100;
    int batch_size = 32;

    // Generate training data
    int num_samples = 1000;
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto state_tensor = torch::zeros({num_samples, 2}, cpu_options);
    auto control_tensor = torch::zeros({num_samples, 1}, cpu_options);
    auto next_state_tensor = torch::zeros({num_samples, 2}, cpu_options);

    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd state(2);
        state << (2.0 * rand() / RAND_MAX - 1.0) * M_PI,
                 (2.0 * rand() / RAND_MAX - 1.0) * 5.0;
        Eigen::VectorXd control(1);
        control << (2.0 * rand() / RAND_MAX - 1.0) * 2.0;

        auto next_state = analytical_pendulum.getDiscreteDynamics(state, control, 0.0);

        // Copy to torch Tensors
        state_tensor[i] = torch::from_blob(state.data(), {2}, cpu_options).clone();
        control_tensor[i] = torch::from_blob(control.data(), {1}, cpu_options).clone();
        next_state_tensor[i] = torch::from_blob(next_state.data(), {2}, cpu_options).clone();
    }

    // Move to GPU if available
    torch::Device device(use_gpu ? torch::kCUDA : torch::kCPU);
    if (use_gpu) {
        state_tensor = state_tensor.to(device).to(torch::kFloat32);
        control_tensor = control_tensor.to(device).to(torch::kFloat32);
        next_state_tensor = next_state_tensor.to(device).to(torch::kFloat32);
    } else {
        state_tensor = state_tensor.to(device);
        control_tensor = control_tensor.to(device);
        next_state_tensor = next_state_tensor.to(device);
    }

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        int num_batches = num_samples / batch_size;

        for (int batch = 0; batch < num_batches; ++batch) {
            auto indices = torch::randperm(num_samples, 
                torch::TensorOptions().dtype(torch::kLong).device(device));
            indices = indices.slice(0, 0, batch_size);

            auto batch_states = state_tensor.index_select(0, indices);
            auto batch_controls = control_tensor.index_select(0, indices);
            auto batch_next_states = next_state_tensor.index_select(0, indices);

            optimizer.zero_grad();
            auto pred_next_states = model->forward({batch_states, batch_controls});
            if (use_gpu) {
                pred_next_states = pred_next_states.to(torch::kFloat32);
            }
            auto loss = torch::mse_loss(pred_next_states, batch_next_states);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<double>();
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch 
                      << ", Avg Loss: " << total_loss / num_batches << std::endl;
        }
    }

    // Test final performance
    Eigen::VectorXd test_state(2);
    test_state << M_PI/4, 0.0;
    Eigen::VectorXd test_control(1);
    test_control << 0.0;

    // Move test inputs to torch
    auto cpu_options_64 = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto test_state_tensor = torch::from_blob(test_state.data(), {1, 2}, cpu_options_64).clone();
    auto test_control_tensor = torch::from_blob(test_control.data(), {1, 1}, cpu_options_64).clone();
    if (use_gpu) {
        test_state_tensor = test_state_tensor.to(device).to(torch::kFloat32);
        test_control_tensor = test_control_tensor.to(device).to(torch::kFloat32);
    }
    auto pred = model->forward({test_state_tensor, test_control_tensor});
    pred = pred.to(torch::kCPU).to(torch::kFloat64); // back to CPU/double

    // Copy to Eigen
    Eigen::VectorXd pred_eigen(2);
    std::memcpy(pred_eigen.data(), pred.data_ptr<double>(), sizeof(double) * 2);

    // Compare to analytical
    auto analytical_next = analytical_pendulum.getDiscreteDynamics(test_state, test_control, 0.0);
    double error = (analytical_next - pred_eigen).norm();
    std::cout << "Test error: " << error << std::endl;
    EXPECT_LT(error, 0.1);
}

