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
#include "cddp_core/torch_dynamical_system.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

class PendulumModel : public DynamicsModelInterface {
public:
    PendulumModel(double length = 1.0, double mass = 1.0, double damping = 0.0) 
        : length_(length), mass_(mass), damping_(damping),
          device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        
        // Create linear layers
        linear1 = register_module("linear1", torch::nn::Linear(3, 32));
        linear2 = register_module("linear2", torch::nn::Linear(32, 32));
        linear3 = register_module("linear3", torch::nn::Linear(32, 2));

        // Move to appropriate device and dtype
        this->to(device_);
        if (device_.is_cpu()) {
            this->to(torch::kFloat64);
        } else {
            this->to(torch::kFloat32);  // Use Float32 for GPU
        }
        initialize_weights();
    }

    torch::Tensor forward(std::vector<torch::Tensor> inputs) override {
        auto state = inputs[0].to(device_);
        auto control = inputs[1].to(device_);
        
        if (device_.is_cuda()) {
            state = state.to(torch::kFloat32);
            control = control.to(torch::kFloat32);
        }
        
        auto x = torch::cat({state, control}, /*dim=*/1);
        x = torch::tanh(linear1(x));
        x = torch::tanh(linear2(x));
        x = linear3(x);
        
        if (device_.is_cuda()) {
            x = x.to(torch::kFloat64);
        }
        
        return x;
    }

private:
    void initialize_weights() {
        torch::NoGradGuard no_grad;
        
        double angle_scale = 0.1;
        double velocity_scale = 0.1;
        
        auto w = linear3->weight.data();
        w[0][0] = angle_scale;
        w[1][1] = velocity_scale;
        linear3->weight.data() = w.to(torch::kFloat64);
        
        auto b = linear3->bias.data();
        b[0] = 0.0;
        b[1] = -9.81 / length_ * angle_scale;
        linear3->bias.data() = b.to(torch::kFloat64);
    }

    torch::Tensor createTensor(const Eigen::VectorXd& eigen_vec) {
        // First create tensor on CPU
        auto cpu_tensor = torch::from_blob(
            const_cast<double*>(eigen_vec.data()),
            {eigen_vec.size()},
            torch::kFloat64
        ).clone();  // Clone to own the memory
        
        // Then move to target device
        return cpu_tensor.to(device_);
    }

    double length_, mass_, damping_;
    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr};
    torch::Device device_;
};

void printTensorInfo(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << ":\n"
              << " - shape: [" << tensor.sizes() << "]\n"
              << " - dtype: " << tensor.dtype() << "\n"
              << " - device: " << tensor.device() << "\n"
              << " - values: " << tensor << "\n" << std::endl;
}

void printVectorInfo(const Eigen::VectorXd& vec, const std::string& name) {
    std::cout << name << ":\n"
              << " - size: " << vec.size() << "\n"
              << " - values: " << vec.transpose() << "\n" << std::endl;
}

// TEST(TorchPendulumTest, DiscreteDynamics) {
//     // Parameters
//     double timestep = 0.01;
//     double length = 1.0;
//     double mass = 1.0;
//     double damping = 0.0;
//     std::string integration_type = "rk4";

//     // Create analytical pendulum for comparison
//     cddp::Pendulum analytical_pendulum(timestep, length, mass, damping, integration_type);

//     // Create torch model and system
//     auto model = std::make_shared<PendulumModel>(length, mass, damping);
//     bool use_gpu = torch::cuda::is_available();
//     TorchDynamicalSystem torch_pendulum(2, 1, timestep, integration_type, model, use_gpu);

//     // Create device object
//     torch::Device device(use_gpu ? torch::kCUDA : torch::kCPU);

//     // Initial test state and control
//     Eigen::VectorXd test_state(2);
//     test_state << M_PI/4, 0.0;  // 45 degrees, no initial velocity
//     Eigen::VectorXd test_control(1);
//     test_control << 0.0;  // No control input

//     // Print initial conditions
//     std::cout << "\nInitial Test Conditions:" << std::endl;
//     printVectorInfo(test_state, "Test State");
//     printVectorInfo(test_control, "Test Control");

//     // Get analytical prediction
//     auto analytical_next = analytical_pendulum.getDiscreteDynamics(test_state, test_control);
//     std::cout << "\nAnalytical Model Prediction:" << std::endl;
//     printVectorInfo(analytical_next, "Analytical Next State");

//     // Get torch prediction
//     auto torch_next = torch_pendulum.getDiscreteDynamics(test_state, test_control);
//     std::cout << "\nTorch Model Prediction:" << std::endl;
//     printVectorInfo(torch_next, "Torch Next State");

//     // Print error analysis
//     double error = (analytical_next - torch_next).norm();
//     std::cout << "\nError Analysis:" << std::endl;
//     std::cout << "Total Error (L2 norm): " << error << std::endl;
//     std::cout << "Component-wise errors:" << std::endl;
//     std::cout << "Position error: " << std::abs(analytical_next[0] - torch_next[0]) << std::endl;
//     std::cout << "Velocity error: " << std::abs(analytical_next[1] - torch_next[1]) << std::endl;

//     // Basic assertions
//     ASSERT_EQ(torch_pendulum.getStateDim(), 2);
//     ASSERT_EQ(torch_pendulum.getControlDim(), 1);
//     ASSERT_DOUBLE_EQ(torch_pendulum.getTimestep(), 0.01);
//     ASSERT_EQ(torch_pendulum.getIntegrationType(), "rk4");

//     // Prediction accuracy tests
//     EXPECT_NEAR((analytical_next - torch_next).norm(), 0.0, 0.1)
//         << "Total prediction error too large";
//     EXPECT_NEAR(analytical_next[0], torch_next[0], 0.1)
//         << "Position prediction error too large";
//     EXPECT_NEAR(analytical_next[1], torch_next[1], 0.1)
//         << "Velocity prediction error too large";

//     // Test with different initial conditions
//     std::vector<Eigen::VectorXd> test_states = {
//         (Eigen::VectorXd(2) << 0.0, 0.0).finished(),  // At rest
//         (Eigen::VectorXd(2) << M_PI/2, 0.0).finished(),  // 90 degrees
//         (Eigen::VectorXd(2) << 0.0, 1.0).finished(),  // With initial velocity
//         (Eigen::VectorXd(2) << M_PI/4, -1.0).finished()  // 45 degrees with negative velocity
//     };

//     std::cout << "\nTesting multiple initial conditions:" << std::endl;
//     for (const auto& state : test_states) {
//         std::cout << "\nTesting state: " << state.transpose() << std::endl;
        
//         auto analytical = analytical_pendulum.getDiscreteDynamics(state, test_control);
//         auto torch = torch_pendulum.getDiscreteDynamics(state, test_control);
        
//         double test_error = (analytical - torch).norm();
//         std::cout << "Error: " << test_error << std::endl;
        
//         EXPECT_NEAR(test_error, 0.0, 0.1)
//             << "Large error for initial state: " << state.transpose();
//     }

//     // Test Jacobian computation
//     auto state_jacobian = torch_pendulum.getStateJacobian(test_state, test_control);
//     auto control_jacobian = torch_pendulum.getControlJacobian(test_state, test_control);

//     std::cout << "\nJacobian Analysis:" << std::endl;
//     std::cout << "State Jacobian:\n" << state_jacobian << std::endl;
//     std::cout << "Control Jacobian:\n" << control_jacobian << std::endl;

//     // Verify Jacobian dimensions
//     ASSERT_EQ(state_jacobian.rows(), 2);
//     ASSERT_EQ(state_jacobian.cols(), 2);
//     ASSERT_EQ(control_jacobian.rows(), 2);
//     ASSERT_EQ(control_jacobian.cols(), 1);
// }


TEST(TorchPendulumTest, Training) {
    // Parameters
    double timestep = 0.01;
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.1;

    // Create pendulums
    cddp::Pendulum analytical_pendulum(timestep, length, mass, damping, "rk4");
    auto model = std::make_shared<PendulumModel>(length, mass, damping);
    bool use_gpu = torch::cuda::is_available();

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    int num_epochs = 100;
    int batch_size = 32;
    
    // Generate training data
    int num_samples = 1000;
    std::vector<Eigen::VectorXd> states, next_states;
    std::vector<Eigen::VectorXd> controls;

    // Create tensors initially on CPU with Float64
    auto cpu_options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(torch::kCPU);

    auto state_tensor = torch::zeros({num_samples, 2}, cpu_options);
    auto control_tensor = torch::zeros({num_samples, 1}, cpu_options);
    auto next_state_tensor = torch::zeros({num_samples, 2}, cpu_options);

    // Fill tensors
    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd state(2);
        state << (2.0 * (double)rand() / RAND_MAX - 1.0) * M_PI,
                 (2.0 * (double)rand() / RAND_MAX - 1.0) * 5.0;
        
        Eigen::VectorXd control(1);
        control << (2.0 * (double)rand() / RAND_MAX - 1.0) * 2.0;

        states.push_back(state);
        controls.push_back(control);
        next_states.push_back(analytical_pendulum.getDiscreteDynamics(state, control));

        state_tensor[i] = torch::from_blob(state.data(), {2}, cpu_options).clone();
        control_tensor[i] = torch::from_blob(control.data(), {1}, cpu_options).clone();
        next_state_tensor[i] = torch::from_blob(next_states[i].data(), {2}, cpu_options).clone();
    }

    // Move to GPU if available and convert to Float32 for GPU operations
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
            // Create indices with appropriate dtype for GPU
            auto indices = torch::randperm(num_samples, 
                torch::TensorOptions()
                    .dtype(torch::kLong)  // Important: indices must be Long
                    .device(device));
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
                      << ", Average Loss: " << total_loss / num_batches 
                      << std::endl;
        }
    }

    // Testing
    Eigen::VectorXd test_state(2);
    test_state << M_PI/4, 0.0;
    Eigen::VectorXd test_control(1);
    test_control << 0.0;

    // Create test tensors on CPU first
    auto test_state_tensor = torch::from_blob(test_state.data(), {1, 2}, cpu_options).clone();
    auto test_control_tensor = torch::from_blob(test_control.data(), {1, 1}, cpu_options).clone();

    if (use_gpu) {
        test_state_tensor = test_state_tensor.to(device).to(torch::kFloat32);
        test_control_tensor = test_control_tensor.to(device).to(torch::kFloat32);
    } else {
        test_state_tensor = test_state_tensor.to(device);
        test_control_tensor = test_control_tensor.to(device);
    }

    auto pred = model->forward({test_state_tensor, test_control_tensor});
    pred = pred.to(torch::kCPU).to(torch::kFloat64);  // Move back to CPU and double precision

    Eigen::VectorXd pred_eigen(2);
    std::memcpy(pred_eigen.data(), pred.data_ptr<double>(), sizeof(double) * 2);

    auto analytical_next = analytical_pendulum.getDiscreteDynamics(test_state, test_control);
    double error = (analytical_next - pred_eigen).norm();
    EXPECT_LT(error, 0.1);
}