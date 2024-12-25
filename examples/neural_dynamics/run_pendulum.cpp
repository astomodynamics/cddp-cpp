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
#include <string>
#include <filesystem>
#include <cmath>       // for std::fmod, std::isfinite

#include "cddp.hpp"
#include "cddp_core/neural_dynamical_system.hpp"

namespace plt = matplotlibcpp;
namespace fs  = std::filesystem;
using namespace cddp;

// ---------------------------------------------------------------------------
// PendulumModel: same class definition as in your other test code or header
// ---------------------------------------------------------------------------
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

    double angle_scale    = 0.1;
    double velocity_scale = 0.1;

    auto w = linear3->weight.data();
    w[0][0] = angle_scale;     
    w[1][1] = velocity_scale;  
    linear3->weight.data() = w;

    auto b = linear3->bias.data();
    b[0] = 0.0;
    b[1] = -9.81 / length_ * angle_scale;
    linear3->bias.data() = b;
}

torch::Tensor PendulumModel::forward(std::vector<torch::Tensor> inputs)
{
    // Expect 2 inputs: [state, control]
    auto state   = inputs[0].to(device_);
    auto control = inputs[1].to(device_);

    if (device_.is_cuda()) {
        // If on GPU, we cast to float32 for the linear layers
        state   = state.to(torch::kFloat32);
        control = control.to(torch::kFloat32);
    }

    // shape checks: state=[batch,2], control=[batch,1]
    auto x = torch::cat({state, control}, /*dim=*/1);
    x = torch::tanh(linear1(x));
    x = torch::tanh(linear2(x));
    x = linear3(x);

    // Convert back to double if needed
    if (device_.is_cuda()) {
        x = x.to(torch::kFloat64);
    }
    return x;
}

// ---------------------------------------------------------------------------
// Helper for printing Eigen vectors (debugging)
// ---------------------------------------------------------------------------
void printVectorInfo(const Eigen::VectorXd& vec, const std::string& name)
{
    std::cout << name << ":\n"
              << " - size: " << vec.size() << "\n"
              << " - values: " << vec.transpose() << "\n" << std::endl;
}

// ---------------------------------------------------------------------------
// main()
// We skip the built-in getDiscreteDynamics and just do "pure neural forward".
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // 1. Load trained model from file
    std::string model_file = "../examples/neural_dynamics/neural_models/pendulum_model.pt";
    if (!fs::exists(model_file)) {
        std::cout << "Trained model file not found: " << model_file << std::endl;
        return 0;
    }

    auto model = std::make_shared<PendulumModel>(1.0, 1.0, 0.1);
    try {
        torch::load(model, model_file);
        std::cout << "Loaded trained model from: " << model_file << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "Could not load model from " << model_file
                  << ": " << e.msg() << std::endl;
        return 0;
    }

    // 2. Analytical pendulum
    double timestep = 0.01;
    double length   = 1.0;
    double mass     = 1.0;
    double damping  = 0.0;      // note: if training used damping=0.1, mismatch here
    std::string integrator = "euler";

    cddp::Pendulum analytical_pendulum(timestep, length, mass, damping, integrator);

    // 3. Initial conditions
    Eigen::VectorXd x0(2);
    x0 << M_PI/2, 0.0;
    Eigen::VectorXd u0(1);
    u0 << 0.0;

    // 4. Prepare data logging
    int N = 300;
    Eigen::VectorXd x_ana = x0;    // for analytical
    Eigen::VectorXd x_nn  = x0;    // for neural

    std::vector<double> time_data(N+1), angle_ana(N+1), angle_nn(N+1);
    std::vector<double> vel_ana(N+1), vel_nn(N+1);

    time_data[0]   = 0.0;
    angle_ana[0]   = x_ana[0];
    vel_ana[0]     = x_ana[1];
    angle_nn[0]    = x_nn[0];
    vel_nn[0]      = x_nn[1];

    // 5. Loop for N steps
    for(int i=1; i<=N; ++i) {
        double t = i * timestep;
        time_data[i] = t;

        // Analytical update: one step
        x_ana = analytical_pendulum.getDiscreteDynamics(x_ana, u0);

        // Pure neural forward:
        //  a) Convert x_nn, u0 to Tensors
        //  b) Call model->forward()
        //  c) That result is the predicted "next state".
        {
            // Build a [1,2] state tensor
            auto state_tensor = torch::from_blob(x_nn.data(), {1, 2}, 
                                 torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)).clone();

            // Build a [1,1] control tensor
            auto ctrl_tensor  = torch::from_blob(u0.data(), {1, 1}, 
                                 torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU)).clone();

            // Pass them to the model
            auto next_state_torch = model->forward({state_tensor, ctrl_tensor}); 
            // next_state_torch shape: [1, 2]

            // Move to CPU double if not already
            next_state_torch = next_state_torch.to(torch::kCPU).to(torch::kFloat64);

            // Extract numeric data
            double angle_pred = next_state_torch[0][0].item<double>();
            double vel_pred   = next_state_torch[0][1].item<double>();

            if (!std::isfinite(angle_pred) || !std::isfinite(vel_pred)) {
                std::cerr << "NaN at step " << i << std::endl;
                break;
            }

            // Check for NaNs
            if (!std::isfinite(angle_pred) || !std::isfinite(vel_pred)) {
                std::cerr << "[WARNING] Found NaN or Inf at step " << i 
                          << " for neural pendulum. Stopping.\n"
                          << "  angle_pred=" << angle_pred 
                          << ", vel_pred=" << vel_pred << "\n";
                break;
            }

            // Update x_nn
            x_nn[0] = angle_pred;
            x_nn[1] = vel_pred;
        }

        // Log data
        angle_ana[i] = x_ana[0];
        vel_ana[i]   = x_ana[1];
        angle_nn[i]  = x_nn[0];
        vel_nn[i]    = x_nn[1];

        // Optional: wrap angles to [0,2*pi] if desired
        x_ana[0] = std::fmod(x_ana[0], 2.0 * M_PI);
        if (x_ana[0] < 0.0) {
            x_ana[0] += 2.0 * M_PI;
        }
        x_nn[0]  = std::fmod(x_nn[0], 2.0 * M_PI);
        if (x_nn[0] < 0.0) {
            x_nn[0] += 2.0 * M_PI;
        }

        // Print debugging info
        std::cout << "t=" << t 
                  << ", x_ana=" << x_ana.transpose() 
                  << ", x_nn=" << x_nn.transpose() << std::endl;
    }

    // 6. Evaluate final L2 difference in state
    double err_norm = (x_ana - x_nn).norm();
    std::cout << "Final L2 error: " << err_norm << std::endl;

    // 7. Plot results
    // Create a directory if it doesn't exist
    std::string fig_dir = "../examples/neural_dynamics/data";
    if (!fs::exists(fig_dir)) {
        fs::create_directories(fig_dir);
    }
    std::string fig_path = fig_dir + "/pendulum_comparison.png";

    plt::figure();
    plt::subplot(2,1,1);
    plt::plot(time_data, angle_ana, {{"label", "Angle (Analytical)"}});
    plt::plot(time_data, angle_nn,  {{"label", "Angle (Neural)"}});

    plt::ylabel("Angle (rad)");
    plt::legend();

    plt::subplot(2,1,2);
    plt::plot(time_data, vel_ana, {{"label", "Vel (Analytical)"}});
    plt::plot(time_data, vel_nn,  {{"label", "Vel (Neural)"}});

    plt::xlabel("Time (s)");
    plt::ylabel("Angular Velocity (rad/s)");
    plt::legend();

    plt::save(fig_path);
    std::cout << "Saved plot: " << fig_path << std::endl;

    return 0;
}
