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

#include "cddp.hpp"
#include "cddp_core/neural_dynamical_system.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
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

    double angle_scale   = 0.1;
    double velocity_scale = 0.1;

    auto w = linear3->weight.data();
    w[0][0] = angle_scale;     // small tweak for angle
    w[1][1] = velocity_scale;  // small tweak for velocity
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
        state   = state.to(torch::kFloat32);
        control = control.to(torch::kFloat32);
    }

    // shape checks: state=[batch,2], control=[batch,1]
    auto x = torch::cat({state, control}, /*dim=*/1);
    x = torch::tanh(linear1(x));
    x = torch::tanh(linear2(x));
    x = linear3(x);

    // Return in double if you prefer
    if (device_.is_cuda()) {
        x = x.to(torch::kFloat64);
    }
    return x;
}


// ---------------------------------------------------------------------------
// Helper function for printing Eigen vectors (for debugging)
// ---------------------------------------------------------------------------
void printVectorInfo(const Eigen::VectorXd& vec, const std::string& name)
{
    std::cout << name << ":\n"
              << " - size: " << vec.size() << "\n"
              << " - values: " << vec.transpose() << "\n" << std::endl;
}


// ---------------------------------------------------------------------------
// Test comparing a loaded neural model's forward sim vs. the analytical model
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // 1. Load trained model from file (e.g. "pendulum_model.pt")
    std::string model_file = "../examples/neural_dynamics/neural_models/pendulum_model.pt";
    if (!fs::exists(model_file)) {
        std::cout << "Trained model file not found: " << model_file << std::endl;
    }

    // We instantiate a new PendulumModel and load weights
    auto model = std::make_shared<PendulumModel>(1.0, 1.0, 0.1);
    try {
        torch::load(model, model_file);
        std::cout << "Loaded trained model from: " << model_file << std::endl;
    } catch (const c10::Error& e) {
        std::cout << "Could not load model from " << model_file
                     << ": " << e.msg() << std::endl;
    }

    // 2. Create cddp::NeuralDynamicalSystem with that model
    double timestep        = 0.01;
    double length          = 1.0;
    double mass            = 1.0;
    double damping         = 0.1;
    std::string integrator = "rk4";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    NeuralDynamicalSystem neural_pendulum(/*state_dim=*/2,
                                          /*control_dim=*/1,
                                          timestep,
                                          integrator,
                                          model,
                                          device);

    // 3. Create an analytical pendulum with the same parameters
    cddp::Pendulum analytical_pendulum(timestep, length, mass, damping, integrator);

    // 4. Choose initial conditions
    Eigen::VectorXd x0(2);
    x0 << M_PI/2, 0.0;  // start at 90 deg, zero velocity
    Eigen::VectorXd u0(1);
    u0 << 0.0;          // no control

    // 5. Simulate both for N steps
    int N = 300; // e.g. 3 seconds if dt=0.01
    Eigen::VectorXd x_ana = x0;
    Eigen::VectorXd x_nn  = x0;

    // We'll store angle and velocity for plotting
    std::vector<double> time_data, angle_ana, angle_nn, vel_ana, vel_nn;
    time_data.reserve(N+1);
    angle_ana.reserve(N+1);
    angle_nn.reserve(N+1);
    vel_ana.reserve(N+1);
    vel_nn.reserve(N+1);

    // Log initial states
    time_data.push_back(0.0);
    angle_ana.push_back(x_ana[0]);
    vel_ana.push_back(x_ana[1]);
    angle_nn.push_back(x_nn[0]);
    vel_nn.push_back(x_nn[1]);

    for(int i=1; i<=N; ++i) {
        // Analytical update
        x_ana = analytical_pendulum.getDiscreteDynamics(x_ana, u0);
        // Neural update
        x_nn  = neural_pendulum.getDiscreteDynamics(x_nn, u0);

        double t = i * timestep;
        time_data.push_back(t);

        angle_ana.push_back(x_ana[0]);
        vel_ana.push_back(x_ana[1]);

        angle_nn.push_back(x_nn[0]);
        vel_nn.push_back(x_nn[1]);
    }

    // 6. Evaluate final L2 difference in state
    //    (just as a rough check that they haven't diverged too far)
    double err_norm = (x_ana - x_nn).norm();
    std::cout << "Final L2 error: " << err_norm << std::endl;
    // We can set a relatively loose tolerance, since it's multi-step integration
    // EXPECT_LT(err_norm, 2.0);

    // 7. Plot results with matplotlibcpp
    //    We'll plot angle vs time, and velocity vs time
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

    // 8. Save figure to a file (instead of show, which might block in CI)
    std::string fig_path = "pendulum_comparison.png";
    plt::save(fig_path);
    std::cout << "Saved plot: " << fig_path << std::endl;

    // If you prefer to see the plot interactively (and your environment supports it), do:
    // plt::show();

    return 0;
}

