/*
 Copyright 2024 Tomo

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

// Standard headers
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cddp.hpp"

struct ODEFuncImpl : public torch::nn::Module {
    ODEFuncImpl(int64_t hidden_dim=32) {
        net = register_module("net", torch::nn::Sequential(
            torch::nn::Linear(/*in_features=*/2, hidden_dim),
            torch::nn::Tanh(),
            torch::nn::Linear(hidden_dim, hidden_dim),
            torch::nn::Tanh(),
            torch::nn::Linear(hidden_dim, 2)
        ));
    }

    // forward(t, y) -> dy/dt
    torch::Tensor forward(const torch::Tensor &t, const torch::Tensor &y) {
        return net->forward(y);
    }

    torch::nn::Sequential net;
};
TORCH_MODULE(ODEFunc);

torch::Tensor rk4_step(
    ODEFunc &func,
    const torch::Tensor &t,
    const torch::Tensor &y,
    double dt
) {
    auto half_dt = dt * 0.5;
    auto k1 = func->forward(t, y);
    auto k2 = func->forward(t + half_dt, y + half_dt * k1);
    auto k3 = func->forward(t + half_dt, y + half_dt * k2);
    auto k4 = func->forward(t + dt,      y + dt * k3);
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

struct NeuralODEImpl : public torch::nn::Module {
    NeuralODEImpl(int64_t hidden_dim=32) {
        func_ = register_module("func", ODEFunc(hidden_dim));
    }

    // forward(y0, t, dt) -> entire trajectory
    torch::Tensor forward(const torch::Tensor &y0, const torch::Tensor &t, double dt)
    {
        int64_t batch_size = y0.size(0); 
        int64_t steps      = t.size(0);

        // shape: [B, steps, 2]
        torch::Tensor trajectory = torch::zeros({batch_size, steps, 2},
            torch::TensorOptions().device(y0.device()).dtype(y0.dtype()));

        // first step is the initial state
        trajectory.select(1, 0) = y0;

        auto state = y0.clone();
        for (int64_t i = 0; i < steps - 1; ++i) {
            auto t_i = t[i];
            state = rk4_step(func_, t_i, state, dt);

            // Wrap theta to [0.0, 2*pi]
            state.select(1, 0) = torch::fmod(state.select(1, 0), 2.0 * M_PI);
            state.select(1, 0) = (state.select(1, 0) < 0).to(torch::kFloat32) * (2.0 * M_PI) + state.select(1, 0);

            trajectory.select(1, i+1) = state;
        }

        return trajectory;
    }

    ODEFunc func_;
};
TORCH_MODULE(NeuralODE);


int main(int argc, char* argv[])
{
    // 1) Parse command line arguments
    std::string model_file = "../examples/neural_dynamics/neural_models/neural_pendulum.pth";
    float init_theta = 1.56f;
    float init_thetadot = 0.0f;
    int64_t seq_length = 100; // FIXME:

    if (argc > 1) model_file   = argv[1];
    if (argc > 2) init_theta   = std::stof(argv[2]);
    if (argc > 3) init_thetadot= std::stof(argv[3]);
    if (argc > 4) seq_length   = std::stoll(argv[4]);

    std::cout << "Model file: " << model_file << std::endl;
    std::cout << "Initial state: (theta=" << init_theta << ", theta_dot=" << init_thetadot << ")" << std::endl;
    std::cout << "Sequence length: " << seq_length << std::endl;

    // 2) Setup device
    torch::Device device = torch::kCPU;

    // 3) Load the trained model
    auto neural_ode = NeuralODE(/*hidden_dim=*/32); 
    torch::load(neural_ode, model_file);
    neural_ode->to(device);
    neural_ode->eval(); 

    // 4) Prepare the initial state, time vector
    auto y0 = torch::tensor({init_theta, init_thetadot}).view({1,2}).to(device);

    float dt = 0.02f; // FIXME:
    auto t_cpu = torch::arange(seq_length, torch::kInt64).to(torch::kFloat32) * dt;
    auto t = t_cpu.to(device);

    // 5) Run the neural ODE to get the predicted trajectory
    auto pred_traj = neural_ode->forward(y0, t, dt); // shape: [1, seq_length, 2]
    pred_traj = pred_traj.squeeze(0).cpu(); // shape [seq_length, 2]

    // 6) Generate the "true" trajectory from cddp::Pendulum
    cddp::Pendulum pendulum(// FIXME:
        /*dt=*/0.02,  /*length=*/1.0,
        /*mass=*/1.0, /*damping=*/0.01,
        /*integration_type=*/"rk4"
    );
    // zero torque
    Eigen::VectorXd control(1);
    control.setZero();

    // initial state
    Eigen::VectorXd state(2);
    state << init_theta, init_thetadot;

    std::vector<float> theta_vec_nn(seq_length);
    std::vector<float> thetadot_vec_nn(seq_length);
    std::vector<float> theta_vec_true(seq_length);
    std::vector<float> thetadot_vec_true(seq_length);

    // fill these vectors in your loop:
    for (int64_t i = 0; i < seq_length; ++i) {
        theta_vec_nn[i]      = pred_traj[i][0].item<float>();
        thetadot_vec_nn[i]   = pred_traj[i][1].item<float>();
        theta_vec_true[i]    = static_cast<float>(state(0));
        thetadot_vec_true[i] = static_cast<float>(state(1));
        if (i < seq_length - 1) {
            state = pendulum.getDiscreteDynamics(state, control);
            // Wrap theta to [0.0, 2*pi]
            state(0) = std::fmod(state(0), 2.0 * M_PI);
            if (state(0) < 0) {
                state(0) += 2.0 * M_PI;
            }
        }
    }

    // Create a 2D tensor of shape [seq_length, 2] for the true trajectory
    auto true_tensor = torch::empty({seq_length, 2}, torch::kFloat32);
    for (int64_t i = 0; i < seq_length; ++i) {
        true_tensor[i][0] = theta_vec_true[i];
        true_tensor[i][1] = thetadot_vec_true[i];
    }
    true_tensor = true_tensor.to(device);

    // 7) Compare predicted vs. true
    auto mse = torch::mse_loss(pred_traj, true_tensor);
    float mse_val = mse.item<float>();

    std::cout << "Comparison result:\n";
    std::cout << " - MSE: " << mse_val << std::endl;

    // Print a few sample points
    std::cout << "\nIndex | True (theta, theta_dot) | Pred (theta, theta_dot)\n";
    std::cout << "---------------------------------------------------------\n";
    for (int64_t i = 0; i < std::min<int64_t>(seq_length, 5); ++i) {
        auto t_th   = true_tensor[i][0].item<float>();
        auto t_td   = true_tensor[i][1].item<float>();
        auto p_th    = pred_traj[i][0].item<float>();
        auto p_td    = pred_traj[i][1].item<float>();
        std::cout << i << "     | ("
                  << t_th << ", " << t_td << ") | ("
                  << p_th << ", " << p_td << ")\n";
    }

    std::string out_file = "pendulum_compare.csv";
    {
        std::ofstream ofs(out_file);
        ofs << "index,true_theta,true_thetadot,pred_theta,pred_thetadot\n";
        for (int64_t i = 0; i < seq_length; ++i) {
            auto t_th   = true_tensor[i][0].item<float>();
            auto t_td   = true_tensor[i][1].item<float>();
            auto p_th    = pred_traj[i][0].item<float>();
            auto p_td    = pred_traj[i][1].item<float>();
            ofs << i << "," 
                << t_th << "," << t_td << ","
                << p_th << "," << p_td << "\n";
        }
        ofs.close();
        std::cout << "Saved CSV: " << out_file << std::endl;
    }

    // // 8) Plot the trajectories 
    // // plt args: (x, y, color, linestyle, linewidth, label)

    // plt::figure_size(800, 400);
    // plt::subplot(1, 2, 1);
    // plt::title("True vs Predicted (Theta)");
    // plt::plot(theta_vec_true, {{"color", "red"}, {"linestyle", "-"}, {"label", "True theta"}});
    // plt::plot(theta_vec_nn, {{"color", "blue"}, {"linestyle", "--"}, {"label", "Predicted theta"}});
    // plt::legend();
    // plt::xlabel("Time step");
    // plt::ylabel("Theta");

    // plt::subplot(1, 2, 2);
    // plt::title("True vs Predicted (Theta_dot)");
    // plt::plot(thetadot_vec_true, {{"color", "red"}, {"linestyle", "-"}, {"label", "True theta_dot"}});
    // plt::plot(thetadot_vec_nn, {{"color", "blue"}, {"linestyle", "--"}, {"label", "Predicted theta_dot"}});
    // plt::legend();
    // plt::xlabel("Time step");
    // plt::ylabel("Theta_dot");

    // plt::save("../examples/neural_dynamics/neural_models/pendulum_compare.png");
    // std::cout << "Saved plot: pendulum_compare.png" << std::endl;

    return 0;
}
