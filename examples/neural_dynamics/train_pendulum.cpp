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
/*
 * Build & run:
 *   $ ./examples/train_pendulum [pendulum_dataset.csv] [num_epochs] [batch_size]
 * i.e.
 *   $ ./examples/train_pendulum pendulum_dataset.csv 32 100
 */

#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>      // for std::random_shuffle or std::shuffle
#include <algorithm>   // for std::min, std::shuffle
#include <numeric>     // for std::iota
#include "cddp.hpp"

struct ODEFuncImpl : public torch::nn::Module {
    // net: 2 -> hidden_dim -> hidden_dim -> 2
    ODEFuncImpl(int64_t hidden_dim=32) {
        net = register_module("net", torch::nn::Sequential(
            torch::nn::Linear(/*in_features=*/2, hidden_dim),
            torch::nn::Tanh(),
            torch::nn::Linear(hidden_dim, hidden_dim),
            torch::nn::Tanh(),
            torch::nn::Linear(hidden_dim, 2)
        ));
    }
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

    torch::Tensor forward(const torch::Tensor &y0,
                          const torch::Tensor &t,
                          double dt)
    {
        int64_t batch_size = y0.size(0);
        int64_t steps      = t.size(0);

        torch::Tensor trajectory = torch::zeros({batch_size, steps, 2},
            torch::TensorOptions().device(y0.device()).dtype(y0.dtype()));

        trajectory.select(1, 0) = y0;

        auto state = y0.clone();
        for (int64_t i = 0; i < steps - 1; ++i) {
            // t[i], shape=()
            auto t_i = t[i];
            state = rk4_step(func_, t_i, state, dt);
            trajectory.select(1, i+1) = state;
        }

        return trajectory;
    }

    ODEFunc func_;
};
TORCH_MODULE(NeuralODE);

class PendulumDataset : public torch::data::Dataset<PendulumDataset>
{
public:
    // Constructor
    explicit PendulumDataset(const std::string &csv_file, int64_t seq_length=200)
        : seq_length_(seq_length)
        , pendulum_(/*FIXME: change and match constants*/
              /*timestep=*/0.02,  /*length=*/0.5,
              /*mass=*/1.0,      /*damping=*/0.01,
              /*integration_type=*/"rk4"
          )
    {
        // 1) Read CSV into initial_states_ vector
        std::ifstream file(csv_file);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV: " + csv_file);
        }

        {
            std::string header_line;
            if (std::getline(file, header_line)) {
                std::cout << "Skipping header: " << header_line << std::endl;
            }
        }

        // read lines
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<double> vals;
            while (!ss.eof()) {
                std::string cell;
                if (!std::getline(ss, cell, ',')) break;
                if (!cell.empty()) {
                    vals.push_back(std::stod(cell));
                }
            }
            if (vals.size() < 2) {
                // not enough columns
                continue;
            }
            // store (theta, theta_dot) as float
            initial_states_.push_back({(float)vals[0], (float)vals[1]});
        }
        file.close();

        std::cout << "Loaded " << initial_states_.size() 
                  << " initial states from " << csv_file << std::endl;

        // 2) Generate trajectories with the cddp::Pendulum
        generate_trajectories();

        // 3) Convert to Tensors
        states_tensor_ = torch::from_blob(
            initial_states_.data(),
            {(long)initial_states_.size(), 2},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        trajectories_tensor_ = torch::from_blob(
            trajectories_.data(),
            {(long)initial_states_.size(), seq_length_, 2},
            torch::TensorOptions().dtype(torch::kFloat32)
        ).clone();

        float dt = 0.01f;
        t_ = torch::arange(seq_length_, torch::kInt64).to(torch::kFloat32).mul(dt);
    }

    // override size()
    torch::optional<size_t> size() const override {
        return initial_states_.size();
    }

    torch::data::Example<> get(size_t idx) override {
        // x = initial state [2]
        auto x = states_tensor_[idx];
        // y = entire trajectory [seq_length_, 2]
        auto y = trajectories_tensor_[idx];
        return {x, y};
    }

    // Optionally expose the time vector
    torch::Tensor get_time_vector() const {
        return t_;
    }

private:
    void generate_trajectories() {
        trajectories_.resize(initial_states_.size() * seq_length_ * 2, 0.f);

        // Create a zero control input
        Eigen::VectorXd control(1);
        control.setZero(); // torque = 0

        std::cout << "Generating " << initial_states_.size() << " trajectories of length " 
                  << seq_length_ << std::endl;

        for (size_t i = 0; i < initial_states_.size(); ++i) {
            // Convert our float pair into an Eigen::VectorXd
            float theta     = initial_states_[i][0];
            float theta_dot = initial_states_[i][1];
            Eigen::VectorXd state(2);
            state << theta, theta_dot;

            // if (i == 0) {
            //     std::cout << "Initial state [" << i << "]: theta=" << theta 
            //              << ", theta_dot=" << theta_dot << std::endl;
            // }

            for (int64_t j = 0; j < seq_length_; ++j) {
                // store in trajectories_
                size_t base_idx = i * seq_length_ * 2 + j * 2;

                trajectories_[base_idx + 0] = static_cast<float>(state(0));
                trajectories_[base_idx + 1] = static_cast<float>(state(1));

                // if j < seq_length_-1, step forward
                if (j < seq_length_ - 1) {
                    state = pendulum_.getDiscreteDynamics(state, control);
                }
            }
        }
        std::cout << "Trajectory generation complete." << std::endl;
    }

private:
    int64_t seq_length_;
    std::vector<std::array<float, 2>> initial_states_;
    std::vector<float> trajectories_; // size = num_samples * seq_length_ * 2

    torch::Tensor states_tensor_;       // shape: [num_samples, 2]
    torch::Tensor trajectories_tensor_; // shape: [num_samples, seq_length_, 2]
    torch::Tensor t_;                   // shape: [seq_length_]

    cddp::Pendulum pendulum_;
};


int main(int argc, char* argv[])
{
    // 1. Decide on device (GPU if available)
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // 2. Parse command line args
    std::string csv_path  = "../examples/neural_dynamics/data";
    std::string csv_file  = csv_path + "/pendulum_dataset.csv";
    std::string model_path = "../examples/neural_dynamics/neural_models";
    std::string model_file = model_path + "/neural_pendulum.pth";
    // Create a directory if it doesn't exist
    if (!std::filesystem::exists(model_path)) {
        std::filesystem::create_directories(model_path);
    }

    int64_t batch_size   = 32;
    int64_t num_epochs   = 1000; // FIXME: 
    if (argc > 1) csv_file   = csv_path + "/" + std::string(argv[1]);
    if (argc > 2) batch_size = std::stoll(argv[2]);
    if (argc > 3) num_epochs = std::stoll(argv[3]);

    // 3. Create dataset & dataloader
    int64_t seq_length = 100; // FIXME: horizon length
    PendulumDataset dataset(csv_file, seq_length);

    // 4. Load the dataset into a DataLoader
    auto data_loader = torch::data::make_data_loader(
        dataset.map(torch::data::transforms::Stack<>()),
        /*batch_size=*/batch_size
    );

    // 5. Create the NeuralODE model
    auto model = NeuralODE(/*hidden_dim=*/32);
    model->to(device);
    std::cout << "Training on " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;

    // 6. Create optimizer
    double learning_rate = 1e-3;
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    // 7. Time vector (CPU, then push to device)
    float dt = 0.02f; // FIXME:
    auto t = dataset.get_time_vector().to(device);

    // 8. Training loop
    std::vector<double> losses;

    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        double epoch_loss = 0.0;
        int batch_count = 0;

        for (auto& batch : *data_loader) {
            // batch.data shape = [B, 2]
            // batch.target shape = [B, seq_length, 2]
            auto data   = batch.data.to(device);
            auto target = batch.target.to(device);

            optimizer.zero_grad();

            auto output = model->forward(data, t, /*dt=*/0.01);
            auto loss = torch::mse_loss(output, target);

            // For older libTorch versions:
            float loss_val = loss.item().toFloat();

            loss.backward();
            optimizer.step();

            epoch_loss += loss_val;
            batch_count++;
        }

        if (epoch % 5 == 0) {
            double avg_loss = epoch_loss / static_cast<double>(batch_count);
            torch::save(model, model_file + std::to_string(epoch) + ".pth");
            losses.push_back(avg_loss);

            std::cout << "Epoch " << epoch << " / " << num_epochs
                      << " | Avg loss: " << avg_loss << std::endl;
        }
        
    }

    std::cout << "Training complete." << std::endl;

    // 9. Save the model
    torch::save(model, model_file);
    std::cout << "Model saved to " << model_file << std::endl;

    // 10. Plot the loss
    plt::figure();
    plt::plot(losses);
    plt::title("Training Loss");
    plt::xlabel("Epoch");
    plt::ylabel("MSE Loss");
    plt::save(model_path + "/training_loss.png");
    std::cout << "Saved plot: " << model_path + "/training_loss.png" << std::endl;

    return 0;
}
