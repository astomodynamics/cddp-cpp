/*
 * train_pendulum_manual.cpp
 *
 * Example approach: manually load the entire dataset into Tensors,
 * and manually create mini-batches. This avoids using DataLoader
 * transforms such as Shuffle, Batch, Stack, etc.
 *
 * Build & run:
 *   $ g++ train_pendulum_manual.cpp -ltorch -lc10 -o train_pendulum_manual
 *   $ ./train_pendulum_manual [pendulum_dataset.csv] [num_epochs] [batch_size]
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

//---------------------------------------//
//   PendulumModel (MLP) Implementation
//---------------------------------------//
struct PendulumModelImpl : public torch::nn::Module {
public:
    PendulumModelImpl(double length=1.0, double mass=1.0, double damping=0.1)
        : length_(length), mass_(mass), damping_(damping)
    {
        linear1 = register_module("linear1", torch::nn::Linear(3, 32));
        linear2 = register_module("linear2", torch::nn::Linear(32, 32));
        linear3 = register_module("linear3", torch::nn::Linear(32, 2));
        initialize_weights();
    }

    torch::Tensor forward(const torch::Tensor &x) {
        auto y = torch::tanh(linear1(x));
        y = torch::tanh(linear2(y));
        y = linear3(y);
        return y;
    }

private:
    void initialize_weights() {
        torch::NoGradGuard no_grad;

        double angle_scale    = 0.1;
        double velocity_scale = 0.1;

        auto w = linear3->weight;
        auto b = linear3->bias;

        if (w.size(1) >= 2) {
            // Just a small manual tweak for initialization
            w.data()[0][0] = angle_scale;
            w.data()[1][1] = velocity_scale;
        }
        b.data()[0] = 0.0;
        b.data()[1] = -9.81 / length_ * angle_scale;
    }

    double length_, mass_, damping_;
public:
    torch::nn::Linear linear1{nullptr}, linear2{nullptr}, linear3{nullptr};
};
TORCH_MODULE(PendulumModel);


//---------------------------------------//
//   Main: Manual Data Loading & Training
//---------------------------------------//
int main(int argc, char* argv[])
{
    // 1. Decide on device (GPU if available)
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // 2. Parse command line args
    std::string csv_path  = "../examples/neural_dynamics/data/pendulum_dataset.csv";
    int num_epochs        = 50;
    int batch_size        = 32;

    if (argc > 1) {
        csv_path = argv[1];
    }
    if (argc > 2) {
        num_epochs = std::stoi(argv[2]);
    }
    if (argc > 3) {
        batch_size = std::stoi(argv[3]);
    }

    // 3. Read entire CSV into two big CPU-based arrays (vectors).
    //    CSV columns: theta,theta_dot,control,theta_next,theta_dot_next
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: could not open CSV file: " << csv_path << std::endl;
        return -1;
    }

    // Skip header
    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: CSV file is empty or invalid." << std::endl;
        return -1;
    }

    // We'll store the values in std::vectors first
    std::vector<double> input_vals;   // for [theta, theta_dot, control]
    std::vector<double> target_vals;  // for [theta_next, theta_dot_next]
    size_t num_samples = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> vals;
        while (std::getline(ss, cell, ',')) {
            vals.push_back(std::stod(cell));
        }
        if (vals.size() < 5) {
            continue; // skip invalid line
        }
        // Layout: [0] -> theta, [1] -> theta_dot, [2] -> control,
        //         [3] -> theta_next, [4] -> theta_dot_next
        input_vals.push_back(vals[0]);
        input_vals.push_back(vals[1]);
        input_vals.push_back(vals[2]);

        target_vals.push_back(vals[3]);
        target_vals.push_back(vals[4]);

        if (num_samples < 5) {
        std::cout << "Sample " << num_samples << ": "
                  << "input=[" << vals[0] << ", " << vals[1] << ", " << vals[2] << "], "
                  << "target=[" << vals[3] << ", " << vals[4] << "]" << std::endl;
        }
    

        num_samples++;
    }
    file.close();

    std::cout << "Loaded " << num_samples << " samples from " << csv_path << std::endl;

    // 4. Convert those vectors to CPU Tensors
    //    We must do from_blob(...) with CPU memory, then clone() if needed.
    auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    torch::Tensor inputs_tensor = torch::from_blob(
        input_vals.data(), {static_cast<long>(num_samples), 3}, cpu_opts).clone();
    torch::Tensor targets_tensor = torch::from_blob(
        target_vals.data(), {static_cast<long>(num_samples), 2}, cpu_opts).clone();

    // 5. Build the model
    const double length = 1.0;
    const double mass   = 1.0;
    const double damping = 0.0;
    auto model = PendulumModel(length, mass, damping);

    // 6. Move model & data to GPU if available
    if (device == torch::kCUDA) {
        std::cout << "Training on GPU (float64)." << std::endl;
        model->to(torch::kCUDA, /*dtype=*/torch::kFloat64);  
        inputs_tensor = inputs_tensor.to(torch::kCUDA);
        targets_tensor = targets_tensor.to(torch::kCUDA);
    } else {
        std::cout << "Training on CPU (float64)." << std::endl;
        model->to(torch::kCPU, /*dtype=*/torch::kFloat64);
    }

    // 7. Create optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // 8. Training loop: manual batching
    std::vector<int64_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    int64_t num_batches = (num_samples + batch_size - 1) / batch_size;
    std::default_random_engine rng(std::random_device{}());

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Shuffle the sample indices
        std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss = 0.0;

        // For each mini-batch
        for (int64_t b = 0; b < num_batches; ++b) {
            int64_t start_idx = b * batch_size;
            int64_t end_idx   = std::min<int64_t>(start_idx + batch_size, num_samples);
            int64_t sz        = end_idx - start_idx;

            // Gather the batch indices
            std::vector<int64_t> batch_idxs(
                indices.begin() + start_idx,
                indices.begin() + end_idx
            );

            auto idx_tensor = torch::from_blob(
                batch_idxs.data(), {static_cast<long>(sz)}, cpu_opts
            ).clone().to(device, torch::kLong);

            // Slice the input and target
            auto batch_inputs  = inputs_tensor.index_select(0, idx_tensor);
            auto batch_targets = targets_tensor.index_select(0, idx_tensor);

            optimizer.zero_grad();
            auto preds = model->forward(batch_inputs);

            // Print preds 
            std::cout << "Preds: " << preds << std::endl;

            auto loss  = torch::mse_loss(preds, batch_targets);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
        }

        // Print progress every 5 epochs
        if (epoch % 5 == 0) {
            double avg_loss = epoch_loss / num_batches;
            std::cout << "Epoch " << epoch << " / " << num_epochs
                      << " | Avg loss: " << avg_loss << std::endl;
        }
    }

    // 9. Save the model
    std::string model_dir = "../examples/neural_dynamics/neural_models";
    if (!std::filesystem::exists(model_dir)) {
        std::filesystem::create_directories(model_dir);
    }
    std::string model_path = model_dir + "/pendulum_model.pt";

    try {
        torch::save(model, model_path);
        std::cout << "Model saved to " << model_path << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Error saving model: " << e.msg() << std::endl;
    }

    std::cout << "Training complete." << std::endl;
    return 0;
}
