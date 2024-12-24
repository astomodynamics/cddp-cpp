#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <torch/torch.h>

// If you want to store CSV or custom format, define your output path
static const std::string kDatasetFile = "pendulum_data.pt";

/**
 * @brief Print a simple progress bar to the console.
 *
 * @param current  The current iteration (0-based).
 * @param total    The total number of iterations.
 * @param barWidth The width (in characters) of the bar portion.
 */
void printProgressBar(int current, int total, int barWidth = 50) {
    float progress = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)      std::cout << "=";
        else if (i == pos)std::cout << ">";
        else              std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    // (Optional) parse command-line arguments for number of samples, etc.
    int num_samples = 1000; 
    if (argc > 1) {
        num_samples = std::stoi(argv[1]);
    }

    double timestep = 0.01;
    double length = 1.0;
    double mass   = 1.0;
    double damping = 0.1;

    std::cout << "Generating " << num_samples << " samples of (state, control, next_state) ..." << std::endl;

    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto state_tensor      = torch::zeros({num_samples, 2}, cpu_options);
    auto control_tensor    = torch::zeros({num_samples, 1}, cpu_options);
    auto next_state_tensor = torch::zeros({num_samples, 2}, cpu_options);

    // If you have a class "Pendulum" or "AnalyticalPendulum" implementing
    // `getDiscreteDynamics(state, control)`, use it:
    // cddp::Pendulum reference_pendulum(timestep, length, mass, damping, "rk4");

    // For demonstration, let's do a placeholder where next_state = state + random noise
    // In practice, you'd call reference_pendulum.getDiscreteDynamics(...) or similar.
    std::default_random_engine rng(1234);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> velocity_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> control_dist(-2.0, 2.0);

    for (int i = 0; i < num_samples; ++i) {
        Eigen::VectorXd state(2), control(1), next_state(2);

        state << angle_dist(rng), velocity_dist(rng);
        control << control_dist(rng);

        // If using a real dynamics function, do:
        // next_state = reference_pendulum.getDiscreteDynamics(state, control);
        // Placeholder: next_state = state + 0.01 * random noise
        next_state = state + Eigen::VectorXd::Random(2) * 0.01;

        // Copy to Tensors
        state_tensor[i]      = torch::from_blob(state.data(), {2}, cpu_options).clone();
        control_tensor[i]    = torch::from_blob(control.data(), {1}, cpu_options).clone();
        next_state_tensor[i] = torch::from_blob(next_state.data(), {2}, cpu_options).clone();

        // Update progress bar every so often (e.g., every 50 samples)
        if (i % 50 == 0) {
            printProgressBar(i, num_samples);
        }
    }
    // Final update to show 100% complete
    printProgressBar(num_samples, num_samples);
    std::cout << std::endl; // Move to the next line

    // 2. Save to disk – for instance, in a single .pt (PyTorch-like) file
    // We can create a simple torch::dict of Tensors and save via torch::save:
    torch::Dict<std::string, torch::Tensor> dataset_map;
    dataset_map.insert("state",       state_tensor);
    dataset_map.insert("control",     control_tensor);
    dataset_map.insert("next_state",  next_state_tensor);

    try {
        torch::save(dataset_map, kDatasetFile);
        std::cout << "Dataset saved to " << kDatasetFile << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error saving the dataset: " << e.msg() << std::endl;
        return -1;
    }

    return 0;
}
