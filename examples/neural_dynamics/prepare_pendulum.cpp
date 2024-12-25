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
#include <filesystem>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "cddp.hpp"

namespace plt = matplotlibcpp;

// Helper function to print a progress bar
void printProgressBar(int current, int total, int barWidth = 50) {
    float progress = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            std::cout << "=";
        } else if (i == pos) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    int num_samples = 10000;
    if (argc > 1) {
        num_samples = std::stoi(argv[1]);
    }
    
    // Create dataset directory if it doesn't exist
    std::string dataset_dir = "../examples/neural_dynamics/data";
    if (!std::filesystem::exists(dataset_dir)) {
        std::filesystem::create_directory(dataset_dir);
    }

    // CSV filename
    std::string csv_filename = dataset_dir + "/pendulum_dataset.csv";
    if (argc > 2) {
        csv_filename = argv[2];
    }

    // Random distributions
    std::default_random_engine rng(1234);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> velocity_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> control_dist(-2.0, 2.0);

    // Open CSV file
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open file " << csv_filename << std::endl;
        return -1;
    }

    // Write header line
    csv_file << "theta,theta_dot,control,theta_next,theta_dot_next\n";

    std::cout << "Generating " << num_samples << " samples of (state, control, next_state) ..." << std::endl;

    // Create a Pendulum instance
    double timestep = 0.01;
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.0;
    std::string integration_type = "rk4";
    cddp::Pendulum pendulum(timestep, length, mass, damping, integration_type);

    std::vector<double> theta, theta_dot, control_vec;
    std::vector<double> theta_next, theta_dot_next;
    Eigen::VectorXd state(2), control(1), next_state(2);

    state << M_PI/3, 0.0;

    for (int i = 0; i < num_samples; ++i) {
        // Sample random control
        // control << control_dist(rng);
        control << 0.0;

        // Store
        theta.push_back(state[0]);
        theta_dot.push_back(state[1]);
        control_vec.push_back(control[0]);

        // Mesurement
        // next_state = state + Eigen::VectorXd::Random(2) * 0.01;
        next_state = pendulum.getContinuousDynamics(state, control) + Eigen::VectorXd::Random(2) * 0.001;

        // Write row to CSV
        // Format: theta, theta_dot, control, theta_next, theta_dot_next
        csv_file
            << state[0] << "," << state[1] << ","
            << control[0] << ","
            << next_state[0] << "," << next_state[1] << "\n";

        // Update progress bar every so often
        if (i % 50 == 0) {
            printProgressBar(i, num_samples);
        }

        
        state = pendulum.getDiscreteDynamics(state, control);
        // Normalize angle to [0, 2*pi]
        state[0] = std::fmod(state[0], 2.0 * M_PI);
        if (state[0] < 0.0) {
            state[0] += 2.0 * M_PI;
        }
    }
    // Final update to show 100% complete
    printProgressBar(num_samples, num_samples);
    std::cout << std::endl;

    csv_file.close();

    // Plot the dataset
    plt::figure();
    plt::plot(theta, "-");
    plt::title("Pendulum Dataset");
    plt::ylabel("Theta");
    plt::save("../examples/neural_dynamics/data/pendulum_dataset.png");
    // plt::show();

    std::cout << "Dataset saved to " << csv_filename << std::endl;
    return 0;
}
