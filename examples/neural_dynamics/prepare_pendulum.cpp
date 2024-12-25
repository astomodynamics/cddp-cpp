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

/**
 * @brief Print a simple progress bar in the console.
 */
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
    // Number of data samples to generate
    int n_samples = 100; 
    if (argc > 1) {
        n_samples = std::stoi(argv[1]);
    }

    // CSV filename
    std::string csv_filename = "pendulum_dataset.csv";
    if (argc > 2) {
        csv_filename = argv[2];
    }

    // Create dataset directory if it doesn't exist
    std::string dataset_dir = "../examples/neural_dynamics/data";
    if (!std::filesystem::exists(dataset_dir)) {
        std::filesystem::create_directory(dataset_dir);
    }
    // Full path to CSV
    csv_filename = dataset_dir + "/" + csv_filename;

    // Random number engine + distributions
    std::default_random_engine rng(1234);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0*M_PI);
    std::uniform_real_distribution<double> velocity_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> control_dist(-10.0, 10.0);

    // Prepare pendulum system.
    double dt      = 0.01;
    double length  = 1.0;
    double mass    = 1.0;
    double damping = 0.01; 
    std::string integration_type = "rk4";

    cddp::Pendulum pendulum(dt, length, mass, damping, integration_type);

    // Open CSV file
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open file " << csv_filename << std::endl;
        return -1;
    }

    // CSV header: 
    //   theta, theta_dot, control, theta_next, theta_dot_next
    csv_file << "theta,theta_dot,control,theta_next,theta_dot_next\n";

    // For console output
    std::cout << "Generating " << n_samples << " samples..." << std::endl;

    // Allocate some storage for states
    Eigen::VectorXd state(2), control(1);

    // Storage for plotting
    std::vector<double> all_theta;
    std::vector<double> all_theta_dot;

    // Main loop: each sample is one step from random initial conditions
    for (int i = 0; i < n_samples; ++i) {
        // 1) Sample random initial state
        double init_theta    = angle_dist(rng);
        double init_thetadot = velocity_dist(rng);
        state << init_theta, init_thetadot;

        // 2) Sample a random control
        control << control_dist(rng);

        // 3) Integrate one step to get the next state (RK4 inside)
        Eigen::VectorXd next_state = pendulum.getDiscreteDynamics(state, control);

        // 4) Write the row to CSV
        csv_file 
            << state[0]      << ","  // theta
            << state[1]      << ","  // theta_dot
            << control[0]    << ","  // control
            << next_state[0] << ","  // theta_next
            << next_state[1] << "\n";// theta_dot_next

        // Keep track of current state for plotting
        all_theta.push_back(state[0]);
        all_theta_dot.push_back(state[1]);

        // Progress bar (optional)
        if ((i+1) % 200 == 0 || i == n_samples - 1) {
            printProgressBar(i+1, n_samples);
        }
    }

    // Final update for the progress bar
    printProgressBar(n_samples, n_samples);
    std::cout << std::endl;

    // Close file
    csv_file.close();
    std::cout << "Dataset saved to " << csv_filename << std::endl;

    plt::figure_size(1500, 500); // figsize=(15,5) roughly

    // 1) Plot theta distribution
    plt::subplot(1, 3, 1);
    plt::hist(all_theta, 50);  // bins=50
    plt::title("Theta Distribution");
    plt::xlabel("Theta (rad)");

    // 2) Plot theta_dot distribution
    plt::subplot(1, 3, 2);
    plt::hist(all_theta_dot, 50); // bins=50
    plt::title("Angular Velocity Distribution");
    plt::xlabel("Theta_dot (rad/s)");

    // 3) Plot phase space
    plt::subplot(1, 3, 3);
    plt::scatter(all_theta, all_theta_dot, /*size=*/2.0,
        {{"alpha", "0.1"}} ); 
    plt::title("Phase Space");
    plt::xlabel("Theta (rad)");
    plt::ylabel("Theta_dot (rad/s)");

    plt::save("../examples/neural_dynamics/data/pendulum_dataset.png");
    // plt::show();
    return 0;
}
