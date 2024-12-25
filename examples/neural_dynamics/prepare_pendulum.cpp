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
    // Number of random trajectories and steps per trajectory
    int num_trajectories     = 10;    // e.g. 10 initial points
    int steps_per_trajectory = 1000;  // e.g. each trajectory 1000 steps
    if (argc > 1) {
        num_trajectories = std::stoi(argv[1]);
    }
    if (argc > 2) {
        steps_per_trajectory = std::stoi(argv[2]);
    }

    // Create dataset directory if it doesn't exist
    std::string dataset_dir = "../examples/neural_dynamics/data";
    if (!std::filesystem::exists(dataset_dir)) {
        std::filesystem::create_directory(dataset_dir);
    }

    // CSV filename
    std::string csv_filename = dataset_dir + "/pendulum_dataset.csv";
    if (argc > 3) {
        csv_filename = argv[3];
    }

    // Random number engine + distributions
    std::default_random_engine rng(1234);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> velocity_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> control_dist(-2.0, 2.0);

    // (Optional) noise if you want measurement noise
    std::normal_distribution<double> noise_dist(0.0, 0.01); // small noise

    // Open CSV file
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open file " << csv_filename << std::endl;
        return -1;
    }

    // CSV header: 
    // We'll treat columns as:
    //   theta, theta_dot, control,  measured_theta_dot, measured_theta_ddot
    csv_file << "theta,theta_dot,control,theta_dot_meas,theta_ddot_meas\n";

    // For console output
    int total_samples = num_trajectories * steps_per_trajectory;
    std::cout << "Generating " << total_samples 
              << " samples (across " << num_trajectories << " trajectories)..." 
              << std::endl;

    // Create a Pendulum instance (for continuous-time dynamics)
    double timestep = 0.01;
    double length   = 1.0;
    double mass     = 1.0;
    double damping  = 0.0;
    std::string integration_type = "rk4";
    cddp::Pendulum pendulum(timestep, length, mass, damping, integration_type);

    Eigen::VectorXd state(2), control(1);

    // We'll track how many total samples we've processed
    int sample_count = 0;

    // For each random trajectory
    for (int traj_i = 0; traj_i < num_trajectories; ++traj_i) {
        // 1) Sample a random initial state
        double init_theta = angle_dist(rng);
        double init_thetadot = velocity_dist(rng);
        state << init_theta, init_thetadot;

        // 2) Optionally randomize or fix control each step
        //    We'll do a new random control every step here.
        for (int step_i = 0; step_i < steps_per_trajectory; ++step_i) {
            control << control_dist(rng); // random control in [-2, 2]

            // 3) Current state is (theta, theta_dot). We'll measure the
            //    continuous-time dynamics: [theta_dot, theta_ddot]
            Eigen::VectorXd cont_dyn = pendulum.getContinuousDynamics(state, control);

            // 4) Add some small noise if desired
            cont_dyn[0] += noise_dist(rng); // noise in measured theta_dot
            cont_dyn[1] += noise_dist(rng); // noise in measured theta_ddot

            // 5) Write the row to CSV
            csv_file 
                << state[0]      << ","  // theta
                << state[1]      << ","  // theta_dot
                << control[0]    << ","  // control
                << cont_dyn[0]   << ","  // measured theta_dot
                << cont_dyn[1]   << "\n";// measured theta_ddot

            // 6) Step forward one time step with discrete integration
            //    to update "state" for the next iteration
            state = pendulum.getDiscreteDynamics(state, control);

            // Optionally wrap angle to [-pi, pi] or [0, 2*pi] (your preference)
            state[0] = std::fmod(state[0], 2.0 * M_PI);
            if (state[0] < 0.0) {
                state[0] += 2.0 * M_PI;
            }

            // 7) Progress bar
            sample_count++;
            if (sample_count % 50 == 0) {
                printProgressBar(sample_count, total_samples);
            }
        }
    }

    // Final update for the progress bar
    printProgressBar(total_samples, total_samples);
    std::cout << std::endl;

    // Close file
    csv_file.close();
    std::cout << "Dataset saved to " << csv_filename << std::endl;

    return 0;
}
