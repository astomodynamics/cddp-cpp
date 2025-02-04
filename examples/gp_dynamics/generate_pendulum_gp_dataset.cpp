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
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <filesystem>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "cddp.hpp"  // Assumes this header includes the Pendulum class and matplotlibcpp.hpp

using namespace cddp;
namespace fs = std::filesystem;

// Helper function: Wrap a given angle (in radians) to the interval [0, 2*pi)
double wrapTheta(double theta) {
    double two_pi = 2.0 * M_PI;
    theta = std::fmod(theta, two_pi);
    if (theta < 0)
        theta += two_pi;
    return theta;
}

int main(int argc, char* argv[]) {
    // --- Dataset Generation Parameters ---
    std::string default_output_csv = "pendulum_dataset.csv";
    int default_num_samples = 1000;
    std::string default_label_type = "continuous";

    // Parse command-line arguments or use defaults.
    std::string output_csv = (argc >= 2) ? argv[1] : default_output_csv;
    int num_samples = (argc >= 3) ? std::stoi(argv[2]) : default_num_samples;
    std::string label_type = (argc >= 4) ? argv[3] : default_label_type;
    bool use_continuous = (label_type == "continuous");

    // Define default directory if none is provided in the output file.
    std::string default_directory = "../examples/gp_dynamics/data";
    fs::path out_path(output_csv);
    if (out_path.parent_path().empty()) {
        out_path = fs::path(default_directory) / out_path;
    }
    // Create the directory if it does not exist.
    fs::path out_dir = out_path.parent_path();
    if (!fs::exists(out_dir)) {
        std::cout << "Directory " << out_dir << " does not exist. Creating directory..." << std::endl;
        if (!fs::create_directories(out_dir)) {
            std::cerr << "Failed to create directory: " << out_dir << std::endl;
            return 1;
        }
    }

    // --- Pendulum Parameters ---
    double dt = 0.02;              // time step [s]
    double length = 1.0;           // pendulum length [m]
    double mass = 1.0;             // pendulum mass [kg]
    double damping = 0.01;         // damping coefficient
    std::string integration_type = "rk4";  // integration method

    // Create the pendulum object.
    Pendulum pendulum(dt, length, mass, damping, integration_type);

    // --- Random Number Generators ---
    std::random_device rd;
    std::mt19937 gen(rd());
    // For theta: uniformly sample in [-pi, pi] then wrap to [0, 2pi)
    std::uniform_real_distribution<> theta_dist(-3.14159, 3.14159);
    // For theta_dot: uniformly sample in [-5, 5]
    std::uniform_real_distribution<> theta_dot_dist(-5.0, 5.0);
    // For control (torque): uniformly sample in [-1, 1]
    std::uniform_real_distribution<> torque_dist(-1.0, 1.0);

    // --- Open CSV File for Dataset Generation ---
    std::ofstream ofs(out_path.string());
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << out_path.string() << std::endl;
        return 1;
    }
    // Write CSV header.
    ofs << "theta,theta_dot,torque,label1,label2\n";

    // --- Generate Dataset ---
    for (int i = 0; i < num_samples; ++i) {
        double theta = theta_dist(gen);
        theta = wrapTheta(theta);
        double theta_dot = theta_dot_dist(gen);
        double torque = torque_dist(gen);

        // Build the state and control vectors.
        Eigen::VectorXd state(2);
        state << theta, theta_dot;
        Eigen::VectorXd control(1);
        control << torque;

        Eigen::VectorXd label;
        if (use_continuous) {
            label = pendulum.getContinuousDynamics(state, control);
        } else {
            label = pendulum.getDiscreteDynamics(state, control);
        }

        ofs << state(0) << "," << state(1) << "," << control(0) << ",";
        ofs << label(0) << "," << label(1) << "\n";
    }
    ofs.close();
    std::cout << "Dataset generation complete: " << num_samples 
              << " samples written to " << out_path.string() << std::endl;

    // --- Plotting State Trajectories ---
    int horizon = 200;       // number of time steps per trajectory
    int nTrajectories = 10;  // number of trajectories to simulate

    // Containers for storing trajectories.
    std::vector<std::vector<double>> trajectories_theta(nTrajectories, std::vector<double>(horizon));
    std::vector<std::vector<double>> trajectories_theta_dot(nTrajectories, std::vector<double>(horizon));
    std::vector<double> time(horizon);
    for (int t = 0; t < horizon; ++t) {
        time[t] = t * dt;
    }

    // For plotting, sample initial theta directly in [0, 2pi].
    std::uniform_real_distribution<> theta_init_dist(0.0, 2 * 3.14159);
    for (int traj = 0; traj < nTrajectories; ++traj) {
        double init_theta = theta_init_dist(gen);
        double init_theta_dot = theta_dot_dist(gen);
        Eigen::VectorXd state(2);
        state << init_theta, init_theta_dot;

        trajectories_theta[traj][0] = state(0);
        trajectories_theta_dot[traj][0] = state(1);

        // Propagate the system for the given horizon.
        for (int t = 1; t < horizon; ++t) {
            double torque = torque_dist(gen);
            Eigen::VectorXd control(1);
            control << torque;

            state = pendulum.getDiscreteDynamics(state, control);
            // Wrap theta value after propagation.
            state(0) = wrapTheta(state(0));

            trajectories_theta[traj][t] = state(0);
            trajectories_theta_dot[traj][t] = state(1);
        }
    }

    // Plot trajectories using matplotlibcpp.
    namespace plt = matplotlibcpp;
    
    // Plot Theta trajectories.
    plt::figure();
    for (int traj = 0; traj < nTrajectories; ++traj) {
        plt::plot(time, trajectories_theta[traj]);
    }
    plt::title("Pendulum Theta Trajectories (wrapped to [0, 2pi])");
    plt::xlabel("Time [s]");
    plt::ylabel("Theta [rad]");

    // Plot Theta_dot trajectories.
    plt::figure();
    for (int traj = 0; traj < nTrajectories; ++traj) {
        plt::plot(time, trajectories_theta_dot[traj]);
    }
    plt::title("Pendulum Theta_dot Trajectories");
    plt::xlabel("Time [s]");
    plt::ylabel("Theta_dot [rad/s]");

    plt::show();

    return 0;
}
