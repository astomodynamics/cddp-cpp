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
#include "cddp.hpp"  // Assumes this header includes the CartPole class and matplotlibcpp.hpp

using namespace cddp;
namespace fs = std::filesystem;

double wrapAngle(double theta) {
    double two_pi = 2.0 * M_PI;
    theta = std::fmod(theta, two_pi);
    if (theta < 0)
        theta += two_pi;
    return theta;
}

int main(int argc, char* argv[]) {
    // --- Dataset Generation Parameters ---
    std::string default_output_csv = "cartpole_dataset.csv";
    int default_num_samples = 1000;
    std::string default_label_type = "continuous";  // Use continuous dynamics by default

    // Parse command-line arguments or use defaults.
    std::string output_csv = (argc >= 2) ? argv[1] : default_output_csv;
    int num_samples = (argc >= 3) ? std::stoi(argv[2]) : default_num_samples;
    std::string label_type = (argc >= 4) ? argv[3] : default_label_type;
    bool use_continuous = (label_type == "continuous");

    // Define default directory if no directory is provided.
    std::string default_directory = "../examples/gp_dynamics/data";
    fs::path out_path(output_csv);
    if (out_path.parent_path().empty()) {
        out_path = fs::path(default_directory) / out_path;
    }
    // Create directory if it does not exist.
    fs::path out_dir = out_path.parent_path();
    if (!fs::exists(out_dir)) {
        std::cout << "Directory " << out_dir << " does not exist. Creating directory..." << std::endl;
        if (!fs::create_directories(out_dir)) {
            std::cerr << "Failed to create directory: " << out_dir << std::endl;
            return 1;
        }
    }

    // --- Set up CartPole Parameters ---
    double dt = 0.02;              // time step [s]
    std::string integration_type = "rk4";  // integration method
    // Typical CartPole parameters (adjust as needed):
    double cart_mass = 1.0;
    double pole_mass = 0.1;
    double pole_length = 0.5;
    double gravity = 9.81;
    double damping = 0.01;

    // Create the CartPole object.
    CartPole cartpole(dt, integration_type, cart_mass, pole_mass, pole_length, gravity, damping);

    // --- Set up Random Number Generators for State and Control ---
    std::random_device rd;
    std::mt19937 gen(rd());
    // For the cart position x: typically in a limited range (e.g., [-2.4, 2.4])
    std::uniform_real_distribution<> x_dist(-2.4, 2.4);
    // For the pole angle theta: assume small angles (e.g., [-0.2, 0.2] radians)
    std::uniform_real_distribution<> theta_dist(-0.2, 0.2);
    // For the cart velocity x_dot: e.g., [-1, 1]
    std::uniform_real_distribution<> x_dot_dist(-1.0, 1.0);
    // For the pole angular velocity theta_dot: e.g., [-1, 1]
    std::uniform_real_distribution<> theta_dot_dist(-1.0, 1.0);
    // For the control force: e.g., [-10, 10]
    std::uniform_real_distribution<> force_dist(-10.0, 10.0);

    // --- Open CSV File for Dataset Generation ---
    std::ofstream ofs(out_path.string());
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << out_path.string() << std::endl;
        return 1;
    }
    // Write CSV header.
    ofs << "x,theta,x_dot,theta_dot,force,label1,label2,label3,label4\n";

    // --- Generate Dataset ---
    for (int i = 0; i < num_samples; ++i) {
        double x = x_dist(gen);
        double theta = theta_dist(gen);
        double x_dot = x_dot_dist(gen);
        double theta_dot = theta_dot_dist(gen);
        double force = force_dist(gen);

        // Build state and control vectors.
        Eigen::VectorXd state(4);
        state << x, theta, x_dot, theta_dot;
        Eigen::VectorXd control(1);
        control << force;

        Eigen::VectorXd label;
        if (use_continuous) {
            label = cartpole.getContinuousDynamics(state, control);
        } else {
            label = cartpole.getDiscreteDynamics(state, control);
        }

        ofs << state(0) << "," << state(1) << "," 
            << state(2) << "," << state(3) << "," 
            << control(0) << ",";
        ofs << label(0) << "," << label(1) << ","
            << label(2) << "," << label(3) << "\n";
    }
    ofs.close();
    std::cout << "Dataset generation complete: " << num_samples 
              << " samples written to " << out_path.string() << std::endl;

    // --- Simulate and Plot Time-Series Trajectories ---
    int horizon = 200;       // time steps per trajectory
    int nTrajectories = 10;  // number of trajectories to simulate

    // Containers for storing trajectories for each state variable.
    std::vector<std::vector<double>> traj_x(nTrajectories, std::vector<double>(horizon));
    std::vector<std::vector<double>> traj_theta(nTrajectories, std::vector<double>(horizon));
    std::vector<std::vector<double>> traj_x_dot(nTrajectories, std::vector<double>(horizon));
    std::vector<std::vector<double>> traj_theta_dot(nTrajectories, std::vector<double>(horizon));
    std::vector<double> time(horizon);
    for (int t = 0; t < horizon; ++t) {
        time[t] = t * dt;
    }

    // Simulate trajectories.
    for (int traj = 0; traj < nTrajectories; ++traj) {
        // Sample a random initial state.
        double init_x = x_dist(gen);
        double init_theta = theta_dist(gen);
        double init_x_dot = x_dot_dist(gen);
        double init_theta_dot = theta_dot_dist(gen);
        Eigen::VectorXd state(4);
        state << init_x, init_theta, init_x_dot, init_theta_dot;

        traj_x[traj][0] = state(0);
        traj_theta[traj][0] = state(1);
        traj_x_dot[traj][0] = state(2);
        traj_theta_dot[traj][0] = state(3);

        // Propagate the system.
        for (int t = 1; t < horizon; ++t) {
            double force = force_dist(gen);
            Eigen::VectorXd control(1);
            control << force;

            state = cartpole.getDiscreteDynamics(state, control);

            traj_x[traj][t] = state(0);
            traj_theta[traj][t] = state(1);
            traj_x_dot[traj][t] = state(2);
            traj_theta_dot[traj][t] = state(3);
        }
    }

    // --- Plot using matplotlibcpp ---
    namespace plt = matplotlibcpp;
    
    // Plot Cart Position (x) trajectories.
    plt::figure();
    for (int traj = 0; traj < nTrajectories; ++traj) {
        plt::plot(time, traj_x[traj]);
    }
    plt::title("Cart Position (x) Trajectories");
    plt::xlabel("Time [s]");
    plt::ylabel("x [m]");

    // Plot Pole Angle (theta) trajectories.
    plt::figure();
    for (int traj = 0; traj < nTrajectories; ++traj) {
        plt::plot(time, traj_theta[traj]);
    }
    plt::title("Pole Angle (theta) Trajectories");
    plt::xlabel("Time [s]");
    plt::ylabel("theta [rad]");

    // Optionally, plot velocities.
    plt::figure();
    for (int traj = 0; traj < nTrajectories; ++traj) {
        plt::plot(time, traj_x_dot[traj]);
    }
    plt::title("Cart Velocity (x_dot) Trajectories");
    plt::xlabel("Time [s]");
    plt::ylabel("x_dot [m/s]");

    plt::figure();
    for (int traj = 0; traj < nTrajectories; ++traj) {
        plt::plot(time, traj_theta_dot[traj]);
    }
    plt::title("Pole Angular Velocity (theta_dot) Trajectories");
    plt::xlabel("Time [s]");
    plt::ylabel("theta_dot [rad/s]");

    plt::show();

    return 0;
}
