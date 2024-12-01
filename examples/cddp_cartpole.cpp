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
#include <filesystem>

#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main() {
    int state_dim = 4;
    int control_dim = 1;
    int horizon = 100;
    double timestep = 0.05;

    // Create a cartpole instance with custom parameters
    double cart_mass = 1.0;
    double pole_mass = 0.2;
    double pole_length = 0.5;
    double gravity = 9.81;
    double damping = 0.0; // TODO: Implement damping term
    std::string integration_type = "rk4";

    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::CartPole>(
        timestep, integration_type, cart_mass, pole_mass, pole_length, gravity, damping);

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0,0) = 50.0;  // Final cart position cost
    Qf(1,1) = 50.0;   // Final cart velocity cost
    Qf(2,2) = 50.0;  // Final pole angle cost
    Qf(3,3) = 50.0;   // Final pole angular velocity cost

    // Goal state: cart at origin, pole upright, zero velocities
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state << 0.0, M_PI, 0.0, 0.0;

    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state (cart at rest, pole hanging down)
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, 0.0, 0.0;  // x=0, theta=0.0, v=0, dtheta=0

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -100.0;  // Maximum negative force
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 100.0;   // Maximum positive force
    
    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 50;
    options.max_line_search_iterations = 11;
    options.regularization_type = "control";
    options.regularization_control = 1e-8;
    cddp_solver.setOptions(options);

    // Initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve
    cddp::CDDPSolution solution = cddp_solver.solve();

    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create plot directory
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Extract solution data
    std::vector<double> x_arr, x_dot_arr, theta_arr, theta_dot_arr, force_arr, time_arr, time_arr2;
    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        theta_arr.push_back(X_sol[i](1));
        x_dot_arr.push_back(X_sol[i](2));
        theta_dot_arr.push_back(X_sol[i](3));
    }
    for (size_t i = 0; i < U_sol.size(); ++i) {
        force_arr.push_back(U_sol[i](0));
        time_arr2.push_back(t_sol[i]);
    }

    // Plot results (2x2)
    plt::figure_size(1200, 800);
    plt::subplot(2, 2, 1);
    plt::title("Cart Position");
    plt::plot(time_arr, x_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::grid(true);

    plt::subplot(2, 2, 2);
    plt::title("Cart Velocity");
    plt::plot(time_arr, x_dot_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [m/s]");
    plt::grid(true);

    plt::subplot(2, 2, 3);
    plt::title("Pole Angle");
    plt::plot(time_arr, theta_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [rad]");
    plt::grid(true);

    plt::subplot(2, 2, 4);
    plt::title("Pole Angular Velocity");
    plt::plot(time_arr, theta_dot_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Angular Velocity [rad/s]");
    plt::grid(true);

    plt::tight_layout();
    plt::save(plotDirectory + "/cartpole_results.png");
    plt::clf();

    // Plot control inputs
    plt::figure_size(800, 600);
    plt::title("Control Inputs");
    plt::plot(time_arr2, force_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Force [N]");
    plt::grid(true);

    plt::save(plotDirectory + "/cartpole_control_inputs.png");
    plt::clf();


    // Animation
    plt::figure_size(800, 600);
    plt::title("CartPole Animation");
    plt::xlabel("x");
    plt::ylabel("y");

    double cart_width = 0.3;
    double cart_height = 0.2;
    double pole_width = 0.05;

    for (int i = 0; i < X_sol.size(); ++i) {
        if (i % 10 == 0) {
            plt::clf();

            double x = x_arr[i];
            double theta = theta_arr[i];

            // Cart corners
            std::vector<double> cart_x = {
                x - cart_width/2, x + cart_width/2,
                x + cart_width/2, x - cart_width/2,
                x - cart_width/2
            };
            std::vector<double> cart_y = {
                -cart_height/2, -cart_height/2,
                cart_height/2, cart_height/2,
                -cart_height/2
            };
            plt::plot(cart_x, cart_y, "k-");

            // Pole
            double pole_end_x = x + pole_length * std::sin(theta);
            double pole_end_y = -pole_length * std::cos(theta);
            std::vector<double> pole_x = {x, pole_end_x};
            std::vector<double> pole_y = {0, pole_end_y};
            plt::plot(pole_x, pole_y, "b-");

            // Plot pole bob
            std::vector<double> circle_x, circle_y;
            for (int j = 0; j <= 20; ++j) {
                double t = 2 * M_PI * j / 20;
                circle_x.push_back(pole_end_x + pole_width * std::cos(t));
                circle_y.push_back(pole_end_y + pole_width * std::sin(t));
            }
            plt::plot(circle_x, circle_y, "b-");

            // Set fixed axis limits for stable animation
            double view_width = 3.0;
            plt::xlim(-0.5 - view_width/2, 0.5 + view_width/2);
            plt::ylim(-view_width/2, view_width/2);
            // plt::axis("equal");

            std::string filename = plotDirectory + "/cartpole_frame_" + std::to_string(i) + ".png";
            plt::save(filename);
            plt::pause(0.01);
        }
    }
}

// Create gif from images using ImageMagick:
// convert -delay 5 ../results/tests/cartpole_frame_*.png ../results/tests/cartpole.gif