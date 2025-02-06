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
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 100;
    double timestep = 0.02;
    // Create a pendulum instance 
    double mass = 1.0; 
    double length = 0.5; 
    double damping = 0.01;
    std::string integration_type = "euler";

    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Pendulum>(timestep, length, mass, damping, integration_type);

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0, 0.0,
        0.0, 100.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0;  // Upright position with zero velocity

    std::vector<Eigen::VectorXd> empty_reference_states;
    // empty_reference_states.back() << 0.0, 0.0;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);
    
    // Initial state (pendulum pointing down)
    Eigen::VectorXd initial_state(state_dim);
    initial_state << M_PI, 0.0;  // Zero angle and angular velocity

    // Construct zero control sequence
    std::vector<Eigen::VectorXd> zero_control_sequence(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Construct initial trajectory 
    std::vector<Eigen::VectorXd> X_init(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    for (int t = 0; t < horizon + 1; ++t) {
        X_init[t] = initial_state;
    }

    // Calculate initial cost
    double J = 0.0;
    for (int t = 0; t < horizon; ++t) {
        J += objective->running_cost(X_init[t], zero_control_sequence[t], t);
    }
    J += objective->terminal_cost(X_init[horizon]);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -100.0;  // Maximum negative torque
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 100.0;   // Maximum positive torque
    
    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 10;
    options.regularization_type = "none";
    options.regularization_control = 1e-7;
    cddp_solver.setOptions(options);


    // Initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon + 1; ++i) {
        X[i] = initial_state;
    }
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
    std::vector<double> theta_arr, theta_dot_arr, torque_arr;
    for (const auto& x : X_sol) {
        theta_arr.push_back(x(0));
        theta_dot_arr.push_back(x(1));
    }
    for (const auto& u : U_sol) {
        torque_arr.push_back(u(0));
    }

     // Plot results
    plt::subplot(2, 1, 1);
    plt::named_plot("Angle", theta_arr);
    plt::named_plot("Angular Velocity", theta_dot_arr);
    plt::title("State Trajectory");
    plt::legend();

    plt::subplot(2, 1, 2);
    plt::named_plot("Torque", torque_arr);
    plt::title("Control Input");
    plt::legend();
    
    plt::save(plotDirectory + "/pendulum_cddp_test.png");

    // Animation
    plt::figure_size(800, 600);

    double pendulum_length = length;
    double bob_radius = 0.1;

    for (int i = 0; i < X_sol.size(); ++i) {
        if (i % 5 == 0) {
            plt::clf();

            double theta = theta_arr[i];
            
            // Calculate pendulum bob position (x, y)
            double x = pendulum_length * std::sin(theta);
            double y = pendulum_length * std::cos(theta);

            // Plot pendulum rod
            std::vector<double> rod_x = {0, x};
            std::vector<double> rod_y = {0, y};
            plt::plot(rod_x, rod_y, "k-");

            // Plot pendulum bob (circle)
            std::vector<double> circle_x, circle_y;
            for (int j = 0; j <= 50; ++j) {
                double t = 2 * M_PI * j / 50;
                circle_x.push_back(x + bob_radius * std::cos(t));
                circle_y.push_back(y + bob_radius * std::sin(t));
            }
            plt::plot(circle_x, circle_y, "b-");

            // Plot trajectory trace
            std::vector<double> trace_x, trace_y;
            for (int j = std::max(0, i - 50); j < i; ++j) {
                trace_x.push_back(pendulum_length * std::sin(theta_arr[j]));
                trace_y.push_back(pendulum_length * std::cos(theta_arr[j]));
            }
            if (!trace_x.empty()) {
                plt::plot(trace_x, trace_y, "r--");
            }

            // Set fixed axis limits for stable animation
            plt::xlim(-0.7, 0.7);
            plt::ylim(-0.7, 0.7);
            // plt::axis("equal");  // Make the aspect ratio 1:1

            // Save frame
            std::string filename = plotDirectory + "/pendulum_frame_" + std::to_string(i) + ".png";
            plt::save(filename);
            
            plt::pause(0.01);
        }
    }

}

// Create gif from images using ImageMagick:
// convert -delay 100 ../results/tests/pendulum_frame_*.png ../results/tests/pendulum.gif