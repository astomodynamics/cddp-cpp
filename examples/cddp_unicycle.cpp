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
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(timestep, integration_type); // Create unique_ptr

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 50.0, 0.0, 0.0,
          0.0, 50.0, 0.0,
          0.0, 0.0, 10.0;
    Qf = 0.5 * Qf;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Create an empty vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> empty_reference_states; 
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial and target states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0; 

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    
    // Add the constraint to the solver
    cddp_solver.addConstraint(std::string("ControlBoxConstraint"), std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Set options
    cddp::CDDPOptions options;
    options.max_iterations = 10;
    options.barrier_coeff = 1e-2;
    options.barrier_factor = 0.1;
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    // cddp::CDDPSolution solution = cddp_solver.solve("CLCDDP");
    cddp::CDDPSolution solution = cddp_solver.solve("LogCDDP");

    // Extract solution
    auto X_sol = solution.state_sequence; // size: horizon + 1
    auto U_sol = solution.control_sequence; // size: horizon
    auto t_sol = solution.time_sequence; // size: horizon + 1

    // Create directory for saving plot (if it doesn't exist)
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Plot the solution (x-y plane)
    std::vector<double> x_arr, y_arr, theta_arr;
    for (const auto& x : X_sol) {
        x_arr.push_back(x(0));
        y_arr.push_back(x(1));
        theta_arr.push_back(x(2));
    }

    // Plot the solution (control inputs)
    std::vector<double> v_arr, omega_arr;
    for (const auto& u : U_sol) {
        v_arr.push_back(u(0));
        omega_arr.push_back(u(1));
    }

    // Plot the solution by subplots
    plt::subplot(2, 1, 1);
    plt::plot(x_arr, y_arr);
    plt::title("State Trajectory");
    plt::xlabel("x");
    plt::ylabel("y");

    plt::subplot(2, 1, 2);
    plt::plot(v_arr);
    plt::plot(omega_arr);
    plt::title("Control Inputs");
    plt::save(plotDirectory + "/unicycle_cddp_test.png");

    // Create figure and axes
    plt::figure_size(800, 600);
    plt::title("unicycle Trajectory");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::xlim(-1, 3); // Adjust limits as needed
    plt::ylim(-1, 3); // Adjust limits as needed

    // Car dimensions
    double car_length = 0.2;
    double car_width = 0.1;

    // Animation function
    for (int i = 0; i < X_sol.size(); ++i) {

        if (i % 5 == 0) {
            // Clear previous plot
			plt::clf();

            // Plot car as a box with heading
            double x = x_arr[i];
            double y = y_arr[i];
            double theta = theta_arr[i];
            
            // Calculate car corner points
            std::vector<double> car_x(5);
            std::vector<double> car_y(5);

            car_x[0] = x + car_length/2 * cos(theta) - car_width/2 * sin(theta);
            car_y[0] = y + car_length/2 * sin(theta) + car_width/2 * cos(theta);

            car_x[1] = x + car_length/2 * cos(theta) + car_width/2 * sin(theta);
            car_y[1] = y + car_length/2 * sin(theta) - car_width/2 * cos(theta);

            car_x[2] = x - car_length/2 * cos(theta) + car_width/2 * sin(theta);
            car_y[2] = y - car_length/2 * sin(theta) - car_width/2 * cos(theta);

            car_x[3] = x - car_length/2 * cos(theta) - car_width/2 * sin(theta);
            car_y[3] = y - car_length/2 * sin(theta) + car_width/2 * cos(theta);

            car_x[4] = car_x[0]; // Close the shape
            car_y[4] = car_y[0];

            // Plot the car
            plt::plot(car_x, car_y, "k-");        

            // Plot trajectory up to current point
            plt::plot(std::vector<double>(x_arr.begin(), x_arr.begin() + i + 1), 
                    std::vector<double>(y_arr.begin(), y_arr.begin() + i + 1), "b-");

            // Add plot title
            plt::title("unicycle Trajectory");

            // Set labels
            plt::xlabel("x");
            plt::ylabel("y");

            // Adjust limits as needed
            plt::xlim(-1, 3); 
            plt::ylim(-1, 3); 

            // Enable legend
            plt::legend();

            // Save current frame as an image
            std::string filename = plotDirectory + "/unicycle_frame_" + std::to_string(i) + ".png";
            plt::save(filename);
            
            // Display plot continuously
            plt::pause(0.01); // Pause for a short time

        }
    };
}

// Create gif from images using ImageMagick
// Installation:
// $ sudo apt-get install imagemagick

// convert -delay 100 ../results/tests/unicycle_frame_*.png ../results/tests/unicycle.gif 