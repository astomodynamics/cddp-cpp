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

#include "cddp.hpp"  // Adjust include as needed for your CDDP framework and DubinsCar definition

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main() {
    // Problem parameters
    const int state_dim = 3;        // [x, y, theta]
    const int control_dim = 1;      // [omega]
    const int horizon = 100;        // planning horizon
    const double timestep = 0.03;   // integration step
    const std::string integration_type = "euler";

    // Create a DubinsCar instance (constant speed + single steering input)
    double forward_speed = 1.0;  // For example, 1.0 m/s
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::DubinsCar>(forward_speed, timestep, integration_type);

    // Create objective function
    // State cost matrix Q (typically zero or small if you only penalize final state heavily)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);

    // Control cost matrix R
    // For single control dimension, this is a 1x1 matrix:
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    // Final state cost matrix Qf
    // For example, a heavier penalty on final position/orientation
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0,0) = 50.0;  // x
    Qf(1,1) = 50.0;  // y
    Qf(2,2) = 10.0;  // theta
    Qf = 0.5 * Qf;   // scaling

    // Goal state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI / 2.0;

    // Create an empty reference-state sequence (if needed for time-varying references)
    std::vector<Eigen::VectorXd> empty_reference_states; 

    // Build the quadratic objective
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define box constraints on control (here, single dimension: -pi to pi)
    Eigen::VectorXd control_lower_bound(control_dim);
    Eigen::VectorXd control_upper_bound(control_dim);
    control_lower_bound << -M_PI;  // min turn rate
    control_upper_bound <<  M_PI;  // max turn rate

    // Add the constraint to the solver
    cddp_solver.addConstraint(
        "ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // (Optional) retrieve the constraint object
    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Set CDDP options
    cddp::CDDPOptions options;
    options.max_iterations = 10;   // for demonstration
    options.barrier_coeff = 1e-2;
    options.barrier_factor = 0.1;
    cddp_solver.setOptions(options);

    // Set an initial guess for the trajectory
    // States (X) is horizon+1 in length, each dimension: 3
    // Controls (U) is horizon in length, each dimension: 1
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon,      Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    // Possible algorithms: "CLCDDP", "LogCDDP", etc. 
    cddp::CDDPSolution solution = cddp_solver.solve("LogCDDP");

    // Extract the solution sequences
    auto X_sol = solution.state_sequence;   // size: horizon + 1
    auto U_sol = solution.control_sequence; // size: horizon
    auto t_sol = solution.time_sequence;    // size: horizon + 1

    // Create directory for saving plots (if it doesn't exist)
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Gather data for plotting
    std::vector<double> x_arr, y_arr, theta_arr;
    for (const auto& x : X_sol) {
        x_arr.push_back(x(0));
        y_arr.push_back(x(1));
        theta_arr.push_back(x(2));
    }

    // For single-dim control: just store the steering rate
    std::vector<double> omega_arr;
    for (const auto& u : U_sol) {
        omega_arr.push_back(u(0));
    }

    // Plot state trajectory & control
    plt::subplot(2, 1, 1);
    plt::plot(x_arr, y_arr);
    plt::title("DubinsCar State Trajectory");
    plt::xlabel("x");
    plt::ylabel("y");

    plt::subplot(2, 1, 2);
    plt::plot(omega_arr);
    plt::title("Steering Rate Control (omega)");
    plt::save(plotDirectory + "/dubins_car_cddp_test.png");

    // Optional: Generate an animation 
    //           (requires multiple frames, so you may want to store and 
    //            convert them to a GIF afterward).
    plt::figure_size(800, 600);
    plt::title("DubinsCar Trajectory");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::xlim(-1, 3);
    plt::ylim(-1, 3);

    // Car dimensions
    double car_length = 0.2;
    double car_width  = 0.1;

    // Animation loop
    for (int i = 0; i < static_cast<int>(X_sol.size()); ++i) {
        if (i % 5 == 0) {
            plt::clf();

            // Current pose
            double x     = x_arr[i];
            double y     = y_arr[i];
            double theta = theta_arr[i];

            // Compute corners of the car rectangle
            std::vector<double> car_x(5), car_y(5);

            // Front-left corner
            car_x[0] = x +  car_length/2.0 * cos(theta) - car_width/2.0 * sin(theta);
            car_y[0] = y +  car_length/2.0 * sin(theta) + car_width/2.0 * cos(theta);

            // Front-right corner
            car_x[1] = x +  car_length/2.0 * cos(theta) + car_width/2.0 * sin(theta);
            car_y[1] = y +  car_length/2.0 * sin(theta) - car_width/2.0 * cos(theta);

            // Rear-right corner
            car_x[2] = x -  car_length/2.0 * cos(theta) + car_width/2.0 * sin(theta);
            car_y[2] = y -  car_length/2.0 * sin(theta) - car_width/2.0 * cos(theta);

            // Rear-left corner
            car_x[3] = x -  car_length/2.0 * cos(theta) - car_width/2.0 * sin(theta);
            car_y[3] = y -  car_length/2.0 * sin(theta) + car_width/2.0 * cos(theta);

            // Close the shape
            car_x[4] = car_x[0];
            car_y[4] = car_y[0];

            // Plot the car
            plt::plot(car_x, car_y, "k-");        

            // Plot the trajectory so far
            plt::plot(std::vector<double>(x_arr.begin(), x_arr.begin() + i + 1),
                      std::vector<double>(y_arr.begin(), y_arr.begin() + i + 1),
                      "b-");

            // Title and axes
            plt::title("DubinsCar Trajectory");
            plt::xlabel("x");
            plt::ylabel("y");
            plt::xlim(-1, 3);
            plt::ylim(-1, 3);

            // Save frame
            std::string filename = plotDirectory + "/dubins_car_frame_" + std::to_string(i) + ".png";
            plt::save(filename);

            // Brief pause for "animation" effect
            plt::pause(0.01);
        }
    }

    // Optionally, you can convert frames to a GIF via ImageMagick:
    //   convert -delay 100 ../results/tests/dubins_car_frame_*.png ../results/tests/dubins_car.gif

    return 0;
}
