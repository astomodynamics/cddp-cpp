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
    int state_dim = 4;  // [x, y, theta, v]
    int control_dim = 2;  // [acceleration, steering_angle]
    int horizon = 100;
    double timestep = 0.1;
    std::string integration_type = "euler";

    // Create a bicycle instance 
    double wheelbase = 1.5;  // wheelbase length in meters
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Bicycle>(timestep, wheelbase, integration_type);

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    // Weights for final state [x, y, theta, v]
    Qf << 50.0, 0.0, 0.0, 0.0,
          0.0, 50.0, 0.0, 0.0,
          0.0, 0.0, 10.0, 0.0,
          0.0, 0.0, 0.0, 10.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 5.0, 5.0, M_PI/2.0, 0.0;  // Target: [x=2, y=2, theta=90deg, v=0]

    // Create an empty vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> empty_reference_states; 
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state: [x=0, y=0, theta=45deg, v=1.0]
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0, 1.0;

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define constraints for [acceleration, steering_angle]
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -10.0, -M_PI/3;  // Max deceleration and steering angle
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 10.0, M_PI/3;    // Max acceleration and steering angle
    
    // Add the constraint to the solver
    cddp_solver.addConstraint(std::string("ControlBoxConstraint"), 
                            std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Set options
    cddp::CDDPOptions options;
    options.max_iterations = 10;
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve();

    // Extract solution
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create directory for saving plot
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Extract trajectory data
    std::vector<double> x_arr, y_arr, theta_arr, v_arr;
    for (const auto& x : X_sol) {
        x_arr.push_back(x(0));
        y_arr.push_back(x(1));
        theta_arr.push_back(x(2));
        v_arr.push_back(x(3));
    }

    // Extract control inputs
    std::vector<double> acc_arr, steering_arr;
    for (const auto& u : U_sol) {
        acc_arr.push_back(u(0));
        steering_arr.push_back(u(1));
    }

    // Plot states and controls
    plt::figure_size(1200, 800);
    
    // State trajectories
    plt::subplot(2, 2, 1);
    plt::plot(x_arr, y_arr);
    plt::title("Position Trajectory");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");

    plt::subplot(2, 2, 2);
    plt::plot(t_sol, theta_arr);
    plt::title("Heading Angle");
    plt::xlabel("Time [s]");
    plt::ylabel("theta [rad]");

    plt::subplot(2, 2, 3);
    plt::plot(t_sol, v_arr);
    plt::title("Velocity");
    plt::xlabel("Time [s]");
    plt::ylabel("v [m/s]");

    // Control inputs
    plt::subplot(2, 2, 4);
    // plt::plot(acc_arr, "b-", {{"label", "Acceleration"}});
    // plt::plot(steering_arr, "r--", {{"label", "Steering"}});
    // plt::title("Control Inputs");
    // plt::xlabel("Step");
    // plt::ylabel("Control");
    // plt::legend();
    
    // plt::save(plotDirectory + "/bicycle_cddp_results.png");

    // Animation
    plt::figure_size(800, 600);
    plt::title("Bicycle Trajectory");
    plt::xlabel("x [m]");
    plt::ylabel("y [m]");
    plt::xlim(-1, 6);
    plt::ylim(-1, 6);

    // Bicycle dimensions
    double car_length = 0.3;
    double car_width = 0.15;

    // Animation loop
    for (int i = 0; i < X_sol.size(); ++i) {
        if (i % 5 == 0) {
            plt::clf();

            // Plot bicycle as a rectangle with heading
            double x = x_arr[i];
            double y = y_arr[i];
            double theta = theta_arr[i];
            
            // Calculate corners
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

            car_x[4] = car_x[0];
            car_y[4] = car_y[0];

            // Plot bicycle shape
            plt::plot(car_x, car_y, "k-");

            // Plot wheels (as small lines)
            double wheel_length = car_width * 0.8;
            // Front wheel (steerable)
            double steering_angle = U_sol[std::min(i, (int)U_sol.size()-1)](1);
            double front_wheel_x[2] = {x + car_length/2 * cos(theta), 
                                     x + car_length/2 * cos(theta) + wheel_length * cos(theta + steering_angle)};
            double front_wheel_y[2] = {y + car_length/2 * sin(theta),
                                     y + car_length/2 * sin(theta) + wheel_length * sin(theta + steering_angle)};
            plt::plot(std::vector<double>(front_wheel_x, front_wheel_x + 2),
                     std::vector<double>(front_wheel_y, front_wheel_y + 2), "r-");

            // Plot trajectory
            plt::plot(std::vector<double>(x_arr.begin(), x_arr.begin() + i + 1), 
                     std::vector<double>(y_arr.begin(), y_arr.begin() + i + 1), "b-");

            plt::title("Bicycle Trajectory");
            plt::xlabel("x [m]");
            plt::ylabel("y [m]");
            plt::xlim(-1, 6);
            plt::ylim(-1, 6);
            plt::legend();

            std::string filename = plotDirectory + "/bicycle_" + std::to_string(i) + ".png";
            plt::save(filename);
            plt::pause(0.01);
        }
    }
}

// Create gif from images using ImageMagick
// Installation:
// $ sudo apt-get install imagemagick

// convert -delay 100 ../results/tests/bicycle_*.png ../results/tests/bicycle.gif