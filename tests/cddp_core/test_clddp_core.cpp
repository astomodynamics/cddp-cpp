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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

TEST(CDDPTest, SolveCLDDP) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a dubins car instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::DubinsCar>(timestep, integration_type); // Create unique_ptr

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

    // // Create CDDP solver
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
    options.max_iterations = 20;
    options.max_cpu_time = 1e-1;
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solveCLDDP();

    ASSERT_TRUE(solution.converged);

    // // Extract solution
    // auto X_sol = solution.state_sequence; // size: horizon + 1
    // auto U_sol = solution.control_sequence; // size: horizon
    // auto t_sol = solution.time_sequence; // size: horizon + 1

    // // Plot the solution (x-y plane)
    // std::vector<double> x_arr, y_arr;
    // for (const auto& x : X_sol) {
    //     x_arr.push_back(x(0));
    //     y_arr.push_back(x(1));
    // }

    // // Plot the solution (control inputs)
    // std::vector<double> v_arr, omega_arr;
    // for (const auto& u : U_sol) {
    //     v_arr.push_back(u(0));
    //     omega_arr.push_back(u(1));
    // }

    // // Plot the solution by subplots
    // plt::subplot(2, 1, 1);
    // plt::plot(x_arr, y_arr);
    // plt::title("State Trajectory");
    // plt::xlabel("x");
    // plt::ylabel("y");

    // plt::subplot(2, 1, 2);
    // plt::plot(v_arr);
    // plt::plot(omega_arr);
    // plt::title("Control Inputs");
    // plt::show();

    // // Assertions
    // ASSERT_TRUE(solution.converged); // Check if the solver converged
    // // Add more assertions based on expected behavior

}