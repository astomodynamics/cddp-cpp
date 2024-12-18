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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

TEST(SQPTest, SolveDubinsCar) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a dubins car instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::DubinsCar>(
        timestep, integration_type
    );

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 50.0, 0.0, 0.0,
          0.0, 50.0, 0.0,
          0.0, 0.0, 10.0;
    Qf = 0.5 * Qf;

    // Goal state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Create reference trajectory
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    );

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;

    // Create SQP solver
    cddp::SQPOptions options;
    options.max_iterations = 10;
    options.ftol = 1e-4;
    options.xtol = 1e-4;
    options.verbose = true;
    options.osqp_verbose = true;

    // Create SQP solver
    cddp::SQPSolver sqp_solver(initial_state, goal_state, horizon, timestep);
    sqp_solver.setDynamicalSystem(std::move(system));
    sqp_solver.setObjective(std::move(objective));
    sqp_solver.setOptions(options);

    // Define control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    
    // Add control box constraint
    sqp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound)
    );

    auto constraint = sqp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint"); 
    Eigen::VectorXd lb = constraint->getLowerBound();
    ASSERT_NE(constraint, nullptr);
    ASSERT_EQ(lb.size(), control_dim);
    ASSERT_EQ(lb, control_lower_bound);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    sqp_solver.setInitialTrajectory(X, U);

    // double cost = objective->evaluate(X, U);
    // std::cout  << "Initial cost: " << cost << std::endl;

    // Solve the problem
    cddp::SQPResult solution = sqp_solver.solve();

    // Verify solution
    // ASSERT_TRUE(solution.success);
    EXPECT_GT(solution.iterations, 0);
    // EXPECT_LT(solution.iterations, options.max_iterations);
    EXPECT_GT(solution.solve_time, 0.0);

    // Verify trajectories
    ASSERT_EQ(solution.X.size(), horizon + 1);
    ASSERT_EQ(solution.U.size(), horizon);


    // Check initial and final states
    EXPECT_NEAR((solution.X.front() - initial_state).norm(), 0.0, 1e-3);
    EXPECT_NEAR((solution.X.back() - goal_state).norm(), 0.0, 0.1);

    // // Check control bounds
    // for (const auto& u : solution.U) {
    //     EXPECT_LE((u - control_lower_bound).minCoeff(), 1e-3);
    //     EXPECT_LE((u - control_upper_bound).maxCoeff(), 1e-3);
    // }

    // // Extract trajectories for plotting
    // auto X_sol = solution.X;
    // auto U_sol = solution.U;
    
    // // Extract states and controls
    // std::vector<double> x_arr, y_arr, theta_arr;
    // std::vector<double> v_arr, omega_arr;
    
    // for (size_t i = 0; i < X_sol.size(); ++i) {
    //     x_arr.push_back(X_sol[i](0));
    //     y_arr.push_back(X_sol[i](1));
    //     theta_arr.push_back(X_sol[i](2));
        
    //     if (i < U_sol.size()) {
    //         v_arr.push_back(U_sol[i](0));
    //         omega_arr.push_back(U_sol[i](1));
    //     }
    // }

    // // Create plot directory if needed
    // const std::string plotDirectory = "../results/tests";
    // if (!fs::exists(plotDirectory)) {
    //     fs::create_directory(plotDirectory);
    // }

    // // Plot results
    // plt::figure_size(800, 600);
    // plt::subplot(2, 1, 1);
    // plt::plot(x_arr, y_arr, "b-", {{"label", "Trajectory"}});
    // plt::plot({initial_state(0)}, {initial_state(1)}, "go", {{"label", "Start"}});
    // plt::plot({goal_state(0)}, {goal_state(1)}, "ro", {{"label", "Goal"}});
    // plt::title("State Trajectory");
    // plt::xlabel("x");
    // plt::ylabel("y");
    // plt::legend();
    // plt::grid(true);

    // plt::subplot(2, 1, 2);
    // plt::plot(v_arr, "b-", {{"label", "v"}});
    // plt::plot(omega_arr, "r-", {{"label", "omega"}});
    // plt::plot(std::vector<double>(U_sol.size(), -1.0), "k--");
    // plt::plot(std::vector<double>(U_sol.size(), 1.0), "k--");
    // plt::title("Control Inputs");
    // plt::xlabel("Time step");
    // plt::ylabel("Control");
    // plt::legend();
    // plt::grid(true);

    // plt::save(plotDirectory + "/dubins_car_sqp_test.png");
    // plt::close();

    // // Also create an animation showing vehicle motion
    // plt::figure_size(800, 600);
    
    // // Car dimensions for visualization
    // double car_length = 0.2;
    // double car_width = 0.1;

    // for (size_t i = 0; i < X_sol.size(); i += 5) {
    //     plt::clf();
        
    //     // Current state
    //     double x = x_arr[i];
    //     double y = y_arr[i];
    //     double theta = theta_arr[i];

    //     // Calculate car corners
    //     std::vector<double> car_x = {
    //         x + car_length/2 * cos(theta) - car_width/2 * sin(theta),
    //         x + car_length/2 * cos(theta) + car_width/2 * sin(theta),
    //         x - car_length/2 * cos(theta) + car_width/2 * sin(theta),
    //         x - car_length/2 * cos(theta) - car_width/2 * sin(theta),
    //         x + car_length/2 * cos(theta) - car_width/2 * sin(theta)  // Close the polygon
    //     };

    //     std::vector<double> car_y = {
    //         y + car_length/2 * sin(theta) + car_width/2 * cos(theta),
    //         y + car_length/2 * sin(theta) - car_width/2 * cos(theta),
    //         y - car_length/2 * sin(theta) - car_width/2 * cos(theta),
    //         y - car_length/2 * sin(theta) + car_width/2 * cos(theta),
    //         y + car_length/2 * sin(theta) + car_width/2 * cos(theta)  // Close the polygon
    //     };

    //     // Plot trajectory up to current point
    //     plt::plot(std::vector<double>(x_arr.begin(), x_arr.begin() + i + 1),
    //              std::vector<double>(y_arr.begin(), y_arr.begin() + i + 1),
    //              "b-", {{"label", "Path"}});
                 
    //     // Plot vehicle
    //     plt::plot(car_x, car_y, "k-");
        
    //     // Plot start and goal
    //     plt::plot({initial_state(0)}, {initial_state(1)}, "go", {{"label", "Start"}});
    //     plt::plot({goal_state(0)}, {goal_state(1)}, "ro", {{"label", "Goal"}});

    //     plt::title("Dubins Car SQP Solution");
    //     plt::xlabel("x");
    //     plt::ylabel("y");
    //     plt::grid(true);
    //     plt::legend();
    //     plt::xlim(-0.5, 2.5);
    //     plt::ylim(-0.5, 2.5);

    //     std::string filename = plotDirectory + "/dubins_car_sqp_" + std::to_string(i) + ".png";
    //     plt::save(filename);
    // }

    // plt::close();
}