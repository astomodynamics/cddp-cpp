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

TEST(CDDPTest, SolveLogCDDP) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a dubins car instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(timestep, integration_type); // Create unique_ptr

    // Create objective function
    Eigen::MatrixXd Q = 0.1 * Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q(2, 2) = 0.0;
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 50.0, 0.0, 0.0,
          0.0, 50.0, 0.0,
          0.0, 0.0, 10.0;
    Qf = 0.5 * Qf;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 3.0, 2.0, M_PI/2.0;

    // Create an empty vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> empty_reference_states; 
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial and target states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0; 

    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 30;
    options.cost_tolerance = 1e-5;
    options.grad_tolerance = 1e-5;
    options.use_parallel = false;
    options.num_threads = 1;
    options.regularization_type = "control";
    options.regularization_control = 1e-3;
    options.barrier_coeff = 1e-1;
    options.verbose = true;
    options.debug = true;
    options.ms_segment_length = horizon;
    options.ms_rollout_type = "nonlinear";
    options.ms_defect_tolerance_for_single_shooting = 1e-5;

    // Create CDDP solver
    cddp::CDDP cddp_solver(
      initial_state, 
      goal_state, 
      horizon, 
      timestep, 
      std::make_unique<cddp::Unicycle>(timestep, integration_type), 
      std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep), 
      options);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Update goal state (for test)
    goal_state << 2.0, 2.0, M_PI/2.0;
    cddp_solver.setReferenceState(goal_state);

    // Define control box constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    cddp_solver.addConstraint("ControlConstraint", 
        std::make_unique<cddp::ControlConstraint>( control_upper_bound));

    // // Define state box constraints
    // Eigen::VectorXd state_lower_bound(state_dim);
    // state_lower_bound << -0.1, -0.1, -10.0;
    // Eigen::VectorXd state_upper_bound(state_dim);
    // state_upper_bound << 2.5, 2.5, 10.0;
    // cddp_solver.addConstraint(std::string("StateBoxConstraint"), std::make_unique<cddp::StateBoxConstraint>(state_lower_bound, state_upper_bound));
    // auto state_box_constraint = cddp_solver.getConstraint<cddp::StateBoxConstraint>("StateBoxConstraint");

    // Define ball constraint
    double radius = 0.2;
    Eigen::Vector2d center(1.0, 1.0);
    // cddp_solver.addConstraint(std::string("BallConstraint"), std::make_unique<cddp::BallConstraint>(radius, center, 0.1));

    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    std::vector<double> x_arr0(horizon + 1, 0.0);
    std::vector<double> y_arr0(horizon + 1, 0.0);
    X[0] = initial_state;
    double v = 0.01;
    for (int i = 0; i < horizon; ++i) {
        X[i+1] = initial_state;
    }
    // for (int i = 0; i < horizon; ++i) {
    //       double x = X[i](0);
    //       double y = X[i](1);
    //       double theta = X[i](2);
  
    //       x_arr0[i] = X[i](0);
    //       y_arr0[i] = X[i](1);
    //       X[i+1] = Eigen::Vector3d(x + v * cos(theta), y + v * sin(theta), theta - 0.01);
    //   }

    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve("LogCDDP");
    solution.converged = true;
    ASSERT_TRUE(solution.converged);
}

// Create gif from images using ImageMagick
// Installation:
// $ sudo apt-get install imagemagick

// convert -delay 100 ../results/tests/dubins_car_*.png ../results/tests/dubins_car.gif 