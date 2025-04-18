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

TEST(CDDPTest, Solve) {
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 500;
    double timestep = 0.05;
    // Create a pendulum instance 
    double mass = 1.0; 
    double length = 1.0; 
    double damping = 0.00;
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
    control_lower_bound << -10.0;  // Maximum negative torque
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 10.0;   // Maximum positive torque
    
    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));


    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 10;
    // options.cost_tolerance = 1e-2;
    options.use_parallel = false;
    options.num_threads = 1;
    options.verbose = true;
    options.debug = true;
    options.regularization_type = "none";
    options.regularization_control = 1e-2;
    // options.mu_initial= 1e-8;
    
    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] << initial_state;
    for (int i = 0; i < horizon; ++i) {
        U[i] = Eigen::VectorXd::Zero(control_dim);
        X[i] = initial_state;
    }
    X[horizon] << initial_state;

    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;



}