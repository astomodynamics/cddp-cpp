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
#include "sqp_core/sqp.hpp"

TEST(SQPTest, SolveUnicycle) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(
        timestep, integration_type
    );

    // Create objective function
    Eigen::MatrixXd Q = 5 * Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 1200.0, 0.0, 0.0,
          0.0, 1200.0, 0.0,
          0.0, 0.0, 700.0;

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
    options.max_iterations = 20;
    options.min_iterations = 3;
    options.ftol = 1e-6;
    options.xtol = 1e-6;
    options.gtol = 1e-6;
    options.eta = 1.0;
    options.trust_region_radius = 100.0;
    options.merit_penalty = 100.0;
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
    std::cout << "Initial state: " << initial_state.transpose() << std::endl;
    std::cout << "Final state: " << solution.X.back().transpose() << std::endl;
    std::cout << "Goal state: " << goal_state.transpose() << std::endl;
    EXPECT_NEAR((solution.X.front() - initial_state).norm(), 0.0, 1e-3);
    // EXPECT_NEAR((solution.X.back() - goal_state).norm(), 0.0, 0.1);

    // Extract trajectories for plotting
    auto X_sol = solution.X;
    auto U_sol = solution.U;
    
    // Extract states and controls
    std::vector<double> x_arr, y_arr, theta_arr;
    std::vector<double> v_arr, omega_arr;
    
    for (size_t i = 0; i < X_sol.size(); ++i) {
        x_arr.push_back(X_sol[i](0));
        y_arr.push_back(X_sol[i](1));
        theta_arr.push_back(X_sol[i](2));
        
        if (i < U_sol.size()) {
            v_arr.push_back(U_sol[i](0));
            omega_arr.push_back(U_sol[i](1));
        }
    }
}