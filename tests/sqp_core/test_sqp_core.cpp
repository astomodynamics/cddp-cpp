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
#include "sqp_core/sqp_core.hpp"   

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

TEST(SQPIPOPTTest, CheckPointers) {
    // Problem parameters.
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle dynamical system.
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Create the objective function.
    Eigen::MatrixXd Q = 5 * Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 1200.0, 0.0, 0.0,
          0.0, 1200.0, 0.0,
          0.0, 0.0, 700.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI / 2.0;

    // Create reference trajectory (empty in this example).
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    );

    // Initial state.
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    // Create IPOPT-specific SCP options.
    cddp::SCPOptions options;
    options.max_iterations = 3;
    options.min_iterations = 3;
    options.ftol = 1e-6;
    options.xtol = 1e-6;
    options.gtol = 1e-6;
    options.merit_penalty = 100.0;
    options.verbose = true;
    options.trust_region_radius = 100.0;
    options.ipopt_max_iter = 100;

    // Create the SCP solver (using IPOPT).
    cddp::SCPSolver sqp_solver(initial_state, goal_state, horizon, timestep);
    sqp_solver.setDynamicalSystem(std::move(system));
    sqp_solver.setObjective(std::move(objective));
    sqp_solver.setOptions(options);

    // Test that the dynamical system and objective are properly set.
    ASSERT_NE(sqp_solver.getDynamicalSystem(), nullptr)
        << "Dynamical system pointer is null.";
    ASSERT_NE(sqp_solver.getObjective(), nullptr)
        << "Objective pointer is null.";
    Eigen::VectorXd ref_state = sqp_solver.getObjective()->getReferenceState();
    ASSERT_GT(ref_state.size(), 0)
        << "Reference state is empty.";
    std::cout << "[TEST] Objective's reference state: " << ref_state.transpose() << std::endl;
}

TEST(SQPIPOPTTest, SolveUnicycle) {
    // Problem parameters.
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle dynamical system.
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Create the objective function.
    Eigen::MatrixXd Q = 5 * Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 1200.0, 0.0, 0.0,
          0.0, 1200.0, 0.0,
          0.0, 0.0, 700.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI / 2.0;

    // Create reference trajectory (empty in this example).
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    );

    // Initial state.
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    // Create IPOPT-specific SCP options.
    cddp::SCPOptions options;
    options.max_iterations = 5;
    options.min_iterations = 3;
    options.ftol = 1e-6;
    options.xtol = 1e-6;
    options.gtol = 1e-6;
    options.merit_penalty = 100.0;
    options.verbose = true;
    options.trust_region_radius = 100.0;
    options.ipopt_print_level = 5;

    // Create the SCP solver.
    cddp::SCPSolver sqp_solver(initial_state, goal_state, horizon, timestep);
    sqp_solver.setDynamicalSystem(std::move(system));
    sqp_solver.setObjective(std::move(objective));
    sqp_solver.setOptions(options);

    // Define control constraints.
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    
    // Add control box constraint.
    sqp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound)
    );

    auto constraint = sqp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint"); 
    ASSERT_NE(constraint, nullptr);
    ASSERT_EQ(constraint->getLowerBound().size(), control_dim);
    ASSERT_EQ(constraint->getLowerBound(), control_lower_bound);

    // Set initial trajectory (all zeros).
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    sqp_solver.setInitialTrajectory(X, U);

    // Solve the problem.
    cddp::SCPResult solution = sqp_solver.solve();

    // Verify basic solution properties.
    EXPECT_GT(solution.iterations, 0);
    EXPECT_GT(solution.solve_time, 0.0);

    // Verify trajectory sizes.
    ASSERT_EQ(solution.X.size(), horizon + 1);
    ASSERT_EQ(solution.U.size(), horizon);

    // Check initial and final states.
    std::cout << "Initial state: " << initial_state.transpose() << std::endl;
    std::cout << "Final state: " << solution.X.back().transpose() << std::endl;
    std::cout << "Goal state: " << goal_state.transpose() << std::endl;
    EXPECT_NEAR((solution.X.front() - initial_state).norm(), 0.0, 1e-3);
    // Optionally check the final state closeness to the goal.
    // EXPECT_NEAR((solution.X.back() - goal_state).norm(), 0.0, 0.1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
