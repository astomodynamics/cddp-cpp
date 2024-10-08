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

TEST(CDDPTest, Solve) {
    // Problem parameters
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 30;
    double timestep = 0.1;

    // Create a pendulum instance 
    double mass = 1.0; 
    double length = 1.0; 
    double gravity = 9.81;
    cddp::Pendulum pendulum(mass, length, gravity, timestep); 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Pendulum>(mass, length, gravity, timestep); // Create unique_ptr

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    // Create an empty vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> empty_reference_states; 
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial and target states
    Eigen::VectorXd initial_state(state_dim);

    initial_state << 0.1, 0.0;  // Start at a small angle, zero velocity

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define constraints
    Eigen::VectorXd control_lower_bound(1);
    control_lower_bound << -2.0;
    Eigen::VectorXd control_upper_bound(1);
    control_upper_bound << 2.0;
    // Add the constraint to the solver
    cddp_solver.addConstraint(std::string("ControlBoxConstraint"), std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound)); // Moved here

    // Test for  evalueate
    Eigen::VectorXd state(2);
    state << 0.1, 0.0;
    Eigen::VectorXd control(1);
    control << 0.0;

    // Check if the constraint set is not empty
    ASSERT_FALSE(cddp_solver.getConstraintSet().empty()); 

    // Check the size of the constraint set
    ASSERT_EQ(cddp_solver.getConstraintSet().size(), 1);


    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");
    auto value = constraint.evaluate(state, control);
    ASSERT_TRUE(value[0] == 0.0);

    // Solve the problem
    // cddp::CDDPSolution solution = cddp_solver.solve();

    // // Assertions
    // ASSERT_TRUE(solution.converged); // Check if the solver converged
    // // Add more assertions based on expected behavior

}