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
#include <random>
#include <cmath>
#include "gtest/gtest.h"
#include "cddp.hpp"

TEST(IPDDPPendulumTest, Solve) {
    // Problem parameters for the pendulum swing-up
    const int state_dim = 2;    // [theta, theta_dot]
    const int control_dim = 1;  // [torque]
    const int horizon = 500;
    const double timestep = 0.05;
    const std::string integration_type = "euler";

    // Create a Pendulum instance with parameters: length, mass, damping
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.0;
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Pendulum>(timestep, length, mass, damping, integration_type);

    // Set initial state (near the downward equilibrium) and goal state (upright position)
    Eigen::VectorXd initial_state(state_dim);
    initial_state << -M_PI, 0.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0;

    // Create the quadratic objective for the pendulum swing-up
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.025 / timestep * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 5.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Create the CDDP solver instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Set control constraints (upper bound of 0.25 torque)
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 0.25;
    cddp_solver.addConstraint("ControlConstraint", std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Set solver options
    cddp::CDDPOptions options;
    options.max_iterations = 500;
    options.verbose = false; // disable verbose output for unit tests
    options.cost_tolerance = 1e-7;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "none";
    options.debug = false;
    options.use_parallel = false;
    options.num_threads = 1;
    options.barrier_coeff = 1e-1;
    cddp_solver.setOptions(options);

    // Initialize trajectories for states X and controls U
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    for (auto& u : U) {
        // u(0) = d(gen);
        u << 0.01;
    }
    X[0] = initial_state;

    double J = 0.0;
    for (size_t t = 0; t < horizon; t++) {
        J += cddp_solver.getObjective().running_cost(X[t], U[t], t);
        X[t + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[t], U[t], t * timestep);
    }
    J += cddp_solver.getObjective().terminal_cost(X.back());
    std::cout << "Initial cost: " << J << std::endl;
    std::cout << "Initial state: " << X[0].transpose() << std::endl;
    std::cout << "Final state (pre-solution): " << X.back().transpose() << std::endl;

    // Set the initial trajectory in the solver
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem using IPDDP
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Check that the solution converged
    EXPECT_TRUE(solution.converged);

    double tol = 0.2;
    Eigen::VectorXd final_state = solution.state_sequence.back();
    for (int i = 0; i < state_dim; ++i) {
        EXPECT_NEAR(final_state(i), goal_state(i), tol);
    }

    std::cout << "Optimized final state: " << final_state.transpose() << std::endl;
}
