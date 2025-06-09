/*
 Copyright 2025 Tomo Sasaki

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
#include <chrono>
#include <matplot/matplot.h>
#include <sys/stat.h>
#include <random>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp"

TEST(IPDDPTest, SolvePendulum)
{
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
    goal_state << 0.0, 0.0; // Upright position with zero velocity

    std::vector<Eigen::VectorXd> empty_reference_states;
    // empty_reference_states.back() << 0.0, 0.0;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state (pendulum pointing down)
    Eigen::VectorXd initial_state(state_dim);
    initial_state << M_PI, 0.0; // Zero angle and angular velocity

    // Construct zero control sequence
    std::vector<Eigen::VectorXd> zero_control_sequence(horizon, Eigen::VectorXd::Zero(control_dim));

    // Construct initial trajectory
    std::vector<Eigen::VectorXd> X_init(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    for (int t = 0; t < horizon + 1; ++t)
    {
        X_init[t] = initial_state;
    }

    // Calculate initial cost
    double J = 0.0;
    for (int t = 0; t < horizon; ++t)
    {
        J += objective->running_cost(X_init[t], zero_control_sequence[t], t);
    }
    J += objective->terminal_cost(X_init[horizon]);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -10.0; // Maximum negative torque
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 10.0; // Maximum positive torque

    cddp_solver.addPathConstraint("ControlConstraint",
                              std::make_unique<cddp::ControlConstraint>( control_upper_bound));

    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 100;       
    options.tolerance = 1e-3;            // KKT/optimality tolerance
    options.acceptable_tolerance = 1e-4; // Cost change tolerance
    options.enable_parallel = false;
    options.num_threads = 1;
    options.verbose = true;
    options.debug = true;
    options.regularization.initial_value = 1e-6;
    options.return_iteration_info = true; // Get detailed iteration history

    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] << initial_state;
    for (int i = 0; i < horizon; ++i)
    {
        U[i] = Eigen::VectorXd::Zero(control_dim);
        X[i] = initial_state;
    }
    X[horizon] << initial_state;

    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    std::cout << "\n=== First solve (cold start) ===" << std::endl;
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Check convergence
    auto status_message = std::any_cast<std::string>(solution.at("status_message"));
    auto iterations_completed = std::any_cast<int>(solution.at("iterations_completed"));
    auto solve_time_ms = std::any_cast<double>(solution.at("solve_time_ms"));
    auto final_objective = std::any_cast<double>(solution.at("final_objective"));

    std::cout << "\n=== Convergence Analysis ===" << std::endl;
    std::cout << "Status: " << status_message << std::endl;
    std::cout << "Converged: " << (status_message == "OptimalSolutionFound" || status_message == "AcceptableSolutionFound" ? "YES" : "NO") << std::endl;
    std::cout << "Iterations: " << iterations_completed << std::endl;
    std::cout << "Solve time: " << solve_time_ms << " ms" << std::endl;
    std::cout << "Final cost: " << final_objective << std::endl;

    // Extract trajectories
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points"));

    // Print final state
    Eigen::VectorXd final_state = X_sol.back();
    std::cout << "Final state: [" << final_state.transpose() << "]" << std::endl;
    std::cout << "Goal state:  [" << goal_state.transpose() << "]" << std::endl;
    std::cout << "Final error: " << (final_state - goal_state).norm() << std::endl;

    // Test assertions
    EXPECT_TRUE(status_message == "OptimalSolutionFound" || status_message == "AcceptableSolutionFound") << "Algorithm should converge";
    EXPECT_GT(iterations_completed, 0) << "Should take at least one iteration";
    EXPECT_LT(final_objective, J) << "Final cost should be better than initial cost";

    // =========================================================================
    // Test warm start capability
    // =========================================================================
    std::cout << "\n=== Testing warm start ===" << std::endl;

    // Enable warm start and use previous solution as initial guess
    cddp::CDDPOptions warm_options = options;
    warm_options.warm_start = true;
    warm_options.max_iterations = 10; // Fewer iterations for warm start
    warm_options.verbose = false;     // Less verbose for warm start test
    warm_options.tolerance = 1e-3;            // KKT/optimality tolerance
    warm_options.acceptable_tolerance = 1e-4; // Cost change tolerance
    warm_options.enable_parallel = false;
    warm_options.num_threads = 1;
    warm_options.debug = false;
    
    // Create a new solver for warm start test
    auto hcw_system_warmstart = std::make_unique<cddp::Pendulum>(timestep, length, mass, damping, integration_type);

    // Create new objective
    std::vector<Eigen::VectorXd> empty_reference_states_warmstart;
    auto objective_warmstart = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states_warmstart, timestep);

    cddp::CDDP warm_solver(initial_state, goal_state, horizon, timestep);
    warm_solver.setDynamicalSystem(std::move(hcw_system_warmstart));
    warm_solver.setObjective(std::move(objective_warmstart));
    warm_solver.addPathConstraint("ControlConstraint",
                              std::make_unique<cddp::ControlConstraint>( control_upper_bound));
    warm_solver.setOptions(warm_options);

    // Use previous solution as warm start
    warm_solver.setInitialTrajectory(X_sol, U_sol);

    // Solve with warm start
    auto start_time = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution warm_solution = warm_solver.solve("IPDDP");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto warm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Extract warm start results
    auto warm_status = std::any_cast<std::string>(warm_solution.at("status_message"));
    auto warm_iterations = std::any_cast<int>(warm_solution.at("iterations_completed"));
    auto warm_solve_time = std::any_cast<double>(warm_solution.at("solve_time_ms"));
    auto warm_final_cost = std::any_cast<double>(warm_solution.at("final_objective"));

    std::cout << "Warm start status: " << warm_status << std::endl;
    std::cout << "Warm start iterations: " << warm_iterations << std::endl;
    std::cout << "Warm start solve time: " << warm_solve_time << " ms" << std::endl;
    std::cout << "Warm start final cost: " << warm_final_cost << std::endl;

    // Warm start should converge faster or in fewer iterations
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Cold start: " << iterations_completed << " iterations, " << solve_time_ms << " ms" << std::endl;
    std::cout << "Warm start: " << warm_iterations << " iterations, " << warm_solve_time << " ms" << std::endl;

    if (warm_iterations <= iterations_completed)
    {
        std::cout << "✓ Warm start used fewer or equal iterations" << std::endl;
    }
    else
    {
        std::cout << "✗ Warm start used more iterations (this can happen)" << std::endl;
    }

    if (warm_solve_time <= solve_time_ms * 1.2)
    { // Allow 20% tolerance
        std::cout << "✓ Warm start was faster or comparable" << std::endl;
    }
    else
    {
        std::cout << "✗ Warm start was slower" << std::endl;
    }

    // Both should converge
    EXPECT_TRUE(warm_status == "OptimalSolutionFound" || warm_status == "AcceptableSolutionFound") << "Warm start should also converge";
    EXPECT_LE(warm_iterations, iterations_completed + 5) << "Warm start should not take significantly more iterations";
}

TEST(IPDDPTest, SolveUnicycle) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a dubins car instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(timestep, integration_type); // Create unique_ptr

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

    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 20;
    options.tolerance = 1e-2;
    options.enable_parallel = true;
    options.num_threads = 10;
    options.verbose = true;
    options.debug = false;

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    
    // Add the constraint to the solver
    cddp_solver.addPathConstraint("ControlConstraint", std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");
    // cddp::CDDPSolution solution = cddp_solver.solveIPDDP();

    auto status = std::any_cast<std::string>(solution.at("status_message"));
    ASSERT_TRUE(status == "OptimalSolutionFound" || status == "AcceptableSolutionFound");

    // Extract solution
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory")); // size: horizon + 1
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory")); // size: horizon
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points")); // size: horizon + 1
}

namespace cddp
{
    class CarParkingObjective : public NonlinearObjective
    {
    public:
        CarParkingObjective(const Eigen::VectorXd &goal_state, double timestep)
            : NonlinearObjective(timestep), reference_state_(goal_state)
        {
            // Control cost coefficients: cu = 1e-2*[1 .01]
            cu_ = Eigen::Vector2d(1e-2, 1e-4);

            // Final cost coefficients: cf = [.1 .1 1 .3]
            cf_ = Eigen::Vector4d(0.1, 0.1, 1.0, 0.3);

            // Smoothness scales for final cost: pf = [.01 .01 .01 1]
            pf_ = Eigen::Vector4d(0.01, 0.01, 0.01, 1.0);

            // Running cost coefficients: cx = 1e-3*[1 1]
            cx_ = Eigen::Vector2d(1e-3, 1e-3);

            // Smoothness scales for running cost: px = [.1 .1]
            px_ = Eigen::Vector2d(0.1, 0.1);
        }

        double running_cost(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control,
                            int index) const override
        {
            // Control cost: lu = cu*u.^2
            double lu = cu_.dot(control.array().square().matrix());

            // Running cost on distance from origin: lx = cx*sabs(x(1:2,:),px)
            Eigen::VectorXd xy_state = state.head(2);
            double lx = cx_.dot(sabs(xy_state, px_));

            return lu + lx;
        }

        double terminal_cost(const Eigen::VectorXd &final_state) const override
        {
            // Final state cost: llf = cf*sabs(x(:,final),pf);
            return cf_.dot(sabs(final_state, pf_)) + running_cost(final_state, Eigen::VectorXd::Zero(2), 0);
        }

    private:
        // Helper function for smooth absolute value (pseudo-Huber)
        Eigen::VectorXd sabs(const Eigen::VectorXd &x, const Eigen::VectorXd &p) const
        {
            return ((x.array().square() / p.array().square() + 1.0).sqrt() * p.array() - p.array()).matrix();
        }

        Eigen::VectorXd reference_state_;
        Eigen::Vector2d cu_; // Control cost coefficients
        Eigen::Vector4d cf_; // Final cost coefficients
        Eigen::Vector4d pf_; // Smoothness scales for final cost
        Eigen::Vector2d cx_; // Running cost coefficients
        Eigen::Vector2d px_; // Smoothness scales for running cost
    };
} // namespace cddp

TEST(IPDDPTest, SolveCar)
{
    int state_dim = 4;   // [x y theta v]
    int control_dim = 2; // [wheel_angle acceleration]
    int horizon = 500;   
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create car instance
    double wheelbase = 2.0;
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Car>(timestep, wheelbase, integration_type);

    // Initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 1.0, 1.0, 1.5 * M_PI, 0.0; // Start at (1,1) facing backwards

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0, 0.0, 0.0; // Park at origin facing forward

    // Create the nonlinear objective
    auto objective = std::make_unique<cddp::CarParkingObjective>(goal_state, timestep);

    // Construct initial control sequence
    std::vector<Eigen::VectorXd> initial_control_sequence(horizon, Eigen::VectorXd::Zero(control_dim));
    for (auto &u : initial_control_sequence)
    {
        u << 0.01, 0.01; // Small initial controls
    }

    // Construct initial trajectory
    std::vector<Eigen::VectorXd> X_init(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    X_init[0] = initial_state;
    
    // Forward simulate initial trajectory
    for (int t = 0; t < horizon; ++t)
    {
        X_init[t + 1] = system->getDiscreteDynamics(X_init[t], initial_control_sequence[t], t * timestep);
    }

    // Calculate initial cost
    double J = 0.0;
    for (int t = 0; t < horizon; ++t)
    {
        J += objective->running_cost(X_init[t], initial_control_sequence[t], t);
    }
    J += objective->terminal_cost(X_init[horizon]);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -0.5, -2.0; // [steering_angle, acceleration]
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 0.5, 2.0;

    cddp_solver.addPathConstraint("ControlConstraint",
                              std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 300;         // Reasonable number for testing
    options.tolerance = 1e-6;            // KKT/optimality tolerance
    options.acceptable_tolerance = 1e-6; // Cost change tolerance
    options.enable_parallel = false;
    options.num_threads = 1;
    options.verbose = true;
    options.debug = true;
    options.regularization.initial_value = 1e-4;
    options.return_iteration_info = true; // Get detailed iteration history

    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    cddp_solver.setInitialTrajectory(X_init, initial_control_sequence);

    // Solve the problem
    std::cout << "\n=== First solve (cold start) ===" << std::endl;
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Check convergence
    auto status_message = std::any_cast<std::string>(solution.at("status_message"));
    auto iterations_completed = std::any_cast<int>(solution.at("iterations_completed"));
    auto solve_time_ms = std::any_cast<double>(solution.at("solve_time_ms"));
    auto final_objective = std::any_cast<double>(solution.at("final_objective"));

    std::cout << "\n=== Convergence Analysis ===" << std::endl;
    std::cout << "Status: " << status_message << std::endl;
    std::cout << "Converged: " << (status_message == "OptimalSolutionFound" || status_message == "AcceptableSolutionFound" ? "YES" : "NO") << std::endl;
    std::cout << "Iterations: " << iterations_completed << std::endl;
    std::cout << "Solve time: " << solve_time_ms << " ms" << std::endl;
    std::cout << "Initial cost: " << J << std::endl;
    std::cout << "Final cost: " << final_objective << std::endl;

    // Extract trajectories
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points"));

    // Print final state
    Eigen::VectorXd final_state = X_sol.back();
    std::cout << "Initial state: [" << initial_state.transpose() << "]" << std::endl;
    std::cout << "Final state: [" << final_state.transpose() << "]" << std::endl;
    std::cout << "Goal state:  [" << goal_state.transpose() << "]" << std::endl;
    std::cout << "Final error: " << (final_state - goal_state).norm() << std::endl;

    // Test assertions
    EXPECT_TRUE(status_message == "OptimalSolutionFound" || status_message == "AcceptableSolutionFound") 
        << "Algorithm should converge";
    EXPECT_GT(iterations_completed, 0) << "Should take at least one iteration";
    EXPECT_LT(final_objective, J) << "Final cost should be better than initial cost";

    // =========================================================================
    // Test warm start capability
    // =========================================================================
    std::cout << "\n=== Testing warm start ===" << std::endl;

    // Enable warm start and use previous solution as initial guess
    cddp::CDDPOptions warm_options = options;
    warm_options.warm_start = true;
    warm_options.max_iterations = 1; // Fewer iterations for warm start
    warm_options.verbose = false;     // Less verbose for warm start test

    // Create a new solver for warm start test
    auto car_system_warmstart = std::make_unique<cddp::Car>(timestep, wheelbase, integration_type);

    // Create new objective
    auto objective_warmstart = std::make_unique<cddp::CarParkingObjective>(goal_state, timestep);

    cddp::CDDP warm_solver(initial_state, goal_state, horizon, timestep);
    warm_solver.setDynamicalSystem(std::move(car_system_warmstart));
    warm_solver.setObjective(std::move(objective_warmstart));
    warm_solver.addPathConstraint("ControlConstraint",
                              std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    warm_solver.setOptions(warm_options);

    // Use previous solution as warm start
    warm_solver.setInitialTrajectory(X_sol, U_sol);

    // Solve with warm start
    auto start_time = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution warm_solution = warm_solver.solve("IPDDP");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto warm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Extract warm start results
    auto warm_status = std::any_cast<std::string>(warm_solution.at("status_message"));
    auto warm_iterations = std::any_cast<int>(warm_solution.at("iterations_completed"));
    auto warm_solve_time = std::any_cast<double>(warm_solution.at("solve_time_ms"));
    auto warm_final_cost = std::any_cast<double>(warm_solution.at("final_objective"));

    std::cout << "Warm start status: " << warm_status << std::endl;
    std::cout << "Warm start iterations: " << warm_iterations << std::endl;
    std::cout << "Warm start solve time: " << warm_solve_time << " ms" << std::endl;
    std::cout << "Warm start final cost: " << warm_final_cost << std::endl;

    // Warm start should converge faster or in fewer iterations
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Cold start: " << iterations_completed << " iterations, " << solve_time_ms << " ms" << std::endl;
    std::cout << "Warm start: " << warm_iterations << " iterations, " << warm_solve_time << " ms" << std::endl;

    if (warm_iterations <= iterations_completed)
    {
        std::cout << "✓ Warm start used fewer or equal iterations" << std::endl;
    }
    else
    {
        std::cout << "✗ Warm start used more iterations (this can happen)" << std::endl;
    }

    if (warm_solve_time <= solve_time_ms * 1.2)
    { // Allow 20% tolerance
        std::cout << "✓ Warm start was faster or comparable" << std::endl;
    }
    else
    {
        std::cout << "✗ Warm start was slower" << std::endl;
    }

    // Both should converge
    EXPECT_TRUE(warm_status == "OptimalSolutionFound" || warm_status == "AcceptableSolutionFound") 
        << "Warm start should also converge";
    EXPECT_LE(warm_iterations, iterations_completed + 10) << "Warm start should not take significantly more iterations";

    // Verify that the car moves towards the goal
    double initial_distance = (initial_state.head(2) - goal_state.head(2)).norm();
    double final_distance = (final_state.head(2) - goal_state.head(2)).norm();
    EXPECT_LT(final_distance, initial_distance) << "Car should move closer to the goal position";
    
    // Check that final position is reasonably close to goal (within 0.5 units)
    EXPECT_LT(final_distance, 0.5) << "Car should park reasonably close to the goal";
}