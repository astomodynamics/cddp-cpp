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
    options.debug = false;
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


TEST(IPDDPTest, SolveQuadrotor)
{
    // For quaternion-based quadrotor, state_dim = 13:
    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    int state_dim = 13;
    int control_dim = 4; // [f1, f2, f3, f4]
    int horizon = 400;
    double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.2;         // 1 kg
    double arm_length = 0.165; // 20 cm
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 7.782e-3; // Ixx
    inertia_matrix(1, 1) = 7.782e-3; // Iyy
    inertia_matrix(2, 2) = 1.439e-2; // Izz

    std::string integration_type = "rk4";

    // Create the dynamical system
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);

    // For propagation, create a direct instance
    cddp::Quadrotor quadrotor(timestep, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    // penalize [x, y, z, qw, qx, qy, qz] more (the orientation/quaternion part)
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;
    Q(2, 2) = 1.0;
    Q(3, 3) = 1.0;
    Q(4, 4) = 1.0;
    Q(5, 5) = 1.0;
    Q(6, 6) = 1.0;

    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Figure-8 trajectory parameters
    double figure8_scale = 3.0;     // 3m
    double constant_altitude = 2.0; // 2m
    double total_time = horizon * timestep;
    double omega = 2.0 * M_PI / total_time; // completes 1 cycle over the horizon

    std::vector<Eigen::VectorXd> figure8_reference_states;
    figure8_reference_states.reserve(horizon + 1);

    for (int i = 0; i <= horizon; ++i)
    {
        double t = i * timestep;
        double angle = omega * t;

        // Lemniscate of Gerono for (x, y)
        // x = A cos(angle)
        // y = A sin(angle)*cos(angle)
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);
        ref_state(0) = figure8_scale * std::cos(angle);
        ref_state(1) = figure8_scale * std::sin(angle) * std::cos(angle);
        ref_state(2) = constant_altitude;

        // Identity quaternion: [1, 0, 0, 0]
        ref_state(3) = 1.0;
        ref_state(4) = 0.0;
        ref_state(5) = 0.0;
        ref_state(6) = 0.0;

        figure8_reference_states.push_back(ref_state);
    }

    // Hover at the starting point of the figure-8
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = figure8_scale; // x
    goal_state(2) = constant_altitude;
    goal_state(3) = 1.0; // qw

    // Create the objective
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, figure8_reference_states, timestep);

    // Start the same figure-8 starting point
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = figure8_scale;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0;

    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 500;          // Reasonable for testing
    options.tolerance = 1e-6;            // KKT/optimality tolerance
    options.acceptable_tolerance = 1e-5; // Cost change tolerance
    options.enable_parallel = false;
    options.num_threads = 1;
    options.verbose = true;
    options.debug = false;
    options.ipddp.barrier.mu_initial = 1e-0;
    options.ipddp.barrier.mu_update_factor = 0.5;
    options.ipddp.barrier.mu_update_power = 1.2;
    options.regularization.initial_value = 1e-4;
    options.return_iteration_info = true;

    // Instantiate CDDP solver
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::move(system),
        std::move(objective),
        options);

    // Control constraints
    double min_force = 0.0;
    double max_force = 4.0;
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
    cddp_solver.addPathConstraint("ControlConstraint", std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));

    // Initial trajectory guess
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }

    X[0] = initial_state;
    for (int i = 0; i < horizon; ++i)
    {
        X[i + 1] = quadrotor.getDiscreteDynamics(X[i], U[i], i * timestep);
    }
    // X = figure8_reference_states;
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    std::cout << "\n=== Quadrotor Test: Cold Start ===" << std::endl;
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Check convergence
    auto status_message = std::any_cast<std::string>(solution.at("status_message"));
    auto iterations_completed = std::any_cast<int>(solution.at("iterations_completed"));
    auto solve_time_ms = std::any_cast<double>(solution.at("solve_time_ms"));
    auto final_objective = std::any_cast<double>(solution.at("final_objective"));

    std::cout << "\n=== Quadrotor Convergence Analysis ===" << std::endl;
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
    std::cout << "Initial state: [" << initial_state.head(7).transpose() << "]" << std::endl;
    std::cout << "Final state:   [" << final_state.head(7).transpose() << "]" << std::endl;
    std::cout << "Goal state:    [" << goal_state.head(7).transpose() << "]" << std::endl;
    
    // Calculate position error
    Eigen::Vector3d final_position = final_state.head(3);
    Eigen::Vector3d goal_position = goal_state.head(3);
    double position_error = (final_position - goal_position).norm();
    std::cout << "Position error: " << position_error << " m" << std::endl;

    // Check quaternion norm (should be close to 1)
    Eigen::Vector4d final_quat = final_state.segment(3, 4);
    double quat_norm = final_quat.norm();
    std::cout << "Final quaternion norm: " << quat_norm << std::endl;

    // Test assertions
    EXPECT_TRUE(status_message == "OptimalSolutionFound" || status_message == "AcceptableSolutionFound") 
        << "Quadrotor algorithm should converge";
    EXPECT_GT(iterations_completed, 0) << "Should take at least one iteration";
    
    // Since initial state and goal state are the same, just check final position is close
    // (the quadrotor should stay near the hover point after following the trajectory)
    
    // Check that quaternion norm is preserved (should be close to 1)
    EXPECT_NEAR(quat_norm, 1.0, 0.1) << "Quaternion norm should be approximately 1";
    
    // Check that the quadrotor reaches reasonably close to the goal (within 0.5 meters)
    EXPECT_LT(position_error, 0.5) << "Quadrotor should reach close to the goal position";

    // =========================================================================
    // Test warm start capability
    // =========================================================================
    std::cout << "\n=== Quadrotor Test: Warm Start ===" << std::endl;

    // Enable warm start and use previous solution as initial guess
    cddp::CDDPOptions warm_options = options;
    warm_options.warm_start = true;
    warm_options.max_iterations = 150; // Allow sufficient iterations for warm start convergence
    warm_options.verbose = true;    // Less verbose for warm start test

    // Create a new solver for warm start test
    auto quadrotor_system_warmstart = std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);

    // Create new objective
    auto objective_warmstart = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, figure8_reference_states, timestep);

    cddp::CDDP warm_solver(initial_state, goal_state, horizon, timestep);
    warm_solver.setDynamicalSystem(std::move(quadrotor_system_warmstart));
    warm_solver.setObjective(std::move(objective_warmstart));
    warm_solver.addPathConstraint("ControlConstraint",
                              std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
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

    // Performance comparison
    std::cout << "\n=== Quadrotor Performance Comparison ===" << std::endl;
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

    // Both should converge
    EXPECT_TRUE(warm_status == "OptimalSolutionFound" || warm_status == "AcceptableSolutionFound") 
        << "Warm start should also converge";
    EXPECT_LE(warm_iterations, iterations_completed + 20) << "Warm start should not take significantly more iterations";
}

