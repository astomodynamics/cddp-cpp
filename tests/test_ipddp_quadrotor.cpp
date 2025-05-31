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

#include "gtest/gtest.h"
#include "cddp.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>
#include <string>

namespace cddp {

TEST(QuadrotorMSIPDDPTest, ReachesGoalState) {
    const int state_dim = 13;    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    const int control_dim = 4; // [f1, f2, f3, f4]
    const int horizon = 400; 
    const double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 0.01; // Ixx
    inertia_matrix(1, 1) = 0.01; // Iyy
    inertia_matrix(2, 2) = 0.02; // Izz
    std::string integration_type = "rk4";

    auto system = std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices (dimensions updated for state_dim = 13)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;
    Q(2, 2) = 1.0;
    Q(3, 3) = 1.0;
    Q(4, 4) = 1.0;
    Q(5, 5) = 1.0;
    Q(6, 6) = 1.0;

    // Control cost matrix (penalize aggressive control inputs)
    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    // Terminal cost matrix (important for stability)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Parameters for the circular trajectory
    double circle_radius = 3.0;              // e.g., 3m radius
    Eigen::Vector2d circle_center(0.0, 0.0); // center of the circle in the x-y plane
    double constant_altitude = 2.0;          // fixed altitude (z)
    double total_time = horizon * timestep;  // total duration
    // omega is chosen so that the quadrotor completes one full circle over the time horizon
    double omega = 2 * M_PI / total_time;

    std::vector<Eigen::VectorXd> reference_states;
    reference_states.reserve(horizon + 1);

    for (int i = 0; i <= horizon; ++i)
    {
        double t = i * timestep;
        double angle = omega * t;

        // Create a reference state of dimension state_dim (13)
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);

        // Position (x, y, z)
        ref_state(0) = circle_center(0) + circle_radius * std::cos(angle);
        ref_state(1) = circle_center(1) + circle_radius * std::sin(angle);
        ref_state(2) = constant_altitude;

        // Orientation: set to identity quaternion [1, 0, 0, 0]
        ref_state(3) = 1.0; // qw
        ref_state(4) = 0.0; // qx
        ref_state(5) = 0.0; // qy
        ref_state(6) = 0.0; // qz                              // z velocity

        reference_states.push_back(ref_state);
    }

    // Goal state: hover at position (3,0,2) with identity quaternion and zero velocities.
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = circle_center(0) + circle_radius;
    goal_state(2) = constant_altitude;
    goal_state(3) = 1.0; // Identity quaternion: qw = 1

    // Initial state (at origin with identity quaternion)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = circle_center(0) + circle_radius;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0; // Identity quaternion: qw = 1

    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, reference_states, timestep);

     // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 10000;
    options.verbose = true;
    options.debug = false;
    options.use_parallel = true;
    options.num_threads = 10;
    options.cost_tolerance = 1e-3;
    options.grad_tolerance = 1e-2;
    options.regularization_type = "control";
    options.regularization_control = 1e-4;
    options.regularization_state = 0.0;
    options.barrier_coeff = 1e-1;
    options.ms_segment_length = horizon / 10;
    options.ms_rollout_type = "nonlinear";
    options.ms_defect_tolerance_for_single_shooting = 1e-5;
    options.barrier_update_factor = 0.2;
    options.barrier_update_power = 1.2;
    options.minimum_reduction_ratio = 1e-4;

    // Create the CDDP solver
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::move(system),
        std::move(objective),
        options);

    // Control constraints (motor thrust limits)
    double min_force = 0.0; // Motors can only produce thrust upward
    double max_force = 4.0; // Maximum thrust per motor
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
    cddp_solver.addConstraint("ControlConstraint",
                               std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
    
    // Initial trajectory: allocate state and control trajectories
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    // Initialize with hovering thrust (each motor provides mg/4)
    double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }
    // Initialize state by the actual reference trajectory
    X = reference_states;
    cddp_solver.setInitialTrajectory(X, U);

    cddp::CDDPSolution solution = cddp_solver.solve("MSIPDDP");
    EXPECT_TRUE(solution.converged);
}

TEST(QuadrotorLogDDPTest, ReachesGoalState) {
    const int state_dim = 13;    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    const int control_dim = 4; // [f1, f2, f3, f4]
    const int horizon = 400; 
    const double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 0.01; // Ixx
    inertia_matrix(1, 1) = 0.01; // Iyy
    inertia_matrix(2, 2) = 0.02; // Izz
    std::string integration_type = "rk4";

    auto system = std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices (dimensions updated for state_dim = 13)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;
    Q(2, 2) = 1.0;
    Q(3, 3) = 1.0;
    Q(4, 4) = 1.0;
    Q(5, 5) = 1.0;
    Q(6, 6) = 1.0;

    // Control cost matrix (penalize aggressive control inputs)
    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    // Terminal cost matrix (important for stability)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Parameters for the circular trajectory
    double circle_radius = 3.0;              // e.g., 3m radius
    Eigen::Vector2d circle_center(0.0, 0.0); // center of the circle in the x-y plane
    double constant_altitude = 2.0;          // fixed altitude (z)
    double total_time = horizon * timestep;  // total duration
    // omega is chosen so that the quadrotor completes one full circle over the time horizon
    double omega = 2 * M_PI / total_time;

    std::vector<Eigen::VectorXd> reference_states;
    reference_states.reserve(horizon + 1);

    for (int i = 0; i <= horizon; ++i)
    {
        double t = i * timestep;
        double angle = omega * t;

        // Create a reference state of dimension state_dim (13)
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);

        // Position (x, y, z)
        ref_state(0) = circle_center(0) + circle_radius * std::cos(angle);
        ref_state(1) = circle_center(1) + circle_radius * std::sin(angle);
        ref_state(2) = constant_altitude;

        // Orientation: set to identity quaternion [1, 0, 0, 0]
        ref_state(3) = 1.0; // qw
        ref_state(4) = 0.0; // qx
        ref_state(5) = 0.0; // qy
        ref_state(6) = 0.0; // qz                              // z velocity

        reference_states.push_back(ref_state);
    }

    // Goal state: hover at position (3,0,2) with identity quaternion and zero velocities.
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = circle_center(0) + circle_radius;
    goal_state(2) = constant_altitude;
    goal_state(3) = 1.0; // Identity quaternion: qw = 1

    // Initial state (at origin with identity quaternion)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = circle_center(0) + circle_radius;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0; // Identity quaternion: qw = 1

    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, reference_states, timestep);

     // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 10000;
    options.verbose = true;
    options.debug = false;
    options.use_parallel = true;
    options.num_threads = 10;
    options.cost_tolerance = 1e-3;
    options.grad_tolerance = 1e-2;
    options.regularization_type = "control";
    options.regularization_control = 1e-4;
    options.regularization_state = 0.0;
    options.barrier_coeff = 1e-1;
    options.ms_segment_length = horizon / 10;
    options.ms_rollout_type = "nonlinear";
    options.ms_defect_tolerance_for_single_shooting = 1e-5;
    options.barrier_update_factor = 0.2;
    options.barrier_update_power = 1.2;
    options.minimum_reduction_ratio = 1e-4;

    // Create the CDDP solver
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::move(system),
        std::move(objective),
        options);

    // Control constraints (motor thrust limits)
    double min_force = 0.0; // Motors can only produce thrust upward
    double max_force = 4.0; // Maximum thrust per motor
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
    cddp_solver.addConstraint("ControlConstraint",
                               std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
    
    // Initial trajectory: allocate state and control trajectories
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    // Initialize with hovering thrust (each motor provides mg/4)
    double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }
    // Initialize state by the actual reference trajectory
    X = reference_states;
    cddp_solver.setInitialTrajectory(X, U);

    cddp::CDDPSolution solution = cddp_solver.solve("LogDDP");
    EXPECT_TRUE(solution.converged);
}

} // namespace cddp
