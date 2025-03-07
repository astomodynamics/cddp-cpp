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
#include <memory>
#include <cmath>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

// Helper function: Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw)
Eigen::Vector3d quaternionToEuler(double qw, double qx, double qy, double qz)
{
    // Roll (phi)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    double phi = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (theta)
    double sinp = 2.0 * (qw * qy - qz * qx);
    double theta = (std::abs(sinp) >= 1.0) ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);

    // Yaw (psi)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double psi = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(phi, theta, psi);
}

// Helper function: Compute rotation matrix from a unit quaternion [qw, qx, qy, qz]
Eigen::Matrix3d getRotationMatrixFromQuaternion(double qw, double qx, double qy, double qz)
{
    Eigen::Matrix3d R;
    R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
    R(0, 1) = 2 * (qx * qy - qz * qw);
    R(0, 2) = 2 * (qx * qz + qy * qw);

    R(1, 0) = 2 * (qx * qy + qz * qw);
    R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
    R(1, 2) = 2 * (qy * qz - qx * qw);

    R(2, 0) = 2 * (qx * qz - qy * qw);
    R(2, 1) = 2 * (qy * qz + qx * qw);
    R(2, 2) = 1 - 2 * (qx * qx + qy * qy);

    return R;
}

// Helper: Transform quadrotor frame points (motor positions) to world coordinates using quaternion.
std::vector<std::vector<double>> transformQuadrotorFrame(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &quat, // [qw, qx, qy, qz]
    double arm_length)
{
    // Define quadrotor motor positions in the body frame
    std::vector<Eigen::Vector3d> body_points = {
        Eigen::Vector3d(arm_length, 0, 0),  // Front motor
        Eigen::Vector3d(0, arm_length, 0),  // Right motor
        Eigen::Vector3d(-arm_length, 0, 0), // Back motor
        Eigen::Vector3d(0, -arm_length, 0)  // Left motor
    };

    Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat[0], quat[1], quat[2], quat[3]);

    std::vector<std::vector<double>> world_points(3, std::vector<double>());
    for (const auto &pt : body_points)
    {
        Eigen::Vector3d wp = position + R * pt;
        world_points[0].push_back(wp.x());
        world_points[1].push_back(wp.y());
        world_points[2].push_back(wp.z());
    }
    return world_points;
}

int main()
{
    // For quaternion-based quadrotor, state_dim = 13:
    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    int state_dim = 13;
    int control_dim = 4; // [f1, f2, f3, f4]
    int horizon = 400;   // Longer horizon for 3D maneuvers
    double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.0;       // 1kg quadrotor
    double arm_length = 0.2; // 20cm arm length
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 0.01; // Ixx
    inertia_matrix(1, 1) = 0.01; // Iyy
    inertia_matrix(2, 2) = 0.02; // Izz

    std::string integration_type = "rk4";

    // Create the dynamical system (quadrotor) as a unique_ptr
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);

    // For propagation (e.g., initial trajectory), also create an instance (if needed)
    cddp::Quadrotor quadrotor(timestep, mass, inertia_matrix, arm_length, integration_type);

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

    // Parameters for the vertical figure-8 trajectory
    double figure8_scale = 3.0;              // Scale of the figure-8 (in meters)
    double constant_altitude = 2.0;          // Base altitude for the figure-8 (z)
    double constant_y = 0.0;                 // Fixed y-coordinate, defining the vertical (x–z) plane
    double total_time = horizon * timestep;  // Total duration
    double omega = 2 * M_PI / total_time;      // Complete one figure-8 over the horizon

    std::vector<Eigen::VectorXd> figure8_reference_states;
    figure8_reference_states.reserve(horizon + 1);

    for (int i = 0; i <= horizon; ++i)
    {
        double t = i * timestep;
        double angle = omega * t;

        // Create a reference state of dimension state_dim (13)
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);

        // Position (x, y, z) using a vertical lemniscate of Gerono in the x–z plane:
        // x = figure8_scale * cos(angle)
        // y = constant_y (fixed)
        // z = constant_altitude + figure8_scale * sin(angle) * cos(angle)
        ref_state(0) = figure8_scale * std::cos(angle);  
        ref_state(1) = constant_y;                        
        ref_state(2) = constant_altitude + figure8_scale * std::sin(angle) * std::cos(angle);

        // Orientation: set to identity quaternion [1, 0, 0, 0]
        ref_state(3) = 1.0; // qw
        ref_state(4) = 0.0; // qx
        ref_state(5) = 0.0; // qy
        ref_state(6) = 0.0; // qz

        figure8_reference_states.push_back(ref_state);
    }

    // Goal state: hover at the starting point of the figure-8 (x = figure8_scale, y = constant_y, z = constant_altitude)
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = figure8_scale;
    goal_state(1) = constant_y;
    goal_state(2) = constant_altitude;
    goal_state(3) = 1.0; // Identity quaternion: qw = 1

    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, figure8_reference_states, timestep);

    // Initial state (at the start of the figure-8)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = figure8_scale;
    initial_state(1) = constant_y;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0; // Identity quaternion: qw = 1

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
    options.barrier_coeff = 1e-3;

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
    double min_force = 0.0; // Motors can only produce upward thrust
    double max_force = 4.0; // Maximum thrust per motor
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);

    // Initial trajectory: allocate state and control trajectories
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    // Initialize with hovering thrust (each motor provides mg/4)
    double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }
    // Propagate initial trajectory using discrete dynamics
    for (size_t i = 0; i < static_cast<size_t>(horizon); ++i)
    {
        X[i + 1] = quadrotor.getDiscreteDynamics(X[i], U[i]);
    }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the optimal control problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    std::cout << "Final state: " << X_sol.back().transpose() << std::endl;

    // Create plot directory if it doesn't exist
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory))
    {
        fs::create_directory(plotDirectory);
    }

    // Extract solution data for plotting.
    // Convert quaternion (indices 3-6) to Euler angles for attitude visualization.
    std::vector<double> time_arr, x_arr, y_arr, z_arr;
    std::vector<double> phi_arr, theta_arr, psi_arr;
    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        y_arr.push_back(X_sol[i](1));
        z_arr.push_back(X_sol[i](2));
        Eigen::Vector4d quat;
        quat << X_sol[i](3), X_sol[i](4), X_sol[i](5), X_sol[i](6);
        Eigen::Vector3d euler = quaternionToEuler(quat(0), quat(1), quat(2), quat(3));
        phi_arr.push_back(euler(0));
        theta_arr.push_back(euler(1));
        psi_arr.push_back(euler(2));
    }

    // Extract control data
    std::vector<double> time_arr2(time_arr.begin(), time_arr.end() - 1);
    std::vector<double> f1_arr, f2_arr, f3_arr, f4_arr;
    for (const auto &u : U_sol)
    {
        f1_arr.push_back(u(0));
        f2_arr.push_back(u(1));
        f3_arr.push_back(u(2));
        f4_arr.push_back(u(3));
    }

    // Plot the XZ trajectory (vertical profile of the figure-eight)
    figure();
    plot(x_arr, z_arr, "b-")->line_width(2);
    xlabel("X Position (m)");
    ylabel("Z Position (m)");
    title("Quadrotor Vertical Figure-Eight Trajectory (XZ Plane)");
    grid(true);
    
    // Save the trajectory plot
    save(plotDirectory + "/quadrotor_figure_eight_xz.png");
    
    std::cout << "Plots saved to: " << plotDirectory << std::endl;

    return 0;
}
