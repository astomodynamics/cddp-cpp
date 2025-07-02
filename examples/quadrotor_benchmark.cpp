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
#include <cmath>
#include <filesystem>
#include <memory>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

// Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw)
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

// Compute rotation matrix from a unit quaternion [qw, qx, qy, qz]
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

// Transform quadrotor frame points (motor positions) to world coordinates using quaternion
std::vector<std::vector<double>> transformQuadrotorFrame(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &quat, // [qw, qx, qy, qz]
    double arm_length)
{
    // Motor positions in body frame
    std::vector<Eigen::Vector3d> body_points = {
        Eigen::Vector3d(arm_length, 0, 0),  // Front
        Eigen::Vector3d(0, arm_length, 0),  // Right
        Eigen::Vector3d(-arm_length, 0, 0), // Back
        Eigen::Vector3d(0, -arm_length, 0)  // Left
    };

    Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat[0], quat[1], quat[2], quat[3]);

    // Prepare return container
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

int main() {
    // --------------------------
    // 1. Shared problem setup
    // --------------------------
    
    // For quaternion-based quadrotor, state_dim = 13:
    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    int state_dim = 13;
    int control_dim = 4; // [f1, f2, f3, f4]
    int horizon = 400;
    double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.2;         // 1.2 kg
    double arm_length = 0.165; // 16.5 cm
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 7.782e-3; // Ixx
    inertia_matrix(1, 1) = 7.782e-3; // Iyy
    inertia_matrix(2, 2) = 1.439e-2; // Izz

    std::string integration_type = "rk4";

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

    // Start the same figure-8 starting point
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = figure8_scale;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0;

    // Control constraints
    double min_force = 0.0;
    double max_force = 4.0;
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);

    // Initial trajectory guess (hover thrust)
    double hover_thrust = mass * 9.81 / 4.0;

    // Create a directory for saving plots
    const std::string plotDirectory = "../results/benchmark";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Helper function to create initial trajectory
    auto createInitialTrajectory = [&]() {
        std::vector<Eigen::VectorXd> X_init = figure8_reference_states; // Use reference as initial guess
        std::vector<Eigen::VectorXd> U_init(horizon, hover_thrust * Eigen::VectorXd::Ones(control_dim));
        return std::make_pair(X_init, U_init);
    };

    auto [X_init, U_init] = createInitialTrajectory();

    // --------------------------------------------------------
    // 2. Baseline #1: ASDDP 
    // --------------------------------------------------------
    std::cout << "Solving with ASDDP..." << std::endl;
    
    cddp::CDDPOptions options_asddp;
    options_asddp.max_iterations = 1000;
    options_asddp.verbose = true;
    options_asddp.debug = false;
    options_asddp.enable_parallel = false;
    options_asddp.num_threads = 1;
    options_asddp.tolerance = 1e-5;
    options_asddp.acceptable_tolerance = 1e-4;
    options_asddp.regularization.initial_value = 1e-1;
    options_asddp.use_ilqr = true;

    cddp::CDDP solver_asddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, figure8_reference_states, timestep),
        options_asddp
    );
    
    solver_asddp.setInitialTrajectory(X_init, U_init);

    // Add constraints
    solver_asddp.addPathConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Solve for baseline #1
    auto start_time_asddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_asddp = solver_asddp.solve(cddp::SolverType::ASDDP);
    auto end_time_asddp = std::chrono::high_resolution_clock::now();
    auto solve_time_asddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_asddp - start_time_asddp).count();
    
    auto X_asddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_asddp.at("state_trajectory"));
    auto U_asddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_asddp.at("control_trajectory"));
    double cost_asddp = std::any_cast<double>(sol_asddp.at("final_objective"));
    std::cout << "ASDDP Optimal Cost: " << cost_asddp << std::endl;

    // Extract data for plotting
    std::vector<double> x_asddp, y_asddp, z_asddp;
    std::vector<double> phi_asddp, theta_asddp, psi_asddp;
    
    for (size_t i = 0; i < X_asddp_sol.size(); ++i)
    {
        x_asddp.push_back(X_asddp_sol[i](0));
        y_asddp.push_back(X_asddp_sol[i](1));
        z_asddp.push_back(X_asddp_sol[i](2));

        double qw = X_asddp_sol[i](3);
        double qx = X_asddp_sol[i](4);
        double qy = X_asddp_sol[i](5);
        double qz = X_asddp_sol[i](6);

        Eigen::Vector3d euler = quaternionToEuler(qw, qx, qy, qz);
        phi_asddp.push_back(euler(0));
        theta_asddp.push_back(euler(1));
        psi_asddp.push_back(euler(2));
    }

    // --------------------------------------------------------
    // 3. Baseline #2: LogDDP    
    // --------------------------------------------------------
    std::cout << "Solving with LogDDP..." << std::endl;
    
    cddp::CDDPOptions options_logddp;
    options_logddp.max_iterations = 1000;
    options_logddp.verbose = true;
    options_logddp.debug = false;
    options_logddp.tolerance = 1e-5;
    options_logddp.acceptable_tolerance = 1e-4;
    options_logddp.regularization.initial_value = 1e-4;
    options_logddp.log_barrier.barrier.mu_initial = 1e-0;
    options_logddp.log_barrier.barrier.mu_update_factor = 0.2;
    options_logddp.log_barrier.relaxed_log_barrier_delta = 1e-5;
    options_logddp.use_ilqr = true;

    cddp::CDDP solver_logddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, figure8_reference_states, timestep),
        options_logddp
    );

    solver_logddp.setInitialTrajectory(X_init, U_init);

    // Add constraints for LogDDP
    solver_logddp.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
    
    // Solve for baseline #2: LogDDP
    auto start_time_logddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_logddp = solver_logddp.solve("LogDDP");
    auto end_time_logddp = std::chrono::high_resolution_clock::now();
    auto solve_time_logddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_logddp - start_time_logddp).count();
    
    auto X_logddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_logddp.at("state_trajectory"));
    auto U_logddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_logddp.at("control_trajectory"));
    double cost_logddp = std::any_cast<double>(sol_logddp.at("final_objective"));
    std::cout << "LogDDP Optimal Cost: " << cost_logddp << std::endl;

    // Extract data for plotting
    std::vector<double> x_logddp, y_logddp, z_logddp;
    std::vector<double> phi_logddp, theta_logddp, psi_logddp;

    for (size_t i = 0; i < X_logddp_sol.size(); ++i)
    {
        x_logddp.push_back(X_logddp_sol[i](0));
        y_logddp.push_back(X_logddp_sol[i](1));
        z_logddp.push_back(X_logddp_sol[i](2));

        double qw = X_logddp_sol[i](3);
        double qx = X_logddp_sol[i](4);
        double qy = X_logddp_sol[i](5);
        double qz = X_logddp_sol[i](6);

        Eigen::Vector3d euler = quaternionToEuler(qw, qx, qy, qz);
        phi_logddp.push_back(euler(0));
        theta_logddp.push_back(euler(1));
        psi_logddp.push_back(euler(2));
    }

    // --------------------------------------------------------
    // 4. Baseline #3: IPDDP    
    // --------------------------------------------------------
    std::cout << "Solving with IPDDP..." << std::endl;
    
    cddp::CDDPOptions options_ipddp;
    options_ipddp.max_iterations = 100;
    options_ipddp.verbose = true;
    options_ipddp.debug = false;
    options_ipddp.tolerance = 1e-5;
    options_ipddp.acceptable_tolerance = 1e-4;
    options_ipddp.regularization.initial_value = 1e-3;
    options_ipddp.ipddp.barrier.mu_initial = 1e-0;
    options_ipddp.ipddp.barrier.mu_update_factor = 0.2;
    options_ipddp.ipddp.barrier.mu_update_power = 1.2;
    options_ipddp.filter.merit_acceptance_threshold = 1e-4;
    options_ipddp.filter.violation_acceptance_threshold = 1e-6;
    options_ipddp.filter.max_violation_threshold = 1e+7;
    options_ipddp.filter.min_violation_for_armijo_check = 1e-7;
    options_ipddp.filter.armijo_constant = 1e-4;
    options_ipddp.use_ilqr = true;
    options_ipddp.enable_parallel = true;
    options_ipddp.num_threads = 10;

    cddp::CDDP solver_ipddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, figure8_reference_states, timestep),
        options_ipddp
    );

    solver_ipddp.setInitialTrajectory(X_init, U_init);

    // Add constraints for IPDDP
    solver_ipddp.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
    
    // Solve for baseline #3: IPDDP
    auto start_time_ipddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_ipddp = solver_ipddp.solve(cddp::SolverType::IPDDP);
    auto end_time_ipddp = std::chrono::high_resolution_clock::now();
    auto solve_time_ipddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ipddp - start_time_ipddp).count();
    
    auto X_ipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp.at("state_trajectory"));
    auto U_ipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp.at("control_trajectory"));
    double cost_ipddp = std::any_cast<double>(sol_ipddp.at("final_objective"));
    std::cout << "IPDDP Optimal Cost: " << cost_ipddp << std::endl;

    // Extract data for plotting
    std::vector<double> x_ipddp, y_ipddp, z_ipddp;
    std::vector<double> phi_ipddp, theta_ipddp, psi_ipddp;

    for (size_t i = 0; i < X_ipddp_sol.size(); ++i)
    {
        x_ipddp.push_back(X_ipddp_sol[i](0));
        y_ipddp.push_back(X_ipddp_sol[i](1));
        z_ipddp.push_back(X_ipddp_sol[i](2));

        double qw = X_ipddp_sol[i](3);
        double qx = X_ipddp_sol[i](4);
        double qy = X_ipddp_sol[i](5);
        double qz = X_ipddp_sol[i](6);

        Eigen::Vector3d euler = quaternionToEuler(qw, qx, qy, qz);
        phi_ipddp.push_back(euler(0));
        theta_ipddp.push_back(euler(1));
        psi_ipddp.push_back(euler(2));
    }

    // --------------------------------------------------------
    // 5. Baseline #4: MSIPDDP    
    // --------------------------------------------------------
    std::cout << "Solving with MSIPDDP..." << std::endl;
    
    cddp::CDDPOptions options_msipddp;
    options_msipddp.max_iterations = 100;
    options_msipddp.verbose = true;
    options_msipddp.debug = false;
    options_msipddp.tolerance = 1e-5;
    options_msipddp.acceptable_tolerance = 1e-4;
    options_msipddp.regularization.initial_value = 1e-3;
    options_msipddp.msipddp.segment_length = horizon / 10;
    options_msipddp.msipddp.rollout_type = "hybrid";
    options_msipddp.msipddp.barrier.mu_initial = 1e-0;
    options_msipddp.msipddp.barrier.mu_update_factor = 0.2;
    options_msipddp.msipddp.barrier.mu_update_power = 1.2;
    options_msipddp.use_ilqr = true;
    options_msipddp.enable_parallel = true;
    options_msipddp.num_threads = 10;
    
    cddp::CDDP solver_msipddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, figure8_reference_states, timestep),
        options_msipddp
    );

    solver_msipddp.setInitialTrajectory(X_init, U_init);

    // Add constraints for MSIPDDP
    solver_msipddp.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));

    // Solve for baseline #4: MSIPDDP
    auto start_time_msipddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_msipddp = solver_msipddp.solve(cddp::SolverType::MSIPDDP);
    auto end_time_msipddp = std::chrono::high_resolution_clock::now();
    auto solve_time_msipddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_msipddp - start_time_msipddp).count();
    
    auto X_msipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_msipddp.at("state_trajectory"));
    auto U_msipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_msipddp.at("control_trajectory"));
    double cost_msipddp = std::any_cast<double>(sol_msipddp.at("final_objective"));
    std::cout << "MSIPDDP Optimal Cost: " << cost_msipddp << std::endl;

    // Extract data for plotting
    std::vector<double> x_msipddp, y_msipddp, z_msipddp;
    std::vector<double> phi_msipddp, theta_msipddp, psi_msipddp;

    for (size_t i = 0; i < X_msipddp_sol.size(); ++i)
    {
        x_msipddp.push_back(X_msipddp_sol[i](0));
        y_msipddp.push_back(X_msipddp_sol[i](1));
        z_msipddp.push_back(X_msipddp_sol[i](2));

        double qw = X_msipddp_sol[i](3);
        double qx = X_msipddp_sol[i](4);
        double qy = X_msipddp_sol[i](5);
        double qz = X_msipddp_sol[i](6);

        Eigen::Vector3d euler = quaternionToEuler(qw, qx, qy, qz);
        phi_msipddp.push_back(euler(0));
        theta_msipddp.push_back(euler(1));
        psi_msipddp.push_back(euler(2));
    }

    // --------------------------------------------------------
    // 6. Reference trajectory for comparison
    // --------------------------------------------------------
    std::vector<double> x_ref, y_ref, z_ref;
    for (const auto& ref_state : figure8_reference_states) {
        x_ref.push_back(ref_state(0));
        y_ref.push_back(ref_state(1));
        z_ref.push_back(ref_state(2));
    }

    // --------------------------------------------------------
    // 7. Plot all trajectories comparison
    // --------------------------------------------------------
    auto main_figure = figure(true);
    main_figure->size(2400, 800); // 1x4 layout

    // --- Subplot 1: ASDDP ---
    auto ax_asddp = subplot(1, 4, 0);

    auto traj3d_asddp = plot3(ax_asddp, x_asddp, y_asddp, z_asddp);
    traj3d_asddp->display_name("ASDDP Trajectory");
    traj3d_asddp->line_style("-");
    traj3d_asddp->line_width(2);
    traj3d_asddp->color("blue");

    hold(ax_asddp, true);

    // Reference trajectory
    auto ref3d_asddp = plot3(ax_asddp, x_ref, y_ref, z_ref);
    ref3d_asddp->display_name("Reference");
    ref3d_asddp->line_style("--");
    ref3d_asddp->line_width(1);
    ref3d_asddp->color("red");

    // Project trajectory onto x-y plane at z=0
    auto proj_xy_asddp = plot3(ax_asddp, x_asddp, y_asddp, std::vector<double>(x_asddp.size(), 0.0));
    proj_xy_asddp->display_name("X-Y Projection");
    proj_xy_asddp->line_style("--");
    proj_xy_asddp->line_width(1);
    proj_xy_asddp->color("gray");

    xlabel(ax_asddp, "X [m]");
    ylabel(ax_asddp, "Y [m]");
    zlabel(ax_asddp, "Z [m]");
    xlim(ax_asddp, {-4, 4});
    ylim(ax_asddp, {-2, 2});
    zlim(ax_asddp, {0, 4});
    title(ax_asddp, "ASDDP");
    auto leg_asddp = matplot::legend(ax_asddp);
    leg_asddp->location(legend::general_alignment::topleft);
    grid(ax_asddp, true);

    // --- Subplot 2: LogDDP ---
    auto ax_logddp = subplot(1, 4, 1);

    auto traj3d_logddp = plot3(ax_logddp, x_logddp, y_logddp, z_logddp);
    traj3d_logddp->display_name("LogDDP Trajectory");
    traj3d_logddp->line_style("-");
    traj3d_logddp->line_width(2);
    traj3d_logddp->color("blue");

    hold(ax_logddp, true);

    auto ref3d_logddp = plot3(ax_logddp, x_ref, y_ref, z_ref);
    ref3d_logddp->display_name("Reference");
    ref3d_logddp->line_style("--");
    ref3d_logddp->line_width(1);
    ref3d_logddp->color("red");

    auto proj_xy_logddp = plot3(ax_logddp, x_logddp, y_logddp, std::vector<double>(x_logddp.size(), 0.0));
    proj_xy_logddp->display_name("X-Y Projection");
    proj_xy_logddp->line_style("--");
    proj_xy_logddp->line_width(1);
    proj_xy_logddp->color("gray");

    xlabel(ax_logddp, "X [m]");
    ylabel(ax_logddp, "Y [m]");
    zlabel(ax_logddp, "Z [m]");
    xlim(ax_logddp, {-4, 4});
    ylim(ax_logddp, {-2, 2});
    zlim(ax_logddp, {0, 4});
    title(ax_logddp, "LogDDP");
    auto leg_logddp = matplot::legend(ax_logddp);
    leg_logddp->location(legend::general_alignment::topleft);
    grid(ax_logddp, true);

    // --- Subplot 3: IPDDP ---
    auto ax_ipddp = subplot(1, 4, 2);

    auto traj3d_ipddp = plot3(ax_ipddp, x_ipddp, y_ipddp, z_ipddp);
    traj3d_ipddp->display_name("IPDDP Trajectory");
    traj3d_ipddp->line_style("-");
    traj3d_ipddp->line_width(2);
    traj3d_ipddp->color("blue");

    hold(ax_ipddp, true);

    auto ref3d_ipddp = plot3(ax_ipddp, x_ref, y_ref, z_ref);
    ref3d_ipddp->display_name("Reference");
    ref3d_ipddp->line_style("--");
    ref3d_ipddp->line_width(1);
    ref3d_ipddp->color("red");

    auto proj_xy_ipddp = plot3(ax_ipddp, x_ipddp, y_ipddp, std::vector<double>(x_ipddp.size(), 0.0));
    proj_xy_ipddp->display_name("X-Y Projection");
    proj_xy_ipddp->line_style("--");
    proj_xy_ipddp->line_width(1);
    proj_xy_ipddp->color("gray");

    xlabel(ax_ipddp, "X [m]");
    ylabel(ax_ipddp, "Y [m]");
    zlabel(ax_ipddp, "Z [m]");
    xlim(ax_ipddp, {-4, 4});
    ylim(ax_ipddp, {-2, 2});
    zlim(ax_ipddp, {0, 4});
    title(ax_ipddp, "IPDDP");
    auto leg_ipddp = matplot::legend(ax_ipddp);
    leg_ipddp->location(legend::general_alignment::topleft);
    grid(ax_ipddp, true);

    // --- Subplot 4: MSIPDDP ---
    auto ax_msipddp = subplot(1, 4, 3);

    auto traj3d_msipddp = plot3(ax_msipddp, x_msipddp, y_msipddp, z_msipddp);
    traj3d_msipddp->display_name("MSIPDDP Trajectory");
    traj3d_msipddp->line_style("-");
    traj3d_msipddp->line_width(2);
    traj3d_msipddp->color("blue");

    hold(ax_msipddp, true);

    auto ref3d_msipddp = plot3(ax_msipddp, x_ref, y_ref, z_ref);
    ref3d_msipddp->display_name("Reference");
    ref3d_msipddp->line_style("--");
    ref3d_msipddp->line_width(1);
    ref3d_msipddp->color("red");

    auto proj_xy_msipddp = plot3(ax_msipddp, x_msipddp, y_msipddp, std::vector<double>(x_msipddp.size(), 0.0));
    proj_xy_msipddp->display_name("X-Y Projection");
    proj_xy_msipddp->line_style("--");
    proj_xy_msipddp->line_width(1);
    proj_xy_msipddp->color("gray");

    xlabel(ax_msipddp, "X [m]");
    ylabel(ax_msipddp, "Y [m]");
    zlabel(ax_msipddp, "Z [m]");
    xlim(ax_msipddp, {-4, 4});
    ylim(ax_msipddp, {-2, 2});
    zlim(ax_msipddp, {0, 4});
    title(ax_msipddp, "MSIPDDP");
    auto leg_msipddp = matplot::legend(ax_msipddp);
    leg_msipddp->location(legend::general_alignment::topleft);
    grid(ax_msipddp, true);

    main_figure->draw();
    main_figure->save(plotDirectory + "/quadrotor_lemniscate_3d_comparison.png");
    std::cout << "Saved combined 3D trajectory plot to "
              << (plotDirectory + "/quadrotor_lemniscate_3d_comparison.png") << std::endl;

    // --------------------------------------------------------
    // 8. Plot computation times
    // --------------------------------------------------------
    auto time_figure = figure(true);
    time_figure->size(1200, 600);

    std::vector<double> solve_times = {
        solve_time_asddp / 1000000.0,      // Convert to seconds
        solve_time_logddp / 1000000.0,     // Convert to seconds
        solve_time_ipddp / 1000000.0,      // Convert to seconds
        solve_time_msipddp / 1000000.0     // Convert to seconds
    };

    std::vector<std::string> solver_names = {
        "ASDDP", "LogDDP", "IPDDP", "MSIPDDP"
    };

    auto ax_times = time_figure->current_axes();
    auto b = bar(ax_times, solve_times);
    ax_times->xticks(matplot::iota(1.0, static_cast<double>(solver_names.size())));
    ax_times->xticklabels(solver_names);
    title(ax_times, "Quadrotor Figure-8 Solver Computation Time Comparison");
    xlabel(ax_times, "Solver");
    ylabel(ax_times, "Solve Time (seconds)");
    grid(ax_times, true);

    time_figure->draw();
    time_figure->save(plotDirectory + "/quadrotor_computation_time_comparison.png");
    std::cout << "Saved computation time plot to "
              << (plotDirectory + "/quadrotor_computation_time_comparison.png") << std::endl;

    // --------------------------------------------------------
    // 9. Print summary
    // --------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "     Quadrotor Figure-8 Benchmark Summary\n";
    std::cout << "========================================\n";
    std::cout << "Solver    | Final Cost | Solve Time (s)\n";
    std::cout << "----------|------------|---------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "ASDDP     | " << std::setw(10) << cost_asddp 
              << " | " << std::setw(13) << solve_time_asddp / 1000000.0 << "\n";
    std::cout << "LogDDP    | " << std::setw(10) << cost_logddp 
              << " | " << std::setw(13) << solve_time_logddp / 1000000.0 << "\n";
    std::cout << "IPDDP     | " << std::setw(10) << cost_ipddp 
              << " | " << std::setw(13) << solve_time_ipddp / 1000000.0 << "\n";
    std::cout << "MSIPDDP   | " << std::setw(10) << cost_msipddp 
              << " | " << std::setw(13) << solve_time_msipddp / 1000000.0 << "\n";
    std::cout << "========================================\n\n";

    return 0;
}
