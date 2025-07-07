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

// Quadrotor Model Predictive Control (MPC) with Collision Avoidance
// This example demonstrates:
// - Multi-rate MPC: Control at 10Hz, simulation at 100Hz
// - Collision avoidance with multiple obstacles
// - Figure-8 reference trajectory 
// - Quaternion-based attitude representation
// - Animated visualization with obstacles

#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <memory>
#include <cstdlib>

#include "cddp.hpp"
#include "matplot/matplot.h"

namespace fs = std::filesystem;
using namespace matplot;

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

int main()
{
    // --------------------------
    // 1. Problem and MPC Setup
    // --------------------------
    const int state_dim = 13;   // [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    const int control_dim = 4;  // [f1, f2, f3, f4]
    const int mpc_horizon = 20; // MPC prediction horizon
    const double mpc_timestep = 0.1; // MPC timestep (aligned with MPC update rate)
    const std::string integration_type = "rk4";

    // Simulation parameters
    const double sim_time = 12.0;
    const double sim_dt = 0.01;      // Simulation timestep (100 Hz)
    const double mpc_dt = 0.1;       // MPC update rate (10 Hz)
    const int mpc_update_freq = static_cast<int>(mpc_dt / sim_dt); // Update MPC every 10 sim steps

    // Quadrotor parameters (matching figure-eight example)
    const double mass = 1.2;         // 1.2kg quadrotor
    const double arm_length = 0.165; // 16.5cm arm length
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 7.782e-3; // Ixx
    inertia_matrix(1, 1) = 7.782e-3; // Iyy
    inertia_matrix(2, 2) = 1.439e-2; // Izz

    // Create a quadrotor instance for dynamics propagation (using sim_dt for actual simulation)
    auto dyn_system_template = std::make_unique<cddp::Quadrotor>(sim_dt, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices (matching figure-eight example)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(0, 0) = 1.0;  // x position
    Q(1, 1) = 1.0;  // y position
    Q(2, 2) = 1.0;  // z position
    // Quaternion components not penalized in tracking
    // Velocity components not penalized

    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Define obstacle parameters (used later for MPC collision avoidance)
    double obstacle_radius = 0.5; // 50 cm (same as ball_radius in trajectory optimization)
    double figure8_scale = 3.0;     // 3m radius
    double constant_altitude = 2.0; // 2m altitude
    Eigen::Vector3d obstacle_center(0.0, 0.0, constant_altitude); // Center of the obstacle

    // Control constraints (motor thrust limits - matching figure-eight)
    const double min_force = 0.0;
    const double max_force = 4.0;

    // --------------------------
    // Step 1: Solve EXACT SAME trajectory optimization as figure-eight example (without obstacle)
    // --------------------------
    std::cout << "Step 1: Computing figure-8 reference trajectory (same as figure_eight_horizontal_safe)..." << std::endl;
    
    // Trajectory parameters (exact same as figure-eight)
    int horizon = 400;
    double timestep = 0.02;
    double total_time = horizon * timestep;
    double omega = 2.0 * M_PI / total_time; // completes 1 cycle
    
    // Generate figure-8 reference states
    std::vector<Eigen::VectorXd> figure8_reference_states;
    figure8_reference_states.reserve(horizon + 1);
    
    for (int i = 0; i <= horizon; ++i)
    {
        double t = i * timestep;
        double angle = omega * t;
        
        // Lemniscate of Gerono for (x, y)
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
    
    // Goal state - hover at starting point
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = figure8_scale; // x
    goal_state(2) = constant_altitude;
    goal_state(3) = 1.0; // qw
    
    // Initial state - same as goal
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = figure8_scale;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0;
    
    // Solver options (exact same as figure-eight)
    cddp::CDDPOptions options;
    options.max_iterations = 100;
    options.verbose = true;
    options.debug = false;
    options.tolerance = 1e-6;
    options.acceptable_tolerance = 1e-7;
    options.use_ilqr = true;
    options.enable_parallel = false;
    options.num_threads = 1;
    
    // Line search options
    options.line_search.max_iterations = 10;
    
    // Regularization options
    options.regularization.initial_value = 1e-3;
    
    // IPDDP-specific options
    options.ipddp.barrier.mu_initial = 1e-1;
    
    // MSIPDDP-specific options
    options.msipddp.barrier.mu_initial = 1e-1;
    options.msipddp.segment_length = 2;
    options.msipddp.rollout_type = "nonlinear";
    options.msipddp.use_controlled_rollout = false;
    
    // Create objective
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, figure8_reference_states, timestep);
    
    // Create system
    auto system = std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);
    
    // Instantiate CDDP solver
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::move(system),
        std::move(objective),
        options);
    
    // Control constraints only (no obstacle in first solve)
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
    cddp_solver.addPathConstraint("ControlConstraint", 
        std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
    
    // Initial trajectory guess
    double hover_thrust = mass * 9.81 / 4.0;
    hover_thrust = std::max(min_force, std::min(max_force, hover_thrust));
    std::vector<Eigen::VectorXd> X_init = figure8_reference_states; // Use reference as initial guess
    std::vector<Eigen::VectorXd> U_init(horizon, hover_thrust * Eigen::VectorXd::Ones(control_dim));
    cddp_solver.setInitialTrajectory(X_init, U_init);
    
    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve("MSIPDDP");
    
    options.max_iterations = 500;
    options.warm_start = true;
    
    // Resolve problem with ball constraint (exact same as figure-eight)
    cddp::CDDP solver_ball(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, figure8_reference_states, timestep),
        options);
    solver_ball.addPathConstraint("ControlConstraint", 
        std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
    
    // Ball constraint (exact same parameters)
    double ball_radius = 0.5; // 50 cm
    Eigen::Vector3d ball_center(0.0, 0.0, constant_altitude); // Center of the ball
    solver_ball.addPathConstraint("BallConstraint", 
        std::make_unique<cddp::BallConstraint>(ball_radius + 0.2, ball_center));
    
    // Initial trajectory from first solution
    auto initial_X = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto initial_U = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    solver_ball.setInitialTrajectory(initial_X, initial_U);
    
    // Solve with ball constraint
    cddp::CDDPSolution solution_ball = solver_ball.solve("MSIPDDP");
    
    // Extract solution
    auto X_opt = std::any_cast<std::vector<Eigen::VectorXd>>(solution_ball.at("state_trajectory"));
    auto U_opt = std::any_cast<std::vector<Eigen::VectorXd>>(solution_ball.at("control_trajectory"));
    auto t_sol = std::any_cast<std::vector<double>>(solution_ball.at("time_points"));
    
    std::cout << "Trajectory optimization completed!" << std::endl;
    std::cout << "Final state = " << X_opt.back().transpose() << std::endl;
    
    // --------------------------
    // Step 2: Use optimized trajectory for MPC reference
    // --------------------------
    std::cout << "\nStep 2: Running MPC with optimized reference trajectory from figure-eight solution..." << std::endl;
    
    // Use the optimized trajectory from figure-eight solution directly as reference
    // The trajectory is already at 0.02s timestep for 8 seconds (400 steps)
    // We'll cycle through it for longer simulation times
    std::vector<Eigen::VectorXd> reference_trajectory;
    int total_sim_steps = static_cast<int>(sim_time / sim_dt);
    
    // For MPC, we need reference beyond simulation time for horizon
    for (int i = 0; i <= total_sim_steps + mpc_horizon * 10; ++i)
    {
        double t = i * sim_dt;
        // Find corresponding index in optimized trajectory (0.02s timestep)
        int opt_idx = static_cast<int>(t / timestep) % X_opt.size();
        reference_trajectory.push_back(X_opt[opt_idx]);
    }

    // Initial state (use exact initial state from optimization)
    Eigen::VectorXd current_state = initial_state;

    // IPDDP Solver Options
    cddp::CDDPOptions options_ipddp;
    options_ipddp.max_iterations = 100;  // Increased for better convergence
    options_ipddp.tolerance = 1e-4;      // Tighter tolerance
    options_ipddp.acceptable_tolerance = 1e-3;  // Add acceptable tolerance
    options_ipddp.verbose = false;
    options_ipddp.debug = false;
    options_ipddp.enable_parallel = true;
    options_ipddp.num_threads = 10;
    options_ipddp.regularization.initial_value = 1e-3;
    options_ipddp.warm_start = true;
    options_ipddp.msipddp.segment_length = 5;
    options_ipddp.msipddp.barrier.mu_initial = 1e-1;  // Add barrier parameter

    // Simulation history storage
    std::vector<Eigen::VectorXd> state_history;
    std::vector<Eigen::VectorXd> control_history;
    std::vector<double> time_history;
    state_history.push_back(current_state);
    time_history.push_back(0.0);

    // Initial trajectory guess for the first MPC solve
    std::vector<Eigen::VectorXd> X_guess(mpc_horizon + 1, current_state);
    std::vector<Eigen::VectorXd> U_guess(mpc_horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Initialize with hovering thrust (already defined and saturated above)
    for (auto &u : U_guess)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }

    // --------------------------
    // 2. MPC Loop
    // --------------------------
    std::cout << "Running MPC with Collision Avoidance for Quadrotor Figure-8 Trajectory..." << std::endl;
    std::cout << "Simulation: " << 1.0/sim_dt << " Hz, MPC: " << 1.0/mpc_dt << " Hz" << std::endl;
    std::cout << "Obstacle: radius = " << obstacle_radius << " m at center [" << obstacle_center.transpose() << "]" << std::endl;
    
    double current_time = 0.0;
    int sim_steps = static_cast<int>(sim_time / sim_dt);

    // Storage for current MPC solution
    std::vector<Eigen::VectorXd> current_mpc_states;
    std::vector<Eigen::VectorXd> current_mpc_controls;
    Eigen::VectorXd current_control = hover_thrust * Eigen::VectorXd::Ones(control_dim); // hover_thrust is already saturated
    
    // Reference tracking state
    int last_ref_idx = 0;
    
    // Goal tolerance for termination
    const double position_tolerance = 0.5; // 50cm tolerance
    const double min_simulation_time = 6.0; // Minimum 6 seconds to ensure at least partial figure-8
    bool goal_reached = false;

    for (int k = 0; k < sim_steps; ++k)
    {
        // Check if goal is reached (after minimum simulation time)
        if (current_time >= min_simulation_time) {
            Eigen::Vector3d position_error = current_state.head(3) - goal_state.head(3);
            double position_error_norm = position_error.norm();
            
            if (position_error_norm < position_tolerance) {
                goal_reached = true;
                std::cout << "\nGoal reached! Position error: " << position_error_norm 
                          << " m (tolerance: " << position_tolerance << " m)" << std::endl;
                std::cout << "Terminating simulation at time: " << current_time << " s" << std::endl;
                break;
            }
        }
        // Only update MPC at specified frequency
        if (k % mpc_update_freq == 0)
        {
            // Find closest point on reference trajectory with directional consistency
            Eigen::Vector3d current_pos = current_state.head(3);
            Eigen::Vector3d current_vel = current_state.segment(7, 3);
            
            // Use the last reference index to prevent backward jumps
            
            // Search window: only look forward from last position
            // Allow some backward search but limited to handle small deviations
            int search_start = std::max(0, last_ref_idx - 50);  // Allow 50 steps back max
            int search_end = std::min((int)reference_trajectory.size(), last_ref_idx + 200);  // Look ahead up to 200 steps
            
            double min_dist_sq = std::numeric_limits<double>::max();
            int closest_idx = last_ref_idx;
            
            // Find closest point within the search window
            for (int i = search_start; i < search_end; ++i)
            {
                Eigen::Vector3d ref_pos = reference_trajectory[i].head(3);
                double dist_sq = (current_pos - ref_pos).squaredNorm();
                
                // Add preference for forward progress
                if (i >= last_ref_idx) {
                    dist_sq *= 0.8;  // 20% bonus for forward points
                }
                
                if (dist_sq < min_dist_sq)
                {
                    min_dist_sq = dist_sq;
                    closest_idx = i;
                }
            }
            
            // Update last reference index
            last_ref_idx = closest_idx;
            
            // Create MPC reference trajectory starting from closest point
            // Add adaptive forward offset based on velocity to encourage progress
            double velocity_mag = current_vel.norm();
            int forward_offset = std::max(5, std::min(20, (int)(velocity_mag * 15)));  // Increased lookahead
            
            int ref_start_idx = std::min(closest_idx + forward_offset, (int)reference_trajectory.size() - 1);
            
            std::vector<Eigen::VectorXd> mpc_ref_traj;
            for (int i = 0; i <= mpc_horizon; ++i)
            {
                // Use finer sampling (every mpc_update_freq steps) for smoother reference
                int idx = std::min(ref_start_idx + i * mpc_update_freq, (int)reference_trajectory.size() - 1);
                mpc_ref_traj.push_back(reference_trajectory[idx]);
            }
            Eigen::VectorXd mpc_goal_state = mpc_ref_traj.back();

            // Create objective for this MPC step
            auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, mpc_goal_state, mpc_ref_traj, mpc_timestep);

            // Create CDDP solver instance for this MPC step
            auto system = std::make_unique<cddp::Quadrotor>(mpc_timestep, mass, inertia_matrix, arm_length, integration_type);
            cddp::CDDP cddp_solver(current_state, mpc_goal_state, mpc_horizon, mpc_timestep,
                                   std::move(system), std::move(objective), options_ipddp);

            // Add control constraints
            Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
            Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
            cddp_solver.addPathConstraint("ControlConstraint",
                std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));

            // Add collision avoidance constraint with safety margin
            double safety_margin = 0.2; // Same margin as in trajectory optimization
            cddp_solver.addPathConstraint("BallConstraint",
                std::make_unique<cddp::BallConstraint>(obstacle_radius + safety_margin, obstacle_center));

            // Set initial trajectory (warm start)
            cddp_solver.setInitialTrajectory(X_guess, U_guess);

            // Solve the OCP
            cddp::CDDPSolution solution = cddp_solver.solve("MSIPDDP");

            // Extract and apply the first control
            auto status = std::any_cast<std::string>(solution.at("status_message"));
            if (status != "OptimalSolutionFound" && status != "AcceptableSolutionFound")
            {
                std::cerr << "Warning: Solver did not converge at time " << current_time << ". Status: " << status << std::endl;
            }

            // Extract solution
            current_mpc_states = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
            current_mpc_controls = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
            
            // Warm start for the next iteration: shift the solution
            for(int i = 0; i < mpc_horizon - 1; ++i)
            {
                X_guess[i] = current_mpc_states[i + 1];
                U_guess[i] = current_mpc_controls[i + 1];
            }
            X_guess[mpc_horizon - 1] = current_mpc_states[mpc_horizon];
            X_guess[mpc_horizon] = current_mpc_states[mpc_horizon];
            U_guess[mpc_horizon - 1] = current_mpc_controls[mpc_horizon - 1];
        }
        
        // Determine which control to apply
        // MPC control index within the current MPC update interval
        int mpc_idx = (k % mpc_update_freq) / mpc_update_freq;  // This will always be 0
        
        // Actually, we should use the first control from the MPC solution
        if (!current_mpc_controls.empty())
        {
            current_control = current_mpc_controls[0];  // Always use first control
        }
        
        // Apply control saturation to enforce bounds at low level
        for (int i = 0; i < control_dim; ++i) {
            current_control(i) = std::max(min_force, std::min(max_force, current_control(i)));
        }
        
        // Propagate system dynamics with saturated control
        current_state = dyn_system_template->getDiscreteDynamics(current_state, current_control, 0.0);

        // Update history
        state_history.push_back(current_state);
        control_history.push_back(current_control);
        current_time += sim_dt;
        time_history.push_back(current_time);

        // Check constraints
        double dist_to_obstacle = (current_state.head(3) - obstacle_center).norm();
        bool collision_violated = dist_to_obstacle < obstacle_radius;
        
        bool control_violated = false;
        for (int i = 0; i < control_dim; ++i) {
            if (current_control(i) < min_force - 1e-6 || current_control(i) > max_force + 1e-6) {
                control_violated = true;
            }
        }
        
        // Progress output (every second)
        if (k % static_cast<int>(1.0 / sim_dt) == 0)
        {
            // Find current progress on reference (for display)
            double progress = (double)last_ref_idx / reference_trajectory.size() * 100.0;
            
            std::cout << "Time: " << current_time 
                      << "s, Position: [" << current_state.head(3).transpose() << "]"
                      << ", Dist to obstacle: " << dist_to_obstacle << " m"
                      << ", Progress: " << progress << "%"
                      << ", MPC updates: " << k/mpc_update_freq;
            
            if (collision_violated) {
                std::cout << " [COLLISION CONSTRAINT VIOLATED!]";
            }
            if (control_violated) {
                std::cout << " [CONTROL CONSTRAINT VIOLATED!]";
            }
            std::cout << std::endl;
        }
    }
    
    if (goal_reached) {
        std::cout << "Simulation finished successfully - goal reached!" << std::endl;
    } else {
        std::cout << "Simulation finished - time limit reached." << std::endl;
    }

    // --------------------------
    // 3. Plotting
    // --------------------------
    // Convert trajectories to plottable vectors
    std::vector<double> x_hist, y_hist, z_hist;
    std::vector<double> phi_hist, theta_hist, psi_hist;
    for(const auto& s : state_history)
    {
        x_hist.push_back(s(0));
        y_hist.push_back(s(1));
        z_hist.push_back(s(2));
        
        // Convert quaternion to Euler angles
        Eigen::Vector3d euler = quaternionToEuler(s(3), s(4), s(5), s(6));
        phi_hist.push_back(euler(0));
        theta_hist.push_back(euler(1));
        psi_hist.push_back(euler(2));
    }

    std::vector<double> f1_hist, f2_hist, f3_hist, f4_hist;
    for(const auto& u : control_history)
    {
        f1_hist.push_back(u(0));
        f2_hist.push_back(u(1));
        f3_hist.push_back(u(2));
        f4_hist.push_back(u(3));
    }

    std::vector<double> x_ref, y_ref, z_ref;
    for(const auto& s : reference_trajectory)
    {
        x_ref.push_back(s(0));
        y_ref.push_back(s(1));
        z_ref.push_back(s(2));
    }

    // Create plot directory
    const std::string plotDirectory = "../results/examples";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Main plotting figure
    auto f1 = figure(true);
    f1->size(1200, 1000);

    // 3D trajectory plot with obstacles
    auto ax1 = subplot(2, 2, 1);
    auto traj_plot = plot3(ax1, x_hist, y_hist, z_hist);
    traj_plot->line_width(2);
    traj_plot->color("blue");
    traj_plot->display_name("Actual Trajectory");
    
    hold(ax1, true);
    auto ref_plot = plot3(ax1, x_ref, y_ref, z_ref);
    ref_plot->line_width(2);
    ref_plot->line_style("--");
    ref_plot->color("red");
    ref_plot->display_name("Reference Trajectory");

    // Plot obstacle as sphere
    int n_sphere = 20; 
    auto phi = linspace(0, M_PI, n_sphere);
    auto theta = linspace(0, 2 * M_PI, n_sphere);

    std::vector<std::vector<double>> sx(n_sphere, std::vector<double>(n_sphere));
    std::vector<std::vector<double>> sy(n_sphere, std::vector<double>(n_sphere));
    std::vector<std::vector<double>> sz(n_sphere, std::vector<double>(n_sphere));

    for (int i = 0; i < n_sphere; i++)
    {
        for (int j = 0; j < n_sphere; j++)
        {
            sx[i][j] = obstacle_center(0) + obstacle_radius * std::sin(phi[i]) * std::cos(theta[j]);
            sy[i][j] = obstacle_center(1) + obstacle_radius * std::sin(phi[i]) * std::sin(theta[j]);
            sz[i][j] = obstacle_center(2) + obstacle_radius * std::cos(phi[i]);
        }
    }
    auto sphere_surf = surf(ax1, sx, sy, sz);
    sphere_surf->edge_color("red");
    sphere_surf->face_alpha(0.5);
    
    // Project trajectory onto x-y plane at z=0
    std::vector<double> z_proj_traj(x_hist.size(), 0.0);
    auto proj_traj = plot3(ax1, x_hist, y_hist, z_proj_traj);
    proj_traj->line_width(1);
    proj_traj->line_style("--");
    proj_traj->color("gray");
    proj_traj->display_name("Trajectory X-Y Projection");
    
    // Project reference onto x-y plane at z=0
    std::vector<double> z_proj_ref(x_ref.size(), 0.0);
    auto proj_ref = plot3(ax1, x_ref, y_ref, z_proj_ref);
    proj_ref->line_width(1);
    proj_ref->line_style(":");
    proj_ref->color({1.0, 0.6, 0.6}); // Light red
    proj_ref->display_name("Reference X-Y Projection");
    
    // Plot projection of obstacle circle on x-y plane
    int n_circle = 100;
    std::vector<double> circle_x(n_circle), circle_y(n_circle), circle_z(n_circle);
    for (int i = 0; i < n_circle; i++)
    {
        double angle = 2.0 * M_PI * i / n_circle;
        circle_x[i] = obstacle_center(0) + obstacle_radius * std::cos(angle);
        circle_y[i] = obstacle_center(1) + obstacle_radius * std::sin(angle);
        circle_z[i] = 0.0; // Projection onto x-y plane
    }
    auto circle_plot = plot3(ax1, circle_x, circle_y, circle_z);
    circle_plot->line_width(2);
    circle_plot->line_style("-");
    circle_plot->color("black");
    circle_plot->display_name("Obstacle X-Y Projection");

    // Mark start and end points
    auto start_scatter = scatter3(ax1, std::vector<double>{x_hist.front()}, std::vector<double>{y_hist.front()}, std::vector<double>{z_hist.front()});
    start_scatter->marker_color("g").marker_size(100);
    auto end_scatter = scatter3(ax1, std::vector<double>{x_hist.back()}, std::vector<double>{y_hist.back()}, std::vector<double>{z_hist.back()});
    end_scatter->marker_color("r").marker_size(100);

    title(ax1, "3D Trajectory with Obstacles");
    xlabel(ax1, "X [m]");
    ylabel(ax1, "Y [m]");
    zlabel(ax1, "Z [m]");
    legend(ax1, "show");
    grid(ax1, true);

    // Position vs time
    auto ax2 = subplot(2, 2, 2);
    plot(ax2, time_history, x_hist, "r-")->line_width(2).display_name("x");
    hold(ax2, true);
    plot(ax2, time_history, y_hist, "g-")->line_width(2).display_name("y");
    plot(ax2, time_history, z_hist, "b-")->line_width(2).display_name("z");
    title(ax2, "Position vs Time");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Position [m]");
    legend(ax2, "show");
    grid(ax2, true);

    // Attitude vs time
    auto ax3 = subplot(2, 2, 3);
    plot(ax3, time_history, phi_hist, "r-")->line_width(2).display_name("roll");
    hold(ax3, true);
    plot(ax3, time_history, theta_hist, "g-")->line_width(2).display_name("pitch");
    plot(ax3, time_history, psi_hist, "b-")->line_width(2).display_name("yaw");
    title(ax3, "Attitude vs Time");
    xlabel(ax3, "Time [s]");
    ylabel(ax3, "Angle [rad]");
    legend(ax3, "show");
    grid(ax3, true);

    // Control inputs vs time
    auto ax4 = subplot(2, 2, 4);
    std::vector<double> control_time_hist = time_history;
    control_time_hist.pop_back(); // control history is one step shorter
    plot(ax4, control_time_hist, f1_hist, "r-")->line_width(2).display_name("f1");
    hold(ax4, true);
    plot(ax4, control_time_hist, f2_hist, "g-")->line_width(2).display_name("f2");
    plot(ax4, control_time_hist, f3_hist, "b-")->line_width(2).display_name("f3");
    plot(ax4, control_time_hist, f4_hist, "k-")->line_width(2).display_name("f4");
    title(ax4, "Motor Forces vs Time");
    xlabel(ax4, "Time [s]");
    ylabel(ax4, "Force [N]");
    legend(ax4, "show");
    grid(ax4, true);

    // Save and show plot
    save(plotDirectory + "/quadrotor_mpc_collision_avoidance.png");
    std::cout << "Saved plot to " << plotDirectory << "/quadrotor_mpc_collision_avoidance.png" << std::endl;

    // --------------------------
    // 4. Animation
    // --------------------------
    std::cout << "Generating animation frames..." << std::endl;
    
    // Animation of the quadrotor frame
    auto f_anim = figure();
    f_anim->size(800, 600);
    auto ax_anim = f_anim->current_axes();

    // For collecting the trajectory as we go
    std::vector<double> anim_x, anim_y, anim_z;
    anim_x.reserve(state_history.size());
    anim_y.reserve(state_history.size());
    anim_z.reserve(state_history.size());

    // Render every Nth frame to reduce #images
    int frame_stride = 20;  // With 100Hz simulation, this gives 5 fps animation
    double prop_radius = 0.02; // radius for small spheres at motor ends
    int frame_count = 0;

    for (size_t i = 0; i < state_history.size(); i += frame_stride)
    {
        ax_anim->clear();
        ax_anim->hold(true);
        ax_anim->grid(true);

        // Current state
        double x = state_history[i](0);
        double y = state_history[i](1);
        double z = state_history[i](2);

        // Accumulate path
        anim_x.push_back(x);
        anim_y.push_back(y);
        anim_z.push_back(z);

        // Plot the partial trajectory so far (in gray)
        if (anim_x.size() > 1) {
            auto path_plot = plot3(ax_anim, anim_x, anim_y, anim_z);
            path_plot->line_width(2);
            path_plot->line_style("-");
            path_plot->color("gray");
        }

        // Plot reference trajectory (in light red dashed)
        std::vector<double> ref_x_show, ref_y_show, ref_z_show;
        int ref_end = std::min((int)reference_trajectory.size(), (int)(i + 200)); // Show future reference
        for (int j = 0; j < ref_end; j += 10) {
            ref_x_show.push_back(reference_trajectory[j](0));
            ref_y_show.push_back(reference_trajectory[j](1));
            ref_z_show.push_back(reference_trajectory[j](2));
        }
        if (!ref_x_show.empty()) {
            auto ref_traj = plot3(ax_anim, ref_x_show, ref_y_show, ref_z_show);
            ref_traj->line_width(1);
            ref_traj->line_style("--");
            ref_traj->color({1.0, 0.6, 0.6}); // Light red
        }

        // Plot obstacle
        int n_sphere_anim = 15; 
        auto phi_anim = linspace(0, M_PI, n_sphere_anim);
        auto theta_anim = linspace(0, 2 * M_PI, n_sphere_anim);

        std::vector<std::vector<double>> sx_anim(n_sphere_anim, std::vector<double>(n_sphere_anim));
        std::vector<std::vector<double>> sy_anim(n_sphere_anim, std::vector<double>(n_sphere_anim));
        std::vector<std::vector<double>> sz_anim(n_sphere_anim, std::vector<double>(n_sphere_anim));

        for (int j = 0; j < n_sphere_anim; j++)
        {
            for (int k = 0; k < n_sphere_anim; k++)
            {
                sx_anim[j][k] = obstacle_center(0) + obstacle_radius * std::sin(phi_anim[j]) * std::cos(theta_anim[k]);
                sy_anim[j][k] = obstacle_center(1) + obstacle_radius * std::sin(phi_anim[j]) * std::sin(theta_anim[k]);
                sz_anim[j][k] = obstacle_center(2) + obstacle_radius * std::cos(phi_anim[j]);
            }
        }
        auto sphere_surf_anim = surf(ax_anim, sx_anim, sy_anim, sz_anim);
        sphere_surf_anim->edge_color("red");
        sphere_surf_anim->face_alpha(0.3);
        
        // Plot x-y projections
        // Project accumulated trajectory onto x-y plane
        if (anim_x.size() > 1) {
            std::vector<double> z_proj_anim(anim_x.size(), 0.0);
            auto proj_path = plot3(ax_anim, anim_x, anim_y, z_proj_anim);
            proj_path->line_width(1);
            proj_path->line_style("--");
            proj_path->color("darkgray");
        }
        
        // Project reference trajectory onto x-y plane
        if (!ref_x_show.empty()) {
            std::vector<double> z_proj_ref_anim(ref_x_show.size(), 0.0);
            auto proj_ref_anim = plot3(ax_anim, ref_x_show, ref_y_show, z_proj_ref_anim);
            proj_ref_anim->line_width(1);
            proj_ref_anim->line_style(":");
            proj_ref_anim->color({1.0, 0.8, 0.8}); // Very light red
        }
        
        // Plot obstacle circle projection on x-y plane
        int n_circle_anim = 50;
        std::vector<double> circle_x_anim(n_circle_anim), circle_y_anim(n_circle_anim), circle_z_anim(n_circle_anim);
        for (int j = 0; j < n_circle_anim; j++)
        {
            double angle = 2.0 * M_PI * j / n_circle_anim;
            circle_x_anim[j] = obstacle_center(0) + obstacle_radius * std::cos(angle);
            circle_y_anim[j] = obstacle_center(1) + obstacle_radius * std::sin(angle);
            circle_z_anim[j] = 0.0; // Projection onto x-y plane
        }
        auto circle_anim = plot3(ax_anim, circle_x_anim, circle_y_anim, circle_z_anim);
        circle_anim->line_width(2);
        circle_anim->line_style("-");
        circle_anim->color("black");

        // Build rotation from quaternion
        Eigen::Vector4d quat(state_history[i](3), state_history[i](4), 
                            state_history[i](5), state_history[i](6));
        Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat(0), quat(1), quat(2), quat(3));

        // Arm endpoints (front, right, back, left)
        std::vector<Eigen::Vector3d> arm_endpoints;
        arm_endpoints.push_back(Eigen::Vector3d(arm_length, 0, 0));
        arm_endpoints.push_back(Eigen::Vector3d(0, arm_length, 0));
        arm_endpoints.push_back(Eigen::Vector3d(-arm_length, 0, 0));
        arm_endpoints.push_back(Eigen::Vector3d(0, -arm_length, 0));

        // Transform to world coords
        for (auto &pt : arm_endpoints)
        {
            pt = Eigen::Vector3d(x, y, z) + R * pt;
        }

        // Front-back arm (blue)
        std::vector<double> fx = {arm_endpoints[0].x(), arm_endpoints[2].x()};
        std::vector<double> fy = {arm_endpoints[0].y(), arm_endpoints[2].y()};
        std::vector<double> fz = {arm_endpoints[0].z(), arm_endpoints[2].z()};
        auto fb_arm = plot3(ax_anim, fx, fy, fz);
        fb_arm->line_width(3.0);
        fb_arm->color("blue");

        // Right-left arm (red)
        std::vector<double> rx = {arm_endpoints[1].x(), arm_endpoints[3].x()};
        std::vector<double> ry = {arm_endpoints[1].y(), arm_endpoints[3].y()};
        std::vector<double> rz = {arm_endpoints[1].z(), arm_endpoints[3].z()};
        auto rl_arm = plot3(ax_anim, rx, ry, rz);
        rl_arm->line_width(3.0);
        rl_arm->color("red");

        // Draw propellers as circles
        auto circle_points = linspace(0, 2 * M_PI, 20);
        for (size_t motor_idx = 0; motor_idx < arm_endpoints.size(); ++motor_idx)
        {
            const auto &motor_pos = arm_endpoints[motor_idx];
            std::vector<double> circ_x, circ_y, circ_z;
            
            // Color based on thrust (if available)
            std::string color = "green";
            if (i < control_history.size()) {
                double thrust = control_history[i](motor_idx);
                double normalized_thrust = thrust / max_force;
                if (normalized_thrust > 0.7) color = "red";
                else if (normalized_thrust > 0.4) color = "orange";
                else color = "green";
            }
            
            for (auto angle : circle_points)
            {
                circ_x.push_back(motor_pos.x() + prop_radius * cos(angle));
                circ_y.push_back(motor_pos.y() + prop_radius * sin(angle));
                circ_z.push_back(motor_pos.z());
            }
            auto prop_plot = plot3(ax_anim, circ_x, circ_y, circ_z);
            prop_plot->line_style("solid");
            prop_plot->line_width(2.0);
            prop_plot->color(color);
        }

        // Add time annotation in the title
        std::string time_str = "Time: " + std::to_string(i * sim_dt).substr(0, 4) + "s";

        title(ax_anim, "Quadrotor MPC with Collision Avoidance - " + time_str);
        xlabel(ax_anim, "X [m]");
        ylabel(ax_anim, "Y [m]");
        zlabel(ax_anim, "Z [m]");
        xlim(ax_anim, {-3, 3});
        ylim(ax_anim, {-2, 2});
        zlim(ax_anim, {0, 4});

        ax_anim->view(30, -45);

        std::string frameFile = plotDirectory + "/quadrotor_mpc_ca_frame_" + std::to_string(frame_count) + ".png";
        f_anim->draw();
        f_anim->save(frameFile);
        
        if (frame_count % 10 == 0) {
            std::cout << "Generated frame " << frame_count << "/" << state_history.size()/frame_stride << std::endl;
        }
        frame_count++;
    }

    // Generate GIF from frames using ImageMagick
    std::cout << "Creating animation..." << std::endl;
    std::string gif_command = "convert -delay 20 -loop 0 " + plotDirectory + "/quadrotor_mpc_ca_frame_*.png " + plotDirectory + "/quadrotor_mpc_collision_avoidance.gif";
    int result = std::system(gif_command.c_str());
    
    if (result == 0) {
        std::cout << "Animation saved to " << plotDirectory << "/quadrotor_mpc_collision_avoidance.gif" << std::endl;
        
        // Clean up frame files
        std::string cleanup_command = "rm " + plotDirectory + "/quadrotor_mpc_ca_frame_*.png";
        (void)std::system(cleanup_command.c_str());
    } else {
        std::cout << "Failed to create animation. Frame files are kept in " << plotDirectory << std::endl;
    }

    // Print final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "=================" << std::endl;
    
    // Check if goal was reached
    Eigen::Vector3d final_position_error = state_history.back().head(3) - goal_state.head(3);
    double final_position_error_norm = final_position_error.norm();
    std::cout << "Final position error: " << final_position_error_norm << " m" << std::endl;
    std::cout << "Goal tolerance: " << position_tolerance << " m" << std::endl;
    std::cout << "Goal reached: " << (goal_reached ? "YES" : "NO") << std::endl;
    std::cout << "Simulation duration: " << time_history.back() << " s" << std::endl;
    
    // Check minimum distance to obstacle and constraint violations
    double min_dist = std::numeric_limits<double>::max();
    double min_clearance = std::numeric_limits<double>::max();
    int collision_violations = 0;
    int control_violations = 0;
    
    for (size_t i = 0; i < state_history.size(); ++i)
    {
        Eigen::Vector3d pos = state_history[i].head(3);
        double dist = (pos - obstacle_center).norm();
        double clearance = dist - obstacle_radius;
        min_dist = std::min(min_dist, dist);
        min_clearance = std::min(min_clearance, clearance);
        
        if (dist < obstacle_radius) {
            collision_violations++;
        }
        
        if (i < control_history.size()) {
            for (int j = 0; j < control_dim; ++j) {
                if (control_history[i](j) < min_force - 1e-6 || 
                    control_history[i](j) > max_force + 1e-6) {
                    control_violations++;
                    break;
                }
            }
        }
    }
    
    std::cout << "Minimum distance to obstacle center: " << min_dist << " m" << std::endl;
    std::cout << "Minimum clearance (dist - radius): " << min_clearance << " m" << std::endl;
    std::cout << "Obstacle radius: " << obstacle_radius << " m" << std::endl;
    std::cout << "\nConstraint Violations:" << std::endl;
    std::cout << "  Collision violations: " << collision_violations << " / " << state_history.size() << " steps";
    if (collision_violations > 0) {
        std::cout << " [FAILED - Collision constraint violated!]";
    } else {
        std::cout << " [PASSED]";
    }
    std::cout << std::endl;
    
    std::cout << "  Control violations: " << control_violations << " / " << control_history.size() << " steps";
    if (control_violations > 0) {
        std::cout << " [FAILED - Control constraints violated!]";
    } else {
        std::cout << " [PASSED]";
    }
    std::cout << std::endl;
    
    // Check control bounds
    double min_control = std::numeric_limits<double>::max();
    double max_control = std::numeric_limits<double>::min();
    for (const auto& u : control_history) {
        for (int i = 0; i < control_dim; ++i) {
            min_control = std::min(min_control, u(i));
            max_control = std::max(max_control, u(i));
        }
    }
    std::cout << "\nControl bounds:" << std::endl;
    std::cout << "  Min control applied: " << min_control << " N (limit: " << min_force << " N)" << std::endl;
    std::cout << "  Max control applied: " << max_control << " N (limit: " << max_force << " N)" << std::endl;

    return 0;
}