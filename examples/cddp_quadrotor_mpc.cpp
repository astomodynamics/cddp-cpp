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

// Quadrotor Model Predictive Control (MPC) Example
// This example demonstrates:
// - Multi-rate MPC: Control at 10Hz, simulation at 100Hz
// - Smooth trajectory generation using S-curve
// - Quaternion-based attitude representation
// - Animated visualization with thrust-based propeller coloring

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

// Helper function: Generate target state for single waypoint navigation
Eigen::VectorXd generateTargetState(int state_dim)
{
    Eigen::VectorXd target_state = Eigen::VectorXd::Zero(state_dim);
    target_state(0) = 3.0; // x position
    target_state(1) = 0.0; // y position  
    target_state(2) = 2.0; // z position
    target_state(3) = 1.0; // qw (identity quaternion)
    target_state(4) = 0.0; // qx
    target_state(5) = 0.0; // qy
    target_state(6) = 0.0; // qz
    // velocities and angular rates remain zero
    return target_state;
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

    // Quadrotor parameters
    const double mass = 1.0;       // 1kg quadrotor
    const double arm_length = 0.2; // 20cm arm length
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 0.01; // Ixx
    inertia_matrix(1, 1) = 0.01; // Iyy
    inertia_matrix(2, 2) = 0.02; // Izz

    // Create a quadrotor instance for dynamics propagation (using sim_dt for actual simulation)
    auto dyn_system_template = std::make_unique<cddp::Quadrotor>(sim_dt, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(0, 0) = 1.0;  // x position
    Q(1, 1) = 1.0;  // y position
    Q(2, 2) = 1.0;  // z position
    Q(3, 3) = 0.0;   // qw
    Q(4, 4) = 0.0;   // qx
    Q(5, 5) = 0.0;   // qy
    Q(6, 6) = 0.0;   // qz
    Q(7, 7) = 0.1;   // vx
    Q(8, 8) = 0.1;   // vy
    Q(9, 9) = 0.1;   // vz

    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 0.01 * Q;

    // Generate target state for single waypoint navigation
    Eigen::VectorXd target_state = generateTargetState(state_dim);

    // Generate waypoint trajectory with smooth ramp to target
    int total_sim_steps = static_cast<int>(sim_time / sim_dt);
    std::vector<Eigen::VectorXd> waypoint_trajectory;
    
    // Time to reach target (seconds)
    const double target_time = 6.0;  // Reach target in 6 seconds instead of 2
    
    for (int i = 0; i <= total_sim_steps + mpc_horizon; ++i)
    {
        double t = i * sim_dt;
        double s = std::min(1.0, t / target_time);  // Progress from 0 to 1
        
        // Use smooth S-curve (smoothstep function)
        double smooth_s = s * s * (3.0 - 2.0 * s);
        
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);
        ref_state.head(3) = smooth_s * target_state.head(3);  // Interpolate position
        ref_state(3) = 1.0;  // Identity quaternion
        
        waypoint_trajectory.push_back(ref_state);
    }

    // Initial state (at origin with identity quaternion)
    Eigen::VectorXd current_state = Eigen::VectorXd::Zero(state_dim);
    current_state(3) = 1.0; // Identity quaternion: qw = 1

    // IPDDP Solver Options
    cddp::CDDPOptions options_ipddp;
    options_ipddp.max_iterations = 20;
    options_ipddp.tolerance = 1e-3;
    options_ipddp.verbose = false;
    options_ipddp.debug = false;
    options_ipddp.enable_parallel = true;
    options_ipddp.num_threads = 10;
    options_ipddp.regularization.initial_value = 1e-3;
    options_ipddp.warm_start = true;

    // Control constraints (motor thrust limits)
    const double min_force = 0.0;
    const double max_force = 4.0;
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);

    // Simulation history storage
    std::vector<Eigen::VectorXd> state_history;
    std::vector<Eigen::VectorXd> control_history;
    std::vector<double> time_history;
    state_history.push_back(current_state);
    time_history.push_back(0.0);

    // Initial trajectory guess for the first MPC solve
    std::vector<Eigen::VectorXd> X_guess(mpc_horizon + 1, current_state);
    std::vector<Eigen::VectorXd> U_guess(mpc_horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Initialize with hovering thrust
    double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U_guess)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }

    // --------------------------
    // 2. MPC Loop
    // --------------------------
    std::cout << "Running IPDDP-based MPC for Quadrotor Point-to-Point Control..." << std::endl;
    std::cout << "Simulation: " << 1.0/sim_dt << " Hz, MPC: " << 1.0/mpc_dt << " Hz" << std::endl;
    double current_time = 0.0;
    int sim_steps = static_cast<int>(sim_time / sim_dt);

    // Storage for current MPC solution
    std::vector<Eigen::VectorXd> current_mpc_states;
    std::vector<Eigen::VectorXd> current_mpc_controls;
    std::vector<Eigen::MatrixXd> current_mpc_gains;
    Eigen::VectorXd current_control = Eigen::VectorXd::Zero(control_dim);
    
    // Initialize with hover thrust
    current_control = (mass * 9.81 / 4.0) * Eigen::VectorXd::Ones(control_dim);

    for (int k = 0; k < sim_steps; ++k)
    {
        // Only update MPC at specified frequency
        if (k % mpc_update_freq == 0)
        {
            // Get current reference trajectory slice for the MPC horizon
            std::vector<Eigen::VectorXd> mpc_ref_traj;
            int ref_start_idx = k;
            for (int i = 0; i <= mpc_horizon; ++i)
            {
                int idx = std::min(ref_start_idx + i * mpc_update_freq, (int)waypoint_trajectory.size() - 1);
                mpc_ref_traj.push_back(waypoint_trajectory[idx]);
            }
            Eigen::VectorXd mpc_goal_state = mpc_ref_traj.back();

            // Create objective for this MPC step
            auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, mpc_goal_state, mpc_ref_traj, mpc_timestep);

            // Create CDDP solver instance for this MPC step
            auto system = std::make_unique<cddp::Quadrotor>(mpc_timestep, mass, inertia_matrix, arm_length, integration_type);
            cddp::CDDP cddp_solver(current_state, mpc_goal_state, mpc_horizon, mpc_timestep,
                                   std::move(system), std::move(objective), options_ipddp);

            // Add control constraints
            cddp_solver.addPathConstraint("ControlConstraint",
                std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));

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
            
            // Extract feedback gains if available
            if (solution.find("control_feedback_gains_K") != solution.end())
            {
                current_mpc_gains = std::any_cast<std::vector<Eigen::MatrixXd>>(solution.at("control_feedback_gains_K"));
            }
        
            
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
        // We need to figure out which MPC control interval we're in
        int steps_per_mpc_interval = static_cast<int>(mpc_timestep / sim_dt);
        int mpc_idx = (k % mpc_update_freq) / steps_per_mpc_interval;
        
        if (!current_mpc_controls.empty() && mpc_idx < current_mpc_controls.size())
        {
            // Option 1: Simple open-loop control
            current_control = current_mpc_controls[mpc_idx];
            
            // Option 2: Use feedback gains if available (commented out for now)
            // if (!current_mpc_gains.empty() && mpc_idx < current_mpc_gains.size() && mpc_idx < current_mpc_states.size())
            // {
            //     Eigen::VectorXd state_error = current_state - current_mpc_states[mpc_idx];
            //     current_control = current_mpc_controls[mpc_idx] - current_mpc_gains[mpc_idx] * state_error;
            //     
            //     // Ensure control constraints are satisfied
            //     for (int i = 0; i < control_dim; ++i)
            //     {
            //         current_control(i) = std::max(control_lower_bound(i), std::min(control_upper_bound(i), current_control(i)));
            //     }
            // }
        }
        
        // Propagate system dynamics
        current_state = dyn_system_template->getDiscreteDynamics(current_state, current_control, 0.0);

        // Update history
        state_history.push_back(current_state);
        control_history.push_back(current_control);
        current_time += sim_dt;
        time_history.push_back(current_time);

        // Progress output (every second)
        if (k % static_cast<int>(1.0 / sim_dt) == 0)
        {
            std::cout << "Time: " << current_time 
                      << "s, Position: [" << current_state.head(3).transpose() << "]" 
                      << ", MPC updates: " << k/mpc_update_freq << std::endl;
        }
    }
    std::cout << "Simulation finished." << std::endl;

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
    for(const auto& s : waypoint_trajectory)
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

    // 3D trajectory plot
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

    // Mark start and end points
    auto start_scatter = scatter3(ax1, std::vector<double>{x_hist.front()}, std::vector<double>{y_hist.front()}, std::vector<double>{z_hist.front()});
    start_scatter->marker_color("g").marker_size(100);
    auto end_scatter = scatter3(ax1, std::vector<double>{x_hist.back()}, std::vector<double>{y_hist.back()}, std::vector<double>{z_hist.back()});
    end_scatter->marker_color("r").marker_size(100);

    title(ax1, "3D Trajectory");
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
    save(plotDirectory + "/quadrotor_mpc.png");
    std::cout << "Saved plot to " << plotDirectory << "/quadrotor_mpc.png" << std::endl;

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
        int ref_end = std::min((int)waypoint_trajectory.size(), (int)(i + 200)); // Show future reference
        for (int j = 0; j < ref_end; j += 10) {
            ref_x_show.push_back(waypoint_trajectory[j](0));
            ref_y_show.push_back(waypoint_trajectory[j](1));
            ref_z_show.push_back(waypoint_trajectory[j](2));
        }
        if (!ref_x_show.empty()) {
            auto ref_traj = plot3(ax_anim, ref_x_show, ref_y_show, ref_z_show);
            ref_traj->line_width(1);
            ref_traj->line_style("--");
            ref_traj->color({1.0, 0.6, 0.6}); // Light red
        }

        // Plot target point
        auto target_pt = scatter3(ax_anim, std::vector<double>{target_state(0)}, 
                                std::vector<double>{target_state(1)}, 
                                std::vector<double>{target_state(2)});
        target_pt->marker_color("red").marker_size(100);

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

        title(ax_anim, "Quadrotor MPC Animation - " + time_str);
        xlabel(ax_anim, "X [m]");
        ylabel(ax_anim, "Y [m]");
        zlabel(ax_anim, "Z [m]");
        xlim(ax_anim, {-1, 4});
        ylim(ax_anim, {-2, 2});
        zlim(ax_anim, {0, 3});

        ax_anim->view(30, -45);

        std::string frameFile = plotDirectory + "/quadrotor_mpc_frame_" + std::to_string(frame_count) + ".png";
        f_anim->draw();
        f_anim->save(frameFile);
        
        if (frame_count % 10 == 0) {
            std::cout << "Generated frame " << frame_count << "/" << state_history.size()/frame_stride << std::endl;
        }
        frame_count++;
    }

    // Generate GIF from frames using ImageMagick
    std::cout << "Creating animation..." << std::endl;
    std::string gif_command = "convert -delay 20 -loop 0 " + plotDirectory + "/quadrotor_mpc_frame_*.png " + plotDirectory + "/quadrotor_mpc.gif";
    int result = std::system(gif_command.c_str());
    
    if (result == 0) {
        std::cout << "Animation saved to " << plotDirectory << "/quadrotor_mpc.gif" << std::endl;
        
        // Clean up frame files
        std::string cleanup_command = "rm " + plotDirectory + "/quadrotor_mpc_frame_*.png";
        std::system(cleanup_command.c_str());
    } else {
        std::cout << "Failed to create animation. Frame files are kept in " << plotDirectory << std::endl;
    }

    // Print final statistics
    Eigen::Vector3d final_pos = current_state.head(3);
    Eigen::Vector3d target_pos = waypoint_trajectory.back().head(3);
    double final_error = (final_pos - target_pos).norm();
    std::cout << "Final position error: " << final_error << " m" << std::endl;
    std::cout << "Final state: [" << current_state.head(3).transpose() << "]" << std::endl;
    std::cout << "Target state: [" << target_pos.transpose() << "]" << std::endl;

    return 0;
}