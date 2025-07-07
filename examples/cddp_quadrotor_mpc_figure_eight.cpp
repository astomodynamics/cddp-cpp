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

// Quadrotor Model Predictive Control (MPC) Figure-8 Example
// This example demonstrates:
// - Multi-rate MPC: Control at 10Hz, simulation at 100Hz
// - Figure-8 reference trajectory tracking
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
    const double sim_time = 12.0;  // Same as working example
    const double sim_dt = 0.01;    // Simulation timestep (100 Hz)
    const double mpc_dt = 0.1;     // MPC update rate (10 Hz)
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

    // Figure-8 trajectory parameters
    const double figure8_scale = 3.0;     // 3m radius
    const double constant_altitude = 2.0; // 2m altitude
    const double figure8_period = 8.0;    // Complete figure-8 in 8 seconds
    const double omega = 2.0 * M_PI / figure8_period;

    // Generate figure-8 reference trajectory with higher resolution for MPC
    int total_sim_steps = static_cast<int>(sim_time / sim_dt);
    std::vector<Eigen::VectorXd> figure8_reference_trajectory;
    
    // Generate reference points at MPC timestep resolution for better alignment
    int total_mpc_steps = static_cast<int>((sim_time + mpc_horizon * mpc_timestep) / mpc_timestep);
    for (int i = 0; i <= total_mpc_steps; ++i)
    {
        double t = i * mpc_timestep;
        double angle = omega * t;
        
        // Lemniscate of Gerono for (x, y)
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);
        ref_state(0) = figure8_scale * std::cos(angle);
        ref_state(1) = figure8_scale * std::sin(angle) * std::cos(angle);
        ref_state(2) = constant_altitude;
        
        // Identity quaternion
        ref_state(3) = 1.0; // qw
        ref_state(4) = 0.0; // qx
        ref_state(5) = 0.0; // qy
        ref_state(6) = 0.0; // qz
        
        // Compute reference velocities for better tracking
        double vx = -figure8_scale * omega * std::sin(angle);
        double vy = figure8_scale * omega * (std::cos(2.0 * angle));
        ref_state(7) = vx;
        ref_state(8) = vy;
        ref_state(9) = 0.0; // vz = 0 for constant altitude
        
        figure8_reference_trajectory.push_back(ref_state);
    }

    // Initial state (start of figure-8)
    Eigen::VectorXd current_state = Eigen::VectorXd::Zero(state_dim);
    current_state(0) = figure8_scale;
    current_state(2) = constant_altitude;
    current_state(3) = 1.0; // Identity quaternion: qw = 1

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
    std::cout << "Running IPDDP-based MPC for Quadrotor Figure-8 Trajectory..." << std::endl;
    std::cout << "Simulation: " << 1.0/sim_dt << " Hz, MPC: " << 1.0/mpc_dt << " Hz" << std::endl;
    std::cout << "Figure-8 parameters: scale = " << figure8_scale << " m, altitude = " << constant_altitude << " m, period = " << figure8_period << " s" << std::endl;
    
    double current_time = 0.0;
    int sim_steps = static_cast<int>(sim_time / sim_dt);

    // Storage for current MPC solution
    std::vector<Eigen::VectorXd> current_mpc_states;
    std::vector<Eigen::VectorXd> current_mpc_controls;
    Eigen::VectorXd current_control = hover_thrust * Eigen::VectorXd::Ones(control_dim);

    for (int k = 0; k < sim_steps; ++k)
    {
        // Only update MPC at specified frequency
        if (k % mpc_update_freq == 0)
        {
            // Get current reference trajectory slice for the MPC horizon
            std::vector<Eigen::VectorXd> mpc_ref_traj;
            int mpc_step = k / mpc_update_freq;
            for (int i = 0; i <= mpc_horizon; ++i)
            {
                int idx = std::min(mpc_step + i, (int)figure8_reference_trajectory.size() - 1);
                mpc_ref_traj.push_back(figure8_reference_trajectory[idx]);
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
        int steps_per_mpc_interval = static_cast<int>(mpc_timestep / sim_dt);
        int mpc_idx = (k % mpc_update_freq) / steps_per_mpc_interval;
        
        if (!current_mpc_controls.empty() && mpc_idx < current_mpc_controls.size())
        {
            current_control = current_mpc_controls[mpc_idx];
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
            // Calculate tracking error using the correct reference index
            int mpc_step = k / mpc_update_freq;
            double ref_time = mpc_step * mpc_timestep;
            double ref_angle = omega * ref_time;
            Eigen::Vector3d ref_pos;
            ref_pos(0) = figure8_scale * std::cos(ref_angle);
            ref_pos(1) = figure8_scale * std::sin(ref_angle) * std::cos(ref_angle);
            ref_pos(2) = constant_altitude;
            
            Eigen::Vector3d current_pos = current_state.head(3);
            double tracking_error = (current_pos - ref_pos).norm();
            
            std::cout << "Time: " << current_time 
                      << "s, Position: [" << current_state.head(3).transpose() << "]"
                      << ", Tracking error: " << tracking_error << " m"
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
    for(size_t i = 0; i < state_history.size(); ++i)
    {
        double t = i * sim_dt;
        double angle = omega * t;
        x_ref.push_back(figure8_scale * std::cos(angle));
        y_ref.push_back(figure8_scale * std::sin(angle) * std::cos(angle));
        z_ref.push_back(constant_altitude);
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

    title(ax1, "3D Figure-8 Trajectory");
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
    
    // Plot reference as dashed lines
    plot(ax2, time_history, x_ref, "r--")->line_width(1).display_name("x_ref");
    plot(ax2, time_history, y_ref, "g--")->line_width(1).display_name("y_ref");
    plot(ax2, time_history, z_ref, "b--")->line_width(1).display_name("z_ref");
    
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
    save(plotDirectory + "/quadrotor_mpc_figure_eight.png");
    std::cout << "Saved plot to " << plotDirectory << "/quadrotor_mpc_figure_eight.png" << std::endl;

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

        // Plot full reference trajectory (in light red dashed)
        std::vector<double> ref_x_show, ref_y_show, ref_z_show;
        int n_ref_points = 100;
        for (int j = 0; j < n_ref_points; ++j) {
            double t = j * sim_time / n_ref_points;
            double angle = omega * t;
            ref_x_show.push_back(figure8_scale * std::cos(angle));
            ref_y_show.push_back(figure8_scale * std::sin(angle) * std::cos(angle));
            ref_z_show.push_back(constant_altitude);
        }
        if (!ref_x_show.empty()) {
            auto ref_traj = plot3(ax_anim, ref_x_show, ref_y_show, ref_z_show);
            ref_traj->line_width(1);
            ref_traj->line_style("--");
            ref_traj->color({1.0, 0.6, 0.6}); // Light red
        }

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

        title(ax_anim, "Quadrotor MPC Figure-8 Animation - " + time_str);
        xlabel(ax_anim, "X [m]");
        ylabel(ax_anim, "Y [m]");
        zlabel(ax_anim, "Z [m]");
        xlim(ax_anim, {-3, 3});
        ylim(ax_anim, {-2, 2});
        zlim(ax_anim, {0, 4});

        ax_anim->view(30, -45);

        std::string frameFile = plotDirectory + "/quadrotor_mpc_figure8_frame_" + std::to_string(frame_count) + ".png";
        f_anim->draw();
        f_anim->save(frameFile);
        
        if (frame_count % 10 == 0) {
            std::cout << "Generated frame " << frame_count << "/" << state_history.size()/frame_stride << std::endl;
        }
        frame_count++;
    }

    // Generate GIF from frames using ImageMagick
    std::cout << "Creating animation..." << std::endl;
    std::string gif_command = "convert -delay 20 -loop 0 " + plotDirectory + "/quadrotor_mpc_figure8_frame_*.png " + plotDirectory + "/quadrotor_mpc_figure_eight.gif";
    int result = std::system(gif_command.c_str());
    
    if (result == 0) {
        std::cout << "Animation saved to " << plotDirectory << "/quadrotor_mpc_figure_eight.gif" << std::endl;
        
        // Clean up frame files
        std::string cleanup_command = "rm " + plotDirectory + "/quadrotor_mpc_figure8_frame_*.png";
        std::system(cleanup_command.c_str());
    } else {
        std::cout << "Failed to create animation. Frame files are kept in " << plotDirectory << std::endl;
    }

    // Print final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "=================" << std::endl;
    
    // Calculate average tracking error
    double total_tracking_error = 0.0;
    double max_tracking_error = 0.0;
    for (size_t i = 0; i < state_history.size(); ++i)
    {
        double t = i * sim_dt;
        double angle = omega * t;
        Eigen::Vector3d ref_pos;
        ref_pos(0) = figure8_scale * std::cos(angle);
        ref_pos(1) = figure8_scale * std::sin(angle) * std::cos(angle);
        ref_pos(2) = constant_altitude;
        
        Eigen::Vector3d actual_pos = state_history[i].head(3);
        double error = (actual_pos - ref_pos).norm();
        total_tracking_error += error;
        max_tracking_error = std::max(max_tracking_error, error);
    }
    double avg_tracking_error = total_tracking_error / state_history.size();
    
    std::cout << "Average tracking error: " << avg_tracking_error << " m" << std::endl;
    std::cout << "Maximum tracking error: " << max_tracking_error << " m" << std::endl;
    
    // Check control bounds
    double min_control = std::numeric_limits<double>::max();
    double max_control = std::numeric_limits<double>::min();
    int control_violations = 0;
    for (const auto& u : control_history) {
        for (int i = 0; i < control_dim; ++i) {
            min_control = std::min(min_control, u(i));
            max_control = std::max(max_control, u(i));
            if (u(i) < min_force - 1e-6 || u(i) > max_force + 1e-6) {
                control_violations++;
            }
        }
    }
    
    std::cout << "\nControl bounds:" << std::endl;
    std::cout << "  Min control applied: " << min_control << " N (limit: " << min_force << " N)" << std::endl;
    std::cout << "  Max control applied: " << max_control << " N (limit: " << max_force << " N)" << std::endl;
    std::cout << "  Control violations: " << control_violations << " / " << control_history.size() * control_dim << " total control values";
    if (control_violations > 0) {
        std::cout << " [FAILED]";
    } else {
        std::cout << " [PASSED]";
    }
    std::cout << std::endl;

    // Final position
    Eigen::Vector3d final_pos = state_history.back().head(3);
    std::cout << "\nFinal position: [" << final_pos.transpose() << "]" << std::endl;

    return 0;
}