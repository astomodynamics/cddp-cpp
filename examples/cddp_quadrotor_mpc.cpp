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

    // Storage for first MPC solution
    std::vector<Eigen::VectorXd> first_mpc_states;
    std::vector<Eigen::VectorXd> first_mpc_controls;
    bool first_solution_saved = false;

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
        
            // Save first MPC solution for analysis
            if (!first_solution_saved && k == 0)
            {
                first_mpc_states = current_mpc_states;
                first_mpc_controls = current_mpc_controls;
                first_solution_saved = true;
                
                // Print first solution analysis
                std::cout << "\n=== First MPC Solution Analysis ===" << std::endl;
            std::cout << "Initial state: [" << current_state.head(3).transpose() << "]" << std::endl;
            std::cout << "Target state: [" << target_state.head(3).transpose() << "]" << std::endl;
            std::cout << "MPC horizon: " << mpc_horizon << " steps, dt: " << mpc_timestep << "s" << std::endl;
            std::cout << "Total prediction time: " << mpc_horizon * mpc_timestep << "s" << std::endl;
            
            // Analyze trajectory feasibility
            std::cout << "\nTrajectory analysis:" << std::endl;
            for (int i = 0; i <= mpc_horizon; i += 5)
            {
                Eigen::Vector3d pos = current_mpc_states[i].head(3);
                Eigen::Vector3d vel = current_mpc_states[i].segment(7, 3);
                std::cout << "t=" << i * mpc_timestep << "s: pos=[" << pos.transpose() 
                          << "], vel=[" << vel.transpose() << "], |vel|=" << vel.norm() << "m/s" << std::endl;
            }
            
            // Check control effort
            std::cout << "\nControl analysis:" << std::endl;
            double total_thrust_avg = 0.0;
            double max_thrust = 0.0;
            for (int i = 0; i < mpc_horizon; ++i)
            {
                double total_thrust = current_mpc_controls[i].sum();
                total_thrust_avg += total_thrust;
                max_thrust = std::max(max_thrust, total_thrust);
                if (i % 5 == 0)
                {
                    std::cout << "t=" << i * mpc_timestep << "s: controls=[" << current_mpc_controls[i].transpose() 
                              << "], total=" << total_thrust << "N" << std::endl;
                }
            }
            total_thrust_avg /= mpc_horizon;
            std::cout << "Average total thrust: " << total_thrust_avg << "N (hover: " << mass * 9.81 << "N)" << std::endl;
            std::cout << "Max total thrust: " << max_thrust << "N" << std::endl;
            
            // Final state analysis
            Eigen::Vector3d final_pos = current_mpc_states.back().head(3);
            double final_error = (final_pos - target_state.head(3)).norm();
            std::cout << "\nPredicted final position: [" << final_pos.transpose() << "]" << std::endl;
            std::cout << "Predicted final error: " << final_error << "m" << std::endl;
            std::cout << "================================\n" << std::endl;
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

    // Create separate plot for first MPC solution
    if (first_solution_saved)
    {
        auto f2 = figure(true);
        f2->size(1200, 800);
        
        // Extract first MPC trajectory data
        std::vector<double> first_x, first_y, first_z;
        std::vector<double> first_time;
        for (int i = 0; i <= mpc_horizon; ++i)
        {
            first_x.push_back(first_mpc_states[i](0));
            first_y.push_back(first_mpc_states[i](1));
            first_z.push_back(first_mpc_states[i](2));
            first_time.push_back(i * mpc_timestep);
        }
        
        // 3D trajectory
        auto ax2_1 = subplot(2, 2, 1);
        auto first_traj = plot3(ax2_1, first_x, first_y, first_z);
        first_traj->line_width(3);
        first_traj->color("blue");
        first_traj->display_name("First MPC Prediction");
        
        hold(ax2_1, true);
        // Add markers at key points
        auto start_pt = scatter3(ax2_1, std::vector<double>{first_x.front()}, 
                                std::vector<double>{first_y.front()}, 
                                std::vector<double>{first_z.front()});
        start_pt->marker_color("g").marker_size(150).display_name("Start");
        
        auto end_pt = scatter3(ax2_1, std::vector<double>{first_x.back()}, 
                              std::vector<double>{first_y.back()}, 
                              std::vector<double>{first_z.back()});
        end_pt->marker_color("r").marker_size(150).display_name("Predicted End");
        
        auto target_pt = scatter3(ax2_1, std::vector<double>{target_state(0)}, 
                                 std::vector<double>{target_state(1)}, 
                                 std::vector<double>{target_state(2)});
        target_pt->marker_color("k").marker_size(150).display_name("Target");
        
        title(ax2_1, "First MPC Predicted Trajectory (3D)");
        xlabel(ax2_1, "X [m]");
        ylabel(ax2_1, "Y [m]");
        zlabel(ax2_1, "Z [m]");
        legend(ax2_1, "show");
        grid(ax2_1, true);
        
        // Position components vs time
        auto ax2_2 = subplot(2, 2, 2);
        plot(ax2_2, first_time, first_x, "r-")->line_width(2).display_name("x");
        hold(ax2_2, true);
        plot(ax2_2, first_time, first_y, "g-")->line_width(2).display_name("y");
        plot(ax2_2, first_time, first_z, "b-")->line_width(2).display_name("z");
        // Add target lines
        plot(ax2_2, first_time, std::vector<double>(first_time.size(), target_state(0)), "r--")->line_width(1);
        plot(ax2_2, first_time, std::vector<double>(first_time.size(), target_state(1)), "g--")->line_width(1);
        plot(ax2_2, first_time, std::vector<double>(first_time.size(), target_state(2)), "b--")->line_width(1);
        title(ax2_2, "First MPC Position Prediction");
        xlabel(ax2_2, "Time [s]");
        ylabel(ax2_2, "Position [m]");
        legend(ax2_2, "show");
        grid(ax2_2, true);
        
        // Velocity profile
        auto ax2_3 = subplot(2, 2, 3);
        std::vector<double> first_vx, first_vy, first_vz, first_v_norm;
        for (const auto& state : first_mpc_states)
        {
            Eigen::Vector3d vel = state.segment(7, 3);
            first_vx.push_back(vel(0));
            first_vy.push_back(vel(1));
            first_vz.push_back(vel(2));
            first_v_norm.push_back(vel.norm());
        }
        plot(ax2_3, first_time, first_vx, "r-")->line_width(2).display_name("vx");
        hold(ax2_3, true);
        plot(ax2_3, first_time, first_vy, "g-")->line_width(2).display_name("vy");
        plot(ax2_3, first_time, first_vz, "b-")->line_width(2).display_name("vz");
        plot(ax2_3, first_time, first_v_norm, "k-")->line_width(3).display_name("|v|");
        title(ax2_3, "First MPC Velocity Prediction");
        xlabel(ax2_3, "Time [s]");
        ylabel(ax2_3, "Velocity [m/s]");
        legend(ax2_3, "show");
        grid(ax2_3, true);
        
        // Control inputs
        auto ax2_4 = subplot(2, 2, 4);
        std::vector<double> first_f1, first_f2, first_f3, first_f4, first_total;
        std::vector<double> first_control_time;
        for (int i = 0; i < mpc_horizon; ++i)
        {
            first_f1.push_back(first_mpc_controls[i](0));
            first_f2.push_back(first_mpc_controls[i](1));
            first_f3.push_back(first_mpc_controls[i](2));
            first_f4.push_back(first_mpc_controls[i](3));
            first_total.push_back(first_mpc_controls[i].sum());
            first_control_time.push_back(i * mpc_timestep);
        }
        plot(ax2_4, first_control_time, first_f1, "r-")->line_width(2).display_name("f1");
        hold(ax2_4, true);
        plot(ax2_4, first_control_time, first_f2, "g-")->line_width(2).display_name("f2");
        plot(ax2_4, first_control_time, first_f3, "b-")->line_width(2).display_name("f3");
        plot(ax2_4, first_control_time, first_f4, "m-")->line_width(2).display_name("f4");
        plot(ax2_4, first_control_time, first_total, "k-")->line_width(3).display_name("Total");
        // Add hover thrust line
        plot(ax2_4, first_control_time, std::vector<double>(first_control_time.size(), mass * 9.81), "k--")
            ->line_width(2).display_name("Hover");
        title(ax2_4, "First MPC Control Prediction");
        xlabel(ax2_4, "Time [s]");
        ylabel(ax2_4, "Force [N]");
        legend(ax2_4, "show");
        grid(ax2_4, true);
        
        save(plotDirectory + "/quadrotor_first_mpc_solution.png");
        std::cout << "Saved first MPC solution plot to " << plotDirectory << "/quadrotor_first_mpc_solution.png" << std::endl;
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