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

// Helper function to create a circular reference trajectory
std::vector<Eigen::VectorXd> create_circular_trajectory(
    double radius, double center_x, double center_y,
    double total_time, double dt)
{
    int num_points = static_cast<int>(total_time / dt) + 1;
    std::vector<Eigen::VectorXd> trajectory;
    trajectory.reserve(num_points);
    double angular_velocity = 2.0 * M_PI / total_time;

    for (int i = 0; i < num_points; ++i)
    {
        double t = i * dt;
        double angle = angular_velocity * t;
        Eigen::VectorXd state(3);
        state(0) = center_x + radius * cos(angle);
        state(1) = center_y + radius * sin(angle);
        state(2) = angle + M_PI / 2.0; // Tangent to the circle
        trajectory.push_back(state);
    }
    return trajectory;
}

int main()
{
    // --------------------------
    // 1. Problem and MPC Setup
    // --------------------------
    const int state_dim = 3;    // [x, y, theta]
    const int control_dim = 2;  // [v, omega]
    const int mpc_horizon = 30; // N in the python notebook
    const double mpc_timestep = 0.1; // dt in the python notebook
    const std::string integration_type = "euler";

    // Simulation parameters
    const double sim_time = 10.0;
    const double sim_dt = 0.1; // controller_dt in python notebook

    // Create a unicycle instance (will be reused)
    auto dyn_system_template = std::make_unique<cddp::Unicycle>(mpc_timestep, integration_type);

    // Quadratic cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q.diagonal() << 30.0, 30.0, 0.1;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    R.diagonal() << 1.0, 0.1;
    Eigen::MatrixXd Qf = 30.0 * Q; // Terminal cost weight from python notebook

    // Create reference trajectory
    auto reference_trajectory = create_circular_trajectory(1.0, 0.0, 0.0, sim_time, sim_dt);

    // Initial state
    Eigen::VectorXd current_state(state_dim);
    current_state << 0.0, -1.0, 0.0;

    // IPDDP Solver Options
    cddp::CDDPOptions options_ipddp;
    options_ipddp.max_iterations = 15;
    options_ipddp.tolerance = 1e-4;
    options_ipddp.verbose = false; // Keep it clean for the MPC loop
    options_ipddp.debug = false;
    options_ipddp.enable_parallel = false; // Can be true for performance
    options_ipddp.num_threads = 1;
    options_ipddp.regularization.initial_value = 1e-4;
    options_ipddp.warm_start = true;

    // Constraint parameters
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.5, M_PI;
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << 0.0, -M_PI;
    
    // Obstacles (from python notebook)
    // We will use the closest one at each step.
    std::vector<Eigen::Vector3d> obstacles;
    obstacles.push_back(Eigen::Vector3d(0.1, 1.3, 0.6));
    obstacles.push_back(Eigen::Vector3d(-0.85, -0.1, 0.2));
    obstacles.push_back(Eigen::Vector3d(-0.7, -0.75, 0.1));

    // Simulation history storage
    std::vector<Eigen::VectorXd> state_history;
    std::vector<Eigen::VectorXd> control_history;
    std::vector<double> time_history;
    state_history.push_back(current_state);
    time_history.push_back(0.0);

    // Initial trajectory guess for the first MPC solve
    std::vector<Eigen::VectorXd> X_guess(mpc_horizon + 1, current_state);
    std::vector<Eigen::VectorXd> U_guess(mpc_horizon, Eigen::VectorXd::Zero(control_dim));

    // --------------------------
    // 2. MPC Loop
    // --------------------------
    std::cout << "Running IPDDP-based MPC for Unicycle Tracking..." << std::endl;
    double current_time = 0.0;
    int sim_steps = static_cast<int>(sim_time / sim_dt);

    for (int k = 0; k < sim_steps; ++k)
    {
        // Get current reference trajectory slice for the MPC horizon
        std::vector<Eigen::VectorXd> mpc_ref_traj;
        int ref_start_idx = k;
        for (int i = 0; i <= mpc_horizon; ++i)
        {
            int idx = std::min(ref_start_idx + i, (int)reference_trajectory.size() - 1);
            mpc_ref_traj.push_back(reference_trajectory[idx]);
        }
        Eigen::VectorXd mpc_goal_state = mpc_ref_traj.back();

        // Create objective for this MPC step
        auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, mpc_goal_state, mpc_ref_traj, mpc_timestep);

        // Create CDDP solver instance for this MPC step
        auto system = std::make_unique<cddp::Unicycle>(mpc_timestep, integration_type);
        cddp::CDDP cddp_solver(current_state, mpc_goal_state, mpc_horizon, mpc_timestep,
                               std::move(system), std::move(objective), options_ipddp);
        
        // Find closest obstacle and add constraint
        Eigen::Vector2d current_pos = current_state.head(2);
        double min_dist = std::numeric_limits<double>::max();
        Eigen::Vector3d closest_obstacle;
        for(const auto& obs : obstacles)
        {
            double dist = (current_pos - obs.head(2)).norm() - obs(2);
            if(dist < min_dist)
            {
                min_dist = dist;
                closest_obstacle = obs;
            }
        }
        
        cddp_solver.addPathConstraint("ControlConstraint",
            std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));
        cddp_solver.addPathConstraint("BallConstraint",
            std::make_unique<cddp::BallConstraint>(closest_obstacle(2), closest_obstacle.head(2)));

        // Set initial trajectory (warm start)
        cddp_solver.setInitialTrajectory(X_guess, U_guess);

        // Solve the OCP
        cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

        // Extract and apply the first control
        auto status = std::any_cast<std::string>(solution.at("status_message"));
        if (status != "OptimalSolutionFound" && status != "AcceptableSolutionFound")
        {
            std::cerr << "Warning: Solver did not converge at step " << k << ". Status: " << status << std::endl;
            // Handle non-convergence, e.g., by applying zero control or previous control
        }

        auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
        Eigen::VectorXd control_to_apply = U_sol[0];
        
        // Propagate system dynamics
        current_state = dyn_system_template->getDiscreteDynamics(current_state, control_to_apply, 0.0);

        // Update history
        state_history.push_back(current_state);
        control_history.push_back(control_to_apply);
        current_time += sim_dt;
        time_history.push_back(current_time);

        // Warm start for the next iteration: shift the solution
        auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
        for(int i = 0; i < mpc_horizon -1; ++i)
        {
            X_guess[i] = X_sol[i+1];
            U_guess[i] = U_sol[i+1];
        }
        X_guess[mpc_horizon -1] = X_sol[mpc_horizon];
        X_guess[mpc_horizon] = X_sol[mpc_horizon]; // Extrapolate or use goal
        U_guess[mpc_horizon -1] = U_sol[mpc_horizon-1]; // Hold last control

        std::cout << "MPC Step: " << k+1 << "/" << sim_steps <<", Time: " << current_time << "s, X: [" << current_state.transpose() << "], U: [" << control_to_apply.transpose() << "]" << std::endl;
    }
    std::cout << "Simulation finished." << std::endl;

    // --------------------------
    // 3. Plotting
    // --------------------------
    // Convert trajectories to plottable vectors
    std::vector<double> x_hist, y_hist, theta_hist;
    for(const auto& s : state_history)
    {
        x_hist.push_back(s(0));
        y_hist.push_back(s(1));
        theta_hist.push_back(s(2));
    }
    std::vector<double> v_hist, omega_hist;
    for(const auto& u : control_history)
    {
        v_hist.push_back(u(0));
        omega_hist.push_back(u(1));
    }
    std::vector<double> x_ref, y_ref;
    for(const auto& s : reference_trajectory)
    {
        x_ref.push_back(s(0));
        y_ref.push_back(s(1));
    }

    auto f = figure(true);
    f->size(1200, 1000);
    f->position(100, 100);

    // Trajectory plot
    auto ax1 = subplot(2, 1, 1);
    plot(ax1, x_hist, y_hist, "b-")->line_width(2).display_name("Actual Trajectory");
    hold(on);
    plot(ax1, x_ref, y_ref, "r--")->line_width(2).display_name("Reference Trajectory");

    // Plot obstacles
    for(const auto& obs : obstacles)
    {
        auto circle = f->draw_circle(obs(0), obs(1), obs(2));
        circle->color("gray");
        circle->face_color("gray");
        circle->face_alpha(0.5);
    }
    
    // Mark start and end
    scatter(ax1, {x_hist.front()}, {y_hist.front()}, 100, "g")->marker_style(line_spec::marker_style::circle).marker_face(true).display_name("Start");
    scatter(ax1, {x_hist.back()}, {y_hist.back()}, 100, "r")->marker_style(line_spec::marker_style::square).marker_face(true).display_name("End");
    
    title(ax1, "Unicycle MPC-CBF Tracking");
    xlabel(ax1, "X [m]");
    ylabel(ax1, "Y [m]");
    legend(ax1, "show");
    grid(ax1, on);
    axis(ax1, "equal");

    // Control plot
    auto ax2 = subplot(2, 1, 2);
    std::vector<double> control_time_hist = time_history;
    control_time_hist.pop_back(); // control history is one step shorter
    plot(ax2, control_time_hist, v_hist, "b-")->line_width(2).display_name("Linear Velocity (v)");
    hold(on);
    plot(ax2, control_time_hist, omega_hist, "g-")->line_width(2).display_name("Angular Velocity (omega)");
    title(ax2, "Control Inputs vs. Time");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Control Value");
    legend(ax2, "show");
    grid(ax2, on);

    // Save and show plot
    const std::string plotDirectory = "../results/examples";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }
    save(plotDirectory + "/unicycle_mpc_cbf.png");
    std::cout << "Saved plot to " << plotDirectory << "/unicycle_mpc_cbf.png" << std::endl;
    show();

    return 0;
} 