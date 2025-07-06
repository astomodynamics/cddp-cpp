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

int main() {
    // --------------------------
    // 1. Problem setup
    // --------------------------
    int state_dim = 3;    // [x, y, theta]
    int control_dim = 2;  // [v, omega]
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle dynamical system instance
    std::unique_ptr<cddp::DynamicalSystem> dyn_system =
        std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Objective weighting matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0,  0.0,   0.0,
           0.0, 100.0,   0.0,
           0.0,   0.0, 100.0;

    // Goal state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Create an empty reference state vector
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;

    // CDDP options
    cddp::CDDPOptions options;
    options.max_iterations = 100;
    options.verbose = true;
    options.debug = false;
    options.enable_parallel = false;
    options.num_threads = 1;
    options.tolerance = 1e-4;
    options.acceptable_tolerance = 1e-3;
    options.regularization.initial_value = 1e-2;
    options.ipddp.barrier.mu_initial = 1e-1;

    // Define control constraint
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.1, M_PI; // clamp velocity and steering

    // Ball constraint parameters
    double radius = 0.4;
    Eigen::Vector2d center(1.0, 1.0);

    // Create a directory for saving plots (if it doesn't exist)
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // --------------------------
    // 2. Solve - NO Ball constraint
    // --------------------------
    cddp::CDDP solver_baseline(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    // Solver with new API already set up with system and objective

    // Add a control constraint
    solver_baseline.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Naive initial trajectory
    std::vector<Eigen::VectorXd> X_baseline_init(horizon + 1, initial_state);
    std::vector<Eigen::VectorXd> U_baseline_init(horizon, Eigen::VectorXd::Zero(control_dim));
    solver_baseline.setInitialTrajectory(X_baseline_init, U_baseline_init);

    // Solve
    cddp::CDDPSolution solution_baseline = solver_baseline.solve(cddp::SolverType::MSIPDDP);
    auto X_baseline_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution_baseline.at("state_trajectory"));   // horizon+1
    auto U_baseline_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution_baseline.at("control_trajectory")); // horizon
    auto T_baseline_sol = std::any_cast<std::vector<double>>(solution_baseline.at("time_points"));    // horizon+1

    // --------------------------
    // 3. Solve - WITH Ball constraint (naive init)
    // --------------------------
    cddp::CDDP solver_ball(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    // Solver with new API already set up with system and objective

    // Add constraints
    solver_ball.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_ball.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));

    // Naive initial trajectory
    std::vector<Eigen::VectorXd> X_ball_init(horizon + 1, initial_state);
    std::vector<Eigen::VectorXd> U_ball_init(horizon, Eigen::VectorXd::Zero(control_dim));
    solver_ball.setInitialTrajectory(X_ball_init, U_ball_init);

    // Solve
    cddp::CDDPSolution solution_ball = solver_ball.solve(cddp::SolverType::MSIPDDP);
    auto X_ball_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution_ball.at("state_trajectory"));
    auto U_ball_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution_ball.at("control_trajectory"));
    auto T_ball_sol = std::any_cast<std::vector<double>>(solution_ball.at("time_points"));

    // --------------------------
    // 4. Solve - WITH Ball constraint (baseline init)
    // --------------------------
    cddp::CDDP solver_ball_with_baseline(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    // Solver with new API already set up with system and objective

    // Add constraints
    solver_ball_with_baseline.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_ball_with_baseline.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));

    // Use baseline solution as initialization
    solver_ball_with_baseline.setInitialTrajectory(X_baseline_sol, U_baseline_sol);

    // Solve
    cddp::CDDPSolution solution_ball_with_baseline = solver_ball_with_baseline.solve(cddp::SolverType::IPDDP);
    auto X_ball_with_baseline_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution_ball_with_baseline.at("state_trajectory"));
    auto U_ball_with_baseline_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution_ball_with_baseline.at("control_trajectory"));
    auto T_ball_with_baseline_sol = std::any_cast<std::vector<double>>(solution_ball_with_baseline.at("time_points"));

    // --------------------------
    // 5. Convert solutions to std::vectors for plotting
    //    We'll do: X vs Y, Theta vs time, Controls vs step
    // --------------------------
    // Baseline
    std::vector<double> x_base, y_base, theta_base, t_base;
    std::vector<double> v_base, omega_base;
    for (size_t i = 0; i < X_baseline_sol.size(); ++i) {
        x_base.push_back(X_baseline_sol[i](0));
        y_base.push_back(X_baseline_sol[i](1));
        theta_base.push_back(X_baseline_sol[i](2));
        if (i < T_baseline_sol.size()) {
            t_base.push_back(T_baseline_sol[i]);
        }
    }
    for (size_t i = 0; i < U_baseline_sol.size(); ++i) {
        v_base.push_back(U_baseline_sol[i](0));
        omega_base.push_back(U_baseline_sol[i](1));
    }

    // Ball (naive init)
    std::vector<double> x_ball, y_ball, theta_ball, t_ball;
    std::vector<double> v_ball, omega_ball;
    for (size_t i = 0; i < X_ball_sol.size(); ++i) {
        x_ball.push_back(X_ball_sol[i](0));
        y_ball.push_back(X_ball_sol[i](1));
        theta_ball.push_back(X_ball_sol[i](2));
        if (i < T_ball_sol.size()) {
            t_ball.push_back(T_ball_sol[i]);
        }
    }
    for (size_t i = 0; i < U_ball_sol.size(); ++i) {
        v_ball.push_back(U_ball_sol[i](0));
        omega_ball.push_back(U_ball_sol[i](1));
    }

    // Ball with baseline init
    std::vector<double> x_ball_bl, y_ball_bl, theta_ball_bl, t_ball_bl;
    std::vector<double> v_ball_bl, omega_ball_bl;
    for (size_t i = 0; i < X_ball_with_baseline_sol.size(); ++i) {
        x_ball_bl.push_back(X_ball_with_baseline_sol[i](0));
        y_ball_bl.push_back(X_ball_with_baseline_sol[i](1));
        theta_ball_bl.push_back(X_ball_with_baseline_sol[i](2));
        if (i < T_ball_with_baseline_sol.size()) {
            t_ball_bl.push_back(T_ball_with_baseline_sol[i]);
        }
    }
    for (size_t i = 0; i < U_ball_with_baseline_sol.size(); ++i) {
        v_ball_bl.push_back(U_ball_with_baseline_sol[i](0));
        omega_ball_bl.push_back(U_ball_with_baseline_sol[i](1));
    }

    // --------------------------
    // 6. Plot: Subplots for the three solutions
    // --------------------------
    auto f1 = figure(true);
    f1->size(1200, 800);

    // Subplot 1: XY Trajectories
    auto ax1 = subplot(3, 1, 0);
    plot(ax1, x_base, y_base, "-b")->display_name("No Ball");
    hold(ax1, true);
    plot(ax1, x_ball, y_ball, "-r")->display_name("Ball (naive init)");
    plot(ax1, x_ball_bl, y_ball_bl, "-m")->display_name("Ball (baseline init)");

    // Also plot the circular region for the ball constraint
    std::vector<double> circle_x, circle_y;
    for (double th = 0.0; th <= 2.0 * M_PI; th += 0.01) {
        circle_x.push_back(center(0) + radius * cos(th));
        circle_y.push_back(center(1) + radius * sin(th));
    }
    plot(ax1, circle_x, circle_y, "--g")->display_name("Ball Constraint");

    title(ax1, "Position Trajectory (X-Y plane)");
    xlabel(ax1, "x [m]");
    ylabel(ax1, "y [m]");
    matplot::legend(ax1);

    // Subplot 2: Heading Angle vs Time
    auto ax2 = subplot(3, 1, 1);
    plot(ax2, t_base, theta_base, "-b")->display_name("No Ball");
    hold(ax2, true);
    plot(ax2, t_ball, theta_ball, "-r")->display_name("Ball (naive init)");
    plot(ax2, t_ball_bl, theta_ball_bl, "-m")->display_name("Ball (baseline init)");

    title(ax2, "Heading Angle vs Time");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Theta [rad]");
    matplot::legend(ax2);

    // Subplot 3: Control Inputs vs Step
    auto ax3 = subplot(3, 1, 2);
    // We'll overlay v and omega in the same subplot for each solution
    // "No Ball" in blue
    auto p1 = plot(ax3, v_base, "-b");
    p1->display_name("v - No Ball");
    hold(ax3, true);
    auto p2 = plot(ax3, omega_base, "--b");
    p2->display_name("omega - No Ball");

    // "Ball naive" in red
    auto p3 = plot(ax3, v_ball, "-r");
    p3->display_name("v - Ball naive");
    auto p4 = plot(ax3, omega_ball, "--r");
    p4->display_name("omega - Ball naive");

    // "Ball baseline init" in magenta
    auto p5 = plot(ax3, v_ball_bl, "-m");
    p5->display_name("v - Ball baseline");
    auto p6 = plot(ax3, omega_ball_bl, "--m");
    p6->display_name("omega - Ball baseline");

    title(ax3, "Control Inputs vs. Step");
    xlabel(ax3, "Step");
    ylabel(ax3, "Control value");
    matplot::legend(ax3);

    // Save the figure
    f1->draw();
    f1->save(plotDirectory + "/trajectory_comparison_ipddp_matplot.png");

    std::cout << "Saved comparison to " << (plotDirectory + "/trajectory_comparison_ipddp_matplot.png") << std::endl;

    // --------------------------
    // 7. (Optional) Animation for the final solution
    //    For demonstration, let's animate the "Ball (baseline init)" solution.
    // --------------------------
    auto f2 = figure(true);
    f2->size(800, 600);
    auto ax_anim = f2->current_axes();
    if (!ax_anim) {
        ax_anim = axes();
    }

    double unicycle_length = 0.35;
    double unicycle_width = 0.15;

    // We'll sample frames every few steps to avoid too many images
    for (size_t i = 0; i < X_ball_with_baseline_sol.size(); ++i) {
        if (i % 5 == 0) {
            ax_anim->clear();
            hold(ax_anim, true);

            double x = x_ball_bl[i];
            double y = y_ball_bl[i];
            double theta = theta_ball_bl[i];

            // Compute corners of a rectangle representing the unicycle
            std::vector<double> car_x(5), car_y(5);
            car_x[0] = x + unicycle_length / 2.0 * cos(theta)
                         - unicycle_width / 2.0 * sin(theta);
            car_y[0] = y + unicycle_length / 2.0 * sin(theta)
                         + unicycle_width / 2.0 * cos(theta);

            car_x[1] = x + unicycle_length / 2.0 * cos(theta)
                         + unicycle_width / 2.0 * sin(theta);
            car_y[1] = y + unicycle_length / 2.0 * sin(theta)
                         - unicycle_width / 2.0 * cos(theta);

            car_x[2] = x - unicycle_length / 2.0 * cos(theta)
                         + unicycle_width / 2.0 * sin(theta);
            car_y[2] = y - unicycle_length / 2.0 * sin(theta)
                         - unicycle_width / 2.0 * cos(theta);

            car_x[3] = x - unicycle_length / 2.0 * cos(theta)
                         - unicycle_width / 2.0 * sin(theta);
            car_y[3] = y - unicycle_length / 2.0 * sin(theta)
                         + unicycle_width / 2.0 * cos(theta);

            car_x[4] = car_x[0];
            car_y[4] = car_y[0];

            // Draw the unicycle rectangle
            auto car_line = plot(ax_anim, car_x, car_y);
            car_line->line_width(2);

            // Plot trajectory up to current frame
            std::vector<double> traj_x(x_ball_bl.begin(), x_ball_bl.begin() + i + 1);
            std::vector<double> traj_y(y_ball_bl.begin(), y_ball_bl.begin() + i + 1);
            auto traj_line = plot(ax_anim, traj_x, traj_y);
            traj_line->line_width(1.5);

            // Also show the ball constraint
            std::vector<double> circle_x_anim, circle_y_anim;
            for (double th = 0.0; th <= 2.0 * M_PI; th += 0.01) {
                circle_x_anim.push_back(center(0) + radius * cos(th));
                circle_y_anim.push_back(center(1) + radius * sin(th));
            }
            plot(ax_anim, circle_x_anim, circle_y_anim, "--g");

            title(ax_anim, "Unicycle Trajectory (Ball Constraint, baseline init)");
            xlabel(ax_anim, "x [m]");
            ylabel(ax_anim, "y [m]");
            xlim(ax_anim, {-0.5, 2.5});
            ylim(ax_anim, {-0.5, 2.5});

            // Save each frame
            std::string filename = plotDirectory + "/unicycle_safe_frame_" + std::to_string(i) + ".png";
            f2->draw();
            f2->save(filename);

            // Small pause so we can see the frames
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // -----------------------------
    // 8. Generate GIF from frames using ImageMagick (if installed)
    // -----------------------------
    {
        std::string gif_command =
            "convert -delay 8 " + plotDirectory + "/unicycle_safe_frame_*.png " +
            plotDirectory + "/unicycle_safe.gif";
        std::system(gif_command.c_str());

        // Cleanup frames if you like
        std::string cleanup_command =
            "rm " + plotDirectory + "/unicycle_safe_frame_*.png";
        std::system(cleanup_command.c_str());
    }

    return 0;
}
