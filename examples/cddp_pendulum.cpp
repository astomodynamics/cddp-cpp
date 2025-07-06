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
#include <random>
#include <string>
#include <chrono>
#include <thread>
#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

int main() {
    // -------------------- Problem Setup --------------------
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 100;
    double timestep = 0.02;
    
    // Create a pendulum instance
    double mass = 1.0;
    double length = 0.5;
    double damping = 0.01;
    std::string integration_type = "euler";
    
    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0, 0.0,
          0.0, 100.0;
    
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0;  // Upright position with zero velocity
    
    std::vector<Eigen::VectorXd> empty_reference_states;
    
    // Initial state (pendulum pointing down)
    Eigen::VectorXd initial_state(state_dim);
    initial_state << M_PI, 0.0;
    
    // Construct a zero control sequence and an initial trajectory (all at the initial state)
    std::vector<Eigen::VectorXd> zero_control_sequence(horizon, Eigen::VectorXd::Zero(control_dim));
    std::vector<Eigen::VectorXd> X_init(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    for (int t = 0; t < horizon + 1; ++t) {
        X_init[t] = initial_state;
    }
    
    // (Optional) Calculate initial cost
    auto temp_objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);
    double J = 0.0;
    for (int t = 0; t < horizon; ++t) {
        J += temp_objective->running_cost(X_init[t], zero_control_sequence[t], t);
    }
    J += temp_objective->terminal_cost(X_init[horizon]);
    
    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -100.0;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 100.0;
    
    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 20;
    options.tolerance = 1e-4;
    options.acceptable_tolerance = 1e-5;
    options.regularization.initial_value = 1e-7;
    
    // Create and configure the CDDP solver with new API
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Pendulum>(timestep, length, mass, damping, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    
    cddp_solver.addPathConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    
    // Set initial trajectory for the solver
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon + 1; ++i) {
        X[i] = initial_state;
    }
    cddp_solver.setInitialTrajectory(X, U);
    
    // Solve the optimal control problem
    cddp::CDDPSolution solution = cddp_solver.solve(cddp::SolverType::IPDDP);
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    
    // Create plot directory if it doesn't exist
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }
    
    // -------------------- Data Extraction --------------------
    std::vector<double> theta_arr, theta_dot_arr, torque_arr;
    for (const auto& x : X_sol) {
        theta_arr.push_back(x(0));
        theta_dot_arr.push_back(x(1));
    }
    for (const auto& u : U_sol) {
        torque_arr.push_back(u(0));
    }
    
    // Build time vectors (state has horizon+1 points; control has horizon points)
    std::vector<double> time_state, time_control;
    for (size_t i = 0; i < theta_arr.size(); ++i)
        time_state.push_back(i * timestep);
    for (size_t i = 0; i < torque_arr.size(); ++i)
        time_control.push_back(i * timestep);
    
    // -------------------- Static Plot --------------------
    auto fig1 = figure(true);
    fig1->size(1200, 800);
    
    // Subplot for state trajectory (angle and angular velocity)
    auto ax1 = subplot(2, 1, 0);
    {
        ax1->hold(true);
        auto plot_handle1 = plot(ax1, time_state, theta_arr, "-o");
        plot_handle1->display_name("Angle");
        plot_handle1->line_width(2);
        plot_handle1->color("b");
        
        auto plot_handle2 = plot(ax1, time_state, theta_dot_arr, "-o");
        plot_handle2->display_name("Angular Velocity");
        plot_handle2->line_width(2);
        plot_handle2->color("r");
        ax1->title("Pendulum State Trajectory");
        ax1->xlabel("Time [s]");
        ax1->ylabel("Value");
        ax1->legend();
        ax1->grid(true);
    
    }
    // Subplot for control input (torque)
    auto ax2 = subplot(2, 1, 1);
    {
        ax2->hold(true);
        auto plot_handle3 = plot(ax2, time_control, torque_arr, "-o");
        plot_handle3->display_name("Torque");
        plot_handle3->line_width(2);
        plot_handle3->color("g");
        
        // Customize the plot
        ax2->title("Control Input");
        ax2->xlabel("Time [s]");
        ax2->ylabel("Torque [Nm]");
        ax2->legend();
        ax2->grid(true);
    }
    
    // Adjust layout if supported
    // tight_layout();  // Uncomment if your matplot version supports it
    save(fig1, plotDirectory + "/pendulum_cddp_test.png");

       // --- Animation ---
   auto fig3 = figure();
   auto ax_fig3 = fig3->current_axes();
   fig3->size(800, 600);
   title("Pendulum Animation");
   double pendulum_length = length;
   double bob_radius = 0.1;


    for (int i = 0; i < X_sol.size(); ++i) {
        // Animate every 5th frame
        if (i % 5 == 0) {
            // Clear previous content.
            cla(ax_fig3);
            hold(ax_fig3, true);
            
            double theta = theta_arr[i];
            // Calculate pendulum bob position (x, y)
            double x = pendulum_length * std::sin(theta);
            double y = pendulum_length * std::cos(theta);
            
            // Plot pendulum rod
            std::vector<double> rod_x = {0, x};
            std::vector<double> rod_y = {0, y};
            auto rod_plot = plot(rod_x, rod_y);
            rod_plot->line_style("k-");
            rod_plot->line_width(2);

            
            // Plot pendulum bob as a circle
            std::vector<double> circle_x, circle_y;
            int num_points = 50;
            for (int j = 0; j <= num_points; ++j) {
                double t = 2 * M_PI * j / num_points;
                circle_x.push_back(x + bob_radius * std::cos(t));
                circle_y.push_back(y + bob_radius * std::sin(t));
            }
            auto circle_plot = plot(circle_x, circle_y);
            circle_plot->line_style("b-");
            circle_plot->line_width(2);
            
            
            // Plot trajectory trace for the last 50 frames, if available
            std::vector<double> trace_x, trace_y;
            int start = std::max(0, i - 50);
            for (int j = start; j < i; ++j) {
                trace_x.push_back(pendulum_length * std::sin(theta_arr[j]));
                trace_y.push_back(pendulum_length * std::cos(theta_arr[j]));
            }
            if (!trace_x.empty()) {
                auto trace_plot = plot(trace_x, trace_y);
                trace_plot->line_style("r--");
                trace_plot->line_width(1);
            }
            // Set fixed axis limits for stable animation.
            ax_fig3->xlim({-0.7, 0.7});
            ax_fig3->ylim({-0.7, 0.7});
            
            // Save the current frame as an image file
            std::string filename = plotDirectory + "/pendulum_frame_" + std::to_string(i) + ".png";
            save(fig3, filename);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }

    // Combine all saved frames into a GIF using ImageMagick's convert tool.
    std::string command = "convert -delay 30 " + plotDirectory + "/pendulum_frame_*.png " + plotDirectory + "/pendulum.gif";
    std::system(command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/pendulum_frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "Animation saved as pendulum.gif" << std::endl;
    
    return 0;
}
