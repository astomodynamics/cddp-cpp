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

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

int main() {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(timestep, integration_type); // Create unique_ptr

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 50.0, 0.0, 0.0,
          0.0, 50.0, 0.0,
          0.0, 0.0, 10.0;
    Qf = 0.5 * Qf;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Create an empty vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> empty_reference_states; 
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial and target states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0; 

    // Set options
    cddp::CDDPOptions options;
    options.max_iterations = 10;
    options.ipddp.barrier.mu_initial = 1e-2;
    options.ipddp.barrier.mu_update_factor = 0.1;

    // Create CDDP solver with new API
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::move(system), std::move(objective), options);

    // Define constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    
    // Add the constraint to the solver
    cddp_solver.addPathConstraint(std::string("ControlBoxConstraint"), std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve(cddp::SolverType::CLDDP);

    // Extract solution
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory")); // size: horizon + 1
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory")); // size: horizon
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points")); // size: horizon + 1

    // Create directory for saving plot (if it doesn't exist)
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Plot the solution (x-y plane)
    std::vector<double> x_arr, y_arr, theta_arr;
    for (const auto& x : X_sol) {
        x_arr.push_back(x(0));
        y_arr.push_back(x(1));
        theta_arr.push_back(x(2));
    }

    // Plot the solution (control inputs)
    std::vector<double> v_arr, omega_arr;
    for (const auto& u : U_sol) {
        v_arr.push_back(u(0));
        omega_arr.push_back(u(1));
    }

    // -----------------------------
    // Plot states and controls
    // -----------------------------
    auto f1 = figure();
    f1->size(1200, 800);

    // First subplot: Position Trajectory
    auto ax1 = subplot(3, 1, 0);
    auto plot_handle = plot(ax1, x_arr, y_arr, "-b");
    plot_handle->line_width(3);
    title(ax1, "Position Trajectory");
    xlabel(ax1, "x [m]");
    ylabel(ax1, "y [m]");

    // Second subplot: Heading Angle vs Time
    auto ax2 = subplot(3, 1, 1);
    auto heading_plot_handle = plot(ax2, t_sol, theta_arr);
    heading_plot_handle->line_width(3);
    title(ax2, "Heading Angle");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "theta [rad]");

    // Fourth subplot: Control Inputs
    auto ax4 = subplot(3, 1, 2);
    auto p1 = plot(ax4, v_arr, "--b");
    p1->line_width(3);
    p1->display_name("Acceleration");

    hold(ax4, true);
    auto p2 = plot(ax4, omega_arr, "--r");
    p2->line_width(3);
    p2->display_name("Steering");

    title(ax4, "Control Inputs");
    xlabel(ax4, "Step");
    ylabel(ax4, "Control");
    matplot::legend(ax4);

    f1->draw();
    f1->save(plotDirectory + "/unicycle_cddp_results.png");

    // -----------------------------
    // Animation: unicycle Trajectory
    // -----------------------------
    auto f2 = figure();
    f2->size(800, 600);
    auto ax_anim = f2->current_axes();
    if (!ax_anim)
    {
        ax_anim = axes();
    }

    double car_length = 0.35;
    double car_width = 0.15;

    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        if (i % 10 == 0)
        {
            ax_anim->clear();
            hold(ax_anim, true);

            double x = x_arr[i];
            double y = y_arr[i];
            double theta = theta_arr[i];

            // Compute unicycle rectangle corners
            std::vector<double> car_x(5), car_y(5);
            car_x[0] = x + car_length / 2 * cos(theta) - car_width / 2 * sin(theta);
            car_y[0] = y + car_length / 2 * sin(theta) + car_width / 2 * cos(theta);
            car_x[1] = x + car_length / 2 * cos(theta) + car_width / 2 * sin(theta);
            car_y[1] = y + car_length / 2 * sin(theta) - car_width / 2 * cos(theta);
            car_x[2] = x - car_length / 2 * cos(theta) + car_width / 2 * sin(theta);
            car_y[2] = y - car_length / 2 * sin(theta) - car_width / 2 * cos(theta);
            car_x[3] = x - car_length / 2 * cos(theta) - car_width / 2 * sin(theta);
            car_y[3] = y - car_length / 2 * sin(theta) + car_width / 2 * cos(theta);
            car_x[4] = car_x[0];
            car_y[4] = car_y[0];

            auto car_line = plot(ax_anim, car_x, car_y);
            car_line->color("black");
            car_line->line_style("solid");
            car_line->line_width(2);
            car_line->display_name("Car");

            // Plot trajectory up to current frame
            std::vector<double> traj_x(x_arr.begin(), x_arr.begin() + i + 1);
            std::vector<double> traj_y(y_arr.begin(), y_arr.begin() + i + 1);
            auto traj_line = plot(ax_anim, traj_x, traj_y);
            traj_line->color("blue");
            traj_line->line_style("solid");
            traj_line->line_width(1.5);
            traj_line->display_name("Trajectory");

            title(ax_anim, "unicycle Trajectory");
            xlabel(ax_anim, "x [m]");
            ylabel(ax_anim, "y [m]");
            xlim(ax_anim, {-1, 2.2});
            ylim(ax_anim, {-1, 2.2});
            // legend(ax_anim);

            std::string filename = plotDirectory + "/unicycle_frame_" + std::to_string(i) + ".png";
            f2->draw();
            f2->save(filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 30 " + plotDirectory + "/unicycle_frame_*.png " + plotDirectory + "/unicycle.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/unicycle_frame_*.png";
    std::system(cleanup_command.c_str());
}
