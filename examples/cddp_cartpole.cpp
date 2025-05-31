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
#include <cmath>
#include <string>
#include <memory>
#include "cddp.hpp"
#include "matplot/matplot.h"
#include <random>

using namespace matplot;
namespace fs = std::filesystem;

int main() {
    int state_dim = 4;
    int control_dim = 1;
    int horizon = 100;
    double timestep = 0.05;

    // Create a CartPole instance with custom parameters.
    double cart_mass = 1.0;
    double pole_mass = 0.2;
    double pole_length = 0.5;
    double gravity = 9.81;
    double damping = 0.0; // TODO: Implement damping term.
    std::string integration_type = "rk4";

    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::CartPole>(
        timestep, integration_type, cart_mass, pole_mass, pole_length, gravity, damping);

    // Cost matrices.
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0,0) = 100.0;  // Final cart position cost.
    Qf(1,1) = 100.0;  // Final cart velocity cost.
    Qf(2,2) = 100.0;  // Final pole angle cost.
    Qf(3,3) = 100.0;  // Final pole angular velocity cost.

    // Goal state: cart at origin, pole upright, zero velocities.
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state << 0.0, M_PI, 0.0, 0.0;

    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state (cart at rest, pole hanging down).
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, 0.0, 0.0;

    // Create CDDP solver.
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints.
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -5.0;  // Maximum negative force.
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 5.0;   // Maximum positive force.
    
    // FIXME: For MSIPDDP 
    cddp_solver.addConstraint("ControlConstraint", 
        std::make_unique<cddp::ControlConstraint>( control_upper_bound));

    // FIXME: For CLDDP
    // cddp_solver.addConstraint("ControlBoxConstraint", 
    //     std::make_unique<cddp::ControlBoxConstraint>( control_lower_bound, control_upper_bound));

    // Solver options.
    cddp::CDDPOptions options;
    options.max_iterations = 500;
    options.cost_tolerance = 1e-7;
    options.grad_tolerance = 1e-6;
    options.regularization_type = "control";
    options.regularization_control = 1e-5;
    options.is_ilqr = true;
    options.use_parallel = true;
    options.num_threads = 12;
    options.debug = false;
    options.barrier_coeff = 1e-1;
    options.ms_segment_length = horizon;
    options.ms_rollout_type = "nonlinear";
    options.ms_defect_tolerance_for_single_shooting = 1e-5;
    options.barrier_update_factor = 0.2;
    options.barrier_update_power = 1.2;
    options.minimum_reduction_ratio = 1e-4;
    cddp_solver.setOptions(options);

    // Initial trajectory.
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    // Generate initial trajectory by constant initial state
    for (int i = 0; i < horizon + 1; ++i) {
        X[i] = initial_state;
    }
    // // Generate initial trajectory by random
    // for (int i = 0; i < horizon + 1; ++i) {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<double> dist(-0.025, 0.025);
    //     X[i] = initial_state;
    //     X[i](0) += dist(gen);
    //     X[i](1) += dist(gen);
    //     X[i](2) += dist(gen);
    //     X[i](3) += dist(gen);
    // }
    // for (int i = 0; i < horizon; ++i) {
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_real_distribution<double> dist(-0.005, 0.005);
    //     U[i] = Eigen::VectorXd::Zero(control_dim);
    //     U[i](0) += dist(gen);
    // }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve.
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create plot directory.
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Create a directory for frame images.
    (void) std::system("mkdir -p frames");

    // Extract solution data.
    std::vector<double> x_arr, x_dot_arr, theta_arr, theta_dot_arr, force_arr, time_arr, time_arr2;
    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        theta_arr.push_back(X_sol[i](1));
        x_dot_arr.push_back(X_sol[i](2));
        theta_dot_arr.push_back(X_sol[i](3));
    }
    for (size_t i = 0; i < U_sol.size(); ++i) {
        force_arr.push_back(U_sol[i](0));
        time_arr2.push_back(t_sol[i]);
    }

    // --- Plot static results (2x2 plots for state trajectories) ---
    auto fig1 = figure();
    fig1->size(1200, 800);

    auto ax1 = subplot(2, 2, 1);
    title(ax1, "Cart Position");
    plot(ax1, time_arr, x_arr)->line_style("b-");
    xlabel(ax1, "Time [s]");
    ylabel(ax1, "Position [m]");
    grid(ax1, true);

    auto ax2 = subplot(2, 2, 2);
    title(ax2, "Cart Velocity");
    plot(ax2, time_arr, x_dot_arr)->line_style("b-");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Velocity [m/s]");
    grid(ax2, true);

    auto ax3 = subplot(2, 2, 3);
    title(ax3, "Pole Angle");
    plot(ax3, time_arr, theta_arr)->line_style("b-");
    xlabel(ax3, "Time [s]");
    ylabel(ax3, "Angle [rad]");
    grid(ax3, true);

    auto ax4 = subplot(2, 2, 4);
    title(ax4, "Pole Angular Velocity");
    plot(ax4, time_arr, theta_dot_arr)->line_style("b-");
    xlabel(ax4, "Time [s]");
    ylabel(ax4, "Angular Velocity [rad/s]");
    grid(ax4, true);

    fig1->save(plotDirectory + "/cartpole_results.png");

    // --- Plot control inputs ---
    auto fig2 = figure();
    fig2->size(800, 600);
    title("Control Inputs");
    plot(time_arr2, force_arr)->line_style("b-");
    xlabel("Time [s]");
    ylabel("Force [N]");
    grid(true);
    fig2->save(plotDirectory + "/cartpole_control_inputs.png");

    // --- Animation ---
    auto fig3 = figure();
    auto ax_fig3 = fig3->current_axes();
    fig3->size(800, 600);
    title("CartPole Animation");
    xlabel("x");
    ylabel("y");

    double cart_width = 0.3;
    double cart_height = 0.2;
    double pole_width = 0.05;

    // Loop over the solution states to generate animation frames.
    for (size_t i = 0; i < X_sol.size(); ++i) {
        if (i % 5 == 0) {
            // Clear previous content.
            cla(ax_fig3);
            hold(ax_fig3, true);

            // Current state.
            double x = x_arr[i];
            double theta = theta_arr[i];

            // Plot the cart as a rectangle centered at (x, 0).
            std::vector<double> cart_x = { x - cart_width/2, x + cart_width/2,
                                           x + cart_width/2, x - cart_width/2,
                                           x - cart_width/2 };
            std::vector<double> cart_y = { -cart_height/2, -cart_height/2,
                                           cart_height/2, cart_height/2,
                                           -cart_height/2 };
            plot(cart_x, cart_y)->line_style("k-");

            // Plot the pole as a line from the top center of the cart.
            double pole_end_x = x + pole_length * std::sin(theta);
            double pole_end_y = cart_height/2 - pole_length * std::cos(theta);
            std::vector<double> pole_x = { x, pole_end_x };
            std::vector<double> pole_y = { cart_height/2, pole_end_y };
            plot(pole_x, pole_y)->line_style("b-");

            // Plot the pole bob as a circle.
            std::vector<double> circle_x, circle_y;
            int num_points = 20;
            for (int j = 0; j <= num_points; ++j) {
                double t = 2 * M_PI * j / num_points;
                circle_x.push_back(pole_end_x + pole_width * std::cos(t));
                circle_y.push_back(pole_end_y + pole_width * std::sin(t));
            }
            plot(circle_x, circle_y)->line_style("b-");

            // Set fixed axis limits for stable animation.
            xlim({-2.0, 2.0});
            ylim({-1.5, 1.5});

            // Save the frame.
            std::string filename = plotDirectory + "/frame_" + std::to_string(i) + ".png";
            fig3->save(filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // Combine all saved frames into a GIF using ImageMagick's convert tool.
    std::string command = "convert -delay 30 " + plotDirectory + "/frame_*.png " + plotDirectory + "/cartpole.gif";
    std::system(command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "Animation saved as cartpole.gif" << std::endl;


    return 0;
}
