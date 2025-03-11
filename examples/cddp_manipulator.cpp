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
#include <string>
#include <filesystem>
#include <random>
#include <chrono>
#include <thread>  // for sleep_for

#include "cddp.hpp"

#include "matplot/matplot.h"
using namespace matplot;
namespace fs = std::filesystem;

int main() {
    // -------------------- System and Problem Setup --------------------
    // System dimensions: state = [q1, q2, q3, dq1, dq2, dq3], control = [tau1, tau2, tau3]
    int state_dim = 6;
    int control_dim = 3;
    int horizon = 200;          // Time horizon for optimization
    double timestep = 0.01;

    // Create a manipulator instance (assumed to be defined in your CDDP framework)
    auto system = std::make_unique<cddp::Manipulator>(timestep, "rk4");

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    // Cost on joint angles
    Q.diagonal().segment(0, 3) = Eigen::Vector3d::Ones();
    // Cost on joint velocities
    Q.diagonal().segment(3, 3) = 0.1 * Eigen::Vector3d::Ones();
    
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 100.0 * Q;  // Terminal cost

    // Initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, -M_PI/2, M_PI, 0.0, 0.0, 0.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << M_PI, -M_PI/6, -M_PI/3, 0.0, 0.0, 0.0;

    // Create quadratic objective (with no reference trajectory)
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Create and configure the CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints: joint torques limited to ±max_torque
    double max_torque = 50.0;
    Eigen::VectorXd control_lower_bound = -max_torque * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_upper_bound =  max_torque * Eigen::VectorXd::Ones(control_dim);
    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 100;
    options.max_line_search_iterations = 20;
    options.verbose = true;
    cddp_solver.setOptions(options);

    // Initialize trajectories: here we use linear interpolation between initial and goal state
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i <= horizon; ++i) {
        double alpha = static_cast<double>(i) / horizon;
        X[i] = (1.0 - alpha) * initial_state + alpha * goal_state;
    }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the optimal control problem
    cddp::CDDPSolution solution = cddp_solver.solve();

    // -------------------- Extract Trajectories for Static Plots --------------------
    std::vector<double> time;     // for state trajectories
    std::vector<double> time_ctrl; // for control trajectories
    // For joint angles and velocities (each has 3 joints)
    std::vector<std::vector<double>> joint_angles(3), joint_velocities(3);
    std::vector<std::vector<double>> joint_torques(3);

    // Loop over the time sequence to extract states and controls
    for (size_t i = 0; i < solution.time_sequence.size(); ++i) {
        time.push_back(solution.time_sequence[i]);
        if (i < solution.state_sequence.size()) {
            for (int j = 0; j < 3; ++j) {
                joint_angles[j].push_back(solution.state_sequence[i](j));
                joint_velocities[j].push_back(solution.state_sequence[i](j + 3));
            }
        }
        if (i < solution.control_sequence.size()) {
            for (int j = 0; j < 3; ++j) {
                joint_torques[j].push_back(solution.control_sequence[i](j));
            }
            time_ctrl.push_back(solution.time_sequence[i]);
        }
    }

    // -------------------- Static Plot: Joint Angles, Velocities, and Torques --------------------
    auto fig1 = figure(true);
    fig1->size(1200, 800);

    // Joint angles subplot
    auto ax1_fig1 = subplot(3, 1, 1);
    for (int j = 0; j < 3; ++j) {
        auto plot_handle = plot(ax1_fig1, time, joint_angles[j], "-o");
        plot_handle->line_width(2);
        plot_handle->display_name("Joint " + std::to_string(j + 1));
    }
    title("Joint Angles");
    xlabel("Time [s]");
    ylabel("Angle [rad]");
    legend();
    grid(on);

    // Joint velocities subplot
    auto ax2_fig1 = subplot(3, 1, 2);
    for (int j = 0; j < 3; ++j) {
        auto plot_handle = plot(ax2_fig1, time, joint_velocities[j], "-o");
        plot_handle->line_width(2);
        plot_handle->display_name("Joint " + std::to_string(j + 1));
    }
    title("Joint Velocities");
    xlabel("Time [s]");
    ylabel("Velocity [rad/s]");
    legend();
    grid(on);

    // Joint torques subplot
    auto ax3_fig1 = subplot(3, 1, 3);
    for (int j = 0; j < 3; ++j) {
        auto plot_handle = plot(ax3_fig1, time_ctrl, joint_torques[j], "-o");
        plot_handle->line_width(2);
        plot_handle->display_name("Joint " + std::to_string(j + 1));
    }
    title("Joint Torques");
    xlabel("Time [s]");
    ylabel("Torque [Nm]");
    legend();
    grid(on);

    // Create a directory for plots if it doesn't exist
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }
    save(fig1, plotDirectory + "/manipulator_cddp_results.png");

    // Create a new figure for animation
    auto fig2 = figure(true);
    fig2->size(800, 600);
    auto ax2_fig2 = fig2->current_axes();
    cddp::Manipulator manipulator(timestep, "rk4");

    // Animate every 5th frame of the solution
    for (size_t i = 0; i < solution.state_sequence.size(); i += 5) {
        ax2_fig2->clear();  // clear current figure
        ax2_fig2->hold(true);  // hold the current plot
        ax2_fig2->grid(true);  // enable grid
        
        const auto& state = solution.state_sequence[i];

        auto transforms = manipulator.getTransformationMatrices(state(0), state(1), state(2));
        Eigen::Matrix4d T01 = transforms[0];
        Eigen::Matrix4d T02 = T01 * transforms[1];
        Eigen::Matrix4d T03 = T02 * transforms[2];
        Eigen::Matrix4d T04 = T03 * transforms[3];

        // Compute end-effector position.
        Eigen::Vector4d r3;
        r3 << manipulator.getLinkLength('c'), 0, manipulator.getLinkLength('b'), 1;
        Eigen::Vector4d r0 = T03 * r3;  // End-effector position in base frame
        // Compute an intermediate (elbow) position
        Eigen::Vector4d rm = T03 * Eigen::Vector4d(0, 0, manipulator.getLinkLength('b'), 1);

        // Prepare data for plotting:
        std::vector<double> x = {0, T03(0, 3), rm(0), r0(0)};
        std::vector<double> y = {0, T03(1, 3), rm(1), r0(1)};
        std::vector<double> z = {0, T03(2, 3), rm(2), r0(2)};
        auto link_line = plot3(x, y, z, "-o");
        link_line->line_width(2);
        link_line->color("blue");

        // Plot joints (using markers only)
        std::vector<double> joint_x = {0, T03(0, 3), rm(0)};
        std::vector<double> joint_y = {0, T03(1, 3), rm(1)};
        std::vector<double> joint_z = {0, T03(2, 3), rm(2)};
        auto joint_markers = plot3(joint_x, joint_y, joint_z, "o");
        joint_markers->color("red");

        // Plot end-effector as a marker
        std::vector<double> ee_x = {r0(0)};
        std::vector<double> ee_y = {r0(1)};
        std::vector<double> ee_z = {r0(2)};
        auto ee_marker = plot3(ee_x, ee_y, ee_z, "o");
        ee_marker->color("red");

        xlabel("X [m]");
        ylabel("Y [m]");
        zlabel("Z [m]");
        title("Manipulator CDDP Solution");
        xlim({-2, 2});
        ylim({-2, 2});
        zlim({-1, 3});
        view(30, -60);

        // Save each frame (e.g., "manipulator_frame_0.png", "manipulator_frame_1.png", …)
        std::string filename = plotDirectory + "/manipulator_frame_" + std::to_string(i/5) + ".png";
        save(fig2, filename);

        // Pause for 10 milliseconds between frames
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    
    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 20 " + plotDirectory + "/manipulator_frame_*.png " + plotDirectory + "/manipulator.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/manipulator_frame_*.png";
    std::system(cleanup_command.c_str());
    
    return 0;
}
