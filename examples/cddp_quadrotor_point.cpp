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
#include <memory>
#include <cmath>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

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

// Helper function: Compute rotation matrix from a unit quaternion [qw, qx, qy, qz]
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

// Helper: Transform quadrotor frame points (motor positions) to world coordinates using quaternion.
std::vector<std::vector<double>> transformQuadrotorFrame(
    const Eigen::Vector3d &position,
    const Eigen::Vector4d &quat, // [qw, qx, qy, qz]
    double arm_length)
{

    // Define quadrotor motor positions in the body frame
    std::vector<Eigen::Vector3d> body_points = {
        Eigen::Vector3d(arm_length, 0, 0),  // Front motor
        Eigen::Vector3d(0, arm_length, 0),  // Right motor
        Eigen::Vector3d(-arm_length, 0, 0), // Back motor
        Eigen::Vector3d(0, -arm_length, 0)  // Left motor
    };

    Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat[0], quat[1], quat[2], quat[3]);

    std::vector<std::vector<double>> world_points(3, std::vector<double>());
    for (const auto &pt : body_points)
    {
        Eigen::Vector3d wp = position + R * pt;
        world_points[0].push_back(wp.x());
        world_points[1].push_back(wp.y());
        world_points[2].push_back(wp.z());
    }
    return world_points;
}

int main()
{
    // For quaternion-based quadrotor, state_dim = 13:
    // [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    int state_dim = 13;
    int control_dim = 4; // [f1, f2, f3, f4]
    int horizon = 200;   // Longer horizon for 3D maneuvers
    double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.0;       // 1kg quadrotor
    double arm_length = 0.2; // 20cm arm length
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 0.01; // Ixx
    inertia_matrix(1, 1) = 0.01; // Iyy
    inertia_matrix(2, 2) = 0.02; // Izz

    std::string integration_type = "rk4";

    // Create the dynamical system (quadrotor) as a unique_ptr
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type);

    // For propagation (e.g., initial trajectory), also create an instance (if needed)
    cddp::Quadrotor quadrotor(timestep, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices (dimensions updated for state_dim = 13)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(4, 4) = 0.1;
    Q(5, 5) = 0.1;
    Q(6, 6) = 0.1;

    // Control cost matrix (penalize aggressive control inputs)
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    // Terminal cost matrix (important for stability)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 500.0; // x position
    Qf(1, 1) = 500.0; // y position
    Qf(2, 2) = 500.0; // z position
    // For orientation (quaternion) we penalize deviation from identity quaternion
    Qf(3, 3) = 1.0;  // qw
    Qf(4, 4) = 1.0;  // qx
    Qf(5, 5) = 1.0;  // qy
    Qf(6, 6) = 1.0;  // qz
    Qf(7, 7) = 10.0; // x velocity
    Qf(8, 8) = 10.0; // y velocity
    Qf(9, 9) = 10.0; // z velocity
    // Qf(10, 10) = 0.1; // roll rate
    // Qf(11, 11) = 0.1; // pitch rate
    // Qf(12, 12) = 0.1; // yaw rate

    // Goal state: hover at position (3,0,2) with identity quaternion and zero velocities.
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = 3.0; // x
    goal_state(2) = 2.0; // z
    // Set identity quaternion [1, 0, 0, 0]
    goal_state(3) = 1.0;
    goal_state(4) = 0.0;
    goal_state(5) = 0.0;
    goal_state(6) = 0.0;

    // No intermediate reference states (optional)
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state (at origin with identity quaternion)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(3) = 1.0; // Identity quaternion: qw = 1

    // Create the CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints (motor thrust limits)
    double min_force = 0.0; // Motors can only produce thrust upward
    double max_force = 5.0; // Maximum thrust per motor
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    cddp_solver.addConstraint("ControlConstraint",
                              std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 2000;
    options.max_line_search_iterations = 15;
    options.regularization_type = "control";
    options.regularization_control = 1e-4;
    cddp_solver.setOptions(options);

    // Initial trajectory: allocate state and control trajectories
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    // Initialize with hovering thrust (each motor provides mg/4)
    double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U)
    {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }
    // Propagate initial trajectory using discrete dynamics
    for (size_t i = 0; i < static_cast<size_t>(horizon); ++i)
    {
        X[i + 1] = quadrotor.getDiscreteDynamics(X[i], U[i]);
    }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the optimal control problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    std::cout << "Final state: " << X_sol.back().transpose() << std::endl;

    // Create plot directory if it doesn't exist
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory))
    {
        fs::create_directory(plotDirectory);
    }
    // Create a directory for frame images.
    (void)std::system("mkdir -p frames");

    // Extract trajectory data
    std::vector<double> time_arr, x_arr, y_arr, z_arr;
    std::vector<double> phi_arr, theta_arr, psi_arr;
    time_arr.reserve(X_sol.size());
    x_arr.reserve(X_sol.size());
    y_arr.reserve(X_sol.size());
    z_arr.reserve(X_sol.size());
    phi_arr.reserve(X_sol.size());
    theta_arr.reserve(X_sol.size());
    psi_arr.reserve(X_sol.size());

    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        y_arr.push_back(X_sol[i](1));
        z_arr.push_back(X_sol[i](2));
        Eigen::Vector4d quat;
        quat << X_sol[i](3), X_sol[i](4), X_sol[i](5), X_sol[i](6);
        Eigen::Vector3d euler = quaternionToEuler(quat(0), quat(1), quat(2), quat(3));
        phi_arr.push_back(euler(0));
        theta_arr.push_back(euler(1));
        psi_arr.push_back(euler(2));
    }

    // Control data
    std::vector<double> time_arr2(time_arr.begin(), time_arr.end() - 1);
    std::vector<double> f1_arr, f2_arr, f3_arr, f4_arr;
    f1_arr.reserve(U_sol.size());
    f2_arr.reserve(U_sol.size());
    f3_arr.reserve(U_sol.size());
    f4_arr.reserve(U_sol.size());

    for (const auto &u : U_sol)
    {
        f1_arr.push_back(u(0));
        f2_arr.push_back(u(1));
        f3_arr.push_back(u(2));
        f4_arr.push_back(u(3));
    }

    // Plotting
    auto f1 = figure();
    f1->size(1200, 800);

    // Position
    auto ax1_f1 = subplot(3, 1, 0);
    auto plot_handle1 = plot(ax1_f1, time_arr, x_arr, "-r");
    plot_handle1->display_name("x");
    plot_handle1->line_width(2);
    hold(ax1_f1, true);
    auto plot_handle2 = plot(ax1_f1, time_arr, y_arr, "-g");
    plot_handle2->display_name("y");
    plot_handle2->line_width(2);
    hold(ax1_f1, true);
    auto plot_handle3 = plot(ax1_f1, time_arr, z_arr, "-b");
    plot_handle3->display_name("z");
    plot_handle3->line_width(2);

    title(ax1_f1, "Position Trajectories");
    xlabel(ax1_f1, "Time [s]");
    ylabel(ax1_f1, "Position [m]");
    legend(ax1_f1);
    grid(ax1_f1, true);

    // Attitude
    auto ax2_f1 = subplot(3, 1, 1);
    hold(ax2_f1, true);
    auto plot_handle4 = plot(ax2_f1, time_arr, phi_arr, "-r");
    plot_handle4->display_name("roll");
    plot_handle4->line_width(2);
    auto plot_handle5 = plot(ax2_f1, time_arr, theta_arr, "-g");
    plot_handle5->display_name("pitch");
    plot_handle5->line_width(2);
    hold(ax2_f1, true);
    auto plot_handle6 = plot(ax2_f1, time_arr, psi_arr, "-b");
    plot_handle6->display_name("yaw");
    plot_handle6->line_width(2);

    title(ax2_f1, "Attitude Angles");
    xlabel(ax2_f1, "Time [s]");
    ylabel(ax2_f1, "Angle [rad]");
    legend(ax2_f1);
    grid(ax2_f1, true);

    // Motor forces
    auto ax3_f1 = subplot(3, 1, 2);
    auto plot_handle7 = plot(ax3_f1, time_arr2, f1_arr, "-r");
    plot_handle7->display_name("f1");
    plot_handle7->line_width(2);
    hold(ax3_f1, true);
    auto plot_handle8 = plot(ax3_f1, time_arr2, f2_arr, "-g");
    plot_handle8->display_name("f2");
    plot_handle8->line_width(2);
    hold(ax3_f1, true);
    auto plot_handle9 = plot(ax3_f1, time_arr2, f3_arr, "-b");
    plot_handle9->display_name("f3");
    plot_handle9->line_width(2);
    hold(ax3_f1, true);
    auto plot_handle10 = plot(ax3_f1, time_arr2, f4_arr, "-k");
    plot_handle10->display_name("f4");
    plot_handle10->line_width(2);

    title(ax3_f1, "Motor Forces");
    xlabel(ax3_f1, "Time [s]");
    ylabel(ax3_f1, "Force [N]");
    legend(ax3_f1);
    grid(ax3_f1, true);

    // Save
    f1->draw();
    f1->save(plotDirectory + "/quadrotor_point_states.png");

    // 3D Trajectory
    auto f2 = figure();
    f2->size(800, 600);
    auto ax2 = f2->current_axes();
    hold(ax2, true);
    auto traj3d = plot3(ax2, x_arr, y_arr, z_arr);
    traj3d->display_name("Trajectory");
    traj3d->line_style("-");
    traj3d->line_width(2);
    traj3d->color("blue");
    // Project trajectory onto x-y plane at z=0
    auto proj_xy = plot3(ax2, x_arr, y_arr, std::vector<double>(x_arr.size(), 0.0));
    proj_xy->display_name("X-Y Projection");
    proj_xy->line_style("--");
    proj_xy->line_width(1);
    proj_xy->color("gray");
    xlabel(ax2, "X [m]");
    ylabel(ax2, "Y [m]");
    zlabel(ax2, "Z [m]");
    xlim(ax2, {0, 5});
    ylim(ax2, {-2, 2});
    zlim(ax2, {0, 5});
    title(ax2, "3D Trajectory (Figure-8)");
    grid(ax2, true);
    f2->draw();
    f2->save(plotDirectory + "/quadrotor_point_3d.png");

    // Animation of the quadrotor frame
    auto f_anim = figure();
    f_anim->size(800, 600);
    auto ax_anim = f_anim->current_axes();

    // For collecting the trajectory as we go
    std::vector<double> anim_x, anim_y, anim_z;
    anim_x.reserve(X_sol.size());
    anim_y.reserve(X_sol.size());
    anim_z.reserve(X_sol.size());

    // Render every Nth frame to reduce #images
    int frame_stride = 15;
    double prop_radius = 0.03; // radius for small spheres at motor ends

    for (size_t i = 0; i < X_sol.size(); i += frame_stride)
    {
        ax_anim->clear();
        ax_anim->hold(true);
        ax_anim->grid(true);

        // Current state
        double x = X_sol[i](0);
        double y = X_sol[i](1);
        double z = X_sol[i](2);

        // Accumulate path
        anim_x.push_back(x);
        anim_y.push_back(y);
        anim_z.push_back(z);

        // Plot the partial trajectory so far (in black dotted line)
        auto path_plot = plot3(anim_x, anim_y, anim_z);
        path_plot->line_width(1.5);
        path_plot->line_style("--");
        path_plot->color("black");

        // Build rotation from quaternion
        Eigen::Vector4d quat(X_sol[i](3), X_sol[i](4), X_sol[i](5), X_sol[i](6));
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

        // Front-back arm
        std::vector<double> fx = {arm_endpoints[0].x(), arm_endpoints[2].x()};
        std::vector<double> fy = {arm_endpoints[0].y(), arm_endpoints[2].y()};
        std::vector<double> fz = {arm_endpoints[0].z(), arm_endpoints[2].z()};
        auto fb_arm = plot3(fx, fy, fz);
        fb_arm->line_width(2.0);
        fb_arm->color("blue");

        // Right-left arm
        std::vector<double> rx = {arm_endpoints[1].x(), arm_endpoints[3].x()};
        std::vector<double> ry = {arm_endpoints[1].y(), arm_endpoints[3].y()};
        std::vector<double> rz = {arm_endpoints[1].z(), arm_endpoints[3].z()};
        auto rl_arm = plot3(rx, ry, rz);
        rl_arm->line_width(2.0);
        rl_arm->color("red");

        auto sphere_points = linspace(0, 2 * M_PI, 15);
        for (const auto &motor_pos : arm_endpoints)
        {
            std::vector<double> circ_x, circ_y, circ_z;
            circ_x.reserve(sphere_points.size());
            circ_y.reserve(sphere_points.size());
            circ_z.reserve(sphere_points.size());
            for (auto angle : sphere_points)
            {
                circ_x.push_back(motor_pos.x() + prop_radius * cos(angle));
                circ_y.push_back(motor_pos.y() + prop_radius * sin(angle));
                circ_z.push_back(motor_pos.z()); // keep the same z for a small ring
            }
            auto sphere_plot = plot3(circ_x, circ_y, circ_z);
            sphere_plot->line_style("solid");
            sphere_plot->line_width(1.5);
            sphere_plot->color("cyan");
        }

        title(ax_anim, "Quadrotor Animation");
        xlabel(ax_anim, "X [m]");
        ylabel(ax_anim, "Y [m]");
        zlabel(ax_anim, "Z [m]");
        xlim(ax_anim, {0, 5});
        ylim(ax_anim, {-5, 5});
        zlim(ax_anim, {0, 5});

        ax_anim->view(30, -30);

        std::string frameFile = plotDirectory + "/quadrotor_anim_frame_" + std::to_string(i / frame_stride) + ".png";
        f_anim->draw();
        f_anim->save(frameFile);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }

    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 30 " + plotDirectory + "/quadrotor_anim_frame_*.png " + plotDirectory + "/quadrotor_point.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/quadrotor_anim_frame_*.png";
    std::system(cleanup_command.c_str());
    return 0;
}

// To create a gif from the saved frames, use ImageMagick (for example):
// convert -delay 5 ../results/tests/quadrotor_frame_*.png ../results/tests/quadrotor.gif
