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
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
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
    int horizon = 400;   // Longer horizon for 3D maneuvers
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
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;
    Q(2, 2) = 1.0;
    Q(3, 3) = 1.0;
    Q(4, 4) = 1.0;
    Q(5, 5) = 1.0;
    Q(6, 6) = 1.0;

    // Control cost matrix (penalize aggressive control inputs)
    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    // Terminal cost matrix (important for stability)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Parameters for the circular trajectory
    double circle_radius = 3.0;              // e.g., 3m radius
    Eigen::Vector2d circle_center(0.0, 0.0); // center of the circle in the x-y plane
    double constant_altitude = 2.0;          // fixed altitude (z)
    double total_time = horizon * timestep;  // total duration
    // omega is chosen so that the quadrotor completes one full circle over the time horizon
    double omega = 2 * M_PI / total_time;

    std::vector<Eigen::VectorXd> circle_reference_states;
    circle_reference_states.reserve(horizon + 1);

    for (int i = 0; i <= horizon; ++i)
    {
        double t = i * timestep;
        double angle = omega * t;

        // Create a reference state of dimension state_dim (13)
        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);

        // Position (x, y, z)
        ref_state(0) = circle_center(0) + circle_radius * std::cos(angle);
        ref_state(1) = circle_center(1) + circle_radius * std::sin(angle);
        ref_state(2) = constant_altitude;

        // Orientation: set to identity quaternion [1, 0, 0, 0]
        ref_state(3) = 1.0; // qw
        ref_state(4) = 0.0; // qx
        ref_state(5) = 0.0; // qy
        ref_state(6) = 0.0; // qz                              // z velocity

        circle_reference_states.push_back(ref_state);
    }

     // Goal state: hover at position (3,0,2) with identity quaternion and zero velocities.
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = circle_center(0) + circle_radius;
    goal_state(2) = constant_altitude;
    goal_state(3) = 1.0; // Identity quaternion: qw = 1

    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, circle_reference_states, timestep);

    // Initial state (at origin with identity quaternion)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(0) = circle_center(0) + circle_radius;
    initial_state(2) = constant_altitude;
    initial_state(3) = 1.0; // Identity quaternion: qw = 1

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 10000;
    options.verbose = true;
    options.debug = false;
    options.use_parallel = true;
    options.num_threads = 10;
    options.cost_tolerance = 1e-3;
    options.grad_tolerance = 1e-2;
    options.regularization_type = "control";
    options.regularization_control = 1e-4;
    options.regularization_state = 0.0;
    options.barrier_coeff = 1e-3;

    // Create the CDDP solver
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::move(system),
        std::move(objective),
        options);

    // Control constraints (motor thrust limits)
    double min_force = 0.0; // Motors can only produce thrust upward
    double max_force = 4.0; // Maximum thrust per motor
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);

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

    // Extract solution data for plotting.
    // For attitude, convert quaternion (indices 3-6) to Euler angles.
    std::vector<double> time_arr, x_arr, y_arr, z_arr;
    std::vector<double> phi_arr, theta_arr, psi_arr;
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

    // Extract control data
    std::vector<double> time_arr2(time_arr.begin(), time_arr.end() - 1);
    std::vector<double> f1_arr, f2_arr, f3_arr, f4_arr;
    for (const auto &u : U_sol)
    {
        f1_arr.push_back(u(0));
        f2_arr.push_back(u(1));
        f3_arr.push_back(u(2));
        f4_arr.push_back(u(3));
    }

    // Plot position trajectories
    plt::figure_size(1200, 800);
    plt::subplot(3, 1, 1);
    plt::title("Position Trajectories");
    plt::plot(time_arr, x_arr, {{"color", "red"}, {"linestyle", "-"}, {"label", "x"}});
    plt::plot(time_arr, y_arr, {{"color", "green"}, {"linestyle", "-"}, {"label", "y"}});
    plt::plot(time_arr, z_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "z"}});
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::legend();
    plt::grid(true);

    // Plot attitude angles (converted from quaternion)
    plt::subplot(3, 1, 2);
    plt::title("Attitude Angles");
    plt::plot(time_arr, phi_arr, {{"color", "red"}, {"linestyle", "-"}, {"label", "roll"}});
    plt::plot(time_arr, theta_arr, {{"color", "green"}, {"linestyle", "-"}, {"label", "pitch"}});
    plt::plot(time_arr, psi_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "yaw"}});
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [rad]");
    plt::legend();
    plt::grid(true);

    // Plot motor forces
    plt::subplot(3, 1, 3);
    plt::title("Motor Forces");
    plt::plot(time_arr2, f1_arr, {{"color", "red"}, {"linestyle", "-"}, {"label", "f1"}});
    plt::plot(time_arr2, f2_arr, {{"color", "green"}, {"linestyle", "-"}, {"label", "f2"}});
    plt::plot(time_arr2, f3_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "f3"}});
    plt::plot(time_arr2, f4_arr, {{"color", "black"}, {"linestyle", "-"}, {"label", "f4"}});
    plt::xlabel("Time [s]");
    plt::ylabel("Force [N]");
    plt::legend();
    plt::grid(true);

    plt::tight_layout();
    plt::save(plotDirectory + "/quadrotor_circle_history.png");
    plt::clf();

    plt::figure();
    plt::plot3(x_arr, y_arr, z_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "trajectory"}});
    plt::xlabel("X [m]");
    plt::ylabel("Y [m]");
    plt::set_zlabel("Z [m]");
    plt::title("3D Trajectory");
    plt::legend();
    plt::grid(true);
    plt::save(plotDirectory + "/quadrotor_circle_3d.png");

    // Animation of the quadrotor frame (optional)
    plt::figure_size(800, 600);
    const long fg = plt::figure();
    plt::title("Quadrotor Animation");

    double prop_radius = 0.03; // Propeller sphere radius

    // Plot settings for arms and trajectory
    std::map<std::string, std::string> arm_front_back_keywords = {{"color", "blue"}, {"linestyle", "-"}, {"linewidth", "2"}};
    std::map<std::string, std::string> arm_right_left_keywords = {{"color", "red"}, {"linestyle", "-"}, {"linewidth", "2"}};
    std::map<std::string, std::string> traj_keywords = {{"color", "black"}, {"linestyle", ":"}, {"linewidth", "1"}};

    std::vector<double> x_traj, y_traj, z_traj;
    for (size_t i = 0; i < X_sol.size(); i += 5)
    { // Render every 5th frame
        plt::clf();
        // Current state
        double x = X_sol[i](0);
        double y = X_sol[i](1);
        double z = X_sol[i](2);
        // Get quaternion from state indices 3-6
        Eigen::Vector4d quat;
        quat << X_sol[i](3), X_sol[i](4), X_sol[i](5), X_sol[i](6);
        // Compute rotation matrix from quaternion
        Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat(0), quat(1), quat(2), quat(3));
        Eigen::Vector3d position(x, y, z);
        x_traj.push_back(x);
        y_traj.push_back(y);
        z_traj.push_back(z);

        // Compute arm endpoints (motor locations)
        std::vector<Eigen::Vector3d> arm_endpoints;
        arm_endpoints.push_back(position + R * Eigen::Vector3d(arm_length, 0, 0));  // Front
        arm_endpoints.push_back(position + R * Eigen::Vector3d(0, arm_length, 0));  // Right
        arm_endpoints.push_back(position + R * Eigen::Vector3d(-arm_length, 0, 0)); // Back
        arm_endpoints.push_back(position + R * Eigen::Vector3d(0, -arm_length, 0)); // Left

        // Plot front-back arm (blue)
        std::vector<double> front_back_x = {arm_endpoints[0](0), arm_endpoints[2](0)};
        std::vector<double> front_back_y = {arm_endpoints[0](1), arm_endpoints[2](1)};
        std::vector<double> front_back_z = {arm_endpoints[0](2), arm_endpoints[2](2)};
        plt::plot3(front_back_x, front_back_y, front_back_z, arm_front_back_keywords, fg);

        // Plot right-left arm (red)
        std::vector<double> right_left_x = {arm_endpoints[1](0), arm_endpoints[3](0)};
        std::vector<double> right_left_y = {arm_endpoints[1](1), arm_endpoints[3](1)};
        std::vector<double> right_left_z = {arm_endpoints[1](2), arm_endpoints[3](2)};
        plt::plot3(right_left_x, right_left_y, right_left_z, arm_right_left_keywords, fg);

        // Plot propeller spheres at each motor location
        for (size_t j = 0; j < arm_endpoints.size(); ++j)
        {
            std::vector<std::vector<double>> sphere_x, sphere_y, sphere_z;
            int resolution = 10;
            for (int u = 0; u < resolution; ++u)
            {
                std::vector<double> x_row, y_row, z_row;
                for (int v = 0; v < resolution; ++v)
                {
                    double theta = u * M_PI / (resolution - 1);
                    double phi = v * 2 * M_PI / (resolution - 1);
                    double x_s = arm_endpoints[j](0) + prop_radius * std::sin(theta) * std::cos(phi);
                    double y_s = arm_endpoints[j](1) + prop_radius * std::sin(theta) * std::sin(phi);
                    double z_s = arm_endpoints[j](2) + prop_radius * std::cos(theta);
                    x_row.push_back(x_s);
                    y_row.push_back(y_s);
                    z_row.push_back(z_s);
                }
                sphere_x.push_back(x_row);
                sphere_y.push_back(y_row);
                sphere_z.push_back(z_row);
            }
            std::map<std::string, std::string> surf_keywords;
            surf_keywords["vmin"] = "0";
            surf_keywords["vmax"] = "1";
            surf_keywords["alpha"] = "0.99";
            surf_keywords["cmap"] = (j == 0 || j == 2) ? "Blues" : "Reds";
            plt::plot_surface(sphere_x, sphere_y, sphere_z, surf_keywords, fg);
        }

        // Plot trajectory
        plt::plot3(x_traj, y_traj, z_traj, traj_keywords, fg);

        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::set_zlabel("Z [m]");
        plt::title("Quadrotor Animation");
        plt::grid(true);
        double plot_size = 3.0;
        plt::xlim(-plot_size, plot_size);
        plt::ylim(-plot_size, plot_size);
        plt::zlim(0, 3);
        plt::view_init(30, -60);

        std::string filename = plotDirectory + "/quadrotor_frame_" + std::to_string(i / 5) + ".png";
        plt::save(filename);
        plt::pause(0.02);
    }
    return 0;
}

// To create a gif from the saved frames, use ImageMagick (for example):
// convert -delay 5 ../results/tests/quadrotor_frame_*.png ../results/tests/quadrotor_circle.gif
