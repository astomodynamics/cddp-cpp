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

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main() {
    // System dimensions
    int state_dim = 12;  // [x, y, z, phi, theta, psi, vx, vy, vz, omega_x, omega_y, omega_z]
    int control_dim = 4; // [f1, f2, f3, f4] (motor forces)
    int horizon = 200;   // Longer horizon for 3D maneuvers
    double timestep = 0.02;

    // Create a quadrotor instance with custom parameters
    double mass = 1.0;  // 1kg quadrotor
    double arm_length = 0.2;  // 20cm arm length
    
    // Inertia matrix for a rough approximation of a quadrotor
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0,0) = 0.01;  // Ixx
    inertia_matrix(1,1) = 0.01;  // Iyy
    inertia_matrix(2,2) = 0.02;  // Izz
    
    std::string integration_type = "rk4";

    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Quadrotor>(
        timestep, mass, inertia_matrix, arm_length, integration_type);
    
    cddp::Quadrotor quadrotor(timestep, mass, inertia_matrix, arm_length, integration_type);

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);

    // Control cost matrix (penalize aggressive control inputs)
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    // Terminal cost matrix (important for stability)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0,0) = 50.0;  // Penalize x position
    Qf(1,1) = 50.0;  // Penalize y position
    Qf(2,2) = 50.0;  // Penalize z position
    Qf(3,3) = 1.0;  // Penalize roll angle
    Qf(4,4) = 1.0;  // Penalize pitch angle
    Qf(5,5) = 1.0;  // Penalize yaw angle
    Qf(6,6) = 10.0;  // Penalize x velocity
    Qf(7,7) = 10.0;  // Penalize y velocity
    Qf(8,8) = 10.0;  // Penalize z velocity
    Qf(9,9) = 0.1;  // Penalize roll rate
    Qf(10,10) = 0.1;  // Penalize pitch rate
    Qf(11,11) = 0.1;  // Penalize yaw rate


    // Goal state: hover at position (1,1,1) with zero angles and velocities
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = 3.0; // x
    goal_state(2) = 2.0; // z

    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state (at origin with small initial angles)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints (motor force limits)
    double min_force = 0.0;    // Motors can't push downward
    double max_force = 5.0;    // Maximum thrust per motor
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    
    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 10;             
    options.max_line_search_iterations = 15;
    options.regularization_type = "control";
    options.regularization_control = 1e-6;
    cddp_solver.setOptions(options);

    // Initial trajectory (linear interpolation for position)
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Initialize with hovering thrust
    double hover_thrust = mass * 9.81 / 4.0;  // Divide by 4 for each motor
    for (auto& u : U) {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }

    // Propagate initial trajectory
    for (size_t i = 0; i < horizon; ++i) {
        X[i+1] = quadrotor.getDiscreteDynamics(X[i], U[i]);
    }
    
    cddp_solver.setInitialTrajectory(X, U);

    // Solve
    cddp::CDDPSolution solution = cddp_solver.solve();

    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Print final state
    std::cout << "Final state: " << X_sol.back().transpose() << std::endl;

    // Create plot directory
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Extract solution data
    std::vector<double> x_arr, y_arr, z_arr;
    std::vector<double> phi_arr, theta_arr, psi_arr;
    std::vector<double> f1_arr, f2_arr, f3_arr, f4_arr;
    std::vector<double> time_arr;

    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        y_arr.push_back(X_sol[i](1));
        z_arr.push_back(X_sol[i](2));
        phi_arr.push_back(X_sol[i](3));
        theta_arr.push_back(X_sol[i](4));
        psi_arr.push_back(X_sol[i](5));
    }

    std::vector<double> time_arr2(time_arr.begin(), time_arr.end()-1);
for (const auto& u : U_sol) {
    f1_arr.push_back(u(0));
    f2_arr.push_back(u(1));
    f3_arr.push_back(u(2));
    f4_arr.push_back(u(3));
}

    // Plot position trajectories
    plt::figure_size(1200, 800);
    plt::subplot(2, 2, 1);
    plt::title("Position Trajectories");
    plt::plot(time_arr, x_arr, {{"color", "red"}, {"linestyle", "-"}, {"label", "x"}});
    plt::plot(time_arr, y_arr, {{"color", "green"}, {"linestyle", "-"}, {"label", "y"}});
    plt::plot(time_arr, z_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "z"}});
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::legend();
    plt::grid(true);

    // Plot attitude angles
    plt::subplot(2, 2, 2);
    plt::title("Attitude Angles");
    plt::plot(time_arr, phi_arr, {{"color", "red"}, {"linestyle", "-"}, {"label", "roll"}});
    plt::plot(time_arr, theta_arr, {{"color", "green"}, {"linestyle", "-"}, {"label", "pitch"}});
    plt::plot(time_arr, psi_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "yaw"}});
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [rad]");
    plt::legend();
    plt::grid(true);

    // Plot motor forces
    plt::subplot(2, 2, 3);
    plt::title("Motor Forces");
    plt::plot(time_arr2, f1_arr, {{"color", "red"}, {"linestyle", "-"}, {"label", "f1"}});
    plt::plot(time_arr2, f2_arr, {{"color", "green"}, {"linestyle", "-"}, {"label", "f2"}});
    plt::plot(time_arr2, f3_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "f3"}});
    plt::plot(time_arr2, f4_arr, {{"color", "black"}, {"linestyle", "-"}, {"label", "f4"}});
    plt::xlabel("Time [s]");
    plt::ylabel("Force [N]");
    plt::legend();
    plt::grid(true);

    // 3D trajectory plot
    plt::subplot(2, 2, 4);
    plt::title("3D Trajectory");
    plt::plot3(x_arr, y_arr, z_arr, {{"color", "blue"}, {"linestyle", "-"}, {"label", "trajectory"}});
    plt::xlabel("X [m]");
    plt::ylabel("Y [m]");
    plt::set_zlabel("Z [m]");
    plt::legend();
    plt::grid(true);

    plt::tight_layout();
    plt::save(plotDirectory + "/quadrotor_results.png");
    plt::clf();



    // // Animation frames
    plt::figure_size(800, 600);
    const long fg = plt::figure();
    plt::title("Quadrotor Animation");

    double prop_radius = 0.03;  // Radius of propeller spheres

    // Plot settings
    std::map<std::string, std::string> arm_front_back_keywords;
    arm_front_back_keywords["color"] = "blue";
    arm_front_back_keywords["linestyle"] = "-";
    arm_front_back_keywords["linewidth"] = "2";

    std::map<std::string, std::string> arm_right_left_keywords;
    arm_right_left_keywords["color"] = "red";
    arm_right_left_keywords["linestyle"] = "-";
    arm_right_left_keywords["linewidth"] = "2";

    std::map<std::string, std::string> traj_keywords;
    traj_keywords["color"] = "black";
    traj_keywords["linestyle"] = ":";
    traj_keywords["linewidth"] = "1";

    // Extract trajectory for animation
    std::vector<double> x_traj, y_traj, z_traj;

    for (size_t i = 0; i < X_sol.size(); i += 5) {  // Render every 5th frame
        plt::clf();
        
        // Current state
        double x = X_sol[i](0);
        double y = X_sol[i](1);
        double z = X_sol[i](2);
        double phi = X_sol[i](3);
        double theta = X_sol[i](4);
        double psi = X_sol[i](5);
        
        // Store trajectory points
        x_traj.push_back(x);
        y_traj.push_back(y);
        z_traj.push_back(z);
        
        // Create rotation matrix
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R = Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitX());
        
        // Transform arm endpoints to world coordinates
        Eigen::Vector3d position(x, y, z);
        std::vector<Eigen::Vector3d> arm_endpoints;
        
        // Front (blue arm)
        arm_endpoints.push_back(position + R * Eigen::Vector3d(arm_length, 0, 0));  // Front
        arm_endpoints.push_back(position + R * Eigen::Vector3d(0, arm_length, 0));  // Right
        arm_endpoints.push_back(position + R * Eigen::Vector3d(-arm_length, 0, 0)); // Back
        arm_endpoints.push_back(position + R * Eigen::Vector3d(0, -arm_length, 0)); // Left

        // Plot front-back arm (blue)
        std::vector<double> front_back_x = {arm_endpoints[0][0], arm_endpoints[2][0]};
        std::vector<double> front_back_y = {arm_endpoints[0][1], arm_endpoints[2][1]};
        std::vector<double> front_back_z = {arm_endpoints[0][2], arm_endpoints[2][2]};
        plt::plot3(front_back_x, front_back_y, front_back_z, arm_front_back_keywords, fg);

        // Plot right-left arm (red)
        std::vector<double> right_left_x = {arm_endpoints[1][0], arm_endpoints[3][0]};
        std::vector<double> right_left_y = {arm_endpoints[1][1], arm_endpoints[3][1]};
        std::vector<double> right_left_z = {arm_endpoints[1][2], arm_endpoints[3][2]};
        plt::plot3(right_left_x, right_left_y, right_left_z, arm_right_left_keywords, fg);

        // Plot propellers as small spheres
        for (size_t j = 0; j < arm_endpoints.size(); ++j) {
            // Generate sphere points
            std::vector<std::vector<double>> sphere_x, sphere_y, sphere_z;
            int resolution = 10;
            for (int u = 0; u < resolution; ++u) {
                std::vector<double> x_row, y_row, z_row;
                for (int v = 0; v < resolution; ++v) {
                    double theta = u * M_PI / (resolution-1);
                    double phi = v * 2 * M_PI / (resolution-1);
                    double x_s = arm_endpoints[j][0] + prop_radius * sin(theta) * cos(phi);
                    double y_s = arm_endpoints[j][1] + prop_radius * sin(theta) * sin(phi);
                    double z_s = arm_endpoints[j][2] + prop_radius * cos(theta);
                    x_row.push_back(x_s);
                    y_row.push_back(y_s);
                    z_row.push_back(z_s);
                }
                sphere_x.push_back(x_row);
                sphere_y.push_back(y_row);
                sphere_z.push_back(z_row);
            }

            // Plot sphere
            std::map<std::string, std::string> surf_keywords;
            surf_keywords["vmin"] = "0";
            surf_keywords["vmax"] = "1";
            surf_keywords["alpha"] = "0.99";
            
            if (j == 0 || j == 2) {  // Front or Back propeller
                surf_keywords["cmap"] = "Blues";
            } else {  // Right or Left propeller
                surf_keywords["cmap"] = "Reds";
            }

            plt::plot_surface(sphere_x, sphere_y, sphere_z, surf_keywords, fg);
        }
        
        // Plot trajectory
        plt::plot3(x_traj, y_traj, z_traj, traj_keywords, fg);
        
        // Set visualization properties
        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::set_zlabel("Z [m]");
        plt::title("Quadrotor Animation");
        plt::grid(true);
        
        // Set axis limits
        double plot_size = 3.0;
        plt::xlim(-plot_size, plot_size);
        plt::ylim(-plot_size, plot_size);
        plt::zlim(0, 3);

        // Set view angle
        plt::view_init(30, -60);
        
        std::string filename = plotDirectory + "/quadrotor_frame_" + 
                            std::to_string(i/5) + ".png";
        plt::save(filename);
        plt::pause(0.02);
    }

    return 0;
}

// Create gif from images using ImageMagick:
// convert -delay 5 ../results/tests/quadrotor_frame_*.png ../results/tests/quadrotor.gif