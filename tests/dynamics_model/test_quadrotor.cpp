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
// Description: Test the quadrotor dynamics model.

#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/quadrotor.hpp"
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(QuadrotorTest, DiscreteDynamics) {
    // Create a quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;  // 1 kg
    double arm_length = 0.2;  // 20 cm
    
    // Diagonal inertia matrix for a symmetric quadrotor
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
               
    std::string integration_type = "rk4";
    cddp::Quadrotor quadrotor(timestep, mass, inertia, arm_length, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> phi_data, theta_data, psi_data;

    // Initial state: hovering with slight initial rotation
    Eigen::VectorXd state = Eigen::VectorXd::Zero(12);
    state(2) = 1.0;  // Start at 1m height
    state(3) = 0.0;  // Small initial roll angle

    // Control input for hover (each motor provides mg/4 force)
    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Simulate for a few steps
    int num_steps = 500;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        z_data.push_back(state[2]);
        phi_data.push_back(state[3]);
        theta_data.push_back(state[4]);
        psi_data.push_back(state[5]);

        // Compute the next state
        state = quadrotor.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(quadrotor.getStateDim(), 12);
    ASSERT_EQ(quadrotor.getControlDim(), 4);
    ASSERT_DOUBLE_EQ(quadrotor.getTimestep(), 0.01);
    ASSERT_EQ(quadrotor.getIntegrationType(), "rk4");

    // Test if quadrotor maintains approximate hover height
    double final_height = z_data.back();
    EXPECT_NEAR(final_height, 1.0, 0.1);

    // // Plot the results in time series 
    // plt::figure();
    // plt::subplot(3, 2, 1);
    // plt::plot(time_data, x_data);
    // plt::title("X Position");
    // plt::xlabel("Time [s]");
    // plt::ylabel("X [m]");
    // plt::grid(true);

    // plt::subplot(3, 2, 2);
    // plt::plot(time_data, phi_data);
    // plt::title("Roll Angle");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Roll [rad]");

    // plt::subplot(3, 2, 3);
    // plt::plot(time_data, y_data);
    // plt::title("Y Position");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Y [m]");
    // plt::grid(true);

    // plt::subplot(3, 2, 4);
    // plt::plot(time_data, theta_data);
    // plt::title("Pitch Angle");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Pitch [rad]");

    // plt::subplot(3, 2, 5);
    // plt::plot(time_data, z_data);
    // plt::title("Z Position");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Z [m]");
    // plt::grid(true);

    // plt::subplot(3, 2, 6);
    // plt::plot(time_data, psi_data);
    // plt::title("Yaw Angle");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Yaw [rad]");

    // plt::tight_layout();

    // plt::save("../results/tests/quadrotor_discrete_dynamics.png");
    // plt::show();
}

TEST(QuadrotorTest, ContinuousDynamics) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
    cddp::Quadrotor quadrotor(timestep, mass, inertia, arm_length, "euler");

    // Test hover condition
    Eigen::VectorXd state = Eigen::VectorXd::Zero(12);
    state(2) = 1.0;  // 1m height

    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Get dynamics
    Eigen::VectorXd state_dot = quadrotor.getContinuousDynamics(state, control);

    // Test expected behavior for hover
    // Position changes should be zero
    EXPECT_NEAR(state_dot[0], 0.0, 1e-10);  // dx/dt
    EXPECT_NEAR(state_dot[1], 0.0, 1e-10);  // dy/dt
    EXPECT_NEAR(state_dot[2], 0.0, 1e-10);  // dz/dt
    
    // Angular rates should be zero
    EXPECT_NEAR(state_dot[3], 0.0, 1e-10);  // dphi/dt
    EXPECT_NEAR(state_dot[4], 0.0, 1e-10);  // dtheta/dt
    EXPECT_NEAR(state_dot[5], 0.0, 1e-10);  // dpsi/dt

    // Test with unbalanced thrust (should create rolling moment)
    control(0) += 0.1;  // Increase front rotor thrust
    control(2) -= 0.1;  // Decrease back rotor thrust
    state_dot = quadrotor.getContinuousDynamics(state, control);
    
    // Should have non-zero rolling moment
    EXPECT_GT(std::abs(state_dot[9]), 0.0);  // Non-zero roll acceleration
}


// Helper function to compute rotation matrix
Eigen::Matrix3d getRotationMatrix(double phi, double theta, double psi) {
    double c_phi = std::cos(phi);
    double s_phi = std::sin(phi);
    double c_theta = std::cos(theta);
    double s_theta = std::sin(theta);
    double c_psi = std::cos(psi);
    double s_psi = std::sin(psi);
    
    Eigen::Matrix3d R;
    R << c_psi*c_theta, c_psi*s_theta*s_phi - s_psi*c_phi, c_psi*s_theta*c_phi + s_psi*s_phi,
         s_psi*c_theta, s_psi*s_theta*s_phi + c_psi*c_phi, s_psi*s_theta*c_phi - c_psi*s_phi,
         -s_theta, c_theta*s_phi, c_theta*c_phi;
    
    return R;
}

// Function to transform quadrotor frame points
std::vector<std::vector<double>> transformQuadrotorFrame(
    const Eigen::Vector3d& position,
    const Eigen::Matrix3d& rotation,
    double arm_length) {
    
    // Define quadrotor frame points in body frame
    std::vector<Eigen::Vector3d> body_points = {
        Eigen::Vector3d(arm_length, 0, 0),   // Front (motor 0)
        Eigen::Vector3d(0, arm_length, 0),   // Right (motor 1)
        Eigen::Vector3d(-arm_length, 0, 0),  // Back (motor 2)
        Eigen::Vector3d(0, -arm_length, 0)   // Left (motor 3)
    };
    
    // Transform points to world frame
    std::vector<std::vector<double>> world_points(3, std::vector<double>());
    for (const auto& point : body_points) {
        Eigen::Vector3d world_point = position + rotation * point;
        world_points[0].push_back(world_point.x());
        world_points[1].push_back(world_point.y());
        world_points[2].push_back(world_point.z());
    }
    
    return world_points;
}

std::vector<std::vector<double>> generatePropellerPoints(
    const Eigen::Vector3d& center,
    const Eigen::Matrix3d& rotation,
    double radius,
    int num_points = 20) {
    
    std::vector<std::vector<double>> circle_points(3, std::vector<double>());
    
    for (int i = 0; i <= num_points; ++i) {
        double angle = 2.0 * M_PI * i / num_points;
        // Generate circle in XY plane
        Eigen::Vector3d point_body(
            radius * std::cos(angle),
            radius * std::sin(angle),
            0.0
        );
        // Transform to world frame
        Eigen::Vector3d point_world = center + rotation * point_body;
        
        circle_points[0].push_back(point_world.x());
        circle_points[1].push_back(point_world.y());
        circle_points[2].push_back(point_world.z());
    }
    
    return circle_points;
}

// TEST(QuadrotorTest, Animation2D) {
//     const long fg = plt::figure();
//     // Create quadrotor instance
//     double timestep = 0.01;
//     double mass = 1.0;
//     double arm_length = 0.2;
    
//     Eigen::Matrix3d inertia;
//     inertia << 0.01, 0, 0,
//                0, 0.01, 0,
//                0, 0, 0.02;
               
//     cddp::Quadrotor quadrotor(timestep, mass, inertia, arm_length, "rk4");

//     // Initial state: slightly tilted hover
//     Eigen::VectorXd state = Eigen::VectorXd::Zero(12);
//     state << 0.0, 0.0, 1.0,           // position
//             0.1, 0.1, 0.0,            // orientation (small roll and pitch)
//             0.0, 0.0, 0.0,            // linear velocity
//             0.0, 0.0, 0.2;            // angular velocity (small yaw rate)

//     // Hover thrust plus small variations
//     double hover_thrust = mass * 9.81 / 4.0;
//     Eigen::VectorXd control(4);
//     control << hover_thrust * 1.1,    // Front (motor 0)
//               hover_thrust * 0.9,     // Right (motor 1)
//               hover_thrust * 1.0,     // Back (motor 2)
//               hover_thrust * 1.0;     // Left (motor 3)

//     // Simulation and animation parameters
//     int num_steps = 1000;
//     double animation_interval = 0.02;  // seconds
//     double prop_radius = 0.1;        // 10cm propeller radius
    
//     // Storage for trajectory
//     std::vector<double> x_traj, y_traj;
    
    // // Plot settings for trajectory
    // std::map<std::string, std::string> traj_keywords;
    // traj_keywords["label"] = "Trajectory";
    // traj_keywords["linestyle"] = ":";
    // traj_keywords["color"] = "black";

    // // Plot settings for arms
    // std::map<std::string, std::string> arm_front_back_keywords;
    // arm_front_back_keywords["color"] = "blue";
    // arm_front_back_keywords["linestyle"] = "-";
    // arm_front_back_keywords["linewidth"] = "2";

    // std::map<std::string, std::string> arm_right_left_keywords;
    // arm_right_left_keywords["color"] = "red";
    // arm_right_left_keywords["linestyle"] = "-";
    // arm_right_left_keywords["linewidth"] = "2";

    // // Plot settings for propellers
    // std::map<std::string, std::string> prop_front_back_keywords;
    // prop_front_back_keywords["color"] = "blue";
    // prop_front_back_keywords["linestyle"] = "--";
    // prop_front_back_keywords["linewidth"] = "1";

    // std::map<std::string, std::string> prop_right_left_keywords;
    // prop_right_left_keywords["color"] = "red";
    // prop_right_left_keywords["linestyle"] = "--";
    // prop_right_left_keywords["linewidth"] = "1";
    
    // // Add legend keywords
    // arm_front_back_keywords["label"] = "Front-Back Arm";
    // arm_right_left_keywords["label"] = "Right-Left Arm";
    // prop_front_back_keywords["label"] = "Front-Back Propellers";
    // prop_right_left_keywords["label"] = "Right-Left Propellers";
    
    // for (int i = 0; i < num_steps; ++i) {
    //     if (i % 5 == 0) {
    //         // Extract current position and orientation
    //         Eigen::Vector3d position(state[0], state[1], state[2]);
    //         Eigen::Matrix3d rotation = getRotationMatrix(state[3], state[4], state[5]);
            
    //         // Transform quadrotor frame to world coordinates
    //         auto frame_points = transformQuadrotorFrame(position, rotation, arm_length);
            
    //         // Store trajectory
    //         x_traj.push_back(position.x());
    //         y_traj.push_back(position.y());

    //         // Clear previous plot
    //         plt::clf();
            
    //         // Plot quadrotor frame
    //         // Front-Back arm (blue)
    //         std::vector<double> front_back_x = {frame_points[0][0], frame_points[0][2]};  // Front to Back
    //         std::vector<double> front_back_y = {frame_points[1][0], frame_points[1][2]};
    //         plt::plot(front_back_x, front_back_y, arm_front_back_keywords);

    //         // Right-Left arm (red)
    //         std::vector<double> right_left_x = {frame_points[0][1], frame_points[0][3]};  // Right to Left
    //         std::vector<double> right_left_y = {frame_points[1][1], frame_points[1][3]};
    //         plt::plot(right_left_x, right_left_y, arm_right_left_keywords);

    //         // Plot propeller circles with matching colors
    //         for (int j = 0; j < 4; ++j) {
    //             Eigen::Vector3d prop_center(
    //                 frame_points[0][j],
    //                 frame_points[1][j],
    //                 frame_points[2][j]
    //             );
                
    //             // Generate circle points in XY plane
    //             std::vector<double> circle_x, circle_y;
    //             int num_points = 50;
    //             for (int k = 0; k <= num_points; ++k) {
    //                 double angle = 2.0 * M_PI * k / num_points;
                    
    //                 // Apply rotation to circle points
    //                 Eigen::Vector3d circle_point_body(
    //                     prop_radius * std::cos(angle),
    //                     prop_radius * std::sin(angle),
    //                     0.0
    //                 );
    //                 Eigen::Vector3d circle_point_world = prop_center + rotation * circle_point_body;
                    
    //                 circle_x.push_back(circle_point_world.x());
    //                 circle_y.push_back(circle_point_world.y());
    //             }
                
    //             // Use matching colors for propellers
    //             if (j == 0 || j == 2) {  // Front or Back propeller
    //                 plt::plot(circle_x, circle_y, prop_front_back_keywords);
    //             } else {  // Right or Left propeller
    //                 plt::plot(circle_x, circle_y, prop_right_left_keywords);
    //             }
    //         }
            
    //         // Plot trajectory
    //         plt::plot(x_traj, y_traj, traj_keywords);
            
    //         // Plot settings
    //         plt::xlabel("X [m]");
    //         plt::ylabel("Y [m]");
    //         plt::title("Quadrotor Top View (X-Y Plane)");
    //         plt::grid(true);
            
    //         // Set axis limits with equal aspect ratio
    //         double plot_size = 2.0;
    //         plt::xlim(-plot_size, plot_size);
    //         plt::ylim(-plot_size, plot_size);
            
    //         // Add annotations for directions
    //         plt::text(1.8, 0.1, "Front");
    //         plt::text(-1.9, 0.1, "Back");
    //         plt::text(0.1, 1.8, "Right");
    //         plt::text(0.1, -1.8, "Left");

    //         // Add legend
    //         plt::legend();

    //         std::string filename = "../results/tests/quadrotor_frame_2d_" + std::to_string(i) + ".png";
    //         plt::save(filename);
            
    //         // Update plot
    //         plt::pause(animation_interval);
            
    //         // Compute next state
    //         state = quadrotor.getDiscreteDynamics(state, control);
    //     }
    // }
    
    // plt::show();
// }

// TEST(QuadrotorTest, StaticVisualization2D) {
//     const long fg = plt::figure();
//     double arm_length = 0.2;
//     double prop_radius = 0.05;
    
//     // Fixed position and zero orientation
//     Eigen::Vector3d position(0, 0, 1);
//     Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  // No rotation
    
//     // Transform quadrotor frame to world coordinates
//     auto frame_points = transformQuadrotorFrame(position, rotation, arm_length);
    
    // Plot settings
    // std::map<std::string, std::string> arm_front_back_keywords;
    // arm_front_back_keywords["color"] = "blue";
    // arm_front_back_keywords["linestyle"] = "-";
    // arm_front_back_keywords["linewidth"] = "2";
    // arm_front_back_keywords["label"] = "Front-Back Arm";

    // std::map<std::string, std::string> arm_right_left_keywords;
    // arm_right_left_keywords["color"] = "red";
    // arm_right_left_keywords["linestyle"] = "-";
    // arm_right_left_keywords["linewidth"] = "2";
    // arm_right_left_keywords["label"] = "Right-Left Arm";

    // std::map<std::string, std::string> prop_front_back_keywords;
    // prop_front_back_keywords["color"] = "blue";
    // prop_front_back_keywords["linestyle"] = "--";
    // prop_front_back_keywords["linewidth"] = "1";
    // prop_front_back_keywords["label"] = "Front-Back Propellers";

    // std::map<std::string, std::string> prop_right_left_keywords;
    // prop_right_left_keywords["color"] = "red";
    // prop_right_left_keywords["linestyle"] = "--";
    // prop_right_left_keywords["linewidth"] = "1";
    // prop_right_left_keywords["label"] = "Right-Left Propellers";
    
    // // Plot frame
    // // Front-Back arm (blue)
    // std::vector<double> front_back_x = {frame_points[0][0], frame_points[0][2]};
    // std::vector<double> front_back_y = {frame_points[1][0], frame_points[1][2]};
    // plt::plot(front_back_x, front_back_y, arm_front_back_keywords);

    // // Right-Left arm (red)
    // std::vector<double> right_left_x = {frame_points[0][1], frame_points[0][3]};
    // std::vector<double> right_left_y = {frame_points[1][1], frame_points[1][3]};
    // plt::plot(right_left_x, right_left_y, arm_right_left_keywords);

    // // Plot propeller circles
    // for (int j = 0; j < 4; ++j) {
    //     Eigen::Vector3d prop_center(
    //         frame_points[0][j],
    //         frame_points[1][j],
    //         frame_points[2][j]
    //     );
        
    //     std::vector<double> circle_x, circle_y;
    //     int num_points = 50;  // Increased points for smoother circles
    //     for (int k = 0; k <= num_points; ++k) {
    //         double angle = 2.0 * M_PI * k / num_points;
    //         Eigen::Vector2d point_body(
    //             prop_radius * cos(angle),
    //             prop_radius * sin(angle)
    //         );
            
    //         circle_x.push_back(prop_center.x() + point_body.x());
    //         circle_y.push_back(prop_center.y() + point_body.y());
    //     }
        
    //     // Use the appropriate keywords map for front/back vs right/left propellers
    //     if (j == 0 || j == 2) {  // Front or Back propeller
    //         plt::plot(circle_x, circle_y, prop_front_back_keywords);
    //     } else {  // Right or Left propeller
    //         plt::plot(circle_x, circle_y, prop_right_left_keywords);
    //     }
    // }
    
    // // Plot settings
    // plt::xlabel("X [m]");
    // plt::ylabel("Y [m]");
    // plt::title("Quadrotor Top View (X-Y Plane)");
    // plt::grid(true);
    
    // // Add text annotations for directions
    // plt::text(0.22, 0.0, "Front");
    // plt::text(-0.22, 0.0, "Back");
    // plt::text(0.0, 0.22, "Right");
    // plt::text(0.0, -0.22, "Left");
    
    // double plot_size = 0.4;  // Adjusted for better view
    // plt::xlim(-plot_size, plot_size);
    // plt::ylim(-plot_size, plot_size);

    // // Set equal aspect ratio
    // std::map<std::string, std::string> aspect_keywords;
    // aspect_keywords["adjustable"] = "box";
    // plt::axis("equal");

    // // Add legend
    // plt::legend();

    // std::string filename = "../results/tests/quadrotor_static_2d.png";
    // plt::save(filename);
    // plt::show();
// }

// Helper function to generate sphere points
std::vector<std::vector<std::vector<double>>> generateSpherePoints(
    double cx, double cy, double cz, double r, int n_points = 20) {
    std::vector<std::vector<std::vector<double>>> sphere(3, std::vector<std::vector<double>>(n_points));
    
    // Generate sphere surface points
    for (int i = 0; i < n_points; ++i) {
        sphere[0][i].resize(n_points);  // x coordinates
        sphere[1][i].resize(n_points);  // y coordinates
        sphere[2][i].resize(n_points);  // z coordinates
        
        double phi = M_PI * (double)i / (n_points - 1);
        
        for (int j = 0; j < n_points; ++j) {
            double theta = 2.0 * M_PI * (double)j / (n_points - 1);
            
            sphere[0][i][j] = cx + r * std::sin(phi) * std::cos(theta);
            sphere[1][i][j] = cy + r * std::sin(phi) * std::sin(theta);
            sphere[2][i][j] = cz + r * std::cos(phi);
        }
    }
    
    return sphere;
}

// TEST(QuadrotorTest, StaticVisualization) {
//     const long fg = plt::figure();
//     double arm_length = 0.2;
//     double prop_radius = 0.05;  // Radius of propeller spheres
    
//     // Fixed position and zero orientation
//     Eigen::Vector3d position(0, 0, 1);
//     Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  // No rotation
    
//     // Transform quadrotor frame to world coordinates
//     auto frame_points = transformQuadrotorFrame(position, rotation, arm_length);
    
//     // Plot settings
//     std::map<std::string, std::string> arm_front_back_keywords;
//     arm_front_back_keywords["color"] = "blue";
//     arm_front_back_keywords["linestyle"] = "-";
//     arm_front_back_keywords["linewidth"] = "5";

//     std::map<std::string, std::string> arm_right_left_keywords;
//     arm_right_left_keywords["color"] = "red";
//     arm_right_left_keywords["linestyle"] = "-";
//     arm_right_left_keywords["linewidth"] = "5";

//     // Plot frame
//     // Front-Back arm (blue)
//     std::vector<double> front_back_x = {frame_points[0][0], frame_points[0][2]};
//     std::vector<double> front_back_y = {frame_points[1][0], frame_points[1][2]};
//     std::vector<double> front_back_z = {frame_points[2][0], frame_points[2][2]};
//     plt::plot3(front_back_x, front_back_y, front_back_z, arm_front_back_keywords, fg);

//     // Right-Left arm (red)
//     std::vector<double> right_left_x = {frame_points[0][1], frame_points[0][3]};
//     std::vector<double> right_left_y = {frame_points[1][1], frame_points[1][3]};
//     std::vector<double> right_left_z = {frame_points[2][1], frame_points[2][3]};
//     plt::plot3(right_left_x, right_left_y, right_left_z, arm_right_left_keywords, fg);

//     // Create custom color map

//     // Plot propeller spheres
//     for (int j = 0; j < 4; ++j) {
//         auto sphere_points = generateSpherePoints(
//             frame_points[0][j],  // x center
//             frame_points[1][j],  // y center
//             frame_points[2][j],  // z center
//             prop_radius
//         );

//         // Surface properties
//         std::map<std::string, std::string> surf_keywords;
//           // Using Set1 colormap for solid colors
//         surf_keywords["vmin"] = "0";     // Fix color range
//         surf_keywords["vmax"] = "1";
//         surf_keywords["alpha"] = "0.99";

//         // Plot surface with color array for solid color
//         if (j == 0 || j == 2) {  // Front or Back propeller
//             surf_keywords["cmap"] = "Blues";  
//         } else {  // Right or Left propeller
//             surf_keywords["cmap"] = "Reds";  
//         }

//         plt::plot_surface(sphere_points[0], sphere_points[1], sphere_points[2], surf_keywords, fg);
//     }
    
//     // Plot settings
//     plt::xlabel("X [m]");
//     plt::ylabel("Y [m]");
//     plt::set_zlabel("Z [m]");
//     plt::title("Static Quadrotor View");
//     plt::grid(true);
    
//     // Set axis limits 
//     double plot_size = 0.3;
//     plt::xlim(-plot_size, plot_size);
//     plt::ylim(-plot_size, plot_size);
//     plt::zlim(position.z() - plot_size, position.z() + plot_size);

//     // Set view angle
//     plt::view_init(30, -60);

//     std::string filename = "../results/tests/quadrotor_static.png";
//     plt::save(filename);
//     plt::show();
// }

TEST(QuadrotorTest, Animation3D) {
    const long fg = plt::figure();
    double arm_length = 0.2;
    double prop_radius = 0.03;  // Radius of propeller spheres
    
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
               
    cddp::Quadrotor quadrotor(timestep, mass, inertia, arm_length, "rk4");

    // Initial state: slightly tilted hover
    Eigen::VectorXd state = Eigen::VectorXd::Zero(12);
    state << 0.0, 0.0, 1.0,           // position
            0.0, 0.0, 0.0,            // orientation 
            0.0, 0.0, 0.0,            // linear velocity
            0.0, 0.0, 0.0;            // angular velocity 

    // Hover thrust plus small variations
    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust * 1.001,    // Front (motor 0)
              hover_thrust * 1.001,     // Right (motor 1)
              hover_thrust * 1.0,     // Back (motor 2)
              hover_thrust * 1.0;     // Left (motor 3)

    // Simulation and animation parameters
    int num_steps = 300;
    double animation_interval = 0.02;
    
    // Storage for trajectory
    std::vector<double> x_traj, y_traj, z_traj;
    
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
    
    // Animation loop
    for (int i = 0; i < num_steps; ++i) {
        if (i % 5 == 0) {  // Update plot every 5 steps
            // Extract current position and orientation
            Eigen::Vector3d position(state[0], state[1], state[2]);
            Eigen::Matrix3d rotation = getRotationMatrix(state[3], state[4], state[5]);
            
            // Transform quadrotor frame to world coordinates
            auto frame_points = transformQuadrotorFrame(position, rotation, arm_length);
            
            // Store trajectory
            x_traj.push_back(position.x());
            y_traj.push_back(position.y());
            z_traj.push_back(position.z());

            // Clear previous plot
            plt::clf();
            
            // Plot frame
            // Front-Back arm (blue)
            std::vector<double> front_back_x = {frame_points[0][0], frame_points[0][2]};
            std::vector<double> front_back_y = {frame_points[1][0], frame_points[1][2]};
            std::vector<double> front_back_z = {frame_points[2][0], frame_points[2][2]};
            plt::plot3(front_back_x, front_back_y, front_back_z, arm_front_back_keywords, fg);

            // Right-Left arm (red)
            std::vector<double> right_left_x = {frame_points[0][1], frame_points[0][3]};
            std::vector<double> right_left_y = {frame_points[1][1], frame_points[1][3]};
            std::vector<double> right_left_z = {frame_points[2][1], frame_points[2][3]};
            plt::plot3(right_left_x, right_left_y, right_left_z, arm_right_left_keywords, fg);

            // Plot propeller spheres
            for (int j = 0; j < 4; ++j) {
                auto sphere_points = generateSpherePoints(
                    frame_points[0][j],  // x center
                    frame_points[1][j],  // y center
                    frame_points[2][j],  // z center
                    prop_radius
                );

                // Surface properties
                std::map<std::string, std::string> surf_keywords;
                // Using Set1 colormap for solid colors
                surf_keywords["vmin"] = "0";     // Fix color range
                surf_keywords["vmax"] = "1";
                surf_keywords["alpha"] = "0.99";

                // Plot surface with color array for solid color
                if (j == 0 || j == 2) {  // Front or Back propeller
                    surf_keywords["cmap"] = "Blues";  
                } else {  // Right or Left propeller
                    surf_keywords["cmap"] = "Reds";  
                }

                plt::plot_surface(sphere_points[0], sphere_points[1], sphere_points[2], surf_keywords, fg);
            }
            
            // Plot trajectory
            plt::plot3(x_traj, y_traj, z_traj, traj_keywords, fg);
            
            // Plot settings
            plt::xlabel("X [m]");
            plt::ylabel("Y [m]");
            plt::set_zlabel("Z [m]");
            plt::title("Quadrotor Animation");
            plt::grid(true);
            
            // Set axis limits
            double plot_size = 5;
            plt::xlim(-plot_size, plot_size);
            plt::ylim(-plot_size, plot_size);
            plt::zlim(-5, 5);  // Center around hover height

            // Set view angle
            plt::view_init(30, -60);

            std::string filename = "../results/tests/quadrotor_frame_" + 
                                 std::to_string(i/5) + ".png";
            plt::save(filename);
            
            // Update plot
            plt::pause(animation_interval);
            
        }
        
        // Compute next state
        state = quadrotor.getDiscreteDynamics(state, control);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
