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
        Eigen::Vector3d(arm_length, 0, 0),   // Front
        Eigen::Vector3d(-arm_length, 0, 0),  // Back
        Eigen::Vector3d(0, arm_length, 0),   // Right
        Eigen::Vector3d(0, -arm_length, 0)   // Left
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

// TEST(QuadrotorTest, Animation) {
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
//     control << hover_thrust * 1.1,    // Front
//               hover_thrust * 0.9,     // Right
//               hover_thrust * 1.0,     // Back
//               hover_thrust * 1.0;     // Left

//     // Simulation and animation
//     int num_steps = 1000;
//     double animation_interval = 0.02;  // seconds
    
//     // Storage for trajectory
//     std::vector<double> x_traj, y_traj, z_traj;
    
//     // Plot settings
//     std::map<std::string, std::string> traj_keywords;
//     traj_keywords["label"] = "Trajectory";
//     traj_keywords["linestyle"] = ":";
//     traj_keywords["color"] = "black";

//     std::map<std::string, std::string> arm1_keywords;
//     arm1_keywords["color"] = "blue";
//     arm1_keywords["linestyle"] = "-";
//     arm1_keywords["linewidth"] = "2";

//     std::map<std::string, std::string> arm2_keywords;
//     arm2_keywords["color"] = "red";
//     arm2_keywords["linestyle"] = "-";
//     arm2_keywords["linewidth"] = "2";

//     // Create figure once
//     // plt::figure();
//     // // plt::grid(true);
//     // double plot_size = 2.0;
//     // plt::xlim(-plot_size, plot_size);
//     // plt::ylim(-plot_size, plot_size);
//     // plt::xlabel("X");
//     // plt::ylabel("Y");
//     // plt::set_zlabel("Z");
    
//     for (int i = 0; i < num_steps; ++i) {
//         if  (i % 5 == 0) {
//             // Extract current position and orientation
//             Eigen::Vector3d position(state[0], state[1], state[2]);
//             Eigen::Matrix3d rotation = getRotationMatrix(state[3], state[4], state[5]);
            
//             // Transform quadrotor frame to world coordinates
//             auto frame_points = transformQuadrotorFrame(position, rotation, arm_length);
            
//             // Store trajectory
//             x_traj.push_back(position.x());
//             y_traj.push_back(position.y());
//             z_traj.push_back(position.z());

//             // Clear previous plot
//             plt::clf();
            
//             // Plot quadrotor frame
//             // Front-Back arm (blue)
//             std::vector<double> arm1_x = {frame_points[0][0], frame_points[0][1]};
//             std::vector<double> arm1_y = {frame_points[1][0], frame_points[1][1]};
//             std::vector<double> arm1_z = {frame_points[2][0], frame_points[2][1]};
//             plt::plot3(arm1_x, arm1_y, arm1_z, arm1_keywords, fg);

//             // Right-Left arm (red)
//             std::vector<double> arm2_x = {frame_points[0][2], frame_points[0][3]};
//             std::vector<double> arm2_y = {frame_points[1][2], frame_points[1][3]};
//             std::vector<double> arm2_z = {frame_points[2][2], frame_points[2][3]};
//             plt::plot3(arm2_x, arm2_y, arm2_z, arm2_keywords, fg);
            
//             // // Plot trajectory
//             plt::plot3(x_traj, y_traj, z_traj, traj_keywords, fg);
            
//             // Plot settings
//             plt::xlabel("X");
//             plt::ylabel("Y");
//             // plt::set_zlabel("Z");
//             plt::grid(true);
            
//             // Set axis limits
//             double plot_size = 4.0;
//             plt::xlim(-plot_size, plot_size);
//             plt::ylim(-plot_size, plot_size);
//             // plt::zlim(0.0, 2.0);

//             std::string filename = "../results/tests/quadrotor_frame_" + std::to_string(i) + ".png";
//             plt::save(filename);
            
//             // Update plot
//             plt::pause(animation_interval);
            
//             // Compute next state
//             state = quadrotor.getDiscreteDynamics(state, control);
//         }
//     }
    
//     // plt::show();
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
