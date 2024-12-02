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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/manipulator.hpp"
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;


TEST(ManipulatorTest, ForwardKinematics) {
    // Create manipulator instance
    double timestep = 0.01;
    cddp::Manipulator manipulator(timestep, "rk4");

    // Set up initial state
    Eigen::VectorXd state = Eigen::VectorXd::Zero(3);

    // Test case 1: 
    state(0) = 0.0;              // theta1 = 0
    state(1) = -M_PI/2;          // theta2 = -90 deg
    state(2) = M_PI/2;           // theta3 = 90 deg
    
    auto T = manipulator.getForwardKinematics(state);
    auto pos = manipulator.getEndEffectorPosition(state);

    // Check values
    EXPECT_NEAR(pos(0), 1.0, 1e-6);
    EXPECT_NEAR(pos(1), 0.2, 1e-6);
    EXPECT_NEAR(pos(2), 1.0, 1e-6); 

    // Test case 2:
    state(0) = M_PI/4;           // theta1 = 45 deg
    state(1) = -M_PI/3;          // theta2 = -60 deg
    state(2) = M_PI * 2 / 3;     // theta3 = 120 deg

    T = manipulator.getForwardKinematics(state);
    pos = manipulator.getEndEffectorPosition(state);

    // Check values 
    EXPECT_NEAR(pos(0), 0.5657, 1e-4);
    EXPECT_NEAR(pos(1), 0.8485, 1e-4);
    EXPECT_NEAR(pos(2), 0.0, 1e-6);

    // Test case 3:
    state(0) = M_PI/2;          // theta1 = 90 deg
    state(1) = M_PI/4;          // theta2 = 45 deg
    state(2) = M_PI/2;          // theta3 = 90 deg

    T = manipulator.getForwardKinematics(state);
    pos = manipulator.getEndEffectorPosition(state);

    // Check values 
    EXPECT_NEAR(pos(0), -0.2, 1e-4);
    EXPECT_NEAR(pos(1), 0.0, 1e-4);
    EXPECT_NEAR(pos(2), -1.4142, 1e-4);
}

TEST(ManipulatorTest, Dynamics) {
    // Create manipulator instance
    double timestep = 0.01;
    cddp::Manipulator manipulator(timestep, "rk4");

    // Test gravity compensation
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(1) = M_PI/4;  // 45 degree angle for second joint
    
    // Compute gravity compensation torques
    Eigen::VectorXd zero_control = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd state_dot = manipulator.getContinuousDynamics(state, zero_control);

    // Check that gravity causes downward acceleration
    EXPECT_GT(std::abs(state_dot(4)), 0.0);  // Should have non-zero acceleration due to gravity
}

TEST(ManipulatorTest, Visualization) {
    // Create manipulator instance
    double timestep = 0.01;
    cddp::Manipulator manipulator(timestep, "rk4");

    // Set up initial state 
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = 0.0;              // theta1 = 0
    state(1) = -M_PI/2;          // theta2 = -90 deg
    state(2) = M_PI/2;           // theta3 = 90 deg

    // Get forward kinematics
    Eigen::Matrix4d T = manipulator.getForwardKinematics(state);
    
    // Get intermediate transformations
    auto transforms = manipulator.getTransformationMatrices(state(0), state(1), state(2));
    Eigen::Matrix4d T01 = transforms[0];
    Eigen::Matrix4d T02 = T01 * transforms[1];
    Eigen::Matrix4d T03 = T02 * transforms[2];
    Eigen::Matrix4d T04 = T03 * transforms[3];

    // Extract joint positions for plotting
    std::vector<double> x = {0.0};  // Base position
    std::vector<double> y = {0.0};
    std::vector<double> z = {0.0};

    // Add joint positions
    x.push_back(T02(0,3));  // First link end
    y.push_back(T02(1,3));
    z.push_back(T02(2,3));

    x.push_back(T03(0,3));  // Second link end
    y.push_back(T03(1,3));
    z.push_back(T03(2,3));

    x.push_back(T04(0,3));  // End effector
    y.push_back(T04(1,3));
    z.push_back(T04(2,3));

    // // Plot manipulator
    // const long fg = plt::figure();
    // plt::figure_size(800, 600);

    // // Plot base
    // std::vector<double> base_x = {0.1, -0.1, -0.1, 0.1, 0.1};
    // std::vector<double> base_y = {0.1, 0.1, -0.1, -0.1, 0.1};
    // std::vector<double> base_z(5, 0.0);
    // std::map<std::string, std::string> base_keywords;
    // base_keywords["color"] = "black";
    // plt::plot3(base_x, base_y, base_z, base_keywords, fg);

    // // Plot manipulator links
    // std::map<std::string, std::string> link_keywords;
    // link_keywords["color"] = "blue";
    // link_keywords["marker"] = "o";
    // plt::plot3(x, y, z, link_keywords, fg);

    // // Plot end-effector
    // std::vector<double> ee_x = {x.back()};
    // std::vector<double> ee_y = {y.back()};
    // std::vector<double> ee_z = {z.back()};
    // std::map<std::string, std::string> ee_keywords;
    // ee_keywords["color"] = "red";
    // ee_keywords["marker"] = "o";
    // ee_keywords["markersize"] = "10";
    // plt::plot3(ee_x, ee_y, ee_z, ee_keywords, fg);

    // // Set plot properties
    // plt::xlabel("X [m]");
    // plt::ylabel("Y [m]");
    // plt::set_zlabel("Z [m]");
    // plt::title("Simplified PUMA Manipulator");
    // plt::grid(true);

    // // Set consistent axis limits
    // double plot_range = 2.5;
    // plt::xlim(-plot_range, plot_range);
    // plt::ylim(-plot_range, plot_range);
    // plt::zlim(-plot_range, plot_range);

    // plt::view_init(30, -60);
    
    // // Save plot
    // std::string plotDirectory = "../results/tests";
    // plt::save(plotDirectory + "/manipulator_pose.png");
    // plt::show();
    // plt::clf();
}

TEST(ManipulatorTest, Animation) {
    // Create manipulator instance
    double timestep = 0.01;
    cddp::Manipulator manipulator(timestep, "rk4");

    // Animation parameters
    int num_frames = 100;
    std::string plotDirectory = "../results/tests";

    // const long fg = plt::figure();

    // // Generate trajectory
    // for (int frame = 0; frame < num_frames; ++frame) {
    //     // Clear previous frame
    //     plt::clf();

    //     // Compute joint angles - example motion
    //     double t = static_cast<double>(frame) / num_frames;
    //     Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    //     state(0) = 0.0;                             // Fixed base
    //     state(1) = -M_PI/2 + M_PI/4 * sin(M_PI*t); // Second joint oscillation
    //     state(2) = M_PI/2 * cos(M_PI*t);           // Third joint oscillation

    //     // Get transformations
    //     auto transforms = manipulator.getTransformationMatrices(state(0), state(1), state(2));
    //     Eigen::Matrix4d T01 = transforms[0];
    //     Eigen::Matrix4d T02 = T01 * transforms[1];
    //     Eigen::Matrix4d T03 = T02 * transforms[2];
    //     Eigen::Matrix4d T04 = T03 * transforms[3];

    //     // Get end-effector position from forward kinematics
    //     Eigen::Vector4d r3;  // End-point wrt Frame 3
    //     r3 << manipulator.getLinkLength('c'), 0, manipulator.getLinkLength('b'), 1;
    //     Eigen::Vector4d r0 = T03 * r3;  // Position of end-effector

    //     // Get elbow position
    //     Eigen::Vector4d rm;  // Intermediate point between O3 and O4
    //     rm = T03 * Eigen::Vector4d(0, 0, manipulator.getLinkLength('b'), 1);

    //     // Plot base square
    //     std::vector<double> base_x = {0.1, -0.1, -0.1, 0.1, 0.1};
    //     std::vector<double> base_y = {0.1, 0.1, -0.1, -0.1, 0.1};
    //     std::vector<double> base_z(5, 0.0);
    //     std::map<std::string, std::string> base_keywords;
    //     base_keywords["color"] = "green";
    //     plt::plot3(base_x, base_y, base_z, base_keywords, fg);

    //     // Plot manipulator links
    //     std::vector<double> x = {0, T03(0,3), rm(0), r0(0)};
    //     std::vector<double> y = {0, T03(1,3), rm(1), r0(1)};
    //     std::vector<double> z = {0, T03(2,3), rm(2), r0(2)};
    //     std::map<std::string, std::string> link_keywords;
    //     link_keywords["color"] = "blue";
    //     link_keywords["linewidth"] = "2";
    //     plt::plot3(x, y, z, link_keywords, fg);

    //     // Plot elbow point
    //     std::vector<double> elbow_x = {rm(0)};
    //     std::vector<double> elbow_y = {rm(1)};
    //     std::vector<double> elbow_z = {rm(2)};
    //     std::map<std::string, std::string> elbow_keywords;
    //     elbow_keywords["color"] = "red";
    //     elbow_keywords["marker"] = "v";
    //     plt::plot3(elbow_x, elbow_y, elbow_z, elbow_keywords, fg);

    //     // Plot end-effector
    //     std::vector<double> ee_x = {r0(0)};
    //     std::vector<double> ee_y = {r0(1)};
    //     std::vector<double> ee_z = {r0(2)};
    //     std::map<std::string, std::string> ee_keywords;
    //     ee_keywords["color"] = "red";
    //     ee_keywords["marker"] = "o";
    //     plt::plot3(ee_x, ee_y, ee_z, ee_keywords, fg);

    //     // Set plot properties
    //     plt::xlabel("X [m]");
    //     plt::ylabel("Y [m]");
    //     plt::set_zlabel("Z [m]");
    //     plt::title("PUMA-like Manipulator Animation");

    //     // Set consistent axis limits
    //     plt::xlim(-1, 1);
    //     plt::ylim(-1, 1);
    //     plt::zlim(-0.5, 1.5);

    //     plt::grid(true);
    //     plt::view_init(30, -60);  // Match MATLAB view angle

    //     // Save frame
    //     plt::save(plotDirectory + "/manipulator_frame_" + std::to_string(frame) + ".png");
    //     plt::pause(0.01);  // Small delay between frames
    // }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
