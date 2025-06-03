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
    Eigen::VectorXd state_dot = manipulator.getContinuousDynamics(state, zero_control, 0.0);

    // Check that gravity causes downward acceleration
    EXPECT_GT(std::abs(state_dot(4)), 0.0);  // Should have non-zero acceleration due to gravity
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
