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
               
    std::string integration_type = "euler";
    cddp::Quadrotor quadrotor(timestep, mass, inertia, arm_length, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> phi_data, theta_data, psi_data;

    // Initial state: hovering with slight initial rotation
    Eigen::VectorXd state = Eigen::VectorXd::Zero(12);
    state(2) = 1.0;  // Start at 1m height
    state(3) = 0.1;  // Small initial roll angle

    // Control input for hover (each motor provides mg/4 force)
    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Simulate for a few steps
    int num_steps = 200;
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
    ASSERT_EQ(quadrotor.getIntegrationType(), "euler");

    // Test if quadrotor maintains approximate hover height
    double final_height = z_data.back();
    EXPECT_NEAR(final_height, 1.0, 0.1);
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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
