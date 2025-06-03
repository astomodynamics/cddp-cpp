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
#include <chrono>
#include <thread>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/spacecraft_linear.hpp"
using namespace cddp;

TEST(HCWTest, DiscreteDynamics) {
    // Create an HCW instance
    double timestep = 1.0;  // 1s timestep
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6371e3 + 500e3, 3));  // For 500km orbit
    double mass = 1.0;  // 100 kg spacecraft
    std::string integration_type = "euler";
    
    cddp::HCW hcw(timestep, mean_motion, mass, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> vx_data, vy_data, vz_data;

    // Initial state: 
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = -37.59664132226163; 
    state(1) = 27.312455860666148;
    state(2) = 13.656227930333074;
    state(3) = 0.015161970413423813;
    state(4) = 0.08348413138390476;
    state(5) = 0.04174206569195238;

    // No control input initially (free drift)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Simulate for several orbits
    int num_steps = 6000;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        z_data.push_back(state[2]);
        vx_data.push_back(state[3]);
        vy_data.push_back(state[4]);
        vz_data.push_back(state[5]);

        // Compute the next state
        state = hcw.getDiscreteDynamics(state, control, 0.0);
    }

    // Basic assertions
    ASSERT_EQ(hcw.getStateDim(), 6);
    ASSERT_EQ(hcw.getControlDim(), 3);
    ASSERT_DOUBLE_EQ(hcw.getTimestep(), 1);
    ASSERT_EQ(hcw.getIntegrationType(), "euler");
}

TEST(HCWTest, ContinuousDynamics) {
    // Create HCW instance
    double timestep = 0.1;
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6.771e6, 3));
    double mass = 100.0;
    cddp::HCW hcw(timestep, mass, mean_motion, "euler");

    // Test hover condition (stationary relative position)
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = 100.0;  // 100m radial offset

    // Calculate required thrust for radial hover
    double hover_thrust = mass * 3.0 * mean_motion * mean_motion * state(0);
    Eigen::VectorXd control(3);
    control << hover_thrust, 0.0, 0.0;  // Radial thrust only

    // Get dynamics
    Eigen::VectorXd state_dot = hcw.getContinuousDynamics(state, control, 0.0);

}

TEST(HCWTest, RelativeTrajectory) {
    // Create HCW instance
    double timestep = 1.0;
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6371e3 + 500e3, 3));  // For 500km orbit
    double mass = 1.0;
    cddp::HCW hcw(timestep, mean_motion, mass, "euler");

    // Initial conditions
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = -37.59664132226163; 
    state(1) = 27.312455860666148;
    state(2) = 13.656227930333074;
    state(3) = 0.015161970413423813;
    state(4) = 0.08348413138390476;
    state(5) = 0.04174206569195238;

    // No control input (natural motion)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Store trajectory points
    std::vector<double> x_data, y_data, z_data;
}

// Helper function to create spacecraft marker coordinates
std::vector<std::vector<double>> createSpacecraftMarker(
    const Eigen::Vector3d& position,
    double size = 1.0) {
    
    std::vector<std::vector<double>> marker(3, std::vector<double>());
    
    // Simple cube-like spacecraft shape
    std::vector<double> dx = {-1, 1, 1, -1, -1, -1, 1, 1, -1};
    std::vector<double> dy = {-1, -1, 1, 1, -1, -1, -1, 1, 1};
    std::vector<double> dz = {-1, -1, -1, -1, -1, 1, 1, 1, 1};
    
    for (size_t i = 0; i < dx.size(); ++i) {
        marker[0].push_back(position.x() + size * dx[i]);
        marker[1].push_back(position.y() + size * dy[i]);
        marker[2].push_back(position.z() + size * dz[i]);
    }
    
    return marker;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}