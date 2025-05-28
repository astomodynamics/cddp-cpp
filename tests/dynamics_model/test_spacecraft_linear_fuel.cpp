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

#include "dynamics_model/spacecraft_linear_fuel.hpp"
#include "cddp.hpp"
using namespace cddp;
using namespace matplot;
namespace plt = matplot;

TEST(SpacecraftLinearFuelTest, DiscreteDynamics) {
    // Create an SpacecraftLinearFuel instance
    double timestep = 1.0;  // 1s timestep
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6371e3 + 500e3, 3));  // For 500km orbit
    double isp = 300.0; // Specific impulse
    double g0 = 9.81; // Standard gravity
    std::string integration_type = "euler";
    
    cddp::SpacecraftLinearFuel model(timestep, mean_motion, isp, g0, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> vx_data, vy_data, vz_data, mass_data, acc_control_data;

    // Initial state: (x, y, z, vx, vy, vz, mass, accumulated_control_effort)
    Eigen::VectorXd state = Eigen::VectorXd::Zero(8);
    state(0) = -37.59664132226163; 
    state(1) = 27.312455860666148;
    state(2) = 13.656227930333074;
    state(3) = 0.015161970413423813;
    state(4) = 0.08348413138390476;
    state(5) = 0.04174206569195238;
    state(6) = 100.0; // Initial mass
    state(7) = 0.0; // Initial accumulated control effort

    // No control input initially (free drift)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Simulate for several orbits
    int num_steps = 6000;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state(0));
        y_data.push_back(state(1));
        z_data.push_back(state(2));
        vx_data.push_back(state(3));
        vy_data.push_back(state(4));
        vz_data.push_back(state(5));
        mass_data.push_back(state(6));
        acc_control_data.push_back(state(7));

        // Compute the next state
        state = model.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(model.getStateDim(), 8);
    ASSERT_EQ(model.getControlDim(), 3);
    ASSERT_DOUBLE_EQ(model.getTimestep(), 1);
    ASSERT_EQ(model.getIntegrationType(), "euler");
}

TEST(SpacecraftLinearFuelTest, RelativeTrajectory) {
    // Create SpacecraftLinearFuel instance
    double timestep = 1.0;
    int num_steps = 6000;
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6371e3 + 500e3, 3));  // For 500km orbit
    double isp = 300.0; // Specific impulse
    double g0 = 9.81; // Standard gravity
    cddp::SpacecraftLinearFuel model(timestep, mean_motion, isp, g0, "euler");


    // Initial conditions
    // Eigen::VectorXd state = Eigen::VectorXd::Zero(8);
    // state(0) = -37.59664132226163; 
    // state(1) = 27.312455860666148;
    // state(2) = 13.656227930333074;
    // state(3) = 0.015161970413423813;
    // state(4) = 0.08348413138390476;
    // state(5) = 0.04174206569195238;
    // state(6) = 100.0; // Initial mass
    // state(7) = 0.0; // Initial accumulated control effort

    Eigen::VectorXd state = Eigen::VectorXd::Zero(8);
    state(0) = 50.0; 
    state(1) = 0.0;
    state(2) = 0.0;
    state(3) = 0.0;
    state(4) = -2*mean_motion*state(0);
    state(5) = 0.0;
    state(6) = 100.0; // Initial mass
    state(7) = 0.0; // Initial accumulated control effort

    // No control input (natural motion)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    std::vector<Eigen::VectorXd> states(num_steps + 1);
    states[0] = state;

    // Simulate dynamics
    for (int i = 0; i < num_steps; ++i) {
        states[i + 1] = model.getDiscreteDynamics(states[i], control);
    }

    // Store trajectory points
    std::vector<double> x_data, y_data, z_data, vx_data, vy_data, vz_data, mass_data, control_x_data, control_y_data, control_z_data;
    for (int i = 0; i < num_steps + 1; ++i) {
        x_data.push_back(states[i](0));
        y_data.push_back(states[i](1));
        z_data.push_back(states[i](2));
        vx_data.push_back(states[i](3));
        vy_data.push_back(states[i](4));
        vz_data.push_back(states[i](5));
        mass_data.push_back(states[i](6));
    }

    // Plot 3d trajectory
    // plt::figure();
    // plt::plot3(x_data, y_data, z_data, "r-");
    // plt::show(); 
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