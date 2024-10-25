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
// Description: Test the pendulum dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/dynamics_model/pendulum.hpp" // Assuming you have the Eigen-based Pendulum class
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(PendulumTest, DiscreteDynamics) {
    // Create a pendulum instance (no device needed for Eigen)
    double mass = 1.0; 
    double length = 1.0; 
    double gravity = 9.81;
    double timestep = 0.05;
    std::string integration_type = "euler";
    cddp::Pendulum pendulum(mass, length, gravity, timestep, integration_type); 

    // Store states for plotting
    std::vector<double> time_data, theta_data, theta_dot_data;

    // Initial state and control (use Eigen vectors)
    Eigen::VectorXd state(2);
    state << 0.1, 0.0;  // Start at a small angle, zero velocity
    Eigen::VectorXd control(1);
    control << 0.0; // No torque initially

    // Simulate for a few steps
    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        theta_data.push_back(state[0]);
        theta_dot_data.push_back(state[1]);

        // Compute the next state
        state = pendulum.getDiscreteDynamics(state, control); 
    }

    // Create directory for saving plot (if it doesn't exist)
    const std::string plotDirectory = "../plots/test";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Plot the results (same as before)
    plt::figure();
    plt::plot(time_data, theta_data, {{"label", "Angle"}});
    plt::plot(time_data, theta_dot_data, {{"label", "Angular Velocity"}});
    plt::xlabel("Time");
    plt::ylabel("State");
    plt::legend();
    plt::save(plotDirectory + "/pendulum_dynamics.png");
    // plt::show();

    // Assert true if the pendulum has the correct state dimension
    ASSERT_EQ(pendulum.getStateDim(), 2);

    // Assert true if the pendulum has the correct control dimension
    ASSERT_EQ(pendulum.getControlDim(), 1);

    // Assert true if the pendulum has the correct timestep
    ASSERT_DOUBLE_EQ(pendulum.getTimestep(), 0.05);

    // Assert true if the pendulum has the correct integration type
    ASSERT_EQ(pendulum.getIntegrationType(), "euler"); 
}