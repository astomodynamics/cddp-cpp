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

// Description: Test the Pendulum dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(PendulumTest, DiscreteDynamics) {
    // Create a Pendulum instance
    double timestep = 0.01;
    std::string integration_type = "rk4";
    cddp::Pendulum pendulum(timestep, integration_type);

    // Store states for plotting
    std::vector<double> time_data, theta_data, theta_dot_data;

    // Initial state and control
    Eigen::VectorXd state(2);
    state << M_PI/4, 0.0;  // Initial angle (45 degrees) and angular velocity
    Eigen::VectorXd control(1);
    control << 0.0;  // No initial torque

    // Simulate for a few steps
    int num_steps = 500;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        theta_data.push_back(state[0]);
        theta_dot_data.push_back(state[1]);

        // Compute the next state
        state = pendulum.getDiscreteDynamics(state, control); 
    }

    // Assert true if the pendulum has the correct state dimension
    ASSERT_EQ(pendulum.getStateDim(), 2);

    // Assert true if the pendulum has the correct control dimension
    ASSERT_EQ(pendulum.getControlDim(), 1);

    // Assert true if the pendulum has the correct timestep
    ASSERT_DOUBLE_EQ(pendulum.getTimestep(), 0.01);

    // Assert true if the pendulum has the correct integration type
    ASSERT_EQ(pendulum.getIntegrationType(), "rk4");

    // Verify that energy decreases due to damping
    double initial_energy = 9.81 * (1.0 - std::cos(M_PI/4));  // mgl(1-cos(theta))
    double final_energy = 9.81 * (1.0 - std::cos(theta_data.back()));
    ASSERT_LT(final_energy, initial_energy);

    // // Plot the results
    // plt::figure();
    // plt::subplot(2, 1, 1);
    // plt::plot(time_data, theta_data);
    // plt::xlabel("Time (s)");
    // plt::ylabel("Angle (rad)");
    // plt::title("Pendulum Angle vs. Time");

    // plt::subplot(2, 1, 2);
    // plt::plot(time_data, theta_dot_data);
    // plt::xlabel("Time (s)");
    // plt::ylabel("Angular Velocity (rad/s)");
    // plt::title("Pendulum Angular Velocity vs. Time");

    // plt::show();
}