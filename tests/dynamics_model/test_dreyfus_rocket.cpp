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
#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(DreyfusRocketTest, DiscreteDynamics) {
    double timestep = 0.05;
    std::string integration_type = "rk4";
    DreyfusRocket rocket(timestep, integration_type);

    std::vector<double> time_data, height_data, velocity_data, energy_data;

    // Initial state: height = 0, velocity = 0
    Eigen::VectorXd state(2);
    state << 0.0, 0.0;
    
    // Initial control: 45 degree thrust angle
    Eigen::VectorXd control(1);
    control << M_PI/4.0;

    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        time_data.push_back(i * timestep);
        height_data.push_back(state[0]);
        velocity_data.push_back(state[1]);
        
        // Calculate total energy (potential + kinetic)
        double potential = rocket.getGravityAcceleration() * state[0];
        double kinetic = 0.5 * state[1] * state[1];
        energy_data.push_back(potential + kinetic);

        state = rocket.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(rocket.getStateDim(), 2);
    ASSERT_EQ(rocket.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(rocket.getTimestep(), 0.05);
    ASSERT_EQ(rocket.getIntegrationType(), "rk4");

    // Test force magnitudes
    EXPECT_DOUBLE_EQ(rocket.getThrustAcceleration(), 64.0);
    EXPECT_DOUBLE_EQ(rocket.getGravityAcceleration(), 32.0);

    // Verify motion under 45-degree thrust
    EXPECT_GT(height_data.back(), height_data.front()); 
    EXPECT_GT(velocity_data.back(), 0.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}