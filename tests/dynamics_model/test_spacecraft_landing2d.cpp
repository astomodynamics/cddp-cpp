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
using namespace cddp;

TEST(SpacecraftLanding2DTest, DiscreteDynamics) {
    double timestep = 0.1;
    std::string integration_type = "rk4";
    SpacecraftLanding2D spacecraft(timestep, integration_type);

    std::vector<double> time_data, x_data, y_data, theta_data, x_vel_data, y_vel_data, w_data;
    std::vector<double> energy_data;

    // Initial state: [x, x_dot, y, y_dot, theta, theta_dot]
    Eigen::VectorXd state(6);
    state << 1000.0,  // x position: 1000m
             0.0,     // x velocity: 0 m/s
             2000.0,  // y position: 2000m 
             -100.0,  // y velocity: -100 m/s (descending)
             0.1,     // theta: slight tilt
             0.0;     // theta_dot: no rotation

    // Control: [thrust_percent, thrust_angle]
    Eigen::VectorXd control(2);
    control << 0.5,   // 50% thrust
              0.0;    // 0 rad thrust angle

    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[2]);
        theta_data.push_back(state[4]);
        x_vel_data.push_back(state[1]);
        y_vel_data.push_back(state[3]);
        w_data.push_back(state[5]);
        
        // Calculate total energy (potential + kinetic)
        double potential = spacecraft.getMass() * spacecraft.getGravity() * state[2];  // mgh
        double kinetic = 0.5 * spacecraft.getMass()* (state[1]*state[1] + state[3]*state[3]) +  // translational
                        0.5 * spacecraft.getInertia() * state[5]*state[5];  // rotational
        
        energy_data.push_back(potential + kinetic);

        state = spacecraft.getDiscreteDynamics(state, control, 0.0);
    }

    // Basic assertions
    ASSERT_EQ(spacecraft.getStateDim(), 6);
    ASSERT_EQ(spacecraft.getControlDim(), 2);
    ASSERT_DOUBLE_EQ(spacecraft.getTimestep(), 0.1);
    ASSERT_EQ(spacecraft.getIntegrationType(), "rk4");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}