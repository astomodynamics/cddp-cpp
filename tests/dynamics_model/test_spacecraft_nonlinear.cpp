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

#include "dynamics_model/spacecraft_nonlinear.hpp"
using namespace cddp;

TEST(SpacecraftNonlinearTest, DiscreteDynamics) {
    // Create a spacecraft instance
    double timestep = 0.01;  // 1s timestep
    double mu = 1.0;  // Earth's gravitational parameter (normalized)
    double mass = 1.0;  // 1 kg spacecraft
    std::string integration_type = "rk4";
    
    SpacecraftNonlinear spacecraft(timestep, integration_type, mass);

    // Store states for plotting
    std::vector<double> time_data, px_data, py_data, pz_data;
    std::vector<double> vx_data, vy_data, vz_data;
    std::vector<double> r0_data, theta_data;

    // Initial state: 
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state << -0.01127,  
            0.0,      
            0.1,    
            0.02,    
            0.02,      
            0.0,      
            0.9,  
            0.0,      
            0.0,      
            1.22838;  

    // No control input initially (free drift)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Simulate for several orbits
    int num_steps = 3000;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        px_data.push_back(state[0]);
        py_data.push_back(state[1]);
        pz_data.push_back(state[2]);
        vx_data.push_back(state[3]);
        vy_data.push_back(state[4]);
        vz_data.push_back(state[5]);
        r0_data.push_back(state[6]);
        theta_data.push_back(state[7]);

        // Compute the next state
        state = spacecraft.getDiscreteDynamics(state, control);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}