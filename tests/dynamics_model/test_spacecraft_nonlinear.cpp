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
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
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

    // Basic assertions
    // ASSERT_EQ(spacecraft.getStateDim(), 10);
    // ASSERT_EQ(spacecraft.getControlDim(), 3);
    // ASSERT_DOUBLE_EQ(spacecraft.getTimestep(), 1.0);
    // ASSERT_EQ(spacecraft.getIntegrationType(), "rk4");

    // // Create directory if it doesn't exist
    // const std::string plotDirectory = "../results/tests";
    // if (!fs::exists(plotDirectory)) {
    //     fs::create_directory(plotDirectory);
    // }

    // // Create a single figure with 2x2 subplots
    // plt::figure_size(1200, 800);

    // // XY plane
    // plt::subplot(2, 2, 1);
    // std::map<std::string, std::string> blue_style = {{"color", "blue"}};
    // plt::plot(px_data, py_data, blue_style);
    // plt::title("X-Y Plane");
    // plt::xlabel("X [m]");
    // plt::ylabel("Y [m]");
    // plt::grid(true);

    // // Find axis limits for consistent scaling
    // double max_range = 0.125;
    // plt::xlim(-max_range, max_range);
    // plt::ylim(-max_range, max_range);

    // // XZ plane
    // plt::subplot(2, 2, 2);
    // std::map<std::string, std::string> red_style = {{"color", "red"}};
    // plt::plot(px_data, pz_data, red_style);
    // plt::title("X-Z Plane");
    // plt::xlabel("X [m]");
    // plt::ylabel("Z [m]");
    // plt::grid(true);
    // plt::xlim(-max_range, max_range);
    // plt::ylim(-max_range, max_range);

    // // YZ plane
    // plt::subplot(2, 2, 3);
    // std::map<std::string, std::string> green_style = {{"color", "green"}};
    // plt::plot(py_data, pz_data, green_style);
    // plt::title("Y-Z Plane");
    // plt::xlabel("Y [m]");
    // plt::ylabel("Z [m]");
    // plt::grid(true);
    // plt::xlim(-max_range, max_range);
    // plt::ylim(-max_range, max_range);

    // 3D plot
    // plt::subplot(2, 2, 4);
    // std::map<std::string, std::string> black_style = {{"color", "black"}};
    // plt::plot3(px_data, py_data, pz_data, black_style);
    // plt::title("3D Trajectory");
    // plt::xlabel("X [m]");
    // plt::ylabel("Y [m]");
    // plt::set_zlabel("Z [m]");
    // plt::grid(true);

    // // Set consistent 3D axis limits
    // plt::xlim(-max_range, max_range);
    // plt::ylim(-max_range, max_range);
    // plt::zlim(-max_range, max_range);

    // Add view angle for 3D plot 
    // plt::view_init(30, 45);

    // plt::tight_layout();
    // plt::save(plotDirectory + "/spacecraft_nonlinear_projections.png");
    // plt::close();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}