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

// Description: Test the Dubins car dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

// #include "cddp-cpp/dynamics_model/dubins_car.hpp"
// #include "cddp-cpp/matplotlibcpp.hpp"
#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(DubinsCarTest, DiscreteDynamics) {
    // Create a Dubins car instance
    double timestep = 0.1;
    std::string integration_type = "euler";
    cddp::DubinsCar dubins_car(timestep, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, theta_data;

    // Initial state and control (use Eigen vectors)
    Eigen::VectorXd state(3);
    state << 0.0, 0.0, 0.0;  // Initial position and orientation
    Eigen::VectorXd control(2);
    control << 1.0, 0.5; // Linear and angular velocity

    // Simulate for a few steps
    int num_steps = 50;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        theta_data.push_back(state[2]);

        // Compute the next state
        state = dubins_car.getDiscreteDynamics(state, control); 
    }

    // Create directory for saving plot (if it doesn't exist)
    const std::string plotDirectory = "../plots/test";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Plot the results
    plt::figure();
    plt::plot(x_data, y_data, {{"label", "Trajectory"}});
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Dubins Car Trajectory");
    plt::legend();
    plt::save(plotDirectory + "/dubins_car_trajectory.png");
    // plt::show();

    // Assert true if the dubins_car has the correct state dimension
    ASSERT_EQ(dubins_car.getStateDim(), 3);

    // Assert true if the dubins_car has the correct control dimension
    ASSERT_EQ(dubins_car.getControlDim(), 2);

    // Assert true if the dubins_car has the correct timestep
    ASSERT_DOUBLE_EQ(dubins_car.getTimestep(), 0.1);

    // Assert true if the dubins_car has the correct integration type
    ASSERT_EQ(dubins_car.getIntegrationType(), "euler"); 
}