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

// Description: Test the DubinsCar dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp" // Adjust if you have a different header for including DubinsCar

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(DubinsCarTest, DiscreteDynamics) {
    // Create a DubinsCar instance
    double speed = 1.0;           // Constant forward speed
    double timestep = 0.1;        // Integration timestep
    std::string integration_type = "euler";
    DubinsCar dubins_car(speed, timestep, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, theta_data;

    // Initial state and control (use Eigen vectors)
    // State = [x, y, theta]
    Eigen::VectorXd state(3);
    state << 0.0, 0.0, 0.0;   // Initial position/orientation

    // Control = [omega]
    Eigen::VectorXd control(1);
    control << 0.5;           // Turn rate

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

    const std::string plotDirectory = "../plots/test";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Optional: Plot results with matplotlibcpp (commented out by default)
    // plt::figure();
    // plt::plot(x_data, y_data, {{"label", "Trajectory"}});
    // plt::xlabel("X");
    // plt::ylabel("Y");
    // plt::title("DubinsCar Trajectory");
    // plt::legend();
    // plt::save(plotDirectory + "/dubins_car_trajectory.png");
    // plt::show();

    // Check state dimension
    ASSERT_EQ(dubins_car.getStateDim(), 3);

    // Check control dimension (should be 1 for DubinsCar with a single steering input)
    ASSERT_EQ(dubins_car.getControlDim(), 1);

    // Check timestep
    ASSERT_DOUBLE_EQ(dubins_car.getTimestep(), 0.1);

    // Check integration type
    ASSERT_EQ(dubins_car.getIntegrationType(), "euler");
}
