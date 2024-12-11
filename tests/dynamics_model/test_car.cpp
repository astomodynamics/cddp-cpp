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
// Description: Test the car dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/car.hpp"
#include "matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(CarTest, DiscreteDynamics) {
    // Create a car instance
    double timestep = 0.03;  // From original MATLAB code
    double wheelbase = 2.0;  // From original MATLAB code
    std::string integration_type = "euler";
    cddp::Car car(timestep, wheelbase, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, theta_data, v_data;

    // Initial state and control (from MATLAB demo)
    Eigen::VectorXd state(4);
    state << 1.0, 1.0, 3*M_PI/2, 0.0;  // Initial state from MATLAB demo
    Eigen::VectorXd control(2);
    control << 0.1, 0.1; // Small steering angle and acceleration

    // Simulate for a few steps
    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        theta_data.push_back(state[2]);
        v_data.push_back(state[3]);

        // Compute the next state
        state = car.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(car.getStateDim(), 4);
    ASSERT_EQ(car.getControlDim(), 2);
    ASSERT_DOUBLE_EQ(car.getTimestep(), 0.03);
    ASSERT_EQ(car.getIntegrationType(), "euler");

    // // Create directory for saving plot (if it doesn't exist)
    // const std::string plotDirectory = "../plots/test";
    // if (!fs::exists(plotDirectory)) {
    //     fs::create_directory(plotDirectory);
    // }

    // // Plot the results
    // // Plot trajectory in X-Y plane
    // plt::figure(1);
    // plt::plot(x_data, y_data, {{"label", "Trajectory"}});
    // plt::xlabel("X Position");
    // plt::ylabel("Y Position");
    // plt::legend();
    // plt::save(plotDirectory + "/car_trajectory.png");

    // // Plot states over time
    // plt::figure(2);
    // plt::plot(time_data, theta_data, {{"label", "Heading"}});
    // plt::plot(time_data, v_data, {{"label", "Velocity"}});
    // plt::xlabel("Time");
    // plt::ylabel("State");
    // plt::legend();
    // plt::save(plotDirectory + "/car_states.png");
    // plt::show();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}