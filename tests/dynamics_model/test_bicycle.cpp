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
// Description: Test the bicycle dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/bicycle.hpp"
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(BicycleTest, DiscreteDynamics) {
    // Create a bicycle instance
    double timestep = 0.05;
    double wheelbase = 2.0; // typical car wheelbase in meters
    std::string integration_type = "euler";
    cddp::Bicycle bicycle(timestep, wheelbase, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, theta_data, v_data;

    // Initial state and control
    Eigen::VectorXd state(4);
    state << 0.0, 0.0, M_PI/4, 1.0;  // Start at origin, 45 degrees heading, 1 m/s velocity
    Eigen::VectorXd control(2);
    control << 0.0, 0.1; // Small acceleration and steering angle

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
        state = bicycle.getDiscreteDynamics(state, control);
    }

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
    // plt::save(plotDirectory + "/bicycle_trajectory.png");

    // // Plot states over time
    // plt::figure(2);
    // plt::plot(time_data, theta_data, {{"label", "Heading"}});
    // plt::plot(time_data, v_data, {{"label", "Velocity"}});
    // plt::xlabel("Time");
    // plt::ylabel("State");
    // plt::legend();
    // plt::save(plotDirectory + "/bicycle_states.png");
    // plt::show();

    // Basic assertions
    ASSERT_EQ(bicycle.getStateDim(), 4);
    ASSERT_EQ(bicycle.getControlDim(), 2);
    ASSERT_DOUBLE_EQ(bicycle.getTimestep(), 0.05);
    ASSERT_EQ(bicycle.getIntegrationType(), "euler");
}

TEST(BicycleTest, Jacobians) {
    // Create bicycle instance
    double timestep = 0.05;
    double wheelbase = 2.0;
    cddp::Bicycle bicycle(timestep, wheelbase, "euler");

    // Test state and control
    Eigen::VectorXd state(4);
    state << 1.0, 2.0, M_PI/6, 1.5;  // Some non-zero state
    Eigen::VectorXd control(2);
    control << 0.5, 0.1;  // Non-zero controls

    // Get analytical Jacobians
    Eigen::MatrixXd A_analytical = bicycle.getStateJacobian(state, control);
    Eigen::MatrixXd B_analytical = bicycle.getControlJacobian(state, control);

    // Get numerical Jacobians
    auto f_A = [&](const Eigen::VectorXd& x) {
        return bicycle.getContinuousDynamics(x, control);
    };
    auto f_B = [&](const Eigen::VectorXd& u) {
        return bicycle.getContinuousDynamics(state, u);
    };
    Eigen::MatrixXd A_numerical = finite_difference_jacobian(f_A, state);
    Eigen::MatrixXd B_numerical = finite_difference_jacobian(f_B, control);

    // Test dimensions
    ASSERT_EQ(A_analytical.rows(), 4);
    ASSERT_EQ(A_analytical.cols(), 4);
    ASSERT_EQ(B_analytical.rows(), 4);
    ASSERT_EQ(B_analytical.cols(), 2);

    // Compare analytical and numerical Jacobians
    double tolerance = 1e-5;
    EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);
    EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);
}

TEST(BicycleTest, ContinuousDynamics) {
    // Create bicycle instance
    double timestep = 0.05;
    double wheelbase = 2.0;
    cddp::Bicycle bicycle(timestep, wheelbase, "euler");

    // Test various states and controls
    Eigen::VectorXd state(4);
    state << 0.0, 0.0, 0.0, 1.0;  // Moving straight
    Eigen::VectorXd control(2);
    control << 0.0, 0.0;  // No acceleration or steering

    // Get dynamics
    Eigen::VectorXd state_dot = bicycle.getContinuousDynamics(state, control);

    // Test expected behavior for straight motion
    EXPECT_NEAR(state_dot[0], 1.0, 1e-10);  // dx/dt = v * cos(theta)
    EXPECT_NEAR(state_dot[1], 0.0, 1e-10);  // dy/dt = v * sin(theta)
    EXPECT_NEAR(state_dot[2], 0.0, 1e-10);  // dtheta/dt = 0
    EXPECT_NEAR(state_dot[3], 0.0, 1e-10);  // dv/dt = 0

    // Test turning behavior
    state << 0.0, 0.0, 0.0, 1.0;
    control << 0.0, 0.1;  // Some steering angle
    state_dot = bicycle.getContinuousDynamics(state, control);
    
    // Should have non-zero angular velocity when steering
    EXPECT_GT(std::abs(state_dot[2]), 0.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}