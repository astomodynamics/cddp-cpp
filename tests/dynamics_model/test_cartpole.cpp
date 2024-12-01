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

TEST(CartPoleTest, DiscreteDynamics) {
    double timestep = 0.05;
    std::string integration_type = "rk4";
    cddp::CartPole cartpole(timestep, integration_type);

    std::vector<double> time_data, x_data, theta_data, energy_data, v_data, w_data;

    Eigen::VectorXd state(4);
    state << 0.0, M_PI/3, 0.0, 0.0;  // Initial position, angle (30 degrees), velocity, angular velocity
    Eigen::VectorXd control(1);
    control << 0.0;  // No initial force

    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        theta_data.push_back(state[1]);
        v_data.push_back(state[2]);
        w_data.push_back(state[3]);
        
        // Calculate total energy (potential + kinetic)
        double potential = -cartpole.getCartMass() * cartpole.getGravity() * cartpole.getPoleLength() * std::cos(state[1]);

        double kinetic = 0.5 * (cartpole.getCartMass() + cartpole.getPoleMass()) * state[2] * state[2] + 
                        0.5 * cartpole.getPoleMass() * cartpole.getPoleLength() * cartpole.getPoleLength() * state[3] * state[3] +
                        cartpole.getPoleMass() * cartpole.getPoleLength() * state[2] * state[3] * std::cos(state[1]);

        energy_data.push_back(potential + kinetic);

        state = cartpole.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(cartpole.getStateDim(), 4);
    ASSERT_EQ(cartpole.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(cartpole.getTimestep(), 0.05);
    ASSERT_EQ(cartpole.getIntegrationType(), "rk4");

    // // Plot the results (position, velocity, angle, angular velocity, energy)
    // plt::figure();
    // plt::subplot(5, 1, 1);
    // plt::plot(time_data, x_data);
    // plt::title("Cart position");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Position [m]");

    // plt::subplot(5, 1, 2);
    // plt::plot(time_data, theta_data);
    // plt::title("Pole angle");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Angle [rad]");

    // plt::subplot(5, 1, 3);
    // plt::plot(time_data, v_data);
    // plt::title("Cart velocity");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Velocity [m/s]");

    // plt::subplot(5, 1, 4);
    // plt::plot(time_data, w_data);
    // plt::title("Pole angular velocity");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Angular velocity [rad/s]");

    // plt::subplot(5, 1, 5);
    // plt::plot(time_data, energy_data);
    // plt::title("Total energy");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Energy [J]");
    // plt::save("../results/tests/cartpole_discrete_dynamics.png");

    // // Animation
    // plt::figure_size(800, 600);
    // plt::title("CartPole Animation");
    // plt::xlabel("x");
    // plt::ylabel("y");

    // double cart_width = 0.3;
    // double cart_height = 0.2;
    // double pole_width = 0.05;
    // double pole_length = cartpole.getPoleLength();

    // state << 0.0, M_PI/3, 0.0, 0.0;  // Reset initial state

    // // create initial cartpole animation
    // for (int i = 0; i < num_steps + 1; ++i) {
    //     if (i % 5 == 0) {
    //         plt::clf();

    //         double x = state(0);
    //         double theta = state(1);

    //         // Cart corners
    //         std::vector<double> cart_x = {
    //             x - cart_width/2, x + cart_width/2,
    //             x + cart_width/2, x - cart_width/2,
    //             x - cart_width/2
    //         };
    //         std::vector<double> cart_y = {
    //             -cart_height/2, -cart_height/2,
    //             cart_height/2, cart_height/2,
    //             -cart_height/2
    //         };
    //         plt::plot(cart_x, cart_y, "k-");

    //         // Pole
    //         double pole_end_x = x + pole_length * std::sin(theta);
    //         double pole_end_y = -pole_length * std::cos(theta);
    //         std::vector<double> pole_x = {x, pole_end_x};
    //         std::vector<double> pole_y = {0, pole_end_y};
    //         plt::plot(pole_x, pole_y, "b-");

    //         // Plot pole bob
    //         std::vector<double> circle_x, circle_y;
    //         for (int j = 0; j <= 20; ++j) {
    //             double t = 2 * M_PI * j / 20;
    //             circle_x.push_back(pole_end_x + pole_width * std::cos(t));
    //             circle_y.push_back(pole_end_y + pole_width * std::sin(t));
    //         }
    //         plt::plot(circle_x, circle_y, "b-");

    //         // Set fixed axis limits for stable animation
    //         double view_width = 4.0;
    //         plt::xlim(x - view_width/2, x + view_width/2);
    //         plt::ylim(-view_width/2, view_width/2);
    //         // plt::axis("equal");

    //         std::string filename = "../results/tests/cartpole_" + std::to_string(i) + ".png";
    //         plt::save(filename);
    //         plt::pause(0.01);
    //     }

    //     // Update state
    //     state = cartpole.getDiscreteDynamics(state, Eigen::VectorXd::Zero(1));

    // }
}

// TEST(CartPoleTest, Jacobians) {
//     double timestep = 0.05;
//     cddp::CartPole cartpole(timestep, "rk4");

//     Eigen::VectorXd state(4);
//     state << 1.0, 0.5, M_PI/4, 0.2;  // Some non-zero state
//     Eigen::VectorXd control(1);
//     control << 0.5;  // Non-zero control

//     Eigen::MatrixXd A_analytical = cartpole.getStateJacobian(state, control);
//     Eigen::MatrixXd B_analytical = cartpole.getControlJacobian(state, control);

//     Eigen::MatrixXd A_numerical = cartpole.getFiniteDifferenceStateJacobian(state, control);
//     Eigen::MatrixXd B_numerical = cartpole.getFiniteDifferenceControlJacobian(state, control);

//     ASSERT_EQ(A_analytical.rows(), 4);
//     ASSERT_EQ(A_analytical.cols(), 4);
//     ASSERT_EQ(B_analytical.rows(), 4);
//     ASSERT_EQ(B_analytical.cols(), 1);

//     double tolerance = 1e-5;
//     EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);
//     EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);
// }

// TEST(CartPoleTest, ContinuousDynamics) {
//     double timestep = 0.05;
//     cddp::CartPole cartpole(timestep, "rk4");

//     Eigen::VectorXd state(4);
//     state << 0.0, 0.0, 0.0, 0.0;  // Equilibrium state
//     Eigen::VectorXd control(1);
//     control << 0.0;  // No force

//     Eigen::VectorXd state_dot = cartpole.getContinuousDynamics(state, control);

//     EXPECT_NEAR(state_dot[0], 0.0, 1e-10);  // dx/dt = 0
//     EXPECT_NEAR(state_dot[1], 0.0, 1e-10);  // dx_dot/dt = 0
//     EXPECT_NEAR(state_dot[2], 0.0, 1e-10);  // dtheta/dt = 0
//     EXPECT_NEAR(state_dot[3], 0.0, 1e-10);  // dtheta_dot/dt = 0

//     // Test with non-zero force
//     control << 1.0;
//     state_dot = cartpole.getContinuousDynamics(state, control);
    
//     EXPECT_GT(std::abs(state_dot[1]), 0.0);  // Should have cart acceleration
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}