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

    std::vector<double> time_data, x_data, theta_data, energy_data;

    Eigen::VectorXd state(4);
    state << 0.0, 0.0, M_PI/6, 0.0;  // Initial position, velocity, angle (30 degrees), angular velocity
    Eigen::VectorXd control(1);
    control << 0.0;  // No initial force

    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        theta_data.push_back(state[2]);
        
        // Calculate total energy (potential + kinetic)
        double potential = cartpole.getPoleMass() * cartpole.getGravity() * 
                          cartpole.getPoleLength() * (1 - std::cos(state[2]));
        double kinetic = 0.5 * cartpole.getCartMass() * state[1] * state[1] + 
                        0.5 * cartpole.getPoleMass() * (
                            state[1] * state[1] + 
                            cartpole.getPoleLength() * cartpole.getPoleLength() * state[3] * state[3]
                        );
        energy_data.push_back(potential + kinetic);

        state = cartpole.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(cartpole.getStateDim(), 4);
    ASSERT_EQ(cartpole.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(cartpole.getTimestep(), 0.05);
    ASSERT_EQ(cartpole.getIntegrationType(), "rk4");

    // // Plot the results
    // plt::figure();
    // plt::subplot(3, 1, 1);
    // plt::plot(time_data, x_data);
    // plt::title("Cart position");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Position [m]");

    // plt::subplot(3, 1, 2);
    // plt::plot(time_data, theta_data);
    // plt::title("Pole angle");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Angle [rad]");

    // plt::subplot(3, 1, 3);
    // plt::plot(time_data, energy_data);
    // plt::title("Total energy");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Energy [J]");
    // // plt::save("../results/tests/cartpole_discrete_dynamics.png");
    // plt::show();
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