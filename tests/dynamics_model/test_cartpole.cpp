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

TEST(CartPoleTest, DiscreteDynamics) {
    double timestep = 0.05;
    std::string integration_type = "rk4";
    cddp::CartPole cartpole(timestep, integration_type);

    std::vector<double> time_data, x_data, theta_data, energy_data, v_data, w_data;

    Eigen::VectorXd state(4);
    state << 0.0, 0.0-0.001, 0.0, 0.0;  // Initial position, angle (30 degrees), velocity, angular velocity
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
    std::cout << "final state: " << state.transpose() << std::endl;
    // Basic assertions
    ASSERT_EQ(cartpole.getStateDim(), 4);
    ASSERT_EQ(cartpole.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(cartpole.getTimestep(), 0.05);
    ASSERT_EQ(cartpole.getIntegrationType(), "rk4");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}