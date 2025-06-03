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
#include <cmath> // For M_PI

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

        state = cartpole.getDiscreteDynamics(state, control, 0.0);
    }
    std::cout << "final state: " << state.transpose() << std::endl;
    // Basic assertions
    ASSERT_EQ(cartpole.getStateDim(), 4);
    ASSERT_EQ(cartpole.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(cartpole.getTimestep(), 0.05);
    ASSERT_EQ(cartpole.getIntegrationType(), "rk4");
}

TEST(CartPoleJacobianTest, Jacobians) {
    double timestep = 0.01; // Timestep for dynamics, not directly used by Jacobian of continuous dynamics
    std::string integration_type = "rk4"; // Not directly used, but needed for constructor
    cddp::CartPole cartpole(timestep, integration_type);

    Eigen::VectorXd state(4);
    // state << 0.1, 0.2, 0.3, 0.4;  // x, theta, x_dot, theta_dot
    state << 0.0, M_PI / 6.0, 0.1, -0.1; // Example state: x=0, theta=30deg, x_dot=0.1, theta_dot=-0.1

    Eigen::VectorXd control(1);
    control << 1.0;  // Example control: force = 1.0

    // Test State Jacobian
    Eigen::MatrixXd analytical_A = cartpole.getStateJacobian(state, control, 0.0);
    Eigen::MatrixXd autodiff_A = cartpole.DynamicalSystem::getStateJacobian(state, control, 0.0);

    // std::cout << "Analytical State Jacobian (A):\n" << analytical_A << std::endl;
    // std::cout << "Autodiff State Jacobian (A):\n" << autodiff_A << std::endl;

    ASSERT_TRUE(analytical_A.isApprox(autodiff_A, 1e-9))
        << "Analytical A:\n" << analytical_A
        << "\nAutodiff A:\n" << autodiff_A;

    // Test Control Jacobian
    Eigen::MatrixXd analytical_B = cartpole.getControlJacobian(state, control, 0.0);
    Eigen::MatrixXd autodiff_B = cartpole.DynamicalSystem::getControlJacobian(state, control, 0.0);

    // std::cout << "Analytical Control Jacobian (B):\n" << analytical_B << std::endl;
    // std::cout << "Autodiff Control Jacobian (B):\n" << autodiff_B << std::endl;

    ASSERT_TRUE(analytical_B.isApprox(autodiff_B, 1e-9))
        << "Analytical B:\n" << analytical_B
        << "\nAutodiff B:\n" << autodiff_B;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}