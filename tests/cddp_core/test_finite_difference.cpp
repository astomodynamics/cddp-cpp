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

TEST(JacobianTest , Pendulum) {
    // Parameters
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.0;

    // Create a pendulum instance 
    double timestep = 0.05;
    std::string integration_type = "euler";
    cddp::Pendulum pendulum(timestep, length, mass, damping, integration_type);

    // Initial state and control (use Eigen vectors)
    Eigen::VectorXd state(2);
    state << 0.1, 0.0;  // Start at a small angle, zero velocity
    Eigen::VectorXd control(1);
    control << 0.0; // No torque initially
    
    // Compute the Jacobians
    Eigen::MatrixXd A = pendulum.getStateJacobian(state, control);
    Eigen::MatrixXd B = pendulum.getControlJacobian(state, control);

    // Check the Jacobians
    Eigen::MatrixXd A_expected = pendulum.getFiniteDifferenceStateJacobian(state, control);
    Eigen::MatrixXd B_expected = pendulum.getFiniteDifferenceControlJacobian(state, control);

    // print the Jacobians
    std::cout << "A = \n" << A << std::endl;
    std::cout << "B = \n" << B << std::endl;

    // Check the Jacobians
    ASSERT_TRUE(A.isApprox(A_expected, 1e-6));
    ASSERT_TRUE(B.isApprox(B_expected, 1e-6));

    // Print the Jacobians
    std::cout << "A = \n" << A << std::endl;
    std::cout << "B = \n" << B << std::endl;
}
