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
    Eigen::MatrixXd A = pendulum.getStateJacobian(state, control, 0.0);
    Eigen::MatrixXd B = pendulum.getControlJacobian(state, control, 0.0);

    // Check the Jacobians
    auto f_A = [&](const Eigen::VectorXd& x) {
        return pendulum.getContinuousDynamics(x, control, 0.0);
    };
    auto f_B = [&](const Eigen::VectorXd& u) {
        return pendulum.getContinuousDynamics(state, u, 0.0);
    };
    Eigen::MatrixXd A_expected = finite_difference_jacobian(f_A, state);
    Eigen::MatrixXd B_expected = finite_difference_jacobian(f_B, control);

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

TEST(JacobianTest, PendulumRK4GetJacobiansUsesDiscreteDynamics) {
    const double length = 1.0;
    const double mass = 1.0;
    const double damping = 0.05;
    const double timestep = 0.2;
    cddp::Pendulum pendulum(timestep, length, mass, damping, "rk4");

    Eigen::VectorXd state(2);
    state << 0.7, -0.25;
    Eigen::VectorXd control(1);
    control << 0.3;

    const auto [A, B] = pendulum.getJacobians(state, control, 0.0);

    auto discrete_wrt_x = [&](const Eigen::VectorXd& x) {
        return pendulum.getDiscreteDynamics(x, control, 0.0);
    };
    auto discrete_wrt_u = [&](const Eigen::VectorXd& u) {
        return pendulum.getDiscreteDynamics(state, u, 0.0);
    };

    const Eigen::MatrixXd A_expected =
        finite_difference_jacobian(discrete_wrt_x, state);
    const Eigen::MatrixXd B_expected =
        finite_difference_jacobian(discrete_wrt_u, control);

    Eigen::MatrixXd A_euler =
        Eigen::MatrixXd::Identity(state.size(), state.size()) +
        timestep * pendulum.getStateJacobian(state, control, 0.0);
    Eigen::MatrixXd B_euler =
        timestep * pendulum.getControlJacobian(state, control, 0.0);

    EXPECT_TRUE(A.isApprox(A_expected, 1e-10));
    EXPECT_TRUE(B.isApprox(B_expected, 1e-10));
    EXPECT_GT((A - A_euler).norm(), 1e-3);
    EXPECT_GT((B - B_euler).norm(), 1e-3);
}
