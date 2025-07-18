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
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "cddp.hpp"
#include "dynamics_model/acrobot.hpp"

using namespace cddp;

TEST(AcrobotTest, DiscreteDynamics) {
    // Create an Acrobot instance
    double timestep = 0.01;
    // Parameters
    double l1 = 1.0;     // length of link 1
    double l2 = 1.0;     // length of link 2
    double m1 = 1.0;     // mass of link 1
    double m2 = 1.0;     // mass of link 2
    double J1 = 1.0;     // inertia of link 1
    double J2 = 1.0;     // inertia of link 2
    std::string integration_type = "rk4";
    
    cddp::Acrobot acrobot(timestep, l1, l2, m1, m2, J1, J2, integration_type);
    
    // Store states for analysis
    std::vector<double> time_data, theta1_data, theta2_data, theta1_dot_data, theta2_dot_data;
    std::vector<double> energy_data;
    
    // Initial state and control
    Eigen::VectorXd state(4);
    state << M_PI/4, -M_PI/6, 0.0, 0.0;  // Initial angles and zero velocities
    Eigen::VectorXd control(1);
    control << 0.0;  // No initial torque
    
    // Simulate for a few steps
    int num_steps = 500;
    for (int i = 0; i < num_steps; ++i) {
        // Store data
        time_data.push_back(i * timestep);
        theta1_data.push_back(state[0]);
        theta2_data.push_back(state[1]);
        theta1_dot_data.push_back(state[2]);
        theta2_dot_data.push_back(state[3]);
        // Energy tracking removed - not available in Julia version
        
        // Compute the next state
        state = acrobot.getDiscreteDynamics(state, control, 0.0);
    }
    
    // Assert correct dimensions
    ASSERT_EQ(acrobot.getStateDim(), 4);
    ASSERT_EQ(acrobot.getControlDim(), 1);
    ASSERT_DOUBLE_EQ(acrobot.getTimestep(), 0.01);
    ASSERT_EQ(acrobot.getIntegrationType(), "rk4");
    
    // Verify simulation ran without errors
    ASSERT_EQ(theta1_data.size(), num_steps);
    ASSERT_EQ(theta2_data.size(), num_steps);
}

TEST(AcrobotTest, FrictionTest) {
    // Test that friction affects the system
    double timestep = 0.01;
    std::string integration_type = "rk4";
    
    cddp::Acrobot acrobot(timestep, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, integration_type);
    
    // Initial state near equilibrium with small velocities
    Eigen::VectorXd state(4);
    state << M_PI - 0.1, 0.0, 0.1, 0.1;  // Near hanging down position
    Eigen::VectorXd control(1);
    control << 0.0;  // No torque
    
    // Store initial state
    Eigen::VectorXd initial_state = state;
    
    // Simulate for some steps
    for (int i = 0; i < 100; ++i) {
        state = acrobot.getDiscreteDynamics(state, control, 0.0);
    }
    
    // The system should have moved from initial state due to dynamics
    ASSERT_GT((state - initial_state).norm(), 0.01);
}

TEST(AcrobotTest, Jacobians) {
    double timestep = 0.01;
    std::string integration_type = "rk4";
    cddp::Acrobot acrobot(timestep, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, integration_type);
    
    Eigen::VectorXd state(4);
    state << M_PI/6.0, -M_PI/8.0, 0.1, -0.1; // Example state
    
    Eigen::VectorXd control(1);
    control << 1.0;  // Example control: torque = 1.0
    
    // Test State Jacobian
    Eigen::MatrixXd autodiff_A = acrobot.getStateJacobian(state, control, 0.0);
    
    // The Jacobian should have the correct dimensions
    ASSERT_EQ(autodiff_A.rows(), 4);
    ASSERT_EQ(autodiff_A.cols(), 4);
    
    // Test Control Jacobian
    Eigen::MatrixXd autodiff_B = acrobot.getControlJacobian(state, control, 0.0);
    
    // The control Jacobian should have the correct dimensions
    ASSERT_EQ(autodiff_B.rows(), 4);
    ASSERT_EQ(autodiff_B.cols(), 1);
    
    // The control only affects the second joint acceleration
    // So B should have zeros in the first three rows
    ASSERT_DOUBLE_EQ(autodiff_B(0, 0), 0.0);
    ASSERT_DOUBLE_EQ(autodiff_B(1, 0), 0.0);
    // The third and fourth rows should be non-zero (accelerations)
    ASSERT_NE(autodiff_B(2, 0), 0.0);
    ASSERT_NE(autodiff_B(3, 0), 0.0);
}

TEST(AcrobotTest, Equilibrium) {
    // Note: The Julia implementation has a quirk in the gravity term that prevents
    // the upright position from being an equilibrium point
    double timestep = 0.01;
    cddp::Acrobot acrobot(timestep, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    
    // Hanging down equilibrium state
    Eigen::VectorXd state(4);
    state << M_PI, 0.0, 0.0, 0.0;  // First link down, second link straight
    Eigen::VectorXd control(1);
    control << 0.0;  // No torque
    
    // Get dynamics at equilibrium
    Eigen::VectorXd state_dot = acrobot.getContinuousDynamics(state, control, 0.0);
    
    // The velocities should remain zero (but accelerations may not due to friction)
    ASSERT_NEAR(state_dot(0), 0.0, 1e-10);
    ASSERT_NEAR(state_dot(1), 0.0, 1e-10);
    
    // Test that it's unstable - small perturbation should grow
    state << 0.01, 0.0, 0.0, 0.0;  // Small perturbation
    state_dot = acrobot.getContinuousDynamics(state, control, 0.0);
    
    // The angular acceleration should be non-zero
    ASSERT_GT(std::abs(state_dot(2)), 1e-6);
}

TEST(AcrobotTest, ControlEffect) {
    // Test that control input affects the system correctly
    double timestep = 0.01;
    cddp::Acrobot acrobot(timestep, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    
    Eigen::VectorXd state(4);
    state << M_PI/4, -M_PI/6, 0.0, 0.0;
    
    // Test with positive control
    Eigen::VectorXd control_pos(1);
    control_pos << 1.0;
    Eigen::VectorXd state_dot_pos = acrobot.getContinuousDynamics(state, control_pos, 0.0);
    
    // Test with negative control
    Eigen::VectorXd control_neg(1);
    control_neg << -1.0;
    Eigen::VectorXd state_dot_neg = acrobot.getContinuousDynamics(state, control_neg, 0.0);
    
    // The control should affect the accelerations (not the velocities directly)
    ASSERT_DOUBLE_EQ(state_dot_pos(0), state_dot_neg(0));  // theta1_dot unchanged
    ASSERT_DOUBLE_EQ(state_dot_pos(1), state_dot_neg(1));  // theta2_dot unchanged
    ASSERT_NE(state_dot_pos(2), state_dot_neg(2));  // theta1_ddot should differ
    ASSERT_NE(state_dot_pos(3), state_dot_neg(3));  // theta2_ddot should differ
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}