/*
 Copyright 2025 Tomo Sasaki

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
// Description: Test the forklift dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/forklift.hpp"
#include "cddp_core/objective.hpp"

using namespace cddp;

TEST(ForkliftTest, DiscreteDynamics) {
    // Create a forklift instance
    double timestep = 0.01;
    double wheelbase = 2.0;
    std::string integration_type = "euler";
    bool rear_steer = true;  // Typical for forklifts
    double max_steering_angle = M_PI / 4;  // 45 degrees
    
    Forklift forklift(timestep, wheelbase, integration_type, rear_steer, max_steering_angle);
    
    // Basic assertions
    ASSERT_EQ(forklift.getStateDim(), 5);
    ASSERT_EQ(forklift.getControlDim(), 2);
    ASSERT_DOUBLE_EQ(forklift.getTimestep(), 0.01);
    ASSERT_EQ(forklift.getIntegrationType(), "euler");
    
    // Test straight line motion
    Eigen::VectorXd state(5);
    state << 0.0, 0.0, 0.0, 1.0, 0.0;  // Start at origin, 1 m/s forward, no steering
    Eigen::VectorXd control(2);
    control << 0.0, 0.0;  // No acceleration, no steering rate
    
    Eigen::VectorXd next_state = forklift.getDiscreteDynamics(state, control, 0.0);
    
    // Should move forward by v * timestep
    EXPECT_NEAR(next_state[0], 0.01, 1e-6);  // x position
    EXPECT_NEAR(next_state[1], 0.0, 1e-6);   // y position
    EXPECT_NEAR(next_state[2], 0.0, 1e-6);   // theta
    EXPECT_NEAR(next_state[3], 1.0, 1e-6);   // velocity
    EXPECT_NEAR(next_state[4], 0.0, 1e-6);   // steering angle
    
    // Test steering dynamics
    state << 0.0, 0.0, 0.0, 0.0, 0.0;
    control << 0.0, 0.5;  // Steering rate
    next_state = forklift.getDiscreteDynamics(state, control, 0.0);
    EXPECT_NEAR(next_state[4], 0.005, 1e-6);  // Steering angle should increase
    
    // Test acceleration
    state << 0.0, 0.0, 0.0, 0.0, 0.0;
    control << 2.0, 0.0;  // Acceleration
    next_state = forklift.getDiscreteDynamics(state, control, 0.0);
    EXPECT_NEAR(next_state[3], 0.02, 1e-6);  // Velocity should increase
}

TEST(ForkliftTest, RearSteerBehavior) {
    // Test rear-steer vs front-steer behavior
    double timestep = 0.01;
    double wheelbase = 2.0;
    
    Forklift forklift_rear(timestep, wheelbase, "euler", true);   // rear-steer
    Forklift forklift_front(timestep, wheelbase, "euler", false); // front-steer
    
    Eigen::VectorXd state(5);
    state << 0.0, 0.0, 0.0, 1.0, M_PI/6;  // 30 degrees steering
    Eigen::VectorXd control(2);
    control << 0.0, 0.0;
    
    Eigen::VectorXd next_rear = forklift_rear.getDiscreteDynamics(state, control, 0.0);
    Eigen::VectorXd next_front = forklift_front.getDiscreteDynamics(state, control, 0.0);
    
    // Rear-steer should turn opposite direction
    EXPECT_NEAR(next_rear[2], -next_front[2], 1e-6);  // theta should be opposite
}

TEST(ForkliftTest, JacobianTest) {
    // Create a forklift instance
    double timestep = 0.01;
    double wheelbase = 2.0;
    std::string integration_type = "euler";
    Forklift forklift(timestep, wheelbase, integration_type);
    
    // Test state
    Eigen::VectorXd state(5);
    state << 0.0, 0.0, 0.0, 1.0, 0.1;  // Moving forward with slight steering
    Eigen::VectorXd control(2);
    control << 0.1, 0.05;  // Small acceleration and steering rate
    
    // Compute the Jacobians
    Eigen::MatrixXd A = forklift.getStateJacobian(state, control, 0.0);
    Eigen::MatrixXd B = forklift.getControlJacobian(state, control, 0.0);
    
    // Check dimensions
    EXPECT_EQ(A.rows(), 5);
    EXPECT_EQ(A.cols(), 5);
    EXPECT_EQ(B.rows(), 5);
    EXPECT_EQ(B.cols(), 2);
    
    // Verify Jacobians are finite
    EXPECT_TRUE(A.allFinite());
    EXPECT_TRUE(B.allFinite());
    
    // Convert to discrete time for numerical comparison
    A *= timestep;
    A.diagonal().array() += 1.0;
    B *= timestep;
    
    // Numerical verification using finite differences
    const double eps = 1e-7;
    Eigen::MatrixXd A_numerical = Eigen::MatrixXd::Zero(5, 5);
    
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        state_plus(i) += eps;
        state_minus(i) -= eps;
        
        Eigen::VectorXd f_plus = forklift.getDiscreteDynamics(state_plus, control, 0.0);
        Eigen::VectorXd f_minus = forklift.getDiscreteDynamics(state_minus, control, 0.0);
        
        A_numerical.col(i) = (f_plus - f_minus) / (2 * eps);
    }
    
    // Compare analytical and numerical Jacobians
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(A(i, j), A_numerical(i, j), 1e-4) 
                << "Mismatch at A(" << i << ", " << j << ")";
        }
    }
}