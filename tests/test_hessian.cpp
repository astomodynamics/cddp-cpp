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

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>

#include "dynamics_model/pendulum.hpp"
#include "dynamics_model/dubins_car.hpp"

namespace cddp {
namespace testing {

// Fixture for pendulum tests
class PendulumHessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a pendulum model
        timestep = 0.01;
        length = 1.0;     // Length of the pendulum [m]
        mass = 1.0;       // Mass [kg]
        damping = 0.1;    // Damping coefficient
        integration = "rk4";
        
        pendulum = std::make_unique<Pendulum>(timestep, length, mass, damping, integration);
        
        // Define a state and control
        state = Eigen::VectorXd::Zero(2);
        state << M_PI / 4.0, 0.0;  // 45-degree angle, zero velocity
        
        control = Eigen::VectorXd::Zero(1);
        control << 1.0;  // Apply a torque of 1 Nm
    }

    std::unique_ptr<Pendulum> pendulum;
    Eigen::VectorXd state;
    Eigen::VectorXd control;
    double timestep;
    double length;
    double mass;
    double damping;
    std::string integration;
};

// Fixture for Dubins car tests
class DubinsCarHessianTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a Dubins car model
        speed = 1.0;      // Constant forward speed [m/s]
        timestep = 0.01;  // Time step [s]
        integration = "rk4";
        
        dubins_car = std::make_unique<DubinsCar>(speed, timestep, integration);
        
        // Define a state and control
        state = Eigen::VectorXd::Zero(3);
        state << 0.0, 0.0, M_PI / 4.0;  // (x, y, theta) = (0, 0, 45°)
        
        control = Eigen::VectorXd::Zero(1);
        control << 0.5;  // Turn rate of 0.5 rad/s
    }

    std::unique_ptr<DubinsCar> dubins_car;
    Eigen::VectorXd state;
    Eigen::VectorXd control;
    double speed;
    double timestep;
    std::string integration;
};

// Test pendulum model parameters
TEST_F(PendulumHessianTest, ParametersAreCorrect) {
    EXPECT_DOUBLE_EQ(pendulum->getLength(), length);
    EXPECT_DOUBLE_EQ(pendulum->getMass(), mass);
    EXPECT_DOUBLE_EQ(pendulum->getDamping(), damping);
    EXPECT_DOUBLE_EQ(pendulum->getTimestep(), timestep);
    EXPECT_EQ(pendulum->getIntegrationType(), integration);
}

// Test pendulum dynamics calculation
TEST_F(PendulumHessianTest, DynamicsCalculation) {
    Eigen::VectorXd xdot = pendulum->getContinuousDynamics(state, control, 0.0);
    
    // Expected values for theta_dot and theta_ddot
    double expected_theta_dot = state(1);  // state(1) is the current angular velocity
    double expected_theta_ddot = (control(0) - pendulum->getDamping() * state(1) + 
                                pendulum->getMass() * pendulum->getGravity() * 
                                pendulum->getLength() * std::sin(state(0))) / 
                               (pendulum->getMass() * pendulum->getLength() * pendulum->getLength());
    
    EXPECT_NEAR(xdot(0), expected_theta_dot, 1e-10);
    EXPECT_NEAR(xdot(1), expected_theta_ddot, 1e-10);
}

// Test pendulum state Jacobian
TEST_F(PendulumHessianTest, StateJacobian) {
    Eigen::MatrixXd A = pendulum->getStateJacobian(state, control, 0.0);
    
    // Expected values for the state Jacobian
    double expected_A_11 = 0.0;  // d(theta_dot)/d(theta)
    double expected_A_12 = 1.0;  // d(theta_dot)/d(theta_dot)
    double expected_A_21 = (pendulum->getGravity() / pendulum->getLength()) * std::cos(state(0));  // d(theta_ddot)/d(theta)
    double expected_A_22 = -pendulum->getDamping() / (pendulum->getMass() * pendulum->getLength() * pendulum->getLength());  // d(theta_ddot)/d(theta_dot)
    
    EXPECT_NEAR(A(0, 0), expected_A_11, 1e-10);
    EXPECT_NEAR(A(0, 1), expected_A_12, 1e-10);
    EXPECT_NEAR(A(1, 0), expected_A_21, 1e-10);
    EXPECT_NEAR(A(1, 1), expected_A_22, 1e-10);
}

// Test pendulum control Jacobian
TEST_F(PendulumHessianTest, ControlJacobian) {
    Eigen::MatrixXd B = pendulum->getControlJacobian(state, control, 0.0);
    
    // Expected values for the control Jacobian
    double expected_B_11 = 0.0;  // d(theta_dot)/d(torque)
    double expected_B_21 = 1.0 / (pendulum->getMass() * pendulum->getLength() * pendulum->getLength());  // d(theta_ddot)/d(torque)
    
    EXPECT_NEAR(B(0, 0), expected_B_11, 1e-10);
    EXPECT_NEAR(B(1, 0), expected_B_21, 1e-10);
}

// Test pendulum state Hessian
TEST_F(PendulumHessianTest, StateHessian) {
    std::vector<Eigen::MatrixXd> state_hessian = pendulum->getStateHessian(state, control, 0.0);
    
    // Check dimensions
    EXPECT_EQ(state_hessian.size(), 2);  // Two state dimensions
    EXPECT_EQ(state_hessian[0].rows(), 2);  // 2x2 matrices
    EXPECT_EQ(state_hessian[0].cols(), 2);
    
    // Expected values for the state Hessian
    // For theta_dot dimension (index 0), all second derivatives should be zero
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(state_hessian[0](i, j), 0.0, 1e-10);
        }
    }
    
    // For theta_ddot dimension (index 1), only d^2(theta_ddot)/d(theta)^2 is non-zero
    double expected_H_100 = -(pendulum->getGravity() / pendulum->getLength()) * std::sin(state(0));  // d^2(theta_ddot)/d(theta)^2
    EXPECT_NEAR(state_hessian[1](0, 0), expected_H_100, 1e-10);
    EXPECT_NEAR(state_hessian[1](0, 1), 0.0, 1e-10);
    EXPECT_NEAR(state_hessian[1](1, 0), 0.0, 1e-10);
    EXPECT_NEAR(state_hessian[1](1, 1), 0.0, 1e-10);
}

// Test pendulum control Hessian
TEST_F(PendulumHessianTest, ControlHessian) {
    std::vector<Eigen::MatrixXd> control_hessian = pendulum->getControlHessian(state, control, 0.0);
    
    // Check dimensions
    EXPECT_EQ(control_hessian.size(), 2);  // Two state dimensions
    EXPECT_EQ(control_hessian[0].rows(), 1);  // 1x1 matrices (control dimension is 1)
    EXPECT_EQ(control_hessian[0].cols(), 1);
    
    // All second derivatives with respect to control should be zero
    for (int k = 0; k < 2; ++k) {
        EXPECT_NEAR(control_hessian[k](0, 0), 0.0, 1e-10);
    }
}

// Test Dubins car model parameters
TEST_F(DubinsCarHessianTest, ParametersAreCorrect) {
    EXPECT_DOUBLE_EQ(dubins_car->getTimestep(), timestep);
    EXPECT_EQ(dubins_car->getIntegrationType(), integration);
}

// Test Dubins car dynamics calculation
TEST_F(DubinsCarHessianTest, DynamicsCalculation) {
    Eigen::VectorXd xdot = dubins_car->getContinuousDynamics(state, control, 0.0);
    
    // Expected values
    double expected_x_dot = speed * std::cos(state(2));
    double expected_y_dot = speed * std::sin(state(2));
    double expected_theta_dot = control(0);
    
    EXPECT_NEAR(xdot(0), expected_x_dot, 1e-10);
    EXPECT_NEAR(xdot(1), expected_y_dot, 1e-10);
    EXPECT_NEAR(xdot(2), expected_theta_dot, 1e-10);
}

// Test Dubins car state Jacobian
TEST_F(DubinsCarHessianTest, StateJacobian) {
    Eigen::MatrixXd A = dubins_car->getStateJacobian(state, control, 0.0);
    
    // Expected values for the state Jacobian
    double expected_A_13 = -speed * std::sin(state(2));  // dx/dtheta
    double expected_A_23 = speed * std::cos(state(2));   // dy/dtheta
    
    EXPECT_NEAR(A(0, 2), expected_A_13, 1e-10);
    EXPECT_NEAR(A(1, 2), expected_A_23, 1e-10);
}

// Test Dubins car control Jacobian
TEST_F(DubinsCarHessianTest, ControlJacobian) {
    Eigen::MatrixXd B = dubins_car->getControlJacobian(state, control, 0.0);
    
    // Expected values for the control Jacobian
    double expected_B_31 = 1.0;  // dtheta/domega
    
    EXPECT_NEAR(B(2, 0), expected_B_31, 1e-10);
}

// Test Dubins car state Hessian
TEST_F(DubinsCarHessianTest, StateHessian) {
    std::vector<Eigen::MatrixXd> state_hessian = dubins_car->getStateHessian(state, control, 0.0);
    
    // Check dimensions
    EXPECT_EQ(state_hessian.size(), 3);  // Three state dimensions
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(state_hessian[i].rows(), 3);  // 3x3 matrices
        EXPECT_EQ(state_hessian[i].cols(), 3);
    }
    
    // Expected values for the state Hessian
    // For x dimension (index 0), only d^2x/dtheta^2 is non-zero
    double expected_H_022 = -speed * std::cos(state(2));  // d^2x/dtheta^2
    EXPECT_NEAR(state_hessian[0](2, 2), expected_H_022, 1e-10);
    
    // For y dimension (index 1), only d^2y/dtheta^2 is non-zero
    double expected_H_122 = -speed * std::sin(state(2));  // d^2y/dtheta^2
    EXPECT_NEAR(state_hessian[1](2, 2), expected_H_122, 1e-10);
    
    // For theta dimension (index 2), all second derivatives should be zero
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(state_hessian[2](i, j), 0.0, 1e-10);
        }
    }
}

// Test Dubins car control Hessian
TEST_F(DubinsCarHessianTest, ControlHessian) {
    std::vector<Eigen::MatrixXd> control_hessian = dubins_car->getControlHessian(state, control, 0.0);
    
    // Check dimensions
    EXPECT_EQ(control_hessian.size(), 3);  // Three state dimensions
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(control_hessian[i].rows(), 1);  // 1x1 matrices (control dimension is 1)
        EXPECT_EQ(control_hessian[i].cols(), 1);
    }
    
    // All second derivatives with respect to control should be zero
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(control_hessian[k](0, 0), 0.0, 1e-10);
    }
}

} // namespace testing
} // namespace cddp