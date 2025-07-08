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
// Description: Test the rate-based quadrotor dynamics model.

#include <iostream>
#include <vector>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/dynamics_model/quadrotor_rate.hpp"
#include "cddp-cpp/cddp_core/helper.hpp"
using namespace cddp;

// Helper: Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw)
Eigen::Vector3d quaternionToEuler(double qw, double qx, double qy, double qz) {
    // Roll (phi)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    double phi = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (theta)
    double sinp = 2.0 * (qw * qy - qz * qx);
    double theta;
    if (std::abs(sinp) >= 1)
        theta = std::copysign(M_PI / 2.0, sinp);
    else
        theta = std::asin(sinp);

    // Yaw (psi)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double psi = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(phi, theta, psi);
}

TEST(QuadrotorRateTest, BasicProperties) {
    // Create a rate-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;         // 1 kg
    double max_thrust = 20.0;  // 20 N
    double max_rate = 0.5;     // 0.5 rad/s
    std::string integration_type = "rk4";
    
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, integration_type);

    // Basic assertions
    ASSERT_EQ(quadrotor.getStateDim(), 10);
    ASSERT_EQ(quadrotor.getControlDim(), 4);
    ASSERT_DOUBLE_EQ(quadrotor.getTimestep(), 0.01);
    ASSERT_EQ(quadrotor.getIntegrationType(), "rk4");
    ASSERT_DOUBLE_EQ(quadrotor.getMass(), mass);
    ASSERT_DOUBLE_EQ(quadrotor.getMaxThrust(), max_thrust);
    ASSERT_DOUBLE_EQ(quadrotor.getMaxRate(), max_rate);
}

TEST(QuadrotorRateTest, HoverDynamics) {
    // Create rate-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, "euler");

    // Test hover condition
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state(2) = 1.0;  // 1m height
    // Set identity quaternion [1, 0, 0, 0]
    state(6) = 1.0;  // qw
    state(7) = 0.0;  // qx
    state(8) = 0.0;  // qy
    state(9) = 0.0;  // qz

    // Control for hover: thrust = mg, zero angular rates
    double hover_thrust = mass * 9.81;
    Eigen::VectorXd control(4);
    control << hover_thrust, 0.0, 0.0, 0.0;  // [thrust, wx, wy, wz]

    // Get continuous dynamics
    Eigen::VectorXd state_dot = quadrotor.getContinuousDynamics(state, control, 0.0);

    // For hover: position derivative should equal the velocity (which is zero)
    EXPECT_NEAR(state_dot[0], 0.0, 1e-10);  // dx/dt
    EXPECT_NEAR(state_dot[1], 0.0, 1e-10);  // dy/dt
    EXPECT_NEAR(state_dot[2], 0.0, 1e-10);  // dz/dt

    // Linear acceleration should be near zero (hover condition)
    EXPECT_NEAR(state_dot[3], 0.0, 1e-10);  // dvx/dt
    EXPECT_NEAR(state_dot[4], 0.0, 1e-10);  // dvy/dt
    EXPECT_NEAR(state_dot[5], 0.0, 1e-10);  // dvz/dt

    // Quaternion derivative should be near zero (zero angular velocity)
    EXPECT_NEAR(state_dot[6], 0.0, 1e-10);  // dqw/dt
    EXPECT_NEAR(state_dot[7], 0.0, 1e-10);  // dqx/dt
    EXPECT_NEAR(state_dot[8], 0.0, 1e-10);  // dqy/dt
    EXPECT_NEAR(state_dot[9], 0.0, 1e-10);  // dqz/dt
}

TEST(QuadrotorRateTest, RotationDynamics) {
    // Create rate-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, "euler");

    // Initial state: hovering with identity quaternion
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state(2) = 1.0;  // 1m height
    state(6) = 1.0;  // qw = 1 (identity quaternion)

    // Apply roll rate
    double hover_thrust = mass * 9.81;
    Eigen::VectorXd control(4);
    control << hover_thrust, 0.1, 0.0, 0.0;  // [thrust, wx=0.1, wy=0, wz=0]

    // Get continuous dynamics
    Eigen::VectorXd state_dot = quadrotor.getContinuousDynamics(state, control, 0.0);

    // Check that quaternion is changing due to angular rate
    EXPECT_GT(std::abs(state_dot[7]), 0.0);  // qx should be changing
    EXPECT_NEAR(state_dot[6], 0.0, 0.01);   // qw should change slightly
    EXPECT_NEAR(state_dot[8], 0.0, 1e-10);  // qy should be zero
    EXPECT_NEAR(state_dot[9], 0.0, 1e-10);  // qz should be zero
}

TEST(QuadrotorRateTest, ThrustDynamics) {
    // Create rate-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, "euler");

    // Initial state: at origin with identity quaternion
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state(6) = 1.0;  // qw = 1 (identity quaternion)

    // Apply thrust greater than hover
    double hover_thrust = mass * 9.81;
    Eigen::VectorXd control(4);
    control << hover_thrust * 1.5, 0.0, 0.0, 0.0;  // 150% hover thrust

    // Get continuous dynamics
    Eigen::VectorXd state_dot = quadrotor.getContinuousDynamics(state, control, 0.0);

    // Check vertical acceleration is positive (upward)
    EXPECT_GT(state_dot[5], 0.0);  // dvz/dt should be positive
    
    // Horizontal accelerations should be zero (no tilt)
    EXPECT_NEAR(state_dot[3], 0.0, 1e-10);  // dvx/dt
    EXPECT_NEAR(state_dot[4], 0.0, 1e-10);  // dvy/dt
}

TEST(QuadrotorRateTest, StateJacobianFiniteDifference) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, "euler");

    // Test state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state(2) = 1.0;  // 1m height
    state(6) = 1.0;  // qw = 1

    double hover_thrust = mass * 9.81;
    Eigen::VectorXd control(4);
    control << hover_thrust, 0.0, 0.0, 0.0;

    // Get analytical Jacobian
    Eigen::MatrixXd A_analytical = quadrotor.getStateJacobian(state, control, 0.0);

    // Get numerical Jacobian
    auto f_A = [&](const Eigen::VectorXd& x) {
        return quadrotor.getContinuousDynamics(x, control, 0.0);
    };
    Eigen::MatrixXd A_numerical = finite_difference_jacobian(f_A, state);

    // Test dimensions
    ASSERT_EQ(A_analytical.rows(), 10);
    ASSERT_EQ(A_analytical.cols(), 10);

    // Compare analytical and numerical Jacobians
    double tolerance = 1e-4;
    EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);

    // Test with non-zero velocities and rotations
    state(3) = 0.2;  // non-zero velocities
    state(4) = 0.2;
    state(5) = 0.2;
    state(7) = 0.1;  // non-zero quaternion components
    state(8) = 0.1;
    state(9) = 0.1;
    
    // Normalize quaternion
    double qnorm = std::sqrt(state(6)*state(6) + state(7)*state(7) + 
                             state(8)*state(8) + state(9)*state(9));
    state(6) /= qnorm;
    state(7) /= qnorm;
    state(8) /= qnorm;
    state(9) /= qnorm;

    // Get Jacobians for non-zero state
    A_analytical = quadrotor.getStateJacobian(state, control, 0.0);
    A_numerical = finite_difference_jacobian(f_A, state);
    
    // Compare again
    EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);
}

TEST(QuadrotorRateTest, ControlJacobianFiniteDifference) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, "euler");

    // Test state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state(2) = 1.0;  // 1m height
    state(6) = 1.0;  // qw = 1

    double hover_thrust = mass * 9.81;
    Eigen::VectorXd control(4);
    control << hover_thrust, 0.0, 0.0, 0.0;

    // Get analytical Jacobian
    Eigen::MatrixXd B_analytical = quadrotor.getControlJacobian(state, control, 0.0);

    // Get numerical Jacobian
    auto f_B = [&](const Eigen::VectorXd& u) {
        return quadrotor.getContinuousDynamics(state, u, 0.0);
    };
    Eigen::MatrixXd B_numerical = finite_difference_jacobian(f_B, control);

    // Test dimensions
    ASSERT_EQ(B_analytical.rows(), 10);
    ASSERT_EQ(B_analytical.cols(), 4);

    // Compare analytical and numerical Jacobians
    double tolerance = 1e-4;
    EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);

    // Test with non-zero state and control
    state(3) = 0.2;  // non-zero velocities
    state(4) = 0.2;
    state(5) = 0.2;
    state(7) = 0.1;  // non-zero quaternion components
    state(8) = 0.1;
    state(9) = 0.1;
    
    // Normalize quaternion
    double qnorm = std::sqrt(state(6)*state(6) + state(7)*state(7) + 
                             state(8)*state(8) + state(9)*state(9));
    state(6) /= qnorm;
    state(7) /= qnorm;
    state(8) /= qnorm;
    state(9) /= qnorm;

    // Non-zero angular rates
    control << hover_thrust*1.1, 0.05, -0.03, 0.02;

    // Get Jacobians for non-zero state and control
    B_analytical = quadrotor.getControlJacobian(state, control, 0.0);
    B_numerical = finite_difference_jacobian(f_B, control);
    
    // Compare again
    EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);
}

TEST(QuadrotorRateTest, DiscreteDynamicsSimulation) {
    // Create a rate-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    std::string integration_type = "rk4";
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate, integration_type);

    // Store states for analysis
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> phi_data, theta_data, psi_data;

    // Initial state: hovering at 1m with identity quaternion
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state(0) = 0.0;  // x
    state(1) = 0.0;  // y
    state(2) = 1.0;  // z = 1m height
    state(6) = 1.0;  // qw = 1 (identity quaternion)

    // Control input: hover with slight roll rate
    double hover_thrust = mass * 9.81;
    Eigen::VectorXd control(4);
    control << hover_thrust, 0.05, 0.0, 0.0;  // Small roll rate

    // Simulate for a few seconds
    int num_steps = 200;  // 2 seconds
    for (int i = 0; i < num_steps; ++i) {
        // Store data
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        z_data.push_back(state[2]);
        
        // Compute Euler angles from quaternion for visualization
        Eigen::Vector4d quat(state(6), state(7), state(8), state(9));
        Eigen::Vector3d euler = quaternionToEuler(quat[0], quat[1], quat[2], quat[3]);
        phi_data.push_back(euler[0]);
        theta_data.push_back(euler[1]);
        psi_data.push_back(euler[2]);

        // Compute the next state
        state = quadrotor.getDiscreteDynamics(state, control, 0.0);
    }

    // Test if quadrotor maintains approximate hover height
    double final_height = z_data.back();
    EXPECT_NEAR(final_height, 1.0, 0.05);  // Within 5cm of initial height

    // Test if roll angle increased due to roll rate
    double final_roll = phi_data.back();
    EXPECT_GT(final_roll, 0.05);  // Should have rolled
}

TEST(QuadrotorRateTest, InvalidParameters) {
    double timestep = 0.01;
    
    // Test negative mass
    EXPECT_THROW(QuadrotorRate(timestep, -1.0, 20.0, 0.5), std::invalid_argument);
    
    // Test zero mass
    EXPECT_THROW(QuadrotorRate(timestep, 0.0, 20.0, 0.5), std::invalid_argument);
    
    // Test negative max thrust
    EXPECT_THROW(QuadrotorRate(timestep, 1.0, -20.0, 0.5), std::invalid_argument);
    
    // Test negative max rate
    EXPECT_THROW(QuadrotorRate(timestep, 1.0, 20.0, -0.5), std::invalid_argument);
}

TEST(QuadrotorRateTest, InvalidDimensions) {
    double timestep = 0.01;
    double mass = 1.0;
    double max_thrust = 20.0;
    double max_rate = 0.5;
    QuadrotorRate quadrotor(timestep, mass, max_thrust, max_rate);
    
    // Test wrong state dimension
    Eigen::VectorXd wrong_state(8);  // Should be 10
    Eigen::VectorXd control(4);
    EXPECT_THROW(quadrotor.getContinuousDynamics(wrong_state, control, 0.0), std::invalid_argument);
    
    // Test wrong control dimension
    Eigen::VectorXd state(10);
    Eigen::VectorXd wrong_control(3);  // Should be 4
    EXPECT_THROW(quadrotor.getContinuousDynamics(state, wrong_control, 0.0), std::invalid_argument);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}