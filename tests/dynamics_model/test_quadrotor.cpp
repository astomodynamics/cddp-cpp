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
// Description: Test the quaternion-based quadrotor dynamics model.

#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/dynamics_model/quadrotor.hpp"
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

// Helper: Compute rotation matrix from quaternion [qw, qx, qy, qz]
Eigen::Matrix3d getRotationMatrixFromQuaternion(double qw, double qx, double qy, double qz) {
    Eigen::Matrix3d R;
    R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
    R(0, 1) = 2 * (qx * qy - qz * qw);
    R(0, 2) = 2 * (qx * qz + qy * qw);
    
    R(1, 0) = 2 * (qx * qy + qz * qw);
    R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
    R(1, 2) = 2 * (qy * qz - qx * qw);
    
    R(2, 0) = 2 * (qx * qz - qy * qw);
    R(2, 1) = 2 * (qy * qz + qx * qw);
    R(2, 2) = 1 - 2 * (qx * qx + qy * qy);
    
    return R;
}

// Transform quadrotor frame points (motor locations) to world coordinates using the quaternion.
std::vector<std::vector<double>> transformQuadrotorFrame(
    const Eigen::Vector3d& position,
    const Eigen::Vector4d& quat,  // [qw, qx, qy, qz]
    double arm_length) {
    
    // Define quadrotor frame points in body frame
    std::vector<Eigen::Vector3d> body_points = {
        Eigen::Vector3d(arm_length, 0, 0),    // Front (motor 0)
        Eigen::Vector3d(0, arm_length, 0),      // Right (motor 1)
        Eigen::Vector3d(-arm_length, 0, 0),     // Back (motor 2)
        Eigen::Vector3d(0, -arm_length, 0)       // Left (motor 3)
    };
    
    // Compute rotation matrix from quaternion
    Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat[0], quat[1], quat[2], quat[3]);
    
    // Transform points to world frame
    std::vector<std::vector<double>> world_points(3, std::vector<double>());
    for (const auto& point : body_points) {
        Eigen::Vector3d world_point = position + R * point;
        world_points[0].push_back(world_point.x());
        world_points[1].push_back(world_point.y());
        world_points[2].push_back(world_point.z());
    }
    
    return world_points;
}

TEST(QuadrotorTest, DiscreteDynamics) {
    // Create a quaternion-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;         // 1 kg
    double arm_length = 0.2;   // 20 cm
    
    // Diagonal inertia matrix for a symmetric quadrotor
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
               
    std::string integration_type = "rk4";
    Quadrotor quadrotor(timestep, mass, inertia, arm_length, integration_type);

    // Store states for plotting (time, position, and Euler angles for visualization)
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> phi_data, theta_data, psi_data;

    // Initial state: hovering with identity quaternion
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(0) = 0.0;  // x
    state(1) = 0.0;  // y
    state(2) = 1.0;  // z = 1m height
    state(3) = 1.0;
    state(4) = 0.0;
    state(5) = 0.0;
    state(6) = 0.0;

    // Control input for hover (each motor provides mg/4 force)
    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Simulate for a few steps
    int num_steps = 500;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        z_data.push_back(state[2]);
        // Compute Euler angles from quaternion for visualization
        Eigen::Vector4d quat(state(3), state(4), state(5), state(6));
        Eigen::Vector3d euler = quaternionToEuler(quat[0], quat[1], quat[2], quat[3]);
        phi_data.push_back(euler[0]);
        theta_data.push_back(euler[1]);
        psi_data.push_back(euler[2]);

        // Compute the next state
        state = quadrotor.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(quadrotor.getStateDim(), 13);
    ASSERT_EQ(quadrotor.getControlDim(), 4);
    ASSERT_DOUBLE_EQ(quadrotor.getTimestep(), 0.01);
    ASSERT_EQ(quadrotor.getIntegrationType(), "rk4");

    // Test if quadrotor maintains approximate hover height
    double final_height = z_data.back();
    EXPECT_NEAR(final_height, 1.0, 0.1);
}

TEST(QuadrotorTest, ContinuousDynamics) {
    // Create quaternion-based quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
    Quadrotor quadrotor(timestep, mass, inertia, arm_length, "euler");

    // Test hover condition
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(2) = 1.0;  // 1m height
    // Set identity quaternion [1, 0, 0, 0]
    state(3) = 1.0;
    state(4) = 0.0;
    state(5) = 0.0;
    state(6) = 0.0;

    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Get continuous dynamics
    Eigen::VectorXd state_dot = quadrotor.getContinuousDynamics(state, control);

    // For hover: position derivative should equal the velocity (which is zero)
    EXPECT_NEAR(state_dot[0], 0.0, 1e-10);  // dx/dt
    EXPECT_NEAR(state_dot[1], 0.0, 1e-10);  // dy/dt
    EXPECT_NEAR(state_dot[2], 0.0, 1e-10);  // dz/dt

    // Quaternion derivative should be near zero (zero angular velocity)
    EXPECT_NEAR(state_dot[3], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[4], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[5], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[6], 0.0, 1e-10);

    // Linear acceleration should be near zero (hover condition)
    EXPECT_NEAR(state_dot[7], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[8], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[9], 0.0, 1e-10);

    // Angular acceleration should be zero
    EXPECT_NEAR(state_dot[10], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[11], 0.0, 1e-10);
    EXPECT_NEAR(state_dot[12], 0.0, 1e-10);

    // Test with unbalanced thrust (should create a nonzero angular acceleration)
    control(0) += 0.1;  // Increase front rotor thrust
    control(2) -= 0.1;  // Decrease back rotor thrust
    state_dot = quadrotor.getContinuousDynamics(state, control);
    
    // Check that the angular acceleration in x (roll) is non-zero
    EXPECT_GT(std::abs(state_dot[10]), 0.0);
}

TEST(QuadrotorTest, StateJacobianFiniteDifference) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
    Quadrotor quadrotor(timestep, mass, inertia, arm_length, "euler");

    // Test state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(2) = 1.0;  // 1m height
    // Set identity quaternion [1, 0, 0, 0]
    state(3) = 1.0;
    state(4) = 0.0;
    state(5) = 0.0;
    state(6) = 0.0;

    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Get analytical Jacobian
    Eigen::MatrixXd A_analytical = quadrotor.getStateJacobian(state, control);

    // Get numerical Jacobian
    auto f_A = [&](const Eigen::VectorXd& x) {
        return quadrotor.getContinuousDynamics(x, control);
    };
    Eigen::MatrixXd A_numerical = finite_difference_jacobian(f_A, state);

    // Test dimensions
    ASSERT_EQ(A_analytical.rows(), 13);
    ASSERT_EQ(A_analytical.cols(), 13);

    // Compare analytical and numerical Jacobians
    double tolerance = 1e-4;
    EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);

    // Test with non-zero velocities and rotations
    state(4) = 0.1;  // non-zero quaternion components
    state(5) = 0.1;
    state(6) = 0.1;
    state(7) = 0.2;  // non-zero linear velocities
    state(8) = 0.2;
    state(9) = 0.2;
    state(10) = 0.1;  // non-zero angular velocities
    state(11) = 0.1;
    state(12) = 0.1;
    
    // Normalize quaternion
    double qnorm = std::sqrt(state(3)*state(3) + state(4)*state(4) + 
                             state(5)*state(5) + state(6)*state(6));
    state(3) /= qnorm;
    state(4) /= qnorm;
    state(5) /= qnorm;
    state(6) /= qnorm;

    // Get Jacobians for non-zero state
    A_analytical = quadrotor.getStateJacobian(state, control);
    A_numerical = finite_difference_jacobian(f_A, state);
    
    // Compare again
    EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);
}

TEST(QuadrotorTest, ControlJacobianFiniteDifference) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
    Quadrotor quadrotor(timestep, mass, inertia, arm_length, "euler");

    // Test state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(2) = 1.0;  // 1m height
    // Set identity quaternion [1, 0, 0, 0]
    state(3) = 1.0;
    state(4) = 0.0;
    state(5) = 0.0;
    state(6) = 0.0;

    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Get analytical Jacobian
    Eigen::MatrixXd B_analytical = quadrotor.getControlJacobian(state, control);

    // Get numerical Jacobian
    auto f_B = [&](const Eigen::VectorXd& u) {
        return quadrotor.getContinuousDynamics(state, u);
    };
    Eigen::MatrixXd B_numerical = finite_difference_jacobian(f_B, control);

    // Test dimensions
    ASSERT_EQ(B_analytical.rows(), 13);
    ASSERT_EQ(B_analytical.cols(), 4);

    // Compare analytical and numerical Jacobians
    double tolerance = 1e-4;
    EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);

    // Test with non-zero velocities and rotations
    state(4) = 0.1;  // non-zero quaternion components
    state(5) = 0.1;
    state(6) = 0.1;
    state(7) = 0.2;  // non-zero linear velocities
    state(8) = 0.2;
    state(9) = 0.2;
    state(10) = 0.1;  // non-zero angular velocities
    state(11) = 0.1;
    state(12) = 0.1;
    
    // Normalize quaternion
    double qnorm = std::sqrt(state(3)*state(3) + state(4)*state(4) + 
                             state(5)*state(5) + state(6)*state(6));
    state(3) /= qnorm;
    state(4) /= qnorm;
    state(5) /= qnorm;
    state(6) /= qnorm;

    // Also try non-uniform control
    control << hover_thrust*1.1, hover_thrust*0.9, hover_thrust*1.1, hover_thrust*0.9;

    // Get Jacobians for non-zero state and non-uniform control
    B_analytical = quadrotor.getControlJacobian(state, control);
    B_numerical = finite_difference_jacobian(f_B, control);
    
    // Compare again
    EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);
}

TEST(QuadrotorTest, StateJacobianAutodiff) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
    Quadrotor quadrotor(timestep, mass, inertia, arm_length, "euler");

    // Test state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(2) = 1.0;  // 1m height
    // Set identity quaternion [1, 0, 0, 0]
    state(3) = 1.0;
    state(4) = 0.0;
    state(5) = 0.0;
    state(6) = 0.0;

    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Get Jacobian using autodiff (which is the default implementation)
    Eigen::MatrixXd A_autodiff = quadrotor.getStateJacobian(state, control);

    // Get numerical Jacobian for comparison
    auto f_A = [&](const Eigen::VectorXd& x) {
        return quadrotor.getContinuousDynamics(x, control);
    };
    Eigen::MatrixXd A_numerical = finite_difference_jacobian(f_A, state);

    // Compare autodiff and numerical Jacobians
    double tolerance = 1e-4;
    EXPECT_NEAR((A_autodiff - A_numerical).norm(), 0.0, tolerance);

    // Try non-zero state
    state(4) = 0.1;  // non-zero quaternion components
    state(5) = 0.1;
    state(6) = 0.1;
    state(7) = 0.2;  // non-zero linear velocities
    state(8) = 0.2;
    state(9) = 0.2;
    state(10) = 0.1;  // non-zero angular velocities
    state(11) = 0.1;
    state(12) = 0.1;
    
    // Normalize quaternion
    double qnorm = std::sqrt(state(3)*state(3) + state(4)*state(4) + 
                             state(5)*state(5) + state(6)*state(6));
    state(3) /= qnorm;
    state(4) /= qnorm;
    state(5) /= qnorm;
    state(6) /= qnorm;

    // Get Jacobians for non-zero state
    A_autodiff = quadrotor.getStateJacobian(state, control);
    A_numerical = finite_difference_jacobian(f_A, state);
    
    // Compare again
    EXPECT_NEAR((A_autodiff - A_numerical).norm(), 0.0, tolerance);
}

TEST(QuadrotorTest, ControlJacobianAutodiff) {
    // Create quadrotor instance
    double timestep = 0.01;
    double mass = 1.0;
    double arm_length = 0.2;
    Eigen::Matrix3d inertia;
    inertia << 0.01, 0, 0,
               0, 0.01, 0,
               0, 0, 0.02;
    Quadrotor quadrotor(timestep, mass, inertia, arm_length, "euler");

    // Test state and control
    Eigen::VectorXd state = Eigen::VectorXd::Zero(13);
    state(2) = 1.0;  // 1m height
    // Set identity quaternion [1, 0, 0, 0]
    state(3) = 1.0;
    state(4) = 0.0;
    state(5) = 0.0;
    state(6) = 0.0;

    double hover_thrust = mass * 9.81 / 4.0;
    Eigen::VectorXd control(4);
    control << hover_thrust, hover_thrust, hover_thrust, hover_thrust;

    // Get Jacobian using autodiff (which is the default implementation)
    Eigen::MatrixXd B_autodiff = quadrotor.getControlJacobian(state, control);

    // Get numerical Jacobian for comparison
    auto f_B = [&](const Eigen::VectorXd& u) {
        return quadrotor.getContinuousDynamics(state, u);
    };
    Eigen::MatrixXd B_numerical = finite_difference_jacobian(f_B, control);

    // Compare autodiff and numerical Jacobians
    double tolerance = 1e-4;
    EXPECT_NEAR((B_autodiff - B_numerical).norm(), 0.0, tolerance);

    // Try non-zero state and non-uniform control
    state(4) = 0.1;  // non-zero quaternion components
    state(5) = 0.1;
    state(6) = 0.1;
    state(7) = 0.2;  // non-zero linear velocities
    state(8) = 0.2;
    state(9) = 0.2;
    state(10) = 0.1;  // non-zero angular velocities
    state(11) = 0.1;
    state(12) = 0.1;
    
    // Normalize quaternion
    double qnorm = std::sqrt(state(3)*state(3) + state(4)*state(4) + 
                             state(5)*state(5) + state(6)*state(6));
    state(3) /= qnorm;
    state(4) /= qnorm;
    state(5) /= qnorm;
    state(6) /= qnorm;

    // Also try non-uniform control
    control << hover_thrust*1.1, hover_thrust*0.9, hover_thrust*1.1, hover_thrust*0.9;

    // Get Jacobians for non-zero state and non-uniform control
    B_autodiff = quadrotor.getControlJacobian(state, control);
    B_numerical = finite_difference_jacobian(f_B, control);
    
    // Compare again
    EXPECT_NEAR((B_autodiff - B_numerical).norm(), 0.0, tolerance);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}