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

#include <iostream>
#include <vector>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/dynamics_model/mrp_attitude.hpp"
#include "cddp-cpp/cddp_core/helper.hpp"

#include "matplot/matplot.h"

using namespace cddp;
using namespace matplot;

// Helper function for skew-symmetric matrix (double)
Eigen::Matrix3d skew_double(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return S;
}

// Helper function for MRP kinematics matrix B(mrp) (double)
Eigen::Matrix3d mrpKinematicsMatrix_double(const Eigen::Vector3d& mrp) {
    double mrp_norm_sq = mrp.squaredNorm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    return (1.0 - mrp_norm_sq) * I + 2.0 * skew_double(mrp) + 2.0 * mrp * mrp.transpose();
}

// Helper function to compute state Jacobian using finite difference
Eigen::MatrixXd computeStateJacobianFD(const cddp::DynamicalSystem& model,
                                     const Eigen::VectorXd& state,
                                     const Eigen::VectorXd& control,
                                     double epsilon = 1e-7) {
    int state_dim = model.getStateDim();
    Eigen::MatrixXd A_numerical(state_dim, state_dim);
    Eigen::VectorXd state_perturbed = state;
    Eigen::VectorXd f_plus, f_minus;

    for (int i = 0; i < state_dim; ++i) {
        // Perturb state element i positively
        state_perturbed = state;
        state_perturbed(i) += epsilon;
        f_plus = model.getContinuousDynamics(state_perturbed, control); // Use continuous dynamics

        // Perturb state element i negatively
        state_perturbed = state;
        state_perturbed(i) -= epsilon;
        f_minus = model.getContinuousDynamics(state_perturbed, control); // Use continuous dynamics

        // Compute finite difference approximation for column i
        A_numerical.col(i) = (f_plus - f_minus) / (2.0 * epsilon);
    }
    return A_numerical;
}

// Helper function to compute control Jacobian using finite difference
Eigen::MatrixXd computeControlJacobianFD(const cddp::DynamicalSystem& model,
                                       const Eigen::VectorXd& state,
                                       const Eigen::VectorXd& control,
                                       double epsilon = 1e-7) {
    int state_dim = model.getStateDim();
    int control_dim = model.getControlDim();
    Eigen::MatrixXd B_numerical(state_dim, control_dim);
    Eigen::VectorXd control_perturbed = control;
    Eigen::VectorXd f_plus, f_minus;

    for (int i = 0; i < control_dim; ++i) {
        // Perturb control element i positively
        control_perturbed = control;
        control_perturbed(i) += epsilon;
        f_plus = model.getContinuousDynamics(state, control_perturbed); // Use continuous dynamics

        // Perturb control element i negatively
        control_perturbed = control;
        control_perturbed(i) -= epsilon;
        f_minus = model.getContinuousDynamics(state, control_perturbed); // Use continuous dynamics

        // Compute finite difference approximation for column i
        B_numerical.col(i) = (f_plus - f_minus) / (2.0 * epsilon);
    }
    return B_numerical;
}

class MrpAttitudeTest : public ::testing::Test {
protected:
    void SetUp() override {
        timestep_ = 0.01;
        inertia_ << 1.0, 0.0, 0.0,
                    0.0, 2.0, 0.0,
                    0.0, 0.0, 3.0;
        model_ = std::make_unique<MrpAttitude>(timestep_, inertia_, "rk4");

        // Test state
        state_ = Eigen::VectorXd::Zero(model_->getStateDim());
        state_ << 0.1, 0.2, 0.3, // mrp
                  0.4, 0.5, 0.6; // omega

        // Test control
        control_ = Eigen::VectorXd::Zero(model_->getControlDim());
        control_ << 0.1, -0.1, 0.2; // tau
    }

    double timestep_;
    Eigen::Matrix3d inertia_;
    std::unique_ptr<MrpAttitude> model_;
    Eigen::VectorXd state_;
    Eigen::VectorXd control_;
};

TEST_F(MrpAttitudeTest, Dimensions) {
    ASSERT_EQ(model_->getStateDim(), 6);
    ASSERT_EQ(model_->getControlDim(), 3);
    ASSERT_DOUBLE_EQ(model_->getTimestep(), timestep_);
    ASSERT_EQ(model_->getIntegrationType(), "rk4");
}

TEST_F(MrpAttitudeTest, ContinuousDynamics) {
    Eigen::VectorXd state_dot = model_->getContinuousDynamics(state_, control_);

    // Manual calculation
    Eigen::Vector3d mrp = state_.segment<3>(0);
    Eigen::Vector3d omega = state_.segment<3>(3);
    Eigen::Vector3d tau = control_.segment<3>(0);

    Eigen::VectorXd expected_state_dot(6);
    expected_state_dot.segment<3>(0) = 0.25 * mrpKinematicsMatrix_double(mrp) * omega;
    expected_state_dot.segment<3>(3) = inertia_.inverse() * (-skew_double(omega) * (inertia_ * omega) + tau);

    ASSERT_EQ(state_dot.size(), 6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(state_dot[i], expected_state_dot[i], 1e-9);
    }
}

TEST_F(MrpAttitudeTest, ContinuousDynamicsAutodiff) {
    // Convert state and control to autodiff types
    VectorXdual2nd state_ad = state_.cast<autodiff::dual2nd>();
    VectorXdual2nd control_ad = control_.cast<autodiff::dual2nd>();

    // Call autodiff dynamics
    VectorXdual2nd state_dot_ad = model_->getContinuousDynamicsAutodiff(state_ad, control_ad);

    // Get standard dynamics for comparison
    Eigen::VectorXd state_dot_double = model_->getContinuousDynamics(state_, control_);

    ASSERT_EQ(state_dot_ad.size(), 6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(state_dot_ad[i].val.val, state_dot_double[i], 1e-9);
    }
}

TEST_F(MrpAttitudeTest, StateJacobianFiniteDifference) {
    // Get analytical Jacobian (calculated via autodiff in the base class)
    Eigen::MatrixXd A_analytical = model_->getStateJacobian(state_, control_);

    // Get numerical Jacobian using finite difference helper
    Eigen::MatrixXd A_numerical = computeStateJacobianFD(*model_, state_, control_);

    ASSERT_EQ(A_analytical.rows(), 6);
    ASSERT_EQ(A_analytical.cols(), 6);
    ASSERT_EQ(A_numerical.rows(), 6);
    ASSERT_EQ(A_numerical.cols(), 6);

    // Compare analytical and numerical Jacobians
    EXPECT_TRUE(A_analytical.isApprox(A_numerical, 1e-6));
}

TEST_F(MrpAttitudeTest, ControlJacobianFiniteDifference) {
    // Get analytical Jacobian (calculated via autodiff in the base class)
    Eigen::MatrixXd B_analytical = model_->getControlJacobian(state_, control_);

    // Get numerical Jacobian using finite difference helper
    Eigen::MatrixXd B_numerical = computeControlJacobianFD(*model_, state_, control_);

    ASSERT_EQ(B_analytical.rows(), 6);
    ASSERT_EQ(B_analytical.cols(), 3);
    ASSERT_EQ(B_numerical.rows(), 6);
    ASSERT_EQ(B_numerical.cols(), 3);

    // Compare analytical and numerical Jacobians
    EXPECT_TRUE(B_analytical.isApprox(B_numerical, 1e-6));
}

TEST_F(MrpAttitudeTest, SimulationAndPlotting) {
    // Simulation parameters
    double simulation_time = 10.0; // seconds
    int num_steps = static_cast<int>(simulation_time / timestep_);

    // Initial state (e.g., zero attitude and angular velocity)
    Eigen::VectorXd current_state = Eigen::VectorXd::Zero(model_->getStateDim());

    // Constant control input (apply a torque around the x-axis)
    Eigen::VectorXd constant_control(model_->getControlDim());
    constant_control << 0.1, 0.0, 0.0;

    // Data storage
    std::vector<double> time_data;
    std::vector<double> mrp_x_data, mrp_y_data, mrp_z_data;
    std::vector<double> omega_x_data, omega_y_data, omega_z_data;

    // Simulation loop
    for (int i = 0; i < num_steps; ++i) {
        // Store data
        time_data.push_back(i * timestep_);
        mrp_x_data.push_back(current_state(MrpAttitude::STATE_MRP_X));
        mrp_y_data.push_back(current_state(MrpAttitude::STATE_MRP_Y));
        mrp_z_data.push_back(current_state(MrpAttitude::STATE_MRP_Z));
        omega_x_data.push_back(current_state(MrpAttitude::STATE_OMEGA_X));
        omega_y_data.push_back(current_state(MrpAttitude::STATE_OMEGA_Y));
        omega_z_data.push_back(current_state(MrpAttitude::STATE_OMEGA_Z));

        // Get next state
        current_state = model_->getDiscreteDynamics(current_state, constant_control);
    }

    // Add final state
    time_data.push_back(num_steps * timestep_);
    mrp_x_data.push_back(current_state(MrpAttitude::STATE_MRP_X));
    mrp_y_data.push_back(current_state(MrpAttitude::STATE_MRP_Y));
    mrp_z_data.push_back(current_state(MrpAttitude::STATE_MRP_Z));
    omega_x_data.push_back(current_state(MrpAttitude::STATE_OMEGA_X));
    omega_y_data.push_back(current_state(MrpAttitude::STATE_OMEGA_Y));
    omega_z_data.push_back(current_state(MrpAttitude::STATE_OMEGA_Z));

    // Plotting
    auto fig = figure(true);
    fig->size(1200, 800);

    // Plot MRP components
    auto ax1 = subplot(2, 1, 0);
    hold(ax1, on);
    plot(ax1, time_data, mrp_x_data, "-r")->line_width(2).display_name("MRP X");
    plot(ax1, time_data, mrp_y_data, "-g")->line_width(2).display_name("MRP Y");
    plot(ax1, time_data, mrp_z_data, "-b")->line_width(2).display_name("MRP Z");
    title(ax1, "MRP Components vs Time");
    xlabel(ax1, "Time [s]");
    ylabel(ax1, "MRP Value");
    legend(ax1);
    grid(ax1, on);
    hold(ax1, off);

    // Plot Angular Velocities
    auto ax2 = subplot(2, 1, 1);
    hold(ax2, on);
    plot(ax2, time_data, omega_x_data, "-r")->line_width(2).display_name("Omega X");
    plot(ax2, time_data, omega_y_data, "-g")->line_width(2).display_name("Omega Y");
    plot(ax2, time_data, omega_z_data, "-b")->line_width(2).display_name("Omega Z");
    title(ax2, "Angular Velocity vs Time");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Angular Velocity [rad/s]");
    legend(ax2);
    grid(ax2, on);
    hold(ax2, off);

    // Show the plot (disable saving for tests unless specifically needed)
    // show(); 
    // If you want to save instead:
    // const std::string plotDirectory = "../results/tests";
    // if (!std::filesystem::exists(plotDirectory)) {
    //     std::filesystem::create_directories(plotDirectory);
    // }
    // save(fig, plotDirectory + "/mrp_attitude_simulation.png");

    // Basic assertion: Check if simulation completed
    ASSERT_EQ(time_data.size(), num_steps + 1);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
} 