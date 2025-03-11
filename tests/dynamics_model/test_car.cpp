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
// Description: Test the car dynamics model.
#include <iostream>
#include <vector>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/car.hpp"
#include "cddp_core/objective.hpp"

using namespace cddp;

TEST(CarTest, DiscreteDynamics) {
    // Create a car instance
    double timestep = 0.03;  // From original MATLAB code
    double wheelbase = 2.0;  // From original MATLAB code
    std::string integration_type = "euler";
    cddp::Car car(timestep, wheelbase, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, theta_data, v_data;

    // Initial state and control (from MATLAB demo)
    Eigen::VectorXd state(4);
    state << 1.0, 1.0, 3*M_PI/2, 0.0;  // Initial state from MATLAB demo
    Eigen::VectorXd control(2);
    control << 0.1, 0.1; // Small steering angle and acceleration

    // Simulate for a few steps
    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        theta_data.push_back(state[2]);
        v_data.push_back(state[3]);

        // Compute the next state
        state = car.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(car.getStateDim(), 4);
    ASSERT_EQ(car.getControlDim(), 2);
    ASSERT_DOUBLE_EQ(car.getTimestep(), 0.03);
    ASSERT_EQ(car.getIntegrationType(), "euler");

    // First Step test
    state << 1.0, 1.0, 3*M_PI/2, 0.0; 
    control << 0.01, 0.01;
    state = car.getDiscreteDynamics(state, control);
    EXPECT_NEAR(state[0], 1.0, 1e-4);
    EXPECT_NEAR(state[1], 1.0, 1e-4);
    EXPECT_NEAR(state[2], 4.7124, 1e-4);
    EXPECT_NEAR(state[3], 0.0003, 1e-4);

    // Second Step test
    state << 1.0, 1.0, 3*M_PI/2, 1.0;
    control << 0.3, 0.1;
    state = car.getDiscreteDynamics(state, control);
    EXPECT_NEAR(state[0], 1.0, 1e-4);
    EXPECT_NEAR(state[1], 0.9713, 1e-4);
    EXPECT_NEAR(state[2], 4.7168, 1e-4);
    EXPECT_NEAR(state[3], 1.0030, 1e-4);
}

TEST(CarTest, JacobianTest) {
    // Create a car instance
    double timestep = 0.03;  // From original MATLAB code
    double wheelbase = 2.0;  // From original MATLAB code
    std::string integration_type = "euler";
    cddp::Car car(timestep, wheelbase, integration_type);

    // Initial state and control (from MATLAB demo)
    Eigen::VectorXd state(4);
    state << 1.0, 1.0, 3*M_PI/2, 0.0;  // Initial state from MATLAB demo
    Eigen::VectorXd control(2);
    control << 0.01, 0.01; // Small steering angle and acceleration

    // Compute the Jacobians
    Eigen::MatrixXd A_numerical = car.getStateJacobian(state, control);
    A_numerical *= timestep;
    A_numerical.diagonal().array() += 1.0; 
    
    Eigen::MatrixXd B_numerical = car.getControlJacobian(state, control);
    B_numerical *= timestep;

    Eigen::MatrixXd A_known(4, 4);
    Eigen::MatrixXd B_known(4, 2);

    // Test values
    A_known << 1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, -0.03,
               0.0, 0.0, 1.0, 0.0001,
               0.0, 0.0, 0.0, 1.0;

    B_known << 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.03;
    
    // Compare values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(A_numerical(i, j), A_known(i, j), 1e-4);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(B_numerical(i, j), B_known(i, j), 1e-4);
        }
    }

    // Next test
    state << 1.0, 1.0, 3*M_PI/2, 1.0;
    control << 0.3, 0.1;

    A_numerical = car.getStateJacobian(state, control);
    A_numerical *= timestep;
    A_numerical.diagonal().array() += 1.0;

    B_numerical = car.getControlJacobian(state, control);
    B_numerical *= timestep;
    
    A_known << 1.0, 0.0, 0.0287, 0.0,
               0.0, 1.0, 0.0, -0.0287,
               0.0, 0.0, 1.0, 0.0044,
               0.0, 0.0, 0.0, 1.0;

    B_known << 0.0, 0.0,
                0.0087, 0.0,
                0.0143, 0.0,
                0.0, 0.03;

    // Compare values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(A_numerical(i, j), A_known(i, j), 1e-4);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(B_numerical(i, j), B_known(i, j), 1e-4);
        }
    }
}

namespace cddp {
class CarParkingObjective : public NonlinearObjective {
public:
    CarParkingObjective(const Eigen::VectorXd& goal_state, double timestep)
        : NonlinearObjective(timestep), reference_state_(goal_state) {
        // Control cost coefficients: cu = 1e-2*[1 .01]
        cu_ = Eigen::Vector2d(1e-2, 1e-4);

        // Final cost coefficients: cf = [.1 .1 1 .3]
        cf_ = Eigen::Vector4d(0.1, 0.1, 1.0, 0.3);

        // Smoothness scales for final cost: pf = [.01 .01 .01 1]
        pf_ = Eigen::Vector4d(0.01, 0.01, 0.01, 1.0);

        // Running cost coefficients: cx = 1e-3*[1 1]
        cx_ = Eigen::Vector2d(1e-3, 1e-3);

        // Smoothness scales for running cost: px = [.1 .1]
        px_ = Eigen::Vector2d(0.1, 0.1);
    }

    double running_cost(const Eigen::VectorXd& state, 
                       const Eigen::VectorXd& control, 
                       int index) const override {
        // Control cost: lu = cu*u.^2
        double lu = cu_.dot(control.array().square().matrix());

        // Running cost on distance from origin: lx = cx*sabs(x(1:2,:),px)
        Eigen::VectorXd xy_state = state.head(2);
        double lx = cx_.dot(sabs(xy_state, px_));

        return lu + lx;
    }

    double terminal_cost(const Eigen::VectorXd& final_state) const override {
        // Final state cost: llf = cf*sabs(x(:,final),pf);
        return cf_.dot(sabs(final_state - reference_state_, pf_));
    }

private:
    // Helper function for smooth absolute value (pseudo-Huber)
    Eigen::VectorXd sabs(const Eigen::VectorXd& x, const Eigen::VectorXd& p) const {
        return ((x.array().square() / p.array().square() + 1.0).sqrt() * p.array() - p.array()).matrix();
    }

    Eigen::VectorXd reference_state_;
    Eigen::Vector2d cu_; // Control cost coefficients
    Eigen::Vector4d cf_; // Final cost coefficients
    Eigen::Vector4d pf_; // Smoothness scales for final cost 
    Eigen::Vector2d cx_; // Running cost coefficients
    Eigen::Vector2d px_; // Smoothness scales for running cost
};
} // namespace cddp

TEST(CarTest , ObjectiveTest) {
    // Create a car instance
    double timestep = 0.03;  // From original MATLAB code
    double wheelbase = 2.0;  // From original MATLAB code
    std::string integration_type = "euler";
    cddp::Car car(timestep, wheelbase, integration_type);

    // Create the objective
    Eigen::VectorXd goal_state(4);
    goal_state << 0.0, 0.0, 0.0, 0.0;  // Origin with zero angle and velocity
    cddp::CarParkingObjective objective(goal_state, timestep);

    // Test running cost
    Eigen::VectorXd state(4);
    state << 1.0, 1.0, 3*M_PI/2, 0.0;  // Initial state from MATLAB demo
    Eigen::VectorXd control(2);
    control << 0.01, 0.01; // Small steering angle and acceleration
    double running_cost = objective.running_cost(state, control, 0);
    EXPECT_NEAR(running_cost, 0.0018 , 1e-4);

    // Another test
    state << 1.0, 1.0, 3*M_PI/2, 1.0;
    control << 0.3, 0.1;
    running_cost = objective.running_cost(state, control, 0);
    EXPECT_NEAR(running_cost, 0.0027, 1e-4);

    control << 0.0, 0.0;
    running_cost = objective.running_cost(state, control, 0);
    double terminal_cost = objective.terminal_cost(state);
    double total_cost = running_cost + terminal_cost;

    EXPECT_NEAR(total_cost, 5.0265, 1e-4);

    // Another test
    state << 1.0, 2.0, 3*M_PI/2, 1.2;
    control << 0.0, 0.0;
    running_cost = objective.running_cost(state, control, 0);
    terminal_cost = objective.terminal_cost(state);
    total_cost = running_cost + terminal_cost;

    EXPECT_NEAR(terminal_cost, 5.169, 1e-4);
}

TEST(CarTest, ObjectiveJacobianTest) {
    // Create a car instance
    double timestep = 0.03;  // From original MATLAB code
    double wheelbase = 2.0;  // From original MATLAB code
    std::string integration_type = "euler";
    cddp::Car car(timestep, wheelbase, integration_type);

    // Create the objective
    Eigen::VectorXd goal_state(4);
    goal_state << 0.0, 0.0, 0.0, 0.0;  // Origin with zero angle and velocity
    cddp::CarParkingObjective objective(goal_state, timestep);

    // Test running cost
    Eigen::VectorXd state(4);
    state << 1.0, 1.0, 3*M_PI/2, 0.0;  // Initial state from MATLAB demo
    Eigen::VectorXd control(2);
    control << 0.01, 0.01;

    Eigen::VectorXd lx = objective.getRunningCostStateGradient(state, control, 0);
    Eigen::VectorXd lu = objective.getRunningCostControlGradient(state, control, 0);

    Eigen::MatrixXd lxx = objective.getRunningCostStateHessian(state, control, 0);
    Eigen::MatrixXd luu = objective.getRunningCostControlHessian(state, control, 0);
    Eigen::MatrixXd lxu = objective.getRunningCostCrossHessian(state, control, 0);

    Eigen::VectorXd lx_known(4);
    lx_known << 0.0010, 0.0010, 0.0, 0.0;
    Eigen::VectorXd lu_known(2);
    lu_known << 0.0002, 0.0000;
    Eigen::MatrixXd lxx_known(4, 4);
    lxx_known << 0.9846e-5, -0.0004e-5, 0.0, 0.0,
                     -0.0004e-5, 0.9846e-5, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0;
    Eigen::MatrixXd luu_known(2, 2);
    luu_known << 0.02, 0.0,
                     0.0, 0.0002;
    Eigen::MatrixXd lxu_known(4, 2);
    lxu_known << 0.0, 0.0,
                     0.0, 0.0,
                     0.0, 0.0,
                     0.0, 0.0;

    // Compare values via norm of difference
    EXPECT_NEAR(lx.norm(), lx_known.norm(), 1e-4);
    EXPECT_NEAR(lu.norm(), lu_known.norm(), 1e-4);
    EXPECT_NEAR((lxx - lxx_known).norm(), 0.0, 1e-4);
    EXPECT_NEAR((luu - luu_known).norm(), 0.0, 1e-4);
    EXPECT_NEAR((lxu - lxu_known).norm(), 0.0, 1e-4);

    // Another test
    state << 1.0, 1.0, 3*M_PI/2, 1.0;
    control << 0.3, 0.1;

    lx = objective.getRunningCostStateGradient(state, control, 0);
    lu = objective.getRunningCostControlGradient(state, control, 0);
    lxx = objective.getRunningCostStateHessian(state, control, 0);
    luu = objective.getRunningCostControlHessian(state, control, 0);
    lxu = objective.getRunningCostCrossHessian(state, control, 0);

    lx_known << 0.0010, 0.0010, 0.0, 0.0;
    lu_known << 0.0060, 0.0000;
    lxx_known << 0.9850e-5, 0.0, 0.0, 0.0,
                0.0, 0.9850e-5, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0;
    luu_known << 0.02, 0.0,
                0.0, 0.0002;
    lxu_known << 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0;   
    
    // Compare values via norm of difference
    EXPECT_NEAR(lx.norm(), lx_known.norm(), 1e-4);
    EXPECT_NEAR(lu.norm(), lu_known.norm(), 1e-4);
    EXPECT_NEAR((lxx - lxx_known).norm(), 0.0, 1e-4);
    EXPECT_NEAR((luu - luu_known).norm(), 0.0, 1e-4);
    EXPECT_NEAR((lxu - lxu_known).norm(), 0.0, 1e-4);



    control << 0.0, 0.0;
    lx = objective.getRunningCostStateGradient(state, control, 0);
    lu = objective.getRunningCostControlGradient(state, control, 0);
    lxx = objective.getRunningCostStateHessian(state, control, 0);
    luu = objective.getRunningCostControlHessian(state, control, 0);
    lxu = objective.getRunningCostCrossHessian(state, control, 0);

    Eigen::VectorXd phi_x = objective.getFinalCostGradient(state);
    Eigen::MatrixXd phi_xx = objective.getFinalCostHessian(state);

    Eigen::VectorXd Jx = phi_x;
    Eigen::MatrixXd Jxx = phi_xx;

    Eigen::VectorXd Jx_known(4);
    Jx_known << 0.1000, 0.1000, 1.0, 0.2121;
    Eigen::MatrixXd Jxx_known(4, 4);
    Jxx_known << 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.1060;

    // Compare values via norm of difference
    EXPECT_NEAR(Jx.norm(), Jx_known.norm(), 1e-4);
    EXPECT_NEAR((Jxx - Jxx_known).norm(), 0.0, 1e-4);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}