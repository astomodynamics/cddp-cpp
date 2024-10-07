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
// Description: Tests for the objective function classes in cddp-cpp.
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "cddp-cpp/cddp_core/objective.hpp"
#include <iostream>
#include <cassert> 

// Helper function to compare vectors with tolerance
bool compareVectors(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double tolerance = 1e-6) {
    if (v1.size() != v2.size()) {
        return false;
    }
    return (v1 - v2).array().abs().maxCoeff() <= tolerance;
}

// Helper function to compare matrices with tolerance
bool compareMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tolerance = 1e-6) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return false;
    }
    return (m1 - m2).array().abs().maxCoeff() <= tolerance;
}

TEST(ObjectiveFunctionTests, QuadraticObjective) {
    // Set up test data (no device needed for Eigen)
    int state_dim = 3; 
    int control_dim = 2;
    int horizon = 5;
    double timestep = 0.1;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim) * 0.1; 
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim) * 2.0;
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    Eigen::MatrixXd X_ref(state_dim, horizon + 1);
    X_ref << 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
             0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
             0.2, 0.3, 0.4, 0.5, 0.6, 0.7;

    goal_state << 1.1, 0.6, 0.3;

    // Create the objective
    cddp::QuadraticObjective objective(Q, R, Qf, goal_state, Eigen::MatrixXd::Zero(0, 0), timestep);

    // Example state and control
    Eigen::VectorXd state(state_dim);
    state << 1.0, 0.5, 0.2;

    Eigen::MatrixXd states(state_dim, horizon + 1);
    states << 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
              0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
              0.2, 0.3, 0.4, 0.5, 0.6, 0.7;

    Eigen::VectorXd control(control_dim);
    control << 0.8, 0.5;

    Eigen::MatrixXd controls(control_dim, horizon);
    controls << 0.8, 0.8, 0.8, 0.8, 0.8,
                0.5, 0.5, 0.5, 0.5, 0.5;

    // Test evaluate
    double cost = objective.evaluate(states, controls);
    double expected_cost = 0.0;
    for (int i = 0; i < states.cols() - 1; i++) {
        Eigen::VectorXd state_error = states.col(i) - goal_state;
        expected_cost += (state_error.transpose() * Q * state_error).value() * timestep;
        expected_cost += (controls.col(i).transpose() * R * controls.col(i)).value() * timestep;
    }
    Eigen::VectorXd state_error = states.col(states.cols() - 1) - goal_state;
    expected_cost += (state_error.transpose() * Qf * state_error).value();
    ASSERT_TRUE(std::abs(cost - expected_cost) < 1e-6);

    // Test gradients
    auto [state_grad, control_grad] = objective.getRunningCostGradients(state, control, 0);
    Eigen::VectorXd expected_state_grad = 2.0 * Q * (state - goal_state) * timestep;
    Eigen::VectorXd expected_control_grad = 2.0 * R * control * timestep;

    ASSERT_TRUE(compareVectors(state_grad, expected_state_grad));
    ASSERT_TRUE(compareVectors(control_grad, expected_control_grad));

    // Test Hessians
    auto [state_hess, control_hess, cross_hess] = objective.getRunningCostHessians(state, control, 0);
    Eigen::MatrixXd expected_state_hess = 2.0 * Q * timestep;
    Eigen::MatrixXd expected_control_hess = 2.0 * R * timestep;
    Eigen::MatrixXd expected_cross_hess = Eigen::MatrixXd::Zero(state_dim, control_dim);

    ASSERT_TRUE(compareMatrices(state_hess, expected_state_hess));
    ASSERT_TRUE(compareMatrices(control_hess, expected_control_hess));
    ASSERT_TRUE(compareMatrices(cross_hess, expected_cross_hess));

    // Test final cost and gradient
    double final_cost = objective.terminal_cost(state);
    Eigen::VectorXd final_cost_grad = objective.getFinalCostGradient(state);
    Eigen::MatrixXd final_cost_hess = objective.getFinalCostHessian(state);

    ASSERT_TRUE(std::abs(final_cost - ((state - goal_state).transpose() * Qf * (state - goal_state)).value()) < 1e-6);
    ASSERT_TRUE(compareVectors(final_cost_grad, 2.0 * Qf * (state - goal_state)));
    ASSERT_TRUE(compareMatrices(final_cost_hess, 2.0 * Qf));
}
    