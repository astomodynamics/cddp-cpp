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
    std::vector<Eigen::VectorXd> X_ref(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    X_ref[0] << 1.0, 0.5, 0.2;
    X_ref[1] << 1.1, 0.6, 0.3;
    X_ref[2] << 1.2, 0.7, 0.4;
    X_ref[3] << 1.3, 0.8, 0.5;
    X_ref[4] << 1.4, 0.9, 0.6;
    X_ref[5] << 1.5, 1.0, 0.7;
    
    goal_state << 1.1, 0.6, 0.3;

    // Create the objective
    cddp::QuadraticObjective objective(Q, R, Qf, goal_state, std::vector<Eigen::VectorXd>(), timestep);
    // Example state and control
    Eigen::VectorXd state(state_dim);
    state << 1.0, 0.5, 0.2;

    std::vector<Eigen::VectorXd>  states(horizon + 1, Eigen::VectorXd::Zero(state_dim));

    states[0] << 1.0, 0.5, 0.2;
    states[1] << 1.1, 0.6, 0.3;
    states[2] << 1.2, 0.7, 0.4;
    states[3] << 1.3, 0.8, 0.5;
    states[4] << 1.4, 0.9, 0.6;
    states[5] << 1.5, 1.0, 0.7;

    Eigen::VectorXd control(control_dim);
    control << 0.8, 0.5;

    std::vector<Eigen::VectorXd> controls(horizon, Eigen::VectorXd::Zero(control_dim));

    controls[0] << 0.8, 0.5;
    controls[1] << 0.8, 0.5;
    controls[2] << 0.8, 0.5;
    controls[3] << 0.8, 0.5;
    controls[4] << 0.8, 0.5;

    // Test evaluate
    double cost = objective.evaluate(states, controls);
    double expected_cost = 0.0;
    for (int i = 0; i < states.size() - 1; i++) {
        Eigen::VectorXd state_error = states.at(i) - goal_state;
        expected_cost += (state_error.transpose() * Q * state_error).value() * timestep;
        expected_cost += (controls.at(i).transpose() * R * controls.at(i)).value() * timestep;
    }
    Eigen::VectorXd state_error = states.back() - goal_state;
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
    Eigen::MatrixXd expected_cross_hess = Eigen::MatrixXd::Zero(control_dim, state_dim);

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


class TestNonlinearObjective : public cddp::NonlinearObjective {
public:
    TestNonlinearObjective(double timestep = 0.1) : NonlinearObjective(timestep) {}

    // Example nonlinear running cost: (x'x)^2 + (u'u)^2
    double running_cost(const Eigen::VectorXd& state, 
                       const Eigen::VectorXd& control,
                       int index) const override {
        double state_cost = std::pow(state.squaredNorm(), 2);
        double control_cost = std::pow(control.squaredNorm(), 2);
        return (state_cost + control_cost) * timestep_;
    }

    // Example nonlinear terminal cost: exp(x'x)
    double terminal_cost(const Eigen::VectorXd& final_state) const override {
        return std::exp(final_state.squaredNorm());
    }
private: 
    double timestep_;
};

TEST(ObjectiveFunctionTests, NonlinearObjective) {
    // Set up test data
    int state_dim = 2;
    int control_dim = 2;
    int horizon = 3;
    double timestep = 0.1;

    TestNonlinearObjective objective(timestep);

    // Create test trajectories
    std::vector<Eigen::VectorXd> states(horizon + 1);
    std::vector<Eigen::VectorXd> controls(horizon);

    for (int i = 0; i <= horizon; i++) {
        states[i] = Eigen::VectorXd::Ones(state_dim) * 0.5;
    }
    for (int i = 0; i < horizon; i++) {
        controls[i] = Eigen::VectorXd::Ones(control_dim) * 0.3;
    }

    // Test evaluate
    double total_cost = objective.evaluate(states, controls);
    double expected_cost = 0.0;
    
    // Calculate expected cost manually
    for (int i = 0; i < horizon; i++) {
        expected_cost += objective.running_cost(states[i], controls[i], i);
    }
    expected_cost += objective.terminal_cost(states.back());
    
    EXPECT_NEAR(total_cost, expected_cost, 1e-6);

    // Test running cost gradients using finite differences
    Eigen::VectorXd state = states[0];
    Eigen::VectorXd control = controls[0];
    
    auto [state_grad, control_grad] = objective.getRunningCostGradients(state, control, 0);
    
    // Test that gradients have correct dimensions
    EXPECT_EQ(state_grad.size(), state_dim);
    EXPECT_EQ(control_grad.size(), control_dim);

    // Verify gradients using finite differences
    double eps = 1e-6;
    for (int i = 0; i < state_dim; i++) {
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        state_plus(i) += eps;
        state_minus(i) -= eps;
        
        double grad_fd = (objective.running_cost(state_plus, control, 0) - 
                         objective.running_cost(state_minus, control, 0)) / (2 * eps);
        EXPECT_NEAR(state_grad(i), grad_fd, 1e-5);
    }

    // Test terminal cost gradient
    Eigen::VectorXd final_grad = objective.getFinalCostGradient(state);
    
    // Verify terminal gradient using finite differences
    for (int i = 0; i < state_dim; i++) {
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        state_plus(i) += eps;
        state_minus(i) -= eps;
        
        double grad_fd = (objective.terminal_cost(state_plus) - 
                         objective.terminal_cost(state_minus)) / (2 * eps);
        EXPECT_NEAR(final_grad(i), grad_fd, 1e-5);
    }

    // Test Hessians
    auto [state_hess, control_hess, cross_hess] = objective.getRunningCostHessians(state, control, 0);
    
    // Check dimensions
    EXPECT_EQ(state_hess.rows(), state_dim);
    EXPECT_EQ(state_hess.cols(), state_dim);
    EXPECT_EQ(control_hess.rows(), control_dim);
    EXPECT_EQ(control_hess.cols(), control_dim);
    EXPECT_EQ(cross_hess.rows(), control_dim);
    EXPECT_EQ(cross_hess.cols(), state_dim);

    // Verify state Hessian using finite differences
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            Eigen::VectorXd state_pp = state;
            Eigen::VectorXd state_pm = state;
            Eigen::VectorXd state_mp = state;
            Eigen::VectorXd state_mm = state;
            
            state_pp(i) += eps; state_pp(j) += eps;
            state_pm(i) += eps; state_pm(j) -= eps;
            state_mp(i) -= eps; state_mp(j) += eps;
            state_mm(i) -= eps; state_mm(j) -= eps;
            
            double hess_fd = (objective.running_cost(state_pp, control, 0) -
                            objective.running_cost(state_pm, control, 0) -
                            objective.running_cost(state_mp, control, 0) +
                            objective.running_cost(state_mm, control, 0)) / (4 * eps * eps);
            
            if (i == j) {  // Allow larger tolerance for diagonal elements due to numerical issues
                EXPECT_NEAR(state_hess(i,j), hess_fd, 1e-4);
            } else {
                EXPECT_NEAR(state_hess(i,j), hess_fd, 1e-5);
            }
        }
    }

    // Test terminal cost Hessian
    Eigen::MatrixXd final_hess = objective.getFinalCostHessian(state);
    EXPECT_EQ(final_hess.rows(), state_dim);
    EXPECT_EQ(final_hess.cols(), state_dim);
}
