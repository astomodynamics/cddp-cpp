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
    std::vector<Eigen::VectorXd> X_ref(horizon, goal_state);

    // Create the objective
    cddp::QuadraticObjective objective(Q, R, Qf, goal_state);

    // Example state and control
    Eigen::VectorXd state(state_dim);
    state << 1.0, 0.5, 0.2;

    Eigen::VectorXd control(control_dim);
    control << 0.8, 0.5;

    // Test evaluate
    double cost = objective.evaluate(state, control);
    double expected_cost = ((state - goal_state).transpose() * Q * (state - goal_state)).value() + (control.transpose() * R * control).value();

    ASSERT_TRUE(std::abs(cost - expected_cost) < 1e-6);

    // // Test gradients
    auto [state_grad, control_grad] = objective.getRunningCostGradients(state, control);
    Eigen::VectorXd expected_state_grad = 2.0 * Q * (state - goal_state);
    Eigen::VectorXd expected_control_grad = 2.0 * R * control;

    ASSERT_TRUE(compareVectors(state_grad, expected_state_grad));
    ASSERT_TRUE(compareVectors(control_grad, expected_control_grad));

    // Test Hessians
    auto [state_hess, control_hess, cross_hess] = objective.getRunningCostHessians(state, control);
    Eigen::MatrixXd expected_state_hess = 2.0 * Q;
    Eigen::MatrixXd expected_control_hess = 2.0 * R;
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
    