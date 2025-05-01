#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "cddp_core/parameterized_objective.hpp"

namespace cddp {
namespace tests {

// Test fixture for ParameterizedQuadraticObjective
class ParameterizedQuadraticObjectiveTest : public ::testing::Test {
protected:
    void SetUp() override {
        state_dim = 4;
        control_dim = 2;
        param_dim = state_dim; // Parameter defines final target offset
        timestep = 0.1;

        // Define weights
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim) * 0.1;
        R = Eigen::MatrixXd::Identity(control_dim, control_dim) * 0.01;
        Qf = Eigen::MatrixXd::Identity(state_dim, state_dim) * 10.0;

        // Define nominal reference state
        nominal_ref_state = Eigen::VectorXd::Zero(state_dim);
        nominal_ref_state << 1.0, 0.5, 0.0, 0.0;

        // Instantiate the objective
        param_objective = std::make_unique<ParameterizedQuadraticObjective>(
            Q, R, Qf, param_dim, nominal_ref_state, 
            std::vector<Eigen::VectorXd>(), // No running reference states for simplicity
            timestep
        );

        // Define sample data
        test_state = Eigen::VectorXd::Zero(state_dim);
        test_state << 0.1, 0.2, 0.3, 0.4;
        test_control = Eigen::VectorXd::Zero(control_dim);
        test_control << 0.5, -0.5;
        test_parameter = Eigen::VectorXd::Zero(param_dim);
        test_parameter << 0.05, -0.05, 0.1, -0.1; // Offset for the final target
    }

    int state_dim, control_dim, param_dim;
    double timestep;
    Eigen::MatrixXd Q, R, Qf;
    Eigen::VectorXd nominal_ref_state;
    std::unique_ptr<ParameterizedQuadraticObjective> param_objective;
    Eigen::VectorXd test_state;
    Eigen::VectorXd test_control;
    Eigen::VectorXd test_parameter;

    Eigen::MatrixXd Q_scaled, R_scaled; // Scaled by timestep
};

TEST_F(ParameterizedQuadraticObjectiveTest, RunningCostCalculation) {
    int index = 0;
    // Running cost uses nominal_ref_state as target since no reference_states_ provided
    Eigen::VectorXd state_error = test_state - nominal_ref_state;
    double expected_state_cost = (state_error.transpose() * param_objective->getQ() * state_error).value();
    double expected_control_cost = (test_control.transpose() * param_objective->getR() * test_control).value();
    double expected_running_cost = expected_state_cost + expected_control_cost;

    double actual_running_cost = param_objective->running_cost(test_state, test_control, test_parameter, index);
    EXPECT_NEAR(actual_running_cost, expected_running_cost, 1e-9);
}

TEST_F(ParameterizedQuadraticObjectiveTest, TerminalCostCalculation) {
    // Terminal cost uses effective target = nominal_ref_state + test_parameter
    Eigen::VectorXd effective_target = nominal_ref_state + test_parameter;
    Eigen::VectorXd state_error = test_state - effective_target;
    double expected_terminal_cost = (state_error.transpose() * param_objective->getQf() * state_error).value();

    double actual_terminal_cost = param_objective->terminal_cost(test_state, test_parameter);
    EXPECT_NEAR(actual_terminal_cost, expected_terminal_cost, 1e-9);
}

TEST_F(ParameterizedQuadraticObjectiveTest, RunningCostStateGradient) {
    int index = 0;
    Eigen::VectorXd state_error = test_state - nominal_ref_state;
    Eigen::VectorXd expected_gradient = 2.0 * param_objective->getQ() * state_error;
    Eigen::VectorXd actual_gradient = param_objective->getRunningCostStateGradient(test_state, test_control, test_parameter, index);
    ASSERT_EQ(actual_gradient.size(), state_dim);
    EXPECT_TRUE(actual_gradient.isApprox(expected_gradient, 1e-9));
}

TEST_F(ParameterizedQuadraticObjectiveTest, RunningCostControlGradient) {
    int index = 0;
    Eigen::VectorXd expected_gradient = 2.0 * param_objective->getR() * test_control;
    Eigen::VectorXd actual_gradient = param_objective->getRunningCostControlGradient(test_state, test_control, test_parameter, index);
     ASSERT_EQ(actual_gradient.size(), control_dim);
    EXPECT_TRUE(actual_gradient.isApprox(expected_gradient, 1e-9));
}

TEST_F(ParameterizedQuadraticObjectiveTest, RunningCostParameterGradient) {
    int index = 0;
    Eigen::VectorXd expected_gradient = Eigen::VectorXd::Zero(param_dim);
    Eigen::VectorXd actual_gradient = param_objective->getRunningCostParameterGradient(test_state, test_control, test_parameter, index);
    ASSERT_EQ(actual_gradient.size(), param_dim);
    EXPECT_TRUE(actual_gradient.isApprox(expected_gradient, 1e-9));
}

TEST_F(ParameterizedQuadraticObjectiveTest, FinalCostStateGradient) {
    Eigen::VectorXd effective_target = nominal_ref_state + test_parameter;
    Eigen::VectorXd state_error = test_state - effective_target;
    Eigen::VectorXd expected_gradient = 2.0 * param_objective->getQf() * state_error;
    Eigen::VectorXd actual_gradient = param_objective->getFinalCostGradient(test_state, test_parameter);
    ASSERT_EQ(actual_gradient.size(), state_dim);
    EXPECT_TRUE(actual_gradient.isApprox(expected_gradient, 1e-9));
}

TEST_F(ParameterizedQuadraticObjectiveTest, FinalCostParameterGradient) {
    Eigen::VectorXd effective_target = nominal_ref_state + test_parameter;
    Eigen::VectorXd state_error = test_state - effective_target;
    Eigen::VectorXd expected_gradient = -2.0 * param_objective->getQf() * state_error;
    Eigen::VectorXd actual_gradient = param_objective->getFinalCostParameterGradient(test_state, test_parameter);
    ASSERT_EQ(actual_gradient.size(), param_dim);
    EXPECT_TRUE(actual_gradient.isApprox(expected_gradient, 1e-9));
}

} // namespace tests
} // namespace cddp 