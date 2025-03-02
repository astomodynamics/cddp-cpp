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
#include "cddp-cpp/cddp_core/constraint.hpp"

TEST(ControlBoxConstraintTest, Evaluate) {
    // Create a constraint with lower and upper bounds
    Eigen::VectorXd lower_bound(2);
    lower_bound << -1.0, -2.0;
    Eigen::VectorXd upper_bound(2);
    upper_bound << 1.0, 2.0;
    cddp::ControlBoxConstraint constraint(lower_bound, upper_bound);

    // Test with a control signal within the bounds
    Eigen::VectorXd state(2); 
    state << 0.5, 1.0;
    Eigen::VectorXd control(2);
    control << 0.5, 1.0;
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(control));

    // Test with a control signal outside the bounds
    control << 1.5, -2.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(control)); 
}

TEST(StateBoxConstraint, Evaluate) {
    // Create a constraint with lower and upper bounds
    Eigen::VectorXd lower_bound(2);
    lower_bound << -1.0, -2.0;
    Eigen::VectorXd upper_bound(2);
    upper_bound << 1.0, 2.0;
    cddp::StateBoxConstraint constraint(lower_bound, upper_bound);

    // Test with a state within the bounds
    Eigen::VectorXd state(2);
    state << 0.5, 1.0;
    Eigen::VectorXd control(2); // Control doesn't matter for this constraint
    control << 0.0;
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(state));

    // Test with a state outside the bounds
    state << 1.5, -2.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(state));
}

TEST(CircleConstraintTest, Evaluate) {
    // Create a circle constraint with a radius of 2.0
    cddp::BallConstraint constraint(2.0, Eigen::Vector2d(0.0, 0.0));

    // Test with a state inside the circle
    Eigen::VectorXd state(2);
    state << 1.0, 1.0;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0; 
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), -2.0, 1e-6); 

    // Test with a state outside the circle
    state << 2.5, 1.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), -8.5, 1e-6);
}

TEST(CircleConstraintTest, Gradients) {
    // Create a circle constraint with a radius of 2.0
    cddp::BallConstraint constraint(2.0, Eigen::Vector2d(0.0, 0.0));

    // Test with a state inside the circle
    Eigen::VectorXd state(2);
    state << 1.0, 1.0;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0; 
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    auto gradients = constraint.getJacobians(state, control);
    ASSERT_TRUE(std::get<0>(gradients).isApprox(Eigen::Vector2d(-2.0, -2.0)));
    ASSERT_TRUE(std::get<1>(gradients).isApprox(Eigen::Vector2d(0.0, 0.0)));

    // Test with a state outside the circle
    state << 2.5, 1.5;
    constraint_value = constraint.evaluate(state, control);
    gradients = constraint.getJacobians(state, control);
    ASSERT_TRUE(std::get<0>(gradients).isApprox(Eigen::Vector2d(-5.0, -3.0)));
    ASSERT_TRUE(std::get<1>(gradients).isApprox(Eigen::Vector2d(0.0, 0.0)));
}

TEST(LinearConstraintTest, Evaluate) {
    // Create a linear constraint with A = [1, 1], b = 1
    Eigen::MatrixXd A(1, 2);
    A << 1.0, 1.0;
    Eigen::VectorXd b(1);
    b << 1.0;
    cddp::LinearConstraint constraint(A, b);

    // Test with a state that satisfies the constraint
    Eigen::VectorXd state(2);
    state << 0.5, 0.5;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0;
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), 1.0, 1e-6);

    // Test with a state that violates the constraint
    state << 0.5, -0.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), 0.0, 1e-6);
}