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

TEST(CircleConstraintTest, Evaluate) {
    // Create a circle constraint with a radius of 2.0
    cddp::CircleConstraint constraint(2.0);

    // Test with a state inside the circle
    Eigen::VectorXd state(2);
    state << 1.0, 1.0;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0; 
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), 2.0, 1e-6); // 1.0^2 + 1.0^2 = 2.0

    // Test with a state outside the circle
    state << 2.5, 1.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), 6.25 + 2.25, 1e-6); // 2.5^2 + 1.5^2 = 8.5
}