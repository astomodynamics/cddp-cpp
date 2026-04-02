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
#include <sys/stat.h>

TEST(ControlConstraintTest, Evaluate) {
  Eigen::VectorXd lower_bound(2);
  lower_bound << -1.0, -2.0;
  Eigen::VectorXd upper_bound(2);
  upper_bound << 1.0, 2.0;
  cddp::ControlConstraint constraint(lower_bound, upper_bound);

  Eigen::VectorXd state(2);
  state << 0.5, 1.0;
  Eigen::VectorXd control(2);
  control << 0.5, 1.0;

  // evaluate() returns [-control; control] (IP-compatible formulation)
  Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
  Eigen::VectorXd expected_value(2 * control.size());
  expected_value.head(control.size()) = -control;
  expected_value.tail(control.size()) = control;
  ASSERT_TRUE(constraint_value.isApprox(expected_value));

  control << 1.5, -2.5;
  constraint_value = constraint.evaluate(state, control);
  expected_value.head(control.size()) = -control;
  expected_value.tail(control.size()) = control;
  ASSERT_TRUE(constraint_value.isApprox(expected_value));
}

TEST(ControlConstraintTest, RawBoundsAndClamp) {
  Eigen::VectorXd lower_bound(2);
  lower_bound << -1.0, -2.0;
  Eigen::VectorXd upper_bound(2);
  upper_bound << 1.0, 2.0;
  cddp::ControlConstraint constraint(lower_bound, upper_bound);

  // Raw bounds for CLDDP / BoxQP
  ASSERT_TRUE(constraint.rawLowerBound().isApprox(lower_bound));
  ASSERT_TRUE(constraint.rawUpperBound().isApprox(upper_bound));

  // Clamp within bounds
  Eigen::VectorXd control(2);
  control << 0.5, 1.0;
  ASSERT_TRUE(constraint.clamp(control).isApprox(control));

  // Clamp outside bounds
  control << 1.5, -2.5;
  Eigen::VectorXd expected(2);
  expected << 1.0, -2.0;
  ASSERT_TRUE(constraint.clamp(control).isApprox(expected));
}

TEST(StateConstraintTest, Evaluate) {
  Eigen::VectorXd lower_bound(2);
  lower_bound << -1.0, -2.0;
  Eigen::VectorXd upper_bound(2);
  upper_bound << 1.0, 2.0;
  cddp::StateConstraint constraint(lower_bound, upper_bound);

  Eigen::VectorXd state(2);
  state << 0.5, 1.0;
  Eigen::VectorXd control(2);
  control << 0.0, 0.0;

  // evaluate() returns [-state; state] (IP-compatible formulation)
  Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
  Eigen::VectorXd expected_value(2 * state.size());
  expected_value.head(state.size()) = -state;
  expected_value.tail(state.size()) = state;
  ASSERT_TRUE(constraint_value.isApprox(expected_value));

  state << 1.5, -2.5;
  constraint_value = constraint.evaluate(state, control);
  expected_value.head(state.size()) = -state;
  expected_value.tail(state.size()) = state;
  ASSERT_TRUE(constraint_value.isApprox(expected_value));
}

TEST(StateConstraintTest, RawBoundsAndClamp) {
  Eigen::VectorXd lower_bound(2);
  lower_bound << -1.0, -2.0;
  Eigen::VectorXd upper_bound(2);
  upper_bound << 1.0, 2.0;
  cddp::StateConstraint constraint(lower_bound, upper_bound);

  ASSERT_TRUE(constraint.rawLowerBound().isApprox(lower_bound));
  ASSERT_TRUE(constraint.rawUpperBound().isApprox(upper_bound));

  Eigen::VectorXd state(2);
  state << 0.5, 1.0;
  ASSERT_TRUE(constraint.clamp(state).isApprox(state));

  state << 1.5, -2.5;
  Eigen::VectorXd expected(2);
  expected << 1.0, -2.0;
  ASSERT_TRUE(constraint.clamp(state).isApprox(expected));
}

TEST(ControlConstraintTest, Jacobians) {
  Eigen::VectorXd lower_bound(2);
  lower_bound << -1.0, -2.0;
  Eigen::VectorXd upper_bound(2);
  upper_bound << 1.0, 2.0;
  cddp::ControlConstraint constraint(lower_bound, upper_bound);

  Eigen::VectorXd state(2);
  state << 0.5, 1.0;
  Eigen::VectorXd control(2);
  control << 0.5, 1.0;

  Eigen::MatrixXd state_jac = constraint.getStateJacobian(state, control);
  Eigen::MatrixXd control_jac = constraint.getControlJacobian(state, control);

  // State Jacobian should be zero
  ASSERT_EQ(state_jac.rows(), 4);
  ASSERT_EQ(state_jac.cols(), 2);
  ASSERT_TRUE(state_jac.isZero());

  // Control Jacobian should be [-I; I]
  Eigen::MatrixXd expected_control_jac(4, 2);
  expected_control_jac.topRows(2) = -Eigen::MatrixXd::Identity(2, 2);
  expected_control_jac.bottomRows(2) = Eigen::MatrixXd::Identity(2, 2);
  ASSERT_TRUE(control_jac.isApprox(expected_control_jac));
}

TEST(StateConstraintTest, Jacobians) {
  Eigen::VectorXd lower_bound(2);
  lower_bound << -1.0, -2.0;
  Eigen::VectorXd upper_bound(2);
  upper_bound << 1.0, 2.0;
  cddp::StateConstraint constraint(lower_bound, upper_bound);

  Eigen::VectorXd state(2);
  state << 0.5, 1.0;
  Eigen::VectorXd control(2);
  control << 0.5, 1.0;

  Eigen::MatrixXd state_jac = constraint.getStateJacobian(state, control);
  Eigen::MatrixXd control_jac = constraint.getControlJacobian(state, control);

  // State Jacobian should be [-I; I]
  Eigen::MatrixXd expected_state_jac(4, 2);
  expected_state_jac.topRows(2) = -Eigen::MatrixXd::Identity(2, 2);
  expected_state_jac.bottomRows(2) = Eigen::MatrixXd::Identity(2, 2);
  ASSERT_TRUE(state_jac.isApprox(expected_state_jac));

  // Control Jacobian should be zero
  ASSERT_EQ(control_jac.rows(), 4);
  ASSERT_EQ(control_jac.cols(), 2);
  ASSERT_TRUE(control_jac.isZero());
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

// New test suite for SecondOrderConeConstraint
TEST(SecondOrderConeConstraintTest, Evaluate) {
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // Opening along positive Y-axis
    double fov = M_PI / 4.0; // 45 degrees half-angle
    double tan_fov = std::tan(fov);
    double epsilon = 1e-8;
    cddp::SecondOrderConeConstraint constraint(origin, axis, fov, epsilon);

    Eigen::VectorXd control(1); // Control doesn't matter
    control << 0.0;

    // Test with a state INSIDE the cone (g <= 0)
    Eigen::VectorXd state_inside(3);
    state_inside << 0.0, 1.0, 0.0; // Point along positive Y-axis
    Eigen::VectorXd constraint_value_inside = constraint.evaluate(state_inside, control);
    ASSERT_LT(constraint_value_inside(0), 0.0); // g = cos(fov)*||p|| - p.axis = cos(fov)*1 - 1 < 0

    // Test with a state OUTSIDE the cone (g > 0)
    Eigen::VectorXd state_outside(3);
    state_outside << 0.0, -1.0, 0.0; // Point along negative Y-axis
    Eigen::VectorXd constraint_value_outside = constraint.evaluate(state_outside, control);
    ASSERT_GT(constraint_value_outside(0), 0.0); // g = cos(fov)*||p|| - p.axis = cos(fov)*1 - (-1) > 0

    // Test with a state ON the boundary (g approx 0)
    Eigen::VectorXd state_boundary(3);
    double y_val = 1.5;
    double radius_at_y = std::abs(y_val) * tan_fov;
    state_boundary << radius_at_y, y_val, 0.0; // Point on boundary at y=1.5
    Eigen::VectorXd constraint_value_boundary = constraint.evaluate(state_boundary, control);
    ASSERT_NEAR(constraint_value_boundary(0), 0.0, 1e-6); // Use tolerance due to epsilon
}

TEST(SecondOrderConeConstraintTest, Gradients) {
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // Opening along positive Y-axis
    double fov = M_PI / 4.0; // 45 degrees
    double cos_fov = std::cos(fov);
    double epsilon = 1e-8;
    cddp::SecondOrderConeConstraint constraint(origin, axis, fov, epsilon);

    Eigen::VectorXd state(3);
    state << 0.1, 0.5, 0.1; // A point inside the cone, in positive Y
    Eigen::VectorXd control(1);
    control << 0.0;

    // Calculate Jacobians
    auto jacobians = constraint.getJacobians(state, control);
    Eigen::MatrixXd state_jacobian = std::get<0>(jacobians);
    Eigen::MatrixXd control_jacobian = std::get<1>(jacobians);

    // Calculate expected state Jacobian using the NEW analytical formula
    // dg/dp_s = cos(fov) * (v^T / reg_norm) - axis^T, where v = p_s - origin
    Eigen::Vector3d p_s = state.head(3);
    Eigen::Vector3d v = p_s - origin; 
    double v_squared = v.squaredNorm(); 
    double reg_norm = std::sqrt(v_squared + epsilon);
    Eigen::RowVector3d expected_dg_dps;
    if (reg_norm > 1e-9) {
        expected_dg_dps = cos_fov * (v.transpose() / reg_norm) - axis.transpose();
    } else {
        expected_dg_dps = -axis.transpose();
    }

    // Construct expected full Jacobian dg/dx = dg/dp_s * [I, 0]
    Eigen::MatrixXd expected_state_jacobian = Eigen::MatrixXd::Zero(1, state.size());
    expected_state_jacobian.leftCols(3) = expected_dg_dps;

    ASSERT_EQ(state_jacobian.rows(), 1);
    ASSERT_EQ(state_jacobian.cols(), state.size());
    ASSERT_TRUE(state_jacobian.isApprox(expected_state_jacobian, 1e-6));

    // Expected control Jacobian should be zero
    ASSERT_EQ(control_jacobian.rows(), 1);
    ASSERT_EQ(control_jacobian.cols(), control.size());
    ASSERT_TRUE(control_jacobian.isApprox(Eigen::MatrixXd::Zero(1, control.size())));
}

TEST(BallConstraintTest, Hessians) {
    // Create a ball constraint with a radius of 2.0 and scale_factor 1.0
    cddp::BallConstraint constraint(2.0, Eigen::Vector2d(0.0, 0.0), 1.0);

    Eigen::VectorXd state(2);
    state << 1.0, 1.0;
    Eigen::VectorXd control(1); 
    control << 0.0;

    auto hessians = constraint.getHessians(state, control);
    std::vector<Eigen::MatrixXd> Hxx_list = std::get<0>(hessians);
    std::vector<Eigen::MatrixXd> Huu_list = std::get<1>(hessians);
    std::vector<Eigen::MatrixXd> Hux_list = std::get<2>(hessians);

    // Expected Hxx for g(x) = -1.0 * ||x - c||^2 is -2.0 * I
    Eigen::MatrixXd expected_Hxx(2, 2);
    expected_Hxx << -2.0,  0.0,
                     0.0, -2.0;
    
    ASSERT_EQ(Hxx_list.size(), 1);
    ASSERT_TRUE(Hxx_list[0].isApprox(expected_Hxx));

    ASSERT_EQ(Huu_list.size(), 1);
    ASSERT_TRUE(Huu_list[0].isApprox(Eigen::MatrixXd::Zero(control.size(), control.size())));

    ASSERT_EQ(Hux_list.size(), 1);
    ASSERT_TRUE(Hux_list[0].isApprox(Eigen::MatrixXd::Zero(control.size(), state.size())));

    // Test with a different scale_factor
    cddp::BallConstraint constraint_scaled(2.0, Eigen::Vector2d(0.0, 0.0), 3.0);
    auto hessians_scaled = constraint_scaled.getHessians(state, control);
    std::vector<Eigen::MatrixXd> Hxx_list_scaled = std::get<0>(hessians_scaled);
    Eigen::MatrixXd expected_Hxx_scaled(2, 2);
    expected_Hxx_scaled << -6.0,  0.0, // -2.0 * 3.0
                            0.0, -6.0;
    ASSERT_EQ(Hxx_list_scaled.size(), 1);
    ASSERT_TRUE(Hxx_list_scaled[0].isApprox(expected_Hxx_scaled));
}

// TEST(LinearConstraintTest, Hessians) {
//     Eigen::MatrixXd A(2, 2);
//     A <<  1.0,  1.0,
//          -1.0,  1.0;
//     Eigen::VectorXd b(2);
//     b << 1.0, 1.0;
//     cddp::LinearConstraint constraint(A, b);

//     Eigen::VectorXd state(2); state << 0.1, 0.2;
//     Eigen::VectorXd control(1); control << 0.0;

//     auto hessians = constraint.getHessians(state, control);
//     std::vector<Eigen::MatrixXd> Hxx_list = std::get<0>(hessians);
//     std::vector<Eigen::MatrixXd> Huu_list = std::get<1>(hessians);
//     std::vector<Eigen::MatrixXd> Hux_list = std::get<2>(hessians);

//     ASSERT_EQ(Hxx_list.size(), A.rows());
//     for (const auto& Hxx : Hxx_list) {
//         ASSERT_TRUE(Hxx.isApprox(Eigen::MatrixXd::Zero(state.size(), state.size())));
//     }
//     ASSERT_EQ(Huu_list.size(), A.rows());
//     for (const auto& Huu : Huu_list) {
//         ASSERT_TRUE(Huu.isApprox(Eigen::MatrixXd::Zero(control.size(), control.size())));
//     }
//     ASSERT_EQ(Hux_list.size(), A.rows());
//     for (const auto& Hux : Hux_list) {
//         ASSERT_TRUE(Hux.isApprox(Eigen::MatrixXd::Zero(control.size(), state.size())));
//     }
// }

// // New test suite for SecondOrderConeConstraint
// TEST(SecondOrderConeConstraintTest, Hessians) {
//     Eigen::Vector3d origin(0.0, 0.0, 0.0);
//     Eigen::Vector3d axis(0.0, 1.0, 0.0); 
//     double fov = M_PI / 4.0; 
//     double epsilon = 1e-8;
//     cddp::SecondOrderConeConstraint constraint(origin, axis, fov, epsilon);

//     Eigen::VectorXd state(3); state << 0.1, 0.5, 0.1; 
//     Eigen::VectorXd control(1); control << 0.0;

//     // Placeholder test: Expect zero Hessians for now as per implementation
//     auto hessians = constraint.getHessians(state, control);
//     std::vector<Eigen::MatrixXd> Hxx_list = std::get<0>(hessians);
//     std::vector<Eigen::MatrixXd> Huu_list = std::get<1>(hessians);
//     std::vector<Eigen::MatrixXd> Hux_list = std::get<2>(hessians);

//     ASSERT_EQ(Hxx_list.size(), 1);
//     ASSERT_TRUE(Hxx_list[0].isApprox(Eigen::MatrixXd::Zero(state.size(), state.size())));
//     ASSERT_EQ(Huu_list.size(), 1);
//     ASSERT_TRUE(Huu_list[0].isApprox(Eigen::MatrixXd::Zero(control.size(), control.size())));
//     ASSERT_EQ(Hux_list.size(), 1);
//     ASSERT_TRUE(Hux_list[0].isApprox(Eigen::MatrixXd::Zero(control.size(), state.size())));
// }
