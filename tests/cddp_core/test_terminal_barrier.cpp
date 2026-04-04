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
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

#include "cddp-cpp/cddp_core/barrier.hpp"
#include "cddp-cpp/cddp_core/constraint.hpp"
#include "cddp-cpp/cddp_core/terminal_constraint.hpp"

TEST(TerminalConstraintTest, EqualityConstraintUsesTargetState) {
  const Eigen::Vector3d target(1.0, -2.0, 0.5);
  cddp::TerminalEqualityConstraint constraint(target);

  const Eigen::Vector3d final_state(1.5, -1.5, 0.2);
  const Eigen::Vector3d expected = final_state - target;

  EXPECT_EQ(constraint.getDualDim(), 3);
  EXPECT_TRUE(constraint.evaluate(final_state).isApprox(expected));
  EXPECT_TRUE(constraint.getLowerBound().isZero());
  EXPECT_TRUE(constraint.getUpperBound().isZero());
  EXPECT_TRUE(
      constraint.getStateJacobian(final_state).isApprox(Eigen::Matrix3d::Identity()));
  EXPECT_NEAR(constraint.computeViolation(final_state), expected.norm(), 1e-12);

  const auto state_hessians = constraint.getStateHessian(final_state);
  ASSERT_EQ(state_hessians.size(), 3U);
  for (const auto &hessian : state_hessians) {
    EXPECT_EQ(hessian.rows(), 3);
    EXPECT_EQ(hessian.cols(), 3);
    EXPECT_TRUE(hessian.isZero());
  }
}

TEST(TerminalConstraintTest, EqualityConstraintRejectsDimensionMismatch) {
  cddp::TerminalEqualityConstraint constraint(Eigen::Vector3d::Zero());

  EXPECT_THROW(constraint.evaluate(Eigen::Vector2d::Zero()), std::invalid_argument);
  EXPECT_THROW(constraint.getStateJacobian(Eigen::Vector2d::Zero()),
               std::invalid_argument);
}

TEST(TerminalConstraintTest, InequalityConstraintEvaluatesBoundsAndViolation) {
  Eigen::Matrix<double, 2, 3> A;
  A << 1.0, 0.0, 0.0, 0.0, 1.0, 1.0;
  Eigen::Vector2d b(1.0, 0.25);
  cddp::TerminalInequalityConstraint constraint(A, b);

  const Eigen::Vector3d final_state(1.5, 0.1, 0.5);
  const Eigen::Vector2d value = A * final_state - b;

  EXPECT_TRUE(constraint.evaluate(final_state).isApprox(value));
  EXPECT_TRUE(
      constraint.getStateJacobian(final_state).isApprox(A));
  EXPECT_TRUE(std::isinf(constraint.getLowerBound()(0)));
  EXPECT_LT(constraint.getLowerBound()(0), 0.0);
  EXPECT_TRUE(constraint.getUpperBound().isApprox(Eigen::Vector2d::Zero()));
  EXPECT_NEAR(constraint.computeViolation(final_state), value.cwiseMax(0.0).sum(),
              1e-12);
}

TEST(RelaxedLogBarrierTest, EvaluateGradientsAndHessiansForLinearConstraint) {
  Eigen::Matrix<double, 1, 2> A;
  A << 1.0, 2.0;
  Eigen::VectorXd b(1);
  b << 4.0;
  cddp::LinearConstraint constraint(A, b);
  cddp::RelaxedLogBarrier barrier(2.0, 0.5);

  const Eigen::Vector2d state(1.0, 1.0);
  const Eigen::VectorXd control(0);

  const double barrier_value = barrier.evaluate(constraint, state, control);
  const auto [grad_x, grad_u] = barrier.getGradients(constraint, state, control);
  const auto [hxx, huu, hux] = barrier.getHessians(constraint, state, control);

  const double slack = b(0) - (A * state)(0);
  EXPECT_NEAR(barrier_value, 2.0 * (-std::log(slack)), 1e-12);
  EXPECT_TRUE(grad_x.isApprox(2.0 * A.transpose(), 1e-12));
  EXPECT_EQ(grad_u.size(), 0);
  EXPECT_TRUE(hxx.isApprox(2.0 * (A.transpose() * A), 1e-12));
  EXPECT_EQ(huu.rows(), 0);
  EXPECT_EQ(huu.cols(), 0);
  EXPECT_EQ(hux.rows(), 0);
  EXPECT_EQ(hux.cols(), state.size());
}

TEST(RelaxedLogBarrierTest, ValidatesRelaxationDelta) {
  EXPECT_THROW(cddp::RelaxedLogBarrier(1.0, 0.0), std::invalid_argument);

  cddp::RelaxedLogBarrier barrier;
  EXPECT_THROW(barrier.setRelaxationDelta(-0.1), std::invalid_argument);
}

TEST(DiscreteBarrierStateTest, ComputesUpdatesAndQuadraticCostTerms) {
  cddp::DiscreteBarrierState barrier_state(2, 1.0, 0.5, 3.0);

  const auto initialized = barrier_state.initializeBarrierStates(3);
  ASSERT_EQ(initialized.size(), 3U);
  for (const auto &value : initialized) {
    EXPECT_TRUE(value.isApprox(Eigen::Vector2d::Ones()));
  }

  const Eigen::Vector2d g_val(-0.5, 1.5);
  const Eigen::Vector2d lower(0.0, 0.0);
  const Eigen::Vector2d upper(1.0, 1.0);
  const Eigen::Vector2d violations =
      barrier_state.computeConstraintViolations(g_val, lower, upper);
  EXPECT_TRUE(violations.isApprox(Eigen::Vector2d(0.5, 0.5)));

  const Eigen::Vector2d current(1.0, 2.0);
  const Eigen::Vector2d updated =
      barrier_state.updateBarrierState(current, violations, 0.1);
  const double decay = std::exp(-0.5 * 0.1);
  EXPECT_TRUE(
      updated.isApprox(Eigen::Vector2d(current(0) * decay + 0.15,
                                       current(1) * decay + 0.15),
                       1e-12));

  EXPECT_TRUE(barrier_state.getBarrierStateDynamicsJacobian(current, 0.1)
                  .isApprox(0.3 * Eigen::Matrix2d::Identity()));

  const Eigen::Vector2d reference(0.25, 0.5);
  const Eigen::Vector2d diff = updated - reference;
  EXPECT_NEAR(barrier_state.evaluateBarrierStateCost(updated, reference),
              0.5 * 3.0 * diff.squaredNorm(), 1e-12);
  EXPECT_TRUE(barrier_state.getBarrierStateCostGradient(updated, reference)
                  .isApprox(3.0 * diff, 1e-12));
  EXPECT_TRUE(barrier_state.getBarrierStateCostHessian(updated)
                  .isApprox(3.0 * Eigen::Matrix2d::Identity(), 1e-12));
}

TEST(DiscreteBarrierStateTest, RejectsInvalidDimensionsAndParameters) {
  EXPECT_THROW(cddp::DiscreteBarrierState(0), std::invalid_argument);
  EXPECT_THROW(cddp::DiscreteBarrierState(1, 0.0), std::invalid_argument);
  EXPECT_THROW(cddp::DiscreteBarrierState(1, 1.0, -0.1), std::invalid_argument);
  EXPECT_THROW(cddp::DiscreteBarrierState(1, 1.0, 0.1, 0.0), std::invalid_argument);

  cddp::DiscreteBarrierState barrier_state(2);
  EXPECT_THROW(barrier_state.computeConstraintViolations(
                   Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(2),
                   Eigen::VectorXd::Zero(2)),
               std::invalid_argument);
  EXPECT_THROW(barrier_state.updateBarrierState(Eigen::VectorXd::Zero(1),
                                                Eigen::VectorXd::Zero(2), 0.1),
               std::invalid_argument);
  EXPECT_THROW(barrier_state.updateBarrierState(Eigen::VectorXd::Zero(2),
                                                Eigen::VectorXd::Zero(2), -0.1),
               std::invalid_argument);
  EXPECT_THROW(barrier_state.evaluateBarrierStateCost(Eigen::VectorXd::Zero(1)),
               std::invalid_argument);
  EXPECT_THROW(barrier_state.getBarrierStateCostGradient(Eigen::VectorXd::Zero(1)),
               std::invalid_argument);
}
