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
#include <array>

#include "gtest/gtest.h"

#include "cddp-cpp/cddp_core/helper.hpp"

namespace {

constexpr int kCentralDifference = 0;
constexpr int kForwardDifference = 1;
constexpr int kBackwardDifference = 2;

double quadratic_cost(const Eigen::VectorXd &x) {
  return x(0) * x(0) + 3.0 * x(0) * x(1) + std::sin(x(1));
}

Eigen::VectorXd vector_function(const Eigen::VectorXd &x) {
  Eigen::VectorXd value(2);
  value << x(0) * x(0) + x(1), x(0) - 2.0 * x(1) * x(1);
  return value;
}

Eigen::VectorXd expected_gradient(const Eigen::VectorXd &x) {
  Eigen::VectorXd grad(2);
  grad << 2.0 * x(0) + 3.0 * x(1), 3.0 * x(0) + std::cos(x(1));
  return grad;
}

Eigen::MatrixXd expected_jacobian(const Eigen::VectorXd &x) {
  Eigen::MatrixXd jac(2, 2);
  jac << 2.0 * x(0), 1.0, 1.0, -4.0 * x(1);
  return jac;
}

Eigen::MatrixXd expected_hessian(const Eigen::VectorXd &x) {
  Eigen::MatrixXd hess(2, 2);
  hess << 2.0, 3.0, 3.0, -std::sin(x(1));
  return hess;
}

} // namespace

TEST(FiniteDifferenceTest, GradientMatchesAnalyticForAllModes) {
  const Eigen::Vector2d x(0.4, -0.2);
  const Eigen::Vector2d grad_expected = expected_gradient(x);

  for (const int mode :
       std::array<int, 3>{kCentralDifference, kForwardDifference,
                          kBackwardDifference}) {
    const Eigen::VectorXd grad =
        cddp::finite_difference_gradient(quadratic_cost, x, 1e-6, mode);
    EXPECT_TRUE(grad.isApprox(grad_expected, 1e-4)) << "mode=" << mode;
  }
}

TEST(FiniteDifferenceTest, JacobianMatchesAnalyticForAllModes) {
  const Eigen::Vector2d x(-0.3, 0.6);
  const Eigen::Matrix2d jac_expected = expected_jacobian(x);

  for (const int mode :
       std::array<int, 3>{kCentralDifference, kForwardDifference,
                          kBackwardDifference}) {
    const Eigen::MatrixXd jac =
        cddp::finite_difference_jacobian(vector_function, x, 1e-6, mode);
    EXPECT_TRUE(jac.isApprox(jac_expected, 1e-4)) << "mode=" << mode;
  }
}

TEST(FiniteDifferenceTest, HessianMatchesAnalyticForAllModes) {
  const Eigen::Vector2d x(0.4, -0.2);
  const Eigen::Matrix2d hess_expected = expected_hessian(x);

  for (const int mode :
       std::array<int, 3>{kCentralDifference, kForwardDifference,
                          kBackwardDifference}) {
    const Eigen::MatrixXd hess =
        cddp::finite_difference_hessian(quadratic_cost, x, 1e-5, mode);
    EXPECT_TRUE(hess.isApprox(hess_expected, 2e-2)) << "mode=" << mode;
  }
}

TEST(FiniteDifferenceTest, InvalidModeReturnsZeroObjects) {
  const Eigen::Vector2d x(1.0, -1.0);

  const Eigen::VectorXd grad =
      cddp::finite_difference_gradient(quadratic_cost, x, 1e-6, 99);
  const Eigen::MatrixXd jac =
      cddp::finite_difference_jacobian(vector_function, x, 1e-6, 99);
  const Eigen::MatrixXd hess =
      cddp::finite_difference_hessian(quadratic_cost, x, 1e-6, 99);

  EXPECT_EQ(grad.size(), x.size());
  EXPECT_TRUE(grad.isZero());
  EXPECT_EQ(jac.rows(), 2);
  EXPECT_EQ(jac.cols(), x.size());
  EXPECT_TRUE(jac.isZero());
  EXPECT_EQ(hess.rows(), x.size());
  EXPECT_EQ(hess.cols(), x.size());
  EXPECT_TRUE(hess.isZero());
}
