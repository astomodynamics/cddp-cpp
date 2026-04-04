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

#include "gtest/gtest.h"

#include "cddp-cpp/cddp_core/helper.hpp"

TEST(HelperConversionTest, EulerRotationRoundTripIsConsistent) {
  const Eigen::Vector3d euler(0.3, -0.2, 0.5);

  const Eigen::Matrix3d rotation = cddp::helper::eulerZYXToRotationMatrix(euler);
  const Eigen::Vector3d recovered = cddp::helper::rotationMatrixToEulerZYX(rotation);
  const Eigen::Matrix3d recovered_rotation =
      cddp::helper::eulerZYXToRotationMatrix(recovered);

  EXPECT_TRUE(rotation.isApprox(recovered_rotation, 1e-9));
}

TEST(HelperConversionTest, RotationMatrixToEulerHandlesGimbalLock) {
  const double half_pi = std::acos(-1.0) / 2.0;
  const Eigen::Vector3d euler(0.5, half_pi, -0.2);

  const Eigen::Matrix3d rotation = cddp::helper::eulerZYXToRotationMatrix(euler);
  const Eigen::Vector3d recovered = cddp::helper::rotationMatrixToEulerZYX(rotation);
  const Eigen::Matrix3d recovered_rotation =
      cddp::helper::eulerZYXToRotationMatrix(recovered);

  EXPECT_TRUE(rotation.isApprox(recovered_rotation, 1e-9));
}

TEST(HelperConversionTest, ShadowAndPrincipalMrpRepresentSameRotation) {
  const Eigen::Vector3d principal_mrp(0.2, -0.1, 0.15);
  const Eigen::Vector3d shadow_mrp =
      -principal_mrp / principal_mrp.squaredNorm();

  const Eigen::Matrix3d principal_rotation =
      cddp::helper::mrpToRotationMatrix(principal_mrp);
  const Eigen::Matrix3d shadow_rotation =
      cddp::helper::mrpToRotationMatrix(shadow_mrp);

  EXPECT_TRUE(principal_rotation.isApprox(shadow_rotation, 1e-9));
}

TEST(HelperConversionTest, QuaternionSingularityProducesFiniteMrp) {
  const Eigen::Vector4d quaternion(0.0, 1.0, 0.0, 0.0);

  const Eigen::Vector3d mrp = cddp::helper::quatToMRP(quaternion);

  EXPECT_TRUE(mrp.allFinite());
  EXPECT_GT(mrp.norm(), 1e6);
}

TEST(HelperConversionTest, SkewMatrixMatchesCrossProduct) {
  const Eigen::Vector3d a(1.0, -2.0, 0.5);
  const Eigen::Vector3d b(-0.2, 0.4, 2.0);

  const Eigen::Matrix3d skew = cddp::helper::skewMatrix(a);

  EXPECT_TRUE(skew.transpose().isApprox(-skew, 1e-12));
  EXPECT_TRUE((skew * b).isApprox(a.cross(b), 1e-12));
}
