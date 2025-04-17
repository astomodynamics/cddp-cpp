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
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <cmath> // For sqrt, atan2, asin, cos, sin
#include "cddp_core/helper.hpp"

namespace cddp
{
     namespace helper
     {

          // --- Conversion to Rotation Matrix --- //

          Eigen::Matrix3d eulerZYXToRotationMatrix(const Eigen::Vector3d &eulerAngles)
          {
               double psi = eulerAngles(0);   // Yaw
               double theta = eulerAngles(1); // Pitch
               double phi = eulerAngles(2);   // Roll

               double cpsi = cos(psi), spsi = sin(psi);
               double cth = cos(theta), sth = sin(theta);
               double cphi = cos(phi), sphi = sin(phi);

               Eigen::Matrix3d Rz, Ry, Rx;
               Rz << cpsi, -spsi, 0,
                   spsi, cpsi, 0,
                   0, 0, 1;

               Ry << cth, 0, sth,
                   0, 1, 0,
                   -sth, 0, cth;

               Rx << 1, 0, 0,
                   0, cphi, -sphi,
                   0, sphi, cphi;

               // ZYX (3-2-1) convention: R = Rz * Ry * Rx
               return Rz * Ry * Rx;
          }

          Eigen::Matrix3d quatToRotationMatrix(const Eigen::Vector4d &q)
          {
               Eigen::Vector4d q_norm = q.normalized(); // Ensure unit quaternion
               double w = q_norm(0);                    // scalar part
               double x = q_norm(1);                    // vector part x
               double y = q_norm(2);                    // vector part y
               double z = q_norm(3);                    // vector part z

               Eigen::Matrix3d R;
               R(0, 0) = 1.0 - 2.0 * (y * y + z * z);
               R(0, 1) = 2.0 * (x * y - w * z);
               R(0, 2) = 2.0 * (x * z + w * y);

               R(1, 0) = 2.0 * (x * y + w * z);
               R(1, 1) = 1.0 - 2.0 * (x * x + z * z);
               R(1, 2) = 2.0 * (y * z - w * x);

               R(2, 0) = 2.0 * (x * z - w * y);
               R(2, 1) = 2.0 * (y * z + w * x);
               R(2, 2) = 1.0 - 2.0 * (x * x + y * y);

               return R;
          }

          Eigen::Matrix3d mrpToRotationMatrix(const Eigen::Vector3d &mrp_in)
          {
               Eigen::Vector3d mrp = mrp_in;
               double s_sq = mrp.squaredNorm();

               // Explicitly handle shadow set: switch to principal MRP if norm > 1
               // The conversion formula works best with the principal set.
               if (s_sq > 1.0 + 1e-9) { // Add small tolerance for floating point
                   mrp = -mrp / s_sq;
                   s_sq = mrp.squaredNorm(); // Recalculate norm squared for the principal set
               }

               double den_inv = 1.0 / (1.0 + s_sq);

               Eigen::Matrix3d R;
               Eigen::Matrix3d S = skewMatrix(mrp); // Use skew of the (potentially switched) mrp

               // Use the formula R = I + (8*S^2 + 4*(1-s^2)*S) / (1+s^2)^2
               // Note: (1+s^2)^2 * den_inv^2 = 1.0
               R = Eigen::Matrix3d::Identity() + (8.0 * S * S + 4.0 * (1.0 - s_sq) * S) * den_inv * den_inv;

               return R;
          }

          Eigen::Vector3d rotationMatrixToEulerZYX(const Eigen::Matrix3d &R)
          {
               double psi, theta, phi;

               // Extract pitch (theta)
               theta = asin(-R(2, 0));

               // Check for gimbal lock (theta = +/- pi/2)
               if (abs(cos(theta)) > 1e-9)
               {
                    // Not in gimbal lock
                    psi = atan2(R(1, 0), R(0, 0));
                    phi = atan2(R(2, 1), R(2, 2));
               }
               else
               {
                    // Gimbal lock: Assuming phi = 0 (arbitrary choice)
                    phi = 0.0;
                    if (theta > 0)
                    { // theta = +pi/2
                         psi = atan2(R(0, 1), R(1, 1));
                    }
                    else
                    { // theta = -pi/2
                         psi = -atan2(R(0, 1), R(1, 1));
                    }
               }

               return Eigen::Vector3d(psi, theta, phi);
          }

          Eigen::Vector4d rotationMatrixToQuat(const Eigen::Matrix3d &R)
          {
               double trace = R.trace();
               double w, x, y, z;

               if (trace > 0.0)
               {
                    double S = sqrt(trace + 1.0) * 2.0; // S = 4*qw
                    w = 0.25 * S;
                    x = (R(2, 1) - R(1, 2)) / S;
                    y = (R(0, 2) - R(2, 0)) / S;
                    z = (R(1, 0) - R(0, 1)) / S;
               }
               else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2)))
               {
                    double S = sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0; // S = 4*qx
                    w = (R(2, 1) - R(1, 2)) / S;
                    x = 0.25 * S;
                    y = (R(0, 1) + R(1, 0)) / S;
                    z = (R(0, 2) + R(2, 0)) / S;
               }
               else if (R(1, 1) > R(2, 2))
               {
                    double S = sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0; // S = 4*qy
                    w = (R(0, 2) - R(2, 0)) / S;
                    x = (R(0, 1) + R(1, 0)) / S;
                    y = 0.25 * S;
                    z = (R(1, 2) + R(2, 1)) / S;
               }
               else
               {
                    double S = sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0; // S = 4*qz
                    w = (R(1, 0) - R(0, 1)) / S;
                    x = (R(0, 2) + R(2, 0)) / S;
                    y = (R(1, 2) + R(2, 1)) / S;
                    z = 0.25 * S;
               }

               Eigen::Vector4d quat(w, x, y, z);
               return quat.normalized(); // Ensure unit quaternion
          }

          Eigen::Vector3d rotationMatrixToMRP(const Eigen::Matrix3d &R)
          {
               // Convert R to quaternion first, then to MRP
               Eigen::Vector4d q = rotationMatrixToQuat(R);
               return quatToMRP(q);
          }

          Eigen::Vector3d quatToEulerZYX(const Eigen::Vector4d &q)
          {
               // Convert quat to rotation matrix, then to Euler
               return rotationMatrixToEulerZYX(quatToRotationMatrix(q));
          }

          Eigen::Vector3d mrpToEulerZYX(const Eigen::Vector3d &mrp)
          {
               // Convert mrp to rotation matrix, then to Euler
               return rotationMatrixToEulerZYX(mrpToRotationMatrix(mrp));
          }

          Eigen::Vector4d eulerZYXToQuat(const Eigen::Vector3d &eulerAngles)
          {
               // Convert Euler to rotation matrix, then to quat
               return rotationMatrixToQuat(eulerZYXToRotationMatrix(eulerAngles));
          }

          Eigen::Vector3d eulerZYXToMRP(const Eigen::Vector3d &eulerAngles)
          {
               // Convert Euler to rotation matrix, then to MRP
               return rotationMatrixToMRP(eulerZYXToRotationMatrix(eulerAngles));
          }

          Eigen::Vector3d quatToMRP(const Eigen::Vector4d &q)
          {
               Eigen::Vector4d q_norm = q.normalized();
               double w = q_norm(0);
               Eigen::Vector3d v = q_norm.tail<3>();

               // Check for singularity (180-degree rotation, w approaches 0)
               if (abs(1.0 + w) < 1e-9)
               {
                    // TODO: Handle singularity properly
                    return v / (1e-9); // Approximate infinity representation
               }

               return v / (1.0 + w);
          }

          Eigen::Vector4d mrpToQuat(const Eigen::Vector3d &mrp)
          {
               double mrp_sq_norm = mrp.squaredNorm();
               double den = 1.0 + mrp_sq_norm;
               double w = (1.0 - mrp_sq_norm) / den;
               Eigen::Vector3d v = (2.0 * mrp) / den;
               Eigen::Vector4d q(w, v(0), v(1), v(2));
               return q; // Already normalized by construction
          }

          Eigen::Matrix3d skewMatrix(const Eigen::Vector3d &v)
          {
               Eigen::Matrix3d S;
               S << 0.0, -v(2), v(1),
                   v(2), 0.0, -v(0),
                   -v(1), v(0), 0.0;
               return S;
          }
     } // namespace helper
} // namespace cddp