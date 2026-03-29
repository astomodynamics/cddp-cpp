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

#ifndef CDDP_HELPER_HPP
#define CDDP_HELPER_HPP

#include <Eigen/Dense>
#include <iostream>

namespace cddp
{
     /**
      * @brief Compute gradient using finite differences
      * @param f Scalar function to differentiate
      * @param x Point at which to evaluate gradient
      * @param h Step size for finite differences
      * @param mode 0 = central, 1 = forward, 2 = backward
      * @return Gradient vector
      */
     template <typename F>
     Eigen::VectorXd finite_difference_gradient(const F &f,
                                                const Eigen::VectorXd &x,
                                                double h = 2e-5,
                                                int mode = 0)
     {
          const int n = x.size();
          Eigen::VectorXd grad(n);
          Eigen::VectorXd x_perturbed = x;

          if (mode == 0)
          {
               // Central differences
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) + h;
                    double f_plus = f(x_perturbed);
                    x_perturbed(i) = x(i) - h;
                    double f_minus = f(x_perturbed);
                    grad(i) = (f_plus - f_minus) / (2.0 * h);
                    x_perturbed(i) = x(i);
               }
          }
          else if (mode == 1)
          {
               // Forward differences
               const double f_x = f(x);
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) + h;
                    grad(i) = (f(x_perturbed) - f_x) / h;
                    x_perturbed(i) = x(i);
               }
          }
          else if (mode == 2)
          {
               // Backward differences
               const double f_x = f(x);
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) - h;
                    grad(i) = (f_x - f(x_perturbed)) / h;
                    x_perturbed(i) = x(i);
               }
          }
          else
          {
               std::cerr << "Invalid mode value for finite difference gradient" << std::endl;
               return Eigen::VectorXd::Zero(n);
          }

          return grad;
     }

     /**
      * @brief Compute Jacobian using finite differences
      * @param f Vector function to differentiate
      * @param x Point at which to evaluate Jacobian
      * @param h Step size for finite differences
      * @param mode 0 = central, 1 = forward, 2 = backward
      * @return Jacobian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_jacobian(const F &f,
                                                const Eigen::VectorXd &x,
                                                double h = 2e-5,
                                                int mode = 0)
     {
          const Eigen::VectorXd f_x = f(x);
          const int m = f_x.size();
          const int n = x.size();
          Eigen::MatrixXd jac(m, n);
          Eigen::VectorXd x_perturbed = x;

          if (mode == 0)
          {
               // Central differences
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) + h;
                    Eigen::VectorXd f_plus = f(x_perturbed);
                    x_perturbed(i) = x(i) - h;
                    Eigen::VectorXd f_minus = f(x_perturbed);
                    jac.col(i) = (f_plus - f_minus) / (2.0 * h);
                    x_perturbed(i) = x(i);
               }
          }
          else if (mode == 1)
          {
               // Forward differences
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) + h;
                    jac.col(i) = (f(x_perturbed) - f_x) / h;
                    x_perturbed(i) = x(i);
               }
          }
          else if (mode == 2)
          {
               // Backward differences
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) - h;
                    jac.col(i) = (f_x - f(x_perturbed)) / h;
                    x_perturbed(i) = x(i);
               }
          }
          else
          {
               std::cerr << "Invalid mode value for finite difference Jacobian" << std::endl;
               return Eigen::MatrixXd::Zero(m, n);
          }

          return jac;
     }

     /**
      * @brief Compute Hessian using finite differences
      * @param f Scalar function to differentiate
      * @param x Point at which to evaluate Hessian
      * @param h Step size for finite differences
      * @param mode 0 = central, 1 = forward, 2 = backward
      * @return Hessian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_hessian(const F &f,
                                               const Eigen::VectorXd &x,
                                               double h = 2e-5,
                                               int mode = 0)
     {
          const int n = x.size();
          Eigen::MatrixXd hess(n, n);
          Eigen::VectorXd x_perturbed = x;

          if (mode == 0)
          {
               // Central differences of gradients
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) + h;
                    Eigen::VectorXd grad_plus = finite_difference_gradient(f, x_perturbed, h);
                    x_perturbed(i) = x(i) - h;
                    Eigen::VectorXd grad_minus = finite_difference_gradient(f, x_perturbed, h);
                    hess.col(i) = (grad_plus - grad_minus) / (2.0 * h);
                    x_perturbed(i) = x(i);
               }
          }
          else if (mode == 1)
          {
               // Forward differences of gradients
               Eigen::VectorXd grad_x = finite_difference_gradient(f, x, h);
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) + h;
                    hess.col(i) = (finite_difference_gradient(f, x_perturbed, h) - grad_x) / h;
                    x_perturbed(i) = x(i);
               }
          }
          else if (mode == 2)
          {
               // Backward differences of gradients
               Eigen::VectorXd grad_x = finite_difference_gradient(f, x, h);
               for (int i = 0; i < n; ++i)
               {
                    x_perturbed(i) = x(i) - h;
                    hess.col(i) = (grad_x - finite_difference_gradient(f, x_perturbed, h)) / h;
                    x_perturbed(i) = x(i);
               }
          }
          else
          {
               std::cerr << "Invalid mode value for finite difference Hessian" << std::endl;
               return Eigen::MatrixXd::Zero(n, n);
          }

          return hess;
     }

     // Forward declarations for attitude conversion helper functions
     namespace helper
     {

          // Conversion to Rotation Matrix
          Eigen::Matrix3d eulerZYXToRotationMatrix(const Eigen::Vector3d &eulerAngles);
          Eigen::Matrix3d quatToRotationMatrix(const Eigen::Vector4d &q);
          Eigen::Matrix3d mrpToRotationMatrix(const Eigen::Vector3d &mrp);

          // Conversion from Rotation Matrix
          Eigen::Vector3d rotationMatrixToEulerZYX(const Eigen::Matrix3d &R);
          Eigen::Vector4d rotationMatrixToQuat(const Eigen::Matrix3d &R);
          Eigen::Vector3d rotationMatrixToMRP(const Eigen::Matrix3d &R);

          // Direct Conversions (use rotation matrix as intermediate for simplicity)
          Eigen::Vector3d quatToEulerZYX(const Eigen::Vector4d &q);
          Eigen::Vector3d mrpToEulerZYX(const Eigen::Vector3d &mrp);
          Eigen::Vector4d eulerZYXToQuat(const Eigen::Vector3d &eulerAngles);
          Eigen::Vector3d eulerZYXToMRP(const Eigen::Vector3d &eulerAngles);
          Eigen::Vector3d quatToMRP(const Eigen::Vector4d &q);
          Eigen::Vector4d mrpToQuat(const Eigen::Vector3d &mrp);

          // Skew Symmetric Matrix
          Eigen::Matrix3d skewMatrix(const Eigen::Vector3d &v);

     } // namespace helper
} // namespace cddp

#endif // CDDP_HELPER_HPP
