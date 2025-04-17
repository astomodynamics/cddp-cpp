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
      * @brief Compute gradient using central finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate gradient
      * @param h Step size for finite differences
      * @return Gradient vector
      */
     template <typename F>
     Eigen::VectorXd finite_difference_gradient_central(const F &f,
                                                        const Eigen::VectorXd &x,
                                                        double h)
     {
          const int n = x.size();
          Eigen::VectorXd grad(n);

          // Compute central differences
          Eigen::VectorXd x_plus = x;
          Eigen::VectorXd x_minus = x;

          for (int i = 0; i < n; ++i)
          {
               x_plus(i) = x(i) + h;
               x_minus(i) = x(i) - h;

               grad(i) = (f(x_plus) - f(x_minus)) / (2.0 * h);

               x_plus(i) = x(i);
               x_minus(i) = x(i);
          }

          return grad;
     }

     /**
      * @brief Compute gradient using forward finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate gradient
      * @param h Step size for finite differences
      * @return Gradient vector
      */
     template <typename F>
     Eigen::VectorXd finite_difference_gradient_forward(const F &f,
                                                        const Eigen::VectorXd &x,
                                                        double h)
     {
          const int n = x.size();
          Eigen::VectorXd grad(n);

          // Compute forward differences
          Eigen::VectorXd x_plus = x;

          for (int i = 0; i < n; ++i)
          {
               x_plus(i) = x(i) + h;

               grad(i) = (f(x_plus) - f(x)) / h;

               x_plus(i) = x(i);
          }

          return grad;
     }

     /**
      * @brief Compute gradient using backward finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate gradient
      * @param h Step size for finite differences
      * @return Gradient vector
      */
     template <typename F>
     Eigen::VectorXd finite_difference_gradient_backward(const F &f,
                                                         const Eigen::VectorXd &x,
                                                         double h)
     {
          const int n = x.size();
          Eigen::VectorXd grad(n);

          // Compute backward differences
          Eigen::VectorXd x_minus = x;

          for (int i = 0; i < n; ++i)
          {
               x_minus(i) = x(i) - h;

               grad(i) = (f(x) - f(x_minus)) / h;

               x_minus(i) = x(i);
          }

          return grad;
     }

     /**
      * @brief Compute gradient using central finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate gradient
      * @param h Step size for finite differences (optional)
      * @param mode mode for differentiating options: 0 for central, 1 for forward, 2 for backward
      * @return Gradient vector
      */
     template <typename F>
     Eigen::VectorXd finite_difference_gradient(const F &f,
                                                const Eigen::VectorXd &x,
                                                double h = 2e-5,
                                                int mode = 0)
     {
          if (mode == 0)
          {
               return finite_difference_gradient_central(f, x, h);
          }
          else if (mode == 1)
          {
               return finite_difference_gradient_forward(f, x, h);
          }
          else if (mode == 2)
          {
               return finite_difference_gradient_backward(f, x, h);
          }
          else
          {
               std::cerr << "Invalid mode value for finite difference gradient" << std::endl;
               return Eigen::VectorXd::Zero(x.size());
          }
          return Eigen::VectorXd::Zero(x.size());
     }

     /**
      * @brief Compute Jacobian using central finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Jacobian
      * @param h Step size for finite differences
      * @return Jacobian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_jacobian_central(const F &f,
                                                        const Eigen::VectorXd &x,
                                                        double h)
     {
          const int m = f(x).size();
          const int n = x.size();
          Eigen::MatrixXd jac(m, n);

          // Compute central differences
          Eigen::VectorXd x_plus = x;
          Eigen::VectorXd x_minus = x;

          for (int i = 0; i < n; ++i)
          {
               x_plus(i) = x(i) + h;
               x_minus(i) = x(i) - h;

               Eigen::VectorXd f_plus = f(x_plus);
               Eigen::VectorXd f_minus = f(x_minus);

               jac.col(i) = (f_plus - f_minus) / (2.0 * h);

               x_plus(i) = x(i);
               x_minus(i) = x(i);
          }

          return jac;
     }

     /**
      * @brief Compute Jacobian using forward finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Jacobian
      * @param h Step size for finite differences
      * @return Jacobian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_jacobian_forward(const F &f,
                                                        const Eigen::VectorXd &x,
                                                        double h)
     {
          const int m = f(x).size();
          const int n = x.size();
          Eigen::MatrixXd jac(m, n);

          // Compute forward differences
          Eigen::VectorXd x_plus = x;

          for (int i = 0; i < n; ++i)
          {
               x_plus(i) = x(i) + h;

               Eigen::VectorXd f_plus = f(x_plus);
               jac.col(i) = (f_plus - f(x)) / h;

               x_plus(i) = x(i);
          }

          return jac;
     }

     /**
      * @brief Compute Jacobian using backward finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Jacobian
      * @param h Step size for finite differences
      * @return Jacobian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_jacobian_backward(const F &f,
                                                         const Eigen::VectorXd &x,
                                                         double h)
     {
          const int m = f(x).size();
          const int n = x.size();
          Eigen::MatrixXd jac(m, n);

          // Compute backward differences
          Eigen::VectorXd x_minus = x;

          for (int i = 0; i < n; ++i)
          {
               x_minus(i) = x(i) - h;

               Eigen::VectorXd f_minus = f(x_minus);
               jac.col(i) = (f(x) - f_minus) / h;

               x_minus(i) = x(i);
          }

          return jac;
     }

     /*
      * @brief Compute Jacobian using central finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Jacobian
      * @param h Step size for finite differences (optional)
      * @param mode mode for differentiating options: 0 for central, 1 for forward, 2 for backward
      * @return Jacobian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_jacobian(const F &f,
                                                const Eigen::VectorXd &x,
                                                double h = 2e-5,
                                                int mode = 0)
     {
          if (mode == 0)
          {
               return finite_difference_jacobian_central(f, x, h);
          }
          else if (mode == 1)
          {
               return finite_difference_jacobian_forward(f, x, h);
          }
          else if (mode == 2)
          {
               return finite_difference_jacobian_backward(f, x, h);
          }
          else
          {
               std::cerr << "Invalid mode value for finite difference Jacobian" << std::endl;
               return Eigen::MatrixXd::Zero(f(x).size(), x.size());
          }
          return Eigen::MatrixXd::Zero(f(x).size(), x.size());
     }

     /**
      * @brief Compute Hessian using central finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Hessian
      * @param h Step size for finite differences
      * @return Hessian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_hessian_central(const F &f,
                                                       const Eigen::VectorXd &x,
                                                       double h)
     {
          const int n = x.size();
          Eigen::MatrixXd hess(n, n);

          // Compute central differences
          Eigen::VectorXd x_plus = x;
          Eigen::VectorXd x_minus = x;

          for (int i = 0; i < n; ++i)
          {
               x_plus(i) = x(i) + h;
               x_minus(i) = x(i) - h;

               Eigen::VectorXd grad_plus = finite_difference_gradient(f, x_plus, h);
               Eigen::VectorXd grad_minus = finite_difference_gradient(f, x_minus, h);

               hess.col(i) = (grad_plus - grad_minus) / (2.0 * h);

               x_plus(i) = x(i);
               x_minus(i) = x(i);
          }

          return hess;
     }

     /**
      * @brief Compute Hessian using forward finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Hessian
      * @param h Step size for finite differences
      * @return Hessian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_hessian_forward(const F &f,
                                                       const Eigen::VectorXd &x,
                                                       double h)
     {
          const int n = x.size();
          Eigen::MatrixXd hess(n, n);

          // Compute forward differences
          Eigen::VectorXd x_plus = x;

          for (int i = 0; i < n; ++i)
          {
               x_plus(i) = x(i) + h;

               Eigen::VectorXd grad_plus = finite_difference_gradient(f, x_plus, h);
               hess.col(i) = (grad_plus - finite_difference_gradient(f, x, h)) / h;

               x_plus(i) = x(i);
          }

          return hess;
     }

     /**
      * @brief Compute Hessian using backward finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Hessian
      * @param h Step size for finite differences
      * @return Hessian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_hessian_backward(const F &f,
                                                        const Eigen::VectorXd &x,
                                                        double h)
     {
          const int n = x.size();
          Eigen::MatrixXd hess(n, n);

          // Compute backward differences
          Eigen::VectorXd x_minus = x;

          for (int i = 0; i < n; ++i)
          {
               x_minus(i) = x(i) - h;

               Eigen::VectorXd grad_minus = finite_difference_gradient(f, x_minus, h);
               hess.col(i) = (finite_difference_gradient(f, x, h) - grad_minus) / h;

               x_minus(i) = x(i);
          }

          return hess;
     }

     /**
      * @brief Compute Hessian using central finite differences
      * @param f Function to differentiate
      * @param x Point at which to evaluate Hessian
      * @param h Step size for finite differences (optional)
      * @param mode mode for differentiating options: 0 for central, 1 for forward, 2 for backward
      * @return Hessian matrix
      */
     template <typename F>
     Eigen::MatrixXd finite_difference_hessian(const F &f,
                                               const Eigen::VectorXd &x,
                                               double h = 2e-5,
                                               int mode = 0)
     {
          if (mode == 0)
          {
               return finite_difference_hessian_central(f, x, h);
          }
          else if (mode == 1)
          {
               return finite_difference_hessian_forward(f, x, h);
          }
          else if (mode == 2)
          {
               return finite_difference_hessian_backward(f, x, h);
          }
          else
          {
               std::cerr << "Invalid mode value for finite difference Hessian" << std::endl;
               return Eigen::MatrixXd::Zero(x.size(), x.size());
          }
          return Eigen::MatrixXd::Zero(x.size(), x.size());
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