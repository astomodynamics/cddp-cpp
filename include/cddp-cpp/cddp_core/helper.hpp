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

namespace cddp
{
/**
 * @brief Compute gradient using central finite differences
 * @param f Function to differentiate
 * @param x Point at which to evaluate gradient
 * @param h Step size for finite differences (optional)
 * @return Gradient vector
 */
template <typename F>
Eigen::VectorXd finite_difference_gradient(const F &f,
                                             const Eigen::VectorXd &x,
                                             double h = 2e-8) 
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

          double f_plus = f(x_plus);
          double f_minus = f(x_minus);

          grad(i) = (f_plus - f_minus) / (2.0 * h);

          x_plus(i) = x(i);
          x_minus(i) = x(i);
     }

     return grad;
}

/*
 * @brief Compute Jacobian using central finite differences
 * @param f Function to differentiate
 * @param x Point at which to evaluate Jacobian
 * @param h Step size for finite differences (optional)
 * @return Jacobian matrix
*/
template <typename F>
Eigen::MatrixXd finite_difference_jacobian(const F &f,
                                             const Eigen::VectorXd &x,
                                             double h = 2e-8)
{
     const int n = x.size();
     const int m = f(x).size();
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
 * @brief Compute Hessian using central finite differences
 * @param f Function to differentiate
 * @param x Point at which to evaluate Hessian
 * @param h Step size for finite differences (optional)
 * @return Hessian matrix
 */
template <typename F>
Eigen::MatrixXd finite_difference_hessian(const F &f,
                                             const Eigen::VectorXd &x,
                                             double h = 2e-8) 
{
     const int n = x.size();
     Eigen::MatrixXd hess(n, n);

     // Compute central differences for second derivatives
     Eigen::VectorXd x_pp = x; // x plus-plus
     Eigen::VectorXd x_pm = x; // x plus-minus
     Eigen::VectorXd x_mp = x; // x minus-plus
     Eigen::VectorXd x_mm = x; // x minus-minus

     // Diagonal terms
     for (int i = 0; i < n; ++i)
     {
          x_pp(i) = x(i) + h;
          x_mm(i) = x(i) - h;

          double f_pp = f(x_pp);
          double f_mm = f(x_mm);
          double f_0 = f(x);

          hess(i, i) = (f_pp - 2.0 * f_0 + f_mm) / (h * h);

          x_pp(i) = x(i);
          x_mm(i) = x(i);
     }

     // Off-diagonal terms
     for (int i = 0; i < n; ++i)
     {
          for (int j = i + 1; j < n; ++j)
          {
               x_pp(i) = x(i) + h;
               x_pp(j) = x(j) + h;
               x_pm(i) = x(i) + h;
               x_pm(j) = x(j) - h;
               x_mp(i) = x(i) - h;
               x_mp(j) = x(j) + h;
               x_mm(i) = x(i) - h;
               x_mm(j) = x(j) - h;

               double f_pp = f(x_pp);
               double f_pm = f(x_pm);
               double f_mp = f(x_mp);
               double f_mm = f(x_mm);

               hess(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
               hess(j, i) = hess(i, j); // Symmetry

               x_pp(i) = x(i);
               x_pp(j) = x(j);
               x_pm(i) = x(i);
               x_pm(j) = x(j);
               x_mp(i) = x(i);
               x_mp(j) = x(j);
               x_mm(i) = x(i);
               x_mm(j) = x(j);
          }
     }

     return hess;
}

/**
 * @brief Configuration options for the box-constrained QP solver
 */
struct BoxQPOptions
{
     int maxIter = 100;           ///< Maximum number of iterations
     double minGrad = 1e-8;       ///< Minimum norm of non-fixed gradient
     double minRelImprove = 1e-8; ///< Minimum relative improvement
     double stepDec = 0.6;        ///< Factor for decreasing stepsize
     double minStep = 1e-22;      ///< Minimal stepsize for linesearch
     double armijo = 0.1;         ///< Armijo parameter
     bool verbose = false;        ///< Print debug info
};

/**
 * @brief Result status from the box-constrained QP solver
 */
enum class BoxQPStatus
{
     HESSIAN_NOT_PD = -1,   ///< Hessian is not positive definite
     NO_DESCENT = 0,        ///< No descent direction found
     MAX_ITER_EXCEEDED = 1, ///< Maximum iterations exceeded
     MAX_LS_EXCEEDED = 2,   ///< Maximum line search iterations exceeded
     NO_BOUNDS = 3,         ///< No bounds, returning Newton point
     SMALL_IMPROVEMENT = 4, ///< Improvement smaller than tolerance
     SMALL_GRADIENT = 5,    ///< Gradient norm smaller than tolerance
     ALL_CLAMPED = 6        ///< All dimensions are clamped
};

/**
 * @brief Results from the box-constrained QP solver
 */
struct BoxQPResult
{
     Eigen::VectorXd x;                  ///< Solution vector
     BoxQPStatus status;                 ///< Result status
     Eigen::LDLT<Eigen::MatrixXd> Hfree; ///< Subspace Cholesky factor
     Eigen::VectorXi free;               ///< Set of free dimensions
     double final_value;                 ///< Final objective value
     double final_grad_norm;             ///< Final gradient norm
     int iterations;                     ///< Number of iterations taken
     int factorizations;                 ///< Number of matrix factorizations
};

/**
 * @brief Box-constrained Quadratic Programming solver
 *
 * Solves optimization problems of the form:
 * minimize    (1/2)x'Hx + g'x
 * subject to  lower ≤ x ≤ upper
 */
class BoxQPSolver
{
public:
     /**
      * @brief Construct a new BoxQPSolver
      * @param options Solver configuration options
      */
     explicit BoxQPSolver(const BoxQPOptions &options = BoxQPOptions());

     /**
      * @brief Solve a box-constrained QP problem
      * @param H Quadratic term (must be positive definite)
      * @param g Linear term
      * @param lower Lower bounds
      * @param upper Upper bounds
      * @param x0 Initial guess (optional)
      * @return BoxQPResult containing solution and solver status
      */
     BoxQPResult solve(const Eigen::MatrixXd &H,
                         const Eigen::VectorXd &g,
                         const Eigen::VectorXd &lower,
                         const Eigen::VectorXd &upper,
                         const Eigen::VectorXd &x0 = Eigen::VectorXd());

     /**
      * @brief Get the current solver options
      * @return const reference to current options
      */
     const BoxQPOptions &getOptions() const { return options_; }

     /**
      * @brief Set new solver options
      * @param options New options to use
      */
     void setOptions(const BoxQPOptions &options) { options_ = options; }

private:
     BoxQPOptions options_; ///< Solver configuration options

     /**
      * @brief Initialize the state vector x
      * @param x0 Initial guess (if provided)
      * @param lower Lower bounds
      * @param upper Upper bounds
      * @param n Problem dimension
      * @return Initialized and bound-feasible x vector
      */
     Eigen::VectorXd initializeX(const Eigen::VectorXd &x0,
                                   const Eigen::VectorXd &lower,
                                   const Eigen::VectorXd &upper,
                                   int n);

     /**
      * @brief Perform line search with Armijo condition
      * @param x Current point
      * @param search Search direction
      * @param gradient Current gradient
      * @param value Current objective value
      * @param H Quadratic term
      * @param g Linear term
      * @param lower Lower bounds
      * @param upper Upper bounds
      * @return std::pair<bool, std::pair<double, Eigen::VectorXd>> Success flag, step size, and new point
      */
     std::pair<bool, std::pair<double, Eigen::VectorXd>> lineSearch(
          const Eigen::VectorXd &x,
          const Eigen::VectorXd &search,
          const Eigen::VectorXd &gradient,
          double value,
          const Eigen::MatrixXd &H,
          const Eigen::VectorXd &g,
          const Eigen::VectorXd &lower,
          const Eigen::VectorXd &upper);

     /**
      * @brief Evaluate the objective function value
      * @param x Point to evaluate
      * @param H Quadratic term
      * @param g Linear term
      * @return Objective value
      */
     double evaluateObjective(const Eigen::VectorXd &x,
                              const Eigen::MatrixXd &H,
                              const Eigen::VectorXd &g) const;

     /**
      * @brief Project a point onto the box constraints
      * @param x Point to project
      * @param lower Lower bounds
      * @param upper Upper bounds
      * @return Projected point
      */
     Eigen::VectorXd projectOntoBox(const Eigen::VectorXd &x,
                                        const Eigen::VectorXd &lower,
                                        const Eigen::VectorXd &upper) const;
};

} // namespace cddp

#endif // CDDP_HELPER_HPP