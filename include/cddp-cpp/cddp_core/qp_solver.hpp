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

#ifndef CDDP_QP_SOLVER_HPP
#define CDDP_QP_SOLVER_HPP

#include <Eigen/Dense>
#include <memory>
#include <random>
#include <vector>

namespace cddp {

/**
 * @brief Solver status codes
 */
enum class QPStatus {
  OPTIMAL = 0,        ///< Optimal solution found
  INFEASIBLE = 1,     ///< Problem is infeasible
  MAX_ITER = 2,       ///< Maximum iterations reached
  NUMERICAL_ERROR = 3 ///< Numerical issues encountered
};

/**
 * @brief Configuration parameters for QP solver
 */
struct QPSolverOptions {
  double eps = 1e-6;         ///< Numerical tolerance
  int max_iterations = 1000; ///< Maximum iterations
  bool warm_start = false;   ///< Use warm start if available
  bool verbose = false;      ///< Print debug info
  int random_seed = 42;      ///< Random seed for initialization
};

/**
 * @brief Result structure containing solution and status information
 */
struct QPResult {
  Eigen::VectorXd x;      ///< Optimal solution
  double objective_value; ///< Optimal objective value
  QPStatus status;        ///< Solution status
  int iterations;         ///< Number of iterations used
  double solve_time;      ///< Solution time in seconds
};

/**
 * @brief Quadratic Programming Solver using SDQP algorithm
 *
 * Solves problems of the form:
 * minimize    1/2 x'Qx + c'x
 * subject to  Ax <= b
 */
class QPSolver {
public:
  /**
   * @brief Constructor
   * @param options Solver configuration options
   */
  explicit QPSolver(const QPSolverOptions &options = QPSolverOptions());

  /**
   * @brief Set problem dimensions
   * @param num_vars Number of variables
   * @param num_constraints Number of constraints
   */
  void setDimensions(int num_vars, int num_constraints);

  /**
   * @brief Set the quadratic cost matrix Q
   * @param Q Quadratic cost matrix (must be positive definite)
   */
  void setHessian(const Eigen::MatrixXd &Q);

  /**
   * @brief Set the linear cost vector c
   * @param c Linear cost vector
   */
  void setGradient(const Eigen::VectorXd &c);

  /**
   * @brief Set the constraint matrix A and vector b
   * @param A Constraint matrix
   * @param b Constraint vector
   */
  void setConstraints(const Eigen::MatrixXd &A, const Eigen::VectorXd &b);

  /**
   * @brief Solve the QP problem
   * @return QPResult containing solution and status
   */
  QPResult solve();

private:
  QPSolverOptions options_;
  int num_vars_;
  int num_constraints_;

  Eigen::MatrixXd Q_; // Quadratic cost matrix
  Eigen::VectorXd c_; // Linear cost vector
  Eigen::MatrixXd A_; // Constraint matrix
  Eigen::VectorXd b_; // Constraint vector

  // Work matrices/vectors
  Eigen::MatrixXd halves_;
  std::vector<Eigen::VectorXd> work_vectors_;
  std::vector<int> work_indices_;

  /**
   * @brief Solve minimum norm problem (internal)
   * @param x Solution vector
   * @return QPStatus solution status
   */
  QPStatus solveMinNorm(Eigen::VectorXd &x);

  /**
   * @brief Generate random permutation (internal)
   * @param n Size of permutation
   * @param perm Output permutation vector
   */
  void generateRandomPermutation(int n, Eigen::VectorXi &perm);

  /**
   * @brief Move element to front of linked list (internal)
   * @param i Index to move
   * @param next Next pointers
   * @param prev Previous pointers
   * @return Previous index
   */
  static int moveToFront(int i, Eigen::VectorXi &next, Eigen::VectorXi &prev);

  // Random number generator
  std::mt19937 rng_;
};

} // namespace cddp

#endif // CDDP_QP_SOLVER_HPP