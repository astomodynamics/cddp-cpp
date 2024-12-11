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

namespace cddp {

/**
 * @brief Configuration options for the box-constrained QP solver
 */
struct BoxQPOptions {
    int maxIter = 100;               ///< Maximum number of iterations
    double minGrad = 1e-8;           ///< Minimum norm of non-fixed gradient
    double minRelImprove = 1e-8;     ///< Minimum relative improvement
    double stepDec = 0.6;            ///< Factor for decreasing stepsize
    double minStep = 1e-22;          ///< Minimal stepsize for linesearch
    double armijo = 0.1;             ///< Armijo parameter
    bool verbose = false;            ///< Print debug info
};

/**
 * @brief Result status from the box-constrained QP solver
 */
enum class BoxQPStatus {
    HESSIAN_NOT_PD = -1,         ///< Hessian is not positive definite
    NO_DESCENT = 0,              ///< No descent direction found
    MAX_ITER_EXCEEDED = 1,       ///< Maximum iterations exceeded
    MAX_LS_EXCEEDED = 2,         ///< Maximum line search iterations exceeded 
    NO_BOUNDS = 3,               ///< No bounds, returning Newton point
    SMALL_IMPROVEMENT = 4,       ///< Improvement smaller than tolerance
    SMALL_GRADIENT = 5,          ///< Gradient norm smaller than tolerance
    ALL_CLAMPED = 6              ///< All dimensions are clamped
};

/**
 * @brief Results from the box-constrained QP solver
 */
struct BoxQPResult {
    Eigen::VectorXd x;               ///< Solution vector
    BoxQPStatus status;              ///< Result status
    Eigen::LDLT<Eigen::MatrixXd> Hfree;  ///< Subspace Cholesky factor
    Eigen::VectorXi free;            ///< Set of free dimensions
    double final_value;              ///< Final objective value
    double final_grad_norm;          ///< Final gradient norm
    int iterations;                  ///< Number of iterations taken
    int factorizations;              ///< Number of matrix factorizations
};

/**
 * @brief Box-constrained Quadratic Programming solver
 * 
 * Solves optimization problems of the form:
 * minimize    (1/2)x'Hx + g'x
 * subject to  lower ≤ x ≤ upper
 */
class BoxQPSolver {
public:
    /**
     * @brief Construct a new BoxQPSolver
     * @param options Solver configuration options
     */
    explicit BoxQPSolver(const BoxQPOptions& options = BoxQPOptions());

    /**
     * @brief Solve a box-constrained QP problem
     * @param H Quadratic term (must be positive definite)
     * @param g Linear term
     * @param lower Lower bounds
     * @param upper Upper bounds
     * @param x0 Initial guess (optional)
     * @return BoxQPResult containing solution and solver status
     */
    BoxQPResult solve(const Eigen::MatrixXd& H, 
                     const Eigen::VectorXd& g,
                     const Eigen::VectorXd& lower, 
                     const Eigen::VectorXd& upper,
                     const Eigen::VectorXd& x0 = Eigen::VectorXd());

    /**
     * @brief Get the current solver options
     * @return const reference to current options
     */
    const BoxQPOptions& getOptions() const { return options_; }

    /**
     * @brief Set new solver options
     * @param options New options to use
     */
    void setOptions(const BoxQPOptions& options) { options_ = options; }

private:
    BoxQPOptions options_;  ///< Solver configuration options

    /**
     * @brief Initialize the state vector x
     * @param x0 Initial guess (if provided)
     * @param lower Lower bounds
     * @param upper Upper bounds
     * @param n Problem dimension
     * @return Initialized and bound-feasible x vector
     */
    Eigen::VectorXd initializeX(const Eigen::VectorXd& x0,
                               const Eigen::VectorXd& lower,
                               const Eigen::VectorXd& upper,
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
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& search,
        const Eigen::VectorXd& gradient,
        double value,
        const Eigen::MatrixXd& H,
        const Eigen::VectorXd& g,
        const Eigen::VectorXd& lower,
        const Eigen::VectorXd& upper);

    /**
     * @brief Evaluate the objective function value
     * @param x Point to evaluate
     * @param H Quadratic term
     * @param g Linear term 
     * @return Objective value
     */
    double evaluateObjective(const Eigen::VectorXd& x,
                           const Eigen::MatrixXd& H,
                           const Eigen::VectorXd& g) const;

    /**
     * @brief Project a point onto the box constraints
     * @param x Point to project
     * @param lower Lower bounds
     * @param upper Upper bounds
     * @return Projected point
     */
    Eigen::VectorXd projectOntoBox(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& lower,
                                  const Eigen::VectorXd& upper) const;
};

} // namespace cddp

#endif // CDDP_HELPER_HPP