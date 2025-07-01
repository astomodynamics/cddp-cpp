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

#ifndef CDDP_LOGDDP_SOLVER_HPP
#define CDDP_LOGDDP_SOLVER_HPP

#include "cddp_core/barrier.hpp"
#include "cddp_core/cddp_core.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace cddp {

/**
 * @brief Log-barrier DDP (LogDDP) solver implementation.
 *
 * This class implements the ISolverAlgorithm interface to provide
 * a log-barrier based DDP solver for handling inequality constraints.
 */
class LogDDPSolver : public ISolverAlgorithm {
public:
  /**
   * @brief Default constructor.
   */
  LogDDPSolver();

  /**
   * @brief Initialize the solver with the given CDDP context.
   * @param context Reference to the CDDP instance containing problem data and
   * options.
   */
  void initialize(CDDP &context) override;

  /**
   * @brief Execute the LogDDP algorithm and return the solution.
   * @param context Reference to the CDDP instance containing problem data and
   * options.
   * @return CDDPSolution containing the results.
   */
  CDDPSolution solve(CDDP &context) override;

  /**
   * @brief Get the name of the solver algorithm.
   * @return String identifier "LogDDP".
   */
  std::string getSolverName() const override;

private:
  // Dynamics storage
  std::vector<Eigen::VectorXd> F_;                 ///< Dynamics evaluations
  std::vector<Eigen::MatrixXd> F_x_;               ///< State jacobians (Fx)
  std::vector<Eigen::MatrixXd> F_u_;               ///< Control jacobians (Fu)
  std::vector<std::vector<Eigen::MatrixXd>> F_xx_; ///< State hessians (Fxx)
  std::vector<std::vector<Eigen::MatrixXd>> F_uu_; ///< Control hessians (Fuu)
  std::vector<std::vector<Eigen::MatrixXd>> F_ux_; ///< Mixed hessians (Fux)

  // Control law parameters
  std::vector<Eigen::VectorXd> k_u_; ///< Feedforward control gains
  std::vector<Eigen::MatrixXd> K_u_; ///< Feedback control gains
  Eigen::Vector2d dV_;               ///< Expected value function change

  // Log-barrier method
  std::map<std::string, std::vector<Eigen::VectorXd>>
      G_; ///< Constraint values g(x,u) - g_ub
  std::unique_ptr<RelaxedLogBarrier>
      relaxed_log_barrier_; ///< Log barrier object
  double mu_;               ///< Barrier parameter
  double relaxation_delta_; ///< Relaxation parameter

  // Filter-based line search
  double constraint_violation_; ///< Current constraint violation measure

  // Multi-shooting parameters
  int ms_segment_length_; ///< Multi-shooting segment length

  /**
   * @brief Pre-compute dynamics jacobians and hessians for all time steps in
   * parallel.
   * @param context Reference to the CDDP context.
   */
  void precomputeDynamicsDerivatives(CDDP &context);

  /**
   * @brief Efficiently compute only jacobians using execution policies.
   * @param context Reference to the CDDP context.
   */
  void precomputeJacobiansOnly(CDDP &context);

  /**
   * @brief Cache-friendly sequential computation with better memory access
   * patterns.
   * @param context Reference to the CDDP context.
   */
  void precomputeDynamicsDerivativesOptimized(CDDP &context);

  /**
   * @brief Evaluate trajectory by computing cost, dynamics, and merit function.
   * @param context Reference to the CDDP context.
   */
  void evaluateTrajectory(CDDP &context);

  /**
   * @brief Reset/initialize the filter for line search.
   * @param context Reference to the CDDP context.
   */
  void resetFilter(CDDP &context);

  /**
   * @brief Perform backward pass (Riccati recursion with log-barrier terms).
   * @param context Reference to the CDDP context.
   * @return True if backward pass succeeds, false otherwise.
   */
  bool backwardPass(CDDP &context);

  /**
   * @brief Perform forward pass with line search.
   * @param context Reference to the CDDP context.
   * @return Best forward pass result.
   */
  ForwardPassResult performForwardPass(CDDP &context);

  /**
   * @brief Perform single forward pass with given step size.
   * @param context Reference to the CDDP context.
   * @param alpha Step size for the forward pass.
   * @return Forward pass result.
   */
  ForwardPassResult forwardPass(CDDP &context, double alpha);

  /**
   * @brief Update barrier parameters.
   * @param context Reference to the CDDP context.
   * @param forward_pass_success Whether the forward pass was successful.
   * @param termination_metric Current termination metric.
   */
  void updateBarrierParameters(CDDP &context, bool forward_pass_success,
                               double termination_metric);

  /**
   * @brief Print iteration information.
   */
  void printIteration(int iter, double cost, double lagrangian, double opt_gap,
                      double regularization, double alpha, double mu,
                      double constraint_violation) const;

  /**
   * @brief Print solution summary.
   * @param solution The solution to print.
   */
  void printSolutionSummary(const CDDPSolution &solution) const;
};

} // namespace cddp

#endif // CDDP_LOGDDP_SOLVER_HPP
