/*
 Copyright 2025 Tomo Sasaki

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

#ifndef CDDP_SOLVER_BASE_HPP
#define CDDP_SOLVER_BASE_HPP

#include "cddp_core/cddp_core.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace cddp {

/**
 * @brief Base class for all DDP solver implementations.
 *
 * Provides a template method solve() that defines the common iteration
 * structure shared by CLDDP, LogDDP, IPDDP, and MSIPDDP. Subclasses
 * override virtual hooks to customize solver-specific behavior.
 */
class CDDPSolverBase : public ISolverAlgorithm {
public:
  virtual ~CDDPSolverBase() = default;

  /**
   * @brief Main solve loop (template method). Marked final.
   *
   * Subclasses customize behavior via protected virtual hooks.
   */
  CDDPSolution solve(CDDP &context) override final;

protected:
  // === Common member variables ===
  std::vector<Eigen::VectorXd> k_u_; ///< Feedforward control gains
  std::vector<Eigen::MatrixXd> K_u_; ///< Feedback control gains
  Eigen::Vector2d dV_;               ///< Expected value function change

  // Dynamics derivatives (pre-computed by barrier/IP solvers)
  std::vector<Eigen::MatrixXd> F_x_; ///< State Jacobians
  std::vector<Eigen::MatrixXd> F_u_; ///< Control Jacobians
  std::vector<std::vector<Eigen::MatrixXd>> F_xx_, F_uu_,
      F_ux_; ///< Dynamics Hessians

  // === Virtual hooks for solver customization ===

  /**
   * @brief Called before the main loop. Evaluate trajectory, reset filter, etc.
   */
  virtual void preIterationSetup(CDDP &context) {}

  /**
   * @brief Perform the backward pass (Riccati recursion).
   * @return True if successful, false if regularization needed.
   */
  virtual bool backwardPass(CDDP &context) = 0;

  /**
   * @brief Optional early convergence check after backward pass, before
   * forward pass. Override to check dual infeasibility without running
   * an unnecessary forward pass (e.g., CLDDP checks inf_du here).
   * @return True if converged.
   */
  virtual bool checkEarlyConvergence(CDDP &context, int iter,
                                     std::string &reason) {
    return false;
  }

  /**
   * @brief Perform a single forward pass trial with given step size.
   */
  virtual ForwardPassResult forwardPass(CDDP &context, double alpha) = 0;

  /**
   * @brief Apply a successful forward pass result to update context state.
   * Default: updates X_, U_, cost_, merit_function_, alpha_pr_.
   * Override to also update dual/slack/costate variables.
   */
  virtual void applyForwardPassResult(CDDP &context,
                                      const ForwardPassResult &result);

  /**
   * @brief Check convergence criteria.
   * @param dJ Change in cost from last iteration.
   * @param dL Change in merit/Lagrangian from last iteration.
   * @param iter Current iteration number.
   * @param[out] reason Termination reason string if converged.
   * @return True if converged.
   */
  virtual bool checkConvergence(CDDP &context, double dJ, double dL, int iter,
                                std::string &reason) = 0;

  /**
   * @brief Called after each iteration (barrier parameter update, etc.).
   * @param forward_pass_success Whether the forward pass succeeded.
   */
  virtual void postIterationUpdate(CDDP &context, bool forward_pass_success) {}

  /**
   * @brief Handle forward pass failure. Default: increase regularization.
   * Override for filter restoration (MSIPDDP).
   * @return True if the solver should break out of the main loop.
   */
  virtual bool handleForwardPassFailure(CDDP &context,
                                        std::string &termination_reason);

  /**
   * @brief Record iteration history. Called after successful forward pass.
   * Default records objective, merit, step lengths, regularization.
   * Override to add barrier/IP specific metrics.
   */
  virtual void recordIterationHistory(const CDDP &context);

  /**
   * @brief Populate solver-specific fields in the solution.
   * Called after the main loop. Override to add barrier/dual/slack fields.
   */
  virtual void populateSolverSpecificSolution(CDDPSolution &solution,
                                              const CDDP &context) {}

  /**
   * @brief Print iteration info. Solver-specific column layouts.
   */
  virtual void printIteration(int iter, const CDDP &context) const = 0;

  /**
   * @brief Print solution summary.
   */
  virtual void printSolutionSummary(const CDDPSolution &solution) const;

  // === Shared implementations ===

  /**
   * @brief Perform forward pass with line search (sequential or parallel).
   */
  ForwardPassResult performForwardPass(CDDP &context);

  /**
   * @brief Pre-compute dynamics Jacobians and optionally Hessians.
   */
  void precomputeDynamicsDerivatives(CDDP &context,
                                     int min_horizon_for_parallel = 50);

  /**
   * @brief Initialize control gain storage to zeros.
   */
  void initializeGains(int horizon, int control_dim, int state_dim);

  /**
   * @brief Build time points vector from context.
   */
  static std::vector<double> buildTimePoints(const CDDP &context);

  /**
   * @brief Compute trajectory cost (running + terminal).
   */
  void computeCost(CDDP &context);

  // History tracking (reuses CDDPSolution::History to avoid type duplication)
  CDDPSolution::History history_;
};

} // namespace cddp

#endif // CDDP_SOLVER_BASE_HPP
