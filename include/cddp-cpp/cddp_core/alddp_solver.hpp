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

#ifndef CDDP_ALDDP_SOLVER_HPP
#define CDDP_ALDDP_SOLVER_HPP

#include "cddp_core/boxqp.hpp"
#include "cddp_core/cddp_solver_base.hpp"
#include "cddp_core/constraint.hpp"
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace cddp {

/**
 * @brief Augmented Lagrangian DDP (ALDDP) solver implementation.
 *
 * Implements an ALTRO-like solver that uses augmented Lagrangian penalization
 * to handle general nonlinear state and control constraints. Features:
 * - AL outer loop with multiplier/penalty updates
 * - BoxQP for control bounds (optional, like CLDDP)
 * - Infeasible state trajectory initialization via slack controls
 * - Square-root backward pass for numerical robustness
 */
class ALDDPSolver : public CDDPSolverBase {
public:
  ALDDPSolver();

  void initialize(CDDP &context) override;
  std::string getSolverName() const override;

protected:
  // CDDPSolverBase hooks
  void preIterationSetup(CDDP &context) override;
  bool backwardPass(CDDP &context) override;
  bool checkEarlyConvergence(CDDP &context, int iter,
                             std::string &reason) override;
  ForwardPassResult forwardPass(CDDP &context, double alpha) override;
  void applyForwardPassResult(CDDP &context,
                              const ForwardPassResult &result) override;
  bool checkConvergence(CDDP &context, double dJ, double dL, int iter,
                        std::string &reason) override;
  void postIterationUpdate(CDDP &context, bool forward_pass_success) override;
  bool handleBackwardPassRegularizationLimit(
      CDDP &context, std::string &termination_reason) override;
  void recordIterationHistory(const CDDP &context) override;
  void populateSolverSpecificSolution(CDDPSolution &solution,
                                      const CDDP &context) override;
  void printIteration(int iter, const CDDP &context) const override;
  void printSolutionSummary(const CDDPSolution &solution) const override;

private:
  // --- AL variables ---
  // Per-constraint, per-timestep Lagrange multipliers (dim = getDualDim())
  std::map<std::string, std::vector<Eigen::VectorXd>> lambda_;
  // Per-constraint, per-timestep penalty weights
  std::map<std::string, std::vector<Eigen::VectorXd>> penalty_;
  // Terminal constraint multipliers and penalties
  std::map<std::string, Eigen::VectorXd> terminal_lambda_;
  std::map<std::string, Eigen::VectorXd> terminal_penalty_;

  // Constraint evaluation cache (only written by evaluateConstraints, read by backward/forward pass)
  std::map<std::string, std::vector<Eigen::VectorXd>> G_;
  std::map<std::string, Eigen::VectorXd> G_terminal_;
  std::map<std::string, std::vector<Eigen::MatrixXd>> G_x_;
  std::map<std::string, std::vector<Eigen::MatrixXd>> G_u_;

  // BoxQP for control bounds
  BoxQPSolver boxqp_solver_;

  // AL outer loop state
  int al_outer_iter_ = 0;
  double inner_tolerance_ = 1e-2;
  double max_constraint_violation_ = 0.0;
  double prev_max_constraint_violation_ = 0.0;
  double prev_outer_cost_ = 0.0;
  bool inner_converged_ = false;
  int inner_iter_count_ = 0;

  // Infeasible initialization
  bool infeasible_start_ = false;
  double current_slack_penalty_ = 1000.0;
  std::vector<Eigen::VectorXd> S_;   // Slack controls s_k (state_dim each)
  std::vector<Eigen::VectorXd> k_s_; // Slack feedforward gains
  std::vector<Eigen::MatrixXd> K_s_; // Slack feedback gains

  // Square-root backward pass workspace
  std::vector<Eigen::MatrixXd> S_chol_; // Upper-triangular Cholesky factors √P_k
  std::vector<Eigen::VectorXd> p_;      // Cost-to-go gradient vectors

  // --- Private helpers ---
  void initializeMultipliersAndPenalties(CDDP &context);
  void initializeSlackControls(CDDP &context);
  void evaluateConstraints(CDDP &context);
  void updateMultipliers(CDDP &context);
  void updatePenalties(CDDP &context);
  double computeMaxConstraintViolation() const;
  double computeMaxSlackNorm() const;

  // Backward pass variants
  bool backwardPassStandard(CDDP &context);
  bool backwardPassSqrt(CDDP &context);

  // AL cost derivative augmentation helpers
  void augmentRunningCostDerivatives(
      const CDDP &context, int t,
      const Eigen::VectorXd &x, const Eigen::VectorXd &u,
      Eigen::VectorXd &l_x, Eigen::VectorXd &l_u,
      Eigen::MatrixXd &l_xx, Eigen::MatrixXd &l_uu,
      Eigen::MatrixXd &l_ux) const;

  void augmentTerminalCostDerivatives(
      const CDDP &context, const Eigen::VectorXd &x_N,
      Eigen::VectorXd &V_x, Eigen::MatrixXd &V_xx) const;

  // Compute AL merit for a given trajectory (thread-safe, no member writes)
  double computeALMerit(
      const CDDP &context,
      const std::vector<Eigen::VectorXd> &X,
      const std::vector<Eigen::VectorXd> &U,
      const std::vector<Eigen::VectorXd> *S_trial = nullptr) const;
};

} // namespace cddp

#endif // CDDP_ALDDP_SOLVER_HPP
