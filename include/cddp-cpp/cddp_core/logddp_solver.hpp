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
#include "cddp_core/cddp_solver_base.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace cddp {

/**
 * @brief Log-barrier DDP (LogDDP) solver implementation.
 *
 * Inherits from CDDPSolverBase (template method pattern) and overrides
 * the virtual hooks for log-barrier specific behavior.
 */
class LogDDPSolver : public CDDPSolverBase {
public:
  LogDDPSolver();

  void initialize(CDDP &context) override;
  std::string getSolverName() const override;

protected:
  // === Virtual hook overrides ===
  void preIterationSetup(CDDP &context) override;
  bool backwardPass(CDDP &context) override;
  ForwardPassResult forwardPass(CDDP &context, double alpha) override;
  void applyForwardPassResult(CDDP &context,
                              const ForwardPassResult &result) override;
  bool checkConvergence(CDDP &context, double dJ, double dL, int iter,
                        std::string &reason) override;
  void postIterationUpdate(CDDP &context, bool forward_pass_success) override;
  void recordIterationHistory(const CDDP &context) override;
  void populateSolverSpecificSolution(CDDPSolution &solution,
                                      const CDDP &context) override;
  void printIteration(int iter, const CDDP &context) const override;

private:
  // Dynamics storage (forward-simulated trajectory)
  std::vector<Eigen::VectorXd> F_;

  // Constraint values g(x,u) - g_ub
  std::map<std::string, std::vector<Eigen::VectorXd>> G_;

  // Log-barrier method
  std::unique_ptr<RelaxedLogBarrier> relaxed_log_barrier_;
  double mu_;
  double relaxation_delta_;

  // Filter-based line search
  double constraint_violation_;

  // Multi-shooting parameters
  int ms_segment_length_;

  /**
   * @brief Evaluate trajectory by computing cost, dynamics, and merit function.
   */
  void evaluateTrajectory(CDDP &context);

  /**
   * @brief Reset/initialize the filter for line search.
   */
  void resetFilter(CDDP &context);
};

} // namespace cddp

#endif // CDDP_LOGDDP_SOLVER_HPP
