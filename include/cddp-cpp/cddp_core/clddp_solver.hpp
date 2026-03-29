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

#ifndef CDDP_CLDDP_SOLVER_HPP
#define CDDP_CLDDP_SOLVER_HPP

#include "cddp_core/boxqp.hpp"
#include "cddp_core/cddp_solver_base.hpp"
#include "cddp_core/constraint.hpp"
#include <Eigen/Dense>
#include <vector>

namespace cddp {

/**
 * @brief Constrained Linear DDP (CLDDP) solver implementation.
 */
class CLDDPSolver : public CDDPSolverBase {
public:
  CLDDPSolver();

  void initialize(CDDP &context) override;
  std::string getSolverName() const override;

protected:
  bool backwardPass(CDDP &context) override;
  ForwardPassResult forwardPass(CDDP &context, double alpha) override;
  bool checkConvergence(CDDP &context, double dJ, double dL, int iter,
                        std::string &reason) override;
  void printIteration(int iter, const CDDP &context) const override;

private:
  BoxQPSolver boxqp_solver_;
};

} // namespace cddp

#endif // CDDP_CLDDP_SOLVER_HPP
