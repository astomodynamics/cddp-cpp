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

#ifndef CDDP_ASDDP_SOLVER_HPP
#define CDDP_ASDDP_SOLVER_HPP

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/constraint.hpp"
#include <Eigen/Dense>
#include <vector>
#include "osqp++.h"

namespace cddp
{

    /**
     * @brief Active Set Constrained DDP (ASDDP) solver implementation.
     *
     * This class implements the ISolverAlgorithm interface to provide
     * an active set constrained DDP solver that uses OSQP to solve
     * QP subproblems at each time step for constraint handling.
     *
     * Uses OSQP as the internal QP solver for handling constrained
     * optimization subproblems in the forward pass.
     */
    class ASDDPSolver : public ISolverAlgorithm
    {
    public:
        /**
         * @brief Default constructor.
         */
        ASDDPSolver();

        /**
         * @brief Initialize the solver with the given CDDP context.
         * @param context Reference to the CDDP instance containing problem data and options.
         */
        void initialize(CDDP &context) override;

        /**
         * @brief Execute the ASDDP algorithm and return the solution.
         * @param context Reference to the CDDP instance containing problem data and options.
         * @return CDDPSolution containing the results.
         */
        CDDPSolution solve(CDDP &context) override;

        /**
         * @brief Get the name of the solver algorithm.
         * @return String identifier "ASDDP".
         */
        std::string getSolverName() const override;

    private:
        // Control law parameters
        std::vector<Eigen::VectorXd> k_u_; ///< Feedforward control gains
        std::vector<Eigen::MatrixXd> K_u_; ///< Feedback control gains
        Eigen::Vector2d dV_;               ///< Expected value function change

        // Q-function matrices for active set method
        std::vector<Eigen::MatrixXd> Q_UU_; ///< Control Hessian matrices
        std::vector<Eigen::MatrixXd> Q_UX_; ///< Control-state cross-derivatives
        std::vector<Eigen::VectorXd> Q_U_;  ///< Control gradients

        /**
         * @brief Perform backward pass (Riccati recursion) with active set method.
         * @param context Reference to the CDDP context.
         * @return True if backward pass succeeds, false otherwise.
         */
        bool backwardPass(CDDP &context);

        /**
         * @brief Perform forward pass with line search and OSQP-based constraint handling.
         * @param context Reference to the CDDP context.
         * @return Best forward pass result.
         */
        ForwardPassResult performForwardPass(CDDP &context);

        /**
         * @brief Perform single forward pass with given step size using OSQP.
         * @param context Reference to the CDDP context.
         * @param alpha Step size for the forward pass.
         * @return Forward pass result.
         */
        ForwardPassResult forwardPass(CDDP &context, double alpha);

        /**
         * @brief Compute the current cost given the trajectories.
         * @param context Reference to the CDDP context.
         */
        void computeCost(CDDP &context);

        /**
         * @brief Print iteration information.
         */
        void printIteration(int iter, double cost, double merit, double inf_du,
                            double regularization, double alpha) const;

        /**
         * @brief Print solution summary.
         * @param solution The solution to print.
         */
        void printSolutionSummary(const CDDPSolution &solution) const;
    };

} // namespace cddp

#endif // CDDP_ASDDP_SOLVER_HPP
