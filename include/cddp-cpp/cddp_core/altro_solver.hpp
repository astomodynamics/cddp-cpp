/*
 * Copyright 2024 Tomo Sasaki
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CDDP_ALTRO_SOLVER_HPP
#define CDDP_ALTRO_SOLVER_HPP

#include "cddp_core/cddp_core.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace cddp
{

    /**
     * @brief Augmented Lagrangian Trajectory Optimizer (ALTRO) solver implementation.
     *
     * This class implements the ISolverAlgorithm interface to provide
     * an augmented lagrangian-based DDP solver for handling constraints.
     */
    class AltroSolver : public ISolverAlgorithm
    {
    public:
        /**
         * @brief Default constructor.
         */
        AltroSolver();

        /**
         * @brief Initialize the solver with the given CDDP context.
         * @param context Reference to the CDDP instance containing problem data and options.
         */
        void initialize(CDDP &context) override;

        /**
         * @brief Execute the ALTRO algorithm and return the solution.
         * @param context Reference to the CDDP instance containing problem data and options.
         * @return CDDPSolution containing the results.
         */
        CDDPSolution solve(CDDP &context) override;

        /**
         * @brief Get the name of the solver algorithm.
         * @return String identifier "Altro".
         */
        std::string getSolverName() const override;

    private:
        // Control law parameters
        std::vector<Eigen::VectorXd> k_u_; ///< Feedforward control gains
        std::vector<Eigen::MatrixXd> K_u_; ///< Feedback control gains

        // Dynamics storage
        std::vector<Eigen::VectorXd> F_;  ///< Dynamics evaluations

        // Altro-specific variables (constraint name -> time trajectory)
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_; ///< Dual variables (Lagrange multipliers)
        std::vector<Eigen::VectorXd> Lambda_; ///< Lagrange multipliers for defect constraints

        double cost_;                                               ///< Current total cost
        double constraint_violation_;                               ///< Current constraint violation measure
        double lagrangian_value_;                                   ///< Augmented Lagrangian value (cost + penalty terms)
        double optimality_gap_;                                     ///< Norm of the gradient of the Lagrangian

        /**
         * @brief Evaluate the trajectory, computing cost, dynamics, and augmented lagrangian.
         * @param context Reference to the CDDP context.
         */
        void evaluateTrajectory(CDDP &context);
        
        /**
         * @brief Perform the backward pass (Riccati recursion with augmented Lagrangian terms).
         * @param context Reference to the CDDP context.
         * @return True if the backward pass succeeds, false otherwise.
         */
        bool backwardPass(CDDP &context);

        /**
         * @brief Perform the forward pass with a line search to find an acceptable step size.
         * @param context Reference to the CDDP context.
         * @return The result of the best forward pass.
         */
        ForwardPassResult performForwardPass(CDDP &context);

        /**
         * @brief Perform a single forward pass with a given step size alpha.
         * @param context Reference to the CDDP context.
         * @param alpha The step size for the forward pass.
         * @return The result of the forward pass.
         */
        ForwardPassResult forwardPass(CDDP &context, double alpha);

        /**
         * @brief Update augmented Lagrangian parameters (multipliers and penalties).
         * @param context Reference to the CDDP context.
         */
        void updateAugmentedLagrangian(CDDP &context);

        /**
         * @brief Print iteration information to the console.
         */
        void printIteration(int iter, double cost, double lagrangian, double grad_norm,
                            double regularization, double alpha, double mu, double constraint_violation) const;

        /**
         * @brief Print a summary of the final solution.
         * @param solution The solution to print.
         */
        void printSolutionSummary(const CDDPSolution &solution) const;
    };

} // namespace cddp

#endif // CDDP_ALTRO_SOLVER_HPP