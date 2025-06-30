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

#ifndef CDDP_IPDDP_SOLVER_HPP
#define CDDP_IPDDP_SOLVER_HPP

#include "cddp_core/cddp_core.hpp"
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>

namespace cddp
{

    /**
     * @brief Interior Point Differential Dynamic Programming (IPDDP) solver implementation.
     *
     * This class implements the ISolverAlgorithm interface to provide
     * an interior point method based DDP solver for handling inequality constraints.
     * Uses primal-dual interior point method with logarithmic barrier.
     */
    class IPDDPSolver : public ISolverAlgorithm
    {
    public:
        /**
         * @brief Default constructor.
         */
        IPDDPSolver();

        /**
         * @brief Initialize the solver with the given CDDP context.
         * @param context Reference to the CDDP instance containing problem data and options.
         */
        void initialize(CDDP &context) override;

        /**
         * @brief Execute the IPDDP algorithm and return the solution.
         * @param context Reference to the CDDP instance containing problem data and options.
         * @return CDDPSolution containing the results.
         */
        CDDPSolution solve(CDDP &context) override;

        /**
         * @brief Get the name of the solver algorithm.
         * @return String identifier "IPDDP".
         */
        std::string getSolverName() const override;

    private:
        // Dynamics storage
        std::vector<Eigen::MatrixXd> F_x_; ///< State jacobians (Fx)
        std::vector<Eigen::MatrixXd> F_u_; ///< Control jacobians (Fu)
        std::vector<std::vector<Eigen::MatrixXd>> F_xx_; ///< State hessians (Fxx)
        std::vector<std::vector<Eigen::MatrixXd>> F_uu_; ///< Control hessians (Fuu)
        std::vector<std::vector<Eigen::MatrixXd>> F_ux_; ///< Mixed hessians (Fux)
        
        // Constraint gradients storage (constraint name -> time trajectory)
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_x_; ///< Constraint state jacobians
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_u_; ///< Constraint control jacobians
        
        // Control law parameters
        std::vector<Eigen::VectorXd> k_u_; ///< Feedforward control gains
        std::vector<Eigen::MatrixXd> K_u_; ///< Feedback control gains
        Eigen::Vector2d dV_;               ///< Expected value function change

        // IPDDP-specific variables (constraint name -> time trajectory)
        std::map<std::string, std::vector<Eigen::VectorXd>> G_; ///< Constraint values g(x,u) - g_ub
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_; ///< Dual variables (Lagrange multipliers)
        std::map<std::string, std::vector<Eigen::VectorXd>> S_; ///< Slack variables
        
        // IPDDP gains for dual and slack variables
        std::map<std::string, std::vector<Eigen::VectorXd>> k_y_; ///< Feedforward gains for dual variables
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_y_; ///< Feedback gains for dual variables
        std::map<std::string, std::vector<Eigen::VectorXd>> k_s_; ///< Feedforward gains for slack variables
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_s_; ///< Feedback gains for slack variables

        // Interior point method parameters
        double mu_;                    ///< Barrier parameter
        std::vector<FilterPoint> filter_;  ///< Filter points for line search
        double constraint_violation_ = 0.0;

        /**
         * @brief Pre-compute dynamics jacobians and hessians for all time steps in parallel.
         * @param context Reference to the CDDP context.
         */
        void precomputeDynamicsDerivatives(CDDP &context);

        /**
         * @brief Pre-compute constraint jacobians for all time steps and constraints in parallel.
         * @param context Reference to the CDDP context.
         */
        void precomputeConstraintGradients(CDDP &context);

        /**
         * @brief Evaluate trajectory by computing cost, constraint values, and merit function.
         * @param context Reference to the CDDP context.
         */
        void evaluateTrajectory(CDDP &context);

        /**
         * @brief Evaluate trajectory for warm start without overwriting the state trajectory.
         * @param context Reference to the CDDP context.
         */
        void evaluateTrajectoryWarmStart(CDDP &context);

        /**
         * @brief Initialize dual and slack variables for warm start consistently with provided trajectory.
         * @param context Reference to the CDDP context.
         */
        void initializeDualSlackVariablesWarmStart(CDDP &context);

        /**
         * @brief Reset/initialize barrier parameters and filter.
         * @param context Reference to the CDDP context.
         */
        void resetBarrierFilter(CDDP &context);

        /**
         * @brief Reset the filter for line search.
         * @param context Reference to the CDDP context.
         */
        void resetFilter(CDDP &context);

        /**
         * @brief Perform backward pass (primal-dual Riccati recursion).
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
         * @brief Update barrier parameter based on convergence progress.
         * @param context Reference to the CDDP context.
         * @param forward_pass_success Whether the forward pass was successful.
         * @param termination_metric Current termination metric.
         */
        void updateBarrierParameters(CDDP &context, bool forward_pass_success, double termination_metric);

        /**
         * @brief Get total dual dimension across all constraints.
         * @param context Reference to the CDDP context.
         * @return Total number of dual variables.
         */
        int getTotalDualDim(const CDDP &context) const;

        /**
         * @brief Print iteration information in IPOPT style.
         */
        void printIteration(int iter, double objective, double inf_pr, double inf_du, 
                            double mu, double step_norm, double regularization, 
                            double alpha_du, double alpha_pr) const;

        /**
         * @brief Print solution summary.
         * @param solution The solution to print.
         */
        void printSolutionSummary(const CDDPSolution &solution) const;
    };

} // namespace cddp

#endif // CDDP_IPDDP_SOLVER_HPP
