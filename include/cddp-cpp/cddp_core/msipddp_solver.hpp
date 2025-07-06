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

#ifndef CDDP_MSIPDDP_SOLVER_HPP
#define CDDP_MSIPDDP_SOLVER_HPP

#include "cddp_core/cddp_core.hpp"
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace cddp
{

    /**
     * @brief Interior Point Differential Dynamic Programming (MSIPDDP) solver.
     *
     * Implements ISolverAlgorithm interface for inequality constrained problems
     * using primal-dual interior point method with logarithmic barrier.
     */
    class MSIPDDPSolver : public ISolverAlgorithm
    {
    public:
        /**
         * @brief Default constructor.
         */
        MSIPDDPSolver();

        /**
         * @brief Initialize solver with CDDP context.
         * @param context CDDP instance with problem data and options.
         */
        void initialize(CDDP &context) override;

        /**
         * @brief Execute MSIPDDP algorithm.
         * @param context CDDP instance with problem data and options.
         * @return Solution containing trajectories and statistics.
         */
        CDDPSolution solve(CDDP &context) override;

        /**
         * @brief Get solver name.
         * @return "MSIPDDP"
         */
        std::string getSolverName() const override;

    protected:
        /**
         * @brief Modify terminal value function for constraints.
         * 
         * This method can be overridden by derived classes (like TCMSIPDDP)
         * to incorporate terminal constraints into the value function.
         * 
         * @param context CDDP context
         * @param V_x Terminal value gradient (input/output)
         * @param V_xx Terminal value Hessian (input/output)
         * @return True if successful
         */
        virtual bool modifyTerminalValueFunction(CDDP &context,
                                               Eigen::VectorXd& V_x,
                                               Eigen::MatrixXd& V_xx);

        // Protected members for derived class access
        double mu_;                       ///< Barrier parameter
        std::vector<FilterPoint> filter_; ///< Filter for line search
        
        // Control law
        std::vector<Eigen::VectorXd> k_u_; ///< Feedforward gains
        std::vector<Eigen::MatrixXd> K_u_; ///< Feedback gains
        Eigen::Vector2d dV_;               ///< Expected value change

        // Interior point variables
        std::map<std::string, std::vector<Eigen::VectorXd>> G_; ///< Constraint values
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_; ///< Dual variables
        std::map<std::string, std::vector<Eigen::VectorXd>> S_; ///< Slack variables
        
        // Terminal constraint variables - separated by type
        // For inequality constraints: g(x) ≤ upper_bound
        std::map<std::string, Eigen::VectorXd> G_ineq_terminal_; ///< Terminal inequality constraint values
        std::map<std::string, Eigen::VectorXd> Y_ineq_terminal_; ///< Terminal inequality dual variables
        std::map<std::string, Eigen::VectorXd> S_ineq_terminal_; ///< Terminal inequality slack variables
        
        // For equality constraints: h(x) = bound
        std::map<std::string, Eigen::VectorXd> G_eq_terminal_; ///< Terminal equality constraint values
        std::map<std::string, Eigen::VectorXd> Y_eq_terminal_; ///< Terminal equality dual variables (no slack needed)
        
        // Terminal constraint derivatives
        std::map<std::string, Eigen::MatrixXd> G_x_ineq_terminal_; ///< Terminal inequality constraint gradients
        std::map<std::string, Eigen::MatrixXd> G_x_eq_terminal_; ///< Terminal equality constraint gradients
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_xx_ineq_terminal_; ///< Terminal inequality Hessians
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_xx_eq_terminal_; ///< Terminal equality Hessians

    private:
        // Dynamics storage
        std::vector<Eigen::VectorXd> F_;   ///< Dynamics evaluations
        std::vector<Eigen::MatrixXd> F_x_;               ///< State jacobians
        std::vector<Eigen::MatrixXd> F_u_;               ///< Control jacobians
        std::vector<std::vector<Eigen::MatrixXd>> F_xx_; ///< State hessians
        std::vector<std::vector<Eigen::MatrixXd>> F_uu_; ///< Control hessians
        std::vector<std::vector<Eigen::MatrixXd>> F_ux_; ///< Mixed hessians

        // Constraint derivatives
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_x_; ///< State gradients
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_u_; ///< Control gradients
        std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>>
            G_xx_; ///< Constraint state hessians (time x dual_dim)
        std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>>
            G_uu_; ///< Constraint control hessians (time x dual_dim)
        std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>>
            G_ux_; ///< Constraint mixed hessians (time x dual_dim)

        // Interior point gains
        std::map<std::string, std::vector<Eigen::VectorXd>> k_y_; ///< Dual feedforward
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_y_; ///< Dual feedback
        std::map<std::string, std::vector<Eigen::VectorXd>> k_s_; ///< Slack feedforward
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_s_; ///< Slack feedback
        
        // Terminal constraint gains (inequality only needs slack gains)
        std::map<std::string, Eigen::VectorXd> k_y_ineq_terminal_; ///< Terminal inequality dual feedforward
        std::map<std::string, Eigen::VectorXd> k_s_ineq_terminal_; ///< Terminal inequality slack feedforward
        std::map<std::string, Eigen::VectorXd> k_y_eq_terminal_; ///< Terminal equality dual feedforward

        // MSIPDDP-specific costate variables and gains
        std::vector<Eigen::VectorXd>
            Lambda_; ///< Costate variables (Lagrange multipliers for dynamics)
        std::vector<Eigen::VectorXd>
            k_lambda_; ///< Feedforward gains for costate variables
        std::vector<Eigen::MatrixXd>
            K_lambda_; ///< Feedback gains for costate variables
        
        // Multi-shooting specific parameters
        int ms_segment_length_; ///< Length of shooting segments

        /**
         * @brief Precompute dynamics derivatives in parallel.
         */
        void precomputeDynamicsDerivatives(CDDP &context);

        /**
         * @brief Precompute constraint gradients in parallel.
         */
        void precomputeConstraintGradients(CDDP &context);

        /**
         * @brief Evaluate trajectory cost and constraints.
         */
        void evaluateTrajectory(CDDP &context);

        /**
         * @brief Evaluate trajectory for warm start.
         */
        void evaluateTrajectoryWarmStart(CDDP &context);

        /**
         * @brief Initialize dual/slack/costate variables for warm start.
         */
        void initializeDualSlackCostateVariablesWarmStart(CDDP &context);

        /**
         * @brief Initialize dual/slack/costate variables for cold start.
         */
        void initializeDualSlackCostateVariables(CDDP &context);

        /**
         * @brief Reset barrier parameters and filter.
         */
        void resetBarrierFilter(CDDP &context);

        /**
         * @brief Reset filter for line search.
         */
        void resetFilter(CDDP &context);

        /**
         * @brief Accept new filter entry with domination check.
         * @param merit_function Merit function value
         * @param constraint_violation Constraint violation measure
         * @return true if entry was accepted, false otherwise
         */
        bool acceptFilterEntry(double merit_function, double constraint_violation);

        /**
         * @brief Check if candidate point is acceptable to filter.
         * @param merit_function Merit function value
         * @param constraint_violation Constraint violation measure  
         * @param options Filter options
         * @param expected_improvement Expected improvement from model
         * @return true if point is acceptable, false otherwise
         */
        bool isFilterAcceptable(double merit_function, double constraint_violation,
                               const SolverSpecificFilterOptions &options,
                               double expected_improvement) const;

        /**
         * @brief Check if filter needs restoration and perform if necessary.
         * @param context CDDP context
         * @return true if restoration was successful, false otherwise
         */
        bool checkAndPerformFilterRestoration(CDDP &context);

        /**
         * @brief Perform backward pass (primal-dual Riccati).
         * @return True if successful.
         */
        virtual bool backwardPass(CDDP &context);

        /**
         * @brief Perform forward pass with line search.
         * @return Best result.
         */
        ForwardPassResult performForwardPass(CDDP &context);

        /**
         * @brief Single forward pass with given step size.
         * @param alpha Step size.
         * @return Forward pass result.
         */
        virtual ForwardPassResult forwardPass(CDDP &context, double alpha);

        /**
         * @brief Update barrier parameter.
         */
        virtual void updateBarrierParameters(CDDP &context, bool forward_pass_success);

        /**
         * @brief Get total dual dimension.
         */
        int getTotalDualDim(const CDDP &context) const;
        
        /**
         * @brief Get total terminal dual dimension.
         */
        int getTotalTerminalDualDim(const CDDP &context) const;

        /**
         * @brief Update iteration history if tracking enabled.
         */
        void updateIterationHistory(
            const CDDPOptions &options, const CDDP &context,
            std::vector<double> &history_objective, std::vector<double> &history_merit_function,
            std::vector<double> &history_step_length_primal, std::vector<double> &history_step_length_dual,
            std::vector<double> &history_dual_infeasibility, std::vector<double> &history_primal_infeasibility,
            std::vector<double> &history_complementary_infeasibility, std::vector<double> &history_barrier_mu,
            double alpha_du) const;

        /**
         * @brief Check convergence criteria.
         * @return True if converged.
         */
        virtual bool checkConvergence(const CDDPOptions &options, const CDDP &context,
                              double dJ, int iter, std::string &termination_reason) const;



        /**
         * @brief Initialize constraint storage containers.
         */
        void initializeConstraintStorage(CDDP &context);

        /**
         * @brief Compute maximum constraint violation.
         */
        double computeMaxConstraintViolation(const CDDP &context) const;

        /**
         * @brief Compute IPOPT-style scaled dual infeasibility.
         * @param context CDDP context with dual/slack variables.
         * @return Scaled dual infeasibility metric.
         */
        virtual double computeScaledDualInfeasibility(const CDDP &context) const;

        /**
         * @brief Print iteration info (IPOPT style).
         */
        void printIteration(int iter, double objective, double inf_pr, double inf_du,
                            double inf_comp, double mu, double step_norm, double regularization,
                            double alpha_du, double alpha_pr) const;

        /**
         * @brief Print solution summary.
         */
        void printSolutionSummary(const CDDPSolution &solution) const;
    };

} // namespace cddp

#endif // CDDP_MSIPDDP_SOLVER_HPP
