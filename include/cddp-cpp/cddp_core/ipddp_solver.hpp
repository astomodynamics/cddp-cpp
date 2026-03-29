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

#include "cddp_core/cddp_solver_base.hpp"
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace cddp
{

    /**
     * @brief Interior Point Differential Dynamic Programming (IPDDP) solver.
     *
     * Inherits from CDDPSolverBase and overrides virtual hooks
     * for primal-dual interior point method with logarithmic barrier.
     */
    class IPDDPSolver : public CDDPSolverBase
    {
    public:
        IPDDPSolver();

        void initialize(CDDP &context) override;
        std::string getSolverName() const override;

    protected:
        // === CDDPSolverBase virtual hook overrides ===
        void preIterationSetup(CDDP &context) override;
        bool backwardPass(CDDP &context) override;
        ForwardPassResult forwardPass(CDDP &context, double alpha) override;
        void applyForwardPassResult(CDDP &context, const ForwardPassResult &result) override;
        bool checkConvergence(CDDP &context, double dJ, double dL, int iter,
                              std::string &reason) override;
        void postIterationUpdate(CDDP &context, bool forward_pass_success) override;
        void recordIterationHistory(const CDDP &context) override;
        void populateSolverSpecificSolution(CDDPSolution &solution,
                                            const CDDP &context) override;
        void printIteration(int iter, const CDDP &context) const override;
        void printSolutionSummary(const CDDPSolution &solution) const override;

    private:
        // Constraint derivatives
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_x_; ///< State gradients
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_u_; ///< Control gradients
        std::map<std::string, std::vector<Eigen::MatrixXd>>
            G_xx_; ///< Constraint state hessians
        std::map<std::string, std::vector<Eigen::MatrixXd>>
            G_uu_; ///< Constraint control hessians
        std::map<std::string, std::vector<Eigen::MatrixXd>>
            G_ux_; ///< Constraint mixed hessians

        // Interior point variables
        std::map<std::string, std::vector<Eigen::VectorXd>> G_; ///< Constraint values
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_; ///< Dual variables
        std::map<std::string, std::vector<Eigen::VectorXd>> S_; ///< Slack variables

        // Interior point gains
        std::map<std::string, std::vector<Eigen::VectorXd>> k_y_; ///< Dual feedforward
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_y_; ///< Dual feedback
        std::map<std::string, std::vector<Eigen::VectorXd>> k_s_; ///< Slack feedforward
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_s_; ///< Slack feedback

        // Barrier method parameters
        double mu_;                       ///< Barrier parameter
        std::vector<FilterPoint> filter_; ///< Filter for line search

        // Pre-allocated workspace for performance optimization
        struct Workspace {
            // Backward pass workspace
            std::vector<Eigen::MatrixXd> A_matrices;
            std::vector<Eigen::MatrixXd> B_matrices;
            std::vector<Eigen::MatrixXd> Q_xx_matrices;
            std::vector<Eigen::MatrixXd> Q_ux_matrices;
            std::vector<Eigen::MatrixXd> Q_uu_matrices;
            std::vector<Eigen::VectorXd> Q_x_vectors;
            std::vector<Eigen::VectorXd> Q_u_vectors;

            // LDLT solver cache
            std::vector<Eigen::LDLT<Eigen::MatrixXd>> ldlt_solvers;
            std::vector<bool> ldlt_valid;

            // Constraint workspace
            Eigen::VectorXd y_combined;
            Eigen::VectorXd s_combined;
            Eigen::VectorXd g_combined;
            Eigen::MatrixXd Q_yu_combined;
            Eigen::MatrixXd Q_yx_combined;
            Eigen::MatrixXd YSinv;
            Eigen::MatrixXd bigRHS;

            // Forward pass workspace
            std::vector<Eigen::VectorXd> delta_x_vectors;

            bool initialized = false;
        } workspace_;

        // === Private helper methods ===
        void precomputeDynamicsDerivatives(CDDP &context);
        void precomputeConstraintGradients(CDDP &context);
        void evaluateTrajectory(CDDP &context);
        void evaluateTrajectoryWarmStart(CDDP &context);
        void initializeDualSlackVariablesWarmStart(CDDP &context);
        void initializeDualSlackVariables(CDDP &context);
        void resetBarrierFilter(CDDP &context);
        void resetFilter(CDDP &context);
        void updateBarrierParameters(CDDP &context, bool forward_pass_success);
        int getTotalDualDim(const CDDP &context) const;
        void initializeConstraintStorage(CDDP &context);
        double computeMaxConstraintViolation(const CDDP &context) const;
        double computeScaledDualInfeasibility(const CDDP &context) const;

        // Legacy print helper (used by printIteration override)
        void printIterationLegacy(int iter, double objective, double inf_pr,
                                  double inf_du, double inf_comp, double mu,
                                  double step_norm, double regularization,
                                  double alpha_du, double alpha_pr) const;
    };

} // namespace cddp

#endif // CDDP_IPDDP_SOLVER_HPP
