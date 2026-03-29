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

#include "cddp_core/cddp_solver_base.hpp"
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

namespace cddp
{

    /**
     * @brief Multiple-Shooting Interior Point DDP (MSIPDDP) solver.
     *
     * Inherits from CDDPSolverBase and overrides virtual hooks
     * for primal-dual interior point method with multi-shooting.
     */
    class MSIPDDPSolver : public CDDPSolverBase
    {
    public:
        MSIPDDPSolver();

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
        bool handleForwardPassFailure(CDDP &context, std::string &termination_reason) override;
        void recordIterationHistory(const CDDP &context) override;
        void populateSolverSpecificSolution(CDDPSolution &solution,
                                            const CDDP &context) override;
        void printIteration(int iter, const CDDP &context) const override;
        void printSolutionSummary(const CDDPSolution &solution) const override;

    private:
        // Dynamics storage (multi-shooting)
        std::vector<Eigen::VectorXd> F_;   ///< Dynamics evaluations

        // Constraint derivatives
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_x_;
        std::map<std::string, std::vector<Eigen::MatrixXd>> G_u_;
        std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>> G_xx_;
        std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>> G_uu_;
        std::map<std::string, std::vector<std::vector<Eigen::MatrixXd>>> G_ux_;

        // Interior point variables
        std::map<std::string, std::vector<Eigen::VectorXd>> G_;
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> S_;

        // Interior point gains
        std::map<std::string, std::vector<Eigen::VectorXd>> k_y_;
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_y_;
        std::map<std::string, std::vector<Eigen::VectorXd>> k_s_;
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_s_;

        // MSIPDDP-specific costate variables and gains
        std::vector<Eigen::VectorXd> Lambda_;
        std::vector<Eigen::VectorXd> k_lambda_;
        std::vector<Eigen::MatrixXd> K_lambda_;

        // Multi-shooting specific parameters
        int ms_segment_length_;

        // Barrier method parameters
        double mu_;
        std::vector<FilterPoint> filter_;

        // Pre-allocated workspace
        struct Workspace {
            std::vector<Eigen::MatrixXd> A_matrices;
            std::vector<Eigen::MatrixXd> B_matrices;
            std::vector<Eigen::MatrixXd> Q_xx_matrices;
            std::vector<Eigen::MatrixXd> Q_ux_matrices;
            std::vector<Eigen::MatrixXd> Q_uu_matrices;
            std::vector<Eigen::VectorXd> Q_x_vectors;
            std::vector<Eigen::VectorXd> Q_u_vectors;

            std::vector<Eigen::LDLT<Eigen::MatrixXd>> ldlt_solvers;
            std::vector<bool> ldlt_valid;

            Eigen::VectorXd y_combined;
            Eigen::VectorXd s_combined;
            Eigen::VectorXd g_combined;
            Eigen::MatrixXd Q_yu_combined;
            Eigen::MatrixXd Q_yx_combined;
            Eigen::MatrixXd YSinv;
            Eigen::MatrixXd bigRHS;

            std::vector<Eigen::VectorXd> delta_x_vectors;
            std::vector<Eigen::VectorXd> d_vectors; ///< Defect vectors

            bool initialized = false;
        } workspace_;

        // === Private helper methods ===
        void precomputeDynamicsDerivatives(CDDP &context);
        void precomputeConstraintGradients(CDDP &context);
        void evaluateTrajectory(CDDP &context);
        void evaluateTrajectoryWarmStart(CDDP &context);
        void initializeDualSlackCostateVariablesWarmStart(CDDP &context);
        void initializeDualSlackCostateVariables(CDDP &context);
        void resetBarrierFilter(CDDP &context);
        void resetFilter(CDDP &context);
        bool acceptFilterEntry(double merit_function, double constraint_violation);
        bool isFilterAcceptable(double merit_function, double constraint_violation,
                               const SolverSpecificFilterOptions &options,
                               double expected_improvement) const;
        bool checkAndPerformFilterRestoration(CDDP &context);
        void updateBarrierParameters(CDDP &context, bool forward_pass_success);
        int getTotalDualDim(const CDDP &context) const;
        void initializeConstraintStorage(CDDP &context);
        double computeMaxConstraintViolation(const CDDP &context) const;
        double computeScaledDualInfeasibility(const CDDP &context) const;

        // Legacy print helper
        void printIterationLegacy(int iter, double objective, double inf_pr,
                                  double inf_du, double inf_comp, double mu,
                                  double step_norm, double regularization,
                                  double alpha_du, double alpha_pr) const;
    };

} // namespace cddp

#endif // CDDP_MSIPDDP_SOLVER_HPP
