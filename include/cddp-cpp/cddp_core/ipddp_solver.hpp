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
    class IPDDPSolverTestAccess;

    /**
     * @brief Interior Point Differential Dynamic Programming (IPDDP) solver.
     *
     * Inherits from CDDPSolverBase and overrides virtual hooks
     * for primal-dual interior point method with logarithmic barrier.
     * Supports path constraints, terminal equality constraints, and
     * terminal inequality constraints via a single-shooting formulation
     * with explicit costate variables.
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
        bool checkEarlyConvergence(CDDP &context, int iter,
                                   std::string &reason) override;
        ForwardPassResult forwardPass(CDDP &context, double alpha) override;
        void applyForwardPassResult(CDDP &context, const ForwardPassResult &result) override;
        bool checkConvergence(CDDP &context, double dJ, double dL, int iter,
                              std::string &reason) override;
        void postIterationUpdate(CDDP &context, bool forward_pass_success) override;
        bool handleForwardPassFailure(CDDP &context,
                                      std::string &termination_reason) override;
        void recordIterationHistory(const CDDP &context) override;
        void populateSolverSpecificSolution(CDDPSolution &solution,
                                            const CDDP &context) override;
        void printIteration(int iter, const CDDP &context) const override;
        void printSolutionSummary(const CDDPSolution &solution) const override;

    private:
        friend class IPDDPSolverTestAccess;

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
        std::map<std::string, std::vector<Eigen::VectorXd>> G_raw_; ///< Raw constraint values
        std::map<std::string, std::vector<Eigen::VectorXd>> G_; ///< Constraint values
        std::map<std::string, std::vector<Eigen::VectorXd>> Y_; ///< Dual variables
        std::map<std::string, std::vector<Eigen::VectorXd>> S_; ///< Slack variables

        // Interior point gains
        std::map<std::string, std::vector<Eigen::VectorXd>> k_y_; ///< Dual feedforward
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_y_; ///< Dual feedback
        std::map<std::string, std::vector<Eigen::VectorXd>> k_s_; ///< Slack feedforward
        std::map<std::string, std::vector<Eigen::MatrixXd>> K_s_; ///< Slack feedback

        // Single-shooting costate variables and gains
        std::vector<Eigen::VectorXd>
            Lambda_; ///< Costate variables (Lagrange multipliers for dynamics)
        std::vector<Eigen::VectorXd>
            k_lambda_; ///< Feedforward gains for costate variables
        std::vector<Eigen::MatrixXd>
            K_lambda_; ///< Feedback gains for costate variables

        // Terminal equality constraint variables
        Eigen::VectorXd Lambda_T_eq_;  ///< Terminal equality multipliers
        Eigen::VectorXd dLambda_T_eq_; ///< Terminal equality multiplier direction

        // Terminal inequality constraint variables
        std::map<std::string, Eigen::VectorXd> G_T_; ///< Terminal inequality values
        std::map<std::string, Eigen::VectorXd> Y_T_; ///< Terminal inequality duals
        std::map<std::string, Eigen::VectorXd> S_T_; ///< Terminal inequality slacks
        std::map<std::string, Eigen::VectorXd> dY_T_; ///< Terminal inequality dual directions
        std::map<std::string, Eigen::VectorXd> dS_T_; ///< Terminal inequality slack directions

        // Search directions
        std::vector<Eigen::VectorXd> dX_; ///< State search directions
        std::vector<Eigen::VectorXd> dU_; ///< Control search directions
        std::map<std::string, std::vector<Eigen::VectorXd>>
            dY_; ///< Dual search directions
        std::map<std::string, std::vector<Eigen::VectorXd>>
            dS_; ///< Slack search directions

        // Barrier method parameters
        double mu_;                       ///< Barrier parameter
        std::vector<FilterPoint> filter_; ///< Filter for line search
        double phi_ = 0.0;               ///< Current filter merit value
        double theta_ = 0.0;             ///< Current filter violation metric
        double filter_theta_ = 0.0;      ///< Current unfloored filter violation metric

        // === Private helper methods ===
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

        // Filter and merit function methods
        bool acceptFilterEntry(double merit_function, double constraint_violation);
        bool isFilterAcceptable(double merit_function,
                                double constraint_violation) const;

        double computeTheta(
            const CDDPOptions &options,
            const std::map<std::string, std::vector<Eigen::VectorXd>> &constraints,
            const std::map<std::string, std::vector<Eigen::VectorXd>> &slacks,
            const std::map<std::string, Eigen::VectorXd> *terminal_constraints = nullptr,
            const std::map<std::string, Eigen::VectorXd> *terminal_slacks = nullptr,
            const Eigen::VectorXd *terminal_equality_residual = nullptr) const;

        double computeBarrierMerit(
            const CDDP &context,
            const std::map<std::string, std::vector<Eigen::VectorXd>> &slacks,
            double cost,
            const std::map<std::string, Eigen::VectorXd> *terminal_slacks = nullptr,
            const Eigen::VectorXd *terminal_equality_multipliers = nullptr,
            const Eigen::VectorXd *terminal_equality_residual = nullptr) const;

        std::pair<double, double> computePrimalAndComplementarity(
            const CDDP &context,
            const std::map<std::string, std::vector<Eigen::VectorXd>> &constraints,
            const std::map<std::string, std::vector<Eigen::VectorXd>> &slacks,
            const std::map<std::string, std::vector<Eigen::VectorXd>> &duals,
            double mu,
            const std::map<std::string, Eigen::VectorXd> *terminal_constraints = nullptr,
            const std::map<std::string, Eigen::VectorXd> *terminal_slacks = nullptr,
            const std::map<std::string, Eigen::VectorXd> *terminal_duals = nullptr,
            const Eigen::VectorXd *terminal_equality_residual = nullptr) const;

        std::pair<double, double> computeMaxStepSizes(const CDDP &context) const;

        // Legacy print helper (used by printIteration override)
        void printIterationLegacy(int iter, double objective, double inf_pr,
                                  double inf_du, double inf_comp, double mu,
                                  double step_norm, double regularization,
                                  double alpha_du, double alpha_pr) const;
    };

} // namespace cddp

#endif // CDDP_IPDDP_SOLVER_HPP
