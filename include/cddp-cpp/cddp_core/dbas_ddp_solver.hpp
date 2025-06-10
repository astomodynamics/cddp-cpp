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

#ifndef CDDP_DBAS_DDP_SOLVER_HPP
#define CDDP_DBAS_DDP_SOLVER_HPP

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/barrier.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <map>

namespace cddp
{

    /**
     * @brief Discrete Barrier State DDP (DBAS-DDP) solver implementation.
     *
     * This class implements the ISolverAlgorithm interface to provide
     * a discrete barrier state based DDP solver for handling inequality constraints
     * with explicit barrier state augmentation and discrete barrier transitions.
     * 
     * Note: DBAS-DDP is incompatible with multi-shooting optimization because:
     * - Barrier states require continuous evolution throughout the entire trajectory
     * - State augmentation needs to maintain temporal relationships across the full horizon
     * - Barrier states carry memory of past constraint violations that would be lost in segmented approaches
     * 
     * Note: DBAS-DDP requires feasible initial guesses because barrier states are part of the
     * augmented state space and need proper initialization. Therefore, conventional line search
     * (focusing on cost reduction) is used instead of filter-based line search.
     */
    class DbasDdpSolver : public ISolverAlgorithm
    {
    public:
        /**
         * @brief Default constructor.
         */
        DbasDdpSolver();

        /**
         * @brief Initialize the solver with the given CDDP context.
         * @param context Reference to the CDDP instance containing problem data and options.
         */
        void initialize(CDDP &context) override;

        /**
         * @brief Execute the DBAS-DDP algorithm and return the solution.
         * @param context Reference to the CDDP instance containing problem data and options.
         * @return CDDPSolution containing the results.
         */
        CDDPSolution solve(CDDP &context) override;

        /**
         * @brief Get the name of the solver algorithm.
         * @return String identifier "DBAS-DDP".
         */
        std::string getSolverName() const override;

    private:
        // Augmented state dimensions including barrier states
        int augmented_state_dim_;        ///< Total augmented state dimension (original + barrier states)
        int original_state_dim_;         ///< Original state dimension
        int barrier_state_dim_;          ///< Number of barrier state variables
        
        // Dynamics storage for augmented system
        std::vector<Eigen::VectorXd> F_aug_;             ///< Augmented dynamics evaluations
        std::vector<Eigen::MatrixXd> F_x_aug_;           ///< Augmented state jacobians
        std::vector<Eigen::MatrixXd> F_u_aug_;           ///< Augmented control jacobians  
        std::vector<std::vector<Eigen::MatrixXd>> F_xx_aug_; ///< Augmented state hessians
        std::vector<std::vector<Eigen::MatrixXd>> F_uu_aug_; ///< Augmented control hessians
        std::vector<std::vector<Eigen::MatrixXd>> F_ux_aug_; ///< Augmented mixed hessians

        // Augmented trajectories
        std::vector<Eigen::VectorXd> X_aug_;             ///< Augmented state trajectory
        std::vector<Eigen::VectorXd> barrier_states_;   ///< Barrier state trajectory
        
        // Control law parameters for augmented system
        std::vector<Eigen::VectorXd> k_u_;               ///< Feedforward control gains
        std::vector<Eigen::MatrixXd> K_u_;               ///< Feedback control gains  
        Eigen::Vector2d dV_;                             ///< Expected value function change

        // Discrete barrier state management
        std::map<std::string, std::vector<Eigen::VectorXd>> barrier_state_values_; ///< Barrier state values for each constraint
        std::map<std::string, std::vector<Eigen::VectorXd>> G_;                     ///< Constraint values
        std::map<std::string, std::unique_ptr<DiscreteBarrierState>> discrete_barrier_managers_; ///< Barrier state managers for each constraint
        std::unique_ptr<RelaxedLogBarrier> relaxed_log_barrier_;                    ///< Log barrier object
        double mu_;                                                                 ///< Barrier parameter
        double relaxation_delta_;                                                   ///< Relaxation parameter
        
        // Discrete barrier state parameters (loaded from options)
        double barrier_state_weight_;                    ///< Weight for barrier state dynamics penalty
        double barrier_state_init_value_;               ///< Initial value for barrier states
        double barrier_state_decay_rate_;               ///< Decay rate for barrier state updates
        bool use_adaptive_barrier_weights_;             ///< Flag for adaptive barrier state weighting
        double max_barrier_state_norm_;                  ///< Maximum allowed barrier state norm
        double barrier_state_regularization_;           ///< Regularization for barrier state dynamics

        // Conventional line search (no filter needed since we require feasible initialization)
        double constraint_violation_;                    ///< Current constraint violation measure
        double previous_barrier_state_norm_;            ///< Previous iteration barrier state norm for convergence checking

        // === Initialization and Validation Methods ===
        
        /**
         * @brief Validate DBAS-DDP options and throw if invalid.
         * @param options DBAS-DDP algorithm options to validate.
         */
        void validateOptions(const DbasDdpAlgorithmOptions &options) const;

        /**
         * @brief Initialize augmented state space including barrier states.
         * @param context Reference to the CDDP context.
         */
        void initializeAugmentedStateSpace(CDDP &context);

        // === Barrier State Management Methods ===
        
        /**
         * @brief Update barrier states based on current constraint violations.
         * @param context Reference to the CDDP context.
         * @param time_step Current time step.
         */
        void updateBarrierStates(CDDP &context, int time_step);

        // === Augmented Dynamics Methods ===
        
        /**
         * @brief Construct augmented dynamics including barrier state dynamics.
         * @param context Reference to the CDDP context.
         * @param time_step Current time step.
         * @param x_orig Original state.
         * @param u Control input.
         * @param barrier_state Current barrier state.
         * @return Augmented dynamics evaluation.
         */
        Eigen::VectorXd evaluateAugmentedDynamics(CDDP &context, int time_step, 
                                                   const Eigen::VectorXd &x_orig, 
                                                   const Eigen::VectorXd &u,
                                                   const Eigen::VectorXd &barrier_state);

        /**
         * @brief Compute augmented system jacobians.
         * @param context Reference to the CDDP context.
         * @param time_step Current time step.
         * @param x_orig Original state.
         * @param u Control input.
         * @param barrier_state Current barrier state.
         * @return Tuple of (F_x_aug, F_u_aug).
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computeAugmentedJacobians(
            CDDP &context, int time_step,
            const Eigen::VectorXd &x_orig, 
            const Eigen::VectorXd &u,
            const Eigen::VectorXd &barrier_state);

        /**
         * @brief Pre-compute augmented dynamics jacobians and hessians for all time steps.
         * @param context Reference to the CDDP context.
         */
        void precomputeAugmentedDynamicsDerivatives(CDDP &context);

        // === Cost and Trajectory Evaluation Methods ===
        
        /**
         * @brief Evaluate augmented trajectory including barrier states.
         * @param context Reference to the CDDP context.
         */
        void evaluateAugmentedTrajectory(CDDP &context);

        /**
         * @brief Compute cost including barrier state penalties.
         * @param context Reference to the CDDP context.
         */
        void computeCost(CDDP &context);

        // === Core DDP Algorithm Methods ===
        
        /**
         * @brief Perform backward pass (Riccati recursion) for augmented system.
         * @param context Reference to the CDDP context.
         * @return True if backward pass succeeds, false otherwise.
         */
        bool backwardPassAugmented(CDDP &context);

        /**
         * @brief Perform forward pass with conventional line search for augmented system.
         * @param context Reference to the CDDP context.
         * @return Best forward pass result.
         */
        ForwardPassResult performAugmentedForwardPass(CDDP &context);

        /**
         * @brief Perform single forward pass with given step size for augmented system.
         * @param context Reference to the CDDP context.
         * @param alpha Step size for the forward pass.
         * @return Forward pass result.
         */
        ForwardPassResult forwardPassAugmented(CDDP &context, double alpha);

        // === Parameter Update Methods ===
        
        /**
         * @brief Update barrier parameters and barrier state weights.
         * @param context Reference to the CDDP context.
         * @param forward_pass_success Whether the forward pass was successful.
         * @param termination_metric Current termination metric.
         */
        void updateBarrierParameters(CDDP &context, bool forward_pass_success, double termination_metric);

        // === State Space Conversion Methods ===
        
        /**
         * @brief Extract original state trajectory from augmented trajectory.
         * @param X_aug Augmented state trajectory.
         * @return Original state trajectory.
         */
        std::vector<Eigen::VectorXd> extractOriginalTrajectory(const std::vector<Eigen::VectorXd> &X_aug);

        /**
         * @brief Extract barrier state trajectory from augmented trajectory.
         * @param X_aug Augmented state trajectory.
         * @return Barrier state trajectory.
         */
        std::vector<Eigen::VectorXd> extractBarrierStateTrajectory(const std::vector<Eigen::VectorXd> &X_aug);

        /**
         * @brief Combine original and barrier states into augmented state.
         * @param x_orig Original state.
         * @param barrier_state Barrier state.
         * @return Augmented state vector.
         */
        Eigen::VectorXd combineStates(const Eigen::VectorXd &x_orig, const Eigen::VectorXd &barrier_state);

        // === Convergence and Health Checking Methods ===
        
        /**
         * @brief Check if barrier states have converged.
         * @param context Reference to the CDDP context.
         * @return True if barrier states have converged within tolerance.
         */
        bool checkBarrierStateConvergence(CDDP &context) const;

        /**
         * @brief Compute total barrier state norm across all constraints.
         * @param context Reference to the CDDP context.
         * @return Total norm of all barrier states.
         */
        double computeBarrierStateNorm(CDDP &context) const;

        /**
         * @brief Check if barrier states are numerically healthy.
         * @param context Reference to the CDDP context.
         * @return True if barrier states are within acceptable bounds.
         */
        bool checkBarrierStateHealth(CDDP &context) const;

        // === Utility and Display Methods ===

        /**
         * @brief Print iteration information.
         */
        void printIteration(int iter, double cost, double merit_function, double inf_du,
                            double regularization, double alpha, double mu, double constraint_violation,
                            double barrier_state_norm) const;

        /**
         * @brief Print solution summary.
         * @param solution The solution to print.
         */
        void printSolutionSummary(const CDDPSolution &solution) const;
    };

} // namespace cddp

#endif // CDDP_DBAS_DDP_SOLVER_HPP