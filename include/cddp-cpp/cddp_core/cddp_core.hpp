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
#ifndef CDDP_CDDP_CORE_HPP
#define CDDP_CDDP_CORE_HPP

#include <iostream> // For std::cout, std::cerr
#include <string>   // For std::string
#include <memory>   // For std::unique_ptr
#include <map>      // For std::map`
#include <iomanip>  // For std::setw
#include <any>      // For std::any
#include <optional> // For std::optional
#include <Eigen/Dense>
#include <vector>
#include <regex>
#include <future>
#include <thread>

#include "cddp_core/dynamical_system.hpp"
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"
#include "cddp_core/barrier.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/options.hpp"

namespace cddp
{

    /**
     * @brief Enumeration of available solver types.
     */
    enum class SolverType {
        CLDDP,    ///< Control-Limited Differential Dynamic Programming
        ASDDP,    ///< Active Set Differential Dynamic Programming  
        LogDDP,   ///< Log-Barrier Differential Dynamic Programming
        IPDDP,    ///< Interior Point Differential Dynamic Programming
        MSIPDDP,  ///< Multi-Shooting Interior Point Differential Dynamic Programming
        ALTRO     ///< Augmented Lagrangian Trajectory Optimizer
    };

    /**
     * @brief Solution data from the CDDP solver, as a map of string keys to `std::any` values.
     *
     * Retrieve values using `std::any_cast<Type>(solution.at("key"))`.
     * Always check `solution.count("key")` before `.at()` to prevent `std::out_of_range`.
     * Handle `std::bad_any_cast` for type mismatches.
     * Optional keys are present only if computed by the specific solver.
     *
     * --- General Information ---
     * - "solver_name":                   std::string (Name of the solver used, e.g., "IPDDP")
     * - "status_message":                std::string (Termination status; possible values:)
     *                                      • "OptimalSolutionFound" - Converged to optimal solution
     *                                      • "AcceptableSolutionFound" - Reached cost tolerance limit
     *                                      • "MaxIterationsReached" - Reached maximum iteration limit
     *                                      • "MaxCpuTimeReached" - Exceeded maximum CPU time limit
     *                                      • "RegularizationLimitReached_Converged" - Reached regularization limit but solution acceptable
     *                                      • "RegularizationLimitReached_NotConverged" - Reached regularization limit, solution not acceptable
     * - "iterations_completed":          int (Number of iterations)
     * - "solve_time_ms":                 double (Total solver time in milliseconds)
     * - "final_objective":               double (Final objective cost J(x,u))
     * - "final_step_length":             double (Final line search step size; use "final_step_length_primal" / "_dual" if distinct)
     *
     * --- Primary Solution Trajectories ---
     * - "time_points":                   std::vector<double> (Time points t_0..t_N)
     * - "state_trajectory":              std::vector<Eigen::VectorXd> (States X_0..X_N)
     * - "control_trajectory":            std::vector<Eigen::VectorXd> (Controls U_0..U_{N-1})
     *
     * --- Iteration History (Optional) ---
     *   (Vectors indexed by iteration number)
     * - "history_objective":             std::vector<double> (Objective J)
     * - "history_merit_function":        std::vector<double> (Merit function value, e.g., for IPMs)
     * - "history_primal_infeasibility":  std::vector<double> (Primal constraint violation norm (inf_pr), including dynamics defects)
     * - "history_dual_infeasibility":    std::vector<double> (Lagrangian gradient norm (inf_du) or other dual infeasibility)
     * - "history_barrier_mu":            std::vector<double> (Barrier parameter (mu) for IPMs; user can log10 for "lg(mu)")
     * - "history_regularization":        std::map<std::string, std::vector<double>> (Regularization values; user can log10 for "lg(rg)")
     * - "history_step_length_primal":    std::vector<double> (Primal step length (alpha_pr))
     * - "history_step_length_dual":      std::vector<double> (Dual step length (alpha_du))
     * - "history_linesearch_iterations": std::vector<int> (Line search sub-iterations (ls))
     *
     * --- Final Metrics (at termination, Optional) ---
     * - "final_primal_infeasibility":    double (Total primal constraint violation norm, including dynamics defects)
     * - "final_dual_infeasibility":      double (Lagrangian gradient norm or other dual infeasibility metric)
     *
     * --- Controller Gains (Feedback Policy, Optional) ---
     * - "control_feedback_gains_k":   std::vector<Eigen::MatrixXd> (Feedback gains K_u)
     *
     * --- Solver-Specific Internal Metrics (at termination, Optional) ---
     * - "final_barrier_parameter_mu":    double (Barrier parameter mu for IPMs; smallness implies complementarity)
     * - "final_regularization":   double (Final regularization value)
     */
    using CDDPSolution = std::map<std::string, std::any>;

    struct ForwardPassResult
    {
        // Core trajectories always computed in a forward pass
        std::vector<Eigen::VectorXd> state_trajectory;
        std::vector<Eigen::VectorXd> control_trajectory;

        // Cost and merit function values
        double cost = 0.0;
        double merit_function = 0.0;

        // Line search step size that produced this result
        double alpha_pr = 1.0;
        double alpha_du = 1.0;

        // Status of this particular forward pass trial
        bool success = false;

        double constraint_violation = 0.0;

        // Optional: Only relevant for certain solver strategies during their forward pass
        std::optional<std::vector<Eigen::VectorXd>> dynamics_trajectory;
        std::optional<std::vector<Eigen::VectorXd>> costate_trajectory;
        std::optional<std::map<std::string, std::vector<Eigen::VectorXd>>> dual_trajectory;
        std::optional<std::map<std::string, std::vector<Eigen::VectorXd>>> slack_trajectory;
        std::optional<std::map<std::string, std::vector<Eigen::VectorXd>>> constraint_eval_trajectory;

        // Default constructor
        ForwardPassResult() = default;
    };

    /**
     * @brief Filter point for filter-based line search.
     * 
     * Each filter point represents a (merit_function, constraint_violation) pair.
     * The filter maintains a set of non-dominated points to guide the line search.
     */
    struct FilterPoint
    {
        double merit_function;        ///< Merit function value (objective + log-barrier terms)
        double constraint_violation;  ///< Constraint violation measure

        // Default constructor
        FilterPoint() : merit_function(0.0), constraint_violation(0.0) {}

        // Constructor with parameters
        FilterPoint(double mf, double cv) : merit_function(mf), constraint_violation(cv) {}

        /**
         * @brief Check if this point dominates another point.
         * @param other The other filter point to compare against.
         * @return True if this point dominates the other (better in both merit and violation).
         */
        bool dominates(const FilterPoint& other) const {
            return merit_function <= other.merit_function && constraint_violation <= other.constraint_violation;
        }
    };

    // Forward declaration
    class CDDP;

    /**
     * @brief Abstract base class for solver algorithm strategies.
     *
     * This interface defines the common operations that any solver algorithm
     * (CLDDP, IPDDP, MSIPDDP, etc.) must implement.
     */
    class ISolverAlgorithm
    {
    public:
        virtual ~ISolverAlgorithm() = default;

        /**
         * @brief Initialize the solver algorithm with the given CDDP context.
         * @param context Reference to the CDDP instance containing problem data and options.
         */
        virtual void initialize(CDDP &context) = 0;

        /**
         * @brief Execute the solver algorithm and return the solution.
         * @param context Reference to the CDDP instance containing problem data and options.
         * @return CDDPSolution containing the results.
         */
        virtual CDDPSolution solve(CDDP &context) = 0;

        /**
         * @brief Get the name of the solver algorithm.
         * @return String identifier for this solver type.
         */
        virtual std::string getSolverName() const = 0;
    };

    class CDDP
    {
    public:
        // Constructor
        CDDP(const Eigen::VectorXd &initial_state,
             const Eigen::VectorXd &reference_state,
             int horizon,
             double timestep,
             std::unique_ptr<DynamicalSystem> system = nullptr,
             std::unique_ptr<Objective> objective = nullptr,
             const CDDPOptions &options = CDDPOptions());

        // --- Accessor methods ---
        // Getters for problem definition
        const DynamicalSystem &getSystem() const { return *system_; }
        const Objective &getObjective() const { return *objective_; }
        const Eigen::VectorXd &getInitialState() const { return initial_state_; }
        const Eigen::VectorXd &getReferenceState() const { return reference_state_; } // Potentially multiple reference states
        const std::vector<Eigen::VectorXd> &getReferenceStates() const { return reference_states_; }
        int getHorizon() const { return horizon_; }
        double getTimestep() const { return timestep_; }
        int getStateDim() const;
        int getControlDim() const;
        int getTotalDualDim() const;
        const CDDPOptions &getOptions() const { return options_; }
        const std::map<std::string, std::unique_ptr<Constraint>> &getConstraintSet() const { return path_constraint_set_; }
        const std::map<std::string, std::unique_ptr<Constraint>> &getTerminalConstraintSet() const { return terminal_constraint_set_; }
        
        // Setters for problem definition
        void setDynamicalSystem(std::unique_ptr<DynamicalSystem> system);
        void setInitialState(const Eigen::VectorXd &initial_state);
        void setReferenceState(const Eigen::VectorXd &reference_state);
        void setReferenceStates(const std::vector<Eigen::VectorXd> &reference_states);
        void setHorizon(int horizon);
        void setTimestep(double timestep);
        void setOptions(const CDDPOptions &options);
        void setObjective(std::unique_ptr<Objective> objective);
        void setInitialTrajectory(const std::vector<Eigen::VectorXd> &X, const std::vector<Eigen::VectorXd> &U);
        void addPathConstraint(std::string constraint_name, std::unique_ptr<Constraint> constraint);
        void addTerminalConstraint(std::string constraint_name, std::unique_ptr<Constraint> constraint);
        bool removePathConstraint(const std::string &constraint_name);
        bool removeTerminalConstraint(const std::string &constraint_name);

        template <typename T>
        T *getConstraint(const std::string &name) const
        {
            auto it = path_constraint_set_.find(name);
            if (it == path_constraint_set_.end())
                return nullptr;
            T *cast_constraint = dynamic_cast<T *>(it->second.get());
            return cast_constraint;
        }

        template <typename T>
        T *getTerminalConstraint(const std::string &name) const
        {
            auto it = terminal_constraint_set_.find(name);
            if (it == terminal_constraint_set_.end())
                return nullptr;
            T *cast_constraint = dynamic_cast<T *>(it->second.get());
            return cast_constraint;
        }

        // --- Core Solver Invocation ---
        /**
         * @brief Solves the optimal control problem using the specified algorithm.
         * @param solver_type Enum identifying the solver algorithm to use.
         * @return CDDPSolution A map containing the solution details.
         */
        CDDPSolution solve(SolverType solver_type = SolverType::CLDDP);
        
        /**
         * @brief Solves the optimal control problem using the specified algorithm (string version for backward compatibility).
         * @param solver_type A string identifying the solver algorithm to use (e.g., "CLDDP", "ASDDP", "LOGDDP", "IPDDP", "MSIPDDP").
         * @return CDDPSolution A map containing the solution details.
         */
        CDDPSolution solve(const std::string& solver_type);

        // --- Public members for strategy access (or provide getters/setters) ---
        // These are the core iterative variables shared across solver strategies.
        std::vector<Eigen::VectorXd> X_; ///< State trajectory (nominal)
        std::vector<Eigen::VectorXd> U_; ///< Control trajectory (nominal)
        double cost_;                    ///< Current total cost
        double merit_function_;          ///< Merit function value
        double inf_pr_;                  ///< Current primal infeasibility (constraint violation norm)
        double inf_du_;                  ///< Current dual infeasibility (Lagrangian gradient norm)
        bool initialized_ = false;       ///< Overall CDDP problem initialization flag

        // Common Line Search parameters that might be managed by CDDP context or passed to strategies
        std::vector<double> alphas_; // Potential alpha values for line search, configured by options_.line_search
        double alpha_pr_;            // Accepted primal step size for the current iteration
        double alpha_du_;            // Accepted dual step size for the current iteration

        // Regularization management
        double regularization_; ///< Current regularization parameter

        // Getters for current iteration metrics
        double getCurrentCost() const { return cost_; }
        double getCurrentMeritFunction() const { return merit_function_; }
        double getCurrentPrimalInfeasibility() const { return inf_pr_; }
        double getCurrentDualInfeasibility() const { return inf_du_; }
        double getCurrentPrimalStepSize() const { return alpha_pr_; }
        double getCurrentDualStepSize() const { return alpha_du_; }
        double getCurrentRegularization() const { return regularization_; }

        /**
         * @brief Check if KKT conditions are satisfied within tolerance.
         * @return True if both inf_pr and inf_du are below tolerance, false otherwise.
         */
        bool isKKTToleranceSatisfied() const;

        /**
         * @brief Increase regularization parameters.
         */
        void increaseRegularization();

        /**
         * @brief Decrease regularization parameters.
         */
        void decreaseRegularization();

        /**
         * @brief Check if regularization limit has been reached.
         * @return True if limit reached, false otherwise.
         */
        bool isRegularizationLimitReached() const;

    private:
        // Problem Definition Data
        std::unique_ptr<DynamicalSystem> system_;
        std::unique_ptr<Objective> objective_;
        std::map<std::string, std::unique_ptr<Constraint>> path_constraint_set_;
        std::map<std::string, std::unique_ptr<Constraint>> terminal_constraint_set_;
        Eigen::VectorXd initial_state_;
        Eigen::VectorXd reference_state_;               // Single desired reference state (if applicable)
        std::vector<Eigen::VectorXd> reference_states_; // Desired reference state trajectory (if applicable)
        int horizon_;
        double timestep_;
        CDDPOptions options_;

        int total_dual_dim_ = 0;

        // Strategy pattern for different solver algorithms
        std::unique_ptr<ISolverAlgorithm> solver_;

        void initializeProblemIfNecessary();
        void printSolverInfo();
        void printOptions(const CDDPOptions &options);
    };

} // namespace cddp
#endif // CDDP_CDDP_CORE_HPP