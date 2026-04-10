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

#include <Eigen/Dense>
#include <future>
#include <iomanip>  // For std::setw
#include <iostream> // For std::cout, std::cerr
#include <map>      // For std::map`
#include <memory>   // For std::unique_ptr
#include <optional> // For std::optional
#include <regex>
#include <string> // For std::string
#include <thread>
#include <vector>
#include <functional>

#include "cddp_core/barrier.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/constraint.hpp"
#include "cddp_core/dynamical_system.hpp"
#include "cddp_core/objective.hpp"
#include "cddp_core/options.hpp"

namespace cddp {

/**
 * @brief Enumeration of available solver types.
 */
enum class SolverType {
  CLDDP,   ///< Control-Limited Differential Dynamic Programming
  LogDDP,  ///< Log-Barrier Differential Dynamic Programming
  IPDDP,   ///< Interior Point Differential Dynamic Programming
  MSIPDDP  ///< Multi-Shooting Interior Point Differential Dynamic Programming
};

/**
 * @brief Solution data from the CDDP solver.
 */
struct CDDPSolution {
  // --- General Information ---
  std::string solver_name;
  std::string status_message = "Running";
  int iterations_completed = 0;
  double solve_time_ms = 0.0;
  double final_objective = 0.0;
  double final_step_length = 0.0;
  double final_regularization = 0.0;

  // --- Primary Solution ---
  std::vector<double> time_points;
  std::vector<Eigen::VectorXd> state_trajectory;
  std::vector<Eigen::VectorXd> control_trajectory;
  std::vector<Eigen::MatrixXd> feedback_gains;

  // --- Final Metrics (IP solvers) ---
  double final_primal_infeasibility = 0.0;
  double final_dual_infeasibility = 0.0;
  double final_complementary_infeasibility = 0.0;
  double final_barrier_mu = 0.0;

  // --- Iteration History (populated if return_iteration_info) ---
  struct History {
    std::vector<double> objective;
    std::vector<double> merit_function;
    std::vector<double> step_length_primal;
    std::vector<double> step_length_dual;
    std::vector<double> dual_infeasibility;
    std::vector<double> primal_infeasibility;
    std::vector<double> complementary_infeasibility;
    std::vector<double> barrier_mu;
    std::vector<double> regularization;

    void reserve(size_t n) {
      objective.reserve(n); merit_function.reserve(n);
      step_length_primal.reserve(n); step_length_dual.reserve(n);
      dual_infeasibility.reserve(n); primal_infeasibility.reserve(n);
      complementary_infeasibility.reserve(n); barrier_mu.reserve(n);
      regularization.reserve(n);
    }
    void clear() {
      objective.clear(); merit_function.clear();
      step_length_primal.clear(); step_length_dual.clear();
      dual_infeasibility.clear(); primal_infeasibility.clear();
      complementary_infeasibility.clear(); barrier_mu.clear();
      regularization.clear();
    }
  } history;
};

struct ForwardPassResult {
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
  double theta = 0.0;      // Filter violation metric
  double inf_pr = 0.0;     // Primal infeasibility
  double inf_comp = 0.0;   // Complementary infeasibility

  // Optional: Only relevant for certain solver strategies during their forward
  // pass
  std::optional<std::vector<Eigen::VectorXd>> dynamics_trajectory;
  std::optional<std::vector<Eigen::VectorXd>> costate_trajectory;
  std::optional<std::map<std::string, std::vector<Eigen::VectorXd>>>
      dual_trajectory;
  std::optional<std::map<std::string, std::vector<Eigen::VectorXd>>>
      slack_trajectory;
  std::optional<std::map<std::string, std::vector<Eigen::VectorXd>>>
      constraint_eval_trajectory;
  std::optional<std::map<std::string, Eigen::VectorXd>>
      terminal_constraint_dual;
  std::optional<std::map<std::string, Eigen::VectorXd>>
      terminal_slack;
  std::optional<std::map<std::string, Eigen::VectorXd>>
      terminal_constraint_value;
  
  // Default constructor
  ForwardPassResult() = default;
};

/**
 * @brief Filter point for filter-based line search.
 *
 * Each filter point represents a (merit_function, constraint_violation) pair.
 * The filter maintains a set of non-dominated points to guide the line search.
 */
struct FilterPoint {
  double
      merit_function; ///< Merit function value (objective + log-barrier terms)
  double constraint_violation; ///< Constraint violation measure

  // Default constructor
  FilterPoint() : merit_function(0.0), constraint_violation(0.0) {}

  // Constructor with parameters
  FilterPoint(double mf, double cv)
      : merit_function(mf), constraint_violation(cv) {}

  /**
   * @brief Check if this point dominates another point.
   * @param other The other filter point to compare against.
   * @return True if this point dominates the other (better in both merit and
   * violation).
   */
  bool dominates(const FilterPoint &other) const {
    return merit_function <= other.merit_function &&
           constraint_violation <= other.constraint_violation;
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
class ISolverAlgorithm {
public:
  virtual ~ISolverAlgorithm() = default;

  /**
   * @brief Initialize the solver algorithm with the given CDDP context.
   * @param context Reference to the CDDP instance containing problem data and
   * options.
   */
  virtual void initialize(CDDP &context) = 0;

  /**
   * @brief Execute the solver algorithm and return the solution.
   * @param context Reference to the CDDP instance containing problem data and
   * options.
   * @return CDDPSolution containing the results.
   */
  virtual CDDPSolution solve(CDDP &context) = 0;

  /**
   * @brief Get the name of the solver algorithm.
   * @return String identifier for this solver type.
   */
  virtual std::string getSolverName() const = 0;
};

class CDDP {
public:
  // Constructor
  CDDP(const Eigen::VectorXd &initial_state,
       const Eigen::VectorXd &reference_state, int horizon, double timestep,
       std::unique_ptr<DynamicalSystem> system = nullptr,
       std::unique_ptr<Objective> objective = nullptr,
       const CDDPOptions &options = CDDPOptions());

  // --- Accessor methods ---
  // Getters for problem definition
  const DynamicalSystem &getSystem() const { return *system_; }
  const Objective &getObjective() const { return *objective_; }
  const Eigen::VectorXd &getInitialState() const { return initial_state_; }
  const Eigen::VectorXd &getReferenceState() const {
    return reference_state_;
  } // Potentially multiple reference states
  const std::vector<Eigen::VectorXd> &getReferenceStates() const {
    return reference_states_;
  }
  int getHorizon() const { return horizon_; }
  double getTimestep() const { return timestep_; }
  int getStateDim() const;
  int getControlDim() const;
  int getTotalDualDim() const;
  const CDDPOptions &getOptions() const { return options_; }
  const std::map<std::string, std::unique_ptr<Constraint>> &
  getConstraintSet() const {
    return path_constraint_set_;
  }
  const std::map<std::string, std::unique_ptr<Constraint>> &
  getTerminalConstraintSet() const {
    return terminal_constraint_set_;
  }

  // Setters for problem definition
  void setDynamicalSystem(std::unique_ptr<DynamicalSystem> system);
  void setInitialState(const Eigen::VectorXd &initial_state);
  void setReferenceState(const Eigen::VectorXd &reference_state);
  void setReferenceStates(const std::vector<Eigen::VectorXd> &reference_states);
  void setHorizon(int horizon);
  void setTimestep(double timestep);
  void setOptions(const CDDPOptions &options);
  void setObjective(std::unique_ptr<Objective> objective);
  void setInitialTrajectory(const std::vector<Eigen::VectorXd> &X,
                            const std::vector<Eigen::VectorXd> &U);
  void addPathConstraint(std::string constraint_name,
                         std::unique_ptr<Constraint> constraint);
  void addTerminalConstraint(std::string constraint_name,
                             std::unique_ptr<Constraint> constraint);
  bool removePathConstraint(const std::string &constraint_name);
  bool removeTerminalConstraint(const std::string &constraint_name);

  template <typename T> T *getConstraint(const std::string &name) const {
    auto it = path_constraint_set_.find(name);
    if (it == path_constraint_set_.end())
      return nullptr;
    T *cast_constraint = dynamic_cast<T *>(it->second.get());
    return cast_constraint;
  }

  template <typename T>
  T *getTerminalConstraint(const std::string &name) const {
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
   * @brief Solves the optimal control problem using the specified algorithm
   * (string version for backward compatibility).
   * @param solver_type A string identifying the solver algorithm to use (e.g.,
   * "CLDDP", "LOGDDP", "IPDDP", "MSIPDDP").
   * @return CDDPSolution A map containing the solution details.
   */
  CDDPSolution solve(const std::string &solver_type);

  // --- External Solver Registration ---
  /**
   * @brief Register an external solver factory function
   * @param solver_name The name of the solver (e.g., "CCIPDDP")
   * @param factory Function that creates the solver instance
   */
  static void registerSolver(const std::string& solver_name, 
                           std::function<std::unique_ptr<ISolverAlgorithm>()> factory);

  /**
   * @brief Check if a solver is registered
   * @param solver_name The name of the solver
   * @return True if solver is registered
   */
  static bool isSolverRegistered(const std::string& solver_name);

  /**
   * @brief Get list of all registered solver names
   * @return Vector of solver names
   */
  static std::vector<std::string> getRegisteredSolvers();

  // --- Public members for strategy access (or provide getters/setters) ---
  // These are the core iterative variables shared across solver strategies.
  std::vector<Eigen::VectorXd> X_; ///< State trajectory (nominal)
  std::vector<Eigen::VectorXd> U_; ///< Control trajectory (nominal)
  double cost_;                    ///< Current total cost
  double merit_function_;          ///< Merit function value
  double inf_pr_; ///< Current primal infeasibility (constraint violation norm)
  double inf_du_; ///< Current dual infeasibility (Lagrangian gradient norm)
  double inf_comp_; ///< Current complementary 
  double step_norm_; ///< Current step norm ||d|| (primal step magnitude)
  bool initialized_ = false; ///< Overall CDDP problem initialization flag

  // Common Line Search parameters that might be managed by CDDP context or
  // passed to strategies
  std::vector<double> alphas_; // Potential alpha values for line search,
                               // configured by options_.line_search
  double alpha_pr_; // Accepted primal step size for the current iteration
  double alpha_du_; // Accepted dual step size for the current iteration

  // Regularization management
  double regularization_; ///< Current regularization parameter
  double terminal_regularization_; ///< Current terminal regularization parameter

  // Getters for current iteration metrics
  double getCurrentCost() const { return cost_; }
  double getCurrentMeritFunction() const { return merit_function_; }
  double getCurrentPrimalInfeasibility() const { return inf_pr_; }
  double getCurrentDualInfeasibility() const { return inf_du_; }
  double getCurrentStepNorm() const { return step_norm_; }
  double getCurrentPrimalStepSize() const { return alpha_pr_; }
  double getCurrentDualStepSize() const { return alpha_du_; }
  double getCurrentRegularization() const { return regularization_; }

  /**
   * @brief Check if KKT conditions are satisfied within tolerance.
   * @return True if both inf_pr and inf_du are below tolerance, false
   * otherwise.
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

  /**
   * @brief Increase terminal regularization parameter.
   */
  void increaseTerminalRegularization();

  /**
   * @brief Decrease terminal regularization parameter.
   */
  void decreaseTerminalRegularization();

  /**
   * @brief Check if terminal regularization limit has been reached.
   * @return True if limit reached, false otherwise.
   */
  bool isTerminalRegularizationLimitReached() const;

  /**
   * @brief Get current terminal regularization value.
   * @return Current terminal regularization parameter.
   */
  double getCurrentTerminalRegularization() const { return terminal_regularization_; }

  /**
   * @brief Print solver information banner.
   */
  void printSolverInfo();

  /**
   * @brief Print solver options.
   * @param options The options to print
   */
  void printOptions(const CDDPOptions &options);

protected:
  /**
   * @brief Create solver instance (virtual for extensibility)
   * @param solver_type String identifying the solver type
   * @return Unique pointer to solver instance, or nullptr if not found
   */
  virtual std::unique_ptr<ISolverAlgorithm> createSolver(const std::string& solver_type);

private:
  // Problem Definition Data
  std::unique_ptr<DynamicalSystem> system_;
  std::unique_ptr<Objective> objective_;
  std::map<std::string, std::unique_ptr<Constraint>> path_constraint_set_;
  std::map<std::string, std::unique_ptr<Constraint>> terminal_constraint_set_;
  Eigen::VectorXd initial_state_;
  Eigen::VectorXd
      reference_state_; // Single desired reference state (if applicable)
  std::vector<Eigen::VectorXd>
      reference_states_; // Desired reference state trajectory (if applicable)
  int horizon_;
  double timestep_;
  CDDPOptions options_;

  int total_dual_dim_ = 0;

  // Strategy pattern for different solver algorithms
  std::unique_ptr<ISolverAlgorithm> solver_;

  // Static registry for external solvers
  static std::map<std::string, std::function<std::unique_ptr<ISolverAlgorithm>()>> external_solver_registry_;

  void initializeProblemIfNecessary();
};

} // namespace cddp
#endif // CDDP_CDDP_CORE_HPP
