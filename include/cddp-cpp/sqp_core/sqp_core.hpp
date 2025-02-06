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
#ifndef CDDP_SCP_CORE_HPP
#define CDDP_SCP_CORE_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <casadi/casadi.hpp>

#include "cddp_core/dynamical_system.hpp"
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"

namespace cddp {

/**
 * @brief Configuration options for SCP solver using IPOPT.
 */
struct SCPOptions {
    int max_iterations = 100;            // Maximum number of iterations
    int min_iterations = 1;              // Minimum number of iterations
    double ftol = 1e-6;                  // Function value tolerance
    double xtol = 1e-6;                  // Step size tolerance
    double gtol = 1e-6;                  // Gradient tolerance
    double merit_penalty = 100.0;        // Penalty for constraint violation in merit function
    bool verbose = false;                // Verbosity flag

    // Trust region parameters
    double trust_region_radius = 100.0;        // Initial trust region radius
    double trust_region_radius_max = 1e6;        // Maximum trust region radius
    double trust_region_increase_factor = 2.0;   // Increase factor when step is good
    double trust_region_decrease_factor = 0.5;   // Decrease factor when step is rejected

    // IPOPT specific options
    int ipopt_max_iter = 1000;
    int ipopt_print_level = 5;
    double ipopt_tol = 1e-6;
};

/**
 * @brief Results from SCP optimization.
 */
struct SCPResult {
    bool success;                          // Whether optimization succeeded
    int iterations;                        // Number of iterations taken
    double objective_value;                // Final objective value
    double constraint_violation;           // Final constraint violation
    std::vector<Eigen::VectorXd> X;        // Optimal state trajectory (size: horizon+1)
    std::vector<Eigen::VectorXd> U;        // Optimal control trajectory (size: horizon)
    std::vector<double> obj_history;       // History of objective values
    std::vector<double> viol_history;      // History of constraint violations
    double solve_time;                     // Solution time in seconds
};

/**
 * @brief Sequential Convex Programming solver using IPOPT (via CasADi).
 *
 * The solver’s API mimics that of your previous SQP solver. In each SCP iteration the
 * dynamics are linearized about the current trajectory and a convex subproblem is solved.
 */
class SCPSolver {
public:
    /**
     * @brief Constructor.
     * @param initial_state Initial state of the system.
     * @param reference_state Desired (goal) state.
     * @param horizon Time horizon (number of control intervals; the state trajectory will have horizon+1 points).
     * @param timestep Time step.
     */
    SCPSolver(const Eigen::VectorXd& initial_state,
                   const Eigen::VectorXd& reference_state,
                   int horizon,
                   double timestep);

    // Setter methods
    void setDynamicalSystem(std::unique_ptr<DynamicalSystem> system) { system_ = std::move(system); }
    void setInitialState(const Eigen::VectorXd& initial_state) { initial_state_ = initial_state; }
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }
    void setHorizon(int horizon) { horizon_ = horizon; }
    void setTimestep(double timestep) { timestep_ = timestep; }
    void setOptions(const SCPOptions& options) { options_ = options; }
    void setObjective(std::unique_ptr<Objective> objective) { objective_ = std::move(objective); }
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X,
                              const std::vector<Eigen::VectorXd>& U);

    /**
     * @brief Solve the optimization problem.
     * @return SCPResult with the solution details.
     */
    SCPResult solve();

    /**
     * @brief Add a constraint to the problem.
     * @param constraint_name Constraint name.
     * @param constraint Constraint object.
     */
    void addConstraint(const std::string& constraint_name, std::unique_ptr<Constraint> constraint) {
        constraint_set_[constraint_name] = std::move(constraint);
    }

    /**
     * @brief Get a specific constraint by name.
     * @tparam T Type of constraint.
     * @param name Constraint name.
     * @return Pointer to the constraint (or nullptr if not found).
     */
    template <typename T>
    T* getConstraint(const std::string& name) const {
        auto it = constraint_set_.find(name);
        if (it == constraint_set_.end()) {
            return nullptr;
        }
        return dynamic_cast<T*>(it->second.get());
    }

    // Getters
    // Returns a pointer to the dynamical system.
    const DynamicalSystem* getDynamicalSystem() const { return system_.get(); }

    // Returns a pointer to the objective.
    const Objective* getObjective() const { return objective_.get(); }

private:
    SCPOptions options_;

    Eigen::VectorXd initial_state_;
    Eigen::VectorXd reference_state_;
    int horizon_;
    double timestep_;

    std::unique_ptr<DynamicalSystem> system_;
    std::unique_ptr<Objective> objective_;
    std::map<std::string, std::unique_ptr<Constraint>> constraint_set_;

    // Current trajectory estimates.
    // X: state trajectory (size: horizon_+1), U: control trajectory (size: horizon_)
    std::vector<Eigen::VectorXd> X_;
    std::vector<Eigen::VectorXd> U_;

    // Helper routines.
    void initializeSCP();
    void computeLinearizedDynamics(const std::vector<Eigen::VectorXd>& X,
                                   const std::vector<Eigen::VectorXd>& U,
                                   std::vector<Eigen::MatrixXd>& A,
                                   std::vector<Eigen::MatrixXd>& B) const;
                                   
    bool satisfies_trust_region_constraints(const std::vector<Eigen::VectorXd>& X,
                                              const std::vector<Eigen::VectorXd>& X_prev,
                                              double Delta) {
        if (X.size() != X_prev.size()) {
            throw std::runtime_error("Trajectory size mismatch in trust-region check.");
        }
        for (size_t t = 0; t < X.size(); ++t) {
            if ((X[t] - X_prev[t]).norm() > Delta) {
                return false;
            }
        }
        return true;
    }
};

} // namespace cddp

#endif // CDDP_SCP_IPOPT_HPP
