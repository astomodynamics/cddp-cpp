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
#ifndef CDDP_SQP_HPP
#define CDDP_SQP_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include "osqp++.h"
#include "cddp_core/dynamical_system.hpp"
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"

namespace cddp {

/**
 * @brief Configuration options for SQP solver
 */
struct SQPOptions {
    int max_iterations = 100;           // Maximum number of iterations
    int min_iterations = 1;             // Minimum number of iterations
    double ftol = 1e-6;                 // Function value tolerance
    double xtol = 1e-6;                 // Step size tolerance
    double gtol = 1e-6;                 // Gradient tolerance
    double eta = 1.0;                   // Merit function penalty weight
    double tau = 0.5;                   // Line search backtracking factor (0<tau<1)
    bool verbose = false;               // Print debug info

    int line_search_max_iterations = 20; // Maximum line search iterations

    // Trust region parameters
    double trust_region_radius = 100.0;       // Initial trust region radius
    double trust_region_radius_max = 1e6;       // Maximum trust region radius
    double trust_region_increase_factor = 2.0;  // Increase factor when step is good
    double trust_region_decrease_factor = 0.5;  // Decrease factor when step is rejected

    double merit_penalty = 100.0;         // Penalty for constraint violation in merit function

    // OSQP specific options
    double osqp_eps_abs = 1e-5;        // Absolute tolerance
    double osqp_eps_rel = 1e-3;        // Relative tolerance
    int osqp_max_iter = 4000;          // Maximum OSQP iterations
    bool osqp_verbose = false;         // OSQP verbosity
    bool warm_start = true;            // Use warm starting
};

/**
 * @brief Results from SQP optimization
 */
struct SQPResult {
    bool success;                      // Whether optimization succeeded
    int iterations;                    // Number of iterations taken
    double objective_value;            // Final objective value
    double constraint_violation;       // Final constraint violation
    std::vector<Eigen::VectorXd> X;    // Optimal state trajectory
    std::vector<Eigen::VectorXd> U;    // Optimal control trajectory
    std::vector<double> obj_history;   // History of objective values
    std::vector<double> viol_history;  // History of constraint violations
    double solve_time;                 // Solution time in seconds
};

/**
 * @brief Sequential Quadratic Programming solver using OSQP.
 */
class SQPSolver {
public:
    /**
     * @brief Constructor.
     * @param initial_state Initial state of the system.
     * @param reference_state Desired (goal) state.
     * @param horizon Time horizon for the problem.
     * @param timestep Time step.
     */
    SQPSolver(const Eigen::VectorXd& initial_state, 
              const Eigen::VectorXd& reference_state,
              int horizon,
              double timestep);

    // Setter methods
    void setDynamicalSystem(std::unique_ptr<DynamicalSystem> system) { system_ = std::move(system); }
    void setInitialState(const Eigen::VectorXd& initial_state) { initial_state_ = initial_state; }
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }
    void setHorizon(int horizon) { horizon_ = horizon; }
    void setTimestep(double timestep) { timestep_ = timestep; }
    void setOptions(const SQPOptions& options);
    void setObjective(std::unique_ptr<Objective> objective) { objective_ = std::move(objective); }
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, 
                              const std::vector<Eigen::VectorXd>& U);

    /**
     * @brief Solve the optimization problem.
     * @return SQPResult with the solution details.
     */
    SQPResult solve();

    /**
     * @brief Add a constraint to the problem.
     * @param constraint_name Constraint name.
     * @param constraint Constraint object.
     */
    void addConstraint(std::string constraint_name, std::unique_ptr<Constraint> constraint) {
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
        if (std::is_same<T, ControlBoxConstraint>::value) {
            if (it == constraint_set_.end()) {
                throw std::runtime_error("ControlBoxConstraint not found");
            }
            return dynamic_cast<T*>(it->second.get());
        }
        if (it == constraint_set_.end()) {
            return nullptr;
        }
        T* cast_constraint = dynamic_cast<T*>(it->second.get());
        return cast_constraint;
    }

private:
    // Member variables
    SQPOptions options_;
    osqp::OsqpSettings osqp_settings_;

    Eigen::VectorXd initial_state_;
    Eigen::VectorXd reference_state_;
    int horizon_;
    double timestep_;   
    
    std::unique_ptr<DynamicalSystem> system_;
    std::unique_ptr<Objective> objective_;
    std::map<std::string, std::unique_ptr<Constraint>> constraint_set_; 

    std::vector<Eigen::VectorXd> X_; // State trajectory
    std::vector<Eigen::VectorXd> U_; // Control trajectory

    // (Optional) cached linearization data
    std::vector<Eigen::MatrixXd> A_; // State Jacobians
    std::vector<Eigen::MatrixXd> B_; // Control Jacobians

    // Helper routines
    void initializeSQP();
    void propagateDynamics(const Eigen::VectorXd& x0, std::vector<Eigen::VectorXd>& U);
    void computeLinearizedDynamics(const std::vector<Eigen::VectorXd>& X,
                                   const std::vector<Eigen::VectorXd>& U,
                                   std::vector<Eigen::MatrixXd>& A,
                                   std::vector<Eigen::MatrixXd>& B);
    void checkDimensions() const;

    /**
     * @brief Formulate the QP subproblem.
     *
     * The QP includes:
     *   - A quadratic cost (from running and terminal costs)
     *   - Dynamics equality constraints,
     *   - Box constraints on states (trust region) and controls,
     *   - And a hard terminal constraint: x_N = reference_state_.
     */
    void formQPSubproblem(const std::vector<Eigen::VectorXd>& X,
                          const std::vector<Eigen::VectorXd>& U,
                          double trust_region_radius,
                          Eigen::SparseMatrix<double>& H,
                          Eigen::VectorXd& g,
                          Eigen::SparseMatrix<double>& A,
                          Eigen::VectorXd& l,
                          Eigen::VectorXd& u);

    double computeConstraintViolation(const std::vector<Eigen::VectorXd>& X,
                                      const std::vector<Eigen::VectorXd>& U) const;
    double computeMeritFunction(const std::vector<Eigen::VectorXd>& X,
                                const std::vector<Eigen::VectorXd>& U,
                                double penalty) const;

    /**
     * @brief Line search using a simple Armijo rule.
     * @return Step–size alpha.
     */
    double lineSearch(const std::vector<Eigen::VectorXd>& X,
                      const std::vector<Eigen::VectorXd>& U,
                      const std::vector<Eigen::VectorXd>& dX,
                      const std::vector<Eigen::VectorXd>& dU);

    /**
     * @brief Extract trajectory updates from the QP solution.
     */
    void extractUpdates(const Eigen::VectorXd& qp_solution,
                        std::vector<Eigen::VectorXd>& dX,
                        std::vector<Eigen::VectorXd>& dU);

    /**
     * @brief Update trajectories using a given step–size alpha.
     */
    void updateTrajectories(const std::vector<Eigen::VectorXd>& X,
                            const std::vector<Eigen::VectorXd>& U,
                            const std::vector<Eigen::VectorXd>& dX,
                            const std::vector<Eigen::VectorXd>& dU,
                            std::vector<Eigen::VectorXd>& X_new,
                            std::vector<Eigen::VectorXd>& U_new,
                            double alpha);
};

} // namespace cddp

#endif // CDDP_SQP_HPP
