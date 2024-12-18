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
#ifndef CDDP_SQP_SOLVER_HPP
#define CDDP_SQP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>
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
    double ftol = 1e-6;                // Function value tolerance
    double xtol = 1e-6;                // Step size tolerance
    double gtol = 1e-6;                // Gradient tolerance
    double eta = 0.1;                  // Merit function parameter
    double tau = 0.5;                  // Line search parameter
    bool verbose = false;              // Print debug info

    int line_search_max_iterations = 20; // Maximum line search backtracking iterations
    
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
 * @brief Sequential Quadratic Programming solver using OSQP
 */
class SQPSolver {
public:
    /**
     * @brief Constructor
     * @param options Solver configuration options
     */
    SQPSolver(const Eigen::VectorXd& initial_state, 
         const Eigen::VectorXd& reference_state,
         int horizon,
         double timestep);

    /**
     * @brief Set the Dynamical System object
     * @param system Dynamical system object (unique_ptr)
     */
    void setDynamicalSystem(std::unique_ptr<DynamicalSystem> system) { system_ = std::move(system); }

    /**
     * @brief Set the Initial state
     * @param initial_state Initial state
     */
    void setInitialState(const Eigen::VectorXd& initial_state) { initial_state_ = initial_state; }

    /**
     * @brief Set the Reference state
     * @param reference_state  Reference state
     */
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }

    /**
     * @brief Set the time horizon for the problem
     * @param horizon Time horizon
     */
    void setHorizon(int horizon) { horizon_ = horizon; }

    /**
     * @brief Set the time step for the problem
     * @param timestep Time step
     */
    void setTimestep(double timestep) { timestep_ = timestep; }

    /**
     * @brief Set the options for the solver
     * @param options Solver options
     */
    void setOptions(const SQPOptions& options) { 
        options_ = options;

        // Initialize OSQP settings
        osqp_settings_.eps_abs = options_.osqp_eps_abs;
        osqp_settings_.eps_rel = options_.osqp_eps_rel;
        osqp_settings_.max_iter = options_.osqp_max_iter;
        osqp_settings_.verbose = options_.osqp_verbose;
        osqp_settings_.warm_start = options_.warm_start;
        osqp_settings_.adaptive_rho = true;
    }

    /**
     * @brief Set the Objective function
     * @param objective Objective function object (unique_ptr)
     */
    void setObjective(std::unique_ptr<Objective> objective) { objective_ = std::move(objective); }

    /**
     * @brief Set the Initial Trajectory 
     * @param X state trajectory
     * @param U control trajectory
     */
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, 
                             const std::vector<Eigen::VectorXd>& U);

    /**
     * @brief Solve the optimization problem
     * @return Solution results
     */
    SQPResult solve();


    /**
     * @brief Add a constraint to the problem
     * 
     * @param constraint_name constraint name given by the user
     * @param constraint constraint object
     */
    void addConstraint(std::string constraint_name, std::unique_ptr<Constraint> constraint) {
        constraint_set_[constraint_name] = std::move(constraint);
    }

    /**
     * @brief Get a specific constraint by name
     * 
     * @tparam T Type of constraint
     * @param name Name of the constraint
     * @return T* Pointer to the constraint
     */
    // Get a specific constraint by name
    template <typename T>
    T* getConstraint(const std::string& name) const {
        // 'find' is a const operation on standard associative containers 
        // and does not modify 'constraint_set_'
        auto it = constraint_set_.find(name);
        
        // Special case for ControlBoxConstraint - must exist
        if (std::is_same<T, ControlBoxConstraint>::value) {
            if (it == constraint_set_.end()) {
                throw std::runtime_error("ControlBoxConstraint not found");
            }
            return dynamic_cast<T*>(it->second.get());
        }
        
        // For other constraints, return nullptr if not found
        if (it == constraint_set_.end()) {
            return nullptr;
        }

        // Try to cast to the requested type
        T* cast_constraint = dynamic_cast<T*>(it->second.get());
        if (!cast_constraint) {
            return nullptr;
        }

        return cast_constraint;
    }


private:
    // Member variables
    SQPOptions options_;
    osqp::OsqpSettings osqp_settings_;

    Eigen::VectorXd initial_state_;      // Initial state of the system
    Eigen::VectorXd reference_state_;      // Desired reference state
    int horizon_;                      // Time horizon for the problem
    double timestep_;   
    
    std::unique_ptr<DynamicalSystem> system_;
    std::unique_ptr<Objective> objective_;
    std::map<std::string, std::unique_ptr<Constraint>> constraint_set_; 

    std::vector<Eigen::VectorXd> X_; // State trajectory
    std::vector<Eigen::VectorXd> U_; // Control trajectory
    double J_;

    std::vector<Eigen::MatrixXd> A_; // State Jacobians
    std::vector<Eigen::MatrixXd> B_; // Control Jacobians

    /**
     * @brief Dynamics propagatition
     */
    void propagateDynamics(const Eigen::VectorXd& x, std::vector<Eigen::VectorXd>& U);

    /**
     * @brief Formulate QP subproblem
     * 
     * @param X State trajectory
     * @param U Control trajectory
     * @param H Hessian matrix (in-place)
     * @param g Gradient vector (in-place)
     * @param A Constraint matrix (in-place)
     * @param l Lower bound (in-place)
     * @param u Upper bound (in-place)
     */
    void formQPSubproblem(const std::vector<Eigen::VectorXd>& X,
                         const std::vector<Eigen::VectorXd>& U,
                         Eigen::SparseMatrix<double>& H,
                         Eigen::VectorXd& g,
                         Eigen::SparseMatrix<double>& A,
                         Eigen::VectorXd& l,
                         Eigen::VectorXd& u);

    double computeConstraintViolation(const std::vector<Eigen::VectorXd>& X,
                                      const std::vector<Eigen::VectorXd>& U) const;

    // Merit function for line search
    double computeMeritFunction(const std::vector<Eigen::VectorXd>& X,
                              const std::vector<Eigen::VectorXd>& U,
                              double eta) const;

    // Line search
    double lineSearch(const std::vector<Eigen::VectorXd>& X,
                     const std::vector<Eigen::VectorXd>& U,
                     const std::vector<Eigen::VectorXd>& dX,
                     const std::vector<Eigen::VectorXd>& dU);

    // Extract trajectory updates from QP solution
    void extractUpdates(const Eigen::VectorXd& qp_solution,
                       std::vector<Eigen::VectorXd>& dX,
                       std::vector<Eigen::VectorXd>& dU);

    // Update trajectories with step
    void updateTrajectories(const std::vector<Eigen::VectorXd>& X,
                          const std::vector<Eigen::VectorXd>& U,
                          const std::vector<Eigen::VectorXd>& dX,
                          const std::vector<Eigen::VectorXd>& dU,
                          double alpha,
                          std::vector<Eigen::VectorXd>& X_new,
                          std::vector<Eigen::VectorXd>& U_new);

    void computeLinearizedDynamics(const std::vector<Eigen::VectorXd>& X,
                                 const std::vector<Eigen::VectorXd>& U,
                                 std::vector<Eigen::MatrixXd>& A,
                                 std::vector<Eigen::MatrixXd>& B);

    void checkDimensions() const;
    void initializeSQP();
};

} // namespace cddp

#endif // CDDP_SQP_SOLVER_HPP