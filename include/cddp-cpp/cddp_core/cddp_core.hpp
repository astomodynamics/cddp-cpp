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
#include <string>  // For std::string
#include <memory> // For std::unique_ptr
#include <map>    // For std::map`
#include <iomanip> // For std::setw
#include <Eigen/Dense>
#include <vector>
#include <regex>
#include <future>
#include <thread>
// #include "torch/torch.h"

#include "cddp_core/dynamical_system.hpp" 
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"
#include "cddp_core/barrier.hpp"
#include "cddp_core/boxqp.hpp"

namespace cddp {

struct CDDPOptions {
    double cost_tolerance = 1e-6;                   // Tolerance for changes in cost function
    double grad_tolerance = 1e-4;                   // Tolerance for cost gradient magnitude
    int max_iterations = 1;                         // Maximum number of iterations
    double max_cpu_time = 0.0;                      // Maximum CPU time for the solver in seconds

    // Line search method
    int max_line_search_iterations = 21;            // Maximum iterations for line search
    double backtracking_coeff = 1.0;                // Maximum step size for line search backtracking
    double backtracking_min = 0.5;                  // Coefficient for line search backtracking
    double backtracking_factor = std::pow(10, (-3.0/10.0));   // Factor for line search backtracking
    double minimum_reduction_ratio = 1e-6;          // Minimum reduction for line search

    // interior-point method
    double mu_initial = 1e-2;                       // Initial barrier coefficient
    double mu_min = 1e-8;                           // Minimum barrier coefficient
    double mu_max = 1e1;                            // Maximum barrier coefficient
    double mu_reduction_ratio = 0.1;                         // Factor for barrier coefficient

    // log-barrier method
    double barrier_coeff = 1e-2;                    // Coefficient for log-barrier method
    double barrier_factor = 0.10;                   // Factor for log-barrier method
    double barrier_tolerance = 1e-8;                // Tolerance for log-barrier method
    double relaxation_coeff = 1.0;                  // Relaxation for log-barrier method
    int barrier_order = 2;                          // Order for log-barrier method
    double filter_acceptance = 1e-8;                            // Small value for filter acceptance
    double constraint_tolerance = 1e-12;             // Tolerance for constraint violation

    // Regularization options
    std::string regularization_type = "control";    // different regularization types: ["none", "control", "state", "both"]
    
    double regularization_state = 1e-6;             // Regularization for state
    double regularization_state_step = 1.0;         // Regularization step for state
    double regularization_state_max = 1e6;          // Maximum regularization
    double regularization_state_min = 1e-8;         // Minimum regularization
    double regularization_state_factor = 1e1;       // Factor for state regularization

    double regularization_control = 1e-6;           // Regularization for control
    double regularization_control_step = 1.0;       // Regularization step for control
    double regularization_control_max = 1e5;        // Maximum regularization
    double regularization_control_min = 1e-8;       // Minimum regularization
    double regularization_control_factor = 1e1;     // Factor for control regularization

    // Other options
    bool verbose = true;                            // Option for debug printing
    bool debug = false;                             // Option for debug mode
    bool header_and_footer = true;                  // Option for header and footer
    bool is_ilqr = true;                            // Option for iLQR
    bool use_parallel = true;                      // Option for parallel computation
    int num_threads = max_line_search_iterations; // Number of threads to use
    bool is_relaxed_log_barrier = false;            // Use relaxed log-barrier method
    bool early_termination = true;                 // Terminate early if cost does not change NOTE: This may be critical for some problems

    // Boxqp options
    double boxqp_max_iterations = 100;              // Maximum number of iterations for boxqp
    double boxqp_min_grad = 1e-8;                   // Minimum gradient norm for boxqp
    double boxqp_min_rel_improve = 1e-8;            // Minimum relative improvement for boxqp
    double boxqp_step_dec = 0.6;                    // Step decrease factor for boxqp
    double boxqp_min_step = 1e-22;                  // Minimum step size for boxqp
    double boxqp_armijo = 0.1;                      // Armijo parameter for boxqp
    bool boxqp_verbose = false;                     // Print debug info for boxqp
};

struct CDDPSolution {
    std::vector<double> time_sequence;
    std::vector<Eigen::VectorXd> control_sequence;
    std::vector<Eigen::VectorXd> state_sequence;
    std::vector<Eigen::VectorXd> dual_sequence;
    std::vector<Eigen::VectorXd> slack_sequence;
    std::vector<double> cost_sequence;
    std::vector<double> lagrangian_sequence;
    std::vector<Eigen::MatrixXd> control_gain;
    std::vector<Eigen::MatrixXd> dual_gain;
    std::vector<Eigen::MatrixXd> slack_gain;
    int iterations;
    double alpha;
    bool converged;
    double solve_time;
};

struct ForwardPassResult {
    std::vector<Eigen::VectorXd> state_sequence;
    std::vector<Eigen::VectorXd> control_sequence;
    std::map<std::string, std::vector<Eigen::VectorXd>> dual_sequence;
    std::map<std::string, std::vector<Eigen::VectorXd>> slack_sequence;
    double cost;
    double lagrangian;
    double alpha = 1.0;
    bool success = false;
    double constraint_violation = 0.0;
};

struct FilterPoint {
    double cost;
    double violation;
    bool dominates(const FilterPoint& other) const {
        return cost <= other.cost && violation <= other.violation;
    }
};

class CDDP {
public:
    // Constructor
    CDDP(const Eigen::VectorXd& initial_state, 
         const Eigen::VectorXd& reference_state,
         int horizon,
         double timestep,
         std::unique_ptr<DynamicalSystem> system = nullptr,
        std::unique_ptr<Objective> objective = nullptr,
        const CDDPOptions& options = CDDPOptions()); 

    // Accessor methods

    // Getters
    const DynamicalSystem& getSystem() const { return *system_; }
    const Objective& getObjective() const { return *objective_; }
    const Eigen::VectorXd& getInitialState() const { return initial_state_; }
    const Eigen::VectorXd& getReferenceState() const { return reference_state_; }
    int getHorizon() const { return horizon_; }
    double getTimestep() const { return timestep_; }
    int getStateDim() const { return system_->getStateDim(); }
    int getControlDim() const { return system_->getControlDim(); }
    int getTotalDualDim() const { return total_dual_dim_; }
    const CDDPOptions& getOptions() const { return options_; }

    // Setters
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
    void setReferenceState(const Eigen::VectorXd& reference_state) {        
        reference_state_ = reference_state; 
        // Update the objective reference state
        objective_->setReferenceState(reference_state);   
    }

    void setReferenceStates(const std::vector<Eigen::VectorXd>& reference_states) {
        reference_states_ = reference_states;
        // Update the objective reference states
        objective_->setReferenceStates(reference_states);
    }

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
    void setOptions(const CDDPOptions& options) { options_ = options; }

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
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U);
    
    /**
     * @brief Add a constraint to the problem
     * 
     * @param constraint_name constraint name given by the user
     * @param constraint constraint object
     */
    void addConstraint(std::string constraint_name, std::unique_ptr<Constraint> constraint) {
        // Insert into the map
        constraint_set_[constraint_name] = std::move(constraint);

        // Increment total_dual_dim_
        int dim = constraint_set_[constraint_name]->getDualDim();
        total_dual_dim_ += dim;

        initialized_ = false; // Reset the initialization flag
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
        auto it = constraint_set_.find(name);
        
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


    // Getter for the constraint set
    const std::map<std::string, std::unique_ptr<Constraint>>& getConstraintSet() const { 
        return constraint_set_; 
    }

    // Initialization methods
    void initializeCDDP();

    // Solve the problem
    CDDPSolution solve(std::string solver_type = "CLCDDP");
    
private:
    // Solver methods
    // CLCDDP methods
    CDDPSolution solveCLCDDP();
    ForwardPassResult solveCLCDDPForwardPass(double alpha);
    bool solveCLCDDPBackwardPass();

    // LogCDDP methods
    CDDPSolution solveLogCDDP();
    ForwardPassResult solveLogCDDPForwardPass(double alpha);
    bool solveLogCDDPBackwardPass();

    // ASCDDP methods
    CDDPSolution solveASCDDP();
    ForwardPassResult solveASCDDPForwardPass(double alpha);
    bool solveASCDDPBackwardPass();
    
    // IPDDP methods
    void initializeIPDDP();
    CDDPSolution solveIPDDP();
    ForwardPassResult solveIPDDPForwardPass(double alpha);
    bool solveIPDDPBackwardPass();

    // Feasible IPDDP methods
    void initializeFeasibleIPDDP();
    CDDPSolution solveFeasibleIPDDP();
    ForwardPassResult solveFeasibleIPDDPForwardPass(double alpha);
    bool solveFeasibleIPDDPBackwardPass();

    // Helper methods
    double computeConstraintViolation(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) const;

    bool checkConvergence(double J_new, double J_old, double dJ, double expected_dV, double gradient_norm);

    // Regularization methods
    void increaseRegularization();
    void decreaseRegularization();
    bool isRegularizationLimitReached() const;

    // Print methods
    void printSolverInfo();
    void printOptions(const CDDPOptions& options);
    void printIteration(int iter, double cost, double lagrangian, 
                        double grad_norm, double lambda_state, 
                        double lambda_control, double alpha, double mu, double constraint_violation);
    void printSolution(const CDDPSolution& solution);

private:
    bool initialized_ = false; // Initialization flag

    // Problem Data
    std::unique_ptr<DynamicalSystem> system_;        
    std::unique_ptr<Objective> objective_;
    std::map<std::string, std::unique_ptr<Constraint>> constraint_set_; 
    std::unique_ptr<LogBarrier> log_barrier_;
    Eigen::VectorXd initial_state_;      
    Eigen::VectorXd reference_state_;      // Desired reference state
    std::vector<Eigen::VectorXd> reference_states_;     // Desired reference states (trajectory)
    int horizon_;                      // Time horizon for the problem
    double timestep_;                  // Time step for the problem
    CDDPOptions options_;              // Options for the solver

    int total_dual_dim_ = 0; // Number of total dual variables across constraints

    // Intermediate trajectories
    std::vector<Eigen::VectorXd> X_;                            // State trajectory
    std::vector<Eigen::VectorXd> U_;                            // Control trajectory
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_;  // Dual trajectory
    std::map<std::string, std::vector<Eigen::VectorXd>> S_; // Slack trajectory 

    // Cost and Lagrangian
    double J_; // Cost 
    double dJ_; // Cost improvement
    double L_; // Lagrangian
    double dL_; // Lagrangian improvement
    Eigen::VectorXd dV_;

    // Line search
    double alpha_; // Line search step size
    std::vector<double> alphas_;

    // Log-barrier
    double mu_; // Barrier coefficient
    double constraint_violation_; // Current constraint violation measure
    double gamma_; // Small value for filter acceptance
    
    // Feedforward and feedback gains
    std::vector<Eigen::VectorXd> k_u_;
    std::vector<Eigen::MatrixXd> K_u_;
    std::map<std::string, std::vector<Eigen::VectorXd>> k_y_;
    std::map<std::string, std::vector<Eigen::MatrixXd>> K_y_;
    std::map<std::string, std::vector<Eigen::VectorXd>> k_s_;
    std::map<std::string, std::vector<Eigen::MatrixXd>> K_s_;

    // Q-function matrices
    std::vector<Eigen::MatrixXd> Q_UU_;
    std::vector<Eigen::MatrixXd> Q_UX_;
    std::vector<Eigen::VectorXd> Q_U_;     

    // Regularization parameters
    double regularization_state_;
    double regularization_state_step_;
    double regularization_control_;
    double regularization_control_step_;   

    // Boxqp solver
    BoxQPOptions boxqp_options_;
    BoxQPSolver boxqp_solver_;

    double optimality_gap_ = 1e+10;
};
} // namespace cddp
#endif // CDDP_CDDP_CORE_HPP