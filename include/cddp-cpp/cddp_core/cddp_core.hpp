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
#include <memory> // For std::unique_ptr
#include <map>    // For std::map
#include <Eigen/Dense>
#include <vector>
#include <regex>
#include "osqp++.h"
// #include "torch/torch.h"

#include "cddp_core/dynamical_system.hpp" 
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"

namespace cddp {

struct CDDPOptions {
    double cost_tolerance = 1e-2;         // Tolerance for changes in cost function
    double grad_tolerance = 1e-2;         // Tolerance for cost gradient magnitude
    int max_iterations = 1;
    double max_cpu_time = 0.0;            // Maximum CPU time for the solver in seconds

    // Line search method
    int max_line_search_iterations = 1; // Maximum iterations for line search
    double backtracking_coeff = 1.0;      // Maximum step size for line search backtracking
    double backtracking_min = 0.5;    // Coefficient for line search backtracking
    double backtracking_factor = std::pow(2, -1);  // Factor for line search backtracking
    double minimum_reduction_ratio = 1e-6;      // Minimum reduction for line search

    // log-barrier method
    double barrier_coeff = 1e-2;          // Coefficient for log-barrier method
    double barrier_factor = 0.90;        // Factor for log-barrier method
    double barrier_tolerance = 1e-6;     // Tolerance for log-barrier method
    double relaxation_coeff = 1.0;             // Relaxation for log-barrier method

    // Active set method
    double active_set_tolerance = 1e-6;  // Tolerance for active set method

    // Regularization options
    std::string regularization_type = "control";          // different regularization types: ["none", "control", "state", "both"]
    
    double regularization_state = 1e-6;       // Regularization for state
    double regularization_state_step = 1.0;  // Regularization step for state
    double regularization_state_max = 1e6;      // Maximum regularization
    double regularization_state_min = 1e-6;     // Minimum regularization
    double regularization_state_factor = 1.5;  // Factor for state regularization

    double regularization_control = 1e-6;       // Regularization for control
    double regularization_control_step = 1.0;  // Regularization step for control
    double regularization_control_max = 1e6;      // Maximum regularization
    double regularization_control_min = 1e-6;     // Minimum regularization
    double regularization_control_factor = 1.5;  // Factor for control regularization

    // Other options
    bool verbose = true;         // Option for debug printing
    bool is_ilqr = true;                  // Option for iLQR
    bool use_parallel = false;            // Option for parallel computation
};

struct CDDPSolution {
    std::vector<double> time_sequence;
    std::vector<Eigen::VectorXd> control_sequence;
    std::vector<Eigen::VectorXd> state_sequence;
    std::vector<double> cost_sequence;
    std::vector<double> lagrangian_sequence;
    int iterations;
    bool converged;
    double solve_time;
};

struct ForwardPassInfo {
    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> U;
    double J;
    double dJ;
    double L; 
    double dL; 
    bool success;
};

class CDDP {
public:
    // Constructor
    CDDP(const Eigen::VectorXd& initial_state, 
         const Eigen::VectorXd& reference_state,
         int horizon,
         double timestep); 

    // Accessor methods

    // Getters
    const DynamicalSystem& getSystem() const { return *system_; }
    const Eigen::VectorXd& getInitialState() const { return initial_state_; }
    const Eigen::VectorXd& getReferenceState() const { return reference_state_; }
    int getHorizon() const { return horizon_; }
    double getTimestep() const { return timestep_; }
    const CDDPOptions& getOptions() const { return options_; }
    // Get the objective, cast to the appropriate type if needed
    template <typename T>
    const T& getObjective() const { 
        return dynamic_cast<const T&>(*objective_); 
    }


    // Setters
    void setDynamicalSystem(std::unique_ptr<DynamicalSystem> system) { system_ = std::move(system); }
    // void setDynamicalSystem(std::unique_ptr<torch::nn::Module> system) { torch_system_ = std::move(system); } 
    void setInitialState(const Eigen::VectorXd& initial_state) { initial_state_ = initial_state; }
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }
    void setHorizon(int horizon) { horizon_ = horizon; }
    void setTimestep(double timestep) { timestep_ = timestep; }
    void setOptions(const CDDPOptions& options) { options_ = options; }
    void setObjective(std::unique_ptr<Objective> objective) { objective_ = std::move(objective); }
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) { X_ = X; U_ = U; }
    void addConstraint(std::string constraint_name, std::unique_ptr<Constraint> constraint) {
        constraint_set_[constraint_name] = std::move(constraint);
    }
        
    // Get a specific constraint by name
    template <typename T>
    T& getConstraint(const std::string& name) {
        auto it = constraint_set_.find(name);
        if (it == constraint_set_.end()) {
            throw std::runtime_error("Constraint not found: " + name);
        }
        try {
            // Note: Returning a non-const reference here
            return dynamic_cast<T&>(*(it->second)); 
        } catch (const std::bad_cast& e) {
            throw std::runtime_error("Invalid constraint type: " + name);
        }
    }

    // Getter for the constraint set
    const std::map<std::string, std::unique_ptr<Constraint>>& getConstraintSet() const { 
        return constraint_set_; 
    }

    // Solve the problem
    CDDPSolution solve();
    CDDPSolution solveCLDDP();

private:
    // Initialization methods
    void initializeCDDP(); // Initialize the CDDP solver

    // Solver methods
    bool solveForwardPass();
    ForwardPassInfo solveForwardPassIteration(double alpha);
    bool solveBackwardPass();

    bool solveCLDDPForwardPass();
    bool solveCLDDPBackwardPass();

    // Helper methods
    void printSolverInfo();
    void printIteration(int iter, double cost, double lagrangian, double grad_norm, double lambda_state, double lambda_control);
    void printOptions(const CDDPOptions& options);
    void printSolution(const CDDPSolution& solution);
    bool checkConvergence(double J_new, double J_old, double dJ, double expected_dV, double gradient_norm);

    // Log-barrier method
    double getLogBarrierCost(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> getLogBarrierCostGradients(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
        return {getLogBarrierCostStateGradient(constraint, state, control, barrier_coeff, relaxation_coeff), getLogBarrierCostControlGradient(constraint, state, control, barrier_coeff, relaxation_coeff)};
    }
    Eigen::VectorXd getLogBarrierCostStateGradient(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff);
    Eigen::VectorXd getLogBarrierCostControlGradient(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getLogBarrierCostHessians(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
        return {getLogBarrierCostStateHessian(constraint, state, control, barrier_coeff, relaxation_coeff), getLogBarrierCostControlHessian(constraint, state, control, barrier_coeff, relaxation_coeff), Eigen::MatrixXd::Zero(control.size(), state.size())};
    }
    Eigen::MatrixXd getLogBarrierCostStateHessian(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff);
    Eigen::MatrixXd getLogBarrierCostControlHessian(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff);

    // Problem Data
    std::unique_ptr<DynamicalSystem> system_;         // Eigen-based dynamical system
    // std::unique_ptr<torch::nn::Module> torch_system_; // Torch-based dynamical system (learned dynamics model)
    Eigen::VectorXd initial_state_;      // Initial state of the system
    Eigen::VectorXd reference_state_;      // Desired reference state
    int horizon_;                      // Time horizon for the problem
    double timestep_;                  // Time step for the problem

    // Options
    CDDPOptions options_;              // Options for the solver

    // Intermediate trajectories
    std::vector<Eigen::VectorXd> X_;                  // State trajectory
    std::vector<Eigen::VectorXd> U_;                  // Control trajectory

    // Intermediate cost
    double J_; // Cost 
    double dJ_; // Cost improvement
    double L_; // Lagrangian
    double dL_; // Lagrangian improvement
    double optimality_gap_ = 1e+10;
    double barrier_coeff_;

    // Cost function
    std::unique_ptr<Objective> objective_; // Objective function

    // Constraints
    std::map<std::string, std::unique_ptr<Constraint>> constraint_set_; 

    // Feedforward and feedback gains
    std::vector<Eigen::VectorXd> k_;
    std::vector<Eigen::MatrixXd> K_;

    // Intermediate value function
    Eigen::VectorXd dV_;
    std::vector<Eigen::VectorXd> V_X_;
    std::vector<Eigen::MatrixXd> V_XX_;

    // Q-function matrices
    std::vector<Eigen::MatrixXd> Q_UU_;
    std::vector<Eigen::MatrixXd> Q_UX_;
    std::vector<Eigen::VectorXd> Q_U_; 

    // QP sover 
    osqp::OsqpSolver osqp_solver_;

    // Regularization parameters
    double regularization_state_;
    double regularization_state_step_;
    double regularization_control_;
    double regularization_control_step_;   
};
}
#endif // CDDP_CDDP_CORE_HPP