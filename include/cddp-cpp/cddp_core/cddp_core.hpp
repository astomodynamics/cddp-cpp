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

#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <vector>
// #include "torch/torch.h"

#include "cddp-cpp/cddp_core/dynamical_system.hpp" 
#include "cddp-cpp/cddp_core/objective.hpp"
// #include "cddp-cpp/cddp_core/constraint.hpp"

namespace cddp {

struct CDDPOptions {
    double cost_tolerance = 1e-4;         // Tolerance for changes in cost function
    double grad_tolerance = 1e-6;         // Tolerance for cost gradient magnitude
    int max_iterations = 1;

    // Line search method
    int max_line_search_iterations = 1; // Maximum iterations for line search
    double backtracking_coeff = 1.0;      // Maximum step size for line search backtracking
    double backtracking_min = 0.5;    // Coefficient for line search backtracking
    double backtracking_factor = std::pow(2, -2);  // Factor for line search backtracking
    double minimum_reduction = 1e-4;      // Minimum reduction for line search

    // log-barrier method
    double barrier_coeff = 1.0;          // Coefficient for log-barrier method
    double barrier_factor = 0.90;        // Factor for log-barrier method
    double barrier_tolerance = 1e-6;     // Tolerance for log-barrier method
    double relaxation_coeff = 5;             // Relaxation for log-barrier method

    // Regularization options
    int regularization_type = 0;          // 0 or 1 for different regularization types
    double regularization_parameter = 1e-6; // Initial regularization parameter 
    double regularization_step = 1.0;     // Step size for regularization
    double regularization_factor = 10.0;  // Factor for regularization
    double regularization_max = 1e6;      // Maximum regularization
    double regularization_min = 1e-6;     // Minimum regularization

    double regularization_x = 1e-6;       // Regularization for state
    double regularization_u = 1e-6;       // Regularization for control
    double regularization_tolerance = 1e-6; // Tolerance for regularization

    // Other options
    bool verbose = true;         // Option for debug printing
    bool is_ilqr = true;                  // Option for iLQR
    bool use_parallel = false;            // Option for parallel computation
};

struct CDDPSolution {
    Eigen::MatrixXd control_sequence;
    Eigen::MatrixXd state_sequence;
    std::vector<double> cost_sequence;
    int iterations;
    bool converged;
    double solve_time;
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
    void setInitialTrajectory(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) { X_ = X; U_ = U; }
    // void addConstraint(std::unique_ptr<Constraint> constraint) { /*constraint_set_.push_back(std::move(constraint));*/ }

    // Solve the problem
    CDDPSolution solve();

private:
    // Initialization methods
    void initializeCDDP(); // Initialize the CDDP solver

    // Solver methods
    bool solveForwardPass();
    bool solveBackwardPass();

    // Helper methods
    void printSolverInfo();
    void printIteration(int iter, double cost, double grad_norm, double lambda);
    void printOptions(const CDDPOptions& options);
    void printSolution(const CDDPSolution& solution);
    bool checkConvergence(double J_new, double J_old, double dJ, double expected_dV, double gradient_norm);

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
    Eigen::MatrixXd X_;                  // State trajectory
    Eigen::MatrixXd U_;                  // Control trajectory

    // Intermediate cost
    double J_;

    // Cost function
    std::unique_ptr<Objective> objective_; // Objective function

    // Constraints
    // std::vector<std::unique_ptr<Constraint>> constraint_set_; // Set of constraints

    // Feedforward and feedback gains
    std::vector<Eigen::VectorXd> k_;
    std::vector<Eigen::MatrixXd> K_;

    // Intermediate value function
    std::vector<double> V_;
    std::vector<Eigen::VectorXd> V_X_;
    std::vector<Eigen::MatrixXd> V_XX_;

    // Q-function matrices
    std::vector<Eigen::MatrixXd> Q_UU_;
    std::vector<Eigen::MatrixXd> Q_UX_;
    std::vector<Eigen::VectorXd> Q_U_; 
};
}
#endif // CDDP_CDDP_CORE_HPP