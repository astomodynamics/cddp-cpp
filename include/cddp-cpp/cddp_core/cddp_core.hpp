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
#include <map>    // For std::map`
#include <iomanip> // For std::setw
#include <Eigen/Dense>
#include <vector>
#include <regex>
#include <future>
#include <thread>
#include "osqp++.h"
// #include "torch/torch.h"

#include "cddp_core/dynamical_system.hpp" 
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"

namespace cddp {

struct CDDPOptions {
    double cost_tolerance = 1e-7;                   // Tolerance for changes in cost function
    double grad_tolerance = 1e-4;                   // Tolerance for cost gradient magnitude
    int max_iterations = 1;                         // Maximum number of iterations
    double max_cpu_time = 0.0;                      // Maximum CPU time for the solver in seconds

    // Line search method
    int max_line_search_iterations = 11;            // Maximum iterations for line search
    double backtracking_coeff = 1.0;                // Maximum step size for line search backtracking
    double backtracking_min = 0.5;                  // Coefficient for line search backtracking
    double backtracking_factor = std::pow(10, (-3.0/10.0));   // Factor for line search backtracking
    double minimum_reduction_ratio = 1e-6;          // Minimum reduction for line search

    // log-barrier method
    double barrier_coeff = 1e-2;                    // Coefficient for log-barrier method
    double barrier_factor = 0.90;                   // Factor for log-barrier method
    double barrier_tolerance = 1e-6;                // Tolerance for log-barrier method
    double relaxation_coeff = 1.0;                  // Relaxation for log-barrier method
    int barrier_order = 2;                          // Order for log-barrier method

    // Active set method
    double active_set_tolerance = 1e-6;             // Tolerance for active set method

    // Regularization options
    std::string regularization_type = "control";    // different regularization types: ["none", "control", "state", "both"]
    
    double regularization_state = 1e-6;             // Regularization for state
    double regularization_state_step = 1.0;         // Regularization step for state
    double regularization_state_max = 1e10;          // Maximum regularization
    double regularization_state_min = 1e-6;         // Minimum regularization
    double regularization_state_factor = 1.6;       // Factor for state regularization

    double regularization_control = 1e-6;           // Regularization for control
    double regularization_control_step = 1.0;       // Regularization step for control
    double regularization_control_max = 1e10;        // Maximum regularization
    double regularization_control_min = 1e-6;       // Minimum regularization
    double regularization_control_factor = 1.6;     // Factor for control regularization

    // Other options
    bool verbose = true;                            // Option for debug printing
    bool debug = false;                             // Option for debug mode
    bool is_ilqr = true;                            // Option for iLQR
    bool use_parallel = true;                      // Option for parallel computation
    int num_threads = max_line_search_iterations; // Number of threads to use
};

struct CDDPSolution {
    std::vector<double> time_sequence;
    std::vector<Eigen::VectorXd> control_sequence;
    std::vector<Eigen::VectorXd> state_sequence;
    std::vector<double> cost_sequence;
    std::vector<double> lagrangian_sequence;
    int iterations;
    double alpha;
    bool converged;
    double solve_time;
};

struct ForwardPassResult {
    std::vector<Eigen::VectorXd> state_sequence;
    std::vector<Eigen::VectorXd> control_sequence;
    double cost;
    double lagrangian;
    double alpha;
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
    const Objective& getObjective() const { return *objective_; }
    const Eigen::VectorXd& getInitialState() const { return initial_state_; }
    const Eigen::VectorXd& getReferenceState() const { return reference_state_; }
    int getHorizon() const { return horizon_; }
    double getTimestep() const { return timestep_; }
    const CDDPOptions& getOptions() const { return options_; }

    // Setters
    // void setDynamicalSystem(std::unique_ptr<torch::nn::Module> system) { torch_system_ = std::move(system); } 
    
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
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) { X_ = X; U_ = U; }
    
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


    // Getter for the constraint set
    const std::map<std::string, std::unique_ptr<Constraint>>& getConstraintSet() const { 
        return constraint_set_; 
    }

    // Solve the problem
    CDDPSolution solve();
    CDDPSolution solveCLDDP();
    CDDPSolution solveLogCDDP();

private:
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
    double expected_;
    double reduction_;

    // Line search
    double alpha_; // Line search step size
    std::vector<double> alphas_;

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

    // Log-barrier method
    std::unique_ptr<LogBarrier> log_barrier_;

    // Regularization parameters
    double regularization_state_;
    double regularization_state_step_;
    double regularization_control_;
    double regularization_control_step_;   

    // Initialization methods
    void initializeCDDP(); // Initialize the CDDP solver

    // Solver methods
    bool solveForwardPass();
    ForwardPassResult solveForwardPassIteration(double alpha);
    bool solveBackwardPass();

    ForwardPassResult solveCLDDPForwardPass(double alpha);
    bool solveCLDDPBackwardPass();

    ForwardPassResult solveLogCDDPForwardPass(double alpha);
    bool solveLogCDDPBackwardPass();

    // Helper methods
    bool checkConvergence(double J_new, double J_old, double dJ, double expected_dV, double gradient_norm);

    

    void printSolverInfo()
    {
        std::cout << "\n";
        std::cout << "\033[34m"; // Set text color to blue
        std::cout << "+---------------------------------------------------------+" << std::endl;
        std::cout << "|    ____ ____  ____  ____    _          ____             |" << std::endl;
        std::cout << "|   / ___|  _ \\|  _ \\|  _ \\  (_)_ __    / ___| _     _    |" << std::endl;
        std::cout << "|  | |   | | | | | | | |_) | | | '_ \\  | |   _| |_ _| |_  |" << std::endl;
        std::cout << "|  | |___| |_| | |_| |  __/  | | | | | | |__|_   _|_   _| |" << std::endl;
        std::cout << "|   \\____|____/|____/|_|     |_|_| |_|  \\____||_|   |_|   |" << std::endl;
        std::cout << "+---------------------------------------------------------+" << std::endl;
        std::cout << "\n";
        std::cout << "Constrained Differential Dynamic Programming\n";
        std::cout << "Author: Tomo Sasaki (@astomodynamics)\n";
        std::cout << "----------------------------------------------------------\n";
        std::cout << "\033[0m"; // Reset text color
        std::cout << "\n";
    }

    void printOptions(const CDDPOptions &options)
    {
        std::cout << "\n========================================\n";
        std::cout << "           CDDP Options\n";
        std::cout << "========================================\n";

        std::cout << "Cost Tolerance: " << std::setw(10) << options.cost_tolerance << "\n";
        std::cout << "Grad Tolerance: " << std::setw(10) << options.grad_tolerance << "\n";
        std::cout << "Max Iterations: " << std::setw(10) << options.max_iterations << "\n";
        std::cout << "Max CPU Time: " << std::setw(10) << options.max_cpu_time << "\n";

        std::cout << "\nLine Search:\n";
        std::cout << "  Max Iterations: " << std::setw(5) << options.max_line_search_iterations << "\n";
        std::cout << "  Backtracking Coeff: " << std::setw(5) << options.backtracking_coeff << "\n";
        std::cout << "  Backtracking Min: " << std::setw(5) << options.backtracking_min << "\n";
        std::cout << "  Backtracking Factor: " << std::setw(5) << options.backtracking_factor << "\n";

        std::cout << "\nLog-Barrier:\n";
        std::cout << "  Barrier Coeff: " << std::setw(5) << options.barrier_coeff << "\n";
        std::cout << "  Barrier Factor: " << std::setw(5) << options.barrier_factor << "\n";
        std::cout << "  Barrier Tolerance: " << std::setw(5) << options.barrier_tolerance << "\n";
        std::cout << "  Relaxation Coeff: " << std::setw(5) << options.relaxation_coeff << "\n";

        std::cout << "\nRegularization:\n";
        std::cout << "  Regularization Type: " << options.regularization_type << "\n";
        std::cout << "  Regularization State: " << std::setw(5) << options.regularization_state << "\n";
        std::cout << "  Regularization State Step: " << std::setw(5) << options.regularization_state_step << "\n";
        std::cout << "  Regularization State Max: " << std::setw(5) << options.regularization_state_max << "\n";
        std::cout << "  Regularization State Min: " << std::setw(5) << options.regularization_state_min << "\n";
        std::cout << "  Regularization State Factor: " << std::setw(5) << options.regularization_state_factor << "\n";

        std::cout << "  Regularization Control: " << std::setw(5) << options.regularization_control << "\n";
        std::cout << "  Regularization Control Step: " << std::setw(5) << options.regularization_control_step << "\n";
        std::cout << "  Regularization Control Max: " << std::setw(5) << options.regularization_control_max << "\n";
        std::cout << "  Regularization Control Min: " << std::setw(5) << options.regularization_control_min << "\n";
        std::cout << "  Regularization Control Factor: " << std::setw(5) << options.regularization_control_factor << "\n";

        std::cout << "\nOther:\n";
        std::cout << "  Print Iterations: " << (options.verbose ? "Yes" : "No") << "\n";
        std::cout << "  iLQR: " << (options.is_ilqr ? "Yes" : "No") << "\n";
        std::cout << "  Use Parallel: " << (options.use_parallel ? "Yes" : "No") << "\n";
        std::cout << "  Num Threads: " << options.num_threads << "\n";

        std::cout << "========================================\n\n";
    }

    void printIteration(int iter, double cost, double lagrangian, double grad_norm, 
                    double lambda_state, double lambda_control, double step_size)
    {
        // Print header for better readability every 10 iterations
        if (iter % 10 == 0)
        {
            std::cout << std::setw(10) << "Iteration"
                    << std::setw(15) << "Objective"
                    << std::setw(15) << "Lagrangian"
                    << std::setw(15) << "Grad Norm"
                    << std::setw(15) << "Step Size"
                    << std::setw(15) << "Reg (State)"
                    << std::setw(15) << "Reg (Control)"
                    << std::endl;
            std::cout << std::string(95, '-') << std::endl;
        }

        // Print iteration details
        std::cout << std::setw(10) << iter
                << std::setw(15) << std::setprecision(6) << cost
                << std::setw(15) << std::setprecision(6) << lagrangian
                << std::setw(15) << std::setprecision(6) << grad_norm
                << std::setw(15) << std::setprecision(6) << step_size
                << std::setw(15) << std::setprecision(6) << lambda_state
                << std::setw(15) << std::setprecision(6) << lambda_control
                << std::endl;
    }

    void printSolution(const CDDPSolution &solution)
    {
        std::cout << "\n========================================\n";
        std::cout << "           CDDP Solution\n";
        std::cout << "========================================\n";

        std::cout << "Converged: " << (solution.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations: " << solution.iterations << "\n";
        std::cout << "Solve Time: " << std::setprecision(4) << solution.solve_time << " micro sec\n";
        std::cout << "Final Cost: " << std::setprecision(6) << solution.cost_sequence.back() << "\n"; // Assuming cost_sequence is not empty

        std::cout << "========================================\n\n";
    }
};
}
#endif // CDDP_CDDP_CORE_HPP