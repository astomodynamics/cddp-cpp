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

#include <iostream> // For std::cout, std::cerr
#include <iomanip> // For std::setw
#include <memory> // For std::unique_ptr
#include <map>    // For std::map
#include <Eigen/Dense>
#include <chrono> // For timing
#include <execution> // For parallel execution policies
#include "osqp++.h"

#include "cddp-cpp/cddp_core/cddp_core.hpp"

namespace cddp {

// Constructor
CDDP::CDDP(const Eigen::VectorXd& initial_state, 
           const Eigen::VectorXd& reference_state,
           int horizon,
           double timestep) 
    : initial_state_(initial_state), 
      reference_state_(reference_state), 
      horizon_(horizon), 
      timestep_(timestep) {

    printSolverInfo();
    
    // initializeCDDP();
}

// Initialize the CDDP solver
void CDDP::initializeCDDP() {
    // Check if the system and objective are set
    if (!system_) {
        std::cerr << "CDDP: Dynamical system is not set" << std::endl;
        throw std::runtime_error("Dynamical system is not set");
    }

    if (!objective_) {
        std::cerr << "CDDP: Objective function is not set" << std::endl;
        throw std::runtime_error("Objective function is not set");
    }

    // Check if reference_state in objective and reference_state in CDDP are the same
    if ((reference_state_ - objective_->getReferenceState()).norm() > 1e-6) {
        std::cerr << "CDDP: Initial state and goal state in the objective function do not match" << std::endl;
        throw std::runtime_error("Initial state and goal state in the objective function do not match");
    }

    // Initialize trajectories (X_ and U_ are std::vectors of Eigen::VectorXd)
    if (X_.size() != horizon_ + 1 && U_.size() != horizon_) {
        X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(system_->getStateDim()));
        U_.resize(horizon_, Eigen::VectorXd::Zero(system_->getControlDim()));
    } else if (X_.size() != horizon_ + 1) {
        X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(system_->getStateDim()));
    } else if (U_.size() != horizon_) {
        U_.resize(horizon_, Eigen::VectorXd::Zero(system_->getControlDim()));
    }

    // Set initial state
    X_[0] = initial_state_;

    // Initialize cost
    J_ = 0.0;

    // Initialize gains and value function approximation
    k_.resize(horizon_, Eigen::VectorXd::Zero(system_->getControlDim()));
    K_.resize(horizon_, Eigen::MatrixXd::Zero(system_->getControlDim(), system_->getStateDim()));
    V_.resize(horizon_ + 1, 0.0);
    V_X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(system_->getStateDim()));
    V_XX_.resize(horizon_ + 1, Eigen::MatrixXd::Zero(system_->getStateDim(), system_->getStateDim()));

    // Initialize Q-function matrices
    Q_UU_.resize(horizon_, Eigen::MatrixXd::Zero(system_->getControlDim(), system_->getControlDim()));
    Q_UX_.resize(horizon_, Eigen::MatrixXd::Zero(system_->getControlDim(), system_->getStateDim()));
    Q_U_.resize(horizon_, Eigen::VectorXd::Zero(system_->getControlDim()));

    // Initialize constraints if empty
    if (constraint_set_.empty()) {
        std::cerr << "CDDP: No constraints are set" << std::endl;
    }
    

}

// Solve the problem
CDDPSolution CDDP::solve() {
    // Initialize CDDP solver
    initializeCDDP();

    if (options_.verbose) {
        printOptions(options_);
    }

    // Initialize solution
    CDDPSolution solution;
    solution.converged = false;
    solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency

    // // Evaluate initial cost
    J_ = objective_->evaluate(X_, U_);
    solution.cost_sequence.push_back(J_);
    std::cout << "Initial Cost: " << J_ << std::endl;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    // Main loop
    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        // Backward pass
        bool backward_pass_success = solveBackwardPass();
        if (!backward_pass_success) {
            break; // Exit if backward pass fails
        }

        // Forward pass
        bool forward_pass_success = solveForwardPass();
        if (!forward_pass_success) {
            break; // Exit if forward pass fails
        }

        // Check convergence
        double J_new = objective_->evaluate(X_, U_);
        // double dJ = J_old - J_new;
        // // ... (Calculate expected_dV and gradient_norm based on your algorithm) ...
        // solution.converged = checkConvergence(J_new, J_old, dJ, expected_dV, gradient_norm);
        // J_old = J_new;

        // if (solution.converged) {
        //     solution.iterations = iter + 1; // Update iteration count
        //     break; // Exit if converged
        // }

        // Print iteration information
        printIteration(iter, J_, 0.0, 0.0); // Assuming lambda is not used

        // Append Latest Cost
        solution.cost_sequence.push_back(J_new);
    }

    // // Finalize solution
    solution.control_sequence = U_;
    solution.state_sequence = X_;
    solution.iterations = solution.converged ? solution.iterations : options_.max_iterations;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 

    solution.solve_time = duration.count(); // Time in microseconds
    printSolution(solution);

    return solution;
}

// Backward pass
bool CDDP::solveBackwardPass() {
    auto active_set_tol = options_.active_set_tolerance;
    // Extract control box constraint
    auto control_box_constraint = constraint_set_.find("ControlBoxConstraint");

    // Terminal cost and its derivatives
    V_.back() = objective_->terminal_cost(X_.back());
    V_X_.back() = objective_->getFinalCostGradient(X_.back());
    V_XX_.back() = objective_->getFinalCostHessian(X_.back());

    // Pre-allocate matrices
    Eigen::MatrixXd Fx(system_->getStateDim(), system_->getStateDim());
    Eigen::MatrixXd Fu(system_->getStateDim(), system_->getControlDim());
    Eigen::MatrixXd A(system_->getStateDim(), system_->getStateDim());
    Eigen::MatrixXd B(system_->getStateDim(), system_->getControlDim());
    Eigen::VectorXd l_x(system_->getStateDim());
    Eigen::VectorXd l_u(system_->getControlDim());
    Eigen::MatrixXd l_xx(system_->getStateDim(), system_->getStateDim());
    Eigen::MatrixXd l_uu(system_->getControlDim(), system_->getControlDim());
    Eigen::MatrixXd l_ux(system_->getControlDim(), system_->getStateDim());
    Eigen::VectorXd Q_x(system_->getStateDim());
    Eigen::VectorXd Q_u(system_->getControlDim());
    Eigen::MatrixXd Q_xx(system_->getStateDim(), system_->getStateDim());
    Eigen::MatrixXd Q_uu(system_->getControlDim(), system_->getControlDim());
    Eigen::MatrixXd Q_ux(system_->getControlDim(), system_->getStateDim());
    Eigen::MatrixXd Q_uu_inv(system_->getControlDim(), system_->getControlDim());
    Eigen::MatrixXd active_constraint_tabl(2 * (system_->getControlDim()), horizon_); 
    Eigen::VectorXd k(system_->getControlDim());
    Eigen::MatrixXd K(system_->getControlDim(), system_->getStateDim());

    // Backward Riccati recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get state and control
        const Eigen::VectorXd& x = X_.at(t);
        const Eigen::VectorXd& u = U_.at(t);

        // Get continuous dynamics Jacobians
        auto [Fx, Fu] = system_->getJacobians(x, u);

        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx; 
        A.diagonal().array() += 1.0; // More efficient way to add identity
        B = timestep_ * Fu;

        // Get cost and its derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);


        // Compute Q-function matrices 
        Q_x = l_x + A.transpose() * V_X_[t + 1];
        Q_u = l_u + B.transpose() * V_X_[t + 1];
        Q_xx = l_xx + A.transpose() * V_XX_[t + 1] * A;
        Q_uu = l_uu + B.transpose() * V_XX_[t + 1] * B;
        Q_ux = l_ux + B.transpose() * V_XX_[t + 1] * A;

        // Symmetrize Q_uu
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            // Add regularization
            // Q_uu.diagonal() += 1e-6;
            std::cout << "Q_uu is not positive definite at " << t << std::endl;
        }

        // TODO: Regularization
        if (options_.regularization_type == 0) {
            Q_uu += options_.regularization_parameter * Eigen::MatrixXd::Identity(system_->getControlDim(), system_->getControlDim());
        } 

        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu);
        if (llt.info() != Eigen::Success) {
            // Decomposition failed
            std::cout << "Cholesky decomposition failed" << std::endl;
            return false;
        }

        /*  --- Identify Active Constraint --- */

        int active_constraint_index = 0;
        Eigen::MatrixXd C(system_->getControlDim(), system_->getControlDim()); // Control constraint matrix
        Eigen::MatrixXd D(system_->getControlDim(), system_->getStateDim()); // State constraint matrix

        // TODO: Implement active set method
        // for (int j = 0; j < system_->getControlDim(); j++) {
        //     if (u(j) < control_box_constraint->getLowerBound()(j) - active_set_tol) {
        //         Eigen::VectorXd e = Eigen::VectorXd::Zero(system_->getControlDim());
        //         e(j) = 1.0;
        //         C.row(active_constraint_index) = e;
        //         D.row(active_constraint_index) = Eigen::VectorXd::Zero(system_->getStateDim());
        //         active_constraint_index += 1;
        //     } else if (u(j) > control_box_constraint->getUpperBound()(j) + active_set_tol) {
        //         Eigen::VectorXd e = Eigen::VectorXd::Zero(system_->getControlDim());
        //         e(j) = -1.0;
        //         C.row(active_constraint_index) = e;
        //         D.row(active_constraint_index) = Eigen::VectorXd::Zero(system_->getStateDim());
        //         active_constraint_index += 1;
        //     }
        // }

        // Store Q-function matrices
        Q_UU_[t] = Q_uu;
        Q_UX_[t] = Q_ux;
        Q_U_[t] = Q_u;

        // Compute gains
        k = -Q_uu.inverse() * Q_u;
        K = -Q_uu.inverse() * Q_ux;

        k_[t] = k;
        K_[t] = K;

        // Compute value function approximation
        // V_[t] = l + V_[t + 1] + 0.5 * k_[t].transpose() * Q_UU_[t] * k_[t] + k_[t].transpose() * Q_U_[t];
        V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
    }

    return true;
}


// Forward pass
bool CDDP::solveForwardPass() {
    bool is_feasible = false;

    // 

    // Extract control box constraint
    auto control_box_constraint = constraint_set_.find("ControlBoxConstraint");

    int iter = 0;

    double alpha = options_.backtracking_coeff;

    // Line-search iteration 
    for (iter = 0; iter < options_.max_line_search_iterations; ++iter) {
        // Initialize cost and constraints
        double J = 0.0;
        std::vector<Eigen::VectorXd> X_new(horizon_ + 1, Eigen::VectorXd::Zero(system_->getStateDim()));
        std::vector<Eigen::VectorXd> U_new(horizon_, Eigen::VectorXd::Zero(system_->getControlDim()));
        X_new[0] = initial_state_;

        for (int t = 0; t < horizon_; ++t) {
            // Get state and control
            const Eigen::VectorXd& x = X_new[t];
            const Eigen::VectorXd& u = U_new[t];

            Eigen::VectorXd dx = x - X_[t];

            // Extract Q-function matrices
            const Eigen::VectorXd& Q_u = Q_U_[t];
            const Eigen::MatrixXd& Q_uu = Q_UU_[t];
            const Eigen::MatrixXd& Q_ux = Q_UX_[t];

            // Extract gains
            const Eigen::VectorXd& k = k_[t];
            const Eigen::MatrixXd& K = K_[t];

            // Create QP problem

            // Create OSQP solver


            // Solve QP problem

            // Extract control


            U_new[t] = u + alpha * k + K * dx;
            

            // Compute new state
            X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);

            // Compute cost
            J += objective_->running_cost(X_new[t + 1], U_new[t], t);
        }
        J += objective_->terminal_cost(X_new.back());
        std::cout << "Cost: " << J << std::endl;

        // // Check constraints
        // is_feasible = true;
        // for (int t = 0; t < horizon_; ++t) {
        //     const Eigen::VectorXd& u = U_new[t];
        //     // Check control box constraint
        //     if (control_box_constraint != constraint_set_.end()) {
        //         if (!control_box_constraint->second->isFeasible(u)) {
        //             is_feasible = false;
        //             break;
        //         }
        //     }
        // }

        // Check if the cost is reduced
        if (J < J_) {
            // Update state and control
            X_ = X_new;
            U_ = U_new;
            J_ = J;
            return true;
        } 
    }

    return true; // Or false if forward pass fails
}

// Helper methods

void CDDP::printSolverInfo() {
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

void CDDP::printOptions(const CDDPOptions& options) {
    std::cout << "\n========================================\n";
    std::cout << "           CDDP Options\n";
    std::cout << "========================================\n";

    std::cout << "Cost Tolerance: " << std::setw(10) << options.cost_tolerance << "\n";
    std::cout << "Grad Tolerance: " << std::setw(10) << options.grad_tolerance << "\n";
    std::cout << "Max Iterations: " << std::setw(10) << options.max_iterations << "\n";

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
    std::cout << "  Type: " << std::setw(10) << options.regularization_type << "\n";
    std::cout << "  State: " << std::setw(10) << options.regularization_x << "\n";
    std::cout << "  Control: " << std::setw(10) << options.regularization_u << "\n";
    std::cout << "  Tolerance: " << std::setw(10) << options.regularization_tolerance << "\n";
    std::cout << "  Factor: " << std::setw(10) << options.regularization_factor << "\n";
    std::cout << "  Max: " << std::setw(10) << options.regularization_max << "\n";
    std::cout << "  Min: " << std::setw(10) << options.regularization_min << "\n";

    std::cout << "\nOther:\n";
    std::cout << "  Print Iterations: " << (options.verbose ? "Yes" : "No") << "\n";
    std::cout << "  iLQR: " << (options.is_ilqr ? "Yes" : "No") << "\n";

    std::cout << "========================================\n\n";
}

void CDDP::printIteration(int iter, double cost, double grad_norm, double lambda) {
    std::cout << "Iteration: " << std::setw(5) << iter << " | ";
    std::cout << "Cost: " << std::setprecision(6) << std::setw(10) << cost << " | ";
    std::cout << "Grad Norm: " << std::setprecision(6) << std::setw(10) << grad_norm << " | ";
    std::cout << "Lambda: " << std::setprecision(6) << std::setw(10) << lambda << "\n";
}

void CDDP::printSolution(const CDDPSolution& solution) {
    std::cout << "\n========================================\n";
    std::cout << "           CDDP Solution\n";
    std::cout << "========================================\n";

    std::cout << "Converged: " << (solution.converged ? "Yes" : "No") << "\n";
    std::cout << "Iterations: " << solution.iterations << "\n";
    std::cout << "Solve Time: " << std::setprecision(4) << solution.solve_time << " micro sec\n";
    std::cout << "Final Cost: " << std::setprecision(6) << solution.cost_sequence.back() << "\n"; // Assuming cost_sequence is not empty

    std::cout << "========================================\n\n";
}
} // namespace cddp