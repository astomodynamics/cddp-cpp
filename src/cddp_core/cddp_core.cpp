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
#include <cmath>  // For std::log
#include <Eigen/Dense>
#include <chrono> // For timing
#include <execution> // For parallel execution policies
#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"
// TODO: gurobi solver compatibility
// #include "gurobi_c++.h"

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
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

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
        X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
        U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    } else if (X_.size() != horizon_ + 1) {
        X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
    } else if (U_.size() != horizon_) {
        U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    }

    // Set initial state
    X_[0] = initial_state_;

    // Initialize cost
    J_ = 0.0;

    barrier_coeff_ = options_.barrier_coeff;

    regularization_state_ = options_.regularization_state;
    regularization_state_step_ = options_.regularization_state_step;
    regularization_control_ = options_.regularization_control;
    regularization_control_step_ = options_.regularization_control_step;

    // Initialize gains and value function approximation
    k_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    K_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
    dV_.resize(2);
    V_X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
    V_XX_.resize(horizon_ + 1, Eigen::MatrixXd::Zero(state_dim, state_dim));

    // Initialize Q-function matrices
    Q_UU_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, control_dim));
    Q_UX_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
    Q_U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));

    // Initialize constraints if empty
    if (constraint_set_.empty()) {
        std::cerr << "CDDP: No constraints are set" << std::endl;
    } // if control constraints are set
    else if (constraint_set_.find("ControlBoxConstraint") != constraint_set_.end()) {
        std::cout << "ControlBoxConstraint is set" << std::endl;
    }
     
    // Initialize OSQP setting
     osqp::OsqpSettings settings;
    settings.warm_start = true;
    settings.verbose = false;
    // settings.max_iter = 1000;
    settings.eps_abs = 1e-3;
    settings.eps_rel = 1e-2;
    // settings.eps_prim_inf = 1e-4;
    // settings.eps_dual_inf = 1e-4;
    // settings.alpha = 1.6;
    
    // Initialize QP solver instance
    osqp::OsqpInstance instance;

    Eigen::SparseMatrix<double> P(control_dim, control_dim);
    P.setIdentity();
    P.makeCompressed();
    instance.objective_matrix = P;

    instance.objective_vector = Eigen::VectorXd::Zero(control_dim);

    Eigen::SparseMatrix<double> A(control_dim, control_dim);
    A.setIdentity();
    A.makeCompressed();
    instance.constraint_matrix = A;
    
    instance.lower_bounds = Eigen::VectorXd::Constant(control_dim, -std::numeric_limits<double>::infinity());
    instance.upper_bounds = Eigen::VectorXd::Constant(control_dim, std::numeric_limits<double>::infinity());

    // Initialize the solver
    osqp_solver_.Init(instance, settings);

    // Ceck if the problem is initialized correctly
    if (osqp_solver_.IsInitialized()) {
        // std::cout << "OSQP Solver is initialized" << std::endl;
    } else {
        std::cerr << "OSQP Solver is not initialized" << std::endl;
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
    solution.time_sequence.reserve(horizon_ + 1);
    for (int t = 0; t <= horizon_; ++t) {
        solution.time_sequence.push_back(timestep_ * t);
    }
    solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency
    solution.lagrangian_sequence.reserve(options_.max_iterations); // Reserve space for efficiency

    // Evaluate initial cost
    J_ = objective_->evaluate(X_, U_);
    solution.cost_sequence.push_back(J_);
    
    // Evaluate Lagrangian
    L_ = J_;

    // // Loop over horizon 
    // for (int t = 0; t < 1; ++t) {
    //     // Evaluate state constraint violation
    //     for (const auto& constraint : constraint_set_) {
    //         if (constraint.first == "ControlBoxConstraint") {
    //             L_ += getLogBarrierCost(*constraint.second, X_[t], U_[t], barrier_coeff_, options_.relaxation_coeff);
    //             // Eigen::VectorXd constraint_violation = constraint.second->evaluate(X_[t], U_[t]);
    //             // if (constraint_violation.minCoeff() < 0) {
    //             //     std::cerr << "CDDP: Constraint violation at time " << t << std::endl;
    //             //     std::cerr << "Constraint violation: " << constraint_violation.transpose() << std::endl;
    //             //     throw std::runtime_error("Constraint violation");
    //             // }
    //         }

    //     }
    // }

    if (options_.verbose) {
        printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_); // Initial iteration information
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main loop of CDDP
    for (int iter = 1; iter <= options_.max_iterations; ++iter) {
        // Check maximum CPU time
        if (options_.max_cpu_time > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
            if (duration.count() * 1e-6 > options_.max_cpu_time) {
                std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                break;
            }
        }

        // 1. Backward pass: Solve Riccati recursion to compute optimal control law
        bool backward_pass_success = false;
        while (!backward_pass_success) {
            backward_pass_success = solveBackwardPass();

            if (!backward_pass_success) {
                std::cerr << "CDDP: Backward pass failed" << std::endl;

                // Increase regularization
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);

                if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                    std::cerr << "CDDP: Regularization limit reached" << std::endl;
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }
        
        // Check termination due to small cost improvement
        if (optimality_gap_ < options_.grad_tolerance && regularization_state_ < 1e-4 && regularization_control_ < 1e-4) {
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_;
            if (regularization_state_ <= options_.regularization_state_min) {
                regularization_state_ = 0.0;
            }
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_;
            if (regularization_control_ <= options_.regularization_control_min) {
                regularization_control_ = 0.0;
            }

            solution.converged = true;
            break;
        }

        bool forward_pass_success = false;
        // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
        if (backward_pass_success) {
            forward_pass_success = solveForwardPass();
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_); 
        }
       
       if (forward_pass_success) {
            // Decrease regularization
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_;
            if (regularization_state_ <= options_.regularization_state_min) {
                regularization_state_ = 0.0;
            }
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_;
            if (regularization_control_ <= options_.regularization_control_min) {
                regularization_control_ = 0.0;
            }

            if (dJ_ < options_.cost_tolerance) {
                solution.converged = true;
                // Append Latest Cost
                solution.cost_sequence.push_back(J_);
                solution.iterations = iter;
                break;
            }
        }
    }

    // Finalize solution
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
    // Initialize variables
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    const auto active_set_tol = options_.active_set_tolerance;

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Terminal cost and its derivatives
    V_X_.back() = objective_->getFinalCostGradient(X_.back());
    V_XX_.back() = objective_->getFinalCostHessian(X_.back());

    // Pre-allocate matrices
    Eigen::MatrixXd A(state_dim, state_dim);
    Eigen::MatrixXd B(state_dim, control_dim);
    Eigen::VectorXd Q_x(state_dim);
    Eigen::VectorXd Q_u(control_dim);
    Eigen::MatrixXd Q_xx(state_dim, state_dim);
    Eigen::MatrixXd Q_uu(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
    Eigen::MatrixXd active_constraint_tabl(2 * (control_dim), horizon_); 
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);

    double Qu_error = 0.0;

    // Backward Riccati recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get state and control
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Extract value function approximation
        const Eigen::VectorXd& V_x = V_X_[t + 1];
        const Eigen::MatrixXd& V_xx = V_XX_[t + 1];

        // Get continuous dynamics Jacobians
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx; 
        A.diagonal().array() += 1.0; // More efficient way to add identity
        B = timestep_ * Fu;

        // Get cost and its derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // TODO: Implement log barrier cost and its derivatives // FIXME: Implement log barrier cost and its derivatives
        // Get log barrier cost and its derivatives
        for (const auto& constraint : constraint_set_) {
            if (constraint.first == "ControlBoxConstraint") {
                const double barrier_cost = getLogBarrierCost(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
                l += barrier_cost;
                const auto [l_x_barrier, l_u_barrier] = getLogBarrierCostGradients(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
                const auto [l_xx_barrier, l_uu_barrier, l_ux_barrier] = getLogBarrierCostHessians(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
                l_x += l_x_barrier;
                l_u += l_u_barrier;
                // l_xx += l_xx_barrier;
                // l_uu += l_uu_barrier;
                // l_ux += l_ux_barrier;
            }
        }

        // Compute Q-function matrices 
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;

        if (options_.regularization_type == "state" || options_.regularization_type == "both") {
            Q_ux = l_ux + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * A;
            Q_uu = l_uu + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * B;
        } else {
            Q_ux = l_ux + B.transpose() * V_xx * A;
            Q_uu = l_uu + B.transpose() * V_xx * B;
        }

        // Symmetrize Q_uu for cholensky decomposition
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Control Regularization
        if (options_.regularization_type == "control" || options_.regularization_type == "both") {
            Q_uu += options_.regularization_control * Eigen::MatrixXd::Identity(control_dim, control_dim);
        } 

        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            // Add regularization
            // Q_uu.diagonal() += 1e-6;
            std::cout << "Q_uu is not positive definite at " << t << std::endl;

            eigenvalues = es.eigenvalues().real();
            if (eigenvalues.minCoeff() <= 0) {
                std::cout << "Q_uu is still not positive definite" << std::endl;
                return false;
            }
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
        Eigen::MatrixXd C(control_dim, control_dim); // Control constraint matrix
        Eigen::MatrixXd D(control_dim, state_dim); // State constraint matrix

        // TODO: Implement active set method
        for (int j = 0; j < control_dim; j++) {
            if (u(j) < control_box_constraint.getLowerBound()(j) - active_set_tol) {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
                e(j) = 1.0;
                C.row(active_constraint_index) = e;
                D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
                active_constraint_index += 1;
            } else if (u(j) > control_box_constraint.getUpperBound()(j) + active_set_tol) {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
                e(j) = -1.0;
                C.row(active_constraint_index) = e;
                D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
                active_constraint_index += 1;
            }
        }

        Eigen::MatrixXd active_constraint_table = Eigen::MatrixXd::Zero(2 * (control_dim), horizon_);
        if (active_constraint_index == 0) {

            Eigen::MatrixXd H = Q_uu.inverse();
            // Feedback Gain Calculation 
            K = -H * Q_ux;
            k = -H * Q_u;
        } else {
            // Shrink C and D matrices
            C.conservativeResize(active_constraint_index, control_dim);
            D.conservativeResize(active_constraint_index, state_dim);

            // Calculate Lagrange Multipliers
            Eigen::MatrixXd Q_uu_inv = Q_uu.inverse();
            Eigen::MatrixXd lambda = (C * Q_uu_inv * C.transpose()).inverse() * C * Q_uu_inv * Q_u;

            // Remove active constraints if lambda is negative
            active_constraint_index = 0;
            std::vector<int> deleted_index_list;

            for (int j = 0; j < control_dim; j++) {
                if (active_constraint_table(j) == 1 && lambda(active_constraint_index) < 0) {
                    C.row(active_constraint_index) = Eigen::VectorXd::Zero(control_dim);
                    active_constraint_table(j, t) = 0;
                    deleted_index_list.push_back(active_constraint_index);
                    active_constraint_index += 1;
                } else if (active_constraint_table(j + control_dim, t) == 1 && lambda(active_constraint_index) < 0) {
                    C.row(active_constraint_index) = Eigen::VectorXd::Zero(control_dim);
                    active_constraint_table(j, t) = 0;
                    deleted_index_list.push_back(active_constraint_index);
                    active_constraint_index += 1;
                }
            }


            Eigen::MatrixXd C_shrinked = Eigen::MatrixXd::Zero(C.rows() - deleted_index_list.size(), control_dim);
            Eigen::MatrixXd D_shrinked = Eigen::MatrixXd::Zero(C.rows() - deleted_index_list.size(),  state_dim);

            // Shrink C and D matrices by taking out negative lambda
            for (int j = 0; j < C.rows(); j++) {
                if (std::find(deleted_index_list.begin(), deleted_index_list.end(), j) == deleted_index_list.end()) {
                    C_shrinked.row(j) = C.row(j);
                    D_shrinked.row(j) = D.row(j);
                }
            }

            // Feedback Gain Calculation
            Eigen::MatrixXd W = (C_shrinked * Q_uu_inv * C_shrinked.transpose()).inverse() * C_shrinked * Q_uu_inv;
            Eigen::MatrixXd H = Q_uu_inv * (Eigen::MatrixXd::Identity(control_dim, control_dim) - C_shrinked.transpose() * W);
            k = -H * Q_u;
            K = -H * Q_ux;
        }

        // Store Q-function matrices
        Q_UU_[t] = Q_uu;
        Q_UX_[t] = Q_ux;
        Q_U_[t] = Q_u;

        // Compute value function approximation
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ = dV_ + dV_step;
        V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_XX_[t] = 0.5 * (V_XX_[t] + V_XX_[t].transpose()); // Symmetrize Hessian

        // Compute optimality gap (Inf-norm) for convergence check
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

        // TODO: Add constraint optimality gap analysis
        optimality_gap_ = Qu_error;
    }

    return true;
}


// Forward pass
bool CDDP::solveForwardPass() {
    bool is_feasible = false;
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    int iter = 0;
    double alpha = options_.backtracking_coeff;
    
    // Pre-allocate matrices
    Eigen::SparseMatrix<double> P(control_dim, control_dim); // Hessian of QP objective
    
    // A is already defined in initializeCDDP
    // Eigen::SparseMatrix<double> A(state_dim, control_dim);
    // A.setIdentity();
    // A.makeCompressed();
    // osqp_solver_.UpdateConstraintMatrix(A);

    // Line-search iteration 
    for (iter = 0; iter < options_.max_line_search_iterations; ++iter) {
        // Initialize cost and constraints
        double J_new = 0.0, dJ = 0.0, expected_dV = 0.0, gradient_norm = 0.0;
        double L_new = 0.0;

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        X_new[0] = initial_state_;

        for (int t = 0; t < horizon_; ++t) {
            // Get state and control
            const Eigen::VectorXd& x = X_new[t];
            const Eigen::VectorXd& u = U_new[t];

            // Deviation from the nominal trajectory
            const Eigen::VectorXd& delta_x = x - X_[t];

            // Extract Q-function matrices
            const Eigen::VectorXd& Q_u = Q_U_[t];
            const Eigen::MatrixXd& Q_uu = Q_UU_[t];
            const Eigen::MatrixXd& Q_ux = Q_UX_[t];

            // Create QP problem
            int numNonZeros = Q_uu.nonZeros(); 
            P.reserve(numNonZeros);
            for (int i = 0; i < Q_uu.rows(); ++i) {
                for (int j = 0; j < Q_uu.cols(); ++j) {
                    if (Q_uu(i, j) != 0) {
                        P.insert(i, j) = Q_uu(i, j);
                    }
                }
            }
            P.makeCompressed(); // Important for efficient storage and operations
            osqp_solver_.UpdateObjectiveMatrix(P);
            
            const Eigen::VectorXd& q = alpha * Q_u + Q_ux * delta_x; // Gradient of QP objective
            osqp_solver_.SetObjectiveVector(q);  

            // Lower and upper bounds
            Eigen::VectorXd lb = 1.0 * (control_box_constraint.getLowerBound() - u);
            Eigen::VectorXd ub = 1.0 * (control_box_constraint.getUpperBound() - u);    
            osqp_solver_.SetBounds(lb, ub);

            // Solve the QP problem TODO: Use SDQP instead of OSQP
            osqp::OsqpExitCode exit_code = osqp_solver_.Solve();

            if (exit_code == osqp::OsqpExitCode::kOptimal) {
                is_feasible = true;
            } else {
                is_feasible = false;
                alpha *= options_.backtracking_factor;
                continue;
            }

            // Extract solution
            double optimal_objective = osqp_solver_.objective_value();
            const Eigen::VectorXd& delta_u = osqp_solver_.primal_solution();

            

            // Extract solution
            U_new[t] += delta_u;

            // Compute cost
            J_new += objective_->running_cost(x, U_new[t], t);

            // Compute new state
            X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);

        }
        J_new += objective_->terminal_cost(X_new.back());

        // Calculate Cost Reduction
        dJ = J_ - J_new;

        double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
        double reduction_ratio = 0.0;
        if (expected > 0.0) {
            reduction_ratio = dJ / expected;
        } else {
            reduction_ratio = std::copysign(1.0, dJ);
            std::cout << "Expected improvement is not positive" << std::endl;
        }
        if (reduction_ratio > options_.minimum_reduction_ratio) {
            // Update state and control
            X_ = X_new;
            U_ = U_new;
            J_ = J_new;
            dJ_ = dJ;
            barrier_coeff_ = options_.barrier_coeff / 10.0;
            return true;
        } else {
            alpha *= options_.backtracking_factor;
        }


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
    }

    return false; // Or false if forward pass fails
}

// Helper methods
double CDDP::getLogBarrierCost(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    Eigen::VectorXd lower_bound = constraint.getLowerBound();
    Eigen::VectorXd upper_bound = constraint.getUpperBound();

    double barrier_cost = 0.0;
    for (int i = 0; i < constraint_value.size(); ++i) {
        // If Constraint is ControlBoxConstraint
        if (constraint.getName() == "ControlBoxConstraint") {
            
            // Calculate the log barrier term for each constraint component
            double upper = upper_bound(i) - constraint_value(i); // 
            double lower = constraint_value(i) - lower_bound(i);

            if (upper > relaxation_coeff) {
                barrier_cost -= barrier_coeff * std::log(upper);
            } else {
                barrier_cost += 0.5 * (std::pow((upper - 2 * relaxation_coeff) / relaxation_coeff, 2) - 1.0) - std::log(relaxation_coeff);
            }

            if (lower > -relaxation_coeff) {
                barrier_cost -= barrier_coeff * std::log(lower);
            } else {
                barrier_cost += 0.5 * (std::pow((lower - 2 * relaxation_coeff) / relaxation_coeff, 2) - 1.0) - std::log(relaxation_coeff);
            }
        


        } else {
            // If Constraint is not ControlBoxConstraint
            if (constraint_value(i) < lower_bound(i)) {
                // Constraint is violated, return a large cost
                return std::numeric_limits<double>::infinity();
            } else {
                // Calculate the log barrier term for each constraint component
                barrier_cost -= barrier_coeff * std::log(lower_bound(i) - constraint_value(i));
            }
        }
    }

    return barrier_cost;
}

Eigen::VectorXd CDDP::getLogBarrierCostStateGradient(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
    Eigen::VectorXd barrier_cost_grad = Eigen::VectorXd::Zero(state.size());
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    Eigen::VectorXd lower_bound = constraint.getLowerBound();
    Eigen::VectorXd upper_bound = constraint.getUpperBound();

    for (int i = 0; i < constraint_value.size(); ++i) {
        // Calculate the log barrier term for each constraint component
        double upper = upper_bound(i) - constraint_value(i);
        double lower = constraint_value(i) - lower_bound(i);

        if (upper > relaxation_coeff) {
            barrier_cost_grad(i) = -barrier_coeff / upper;
        } else {
            barrier_cost_grad(i) = -(upper - 2 * relaxation_coeff) / (relaxation_coeff * relaxation_coeff);
        }

        if (lower > -relaxation_coeff) {
            barrier_cost_grad(i) = -barrier_coeff / lower;
        } else {
            barrier_cost_grad(i) = -(lower - 2 * relaxation_coeff) / (relaxation_coeff * relaxation_coeff);
        }
    }

    return barrier_cost_grad;
}

Eigen::VectorXd CDDP::getLogBarrierCostControlGradient(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
    Eigen::VectorXd barrier_cost_grad = Eigen::VectorXd::Zero(control.size());

    if (constraint.getName() == "ControlBoxConstraint") {
        Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
        Eigen::VectorXd lower_bound = constraint.getLowerBound();
        Eigen::VectorXd upper_bound = constraint.getUpperBound();

        for (int i = 0; i < constraint_value.size(); ++i) {
            // Calculate the log barrier term for each constraint component
            double upper = upper_bound(i) - constraint_value(i);
            double lower = constraint_value(i) - lower_bound(i);

            if (upper > relaxation_coeff) {
                barrier_cost_grad(i) = barrier_coeff / upper;
            } else {
                barrier_cost_grad(i) = (upper - 2 * relaxation_coeff) / (relaxation_coeff * relaxation_coeff);
            }

            if (lower > -relaxation_coeff) {
                barrier_cost_grad(i) = barrier_coeff / lower;
            } else {
                barrier_cost_grad(i) = (lower - 2 * relaxation_coeff) / (relaxation_coeff * relaxation_coeff);
            }
        }
    }

    return barrier_cost_grad;
}

Eigen::MatrixXd CDDP::getLogBarrierCostStateHessian(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
    Eigen::MatrixXd barrier_cost_hess = Eigen::MatrixXd::Zero(state.size(), state.size());
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    Eigen::VectorXd lower_bound = constraint.getLowerBound();
    Eigen::VectorXd upper_bound = constraint.getUpperBound();

    // for (int i = 0; i < constraint_value.size(); ++i) {
    //     // Calculate the log barrier term for each constraint component
    //     double upper = upper_bound(i) - constraint_value(i);
    //     double lower = constraint_value(i) - lower_bound(i);

    //     if (upper > relaxation_coeff) {
    //         barrier_cost_hess(i, i) = barrier_coeff / (upper * upper);
    //     } else {
    //         barrier_cost_hess(i, i) = 1.0 / (relaxation_coeff * relaxation_coeff);
    //     }

    //     if (lower > -relaxation_coeff) {
    //         barrier_cost_hess(i, i) = barrier_coeff / (lower * lower);
    //     } else {
    //         barrier_cost_hess(i, i) = 1.0 / (relaxation_coeff * relaxation_coeff);
    //     }
    // }

    return barrier_cost_hess;
}

Eigen::MatrixXd CDDP::getLogBarrierCostControlHessian(const Constraint& constraint, const Eigen::VectorXd& state, const Eigen::VectorXd& control, double barrier_coeff, double relaxation_coeff) {
    Eigen::MatrixXd barrier_cost_hess = Eigen::MatrixXd::Zero(state.size(), state.size());
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    Eigen::VectorXd lower_bound = constraint.getLowerBound();
    Eigen::VectorXd upper_bound = constraint.getUpperBound();

    for (int i = 0; i < constraint_value.size(); ++i) {
        // Calculate the log barrier term for each constraint component
        double upper = upper_bound(i) - constraint_value(i);
        double lower = constraint_value(i) - lower_bound(i);

        if (upper > relaxation_coeff) {
            barrier_cost_hess(i, i) = barrier_coeff / (upper * upper);
        } else {
            barrier_cost_hess(i, i) = 1.0 / (relaxation_coeff * relaxation_coeff);
        }

        if (lower > -relaxation_coeff) {
            barrier_cost_hess(i, i) = barrier_coeff / (lower * lower);
        } else {
            barrier_cost_hess(i, i) = 1.0 / (relaxation_coeff * relaxation_coeff);
        }
    }

    return barrier_cost_hess;
}



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

    std::cout << "========================================\n\n";
}

void CDDP::printIteration(int iter, double cost, double lagrangian, double grad_norm, double lambda_state, double lambda_control) {
    std::cout << "Iteration: " << std::setw(5) << iter << " | ";
    std::cout << "Cost: " << std::setprecision(6) << std::setw(10) << cost << " | ";
    std::cout << "Lagrangian: " << std::setprecision(6) << std::setw(10) << lagrangian << " | ";
    std::cout << "Grad Norm: " << std::setprecision(6) << std::setw(10) << grad_norm << " | ";
    std::cout << "Reg (state): " << std::setprecision(6) << std::setw(10) << lambda_state << "\n";
    std::cout << "Reg (control): " << std::setprecision(6) << std::setw(10) << lambda_control << "\n";
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



CDDPSolution CDDP::solveCLDDP() {
    // Initialize CDDP solver
    initializeCDDP();

    if (options_.verbose) {
        printOptions(options_);
    }

    // Initialize solution
    CDDPSolution solution;
    solution.converged = false;
    solution.time_sequence.reserve(horizon_ + 1);
    for (int t = 0; t <= horizon_; ++t) {
        solution.time_sequence.push_back(timestep_ * t);
    }
    solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency

    // Evaluate initial cost
    J_ = objective_->evaluate(X_, U_);
    solution.cost_sequence.push_back(J_);

    if (options_.verbose) {
        printIteration(0, J_, 0.0, optimality_gap_, regularization_state_, regularization_control_); // Initial iteration information
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main loop of CDDP
    for (int iter = 1; iter <= options_.max_iterations; ++iter) {
        // Check maximum CPU time
        if (options_.max_cpu_time > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
            if (duration.count() * 1e-6 > options_.max_cpu_time) {
                std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                break;
            }
        }

        // 1. Backward pass: Solve Riccati recursion to compute optimal control law
        bool backward_pass_success = false;
        while (!backward_pass_success) {
            backward_pass_success = solveCLDDPBackwardPass();

            if (!backward_pass_success) {
                std::cerr << "CDDP: Backward pass failed" << std::endl;

                // Increase regularization
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);

                if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                    std::cerr << "CDDP: Regularization limit reached" << std::endl;
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }
        
        // Check termination due to small cost improvement
        if (optimality_gap_ < options_.grad_tolerance && regularization_state_ < 1e-4 && regularization_control_ < 1e-4) {
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_;
            if (regularization_state_ <= options_.regularization_state_min) {
                regularization_state_ = 0.0;
            }
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_;
            if (regularization_control_ <= options_.regularization_control_min) {
                regularization_control_ = 0.0;
            }

            solution.converged = true;
            break;
        }

        bool forward_pass_success = false;
        // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
        if (backward_pass_success) {
            forward_pass_success = solveCLDDPForwardPass();
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, 0.0, optimality_gap_, regularization_state_, regularization_control_); 
        }
       
       if (forward_pass_success) {
            // Decrease regularization
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_;
            if (regularization_state_ <= options_.regularization_state_min) {
                regularization_state_ = 0.0;
            }
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_;
            if (regularization_control_ <= options_.regularization_control_min) {
                regularization_control_ = 0.0;
            }

            if (dJ_ < options_.cost_tolerance) {
                solution.converged = true;
                // Append Latest Cost
                solution.cost_sequence.push_back(J_);
                solution.iterations = iter;
                break;
            }
        }
    }

    // Finalize solution
    solution.control_sequence = U_;
    solution.state_sequence = X_;
    solution.cost_sequence.push_back(J_);
    solution.iterations = solution.converged ? solution.iterations : options_.max_iterations;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
    solution.solve_time = duration.count(); // Time in microseconds
    printSolution(solution);

    return solution;
}

bool CDDP::solveCLDDPBackwardPass() {
    // Initialize variables
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    const auto active_set_tol = options_.active_set_tolerance;

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Terminal cost and its derivatives
    V_X_.back() = objective_->getFinalCostGradient(X_.back());
    V_XX_.back() = objective_->getFinalCostHessian(X_.back());

    // Pre-allocate matrices
    Eigen::MatrixXd A(state_dim, state_dim);
    Eigen::MatrixXd B(state_dim, control_dim);
    Eigen::VectorXd Q_x(state_dim);
    Eigen::VectorXd Q_u(control_dim);
    Eigen::MatrixXd Q_xx(state_dim, state_dim);
    Eigen::MatrixXd Q_uu(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);
    Eigen::SparseMatrix<double> P(control_dim, control_dim); // Hessian of QP objective

    double Qu_error = 0.0;

    // Backward Riccati recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get state and control
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Extract value function approximation
        const Eigen::VectorXd& V_x = V_X_[t + 1];
        const Eigen::MatrixXd& V_xx = V_XX_[t + 1];

        // Get continuous dynamics Jacobians
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx; 
        A.diagonal().array() += 1.0; // More efficient way to add identity
        B = timestep_ * Fu;

        // Get cost and its derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // Compute Q-function matrices 
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;

        if (options_.regularization_type == "state" || options_.regularization_type == "both") {
            Q_ux = l_ux + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * A;
            Q_uu = l_uu + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * B;
        } else {
            Q_ux = l_ux + B.transpose() * V_xx * A;
            Q_uu = l_uu + B.transpose() * V_xx * B;
        }

        // Symmetrize Q_uu for cholensky decomposition
        Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Control Regularization
        if (options_.regularization_type == "control" || options_.regularization_type == "both") {
            Q_uu += options_.regularization_control * Eigen::MatrixXd::Identity(control_dim, control_dim);
        } 

        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            // Add regularization
            // Q_uu.diagonal() += 1e-6;
            std::cout << "Q_uu is not positive definite at " << t << std::endl;

            eigenvalues = es.eigenvalues().real();
            if (eigenvalues.minCoeff() <= 0) {
                std::cout << "Q_uu is still not positive definite" << std::endl;
                return false;
            }
        }

        // Store Q-function matrices
        Q_UU_[t] = Q_uu;
        Q_UX_[t] = Q_ux;
        Q_U_[t] = Q_u;

        /*    Solve Box QP Problem    */   
        int numNonZeros = Q_uu.nonZeros(); 
        P.reserve(numNonZeros);
        for (int i = 0; i < Q_uu.rows(); ++i) {
            for (int j = 0; j < Q_uu.cols(); ++j) {
                if (Q_uu(i, j) != 0) {
                    P.insert(i, j) = Q_uu(i, j);
                }
            }
        }
        P.makeCompressed(); // Important for efficient storage and operations
        osqp_solver_.UpdateObjectiveMatrix(P);

        const Eigen::VectorXd& q = Q_u; // Gradient of QP objective
        osqp_solver_.SetObjectiveVector(q);  

        // Lower and upper bounds
        Eigen::VectorXd lb = 1.0 * (control_box_constraint.getLowerBound() - u);
        Eigen::VectorXd ub = 1.0 * (control_box_constraint.getUpperBound() - u);    
        osqp_solver_.SetBounds(lb, ub);

        // Solve the QP problem TODO: Use SDQP instead of OSQP
        osqp::OsqpExitCode exit_code = osqp_solver_.Solve();

        // Extract solution
        double optimal_objective = osqp_solver_.objective_value();
        k = osqp_solver_.primal_solution();

        // Compute feedback gain
        K = -Q_uu.inverse() * Q_ux;

        // Store feedforward and feedback gain
        k_[t] = k;
        K_[t] = K;

        // Compute value function approximation
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ = dV_ + dV_step;
        V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_XX_[t] = 0.5 * (V_XX_[t] + V_XX_[t].transpose()); // Symmetrize Hessian

        // Compute optimality gap (Inf-norm) for convergence check
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

        // TODO: Add constraint optimality gap analysis
        optimality_gap_ = Qu_error;
    }

    return true;
}

bool CDDP::solveCLDDPForwardPass() {
    bool is_feasible = false;
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    int iter = 0;
    double alpha = options_.backtracking_coeff;

    // Line-search iteration 
    for (iter = 0; iter < options_.max_line_search_iterations; ++iter) {
        // Initialize cost and constraints
        double J_new = 0.0, dJ = 0.0, expected_dV = 0.0, gradient_norm = 0.0;

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        X_new[0] = initial_state_;

        for (int t = 0; t < horizon_; ++t) {
            // Get state and control
            const Eigen::VectorXd& x = X_new[t];
            const Eigen::VectorXd& u = U_new[t];

            // Deviation from the nominal trajectory
            const Eigen::VectorXd& delta_x = x - X_[t];

            // Extract control laws
            const Eigen::VectorXd& k = k_[t];
            const Eigen::MatrixXd& K = K_[t];

            // Create a new solution
            U_new[t] = u + alpha * k + K * delta_x;

            // Compute cost
            J_new += objective_->running_cost(x, U_new[t], t);

            // Compute new state
            X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);

        }
        J_new += objective_->terminal_cost(X_new.back());

        // Calculate Cost Reduction
        dJ = J_ - J_new;

        double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));

        double reduction_ratio = 0.0;
        if (expected > 0.0) {
            reduction_ratio = dJ / expected;
        } else {
            reduction_ratio = std::copysign(1.0, dJ);
            std::cout << "Expected improvement is not positive" << std::endl;
        }

        if (reduction_ratio > options_.minimum_reduction_ratio) {
            // Update state and control
            X_ = X_new;
            U_ = U_new;
            J_ = J_new;
            dJ_ = dJ;
            return true;
        } else {
            alpha *= options_.backtracking_factor;
        }

    }
    return false;
}



} // namespace cddp