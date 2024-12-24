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
#include <iomanip>  // For std::setw
#include <memory>   // For std::unique_ptr
#include <map>      // For std::map
#include <cmath>    // For std::log
#include <Eigen/Dense>
#include <chrono>    // For timing
#include <execution> // For parallel execution policies

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/boxqp.hpp"

namespace cddp
{

// Constructor
CDDP::CDDP(const Eigen::VectorXd &initial_state,
            const Eigen::VectorXd &reference_state,
            int horizon,
            double timestep,
            std::unique_ptr<DynamicalSystem> system,
            std::unique_ptr<Objective> objective,
            const CDDPOptions &options)
    : initial_state_(initial_state),
        reference_state_(reference_state),
        horizon_(horizon),
        timestep_(timestep),
        system_(std::move(system)),
        objective_(std::move(objective)),
        options_(options),
        initialized_(false)
{
    initializeCDDP();
    if (options_.verbose) {
        printSolverInfo();
        printOptions(options_);
    }
}

void CDDP::setInitialTrajectory(const std::vector<Eigen::VectorXd> &X, const std::vector<Eigen::VectorXd> &U)
{
    if (!system_) {
        std::cerr << "CDDP::setInitialTrajectory: No dynamical system provided." << std::endl;
        throw std::runtime_error("CDDP::setInitialTrajectory: No dynamical system provided.");
    }

    if (!objective_) {
        std::cerr << "CDDP::setInitialTrajectory: No objective function provided." << std::endl;
        throw std::runtime_error("CDDP::setInitialTrajectory: No objective function provided.");
    }

    if (X.size() != horizon_ + 1)
    {
        std::cerr << "CDDP::setInitialTrajectory: X has wrong #timesteps" << std::endl;
        throw std::runtime_error("CDDP::setInitialTrajectory: X has wrong #timesteps");
    }
    if (U.size() != horizon_)
    {
        std::cerr << "CDDP::setInitialTrajectory: U has wrong #timesteps" << std::endl;
        throw std::runtime_error("CDDP::setInitialTrajectory: U has wrong #timesteps");
    }

    X_ = X;
    U_ = U;
    J_ = objective_->evaluate(X_, U_);
}

// Initialize the CDDP solver
void CDDP::initializeCDDP()
{
    if (initialized_)
    {
        // Already done—return.
        return;
    }

    if (!system_)
    {
        initialized_ = false;
        if (options_.verbose) {
            std::cerr << "CDDP::initializeCDDP: No dynamical system provided." << std::endl;
        }
        return;
    }

    if (!objective_)
    {
        initialized_ = false;
        if (options_.verbose) {
            std::cerr << "CDDP::initializeCDDP: No objective function provided." << std::endl;
        }
        return;
    }

    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Check if reference_state in objective and reference_state in CDDP are the same
    if ((reference_state_ - objective_->getReferenceState()).norm() > 1e-6)
    {
        std::cerr << "CDDP: Initial state and goal state in the objective function do not match" << std::endl;
        throw std::runtime_error("Initial state and goal state in the objective function do not match");
    }

    // Initialize trajectories (X_ and U_ are std::vectors of Eigen::VectorXd)
    if (X_.size() != horizon_ + 1 && U_.size() != horizon_)
    {
        X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
        U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    }
    else if (X_.size() != horizon_ + 1)
    {
        X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
    }
    else if (U_.size() != horizon_)
    {
        U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    }

    // Initialize cost
    J_ = objective_->evaluate(X_, U_);

    alpha_ = options_.backtracking_coeff;
    for (int i = 0; i < options_.max_line_search_iterations; ++i)
    {
        alphas_.push_back(alpha_);
        alpha_ *= options_.backtracking_factor;
    }
    alpha_ = options_.backtracking_coeff;

    if (options_.regularization_type == "state" || options_.regularization_type == "both")
    {
        regularization_state_ = options_.regularization_state;
        regularization_state_step_ = options_.regularization_state_step;
    } else {
        regularization_state_ = 0.0;
        regularization_state_step_ = 1.0;
    }
    
    if (options_.regularization_type == "control" || options_.regularization_type == "both")
    {
        regularization_control_ = options_.regularization_control;
        regularization_control_step_ = options_.regularization_control_step;
    } else {
        regularization_control_ = 0.0;
        regularization_control_step_ = 1.0;
    }


    // Initialize gains and value reduction
    k_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    K_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
    dV_.resize(2);

    // Initialize Q-function matrices
    Q_UU_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, control_dim));
    Q_UX_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
    Q_U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));

    // Check if ControlBoxConstraint is set
    if (constraint_set_.find("ControlBoxConstraint") != constraint_set_.end())
    {
        std::cout << "ControlBoxConstraint is set" << std::endl;
    }

    // Initialize Log-barrier object
    log_barrier_ = std::make_unique<LogBarrier>(options_.barrier_coeff, options_.relaxation_coeff, options_.barrier_order);

    // Initialize boxqp options
    boxqp_options_.max_iterations = options_.boxqp_max_iterations;
    boxqp_options_.min_grad = options_.boxqp_min_grad;
    boxqp_options_.min_rel_improve = options_.boxqp_min_rel_improve;    
    boxqp_options_.step_dec = options_.boxqp_step_dec;
    boxqp_options_.min_step = options_.boxqp_min_step;
    boxqp_options_.armijo = options_.boxqp_armijo;
    boxqp_options_.verbose = options_.boxqp_verbose;

    boxqp_solver_ = BoxQPSolver(boxqp_options_);
    
    initialized_ = true;
}


CDDPSolution CDDP::solve() {
    // Initialize if not done
    if (!initialized_) {
        initializeCDDP();
    }

    if (!initialized_) {
        std::cerr << "CDDP: Initialization failed" << std::endl;
        throw std::runtime_error("CDDP: Initialization failed");
    }

    // Prepare solution struct
    CDDPSolution solution;
    solution.converged = false;
    solution.alpha = alpha_;
    solution.iterations = 0;
    solution.solve_time = 0.0;
    solution.time_sequence.reserve(horizon_ + 1);
    for (int t = 0; t <= horizon_; ++t) {
        solution.time_sequence.push_back(timestep_ * t);
    }
    solution.control_sequence.reserve(horizon_);
    solution.state_sequence.reserve(horizon_ + 1);
    solution.cost_sequence.reserve(options_.max_iterations); // Reserve space for efficiency
    solution.lagrangian_sequence.reserve(options_.max_iterations); // Reserve space for efficiency
    solution.cost_sequence.push_back(J_);
    solution.lagrangian_sequence.push_back(L_);

    if (options_.verbose) {
        printIteration(0, J_, J_, optimality_gap_, regularization_state_, regularization_control_, alpha_); // Initial iteration information
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;

    // Main loop of CDDP
    while (iter < options_.max_iterations)
    {
        ++iter;
        solution.iterations = iter;

        // Check maximum CPU time
        if (options_.max_cpu_time > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 
            if (duration.count() * 1e-6 > options_.max_cpu_time) {
                if (options_.verbose) {
                    std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
                }
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
                    if (options_.verbose) {
                        std::cerr << "CDDP: Regularization limit reached" << std::endl;
                    }
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
            }
        }
        
        // Check termination due to small cost improvement
        if (optimality_gap_ < options_.grad_tolerance) {
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_ * (regularization_state_ > options_.regularization_state_min ? 1.0 : 0.0);
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_ * (regularization_control_ > options_.regularization_control_min ? 1.0 : 0.0);
            
            if (regularization_state_ < 1e-5 && regularization_control_ < 1e-5) {
                solution.converged = true;
                break;
            }
            
        }

        // 2. Forward pass (either single-threaded or multi-threaded)
        ForwardPassResult best_result;
        best_result.cost = std::numeric_limits<double>::infinity();
        best_result.lagrangian = std::numeric_limits<double>::infinity();
        bool forward_pass_success = false;

        if (!options_.use_parallel) {
            // Single-threaded execution with early termination
            for (double alpha : alphas_) {
                ForwardPassResult result = solveForwardPass(alpha);
                
                if (result.success && result.cost < best_result.cost) {
                    best_result = result;
                    forward_pass_success = true;
                    
                    // Check for early termination
                    if (result.success) {
                        if (options_.debug) {
                            std::cout << "CDDP: Early termination due to successful forward pass" << std::endl;
                        }
                        break;
                    }
                }
            }
        } else { 
            // TODO: Improve multi-threaded execution
            // Multi-threaded execution 
            std::vector<std::future<ForwardPassResult>> futures;
            futures.reserve(alphas_.size());
            
            // Launch all forward passes in parallel
            for (double alpha : alphas_) {
                futures.push_back(std::async(std::launch::async, 
                    [this, alpha]() { return solveForwardPass(alpha); }));
            }
            
            // Collect results from all threads
            for (auto& future : futures) {
                try {
                    if (future.valid()) {
                        ForwardPassResult result = future.get();
                        if (result.success && result.cost < best_result.cost) {
                            best_result = result;
                            forward_pass_success = true;
                        }
                    }
                } catch (const std::exception& e) {
                    if (options_.verbose) {
                        std::cerr << "CDDP: Forward pass thread failed: " << e.what() << std::endl;
                    }
                    continue;
                }
            }
        }

        // Update solution if a feasible forward pass was found
        if (forward_pass_success) {
            if (options_.debug) {
                std::cout << "Best cost: " << best_result.cost << std::endl;
                std::cout << "Best alpha: " << best_result.alpha << std::endl;
            }
            X_ = best_result.state_sequence;
            U_ = best_result.control_sequence;
            dJ_ = J_ - best_result.cost;
            J_ = best_result.cost;
            dL_ = L_ - best_result.lagrangian;
            L_ = best_result.lagrangian;
            alpha_ = best_result.alpha;
            solution.cost_sequence.push_back(J_);
            solution.lagrangian_sequence.push_back(L_);

            // Decrease regularization
            regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
            regularization_state_ *= regularization_state_step_ * (regularization_state_ > options_.regularization_state_min ? 1.0 : 0.0);
            regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
            regularization_control_ *= regularization_control_step_ * (regularization_control_ > options_.regularization_control_min ? 1.0 : 0.0);

            // Check termination
            if (dJ_ < options_.cost_tolerance) {
                solution.converged = true;
                break;
            }
        } else {
            // Increase regularization
            regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
            regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_max);
            regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
            regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_max);

            // Check regularization limit
            if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                std::cerr << "CDDP: Regularization limit reached" << std::endl;
                solution.converged = false;
                break;
            }
        }

        // Print iteration information
        if (options_.verbose) {
            printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_); 
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 

    // Finalize solution
    solution.state_sequence = X_;
    solution.control_sequence = U_;
    solution.alpha = alpha_;
    solution.solve_time = duration.count(); // Time in microseconds
    
    if (options_.verbose) {
        printSolution(solution); 
    }

    return solution;
}

bool CDDP::solveBackwardPass() {
    // Initialize variables
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Extract control box constraint
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Terminal cost and its derivatives]
    Eigen::VectorXd V_x = objective_->getFinalCostGradient(X_.back());
    Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());

    // Pre-allocate matrices
    Eigen::MatrixXd A(state_dim, state_dim);
    Eigen::MatrixXd B(state_dim, control_dim);
    Eigen::VectorXd Q_x(state_dim);
    Eigen::VectorXd Q_u(control_dim);
    Eigen::MatrixXd Q_xx(state_dim, state_dim);
    Eigen::MatrixXd Q_uu(control_dim, control_dim);
    Eigen::MatrixXd Q_uu_reg(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::MatrixXd Q_ux_reg(control_dim, state_dim);
    Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);
    dV_ = Eigen::Vector2d::Zero();

    double Qu_error = 0.0;

    // Backward Riccati recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Get state and control
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Get continuous dynamics Jacobians
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx; 
        A.diagonal().array() += 1.0;
        B = timestep_ * Fu;

        // Get cost and its derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // Compute Q-function matrices 
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;
        Q_ux = l_ux + B.transpose() * V_xx * A;
        Q_uu = l_uu + B.transpose() * V_xx * B;

        // TODO: Apply Cholesky decomposition to Q_uu later?
        // // Symmetrize Q_uu for Cholesky decomposition
        // Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        if (options_.regularization_type == "state" || options_.regularization_type == "both") {
            Q_ux_reg = l_ux + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * A;
            Q_uu_reg = l_uu + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * B;
        } else {
            Q_ux_reg = Q_ux;
            Q_uu_reg = Q_uu;
        } 

        if (options_.regularization_type == "control" || options_.regularization_type == "both") {
            Q_uu_reg.diagonal().array() += regularization_control_;
        }

        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
        const Eigen::VectorXd& eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            if (options_.debug) {
                std::cerr << "CDDP: Q_uu is still not positive definite" << std::endl;
            }
            return false;
        }

        if (control_box_constraint == nullptr) {
            const Eigen::MatrixXd &H = Q_uu_reg.inverse();
            k = -H * Q_u;
            K = -H * Q_ux_reg;
        } else {
            // Solve QP by boxQP
            const Eigen::VectorXd& lb = control_box_constraint->getLowerBound() - u;
            const Eigen::VectorXd& ub = control_box_constraint->getUpperBound() - u;
            const Eigen::VectorXd& x0 = k_[t]; // Initial guess
            
            cddp::BoxQPResult qp_result = boxqp_solver_.solve(Q_uu_reg, Q_u, lb, ub, x0);
            
            // TODO: Better status check
            if (qp_result.status == BoxQPStatus::HESSIAN_NOT_PD || 
                qp_result.status == BoxQPStatus::NO_DESCENT) {
                    if (options_.debug) {
                        std::cerr << "CDDP: BoxQP failed at time step " << t << std::endl;
                    }
                return false;
            }
            
            // Extract solution
            k = qp_result.x;

            // Compute feedback gain matrix
            K = Eigen::MatrixXd::Zero(control_dim, state_dim);
            if (qp_result.free.sum() > 0) {
                // Get indices of free variables
                std::vector<int> free_idx;
                for (int i = 0; i < control_dim; i++) {
                    if (qp_result.free(i)) {
                        free_idx.push_back(i);
                    }
                }

                // Extract relevant parts of Q_ux for free variables
                Eigen::MatrixXd Q_ux_free(free_idx.size(), state_dim);
                for (size_t i = 0; i < free_idx.size(); i++) {
                    Q_ux_free.row(i) = Q_ux_reg.row(free_idx[i]);
                }

                // Compute gains for free variables using the LDLT factorization
                Eigen::MatrixXd K_free = -qp_result.Hfree.solve(Q_ux_free);

                // Put back into full K matrix
                for (size_t i = 0; i < free_idx.size(); i++) {
                    K.row(free_idx[i]) = K_free.row(i);
                }
            }
        }

        // Store feedforward and feedback gain
        k_[t] = k;
        K_[t] = K;

        // Compute value function approximation
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ = dV_ + dV_step;
        V_x = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize Hessian

        // Compute optimality gap (Inf-norm) for convergence check
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

        // TODO: Add constraint optimality gap analysis
        optimality_gap_ = Qu_error;
    }

    if (options_.debug) {
        std::cout << "Qu_error: " << Qu_error << std::endl;
        std::cout << "dV: " << dV_.transpose() << std::endl;
    }
    
    return true;
}

ForwardPassResult CDDP::solveForwardPass(double alpha) {
    // Prepare result struct
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.lagrangian = std::numeric_limits<double>::infinity();
    result.alpha = alpha;

    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Initialize trajectories
    std::vector<Eigen::VectorXd> X_new = X_;
    std::vector<Eigen::VectorXd> U_new = U_;
    X_new[0] = initial_state_;
    double J_new = 0.0;

    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    for (int t = 0; t < horizon_; ++t) {
        const Eigen::VectorXd& x = X_new[t];
        const Eigen::VectorXd& u = U_new[t];
        const Eigen::VectorXd& delta_x = x - X_[t];

        U_new[t] = u + alpha * k_[t] + K_[t] * delta_x;

        if (control_box_constraint != nullptr) {
            U_new[t] = control_box_constraint->clamp(U_new[t]);
        }

        J_new += objective_->running_cost(x, U_new[t], t);
        X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);
    }
    J_new += objective_->terminal_cost(X_new.back());
    double dJ = J_ - J_new;
    double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
    double reduction_ratio = expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);

    // Check if cost reduction is sufficient
    result.success = reduction_ratio > options_.minimum_reduction_ratio;
    result.state_sequence = X_new;
    result.control_sequence = U_new;
    result.cost = J_new;
    result.lagrangian = J_new;

    return result;
}

void CDDP::printSolverInfo()
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

void CDDP::printOptions(const CDDPOptions &options)
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

void CDDP::printIteration(int iter, double cost, double lagrangian, double grad_norm, 
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

void CDDP::printSolution(const CDDPSolution &solution)
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

} // namespace cddp

// // Solve the problem
// CDDPSolution CDDP::solve()
// {
//     // Initialize CDDP solver
//     initializeCDDP();

//     if (options_.verbose)
//     {
//         printOptions(options_);
//     }

//     // Initialize solution
//     CDDPSolution solution;
//     solution.converged = false;
//     solution.time_sequence.reserve(horizon_ + 1);
//     for (int t = 0; t <= horizon_; ++t)
//     {
//         solution.time_sequence.push_back(timestep_ * t);
//     }
//     solution.cost_sequence.reserve(options_.max_iterations);       // Reserve space for efficiency
//     solution.lagrangian_sequence.reserve(options_.max_iterations); // Reserve space for efficiency

//     // Evaluate initial cost
//     J_ = objective_->evaluate(X_, U_);
//     solution.cost_sequence.push_back(J_);

//     // Evaluate Lagrangian
//     L_ = J_;

//     // // Loop over horizon
//     // for (int t = 0; t < 1; ++t) {
//     //     // Evaluate state constraint violation
//     //     for (const auto& constraint : constraint_set_) {
//     //         if (constraint.first == "ControlBoxConstraint") {
//     //             L_ += getLogBarrierCost(*constraint.second, X_[t], U_[t], barrier_coeff_, options_.relaxation_coeff);
//     //             // Eigen::VectorXd constraint_violation = constraint.second->evaluate(X_[t], U_[t]);
//     //             // if (constraint_violation.minCoeff() < 0) {
//     //             //     std::cerr << "CDDP: Constraint violation at time " << t << std::endl;
//     //             //     std::cerr << "Constraint violation: " << constraint_violation.transpose() << std::endl;
//     //             //     throw std::runtime_error("Constraint violation");
//     //             // }
//     //         }

//     //     }
//     // }

//     if (options_.verbose)
//     {
//         printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_); // Initial iteration information
//     }

//     // Start timer
//     auto start_time = std::chrono::high_resolution_clock::now();
//     int iter = 0;

//     // Main loop of CDDP
//     while (iter < options_.max_iterations)
//     {
//         ++iter;

//         // Check maximum CPU time
//         if (options_.max_cpu_time > 0)
//         {
//             auto end_time = std::chrono::high_resolution_clock::now();
//             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//             if (duration.count() * 1e-6 > options_.max_cpu_time)
//             {
//                 std::cerr << "CDDP: Maximum CPU time reached. Returning current solution" << std::endl;
//                 break;
//             }
//         }

//         // 1. Backward pass: Solve Riccati recursion to compute optimal control law
//         bool backward_pass_success = false;
//         while (!backward_pass_success)
//         {
//             backward_pass_success = solveBackwardPass();

//             if (!backward_pass_success)
//             {
//                 std::cerr << "CDDP: Backward pass failed" << std::endl;

//                 // Increase regularization
//                 regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
//                 regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
//                 regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
//                 regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);

//                 if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max)
//                 {
//                     std::cerr << "CDDP: Regularization limit reached" << std::endl;
//                     break; // Exit if regularization limit reached
//                 }
//                 continue; // Continue if backward pass fails
//             }
//         }

//         // Check termination due to small cost improvement
//         if (optimality_gap_ < options_.grad_tolerance && regularization_state_ < 1e-4 && regularization_control_ < 1e-4)
//         {
//             regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
//             regularization_state_ *= regularization_state_step_;
//             if (regularization_state_ <= options_.regularization_state_min)
//             {
//                 regularization_state_ = 0.0;
//             }
//             regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
//             regularization_control_ *= regularization_control_step_;
//             if (regularization_control_ <= options_.regularization_control_min)
//             {
//                 regularization_control_ = 0.0;
//             }

//             solution.converged = true;
//             break;
//         }

//         bool forward_pass_success = false;
//         // 2. Forward pass: line-search to find feasible optimal control sequence if backward pass is successful
//         if (backward_pass_success)
//         {
//             forward_pass_success = solveForwardPass();
//         }

//         // Print iteration information
//         if (options_.verbose)
//         {
//             printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_);
//         }

//         if (forward_pass_success)
//         {
//             // Decrease regularization
//             regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
//             regularization_state_ *= regularization_state_step_;
//             if (regularization_state_ <= options_.regularization_state_min)
//             {
//                 regularization_state_ = 0.0;
//             }
//             regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
//             regularization_control_ *= regularization_control_step_;
//             if (regularization_control_ <= options_.regularization_control_min)
//             {
//                 regularization_control_ = 0.0;
//             }

//             // Append Latest Cost
//             solution.cost_sequence.push_back(J_);

//             if (dJ_ < options_.cost_tolerance)
//             {
//                 solution.converged = true;
//                 solution.iterations = iter;
//                 break;
//             }
//         }
//     }

//     // Finalize solution
//     solution.control_sequence = U_;
//     solution.state_sequence = X_;
//     solution.iterations = solution.converged ? iter : options_.max_iterations;

//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//     solution.solve_time = duration.count(); // Time in microseconds
//     printSolution(solution);

//     return solution;
// }

// // Backward pass
// bool CDDP::solveBackwardPass()
// {
//     // Initialize variables
//     const int state_dim = system_->getStateDim();
//     const int control_dim = system_->getControlDim();
//     const auto active_set_tol = options_.active_set_tolerance;

//     // Extract control box constraint
//     auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

//     // Terminal cost and its derivatives
//     V_X_.back() = objective_->getFinalCostGradient(X_.back());
//     V_XX_.back() = objective_->getFinalCostHessian(X_.back());

//     // Pre-allocate matrices
//     Eigen::MatrixXd A(state_dim, state_dim);
//     Eigen::MatrixXd B(state_dim, control_dim);
//     Eigen::VectorXd Q_x(state_dim);
//     Eigen::VectorXd Q_u(control_dim);
//     Eigen::MatrixXd Q_xx(state_dim, state_dim);
//     Eigen::MatrixXd Q_uu(control_dim, control_dim);
//     Eigen::MatrixXd Q_ux(control_dim, state_dim);
//     Eigen::MatrixXd Q_uu_inv(control_dim, control_dim);
//     Eigen::MatrixXd active_constraint_tabl(2 * (control_dim), horizon_);
//     Eigen::VectorXd k(control_dim);
//     Eigen::MatrixXd K(control_dim, state_dim);

//     double Qu_error = 0.0;

//     // Backward Riccati recursion
//     for (int t = horizon_ - 1; t >= 0; --t)
//     {
//         // Get state and control
//         const Eigen::VectorXd &x = X_[t];
//         const Eigen::VectorXd &u = U_[t];

//         // Extract value function approximation
//         const Eigen::VectorXd &V_x = V_X_[t + 1];
//         const Eigen::MatrixXd &V_xx = V_XX_[t + 1];

//         // Get continuous dynamics Jacobians
//         const auto [Fx, Fu] = system_->getJacobians(x, u);

//         // Convert continuous dynamics to discrete time
//         A = timestep_ * Fx;
//         A.diagonal().array() += 1.0; // More efficient way to add identity
//         B = timestep_ * Fu;

//         // Get cost and its derivatives
//         double l = objective_->running_cost(x, u, t);
//         auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
//         auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

//         // TODO: Implement log barrier cost and its derivatives // FIXME: Implement log barrier cost and its derivatives
//         // Get log barrier cost and its derivatives
//         // for (const auto& constraint : constraint_set_) {
//         //     if (constraint.first == "ControlBoxConstraint") {
//         //         const double barrier_cost = getLogBarrierCost(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
//         //         l += barrier_cost;
//         //         const auto [l_x_barrier, l_u_barrier] = getLogBarrierCostGradients(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
//         //         const auto [l_xx_barrier, l_uu_barrier, l_ux_barrier] = getLogBarrierCostHessians(*constraint.second, x, u, barrier_coeff_, options_.relaxation_coeff);
//         //         l_x += l_x_barrier;
//         //         l_u += l_u_barrier;
//         //         // l_xx += l_xx_barrier;
//         //         // l_uu += l_uu_barrier;
//         //         // l_ux += l_ux_barrier;
//         //     }
//         // }

//         // Compute Q-function matrices
//         Q_x = l_x + A.transpose() * V_x;
//         Q_u = l_u + B.transpose() * V_x;
//         Q_xx = l_xx + A.transpose() * V_xx * A;

//         if (options_.regularization_type == "state" || options_.regularization_type == "both")
//         {
//             Q_ux = l_ux + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * A;
//             Q_uu = l_uu + B.transpose() * (V_xx + regularization_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim)) * B;
//         }
//         else
//         {
//             Q_ux = l_ux + B.transpose() * V_xx * A;
//             Q_uu = l_uu + B.transpose() * V_xx * B;
//         }

//         // Symmetrize Q_uu for cholensky decomposition
//         Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

//         // Control Regularization
//         if (options_.regularization_type == "control" || options_.regularization_type == "both")
//         {
//             Q_uu += options_.regularization_control * Eigen::MatrixXd::Identity(control_dim, control_dim);
//         }

//         // Check eigenvalues of Q_uu
//         Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
//         Eigen::VectorXd eigenvalues = es.eigenvalues().real();
//         if (eigenvalues.minCoeff() <= 0)
//         {
//             // Add regularization
//             // Q_uu.diagonal() += 1e-6;
//             std::cout << "Q_uu is not positive definite at " << t << std::endl;

//             eigenvalues = es.eigenvalues().real();
//             if (eigenvalues.minCoeff() <= 0)
//             {
//                 std::cout << "Q_uu is still not positive definite" << std::endl;
//                 return false;
//             }
//         }

//         // Cholesky decomposition
//         Eigen::LLT<Eigen::MatrixXd> llt(Q_uu);
//         if (llt.info() != Eigen::Success)
//         {
//             // Decomposition failed
//             std::cout << "Cholesky decomposition failed" << std::endl;
//             return false;
//         }

//         /*  --- Identify Active Constraint --- */
//         int active_constraint_index = 0;
//         Eigen::MatrixXd C(control_dim, control_dim); // Control constraint matrix
//         Eigen::MatrixXd D(control_dim, state_dim);   // State constraint matrix

//         // TODO: Implement active set method
//         for (int j = 0; j < control_dim; j++)
//         {
//             if (u(j) <= control_box_constraint->getLowerBound()(j) + active_set_tol)
//             {
//                 Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
//                 e(j) = 1.0;
//                 C.row(active_constraint_index) = -e; // Note the negative sign
//                 D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
//                 active_constraint_index += 1;
//             }
//             else if (u(j) >= control_box_constraint->getUpperBound()(j) - active_set_tol)
//             {
//                 Eigen::VectorXd e = Eigen::VectorXd::Zero(control_dim);
//                 e(j) = 1.0; // No negative here
//                 C.row(active_constraint_index) = e;
//                 D.row(active_constraint_index) = Eigen::VectorXd::Zero(state_dim);
//                 active_constraint_index += 1;
//             }
//         }

//         Eigen::MatrixXd active_constraint_table = Eigen::MatrixXd::Zero(2 * (control_dim), horizon_);
//         if (active_constraint_index == 0)
//         { // No active constraints
//             // Compute the optimal control deviation
//             Eigen::MatrixXd H = Q_uu.inverse();
//             K = -H * Q_ux;
//             k = -H * Q_u;
//         }
//         else
//         {
//             // Extract identified active constraints
//             Eigen::MatrixXd grad_x_g = D.topRows(active_constraint_index);
//             Eigen::MatrixXd grad_u_g = C.topRows(active_constraint_index);

//             // Calculate Lagrange multipliers
//             Eigen::MatrixXd Q_uu_inv = Q_uu.inverse();
//             Eigen::MatrixXd lambda = -(grad_u_g * Q_uu_inv * grad_u_g.transpose()).inverse() * (grad_u_g * Q_uu_inv * Q_u);

//             // Find indices where lambda is non-negative
//             std::vector<int> active_indices;
//             for (int i = 0; i < lambda.rows(); ++i)
//             {
//                 if (lambda(i) >= 0)
//                 {
//                     active_indices.push_back(i);
//                 }
//             }
//             int active_count_new = active_indices.size();

//             // Create new constraint matrices
//             Eigen::MatrixXd C_new = Eigen::MatrixXd::Zero(active_count_new, control_dim);
//             Eigen::MatrixXd D_new = Eigen::MatrixXd::Zero(active_count_new, state_dim);

//             if (active_count_new > 0)
//             {
//                 // Fill new constraint matrices with active constraints
//                 for (int i = 0; i < active_count_new; ++i)
//                 {
//                     C_new.row(i) = grad_u_g.row(active_indices[i]);
//                     D_new.row(i) = grad_x_g.row(active_indices[i]);
//                 }

//                 // Calculate feedback gains
//                 Eigen::MatrixXd W = -(C_new * Q_uu_inv * C_new.transpose()).inverse() * (C_new * Q_uu_inv);
//                 Eigen::MatrixXd H = Q_uu_inv * (Eigen::MatrixXd::Identity(control_dim, control_dim) - C_new.transpose() * W);
//                 k = -H * Q_u;
//                 K = -H * Q_ux + W.transpose() * D_new;
//             }
//             else
//             {
//                 // If no active constraints remain, revert to unconstrained solution
//                 Eigen::MatrixXd H = Q_uu.inverse();
//                 K = -H * Q_ux;
//                 k = -H * Q_u;
//             }
//         }

//         // Store Q-function matrices
//         Q_UU_[t] = Q_uu;
//         Q_UX_[t] = Q_ux;
//         Q_U_[t] = Q_u;

//         // Compute value function approximation
//         Eigen::Vector2d dV_step;
//         dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
//         dV_ = dV_ + dV_step;
//         V_X_[t] = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
//         V_XX_[t] = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
//         V_XX_[t] = 0.5 * (V_XX_[t] + V_XX_[t].transpose()); // Symmetrize Hessian

//         // Compute optimality gap (Inf-norm) for convergence check
//         Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());

//         // TODO: Add constraint optimality gap analysis
//         optimality_gap_ = Qu_error;
//     }

//     return true;
// }

// // Forward pass
// bool CDDP::solveForwardPass()
// {
//     bool is_feasible = false;
//     const int state_dim = system_->getStateDim();
//     const int control_dim = system_->getControlDim();

//     // Extract control box constraint
//     auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

//     int iter = 0;
//     double alpha = options_.backtracking_coeff;

//     // Line-search iteration
//     for (iter = 0; iter < options_.max_line_search_iterations; ++iter)
//     {
//         // Initialize cost and constraints
//         double J_new = 0.0, dJ = 0.0, expected_dV = 0.0, gradient_norm = 0.0;
//         double L_new = 0.0;

//         std::vector<Eigen::VectorXd> X_new = X_;
//         std::vector<Eigen::VectorXd> U_new = U_;
//         X_new[0] = initial_state_;

//         for (int t = 0; t < horizon_; ++t)
//         {
//             // Get state and control
//             const Eigen::VectorXd &x = X_new[t];
//             const Eigen::VectorXd &u = U_new[t];

//             // Deviation from the nominal trajectory
//             const Eigen::VectorXd &delta_x = x - X_[t];

//             // Extract Q-function matrices
//             const Eigen::VectorXd &Q_u = Q_U_[t];
//             const Eigen::MatrixXd &Q_uu = Q_UU_[t];
//             const Eigen::MatrixXd &Q_ux = Q_UX_[t];

//             const Eigen::VectorXd &q = alpha * Q_u + Q_ux * delta_x; // Gradient of QP objective

//             // Lower and upper bounds
//             const Eigen::VectorXd& lb = control_box_constraint->getLowerBound() - u;
//             const Eigen::VectorXd& ub = control_box_constraint->getUpperBound() - u;

//             // Create b vector
//             Eigen::VectorXd b(2 * control_dim);
//             b << ub, -lb;

//             // Create A matrix
//             Eigen::MatrixXd A(2 * control_dim, control_dim); // Upper is identity, lower is -identity
//             A.setIdentity();
//             A.block(control_dim, 0, control_dim, control_dim) = -Eigen::MatrixXd::Identity(control_dim, control_dim);

//             QPSolverOptions options;
//             options.verbose = false;
//             QPSolver solver(options);

//             solver.setDimensions(state_dim, 2*control_dim);
//             solver.setHessian(Q_uu);
//             solver.setGradient(q);
//             solver.setConstraints(A, b);

//             QPResult result = solver.solve();

//             // Extract solution
//             const Eigen::VectorXd &delta_u = result.x;

//             // Extract solution
//             U_new[t] += delta_u;

//             // Compute cost
//             J_new += objective_->running_cost(x, U_new[t], t);

//             // Compute new state
//             X_new[t + 1] = system_->getDiscreteDynamics(x, U_new[t]);
//         }
//         J_new += objective_->terminal_cost(X_new.back());

//         // Calculate Cost Reduction
//         dJ = J_ - J_new;

//         double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
//         double reduction_ratio = 0.0;
//         if (expected > 0.0)
//         {
//             reduction_ratio = dJ / expected;
//         }
//         else
//         {
//             reduction_ratio = std::copysign(1.0, dJ);
//             std::cout << "Expected improvement is not positive" << std::endl;
//         }
//         if (reduction_ratio > options_.minimum_reduction_ratio)
//         {
//             // Update state and control
//             X_ = X_new;
//             U_ = U_new;
//             J_ = J_new;
//             dJ_ = dJ;
//             alpha_ = alpha;
//             // barrier_coeff_ = options_.barrier_coeff / 10.0;
//             return true;
//         }
//         else
//         {
//             alpha *= options_.backtracking_factor;
//         }

//         // // Check constraints
//         // is_feasible = true;
//         // for (int t = 0; t < horizon_; ++t) {
//         //     const Eigen::VectorXd& u = U_new[t];
//         //     // Check control box constraint
//         //     if (control_box_constraint != constraint_set_.end()) {
//         //         if (!control_box_constraint->second->isFeasible(u)) {
//         //             is_feasible = false;
//         //             break;
//         //         }
//         //     }
//         // }
//     }

//     return false; // Or false if forward pass fails
// }

