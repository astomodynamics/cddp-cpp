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
#include <future> // For multi-threading
#include <thread> // For multi-threading

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/helper.hpp"
#include "cddp_core/boxqp.hpp"

namespace cddp
{

void CDDP::initializeIPDDP()
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
            std::cerr << "IPDDP::initializeIPDDP: No dynamical system provided." << std::endl;
        }
        return;
    }

    if (!objective_)
    {
        initialized_ = false;
        if (options_.verbose) {
            std::cerr << "IPDDP::initializeIPDDP: No objective function provided." << std::endl;
        }
        return;
    }

    const int state_dim = getStateDim();
    const int control_dim = getControlDim();

    // Check if reference_state in objective and reference_state in IPDDP are the same
    if ((reference_state_ - objective_->getReferenceState()).norm() > 1e-6)
    {
        std::cerr << "IPDDP: Initial state and goal state in the objective function do not match" << std::endl;
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

    k_u_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    K_u_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));

    Y_.clear();
    S_.clear();
    k_y_.clear();
    K_y_.clear();
    k_s_.clear();
    K_s_.clear();

    for (const auto &constraint : constraint_set_)
    {
        std::string constraint_name = constraint.first;
        // Initialize dual and slack trajectories TODO: Find a better way to initialize
        int dual_dim = constraint.second->getDualDim();
        Y_[constraint_name].resize(horizon_, 0.1 * Eigen::VectorXd::Ones(dual_dim));
        S_[constraint_name].resize(horizon_, 0.01 * Eigen::VectorXd::Ones(dual_dim));

        // Initialize gains
        k_y_[constraint_name].resize(horizon_, Eigen::VectorXd::Zero(dual_dim));
        K_y_[constraint_name].resize(horizon_, Eigen::MatrixXd::Zero(dual_dim, state_dim));
        k_s_[constraint_name].resize(horizon_, Eigen::VectorXd::Zero(dual_dim));
        K_s_[constraint_name].resize(horizon_, Eigen::MatrixXd::Zero(dual_dim, state_dim));
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
    dV_.resize(2); // Cost improvement 

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

    // Initialize Log-barrier object
    log_barrier_ = std::make_unique<LogBarrier>(options_.barrier_coeff, 
                                                options_.relaxation_coeff, 
                                                options_.barrier_order, 
                                                options_.is_relaxed_log_barrier);
    mu_ = options_.barrier_coeff;
    log_barrier_->setBarrierCoeff(mu_);

    // Check if ControlBoxConstraint is set
    if (constraint_set_.find("ControlBoxConstraint") != constraint_set_.end())
    {
        std::cout << "ControlBoxConstraint is set" << std::endl;
    }

    // Initialize filter acceptance parameter
    gamma_ = options_.filter_acceptance;
    constraint_violation_ = 0.0;
    
    // Now initialized
    initialized_ = true;
}


CDDPSolution CDDP::solveIPDDP()
{
    // Initialize if not done
    if (!initialized_) {
        initializeIPDDP();
    }

    if (!initialized_) {
        std::cerr << "IPDDP: Initialization failed" << std::endl;
        throw std::runtime_error("IPDDP: Initialization failed");
    }

    // Initialize log barrier parameters
    mu_ = options_.barrier_coeff;  // Initial barrier coefficient
    log_barrier_->setBarrierCoeff(mu_);

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
    solution.cost_sequence.reserve(options_.max_iterations);
    solution.lagrangian_sequence.reserve(options_.max_iterations); 
    solution.cost_sequence.push_back(J_);

    // Evaluate Lagrangian
    L_ = J_;
    // # TODO: Multi-threading?
    for (int t = 0; t < horizon_; ++t)
    {
        for (const auto &constraint : constraint_set_)
        {
            L_ += log_barrier_->evaluate(*constraint.second, X_[t], U_[t]);
        }
    }
    solution.lagrangian_sequence.push_back(L_);

    if (options_.verbose)
    {
        printIteration(0, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); // Initial iteration information
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;

    // Main loop of CDDP
    while (iter < options_.max_iterations) {
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
            backward_pass_success = solveIPDDPBackwardPass();

            if (!backward_pass_success) {
                std::cerr << "CDDP: Backward pass failed" << std::endl;

                // Increase regularization
                if (options_.regularization_type == "state") {
                    regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                    regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                } else if (options_.regularization_type == "control") {
                    regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                    regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);
                } else if (options_.regularization_type == "both") {
                    regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                    regularization_state_ = std::max(regularization_state_ * regularization_state_step_, options_.regularization_state_min);
                    regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                    regularization_control_ = std::max(regularization_control_ * regularization_control_step_, options_.regularization_control_min);
                } 

                if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                    if (options_.verbose) {
                        std::cerr << "CDDP: Backward pass regularization limit reached" << std::endl;
                    }
                    break; // Exit if regularization limit reached
                }
                continue; // Continue if backward pass fails
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
                ForwardPassResult result = solveIPDDPForwardPass(alpha);
                
                if (result.success && result.cost < best_result.cost) {
                    best_result = result;
                    forward_pass_success = true;
                    
                    // Check for early termination
                    if (result.success) {
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
                    [this, alpha]() { return solveIPDDPForwardPass(alpha); }));
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
            if (options_.regularization_type == "state") {
                regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
                regularization_state_ *= regularization_state_step_;
                if (regularization_state_ < options_.regularization_state_min) {
                    regularization_state_ = options_.regularization_state_min;
                }
            } else if (options_.regularization_type == "control") {
                regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
                regularization_control_ *= regularization_control_step_;
                if (regularization_control_ < options_.regularization_control_min) {
                    regularization_control_ = options_.regularization_control_min;
                }
            } else if (options_.regularization_type == "both") {
                regularization_state_step_ = std::min(regularization_state_step_ / options_.regularization_state_factor, 1 / options_.regularization_state_factor);
                regularization_state_ *= regularization_state_step_;
                if (regularization_state_ < options_.regularization_state_min) {
                    regularization_state_ = options_.regularization_state_min;
                }
                regularization_control_step_ = std::min(regularization_control_step_ / options_.regularization_control_factor, 1 / options_.regularization_control_factor);
                regularization_control_ *= regularization_control_step_;
                if (regularization_control_ < options_.regularization_control_min) {
                    regularization_control_ = options_.regularization_control_min;
                }
            }

            // Check termination
            if (dJ_ < options_.cost_tolerance || (std::max(optimality_gap_, mu_) < options_.grad_tolerance)) {
                solution.converged = true;
                break;
            }
        } else {
            bool early_termination_flag = false; // TODO: Improve early termination
            // Increase regularization
            if (options_.regularization_type == "state") {
                if (regularization_state_ < 1e-2) {
                    early_termination_flag = true; // Early termination if regularization is fairly small
                }
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_state_ = std::min(regularization_state_ * regularization_state_step_, options_.regularization_state_max);
                
            } else if (options_.regularization_type == "control") {
                if (regularization_control_ < 1e-2) {
                    early_termination_flag = true; // Early termination if regularization is fairly small
                }
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                regularization_control_ = std::min(regularization_control_ * regularization_control_step_, options_.regularization_control_max);
            } else if (options_.regularization_type == "both") {
                if (regularization_state_ < 1e-2 ||  
                    regularization_control_ < 1e-2) {
                    early_termination_flag = true; // Early termination if regularization is fairly small
                }
                regularization_state_step_ = std::max(regularization_state_step_ * options_.regularization_state_factor, options_.regularization_state_factor);
                regularization_state_ = std::min(regularization_state_ * regularization_state_step_, options_.regularization_state_max);
                regularization_control_step_ = std::max(regularization_control_step_ * options_.regularization_control_factor, options_.regularization_control_factor);
                regularization_control_ = std::min(regularization_control_ * regularization_control_step_, options_.regularization_control_max);
            } else {
                early_termination_flag = false;
            }

            // Check early termination
            if (options_.early_termination && early_termination_flag) {
                if (dJ_ < options_.cost_tolerance * 1e2 ||
                    (optimality_gap_ < options_.grad_tolerance * 1e1)) 
                {
                    solution.converged = true;
                    if (options_.verbose) {
                        std::cerr << "CDDP: Early termination due to small cost reduction" << std::endl;
                    }
                    break;
                }
            }
            
            // Check regularization limit
            if (regularization_state_ >= options_.regularization_state_max || regularization_control_ >= options_.regularization_control_max) {
                if ((dJ_ < options_.cost_tolerance * 1e2) ||
                    (optimality_gap_ < options_.grad_tolerance * 1e1)) 
                {
                    solution.converged = true;
                }  else
                {
                    // We are forced to large regularization but still not near local min
                    solution.converged = false;
                }
                if (options_.verbose) {
                    std::cerr << "CDDP: Regularization limit reached. "
                            << (solution.converged ? "Treating as converged." : "Not converged.") 
                            << std::endl;
                }
                break;
            }
        }

        // Print iteration information
        if (options_.verbose) {
           printIteration(iter, J_, L_, optimality_gap_, regularization_state_, regularization_control_, alpha_, mu_, constraint_violation_); 
        }

        // Update barrier parameter mu
        if (forward_pass_success && (optimality_gap_ < 0.2 * mu_)) {
            mu_ = std::max(mu_ * options_.barrier_factor, std::min(0.2 * mu_, std::pow(mu_, 1.2)));
        }
        log_barrier_->setBarrierCoeff(mu_);
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

bool CDDP::solveIPDDPBackwardPass() {
    // Setup
    // Initialize variables
    const int state_dim = getStateDim();
    const int control_dim = getControlDim();
    const int total_dual_dim = getTotalDualDim(); // Number of dual variables across constraints

    // Terminal cost and derivatives
    Eigen::VectorXd V_x  = objective_->getFinalCostGradient(X_.back());
    Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose());  // Symmetrize 

    dV_ = Eigen::Vector2d::Zero();
    double Qu_max_norm = 0.0;
    double residual_max = 0.0; // complementary residual measure: r = s ◦ y - mu_
    double dual_norm = 0.0; //

    // Backward Recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        //Expand cost around (x[t], u[t])
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Continuous dynamics 
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Discretize
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx;
        Eigen::MatrixXd B = timestep_ * Fu;

        // Gather dual and slack variables, and constraint values
        Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::VectorXd s = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);
        Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);

        int offset = 0; // offset in [0..total_dual_dim)
        for (auto &cKV : constraint_set_) {
            const std::string &cname = cKV.first;
            auto &constraint        = cKV.second;
            int dual_dim = constraint->getDualDim();

            // Slack & dual at time t and constraint cname
            Eigen::VectorXd y_vec = Y_[cname][t]; // dual variable
            Eigen::VectorXd s_vec = S_[cname][t]; // slack variable

            // Evaluate constraint
            // Eigen::VectorXd g_vec = constraint->evaluate(x, u); // dimension = dual_dim
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::VectorXd g_vec = Eigen::VectorXd::Zero(dual_dim);
            if (cname == "ControlBoxConstraint") {
                Eigen::VectorXd lb = constraint->getLowerBound();
                Eigen::VectorXd ub = constraint->getUpperBound();
                g_vec.head(control_dim) = u - ub;
                g_vec.tail(control_dim) = lb - u;
            }
            
    
            // partial wrt. x => g_x
            // Eigen::MatrixXd g_x = constraint->getStateJacobian(x, u);
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::MatrixXd g_x = Eigen::MatrixXd::Zero(dual_dim, state_dim);

            // partial wrt. u => g_u
            // Eigen::MatrixXd g_u = constraint->getControlJacobian(x, u);
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::MatrixXd g_u = Eigen::MatrixXd::Zero(dual_dim, control_dim);
            if (cname == "ControlBoxConstraint") {
                // top half
                g_u.block(0, 0, dual_dim/2, control_dim) 
                    =  Eigen::MatrixXd::Identity(dual_dim/2, control_dim);
                // bottom half
                g_u.block(control_dim, 0,  dual_dim/2, control_dim) 
                    = -Eigen::MatrixXd::Identity( dual_dim/2, control_dim);
            }

            // Insert into big arrays
            y.segment(offset, dual_dim)   = y_vec;
            s.segment(offset, dual_dim)   = s_vec;
            g.segment(offset, dual_dim)   = g_vec;
            Q_yx.block(offset, 0, dual_dim, state_dim)   = g_x;
            Q_yu.block(offset, 0, dual_dim, control_dim) = g_u;

            offset += dual_dim;
        }

        // Cost & derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // Q expansions from cost
        Eigen::VectorXd Q_x  = l_x + Q_yx.transpose() * y + A.transpose() * V_x;
        Eigen::VectorXd Q_u  = l_u + Q_yx.transpose() * y + B.transpose() * V_x;
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

        Eigen::MatrixXd Y = y.asDiagonal(); // Diagonal matrix with y as diagonal
        Eigen::MatrixXd S = s.asDiagonal(); // Diagonal matrix with s as diagonal
        Eigen::MatrixXd G = g.asDiagonal(); // Diagonal matrix with g as diagonal
        Eigen::MatrixXd S_inv = S.inverse(); // Inverse of S
        Eigen::MatrixXd YSinv = Y * S_inv; // Y * S_inv

        // Residuals:
        // r_p = g + s (primal feasibility)
        Eigen::VectorXd r_p = g + s;

        // r_d   = y.*s - mu (dual feasibility)
        Eigen::VectorXd r_d = y.cwiseProduct(s).array() - mu_;

        // rhat = y .* r_p - r_d = y.*(g + s) - (y.*s - mu)
        Eigen::VectorXd rhat = y.cwiseProduct(r_p) - r_d;           

        // Regularization
        Eigen::MatrixXd Q_ux_reg = Q_ux;
        Eigen::MatrixXd Q_uu_reg = Q_uu;

        // TODO: Add State regularization here
        if (options_.regularization_type == "control" || 
            options_.regularization_type == "both") {
            Q_uu_reg.diagonal().array() += regularization_control_;
        }
        Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg +  Q_yu.transpose() * YSinv * Q_yu);
        if (llt.info() != Eigen::Success) {
            if (options_.debug) {
                std::cerr << "CDDP: Backward pass failed at time " << t << std::endl;
            }
            return false;
        }
        
        Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
        bigRHS.col(0) = Q_u + Q_yu.transpose() * YSinv * rhat;
        Eigen::MatrixXd M = //(control_dim, state_dim)
            Q_ux + Q_yu.transpose() * YSinv * Q_yx; 
        for (int col = 0; col < state_dim; col++) {
            bigRHS.col(col+1) = M.col(col);
        }

        Eigen::MatrixXd R = llt.matrixU();
        // forward/back solve
        Eigen::MatrixXd kK = -R.transpose().triangularView<Eigen::Upper>().solve(
                                R.triangularView<Eigen::Upper>().solve(bigRHS)
                             );

        // parse out feedforward (ku) and feedback (Ku)
        Eigen::VectorXd k_u = kK.col(0); // dimension [control_dim]
        Eigen::MatrixXd K_u(control_dim, state_dim);
        for (int col = 0; col < state_dim; col++) {
            K_u.col(col) = kK.col(col+1);
        }

        // Save gains
        k_u_[t] = k_u;
        K_u_[t] = K_u;


        // Compute gains for constraints 
        Eigen::VectorXd k_y = S_inv * (rhat + S * Q_yu * k_u);
        Eigen::MatrixXd K_y = YSinv * (Q_yx + Q_yu * K_u);
        Eigen::VectorXd k_s = - r_p - Q_yu * k_u;
        Eigen::MatrixXd K_s = - Q_yx - Q_yu * K_u;

        for (auto &cKV : constraint_set_) {
            const std::string &cname = cKV.first;
            auto &constraint        = cKV.second;
            int dual_dim = constraint->getDualDim();

            // Now store gains
            k_y_[cname][t] = k_y.segment(offset, dual_dim);
            K_y_[cname][t] = K_y.block(offset, 0, dual_dim, state_dim);
            k_s_[cname][t] = k_s.segment(offset, dual_dim);
            K_s_[cname][t] = K_s.block(offset, 0, dual_dim, state_dim);

            offset += dual_dim;
        }

        // Update Q expansions
        Q_u  += Q_yu.transpose() * S_inv * rhat;
        Q_x  += Q_yx.transpose() * S_inv * rhat;
        Q_xx += Q_yx.transpose() * YSinv * Q_yx;
        Q_ux += Q_yx.transpose() * YSinv * Q_yu;
        Q_uu += Q_yu.transpose() * YSinv * Q_yu;

        // Update cost improvement
        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        // Update value function
        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;

        // Debug norms
        double Qu_norm = Q_u.lpNorm<Eigen::Infinity>();
        if (Qu_norm > Qu_max_norm) Qu_max_norm = Qu_norm;
        double r_norm = r_d.lpNorm<Eigen::Infinity>();
        if (r_norm > residual_max) residual_max = r_norm; 
    } // end for t

    // Compute optimality gap and print
    optimality_gap_ = std::max(Qu_max_norm, residual_max);

    if (options_.debug) {
        std::cout << "[IPDDP Backward Pass]\n"
                  << "    Qu_max_norm:  " << Qu_max_norm << "\n"
                  << "    residual_max:  " << residual_max << "\n"
                  << "    dV:           " << dV_.transpose() << std::endl;
    }
    return true;
} // end solveIPDDPBackwardPass

ForwardPassResult CDDP::solveIPDDPForwardPass(double alpha) {
    // Prepare result struct
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.lagrangian = std::numeric_limits<double>::infinity();
    result.alpha = alpha;

    const int state_dim = getStateDim();
    const int control_dim = getControlDim();

    double tau = std::max(0.99, 1.0 - mu_);  

    // Copy old trajectories (from the “previous” solution)
    std::vector<Eigen::VectorXd> X_new = X_;  // old states
    std::vector<Eigen::VectorXd> U_new = U_;  // old controls
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;  // old dual
    std::map<std::string, std::vector<Eigen::VectorXd>> S_new = S_;  // old slack

    X_new[0] = initial_state_;
    double cost_new = 0.0;
    double log_cost_new = 0.0;
    double primal_residual = 0.0;
    double sum_log_y = 0.0; 

    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Current filter point
    FilterPoint current{J_, L_};

    for (int t = 0; t < horizon_; ++t) {
        // 1) Update dual & slack
        for (auto &ckv : constraint_set_) {
            const std::string &cname = ckv.first;
            int dual_dim = ckv.second->getDualDim();

            // old y, s
            const Eigen::VectorXd &y_old = Y_[cname][t];
            const Eigen::VectorXd &s_old = S_[cname][t];

            // new y, s
            Eigen::VectorXd y_new = y_old + 
                                    alpha * k_y_[cname][t] + 
                                    K_y_[cname][t] * (X_new[t] - X_[t]); 
            Eigen::VectorXd s_new = s_old + 
                                    alpha * k_s_[cname][t] + 
                                    K_s_[cname][t] * (X_new[t] - X_[t]);
            
            // Enforce minimal feasibility w.r.t. old solution
            for (int i = 0; i < dual_dim; i++) {
                double y_min = (1.0 - tau) * y_old[i]; 
                double s_min = (1.0 - tau) * s_old[i];
                if (y_new[i] < y_min || s_new[i] < s_min) {
                    // fail early
                    // std::cout << "y_new: " << y_new.transpose() << std::endl;
                    // std::cout << "s_new: " << s_new.transpose() << std::endl;
                    // if (options_.debug) {
                    //     std::cerr << "[IPDDP ForwardPass] Feasibility fail at time=" 
                    //               << t << ", constraint=" << cname 
                    //               << " y_new or s_new < (1-tau)*y_old or s_old." 
                    //               << std::endl;
                    // }
                    return result; // success=false, cost=inf => exit
                }
                if (y_new[i] < 0.0) {
                    y_new[i] = 1e-8;
                }
                if (s_new[i] < 0.0) {
                    s_new[i] = 1e-8;
                }
            }

            // Save them
            Y_new[cname][t] = y_new;
            S_new[cname][t] = s_new;
        }

        // 2) Update control
        const Eigen::VectorXd &u_old = U_[t];
        U_new[t] = u_old + alpha * k_u_[t] + K_u_[t] * (X_new[t] - X_[t]);

        // 3) Accumulate cost / measure constraint violation
        double stage_cost = objective_->running_cost(X_new[t], U_new[t], t);

        cost_new += stage_cost;

        // 4) Evaluate constraints c = g(x,u)
        for (auto &ckv : constraint_set_) {
            const std::string &cname = ckv.first;
            auto &constraint = ckv.second;
            int cd = constraint->getDualDim();

            // Evaluate c
            // Eigen::VectorXd c_val = constraint->evaluate(X_new[t+1], U_new[t]);
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::VectorXd c_val = Eigen::VectorXd::Zero(cd);
            if (cname == "ControlBoxConstraint") {
                Eigen::VectorXd lb = constraint->getLowerBound();
                Eigen::VectorXd ub = constraint->getUpperBound();
                c_val.head(control_dim) = U_new[t] - ub;
                c_val.tail(control_dim) = lb - U_new[t];
            }

            // y_new
            Eigen::VectorXd y_new = Y_new[cname][t];
            // primal residual: L1 norm of c + y
            double local_res = (c_val + y_new).lpNorm<1>();
            if (local_res > primal_residual) {
                primal_residual = local_res;
            }
            
            if (y_new.minCoeff() <= 0.0) {
                // log of non-positive => fail or large cost
                if (options_.debug) {
                    std::cerr << "[IPDDP FwdPass] y_new <= 0 => log is invalid at time=" 
                                << t << ", constraint=" << cname << std::endl;
                }
                return result;
            }
            // sum logs
            sum_log_y += (y_new.array().log()).sum();
        }

        // 5) Step the dynamics
        X_new[t+1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
    }

    cost_new += objective_->terminal_cost(X_new.back());
    log_cost_new = cost_new - mu_ * sum_log_y;

    FilterPoint candidate{cost_new, log_cost_new};

    // Filter acceptance criteria  
    // bool sufficient_progress = 
    //             (cost_new < current.cost - gamma_ * candidate.violation) || 
    //             (candidate.violation < (1 - gamma_) * current.violation);
    bool sufficient_progress = (cost_new < current.cost);

    bool acceptable = sufficient_progress && !current.dominates(candidate);

   if (acceptable) {
       double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
       double reduction_ratio = expected > 0.0 ? (J_ - cost_new) / expected : 
                                               std::copysign(1.0, J_ - cost_new);

       result.success = acceptable;
       result.state_sequence = X_new;
       result.control_sequence = U_new;
       result.dual_sequence = Y_new;
       result.slack_sequence = S_new;
       result.cost = cost_new;
       result.lagrangian = log_cost_new;
       result.constraint_violation = primal_residual;
   }

   return result;
} // end solveIPDDPForwardPass
} // namespace cddp