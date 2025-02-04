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

void CDDP::initializeFeasibleIPDDP()
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

    // ------------------------------------------------------------------------
    // To accelerate convergence, especially for box-like constraints, we can
    // tune the initial multipliers (Y_) based on how close the initial guess
    // (U_) is to the bounds. This helps avoid the solver spending extra
    // iterations to scale them up or down.
    // ------------------------------------------------------------------------

    // First, we gather some approximate measure of how close the initial
    // control trajectory is to its constraint boundaries (if any).
    double avg_box_margin = 0.0;
    int box_count = 0;
    if (constraint_set_.find("ControlBoxConstraint") != constraint_set_.end())
    {
        auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");
        if (control_box_constraint)
        {
            Eigen::VectorXd lower_bound = control_box_constraint->getLowerBound();
            Eigen::VectorXd upper_bound = control_box_constraint->getUpperBound();

            // Compute the smallest distance to any boundary across the entire initial control trajectory
            for (int t = 0; t < horizon_; ++t)
            {
                Eigen::VectorXd dist_to_ub = (upper_bound - U_[t]);
                Eigen::VectorXd dist_to_lb = (U_[t] - lower_bound);
                double local_margin = std::min(dist_to_ub.minCoeff(), dist_to_lb.minCoeff());
                avg_box_margin += local_margin;
                box_count++;
            }
            if (box_count > 0)
            {
                avg_box_margin /= box_count;
            }
        }
    }

    // We won't let this margin go to zero (to avoid blowing up multipliers).
    if (avg_box_margin < 1e-3)
    {
        avg_box_margin = 1e-3; // Just a fallback
    }
    double control_box_multiplier_init = 1.0 / avg_box_margin; // A simple scaling

    // ------------------------------------------------------------------------
    // Now proceed to initialize the dual and slack variables for each constraint.
    // ------------------------------------------------------------------------
    for (const auto &constraint : constraint_set_)
    {
        std::string constraint_name = constraint.first;
        int dual_dim = constraint.second->getDualDim();

        Y_[constraint_name].resize(horizon_);
        S_[constraint_name].resize(horizon_);
        k_y_[constraint_name].resize(horizon_);
        K_y_[constraint_name].resize(horizon_);
        k_s_[constraint_name].resize(horizon_);
        K_s_[constraint_name].resize(horizon_);

        // Decide initial Y_ based on the constraint type
        for (int t = 0; t < horizon_; ++t)
        {
            // If we recognize the constraint as a "ControlBoxConstraint," we use
            // the tuned multiplier. Otherwise, we use a default.
            if (constraint_name == "ControlBoxConstraint")
            {
                Y_[constraint_name][t] = control_box_multiplier_init * Eigen::VectorXd::Ones(dual_dim);
            }
            else
            {
                // Default initialization
                Y_[constraint_name][t] = 1.0 * Eigen::VectorXd::Ones(dual_dim);
            }

            // Slack initialization remains a small positive constant.
            S_[constraint_name][t] = 0.01 * Eigen::VectorXd::Ones(dual_dim);

            // Gains set to zero.
            k_y_[constraint_name][t].setZero(dual_dim);
            K_y_[constraint_name][t].setZero(dual_dim, state_dim);
            k_s_[constraint_name][t].setZero(dual_dim);
            K_s_[constraint_name][t].setZero(dual_dim, state_dim);
        }
    }

    // Initialize cost
    J_ = objective_->evaluate(X_, U_);

    // Initialize line search parameters
    alphas_.clear();
    alpha_ = options_.backtracking_coeff;
    for (int i = 0; i < options_.max_line_search_iterations; ++i)
    {
        alphas_.push_back(alpha_);
        alpha_ *= options_.backtracking_factor;
    }
    alpha_ = options_.backtracking_coeff;
    dV_.resize(2); // Cost improvement 

    // Initialize regularization parameters
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
    
    // Initialize filter acceptance parameter
    gamma_ = options_.filter_acceptance;
    constraint_violation_ = 0.0;
    
    // Now initialized
    initialized_ = true;
}


CDDPSolution CDDP::solveFeasibleIPDDP()
{
    // Initialize if not done
    if (!initialized_) {
        initializeFeasibleIPDDP();
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
            backward_pass_success = solveFeasibleIPDDPBackwardPass();

            if (!backward_pass_success) {
                std::cerr << "IPDDP: Backward pass failed" << std::endl;

                // Increase regularization and check limit
                increaseRegularization();

                if (isRegularizationLimitReached()) {
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
                ForwardPassResult result = solveFeasibleIPDDPForwardPass(alpha);
                
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
            // Multi-threaded execution 
            std::vector<std::future<ForwardPassResult>> futures;
            futures.reserve(alphas_.size());
            
            // Launch all forward passes in parallel
            for (double alpha : alphas_) {
                futures.push_back(std::async(std::launch::async, 
                    [this, alpha]() { return solveFeasibleIPDDPForwardPass(alpha); }));
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
            decreaseRegularization();

            // Check termination
            if (dJ_ < options_.cost_tolerance || (std::max(optimality_gap_, mu_) < options_.grad_tolerance)) {
                solution.converged = true;
                break;
            }
        } else {
            bool early_termination_flag = false; // TODO: Improve early termination
            // Increase regularization
            increaseRegularization();

            // TODO: Improve early termination
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
            
            // If limit is reached treat as converged
            if (isRegularizationLimitReached()) {
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
    
    if (options_.header_and_footer) {
        printSolution(solution); 
    }

    return solution;
}

bool CDDP::solveFeasibleIPDDPBackwardPass() {
    // Setup
    // Initialize variables
    const int state_dim = getStateDim();
    const int control_dim = getControlDim();
    const int total_dual_dim = getTotalDualDim();

    // Terminal cost and derivatives
    Eigen::VectorXd V_x  = objective_->getFinalCostGradient(X_.back());
    Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose());  // Symmetrize 

    dV_ = Eigen::Vector2d::Zero();
    double Qu_max_norm = 0.0;
    double residual_max = 0.0; // complementary residual measure: r = s ◦ y - mu_
    double dual_norm = 0.0; // Not used for feasible IPDDP

    // Backward Recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Expand cost around (x[t], u[t])
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Continuous dynamics 
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Discretize
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx;
        Eigen::MatrixXd B = timestep_ * Fu;

        // Extract dual variable and constraint values
        Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
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

            // Evaluate constraint
            Eigen::VectorXd g_vec = Eigen::VectorXd::Zero(dual_dim);
            if (cname == "ControlBoxConstraint") {
                Eigen::VectorXd lb = constraint->getLowerBound();
                Eigen::VectorXd ub = constraint->getUpperBound();
                g_vec.head(control_dim) = u - ub;
                g_vec.tail(control_dim) = lb - u;
            }

            // partial wrt. x => g_x
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::MatrixXd g_x = Eigen::MatrixXd::Zero(dual_dim, state_dim);

            // partial wrt. u => g_u
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::MatrixXd g_u = Eigen::MatrixXd::Zero(dual_dim, control_dim);
            if (cname == "ControlBoxConstraint") {
                // top half
                g_u.block(0, 0, dual_dim/2, control_dim) 
                    =  Eigen::MatrixXd::Identity(dual_dim/2, control_dim);
                // bottom half
                g_u.block(control_dim, 0,  dual_dim/2, control_dim) 
                    = -Eigen::MatrixXd::Identity(dual_dim/2, control_dim);
            }

            // Insert into big arrays
            y.segment(offset, dual_dim)   = y_vec;
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
        Eigen::VectorXd Q_u  = l_u + Q_yu.transpose() * y + B.transpose() * V_x;
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

        // Regularization
        Eigen::MatrixXd Q_ux_reg = Q_ux;
        Eigen::MatrixXd Q_uu_reg = Q_uu;

        if (options_.regularization_type == "control" || 
            options_.regularization_type == "both") {
            Q_uu_reg.diagonal().array() += regularization_control_;
        }
        Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

        Eigen::VectorXd r = y.cwiseProduct(g) + mu_ * Eigen::VectorXd::Ones(total_dual_dim);
        Eigen::VectorXd g_safe = g.array() + 1e-8; // Ensure no division by zero
        Eigen::MatrixXd G = g_safe.asDiagonal(); // Diagonal matrix with g as diagonal
        Eigen::MatrixXd G_inv = G.inverse(); // Inverse of G
        Eigen::MatrixXd YGinv = y.asDiagonal() * G_inv; // Y * G^-1 

        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg - Q_yu.transpose() * YGinv * Q_yu);
        if (llt.info() != Eigen::Success) {
            if (options_.debug) {
                std::cerr << "[IPDDP] Backward pass failed at time " << t << std::endl;
            }
            return false;
        }
        
        Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
        bigRHS.col(0) = Q_u - Q_yu.transpose() * G_inv * r;
        Eigen::MatrixXd M =
            Q_ux - Q_yu.transpose() * YGinv * Q_yx;
        for (int col = 0; col < state_dim; col++) {
            bigRHS.col(col+1) = M.col(col);
        }

        Eigen::MatrixXd R = llt.matrixU();
        Eigen::MatrixXd z = R.triangularView<Eigen::Upper>().solve(bigRHS);
        Eigen::MatrixXd kK = -R.transpose().triangularView<Eigen::Lower>().solve(z);

        // parse out feedforward (ku) and feedback (Ku)
        Eigen::VectorXd k_u = kK.col(0); 
        Eigen::MatrixXd K_u(control_dim, state_dim);
        for (int col = 0; col < state_dim; col++) {
            K_u.col(col) = kK.col(col+1);
        }

        // Save gains
        k_u_[t] = k_u;
        K_u_[t] = K_u;

        // For duals
        Eigen::VectorXd k_y = - G_inv * (r + y.asDiagonal() * Q_yu * k_u);
        Eigen::MatrixXd K_y = - YGinv * (Q_yx + Q_yu * K_u);

        offset = 0;
        for (auto &ckv : constraint_set_) {
            const std::string &cname = ckv.first;
            auto &constraint = ckv.second;
            int dual_dim = constraint->getDualDim();

            k_y_[cname][t] = k_y.segment(offset, dual_dim);
            K_y_[cname][t] = K_y.block(offset, 0, dual_dim, state_dim);

            k_s_[cname][t].setZero(dual_dim);
            K_s_[cname][t].setZero(dual_dim, state_dim);

            offset += dual_dim;
        }

        Q_x  -= Q_yx.transpose() * (G_inv * r);
        Q_u  -= Q_yu.transpose() * (G_inv * r);
        Q_xx -= Q_yx.transpose() * YGinv * Q_yx;
        Q_ux -= Q_yu.transpose() * YGinv * Q_yx;
        Q_uu -= Q_yu.transpose() * YGinv * Q_yu;

        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;

        double Qu_norm = Q_u.lpNorm<Eigen::Infinity>();
        if (Qu_norm > Qu_max_norm) Qu_max_norm = Qu_norm;
        double r_norm = r.lpNorm<Eigen::Infinity>();
        if (r_norm > residual_max) residual_max = r_norm; 
    } 

    // Compute optimality gap and print
    optimality_gap_ = std::max(Qu_max_norm, residual_max);

    if (options_.debug) {
        std::cout << "[IPDDP Backward Pass]\n"
                  << "    Qu_max_norm:  " << Qu_max_norm << "\n"
                  << "    residual_max:  " << residual_max << "\n"
                  << "    dV:           " << dV_.transpose() << std::endl;
    }
    return true;
}

ForwardPassResult CDDP::solveFeasibleIPDDPForwardPass(double alpha) {
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.lagrangian = std::numeric_limits<double>::infinity();
    result.alpha = alpha;

    const int state_dim = getStateDim();
    const int control_dim = getControlDim();

    double tau = std::max(0.99, 1.0 - mu_);  

    // Copy old trajectories (from the “previous” solution)
    std::vector<Eigen::VectorXd> X_new = X_;  
    std::vector<Eigen::VectorXd> U_new = U_;  
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;  

    X_new[0] = initial_state_;
    double cost_new = 0.0;
    double log_cost_new = 0.0;
    double primal_residual = 0.0;

    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Current filter point
    FilterPoint current{J_, L_};

    for (int t = 0; t < horizon_; ++t) {
        // 1) Update dual & slack
        for (auto &ckv : constraint_set_) {
            const std::string &cname = ckv.first;
            auto &constraint = ckv.second;
            int dual_dim = ckv.second->getDualDim();

            const Eigen::VectorXd &y_old = Y_[cname][t];
            Eigen::VectorXd y_new = y_old + 
                                    alpha * k_y_[cname][t] + 
                                    K_y_[cname][t] * (X_new[t] - X_[t]); 

            for (int i = 0; i < dual_dim; i++)
            {
                double y_min = (1.0 - tau) * y_old[i];
                if (y_new[i] < y_min)
                {
                    // Too small => fail
                    return result;
                }
                if (y_new[i] < 0.0) {
                    // clamp to small positive => strictly feasible
                    y_new[i] = 1e-8;
                }
            }
            Y_new[cname][t] = y_new;
        }

        // 2) Update control
        const Eigen::VectorXd &u_old = U_[t];
        U_new[t] = u_old + alpha * k_u_[t] + K_u_[t] * (X_new[t] - X_[t]);

        // If desired, clamp control to box if we want explicit feasibility:
        // if (control_box_constraint != nullptr) {
        //     U_new[t] = control_box_constraint->clamp(U_new[t]);
        // }

        // 3) Evaluate cost
        double stage_cost = objective_->running_cost(X_new[t], U_new[t], t);
        cost_new += stage_cost;

        // 4) Evaluate constraints for residual
        for (auto &ckv : constraint_set_)
        {
            const std::string &cname = ckv.first;
            auto &constraint = ckv.second;
            int dual_dim = constraint->getDualDim();

            Eigen::VectorXd g_val = Eigen::VectorXd::Zero(dual_dim);
            if (cname == "ControlBoxConstraint") {
                Eigen::VectorXd lb = constraint->getLowerBound();
                Eigen::VectorXd ub = constraint->getUpperBound();
                g_val.head(control_dim) = U_new[t] - ub;
                g_val.tail(control_dim) = lb - U_new[t];
            }
            Eigen::VectorXd y_val = Y_new[cname][t];
            double local_res = (g_val + y_val).lpNorm<1>();
            primal_residual = std::max(primal_residual, local_res);
        }

        // 5) Step the system
        X_new[t + 1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
    }

    // Terminal cost
    cost_new += objective_->terminal_cost(X_new.back());

    log_cost_new = cost_new;  // For demonstration, ignoring additional barrier terms etc.

    // Build filter point candidate
    FilterPoint candidate{cost_new, log_cost_new};
    bool sufficient_progress = (cost_new < current.cost);
    bool acceptable = sufficient_progress && !current.dominates(candidate);

    if (acceptable)
    {
        double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
        double reduction_ratio = 1.0;
        if (expected > 1e-12) {
            reduction_ratio = (J_ - cost_new) / expected;
        }

        result.success = true;
        result.state_sequence = X_new;
        result.control_sequence = U_new;
        result.dual_sequence = Y_new;
        result.cost = cost_new;
        result.lagrangian = log_cost_new;
        result.constraint_violation = primal_residual;
    }

    return result;
}

} // namespace cddp
