/*
 * Copyright 2024 Tomo Sasaki
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cddp_core/altro_solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

namespace cddp {

AltroSolver::AltroSolver() 
    : cost_(0.0), constraint_violation_(0.0), lagrangian_value_(0.0), optimality_gap_(0.0) {
}

void AltroSolver::initialize(CDDP& context) {
    const CDDPOptions& options = context.getOptions();
    const int state_dim = context.getStateDim();
    const int control_dim = context.getControlDim();
    const int horizon = context.getHorizon();

    // For warm starts, verify that existing state is valid
    if (options.warm_start) {
        bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) &&
                                 K_u_.size() == static_cast<size_t>(horizon) &&
                                 Lambda_.size() == static_cast<size_t>(horizon));

        if (valid_warm_start && !k_u_.empty()) {
            for (int t = 0; t < horizon; ++t) {
                if (k_u_[t].size() != control_dim ||
                    K_u_[t].rows() != control_dim ||
                    K_u_[t].cols() != state_dim ||
                    Lambda_[t].size() != state_dim) {
                    valid_warm_start = false;
                    break;
                }
            }
        } else {
            valid_warm_start = false;
        }

        // Check dual variables validity for warm start
        if (valid_warm_start) {
            const auto& constraint_set = context.getConstraintSet();
            for (const auto& constraint_pair : constraint_set) {
                const std::string& constraint_name = constraint_pair.first;
                int dual_dim = constraint_pair.second->getDualDim();
                
                if (Y_.find(constraint_name) == Y_.end() ||
                    Y_[constraint_name].size() != static_cast<size_t>(horizon)) {
                    valid_warm_start = false;
                    break;
                }
                
                for (int t = 0; t < horizon; ++t) {
                    if (Y_[constraint_name][t].size() != dual_dim) {
                        valid_warm_start = false;
                        break;
                    }
                }
                if (!valid_warm_start) break;
            }
        }

        if (valid_warm_start) {
            if (options.verbose) {
                std::cout << "ALTRO: Using warm start with existing control gains, dual variables, and defect multipliers" << std::endl;
            }
            // Initialize dynamics storage for warm start
            F_.resize(horizon, Eigen::VectorXd::Zero(state_dim));
            evaluateTrajectory(context);
            return;
        } else if (options.verbose) {
            std::cout << "ALTRO: Warning - warm start requested but no valid solver state found. "
                      << "Falling back to cold start initialization." << std::endl;
        }
    }

    // Cold start: full initialization
    k_u_.resize(horizon);
    K_u_.resize(horizon);
    
    for (int t = 0; t < horizon; ++t) {
        k_u_[t] = Eigen::VectorXd::Zero(control_dim);
        K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
    }

    // Initialize dynamics storage
    F_.resize(horizon, Eigen::VectorXd::Zero(state_dim));

    // Initialize Lagrange multipliers for defect constraints
    Lambda_.resize(horizon);
    for (int t = 0; t < horizon; ++t) {
        Lambda_[t] = Eigen::VectorXd::Constant(state_dim, options.altro.defect_dual_init_scale);
    }

    // Initialize dual variables for constraints
    Y_.clear();
    const auto& constraint_set = context.getConstraintSet();
    
    for (const auto& constraint_pair : constraint_set) {
        const std::string& constraint_name = constraint_pair.first;
        const auto& constraint = constraint_pair.second;
        int dual_dim = constraint->getDualDim();
        
        Y_[constraint_name].resize(horizon, Eigen::VectorXd::Zero(dual_dim));
        
        // Initialize dual variables to small positive values
        for (int t = 0; t < horizon; ++t) {
            Y_[constraint_name][t] = Eigen::VectorXd::Constant(dual_dim, options.altro.dual_var_init_scale);
        }
    }

    // Initialize regularization
    context.regularization_ = options.regularization.initial_value;
}

std::string AltroSolver::getSolverName() const {
    return "Altro";
}

CDDPSolution AltroSolver::solve(CDDP& context) {
    const CDDPOptions& options = context.getOptions();

    // Prepare solution map
    CDDPSolution solution;
    solution["solver_name"] = getSolverName();
    solution["status_message"] = std::string("Running");
    solution["iterations_completed"] = 0;
    solution["solve_time_ms"] = 0.0;

    // Initialize history vectors only if requested
    std::vector<double> history_objective;
    std::vector<double> history_lagrangian;
    std::vector<double> history_step_length_primal;
    std::vector<double> history_dual_infeasibility;
    std::vector<double> history_primal_infeasibility;
    std::vector<double> history_penalty_parameter;

    if (options.return_iteration_info) {
        const size_t expected_size = static_cast<size_t>(options.max_iterations + 1);
        history_objective.reserve(expected_size);
        history_lagrangian.reserve(expected_size);
        history_step_length_primal.reserve(expected_size);
        history_dual_infeasibility.reserve(expected_size);
        history_primal_infeasibility.reserve(expected_size);
        history_penalty_parameter.reserve(expected_size);

        // Initial iteration values
        history_objective.push_back(cost_);
        history_lagrangian.push_back(lagrangian_value_);
        history_step_length_primal.push_back(1.0);
        history_dual_infeasibility.push_back(optimality_gap_);
        history_primal_infeasibility.push_back(constraint_violation_);
        history_penalty_parameter.push_back(options.altro.penalty_scaling);
    }

    if (options.verbose) {
        printIteration(0, cost_, lagrangian_value_, 0.0, context.regularization_, 
                      1.0, options.altro.penalty_scaling, constraint_violation_);
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;
    bool converged = false;
    std::string termination_reason = "MaxIterationsReached";

    // Initial trajectory evaluation
    evaluateTrajectory(context);

    // Main ALTRO loop
    while (iter < options.max_iterations) {
        ++iter;

        // Check maximum CPU time
        if (options.max_cpu_time > 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            if (duration.count() > options.max_cpu_time * 1000) {
                termination_reason = "MaxCpuTimeReached";
                if (options.verbose) {
                    std::cerr << "ALTRO: Maximum CPU time reached. Returning current solution" << std::endl;
                }
                break;
            }
        }

        // 1. Backward pass
        bool backward_pass_success = false;
        while (!backward_pass_success) {
            backward_pass_success = backwardPass(context);

            if (!backward_pass_success) {
                context.increaseRegularization();
                if (context.isRegularizationLimitReached()) {
                    termination_reason = "RegularizationLimitReached_NotConverged";
                    if (options.verbose) {
                        std::cerr << "ALTRO: Backward pass regularization limit reached" << std::endl;
                    }
                    break;
                }
            }
        }

        if (!backward_pass_success)
            break;

        // 2. Forward pass
        ForwardPassResult best_result = performForwardPass(context);

        // Update solution if forward pass succeeded
        if (best_result.success) {
            if (options.debug) {
                std::cout << "[ALTRO: Forward pass] " << std::endl;
                std::cout << "    cost: " << best_result.cost << std::endl;
                std::cout << "    merit_function: " << best_result.merit_function << std::endl;
                std::cout << "    alpha: " << best_result.alpha_pr << std::endl;
                std::cout << "    cv_err: " << best_result.constraint_violation << std::endl;
            }
            
            context.X_ = best_result.state_trajectory;
            context.U_ = best_result.control_trajectory;
            if (best_result.dynamics_trajectory) {
                F_ = *best_result.dynamics_trajectory;
            }
            
            double dJ = cost_ - best_result.cost;
            double dL = lagrangian_value_ - best_result.merit_function;
            cost_ = best_result.cost;
            lagrangian_value_ = best_result.merit_function;
            context.alpha_pr_ = best_result.alpha_pr;
            constraint_violation_ = best_result.constraint_violation;

            // Store history only if requested
            if (options.return_iteration_info) {
                history_objective.push_back(cost_);
                history_lagrangian.push_back(lagrangian_value_);
                history_step_length_primal.push_back(context.alpha_pr_);
                history_dual_infeasibility.push_back(optimality_gap_);
                history_primal_infeasibility.push_back(constraint_violation_);
                history_penalty_parameter.push_back(options.altro.penalty_scaling);
            }

            context.decreaseRegularization();

            // Check convergence
            optimality_gap_ = std::abs(dJ) + std::abs(dL);
            
            if (optimality_gap_ <= options.tolerance && 
                constraint_violation_ <= options.altro.constraint_tolerance) {
                converged = true;
                termination_reason = "OptimalSolutionFound";
                break;
            }
            
            if (std::abs(dJ) < options.acceptable_tolerance && 
                constraint_violation_ <= options.altro.constraint_tolerance) {
                converged = true;
                termination_reason = "AcceptableSolutionFound";
                break;
            }
        } else {
            context.increaseRegularization();

            if (context.isRegularizationLimitReached()) {
                termination_reason = "RegularizationLimitReached_NotConverged";
                converged = false;
                if (options.verbose) {
                    std::cerr << "ALTRO: Regularization limit reached. Not converged." << std::endl;
                }
                break;
            }
        }

        // Print iteration info
        if (options.verbose) {
            printIteration(iter, cost_, lagrangian_value_, optimality_gap_,
                          context.regularization_, context.alpha_pr_, 
                          options.altro.penalty_scaling, constraint_violation_);
        }

        // Update augmented Lagrangian parameters
        updateAugmentedLagrangian(context);
    }

    // Compute final timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Populate final solution
    solution["status_message"] = termination_reason;
    solution["iterations_completed"] = iter;
    solution["solve_time_ms"] = static_cast<double>(duration.count());
    solution["final_objective"] = cost_;
    solution["final_step_length"] = context.alpha_pr_;

    // Add trajectories
    std::vector<double> time_points;
    time_points.reserve(static_cast<size_t>(context.getHorizon() + 1));
    for (int t = 0; t <= context.getHorizon(); ++t) {
        time_points.push_back(t * context.getTimestep());
    }
    solution["time_points"] = time_points;
    solution["state_trajectory"] = context.X_;
    solution["control_trajectory"] = context.U_;

    // Add iteration history if requested
    if (options.return_iteration_info) {
        solution["history_objective"] = history_objective;
        solution["history_lagrangian"] = history_lagrangian;
        solution["history_step_length_primal"] = history_step_length_primal;
        solution["history_dual_infeasibility"] = history_dual_infeasibility;
        solution["history_primal_infeasibility"] = history_primal_infeasibility;
        solution["history_penalty_parameter"] = history_penalty_parameter;
    }

    // Add control gains
    solution["control_feedback_gains_K"] = K_u_;

    // Final metrics
    solution["final_regularization"] = context.regularization_;
    solution["final_penalty_parameter"] = options.altro.penalty_scaling;
    solution["final_primal_infeasibility"] = constraint_violation_;
    solution["final_dual_infeasibility"] = optimality_gap_;
    solution["final_lagrangian"] = lagrangian_value_;

    if (options.verbose) {
        printSolutionSummary(solution);
    }

    return solution;
}

void AltroSolver::evaluateTrajectory(CDDP& context) {
    const auto& X = context.X_;
    const auto& U = context.U_;
    const auto& objective = context.getObjective();
    const auto& system = context.getSystem();
    const auto& constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();
    const double timestep = context.getTimestep();
    const double penalty_scaling = context.getOptions().altro.penalty_scaling;
    
    // Compute cost
    cost_ = 0.0;
    for (int t = 0; t < horizon; ++t) {
        cost_ += objective.running_cost(X[t], U[t], t);
        
        // Store dynamics
        F_[t] = system.getDiscreteDynamics(X[t], U[t], t * timestep);
    }
    cost_ += objective.terminal_cost(X.back());
    
    // Compute constraint violation and augmented Lagrangian terms
    constraint_violation_ = 0.0;
    double penalty_cost = 0.0;
    
    for (const auto& constraint_pair : constraint_set) {
        const std::string& constraint_name = constraint_pair.first;
        const auto& constraint = constraint_pair.second;
        
        for (int t = 0; t < horizon; ++t) {
            // Evaluate constraint
            Eigen::VectorXd g = constraint->evaluate(X[t], U[t]) - constraint->getUpperBound();
            
            // Update constraint violation
            constraint_violation_ += std::max(0.0, g.maxCoeff());
            
            // Augmented Lagrangian terms
            const Eigen::VectorXd& y = Y_[constraint_name][t];
            
            for (int i = 0; i < g.size(); ++i) {
                if (g(i) > 0) {
                    // Active constraint: add linear and quadratic penalty terms
                    penalty_cost += y(i) * g(i) + 0.5 * penalty_scaling * g(i) * g(i);
                } else {
                    // Inactive constraint: only quadratic penalty if multiplier is large
                    double projected_multiplier = std::max(0.0, y(i) + penalty_scaling * g(i));
                    if (projected_multiplier > 0) {
                        penalty_cost += 0.5 * penalty_scaling * g(i) * g(i);
                    }
                }
            }
        }
    }
    
    lagrangian_value_ = cost_ + penalty_cost;
}

bool AltroSolver::backwardPass(CDDP& context) {
    const auto& X = context.X_;
    const auto& U = context.U_;
    const auto& objective = context.getObjective();
    const auto& system = context.getSystem();
    const auto& constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();
    const int state_dim = context.getStateDim();
    const int control_dim = context.getControlDim();
    const double timestep = context.getTimestep();
    const double penalty_scaling = context.getOptions().altro.penalty_scaling;
    const bool is_ilqr = context.getOptions().use_ilqr;
    
    // Terminal cost derivatives
    Eigen::VectorXd V_x = objective.getFinalCostGradient(X.back());
    Eigen::MatrixXd V_xx = objective.getFinalCostHessian(X.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose());
    
    // Backward recursion
    for (int t = horizon - 1; t >= 0; --t) {
        const Eigen::VectorXd& x = X[t];
        const Eigen::VectorXd& u = U[t];
        
        // Get dynamics derivatives
        const auto [Fx, Fu] = system.getJacobians(x, u, t * timestep);
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep * Fx;
        Eigen::MatrixXd B = timestep * Fu;
        
        // Cost derivatives
        auto [l_x, l_u] = objective.getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective.getRunningCostHessians(x, u, t);
        
        // Initialize Q-function with cost terms
        Eigen::VectorXd Q_x = l_x + A.transpose() * V_x;
        Eigen::VectorXd Q_u = l_u + B.transpose() * V_x;
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;
        
        // Add constraint terms to Q-function
        for (const auto& constraint_pair : constraint_set) {
            const std::string& constraint_name = constraint_pair.first;
            const auto& constraint = constraint_pair.second;
            const Eigen::VectorXd& y = Y_[constraint_name][t];
            
            // Evaluate constraint and its derivatives
            Eigen::VectorXd g = constraint->evaluate(x, u) - constraint->getUpperBound();
            Eigen::MatrixXd g_x = constraint->getStateJacobian(x, u);
            Eigen::MatrixXd g_u = constraint->getControlJacobian(x, u);
            
            for (int i = 0; i < g.size(); ++i) {
                if (g(i) > 0 || (g(i) <= 0 && y(i) + penalty_scaling * g(i) > 0)) {
                    // Add first-order terms
                    Q_x += (y(i) + penalty_scaling * g(i)) * g_x.row(i).transpose();
                    Q_u += (y(i) + penalty_scaling * g(i)) * g_u.row(i).transpose();
                    
                    // Add second-order terms (Gauss-Newton approximation)
                    Q_xx += penalty_scaling * g_x.row(i).transpose() * g_x.row(i);
                    Q_ux += penalty_scaling * g_u.row(i).transpose() * g_x.row(i);
                    Q_uu += penalty_scaling * g_u.row(i).transpose() * g_u.row(i);
                }
            }
        }
        
        // Regularization
        double reg = context.regularization_;
        Eigen::MatrixXd Q_uu_reg = Q_uu;
        Q_uu_reg.diagonal().array() += reg;
        Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose());
        
        // Solve for control law
        Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
        if (ldlt.info() != Eigen::Success) {
            return false;
        }
        
        k_u_[t] = -ldlt.solve(Q_u);
        K_u_[t] = -ldlt.solve(Q_ux);
        
        // Update value function
        V_x = Q_x + K_u_[t].transpose() * Q_u + Q_ux.transpose() * k_u_[t] + K_u_[t].transpose() * Q_uu * k_u_[t];
        V_xx = Q_xx + K_u_[t].transpose() * Q_ux + Q_ux.transpose() * K_u_[t] + K_u_[t].transpose() * Q_uu * K_u_[t];
        V_xx = 0.5 * (V_xx + V_xx.transpose());
    }
    
    return true;
}

ForwardPassResult AltroSolver::performForwardPass(CDDP& context) {
    const auto& options = context.getOptions();
    
    ForwardPassResult best_result;
    best_result.success = false;
    best_result.merit_function = std::numeric_limits<double>::infinity();
    
    // Try different step sizes from the context
    for (double alpha : context.alphas_) {
        ForwardPassResult result = forwardPass(context, alpha);
        
        if (result.success && result.merit_function < best_result.merit_function) {
            best_result = result;
        }
        
        // Early termination if we found a good step
        if (result.success && result.merit_function < lagrangian_value_) {
            break;
        }
    }
    
    return best_result;
}

ForwardPassResult AltroSolver::forwardPass(CDDP& context, double alpha) {
    ForwardPassResult result;
    result.success = false;
    result.alpha_pr = alpha;
    result.cost = std::numeric_limits<double>::infinity();
    result.merit_function = std::numeric_limits<double>::infinity();
    
    const auto& X = context.X_;
    const auto& U = context.U_;
    const auto& system = context.getSystem();
    const auto& objective = context.getObjective();
    const auto& constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();
    const int state_dim = context.getStateDim();
    const double timestep = context.getTimestep();
    const double penalty_scaling = context.getOptions().altro.penalty_scaling;
    
    // Initialize new trajectories
    std::vector<Eigen::VectorXd> X_new = X;
    std::vector<Eigen::VectorXd> U_new = U;
    std::vector<Eigen::VectorXd> F_new(horizon);
    
    // Set initial state
    X_new[0] = context.getInitialState();
    
    // Forward rollout
    for (int t = 0; t < horizon; ++t) {
        Eigen::VectorXd dx = X_new[t] - X[t];
        
        // Apply control law
        U_new[t] = U[t] + alpha * k_u_[t] + K_u_[t] * dx;
        
        // Check for NaN/Inf
        if (!U_new[t].allFinite()) {
            return result;
        }
        
        // Integrate dynamics
        F_new[t] = system.getDiscreteDynamics(X_new[t], U_new[t], t * timestep);
        
        if (!F_new[t].allFinite()) {
            return result;
        }
        
        X_new[t + 1] = F_new[t];
    }
    
    // Evaluate new trajectory
    double cost_new = 0.0;
    double constraint_violation_new = 0.0;
    double penalty_cost = 0.0;
    
    // Compute cost
    for (int t = 0; t < horizon; ++t) {
        cost_new += objective.running_cost(X_new[t], U_new[t], t);
    }
    cost_new += objective.terminal_cost(X_new.back());
    
    // Compute constraint terms
    for (const auto& constraint_pair : constraint_set) {
        const std::string& constraint_name = constraint_pair.first;
        const auto& constraint = constraint_pair.second;
        
        for (int t = 0; t < horizon; ++t) {
            Eigen::VectorXd g = constraint->evaluate(X_new[t], U_new[t]) - constraint->getUpperBound();
            const Eigen::VectorXd& y = Y_[constraint_name][t];
            
            constraint_violation_new += std::max(0.0, g.maxCoeff());
            
            for (int i = 0; i < g.size(); ++i) {
                if (g(i) > 0) {
                    penalty_cost += y(i) * g(i) + 0.5 * penalty_scaling * g(i) * g(i);
                } else {
                    double projected_multiplier = std::max(0.0, y(i) + penalty_scaling * g(i));
                    if (projected_multiplier > 0) {
                        penalty_cost += 0.5 * penalty_scaling * g(i) * g(i);
                    }
                }
            }
        }
    }
    
    double merit_function_new = cost_new + penalty_cost;
    
    // Simple acceptance test
    if (merit_function_new < lagrangian_value_ || constraint_violation_new < constraint_violation_) {
        result.success = true;
        result.state_trajectory = X_new;
        result.control_trajectory = U_new;
        result.dynamics_trajectory = F_new;
        result.cost = cost_new;
        result.merit_function = merit_function_new;
        result.constraint_violation = constraint_violation_new;
    }
    
    return result;
}

void AltroSolver::updateAugmentedLagrangian(CDDP& context) {
    const auto& X = context.X_;
    const auto& U = context.U_;
    const auto& constraint_set = context.getConstraintSet();
    const int horizon = context.getHorizon();
    const double penalty_scaling = context.getOptions().altro.penalty_scaling;
    
    // Update dual variables (multipliers)
    for (const auto& constraint_pair : constraint_set) {
        const std::string& constraint_name = constraint_pair.first;
        const auto& constraint = constraint_pair.second;
        
        for (int t = 0; t < horizon; ++t) {
            Eigen::VectorXd g = constraint->evaluate(X[t], U[t]) - constraint->getUpperBound();
            Eigen::VectorXd& y = Y_[constraint_name][t];
            
            // Update multipliers: y_new = max(0, y_old + penalty_scaling * g)
            for (int i = 0; i < g.size(); ++i) {
                y(i) = std::max(0.0, y(i) + penalty_scaling * g(i));
            }
        }
    }
}

void AltroSolver::printIteration(int iter, double cost, double lagrangian, double grad_norm,
                                double regularization, double alpha, double mu, double constraint_violation) const {
    if (iter == 0) {
        std::cout << std::setw(4) << "iter" 
                  << std::setw(12) << "cost" 
                  << std::setw(12) << "lagrangian"
                  << std::setw(12) << "grad_norm"
                  << std::setw(8) << "alpha"
                  << std::setw(10) << "penalty"
                  << std::setw(12) << "viol" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }
    
    std::cout << std::setw(4) << iter
              << std::setw(12) << std::scientific << std::setprecision(3) << cost
              << std::setw(12) << std::scientific << std::setprecision(3) << lagrangian
              << std::setw(12) << std::scientific << std::setprecision(3) << grad_norm
              << std::setw(8) << std::fixed << std::setprecision(2) << alpha
              << std::setw(10) << std::scientific << std::setprecision(2) << mu
              << std::setw(12) << std::scientific << std::setprecision(3) << constraint_violation
              << std::endl;
}

void AltroSolver::printSolutionSummary(const CDDPSolution& solution) const {
    std::cout << "\n=== ALTRO Solution Summary ===" << std::endl;
    auto status_it = solution.find("status_message");
    auto iterations_it = solution.find("iterations_completed");
    auto solve_time_it = solution.find("solve_time_ms");
    auto final_cost_it = solution.find("final_objective");
    auto final_lagrangian_it = solution.find("final_lagrangian");
    auto final_alpha_it = solution.find("final_step_length");

    std::cout << "Status: " << (status_it != solution.end() ? std::any_cast<std::string>(status_it->second) : "N/A") << std::endl;
    std::cout << "Iterations: " << (iterations_it != solution.end() ? std::any_cast<int>(iterations_it->second) : -1) << std::endl;
    std::cout << "Solve time: " << (solve_time_it != solution.end() ? std::any_cast<double>(solve_time_it->second) * 1e-3 : -1.0) << " seconds" << std::endl;
    std::cout << "Final cost: " << (final_cost_it != solution.end() ? std::any_cast<double>(final_cost_it->second) : -1.0) << std::endl;
    std::cout << "Final lagrangian: " << (final_lagrangian_it != solution.end() ? std::any_cast<double>(final_lagrangian_it->second) : -1.0) << std::endl;
    std::cout << "Final step size: " << (final_alpha_it != solution.end() ? std::any_cast<double>(final_alpha_it->second) : -1.0) << std::endl;
    std::cout << "================================\n" << std::endl;
}

} // namespace cddp
