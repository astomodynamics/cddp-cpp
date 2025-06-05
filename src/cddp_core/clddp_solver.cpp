/*
 Copyright 2025 Tomo Sasaki

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

#include "cddp_core/clddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include "cddp_core/boxqp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <future>
#include <execution>

namespace cddp {

CLDDPSolver::CLDDPSolver() : boxqp_solver_(BoxQPOptions()) {}

void CLDDPSolver::initialize(CDDP& context) {
    const CDDPOptions& options = context.getOptions();
    
    int horizon = context.getHorizon();
    int control_dim = context.getControlDim();
    int state_dim = context.getStateDim();
    
    // For warm starts, verify that existing state is valid
    if (options.warm_start) {
        // Check if solver state is properly initialized and compatible
        bool valid_warm_start = (k_u_.size() == static_cast<size_t>(horizon) && 
                                K_u_.size() == static_cast<size_t>(horizon));
        
        if (valid_warm_start && !k_u_.empty()) {
            // Verify dimensions are consistent
            for (int t = 0; t < horizon; ++t) {
                if (k_u_[t].size() != control_dim || 
                    K_u_[t].rows() != control_dim || 
                    K_u_[t].cols() != state_dim) {
                    valid_warm_start = false;
                    break;
                }
            }
        } else {
            valid_warm_start = false;
        }
        
        if (valid_warm_start) {
            // Valid warm start: only update what's necessary
            if (options.verbose) {
                std::cout << "CLDDP: Using warm start with existing control gains" << std::endl;
            }
            
            // Only update BoxQP solver options if they changed
            boxqp_solver_.setOptions(options.box_qp);
            
            // Compute cost for current trajectories
            if (!context.X_.empty() && !context.U_.empty()) {
                computeCost(context);
            }
            
            // Keep existing k_u_, K_u_, and other solver state
            return;
        } else {
            // Invalid warm start: fall back to cold start with warning
            if (options.verbose) {
                std::cout << "CLDDP: Warning - warm start requested but no valid solver state found. "
                          << "Falling back to cold start initialization." << std::endl;
            }
        }
    }
    
    // Cold start: full initialization (also used as fallback for invalid warm start)
    k_u_.resize(horizon);
    K_u_.resize(horizon);
    for (int t = 0; t < horizon; ++t) {
        k_u_[t] = Eigen::VectorXd::Zero(control_dim);
        K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
    }
    
    dV_ = Eigen::Vector2d::Zero();
    
    // Setup BoxQP solver
    boxqp_solver_ = BoxQPSolver(options.box_qp);
    
    // Compute initial cost if trajectories exist
    if (!context.X_.empty() && !context.U_.empty()) {
        computeCost(context);
    }
}

CDDPSolution CLDDPSolver::solve(CDDP& context) {
    const CDDPOptions& options = context.getOptions();
    
    // Prepare solution map
    CDDPSolution solution;
    solution["solver_name"] = getSolverName();
    solution["status_message"] = std::string("Running");
    solution["iterations_completed"] = 0;
    solution["solve_time_ms"] = 0.0;
    
    // Initialize history vectors only if requested
    std::vector<double> history_objective;
    std::vector<double> history_merit_function;
    std::vector<double> history_step_length_primal;
    std::vector<double> history_dual_infeasibility;
    std::vector<double> history_regularization;
    
    if (options.return_iteration_info) {
        history_objective.push_back(context.cost_);
        history_merit_function.push_back(context.merit_function_);
        history_dual_infeasibility.push_back(context.inf_du_);
        history_regularization.push_back(context.regularization_);
    }
    
    if (options.verbose) {
        printIteration(0, context.cost_, context.merit_function_, context.inf_du_, context.regularization_, context.alpha_);
    }
    
    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    int iter = 0;
    bool converged = false;
    
    // Main CLDDP loop
    while (iter < options.max_iterations) {
        ++iter;
        
        // Check maximum CPU time
        if (options.max_cpu_time > 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            if (duration.count() > options.max_cpu_time * 1000) {
                if (options.verbose) {
                    std::cerr << "CLDDP: Maximum CPU time reached. Returning current solution" << std::endl;
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
                    if (options.verbose) {
                        std::cerr << "CLDDP: Backward pass regularization limit reached" << std::endl;
                    }
                    break;
                }
            }
        }
        
        if (!backward_pass_success) break;
        
        // 2. Forward pass
        ForwardPassResult best_result = performForwardPass(context);
        
        // Update solution if forward pass succeeded
        if (best_result.success) {
            context.X_ = best_result.state_trajectory;
            context.U_ = best_result.control_trajectory;
            double dJ = context.cost_ - best_result.cost;
            context.cost_ = best_result.cost;
            context.merit_function_ = best_result.merit_function;
            context.alpha_ = best_result.alpha;
            
            // Store history only if requested
            if (options.return_iteration_info) {
                history_objective.push_back(context.cost_);
                history_merit_function.push_back(context.merit_function_);
                history_step_length_primal.push_back(context.alpha_);
                history_dual_infeasibility.push_back(context.inf_du_);
                history_regularization.push_back(context.regularization_);
            }
            
            context.decreaseRegularization();
            
            // Check convergence
            if (dJ < options.tolerance) {
                converged = true;
                break;
            }
        } else {
            context.increaseRegularization();
            
            // Check if regularization limit reached
            if (context.isRegularizationLimitReached()) {
                double dJ = 0.0; // No cost improvement
                if (dJ < options.acceptable_tolerance) {
                    converged = true;
                } else {
                    converged = false;
                }
                if (options.verbose) {
                    std::cerr << "CLDDP: Regularization limit reached. "
                              << (converged ? "Treating as converged." : "Not converged.")
                              << std::endl;
                }
                break;
            }
        }
        
        // Print iteration info
        if (options.verbose) {
            printIteration(iter, context.cost_, context.merit_function_, context.inf_du_, context.regularization_, context.alpha_);
        }
    }
    
    // Compute final timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Populate final solution
    solution["status_message"] = converged ? std::string("OptimalSolutionFound") : std::string("MaxIterationsReached");
    solution["iterations_completed"] = iter;
    solution["solve_time_ms"] = static_cast<double>(duration.count());
    solution["final_objective"] = context.cost_;
    solution["final_step_length"] = context.alpha_;
    
    // Add trajectories
    std::vector<double> time_points;
    for (int t = 0; t <= context.getHorizon(); ++t) {
        time_points.push_back(t * context.getTimestep());
    }
    solution["time_points"] = time_points;
    solution["state_trajectory"] = context.X_;
    solution["control_trajectory"] = context.U_;
    
    // Add iteration history if requested
    if (options.return_iteration_info) {
        solution["history_objective"] = history_objective;
        solution["history_merit_function"] = history_merit_function;
        solution["history_step_length_primal"] = history_step_length_primal;
        solution["history_dual_infeasibility"] = history_dual_infeasibility;
        std::map<std::string, std::vector<double>> history_regularization_map;
        history_regularization_map["control"] = history_regularization;
        solution["history_regularization"] = history_regularization_map;
    }
    
    // Add control gains
    solution["control_feedforward_gains_k"] = k_u_;
    
    // Final metrics
    std::map<std::string, double> final_regularization;
    final_regularization["control"] = context.regularization_;
    solution["final_regularization_values"] = final_regularization;
    
    if (options.verbose) {
        printSolutionSummary(solution);
    }
    
    return solution;
}

std::string CLDDPSolver::getSolverName() const {
    return "CLDDP";
}

bool CLDDPSolver::backwardPass(CDDP& context) {
    const CDDPOptions& options = context.getOptions();
    const int state_dim = context.getStateDim();
    const int control_dim = context.getControlDim();
    const int horizon = context.getHorizon();
    
    // Extract control box constraint
    auto control_box_constraint = context.getConstraint<ControlBoxConstraint>("ControlBoxConstraint");
    
    // Terminal cost and its derivatives
    Eigen::VectorXd V_x = context.getObjective().getFinalCostGradient(context.X_.back());
    Eigen::MatrixXd V_xx = context.getObjective().getFinalCostHessian(context.X_.back());
    
    // Pre-allocate matrices
    Eigen::MatrixXd A(state_dim, state_dim);
    Eigen::MatrixXd B(state_dim, control_dim);
    Eigen::VectorXd Q_x(state_dim);
    Eigen::VectorXd Q_u(control_dim);
    Eigen::MatrixXd Q_xx(state_dim, state_dim);
    Eigen::MatrixXd Q_uu(control_dim, control_dim);
    Eigen::MatrixXd Q_uu_reg(control_dim, control_dim);
    Eigen::MatrixXd Q_ux(control_dim, state_dim);
    Eigen::VectorXd k(control_dim);
    Eigen::MatrixXd K(control_dim, state_dim);
    
    dV_ = Eigen::Vector2d::Zero();
    double Qu_error = 0.0;
    
    // Backward Riccati recursion
    for (int t = horizon - 1; t >= 0; --t) {
        const Eigen::VectorXd& x = context.X_[t];
        const Eigen::VectorXd& u = context.U_[t];
        
        // Get continuous dynamics Jacobians
        const auto [Fx, Fu] = context.getSystem().getJacobians(x, u, t * context.getTimestep());
        
        // Convert to discrete time
        A = context.getTimestep() * Fx;
        A.diagonal().array() += 1.0;
        B = context.getTimestep() * Fu;
        
        // Get cost and its derivatives
        auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = context.getObjective().getRunningCostHessians(x, u, t);
        
        // Compute Q-function matrices
        Q_x = l_x + A.transpose() * V_x;
        Q_u = l_u + B.transpose() * V_x;
        Q_xx = l_xx + A.transpose() * V_xx * A;
        Q_ux = l_ux + B.transpose() * V_xx * A;
        Q_uu = l_uu + B.transpose() * V_xx * B;
        
        // Apply regularization
        Q_uu_reg = Q_uu;
        Q_uu_reg.diagonal().array() += context.regularization_;
        
        // Check positive definiteness
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu_reg);
        if (es.eigenvalues().real().minCoeff() <= 0) {
            if (options.debug) {
                std::cerr << "CLDDP: Q_uu is not positive definite at time " << t << std::endl;
            }
            return false;
        }
        
        // Solve for control law
        if (control_box_constraint == nullptr) {
            const Eigen::MatrixXd H = Q_uu_reg.inverse();
            k = -H * Q_u;
            K = -H * Q_ux;
        } else {
            // Solve constrained QP
            const Eigen::VectorXd lb = control_box_constraint->getLowerBound() - u;
            const Eigen::VectorXd ub = control_box_constraint->getUpperBound() - u;
            const Eigen::VectorXd x0 = k_u_[t];
            
            BoxQPResult qp_result = boxqp_solver_.solve(Q_uu_reg, Q_u, lb, ub, x0);
            
            if (qp_result.status == BoxQPStatus::HESSIAN_NOT_PD || 
                qp_result.status == BoxQPStatus::NO_DESCENT) {
                if (options.debug) {
                    std::cerr << "CLDDP: BoxQP failed at time step " << t << std::endl;
                }
                return false;
            }
            
            k = qp_result.x;
            
            // Compute feedback gain
            K = Eigen::MatrixXd::Zero(control_dim, state_dim);
            if (qp_result.free.sum() > 0) {
                std::vector<int> free_idx;
                for (int i = 0; i < control_dim; i++) {
                    if (qp_result.free(i)) {
                        free_idx.push_back(i);
                    }
                }
                
                Eigen::MatrixXd Q_ux_free(free_idx.size(), state_dim);
                for (size_t i = 0; i < free_idx.size(); i++) {
                    Q_ux_free.row(i) = Q_ux.row(free_idx[i]);
                }
                
                Eigen::MatrixXd K_free = -qp_result.Hfree.solve(Q_ux_free);
                
                for (size_t i = 0; i < free_idx.size(); i++) {
                    K.row(free_idx[i]) = K_free.row(i);
                }
            }
        }
        
        // Store gains
        k_u_[t] = k;
        K_u_[t] = K;
        
        // Update value function
        Eigen::Vector2d dV_step;
        dV_step << Q_u.dot(k), 0.5 * k.dot(Q_uu * k);
        dV_ += dV_step;
        
        V_x = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize
        
        // Update optimality gap
        Qu_error = std::max(Qu_error, Q_u.lpNorm<Eigen::Infinity>());
    }
    
    context.inf_du_ = Qu_error;
    
    if (options.debug) {
        std::cout << "Qu_error: " << Qu_error << std::endl;
        std::cout << "dV: " << dV_.transpose() << std::endl;
    }
    
    return true;
}

ForwardPassResult CLDDPSolver::performForwardPass(CDDP& context) {
    const CDDPOptions& options = context.getOptions();
    ForwardPassResult best_result;
    best_result.cost = std::numeric_limits<double>::infinity();
    best_result.success = false;
    
    if (!options.enable_parallel) {
        // Single-threaded execution with early termination
        for (double alpha : context.alphas_) {
            ForwardPassResult result = forwardPass(context, alpha);
            
            if (result.success && result.cost < best_result.cost) {
                best_result = result;
                if (result.success) {
                    if (options.debug) {
                        std::cout << "CLDDP: Early termination due to successful forward pass" << std::endl;
                    }
                    break; // Early termination
                }
            }
        }
    } else {
        // Multi-threaded execution
        std::vector<std::future<ForwardPassResult>> futures;
        futures.reserve(context.alphas_.size());
        
        for (double alpha : context.alphas_) {
            futures.push_back(std::async(std::launch::async, 
                [this, &context, alpha]() { return forwardPass(context, alpha); }));
        }
        
        for (auto& future : futures) {
            try {
                if (future.valid()) {
                    ForwardPassResult result = future.get();
                    if (result.success && result.cost < best_result.cost) {
                        best_result = result;
                    }
                }
            } catch (const std::exception& e) {
                if (options.verbose) {
                    std::cerr << "CLDDP: Forward pass thread failed: " << e.what() << std::endl;
                }
            }
        }
    }
    
    return best_result;
}

ForwardPassResult CLDDPSolver::forwardPass(CDDP& context, double alpha) {
    const CDDPOptions& options = context.getOptions();
    
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.merit_function = std::numeric_limits<double>::infinity();
    result.alpha = alpha;
    
    // Initialize trajectories
    result.state_trajectory = context.X_;
    result.control_trajectory = context.U_;
    result.state_trajectory[0] = context.getInitialState();
    
    double J_new = 0.0;
    auto control_box_constraint = context.getConstraint<ControlBoxConstraint>("ControlBoxConstraint");
    
    // Forward simulation
    for (int t = 0; t < context.getHorizon(); ++t) {
        const Eigen::VectorXd& x = result.state_trajectory[t];
        const Eigen::VectorXd& u = result.control_trajectory[t];
        const Eigen::VectorXd delta_x = x - context.X_[t];
        
        // Apply control update
        result.control_trajectory[t] = u + alpha * k_u_[t] + K_u_[t] * delta_x;
        
        // Apply control constraints
        if (control_box_constraint != nullptr) {
            result.control_trajectory[t] = control_box_constraint->clamp(result.control_trajectory[t]);
        }
        
        // Compute running cost
        J_new += context.getObjective().running_cost(x, result.control_trajectory[t], t);
        
        // Propagate dynamics
        result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
            x, result.control_trajectory[t], t * context.getTimestep());
    }
    
    // Add terminal cost
    J_new += context.getObjective().terminal_cost(result.state_trajectory.back());
    
    // Check improvement
    double dJ = context.cost_ - J_new;
    double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
    double reduction_ratio = expected > 0.0 ? dJ / expected : std::copysign(1.0, dJ);
    
    // Use the Armijo constant from filter options for line search acceptance
    result.success = reduction_ratio > options.filter.armijo_constant;
    result.cost = J_new;
    result.merit_function = J_new; // For CLDDP, merit function equals cost
    
    return result;
}

void CLDDPSolver::computeCost(CDDP& context) {
    context.cost_ = 0.0;
    
    // Running costs
    for (int t = 0; t < context.getHorizon(); ++t) {
        context.cost_ += context.getObjective().running_cost(context.X_[t], context.U_[t], t);
    }
    
    // Terminal cost
    context.cost_ += context.getObjective().terminal_cost(context.X_.back());
    context.merit_function_ = context.cost_; // For CLDDP, merit function equals cost
}

void CLDDPSolver::printIteration(int iter, double cost, double merit, double inf_du, 
                                 double regularization, double alpha) const {
    if (iter == 0) {
        std::cout << std::setw(4) << "iter" << " "
                  << std::setw(12) << "objective" << " "
                  << std::setw(10) << "inf_du" << " "
                  << std::setw(8) << "lg(rg)" << " "
                  << std::setw(8) << "alpha" << std::endl;
    }
    
    std::cout << std::setw(4) << iter << " "
              << std::setw(12) << std::scientific << std::setprecision(4) << cost << " "
              << std::setw(10) << std::scientific << std::setprecision(2) << inf_du << " "
              << std::setw(8) << std::fixed << std::setprecision(1) << std::log10(regularization) << " "
              << std::setw(8) << std::fixed << std::setprecision(4) << alpha << std::endl;
}

void CLDDPSolver::printSolutionSummary(const CDDPSolution& solution) const {
    std::cout << "\n========================================\n";
    std::cout << "           CLDDP Solution Summary\n";
    std::cout << "========================================\n";
    
    auto iterations = std::any_cast<int>(solution.at("iterations_completed"));
    auto solve_time = std::any_cast<double>(solution.at("solve_time_ms"));
    auto final_cost = std::any_cast<double>(solution.at("final_objective"));
    auto status = std::any_cast<std::string>(solution.at("status_message"));
    
    std::cout << "Status: " << status << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Solve Time: " << std::setprecision(2) << solve_time << " ms\n";
    std::cout << "Final Cost: " << std::setprecision(6) << final_cost << "\n";
    std::cout << "========================================\n\n";
}

} // namespace cddp
