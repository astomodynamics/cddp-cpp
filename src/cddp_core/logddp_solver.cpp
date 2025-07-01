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

#include "cddp_core/logddp_solver.hpp"
#include "cddp_core/cddp_core.hpp"
#include <chrono>
#include <cmath>
#include <execution>
#include <future>
#include <iomanip>
#include <iostream>

namespace cddp {

LogDDPSolver::LogDDPSolver()
    : mu_(1e-1), relaxation_delta_(1e-5), constraint_violation_(1e+7),
      ms_segment_length_(5) {}

void LogDDPSolver::initialize(CDDP &context) {
  const CDDPOptions &options = context.getOptions();

  int horizon = context.getHorizon();
  int control_dim = context.getControlDim();
  int state_dim = context.getStateDim();

  // Check if reference_state in objective and reference_state in context are
  // the same
  if ((context.getReferenceState() - context.getObjective().getReferenceState())
          .norm() > 1e-6) {
    std::cerr << "LogDDP: Initial state and goal state in the objective "
                 "function do not match"
              << std::endl;
    throw std::runtime_error(
        "Initial state and goal state in the objective function do not match");
  }

  // For warm starts, verify that existing state is valid
  if (options.warm_start) {
    bool valid_warm_start =
        (k_u_.size() == static_cast<size_t>(horizon) &&
         K_u_.size() == static_cast<size_t>(horizon) &&
         F_.size() == static_cast<size_t>(horizon) &&
         context.X_.size() == static_cast<size_t>(horizon + 1) &&
         context.U_.size() == static_cast<size_t>(horizon));

    if (valid_warm_start && !k_u_.empty()) {
      for (int t = 0; t < horizon; ++t) {
        if (k_u_[t].size() != control_dim || K_u_[t].rows() != control_dim ||
            K_u_[t].cols() != state_dim || F_[t].size() != state_dim) {
          valid_warm_start = false;
          break;
        }
      }
    } else {
      valid_warm_start = false;
    }

    if (valid_warm_start) {
      if (options.verbose) {
        std::cout << "LogDDP: Using warm start with existing control gains"
                  << std::endl;
      }

      // Reset barrier parameters for warm start
      mu_ = options.log_barrier.barrier.mu_initial;
      relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
      if (!relaxed_log_barrier_) {
        relaxed_log_barrier_ =
            std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);
      } else {
        relaxed_log_barrier_->setBarrierCoeff(mu_);
        relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
      }

      // Initialize regularization
      context.regularization_ = options.regularization.initial_value;

      // Initialize line search parameters
      context.alphas_.clear();
      double alpha = options.line_search.initial_step_size;
      for (int i = 0; i < options.line_search.max_iterations; ++i) {
        context.alphas_.push_back(alpha);
        alpha *= options.line_search.step_reduction_factor;
      }
      context.alpha_pr_ = options.line_search.initial_step_size;
      dV_ = Eigen::Vector2d::Zero();

      constraint_violation_ = std::numeric_limits<double>::infinity();
      ms_segment_length_ = options.log_barrier.segment_length;

      // Evaluate current trajectory and reset filter
      evaluateTrajectory(context);
      resetFilter(context);
      return;
    } else if (options.verbose) {
      std::cout << "LogDDP: Warning - warm start requested but no valid solver "
                   "state found. "
                << "Falling back to cold start initialization." << std::endl;
    }
  }

  // Cold start: Initialize trajectories with interpolation between initial and
  // reference states
  if (context.X_.size() != static_cast<size_t>(horizon + 1) ||
      context.U_.size() != static_cast<size_t>(horizon)) {
    context.X_.resize(horizon + 1);
    context.U_.resize(horizon);

    // Create X_ initial guess using initial_state and reference_state by
    // interpolating between them
    for (int t = 0; t <= horizon; ++t) {
      context.X_[t] =
          context.getInitialState() +
          t * (context.getReferenceState() - context.getInitialState()) /
              horizon;
    }

    for (int t = 0; t < horizon; ++t) {
      context.U_[t] = Eigen::VectorXd::Zero(control_dim);
    }
  }

  // Resize linearized dynamics storage
  F_.resize(horizon);
  F_x_.resize(horizon);
  F_u_.resize(horizon);
  F_xx_.resize(horizon);
  F_uu_.resize(horizon);
  F_ux_.resize(horizon);

  for (int t = 0; t < horizon; ++t) {
    F_[t] = Eigen::VectorXd::Zero(state_dim);
  }

  k_u_.resize(horizon);
  K_u_.resize(horizon);

  for (int t = 0; t < horizon; ++t) {
    k_u_[t] = Eigen::VectorXd::Zero(control_dim);
    K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
  }

  G_.clear();

  const auto &constraint_set = context.getConstraintSet();
  for (const auto &constraint_pair : constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    int dual_dim = constraint_pair.second->getDualDim();

    G_[constraint_name].resize(horizon);
    for (int t = 0; t < horizon; ++t) {
      G_[constraint_name][t] = Eigen::VectorXd::Zero(dual_dim);
    }
  }

  // Initialize cost using objective evaluation
  context.cost_ = context.getObjective().evaluate(context.X_, context.U_);

  // Initialize line search parameters
  context.alphas_.clear();
  double alpha = options.line_search.initial_step_size;
  for (int i = 0; i < options.line_search.max_iterations; ++i) {
    context.alphas_.push_back(alpha);
    alpha *= options.line_search.step_reduction_factor;
  }
  context.alpha_pr_ = options.line_search.initial_step_size;
  dV_ = Eigen::Vector2d::Zero();

  // Initialize regularization
  context.regularization_ = options.regularization.initial_value;

  constraint_violation_ = std::numeric_limits<double>::infinity();

  ms_segment_length_ = options.log_barrier.segment_length;

  // Check if ms_segment_length_ is valid
  if (ms_segment_length_ < 0) {
    std::cerr << "LogDDP: ms_segment_length_ must be non-negative" << std::endl;
    throw std::runtime_error("LogDDP: ms_segment_length_ must be non-negative");
  }

  const std::string &rollout_type = options.log_barrier.rollout_type;
  if (rollout_type != "linear" && rollout_type != "nonlinear" &&
      rollout_type != "hybrid") {
    std::cerr << "LogDDP: Invalid ms_rollout_type: " << rollout_type
              << std::endl;
    throw std::runtime_error("LogDDP: Invalid ms_rollout_type");
  }

  // Initialize log barrier object
  mu_ = options.log_barrier.barrier.mu_initial;
  relaxation_delta_ = options.log_barrier.relaxed_log_barrier_delta;
  if (!relaxed_log_barrier_) {
    relaxed_log_barrier_ =
        std::make_unique<RelaxedLogBarrier>(mu_, relaxation_delta_);
  }

  // Evaluate initial trajectory
  evaluateTrajectory(context);
  resetFilter(context);
}

std::string LogDDPSolver::getSolverName() const { return "LogDDP"; }

CDDPSolution LogDDPSolver::solve(CDDP &context) {
  const CDDPOptions &options = context.getOptions();

  // Prepare solution map with old-style structure for compatibility
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
  std::vector<double> history_barrier_mu;

  if (options.return_iteration_info) {
    const size_t expected_size =
        static_cast<size_t>(options.max_iterations + 1);
    history_objective.reserve(expected_size);
    history_lagrangian.reserve(expected_size);
    history_step_length_primal.reserve(expected_size);
    history_dual_infeasibility.reserve(expected_size);
    history_primal_infeasibility.reserve(expected_size);
    history_barrier_mu.reserve(expected_size);
  }

  // Initialize trajectories and gaps
  evaluateTrajectory(context); // context.cost_ is computed inside this function
  if (options.return_iteration_info) {
    history_objective.push_back(context.cost_);
  }

  // Reset LogDDP filter
  resetFilter(context); // L_ and constraint_violation_ are computed inside this
                        // function
  if (options.return_iteration_info) {
    history_lagrangian.push_back(context.merit_function_);
    history_dual_infeasibility.push_back(context.inf_du_);
    history_primal_infeasibility.push_back(constraint_violation_);
    history_barrier_mu.push_back(mu_);
  }

  if (options.verbose) {
    printIteration(0, context.cost_, context.merit_function_, context.inf_du_,
                   context.regularization_, context.alpha_pr_, mu_,
                   constraint_violation_);
  }

  // Start timer
  auto start_time = std::chrono::high_resolution_clock::now();
  int iter = 0;
  bool converged = false;
  std::string termination_reason = "MaxIterationsReached";
  double dJ = 0.0;
  double dL = 0.0;

  // Main loop of LogDDP
  while (iter < options.max_iterations) {
    ++iter;

    // Check maximum CPU time
    if (options.max_cpu_time > 0) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          current_time - start_time);
      if (duration.count() > options.max_cpu_time * 1000) {
        termination_reason = "MaxCpuTimeReached";
        if (options.verbose) {
          std::cerr
              << "LogDDP: Maximum CPU time reached. Returning current solution"
              << std::endl;
        }
        break;
      }
    }

    // 1. Backward pass: Solve Riccati recursion to compute optimal control law
    bool backward_pass_success = false;
    while (!backward_pass_success) {
      backward_pass_success = backwardPass(context);

      if (!backward_pass_success) {
        if (options.debug) {
          std::cerr << "LogDDP: Backward pass failed" << std::endl;
        }

        context.increaseRegularization();

        if (context.isRegularizationLimitReached()) {
          if (options.verbose) {
            std::cerr << "LogDDP: Backward pass regularization limit reached!"
                      << std::endl;
          }
          converged = true;
          termination_reason = "RegularizationLimitReached_Converged";
          break;
        }
        continue;
      }
    }

    if (converged) {
      break;
    }

    // 2. Forward pass
    ForwardPassResult best_result = performForwardPass(context);

    bool forward_pass_success = best_result.success;

    // Update solution if a feasible forward pass was found
    if (forward_pass_success) {
      if (options.debug) {
        std::cout << "[LogDDP: Forward pass] " << std::endl;
        std::cout << "    cost: " << best_result.cost << std::endl;
        std::cout << "    logcost: " << best_result.merit_function << std::endl;
        std::cout << "    alpha: " << best_result.alpha_pr << std::endl;
        std::cout << "    rf_err: " << best_result.constraint_violation
                  << std::endl;
      }

      context.X_ = best_result.state_trajectory;
      context.U_ = best_result.control_trajectory;
      if (best_result.dynamics_trajectory)
        F_ = *best_result.dynamics_trajectory;

      dJ = context.cost_ - best_result.cost;
      context.cost_ = best_result.cost;
      dL = context.merit_function_ - best_result.merit_function;
      context.merit_function_ = best_result.merit_function;
      context.alpha_pr_ = best_result.alpha_pr;
      constraint_violation_ = best_result.constraint_violation;

      if (options.return_iteration_info) {
        history_objective.push_back(context.cost_);
        history_lagrangian.push_back(context.merit_function_);
        history_step_length_primal.push_back(context.alpha_pr_);
        history_dual_infeasibility.push_back(context.inf_du_);
        history_primal_infeasibility.push_back(constraint_violation_);
        history_barrier_mu.push_back(mu_);
      }

      context.decreaseRegularization();
    } else {
      context.increaseRegularization();

      if (context.isRegularizationLimitReached()) {
        if (options.debug) {
          std::cerr << "LogDDP: Forward Pass regularization limit reached"
                    << std::endl;
        }
        converged = false;
        termination_reason = "RegularizationLimitReached_NotConverged";
        break;
      }
    }

    // Print iteration information
    if (options.verbose) {
      printIteration(iter, context.cost_, context.merit_function_,
                     context.inf_du_, context.regularization_,
                     context.alpha_pr_, mu_, constraint_violation_);
    }

    // Check termination
    double termination_metric = std::max(context.inf_du_, context.inf_pr_);
    if (termination_metric <= options.tolerance) {
      if (options.debug) {
        std::cout << "LogDDP: Converged due to optimality gap and constraint "
                     "violation."
                  << std::endl;
      }
      converged = true;
      termination_reason = "OptimalSolutionFound";
      break;
    }

    if (std::abs(dJ) < options.acceptable_tolerance &&
        std::abs(dL) < options.acceptable_tolerance) {
      if (options.debug) {
        std::cout
            << "LogDDP: Converged due to small change in cost and Lagrangian."
            << std::endl;
      }
      converged = true;
      termination_reason = "AcceptableSolutionFound";
      break;
    }

    // Barrier update logic
    if (forward_pass_success) {
      mu_ = std::max(options.log_barrier.barrier.mu_min_value,
                     mu_ * options.log_barrier.barrier.mu_update_factor);
    } else {
      mu_ = std::min(options.log_barrier.barrier.mu_initial, mu_ * 5.0);
    }
    relaxed_log_barrier_->setBarrierCoeff(mu_);
    relaxed_log_barrier_->setRelaxationDelta(relaxation_delta_);
    resetFilter(context);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // Populate final solution
  solution["status_message"] = termination_reason;
  solution["iterations_completed"] = iter;
  solution["solve_time_ms"] = static_cast<double>(duration.count());
  solution["final_objective"] = context.cost_;
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
    solution["history_merit_function"] = history_lagrangian;
    solution["history_step_length_primal"] = history_step_length_primal;
    solution["history_dual_infeasibility"] = history_dual_infeasibility;
    solution["history_primal_infeasibility"] = history_primal_infeasibility;
    solution["history_barrier_mu"] = history_barrier_mu;
  }

  // Add control gains
  solution["control_feedback_gains_K"] = K_u_;

  // Final metrics
  solution["final_regularization"] = context.regularization_;
  solution["final_barrier_parameter_mu"] = mu_;
  solution["final_primal_infeasibility"] = constraint_violation_;
  solution["final_dual_infeasibility"] = context.inf_du_;

  if (options.verbose) {
    printSolutionSummary(solution);
  }

  return solution;
}

void LogDDPSolver::evaluateTrajectory(CDDP &context) {
  const int horizon = context.getHorizon();
  double cost = 0.0;

  // Rollout dynamics and calculate cost
  for (int t = 0; t < horizon; ++t) {
    const Eigen::VectorXd &x_t = context.X_[t];
    const Eigen::VectorXd &u_t = context.U_[t];

    // Compute stage cost using the guessed state/control
    cost += context.getObjective().running_cost(x_t, u_t, t);

    // Compute dynamics
    F_[t] = context.getSystem().getDiscreteDynamics(x_t, u_t,
                                                    t * context.getTimestep());
  }

  // Add terminal cost based on the final guessed state
  cost += context.getObjective().terminal_cost(context.X_.back());

  // Store the initial total cost
  context.cost_ = cost;
}

void LogDDPSolver::resetFilter(CDDP &context) {
  // Evaluate log-barrier cost (includes path constraints)
  context.merit_function_ = context.cost_;
  constraint_violation_ = 0.0;
  double defect_violation = 0.0;

  const auto &constraint_set = context.getConstraintSet();

  // Calculate path constraint terms and violation
  for (int t = 0; t < context.getHorizon(); ++t) {
    for (const auto &constraint_pair : constraint_set) {
      const std::string &constraint_name = constraint_pair.first;
      Eigen::VectorXd g_t =
          constraint_pair.second->evaluate(context.X_[t], context.U_[t]) -
          constraint_pair.second->getUpperBound();
      G_[constraint_name][t] = g_t;
      context.merit_function_ += relaxed_log_barrier_->evaluate(
          *constraint_pair.second, context.X_[t], context.U_[t]);

      for (int i = 0; i < g_t.size(); ++i) {
        if (g_t(i) > 0.0) {
          constraint_violation_ += g_t(i);
        }
      }
    }

    // Add defect violation penalty
    Eigen::VectorXd d = F_[t] - context.X_[t + 1];
    defect_violation += d.lpNorm<1>();
  }
  constraint_violation_ += defect_violation;
  context.inf_pr_ = constraint_violation_;
}

void LogDDPSolver::precomputeDynamicsDerivatives(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();
  const double timestep = context.getTimestep();

  // Resize storage
  F_x_.resize(horizon);
  F_u_.resize(horizon);
  F_xx_.resize(horizon);
  F_uu_.resize(horizon);
  F_ux_.resize(horizon);

  // Threshold for when parallelization is worth it
  const int MIN_HORIZON_FOR_PARALLEL = 20;
  const bool use_parallel =
      options.enable_parallel && horizon >= MIN_HORIZON_FOR_PARALLEL;

  if (!use_parallel) {
    // Single-threaded computation - always efficient for small horizons
    for (int t = 0; t < horizon; ++t) {
      const Eigen::VectorXd &x = context.X_[t];
      const Eigen::VectorXd &u = context.U_[t];

      // Compute jacobians
      const auto [Fx, Fu] =
          context.getSystem().getJacobians(x, u, t * timestep);
      F_x_[t] = Fx;
      F_u_[t] = Fu;

      // Compute hessians if not using iLQR
      if (!options.use_ilqr) {
        const auto hessians =
            context.getSystem().getHessians(x, u, t * timestep);
        F_xx_[t] = std::get<0>(hessians);
        F_uu_[t] = std::get<1>(hessians);
        F_ux_[t] = std::get<2>(hessians);
      } else {
        // Initialize empty hessians for iLQR
        F_xx_[t] = std::vector<Eigen::MatrixXd>();
        F_uu_[t] = std::vector<Eigen::MatrixXd>();
        F_ux_[t] = std::vector<Eigen::MatrixXd>();
      }
    }
  } else {
    // Chunked parallel computation - much more efficient
    const int num_threads =
        std::min(options.num_threads,
                 static_cast<int>(std::thread::hardware_concurrency()));
    const int chunk_size = std::max(1, horizon / num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      int start_t = thread_id * chunk_size;
      int end_t = (thread_id == num_threads - 1) ? horizon
                                                 : (thread_id + 1) * chunk_size;

      if (start_t >= horizon)
        break;

      futures.push_back(
          std::async(std::launch::async, [this, &context, &options, start_t,
                                          end_t, timestep]() {
            // Process a chunk of time steps
            for (int t = start_t; t < end_t; ++t) {
              const Eigen::VectorXd &x = context.X_[t];
              const Eigen::VectorXd &u = context.U_[t];

              // Compute jacobians
              const auto [Fx, Fu] =
                  context.getSystem().getJacobians(x, u, t * timestep);
              F_x_[t] = Fx;
              F_u_[t] = Fu;

              // Compute hessians if not using iLQR
              if (!options.use_ilqr) {
                const auto hessians =
                    context.getSystem().getHessians(x, u, t * timestep);
                F_xx_[t] = std::get<0>(hessians);
                F_uu_[t] = std::get<1>(hessians);
                F_ux_[t] = std::get<2>(hessians);
              } else {
                // Initialize empty hessians for iLQR
                F_xx_[t] = std::vector<Eigen::MatrixXd>();
                F_uu_[t] = std::vector<Eigen::MatrixXd>();
                F_ux_[t] = std::vector<Eigen::MatrixXd>();
              }
            }
          }));
    }

    // Wait for all computations to complete
    for (auto &future : futures) {
      try {
        if (future.valid()) {
          future.get();
        }
      } catch (const std::exception &e) {
        if (options.verbose) {
          std::cerr
              << "LogDDP: Dynamics derivatives computation thread failed: "
              << e.what() << std::endl;
        }
        throw;
      }
    }
  }

  if (options.debug) {
    std::cout << "[LogDDP] Pre-computed dynamics derivatives for " << horizon
              << " time steps using "
              << (use_parallel ? "parallel" : "sequential") << " computation"
              << std::endl;
  }
}

bool LogDDPSolver::backwardPass(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();
  const int horizon = context.getHorizon();
  const double timestep = context.getTimestep();
  const auto &constraint_set = context.getConstraintSet();

  // Pre-compute dynamics jacobians and hessians for all time steps
  precomputeDynamicsDerivatives(context);

  // Terminal cost and derivatives (V_x, V_xx at t=N)
  Eigen::VectorXd V_x =
      context.getObjective().getFinalCostGradient(context.X_.back());
  Eigen::MatrixXd V_xx =
      context.getObjective().getFinalCostHessian(context.X_.back());
  V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize

  dV_ = Eigen::Vector2d::Zero();
  double Qu_err = 0.0;

  // Backward Riccati recursion
  for (int t = horizon - 1; t >= 0; --t) {
    const Eigen::VectorXd &x = context.X_[t];
    const Eigen::VectorXd &u = context.U_[t];
    const Eigen::VectorXd &f = F_[t];
    const Eigen::VectorXd &d = f - context.X_[t + 1]; // Defect

    // Use pre-computed dynamics Jacobians
    const Eigen::MatrixXd &Fx = F_x_[t];
    const Eigen::MatrixXd &Fu = F_u_[t];
    const Eigen::MatrixXd &A =
        timestep * Fx + Eigen::MatrixXd::Identity(state_dim, state_dim);
    const Eigen::MatrixXd &B = timestep * Fu;

    // Cost derivatives at (x_t, u_t)
    auto [l_x, l_u] = context.getObjective().getRunningCostGradients(x, u, t);
    auto [l_xx, l_uu, l_ux] =
        context.getObjective().getRunningCostHessians(x, u, t);

    Eigen::VectorXd Q_x = l_x + A.transpose() * (V_x + V_xx * d);
    Eigen::VectorXd Q_u = l_u + B.transpose() * (V_x + V_xx * d);
    Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
    Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
    Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;

    // Add state hessian term if not using iLQR
    if (!options.use_ilqr) {
      // Use pre-computed hessians
      const auto &Fxx = F_xx_[t];
      const auto &Fuu = F_uu_[t];
      const auto &Fux = F_ux_[t];

      for (int i = 0; i < state_dim; ++i) {
        Q_xx += timestep * V_x(i) * Fxx[i];
        Q_ux += timestep * V_x(i) * Fux[i];
        Q_uu += timestep * V_x(i) * Fuu[i];
      }
    }

    // Apply Log-barrier cost gradients and Hessians
    for (const auto &constraint_pair : constraint_set) {
      auto [L_x_relaxed, L_u_relaxed] =
          relaxed_log_barrier_->getGradients(*constraint_pair.second, x, u);
      Q_x += L_x_relaxed;
      Q_u += L_u_relaxed;

      auto [L_xx_relaxed, L_uu_relaxed, L_ux_relaxed] =
          relaxed_log_barrier_->getHessians(*constraint_pair.second, x, u);
      Q_xx += L_xx_relaxed;
      Q_uu += L_uu_relaxed;
      Q_ux += L_ux_relaxed;
    }

    // Regularization
    Eigen::MatrixXd Q_uu_reg = Q_uu;
    Q_uu_reg.diagonal().array() += context.regularization_;
    Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

    Eigen::LDLT<Eigen::MatrixXd> ldlt(Q_uu_reg);
    if (ldlt.info() != Eigen::Success) {
      if (options.debug) {
        std::cerr << "LogDDP: Backward pass failed at time " << t << std::endl;
      }
      return false;
    }

    Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
    bigRHS.col(0) = Q_u;
    Eigen::MatrixXd M = Q_ux;
    for (int col = 0; col < state_dim; col++) {
      bigRHS.col(col + 1) = M.col(col);
    }

    Eigen::MatrixXd kK = -ldlt.solve(bigRHS);

    // parse out feedforward (ku) and feedback (Ku)
    Eigen::VectorXd k_u = kK.col(0);
    Eigen::MatrixXd K_u(control_dim, state_dim);
    for (int col = 0; col < state_dim; col++) {
      K_u.col(col) = kK.col(col + 1);
    }

    // Save gains
    k_u_[t] = k_u;
    K_u_[t] = K_u;

    // Compute value function approximation
    Eigen::Vector2d dV_step;
    dV_step << Q_u.dot(k_u), 0.5 * k_u.dot(Q_uu * k_u);
    dV_ = dV_ + dV_step;
    V_x = Q_x + K_u.transpose() * Q_uu * k_u + Q_ux.transpose() * k_u +
          K_u.transpose() * Q_u;
    V_xx = Q_xx + K_u.transpose() * Q_uu * K_u + Q_ux.transpose() * K_u +
           K_u.transpose() * Q_ux;
    V_xx = 0.5 * (V_xx + V_xx.transpose()); // Symmetrize Hessian

    // Compute optimality gap (Inf-norm) for convergence check
    Qu_err = std::max(Qu_err, Q_u.lpNorm<Eigen::Infinity>());
  }
  context.inf_du_ = Qu_err;

  if (options.debug) {
    std::cout << "[LogDDP Backward Pass]\n"
              << "    Qu_err:  " << std::scientific << std::setprecision(4)
              << Qu_err << "\n"
              << "    rf_err:  " << std::scientific << std::setprecision(4)
              << constraint_violation_ << "\n"
              << "    dV:      " << std::scientific << std::setprecision(4)
              << dV_.transpose() << std::endl;
  }

  return true;
}

ForwardPassResult LogDDPSolver::performForwardPass(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  ForwardPassResult best_result;
  best_result.cost = std::numeric_limits<double>::infinity();
  best_result.merit_function = std::numeric_limits<double>::infinity();
  best_result.success = false;

  if (!options.enable_parallel) {
    // Single-threaded execution with early termination
    for (double alpha_pr : context.alphas_) {
      ForwardPassResult result = forwardPass(context, alpha_pr);

      if (result.success) {
        best_result = result;
        break; // Early termination on first success
      }
    }
  } else {
    // Multi-threaded execution
    std::vector<std::future<ForwardPassResult>> futures;
    futures.reserve(context.alphas_.size());

    for (double alpha_pr : context.alphas_) {
      futures.push_back(
          std::async(std::launch::async, [this, &context, alpha_pr]() {
            return forwardPass(context, alpha_pr);
          }));
    }

    for (auto &future : futures) {
      try {
        if (future.valid()) {
          ForwardPassResult result = future.get();
          if (result.success &&
              result.merit_function < best_result.merit_function) {
            best_result = result;
          }
        }
      } catch (const std::exception &e) {
        if (options.verbose) {
          std::cerr << "LogDDP: Forward pass thread failed: " << e.what()
                    << std::endl;
        }
      }
    }
  }

  return best_result;
}

ForwardPassResult LogDDPSolver::forwardPass(CDDP &context, double alpha) {
  const CDDPOptions &options = context.getOptions();
  const auto &constraint_set = context.getConstraintSet();

  ForwardPassResult result;
  result.success = false;
  result.cost = std::numeric_limits<double>::infinity();
  result.merit_function = std::numeric_limits<double>::infinity();
  result.alpha_pr = alpha;

  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();

  // Initialize trajectories
  result.state_trajectory = context.X_;
  result.control_trajectory = context.U_;
  result.state_trajectory[0] = context.getInitialState();

  std::vector<Eigen::VectorXd> F_new = F_;

  double cost_new = 0.0;
  double merit_function_new = 0.0;
  double rp_err = 0.0;
  double rf_err = 0.0;

  // Rollout loop with multi-shooting logic from original
  for (int t = 0; t < horizon; ++t) {
    const Eigen::VectorXd delta_x_t =
        result.state_trajectory[t] - context.X_[t];

    // Update control
    result.control_trajectory[t] =
        context.U_[t] + alpha * k_u_[t] + K_u_[t] * delta_x_t;

    // --- Rollout Logic from original ---
    Eigen::VectorXd dynamics_eval_for_F_new_t;

    // Determine if the *next* step (t+1) starts a new segment boundary
    bool is_segment_boundary = (ms_segment_length_ > 0) &&
                               ((t + 1) % ms_segment_length_ == 0) &&
                               (t + 1 < horizon);
    bool apply_gap_closing_strategy = is_segment_boundary;

    if (apply_gap_closing_strategy) {
      if (options.log_barrier.rollout_type == "nonlinear") {
        F_new[t] = context.getSystem().getDiscreteDynamics(
            result.state_trajectory[t], result.control_trajectory[t],
            t * context.getTimestep());
        result.state_trajectory[t + 1] = context.X_[t + 1] +
                                         (F_new[t] - F_[t]) +
                                         alpha * (F_[t] - context.X_[t + 1]);
      } else if (options.log_barrier.rollout_type == "hybrid") {
        F_new[t] = context.getSystem().getDiscreteDynamics(
            result.state_trajectory[t], result.control_trajectory[t],
            t * context.getTimestep());
        const auto [Fx, Fu] = context.getSystem().getJacobians(
            context.X_[t], context.U_[t], t * context.getTimestep());
        const Eigen::MatrixXd A =
            Eigen::MatrixXd::Identity(state_dim, state_dim) +
            context.getTimestep() * Fx;
        const Eigen::MatrixXd B = context.getTimestep() * Fu;
        result.state_trajectory[t + 1] =
            context.X_[t + 1] + (A + B * K_u_[t]) * delta_x_t +
            alpha * (B * k_u_[t] + F_[t] - context.X_[t + 1]);
      }
    } else {
      result.state_trajectory[t + 1] = context.getSystem().getDiscreteDynamics(
          result.state_trajectory[t], result.control_trajectory[t],
          t * context.getTimestep());
      F_new[t] = result.state_trajectory[t + 1];
    }

    // Robustness check
    if (!result.state_trajectory[t + 1].allFinite() ||
        !result.control_trajectory[t].allFinite()) {
      if (options.debug) {
        std::cerr
            << "[LogDDP Forward Pass] NaN/Inf detected during rollout at t="
            << t << " for alpha=" << alpha << std::endl;
      }
      result.success = false;
      return result;
    }
  }

  // Cost computation and filter line-search from original
  for (int t = 0; t < horizon; ++t) {
    cost_new += context.getObjective().running_cost(
        result.state_trajectory[t], result.control_trajectory[t], t);

    for (const auto &constraint_pair : constraint_set) {
      const std::string &constraint_name = constraint_pair.first;
      Eigen::VectorXd g_t =
          constraint_pair.second->evaluate(result.state_trajectory[t],
                                           result.control_trajectory[t]) -
          constraint_pair.second->getUpperBound();
      merit_function_new += relaxed_log_barrier_->evaluate(
          *constraint_pair.second, result.state_trajectory[t],
          result.control_trajectory[t]);

      for (int i = 0; i < g_t.size(); ++i) {
        if (g_t(i) > 0.0) {
          rp_err += g_t(i);
        }
      }
    }

    Eigen::VectorXd d = F_new[t] - result.state_trajectory[t + 1];
    rf_err += d.lpNorm<1>();
  }

  cost_new +=
      context.getObjective().terminal_cost(result.state_trajectory.back());
  merit_function_new += cost_new;

  // Filter-based acceptance using original logic with new options structure
  double constraint_violation_old = constraint_violation_;
  double constraint_violation_new = rf_err + rp_err;
  double merit_function_old = context.merit_function_;
  bool filter_acceptance = false;
  double expected_improvement = alpha * dV_(0);

  const auto &filter_opts = options.filter;

  if (constraint_violation_new > filter_opts.max_violation_threshold) {
    if (constraint_violation_new <
        (1.0 - filter_opts.violation_acceptance_threshold) *
            constraint_violation_old) {
      filter_acceptance = true;
    }
  } else if (std::max(constraint_violation_new, constraint_violation_old) <
                 filter_opts.min_violation_for_armijo_check &&
             expected_improvement < 0) {
    if (merit_function_new < merit_function_old + filter_opts.armijo_constant *
                                                      expected_improvement) {
      filter_acceptance = true;
    }
  } else {
    if (merit_function_new <
            merit_function_old - filter_opts.merit_acceptance_threshold *
                                     constraint_violation_old ||
        constraint_violation_new <
            (1.0 - filter_opts.violation_acceptance_threshold) *
                constraint_violation_old) {
      filter_acceptance = true;
    }
  }

  if (filter_acceptance) {
    result.success = true;
    result.cost = cost_new;
    result.merit_function = merit_function_new;
    result.constraint_violation = constraint_violation_new;
    result.dynamics_trajectory = F_new;
  }

  return result;
}

void LogDDPSolver::updateBarrierParameters(CDDP &context,
                                           bool forward_pass_success,
                                           double termination_metric) {
  // This method is called from solve() with the specific barrier update logic
  // from original The actual logic is implemented directly in solve() to match
  // the original
}

void LogDDPSolver::printIteration(int iter, double cost, double lagrangian,
                                  double opt_gap, double regularization,
                                  double alpha, double mu,
                                  double constraint_violation) const {
  if (iter == 0) {
    std::cout << std::setw(4) << "iter" << " " << std::setw(12) << "objective"
              << " " << std::setw(12) << "lagrangian" << " " << std::setw(10)
              << "opt_gap" << " " << std::setw(8) << "lg(rg)" << " "
              << std::setw(8) << "alpha" << " " << std::setw(8) << "lg(mu)"
              << " " << std::setw(10) << "cv_viol" << std::endl;
  }

  std::cout << std::setw(4) << iter << " " << std::setw(12) << std::scientific
            << std::setprecision(4) << cost << " " << std::setw(12)
            << std::scientific << std::setprecision(4) << lagrangian << " "
            << std::setw(10) << std::scientific << std::setprecision(2)
            << opt_gap << " " << std::setw(8) << std::fixed
            << std::setprecision(1) << std::log10(regularization) << " "
            << std::setw(8) << std::fixed << std::setprecision(4) << alpha
            << " " << std::setw(8) << std::fixed << std::setprecision(1)
            << std::log10(mu) << " " << std::setw(10) << std::scientific
            << std::setprecision(2) << constraint_violation << std::endl;
}

void LogDDPSolver::printSolutionSummary(const CDDPSolution &solution) const {
  std::cout << "\n========================================\n";
  std::cout << "           LogDDP Solution Summary\n";
  std::cout << "========================================\n";

  auto iterations = std::any_cast<int>(solution.at("iterations_completed"));
  auto solve_time = std::any_cast<double>(solution.at("solve_time_ms"));
  auto final_cost = std::any_cast<double>(solution.at("final_objective"));
  auto status = std::any_cast<std::string>(solution.at("status_message"));
  auto final_mu =
      std::any_cast<double>(solution.at("final_barrier_parameter_mu"));

  std::cout << "Status: " << status << "\n";
  std::cout << "Iterations: " << iterations << "\n";
  std::cout << "Solve Time: " << std::setprecision(2) << solve_time << " ms\n";
  std::cout << "Final Cost: " << std::setprecision(6) << final_cost << "\n";
  std::cout << "Final Barrier μ: " << std::setprecision(2) << std::scientific
            << final_mu << "\n";
  std::cout << "========================================\n\n";
}

} // namespace cddp
