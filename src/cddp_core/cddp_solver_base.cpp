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

#include "cddp_core/cddp_solver_base.hpp"
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <iomanip>
#include <iostream>

namespace cddp {

// === Template method: solve() ===

CDDPSolution CDDPSolverBase::solve(CDDP &context) {
  const CDDPOptions &options = context.getOptions();

  // Print header/options if requested
  if (options.print_solver_header) {
    context.printSolverInfo();
  }
  if (options.print_solver_options) {
    context.printOptions(options);
  }

  // Prepare solution
  CDDPSolution solution;
  solution.solver_name = getSolverName();
  solution.status_message = "Running";
  solution.iterations_completed = 0;
  solution.solve_time_ms = 0.0;

  // Initialize history
  if (options.return_iteration_info) {
    history_.clear();
    history_.reserve(static_cast<size_t>(options.max_iterations + 1));
  }

  // Solver-specific pre-loop setup (evaluateTrajectory, resetFilter, etc.)
  preIterationSetup(context);

  // Record initial state
  if (options.return_iteration_info) {
    recordIterationHistory(context);
  }

  if (options.verbose) {
    printIteration(0, context);
  }

  // Start timer
  auto start_time = std::chrono::high_resolution_clock::now();
  int iter = 0;
  bool converged = false;
  std::string termination_reason = "MaxIterationsReached";
  double dJ = 0.0;
  double dL = 0.0;

  // Main iteration loop
  while (iter < options.max_iterations) {
    ++iter;

    // Check CPU time limit
    if (options.max_cpu_time > 0) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start_time);
      if (elapsed.count() > options.max_cpu_time * 1000) {
        termination_reason = "MaxCpuTimeReached";
        if (options.verbose) {
          std::cerr << getSolverName()
                    << ": Maximum CPU time reached. Returning current solution"
                    << std::endl;
        }
        break;
      }
    }

    // Backward pass with regularization retry
    bool backward_ok = false;
    while (!backward_ok) {
      backward_ok = backwardPass(context);
      if (!backward_ok) {
        context.increaseRegularization();
        if (context.isRegularizationLimitReached()) {
          converged = handleBackwardPassRegularizationLimit(
              context, termination_reason);
          if (options.verbose) {
            std::cerr << getSolverName()
                      << ": Backward pass regularization limit reached"
                      << std::endl;
          }
          break;
        }
      }
    }
    if (!backward_ok)
      break;

    // Early convergence check (e.g., CLDDP checks inf_du after backward pass)
    if (checkEarlyConvergence(context, iter, termination_reason)) {
      converged = true;
      if (options.return_iteration_info) {
        recordIterationHistory(context);
      }
      if (options.verbose) {
        printIteration(iter, context);
      }
      break;
    }

    // Forward pass with line search
    ForwardPassResult best_result = performForwardPass(context);
    bool fp_success = best_result.success;

    if (fp_success) {
      dJ = context.cost_ - best_result.cost;
      dL = context.merit_function_ - best_result.merit_function;
      applyForwardPassResult(context, best_result);
      if (options.return_iteration_info) {
        recordIterationHistory(context);
      }
      context.decreaseRegularization();

      // Check convergence (only meaningful after successful forward pass)
      converged = checkConvergence(context, dJ, dL, iter, termination_reason);
    } else {
      bool should_break = handleForwardPassFailure(context, termination_reason);
      if (should_break)
        break;
    }

    if (options.verbose) {
      printIteration(iter, context);
    }

    if (converged)
      break;

    postIterationUpdate(context, fp_success);
  }

  // Compute final timing
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  // Populate common solution fields
  solution.status_message = termination_reason;
  solution.iterations_completed = iter;
  solution.solve_time_ms = static_cast<double>(duration.count());
  solution.final_objective = context.cost_;
  solution.final_step_length = context.alpha_pr_;
  solution.time_points = buildTimePoints(context);
  solution.state_trajectory = context.X_;
  solution.control_trajectory = context.U_;
  solution.feedback_gains = K_u_;
  solution.final_regularization = context.regularization_;

  // Move iteration history into solution (same type — no field-by-field copy)
  if (options.return_iteration_info) {
    solution.history = std::move(history_);
  }

  // Solver-specific solution fields
  populateSolverSpecificSolution(solution, context);

  if (options.verbose) {
    printSolutionSummary(solution);
  }

  return solution;
}

// === Default implementations ===

void CDDPSolverBase::applyForwardPassResult(CDDP &context,
                                            const ForwardPassResult &result) {
  context.X_ = result.state_trajectory;
  context.U_ = result.control_trajectory;
  context.cost_ = result.cost;
  context.merit_function_ = result.merit_function;
  context.alpha_pr_ = result.alpha_pr;
  context.alpha_du_ = result.alpha_du;
}

bool CDDPSolverBase::handleBackwardPassRegularizationLimit(
    CDDP &context, std::string &termination_reason) {
  termination_reason = "RegularizationLimitReached_NotConverged";
  return false; // not converged
}

bool CDDPSolverBase::handleForwardPassFailure(CDDP &context,
                                              std::string &termination_reason) {
  context.increaseRegularization();
  if (context.isRegularizationLimitReached()) {
    termination_reason = "RegularizationLimitReached_NotConverged";
    if (context.getOptions().verbose) {
      std::cerr << getSolverName()
                << ": Regularization limit reached. Not converged." << std::endl;
    }
    return true; // break
  }
  return false; // continue
}

void CDDPSolverBase::recordIterationHistory(const CDDP &context) {
  history_.objective.push_back(context.cost_);
  history_.merit_function.push_back(context.merit_function_);
  history_.step_length_primal.push_back(context.alpha_pr_);
  history_.step_length_dual.push_back(context.alpha_du_);
  history_.dual_infeasibility.push_back(context.inf_du_);
  history_.primal_infeasibility.push_back(context.inf_pr_);
  history_.complementary_infeasibility.push_back(context.inf_comp_);
  // Note: barrier_mu is NOT pushed here. IP solvers override recordIterationHistory
  // to push their actual mu_ value. Non-IP solvers (CLDDP) don't use barrier_mu,
  // so it stays empty for them. Consumers must check vector sizes.
  history_.regularization.push_back(context.regularization_);
}

void CDDPSolverBase::printSolutionSummary(const CDDPSolution &solution) const {
  std::cout << "\n========================================\n";
  std::cout << "       " << getSolverName() << " Solution Summary\n";
  std::cout << "========================================\n";

  std::cout << "Status: " << solution.status_message << "\n";
  std::cout << "Iterations: " << solution.iterations_completed << "\n";
  std::cout << "Solve Time: " << std::setprecision(2) << solution.solve_time_ms << " ms\n";
  std::cout << "Final Cost: " << std::setprecision(6) << solution.final_objective << "\n";
  std::cout << "========================================\n\n";
}

// === Shared utilities ===

ForwardPassResult CDDPSolverBase::performForwardPass(CDDP &context) {
  const CDDPOptions &options = context.getOptions();
  ForwardPassResult best_result;
  best_result.cost = std::numeric_limits<double>::infinity();
  best_result.merit_function = std::numeric_limits<double>::infinity();
  best_result.success = false;

  if (!options.enable_parallel) {
    // Single-threaded with early termination
    for (double alpha : context.alphas_) {
      ForwardPassResult result = forwardPass(context, alpha);
      if (result.success) {
        best_result = result;
        break;
      }
    }
  } else {
    // Multi-threaded
    std::vector<std::future<ForwardPassResult>> futures;
    futures.reserve(context.alphas_.size());
    std::exception_ptr first_exception;

    for (double alpha : context.alphas_) {
      futures.push_back(
          std::async(std::launch::async,
                     [this, &context, alpha]() { return forwardPass(context, alpha); }));
    }

    int failed_count = 0;
    std::string last_error;
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
        ++failed_count;
        last_error = e.what();
        if (!first_exception) {
          first_exception = std::current_exception();
        }
        if (options.verbose) {
          std::cerr << getSolverName()
                    << ": Forward pass thread failed: " << e.what()
                    << std::endl;
        }
      }
    }
    if (failed_count > 0 &&
        failed_count == static_cast<int>(futures.size())) {
      std::cerr << getSolverName()
                << ": ALL forward pass threads failed. Last error: "
                << last_error << std::endl;
    }
    if (first_exception) {
      std::rethrow_exception(first_exception);
    }
  }

  return best_result;
}

void CDDPSolverBase::precomputeDynamicsDerivatives(
    CDDP &context, int min_horizon_for_parallel) {
  const CDDPOptions &options = context.getOptions();
  const int horizon = context.getHorizon();
  const int state_dim = context.getStateDim();
  const int control_dim = context.getControlDim();

  // Resize storage
  F_x_.resize(horizon);
  F_u_.resize(horizon);
  if (!options.use_ilqr) {
    F_xx_.resize(horizon);
    F_uu_.resize(horizon);
    F_ux_.resize(horizon);
  }

  auto compute_derivatives = [&](int t) {
    const Eigen::VectorXd &x = context.X_[t];
    const Eigen::VectorXd &u = context.U_[t];
    double time = t * context.getTimestep();

    auto [Fx, Fu] = context.getSystem().getJacobians(x, u, time);
    // Convert to discrete time
    F_x_[t] = context.getTimestep() * Fx;
    F_x_[t].diagonal().array() += 1.0;
    F_u_[t] = context.getTimestep() * Fu;

    if (!options.use_ilqr) {
      auto [Fxx, Fuu, Fux] = context.getSystem().getHessians(x, u, time);
      F_xx_[t].resize(state_dim);
      F_uu_[t].resize(state_dim);
      F_ux_[t].resize(state_dim);
      for (int i = 0; i < state_dim; ++i) {
        F_xx_[t][i] = context.getTimestep() * Fxx[i];
        F_uu_[t][i] = context.getTimestep() * Fuu[i];
        F_ux_[t][i] = context.getTimestep() * Fux[i];
      }
    }
  };

  bool use_parallel =
      options.enable_parallel && horizon >= min_horizon_for_parallel;

  if (use_parallel) {
    std::vector<std::future<void>> futures;
    futures.reserve(horizon);
    for (int t = 0; t < horizon; ++t) {
      futures.push_back(
          std::async(std::launch::async, compute_derivatives, t));
    }
    for (auto &f : futures) {
      f.get();
    }
  } else {
    for (int t = 0; t < horizon; ++t) {
      compute_derivatives(t);
    }
  }
}

void CDDPSolverBase::initializeGains(int horizon, int control_dim,
                                     int state_dim) {
  k_u_.resize(horizon);
  K_u_.resize(horizon);
  for (int t = 0; t < horizon; ++t) {
    k_u_[t] = Eigen::VectorXd::Zero(control_dim);
    K_u_[t] = Eigen::MatrixXd::Zero(control_dim, state_dim);
  }
  dV_ = Eigen::Vector2d::Zero();
}

std::vector<double> CDDPSolverBase::buildTimePoints(const CDDP &context) {
  std::vector<double> time_points;
  time_points.reserve(static_cast<size_t>(context.getHorizon() + 1));
  for (int t = 0; t <= context.getHorizon(); ++t) {
    time_points.push_back(t * context.getTimestep());
  }
  return time_points;
}

void CDDPSolverBase::computeCost(CDDP &context) {
  context.cost_ = 0.0;
  for (int t = 0; t < context.getHorizon(); ++t) {
    context.cost_ +=
        context.getObjective().running_cost(context.X_[t], context.U_[t], t);
  }
  context.cost_ += context.getObjective().terminal_cost(context.X_.back());
  context.merit_function_ = context.cost_;
}

} // namespace cddp
