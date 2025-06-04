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
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory> // For std::unique_ptr
#include <map>    // For std::map

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
    if (options_.header_and_footer) {
        printSolverInfo();
        printOptions(options_);
    }
}

cddp::CDDPSolution CDDP::solve(std::string solver_type) {
    if (solver_type == "CLCDDP" || solver_type == "CLDDP") {
        if (options_.verbose) {
            std::cout << "--------------------" << std::endl;
            std::cout << "Solving with CLCDDP" << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        return solveCLCDDP();
    } else if (solver_type == "LogCDDP" || solver_type == "LogDDP") {
        if (options_.verbose) {
            std::cout << "--------------------" << std::endl;
            std::cout << "Solving with LogDDP" << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        return solveLogDDP();
    } else if (solver_type == "ASCDDP" || solver_type == "ASDDP") {
        if (options_.verbose) {
            std::cout << "--------------------" << std::endl;
            std::cout << "Solving with ASCDDP" << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        return solveASCDDP();
    } else if (solver_type == "IPDDP") {
        if (options_.verbose) {
            std::cout << "--------------------" << std::endl;
            std::cout << "Solving with IPDDP" << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        return solveIPDDP();
    } else if (solver_type == "MSIPDDP") {
        if (options_.verbose) {
            std::cout << "--------------------" << std::endl;
            std::cout << "Solving with MSIPDDP" << std::endl;
            std::cout << "--------------------" << std::endl;
        }
        return solveMSIPDDP();
    } else
    {
        std::cerr << "CDDP::solve: Unknown solver type" << std::endl;
        throw std::runtime_error("CDDP::solve: Unknown solver type");
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
    k_u_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    K_u_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
    dV_.resize(2);

    // Initialize Q-function matrices: USED ONLY FOR ASCDDP
    Q_UU_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, control_dim));
    Q_UX_.resize(horizon_, Eigen::MatrixXd::Zero(control_dim, state_dim));
    Q_U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));

    // Check if ControlBoxConstraint is set
    if (constraint_set_.find("ControlBoxConstraint") != constraint_set_.end())
    {
        std::cout << "ControlBoxConstraint is set" << std::endl;
    }

    // Initialize Log-barrier object
    mu_ = options_.barrier_coeff;
    constraint_violation_ = 0.0;

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

double CDDP::computeConstraintViolation(const std::vector<Eigen::VectorXd>& X,
                                      const std::vector<Eigen::VectorXd>& U) const {
    double total_violation = 0.0;
    for (const auto& constraint : constraint_set_) {
        for (size_t t = 0; t < U.size(); ++t) {
            total_violation += constraint.second->computeViolation(X[t], U[t]);
        }
    }
    return total_violation;
}


void CDDP::increaseRegularization()
{
    // For "state" or "both"
    if (options_.regularization_type == "state" ||
        options_.regularization_type == "both")
    {
        // Increase step
        regularization_state_step_ = std::max(
            regularization_state_step_ * options_.regularization_state_factor,
            options_.regularization_state_factor);

        // Increase actual regularization
        regularization_state_ = std::max(
            regularization_state_ * regularization_state_step_,
            options_.regularization_state_min);
    }

    // For "control" or "both"
    if (options_.regularization_type == "control" ||
        options_.regularization_type == "both")
    {
        // Increase step
        regularization_control_step_ = std::max(
            regularization_control_step_ * options_.regularization_control_factor,
            options_.regularization_control_factor);

        // Increase actual regularization
        regularization_control_ = std::max(
            regularization_control_ * regularization_control_step_,
            options_.regularization_control_min);
    }
}


void CDDP::decreaseRegularization()
{
    // For "state" or "both"
    if (options_.regularization_type == "state" ||
        options_.regularization_type == "both")
    {
        // Decrease step
        regularization_state_step_ = std::min(
            regularization_state_step_ / options_.regularization_state_factor,
            1.0 / options_.regularization_state_factor);

        // Decrease actual regularization
        regularization_state_ = std::max(
            regularization_state_ * regularization_state_step_,
            options_.regularization_state_min);
    }

    // For "control" or "both"
    if (options_.regularization_type == "control" ||
        options_.regularization_type == "both")
    {
        // Decrease step
        regularization_control_step_ = std::min(
            regularization_control_step_ / options_.regularization_control_factor,
            1.0 / options_.regularization_control_factor);

        // Decrease actual regularization
        regularization_control_ = std::max(
            regularization_control_ * regularization_control_step_,
            options_.regularization_control_min);
    }
}


bool CDDP::isRegularizationLimitReached() const
{
    bool state_limit   = (regularization_state_   >= options_.regularization_state_max);
    bool control_limit = (regularization_control_ >= options_.regularization_control_max);

    if (options_.regularization_type == "state")
        return state_limit;
    else if (options_.regularization_type == "control")
        return control_limit;
    else if (options_.regularization_type == "both")
        return (state_limit || control_limit);

    // For "none" or unknown, no limit in practice
    return false;
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
    std::cout << "  Barrier Order: " << std::setw(5) << options.barrier_order << "\n";
    std::cout << "  Filter Acceptance: " << std::setw(5) << options.filter_acceptance << "\n";
    std::cout << "  Constraint Tolerance: " << std::setw(5) << options.constraint_tolerance << "\n";

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
    std::cout << "  Relaxed Log-Barrier: " << (options.is_relaxed_log_barrier ? "Yes" : "No") << "\n";
    std::cout << "  Early Termination: " << (options.early_termination ? "Yes" : "No") << "\n";

    std::cout << "\nBoxQP:\n";
    std::cout << "  BoxQP Max Iterations: " << options.boxqp_max_iterations << "\n";
    std::cout << "  BoxQP Min Grad: " << options.boxqp_min_grad << "\n";
    std::cout << "  BoxQP Min Rel Improve: " << options.boxqp_min_rel_improve << "\n";
    std::cout << "  BoxQP Step Dec: " << options.boxqp_step_dec << "\n";
    std::cout << "  BoxQP Min Step: " << options.boxqp_min_step << "\n";
    std::cout << "  BoxQP Armijo: " << options.boxqp_armijo << "\n";
    std::cout << "  BoxQP Verbose: " << (options.boxqp_verbose ? "Yes" : "No") << "\n";

    std::cout << "\nMSIPDDP:\n";
    std::cout << "  MS Segment Length: " << options.ms_segment_length << "\n";
    std::cout << "  MS Rollout Type: " << options.ms_rollout_type << "\n";
    std::cout << "  MS Defect Tolerance: " << options.ms_defect_tolerance_for_single_shooting << "\n";
    std::cout << "  Barrier Update Factor: " << options.barrier_update_factor << "\n";
    std::cout << "  Barrier Update Power: " << options.barrier_update_power << "\n";


    std::cout << "========================================\n\n";
}

void CDDP::printIteration(int iter, double cost, double lagrangian, double grad_norm,
               double lambda_state, double lambda_control, double step_size, 
               double mu, double constraint_violation)
{
   if (iter % 10 == 0)
   {
       std::cout << std::setw(5) << "Iter"
               << std::setw(12) << "Cost"
               << std::setw(12) << "Lagr"
               << std::setw(10) << "Grad"
               << std::setw(10) << "Step"
               << std::setw(10) << "RegS"
               << std::setw(10) << "RegC" 
               << std::setw(10) << "Mu"
               << std::setw(10) << "Viol"
               << std::endl;
       std::cout << std::string(89, '-') << std::endl;
   }

   std::cout << std::setw(5) << iter
           << std::setw(12) << std::scientific << std::setprecision(3) << cost
           << std::setw(12) << std::scientific << std::setprecision(3) << lagrangian
           << std::setw(10) << std::scientific << std::setprecision(2) << grad_norm
           << std::setw(10) << std::fixed << std::setprecision(3) << step_size
           << std::setw(10) << std::scientific << std::setprecision(2) << lambda_state
           << std::setw(10) << std::scientific << std::setprecision(2) << lambda_control
           << std::setw(10) << std::scientific << std::setprecision(2) << mu
           << std::setw(10) << std::scientific << std::setprecision(2) << constraint_violation
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
