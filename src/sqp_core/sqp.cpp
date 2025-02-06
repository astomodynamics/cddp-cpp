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
#include "sqp_core/sqp.hpp"
#include "cddp_core/helper.hpp"
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace cddp {

SQPSolver::SQPSolver(const Eigen::VectorXd& initial_state,
                     const Eigen::VectorXd& reference_state,
                     int horizon,
                     double timestep)
    : initial_state_(initial_state),
      reference_state_(reference_state),
      horizon_(horizon),
      timestep_(timestep) {
    // OSQP settings will be set when setOptions() is called.
    initializeSQP();
}

void SQPSolver::setOptions(const SQPOptions& options) { 
    options_ = options;
    osqp_settings_.eps_abs = options_.osqp_eps_abs;
    osqp_settings_.eps_rel = options_.osqp_eps_rel;
    osqp_settings_.max_iter = options_.osqp_max_iter;
    osqp_settings_.verbose = options_.osqp_verbose;
    osqp_settings_.warm_start = options_.warm_start;
    osqp_settings_.adaptive_rho = true;
}

void SQPSolver::initializeSQP() {
    if (!system_) {
        return;  // Wait until system is set to know dimensions.
    }
    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();
    X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
    U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));
    X_[0] = initial_state_;
}

void SQPSolver::setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, 
                                     const std::vector<Eigen::VectorXd>& U) {
    X_ = X;
    U_ = U;
    checkDimensions();
}

void SQPSolver::propagateDynamics(const Eigen::VectorXd& x0, 
                                  std::vector<Eigen::VectorXd>& U) {
    if (!system_) {
        throw std::runtime_error("System not set");
    }
    const int horizon = U.size();
    X_.resize(horizon + 1);
    X_[0] = x0;
    for (int t = 0; t < horizon; ++t) {
        X_[t + 1] = system_->getDiscreteDynamics(X_[t], U[t]);
    }
}

void SQPSolver::checkDimensions() const {
    if (!system_ || !objective_) {
        throw std::runtime_error("System or objective not set");
    }
    if (X_.empty() || U_.empty() || X_.size() != U_.size() + 1) {
        throw std::runtime_error("Invalid trajectory dimensions");
    }
    if (X_.size() != horizon_ + 1) {
        throw std::runtime_error("Trajectory length does not match horizon");
    }
    for (const auto& x : X_) {
        if (x.size() != system_->getStateDim()) {
            throw std::runtime_error("Invalid state dimension");
        }
    }
    for (const auto& u : U_) {
        if (u.size() != system_->getControlDim()) {
            throw std::runtime_error("Invalid control dimension");
        }
    }
}

void SQPSolver::computeLinearizedDynamics(const std::vector<Eigen::VectorXd>& X,
                                          const std::vector<Eigen::VectorXd>& U,
                                          std::vector<Eigen::MatrixXd>& A,
                                          std::vector<Eigen::MatrixXd>& B) {
    const int horizon = U.size();
    A.resize(horizon);
    B.resize(horizon);
    for (int t = 0; t < horizon; ++t) {
        auto [At, Bt] = system_->getJacobians(X[t], U[t]);
        A[t] = (At * timestep_ + Eigen::MatrixXd::Identity(At.rows(), At.cols()));
        B[t] = Bt * timestep_;
    }
}

SQPResult SQPSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    SQPResult result;
    result.success = false;
    result.iterations = 0;

    if (!system_ || !objective_) {
        throw std::runtime_error("System or objective not set");
    }

    // Propagate dynamics for initial trajectory.
    propagateDynamics(initial_state_, U_);

    double trust_region_radius = options_.trust_region_radius;
    std::vector<Eigen::VectorXd> X_curr = X_;
    std::vector<Eigen::VectorXd> U_curr = U_;

    double J_curr = objective_->evaluate(X_curr, U_curr);
    double viol_curr = computeConstraintViolation(X_curr, U_curr);

    if (options_.verbose) {
        std::cout << "Initial cost: " << J_curr << std::endl;
        std::cout << "Initial constraint violation: " << viol_curr << std::endl;
    }

    osqp::OsqpInstance instance;
    osqp::OsqpSolver solver;

    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        result.iterations++;

        // 1. Formulate QP subproblem with the current trust region radius.
        Eigen::SparseMatrix<double> H, A;
        Eigen::VectorXd g, l, u;
        formQPSubproblem(X_curr, U_curr, trust_region_radius, H, g, A, l, u);

        instance.objective_matrix = H;
        instance.objective_vector = g;
        instance.constraint_matrix = A;
        instance.lower_bounds = l;
        instance.upper_bounds = u;

        if (iter == 0) {
            auto status = solver.Init(instance, osqp_settings_);
            if (!status.ok()) {
                if (options_.verbose) {
                    std::cout << "OSQP initialization failed: " << status.message() << std::endl;
                }
                return result;
            }
        } else {
            solver.UpdateObjectiveMatrix(H);
            solver.SetObjectiveVector(g);
            solver.UpdateConstraintMatrix(A);
            solver.SetBounds(l, u);
        }

        auto exit_code = solver.Solve();
        if (exit_code != osqp::OsqpExitCode::kOptimal) {
            if (options_.verbose) {
                std::cout << "QP solve failed at iteration " << iter << std::endl;
            }
            break;
        }

        // 2. Extract update directions from the QP solution.
        std::vector<Eigen::VectorXd> dX, dU;
        extractUpdates(solver.primal_solution(), dX, dU);
        
        // 3. Determine step length via line search.
        double alpha = lineSearch(X_curr, U_curr, dX, dU);
        if (options_.verbose) {
            std::cout << "Iteration " << iter << " line search alpha = " << alpha << std::endl;
        }

        // 4. Update trajectories using step–size alpha.
        std::vector<Eigen::VectorXd> X_new, U_new;
        updateTrajectories(X_curr, U_curr, dX, dU, X_new, U_new, alpha);

        double J_new = objective_->evaluate(X_new, U_new);
        double viol_new = computeConstraintViolation(X_new, U_new);
        double merit_new = computeMeritFunction(X_new, U_new, options_.merit_penalty);
        double merit_curr = computeMeritFunction(X_curr, U_curr, options_.merit_penalty);

        if (options_.verbose) {
            std::cout << "Iter " << iter << ": cost=" << J_new 
                      << ", viol=" << viol_new 
                      << ", merit=" << merit_new << std::endl;
        }

        // 5. Accept or reject the step.
        if (merit_new < merit_curr) {
            // Accept the step.
            X_curr = X_new;
            U_curr = U_new;
            J_curr = J_new;
            viol_curr = viol_new;
            // Increase trust region if nearly full step.
            if (alpha > 0.9) {
                trust_region_radius = std::min(options_.trust_region_radius_max, 
                                               trust_region_radius * options_.trust_region_increase_factor);
            }
        } else {
            // Reject the step: contract trust region.
            trust_region_radius *= options_.trust_region_decrease_factor;
        }
        
        result.obj_history.push_back(J_curr);
        result.viol_history.push_back(viol_curr);

        double dJ = std::abs(J_new - J_curr) / (std::abs(J_curr) + 1e-10);
        double max_step = 0.0;
        for (size_t i = 0; i < U_curr.size(); ++i) {
            max_step = std::max(max_step, (U_new[i] - U_curr[i]).lpNorm<Eigen::Infinity>());
        }

        if (dJ < options_.ftol && max_step < options_.xtol && viol_new < options_.gtol && iter >= options_.min_iterations) {
            result.success = true;
            if (options_.verbose) {
                std::cout << "SQP converged at iteration " << iter << std::endl;
                std::cout << "Final cost: " << J_new << ", violation: " << viol_new << std::endl;
            }
            break;
        }
    }

    result.X = X_curr;
    result.U = U_curr;
    result.objective_value = J_curr;
    result.constraint_violation = viol_curr;
    auto end_time = std::chrono::high_resolution_clock::now();
    result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    if (options_.verbose) {
        std::cout << "\nSQP finished in " << result.iterations << " iterations" << std::endl;
        std::cout << "Solve time: " << result.solve_time << " seconds" << std::endl;
        std::cout << "Final cost: " << result.objective_value << std::endl;
        std::cout << "Final constraint violation: " << result.constraint_violation << std::endl;
        std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
    }
    return result;
}

void SQPSolver::formQPSubproblem(const std::vector<Eigen::VectorXd>& X,
                                 const std::vector<Eigen::VectorXd>& U,
                                 double trust_region_radius,
                                 Eigen::SparseMatrix<double>& H,
                                 Eigen::VectorXd& g,
                                 Eigen::SparseMatrix<double>& A,
                                 Eigen::VectorXd& l,
                                 Eigen::VectorXd& u) {
    const double reg = 1e-6;
    const int nx = system_->getStateDim();
    const int nu = system_->getControlDim();
    const int N  = U.size();

    // Compute linearized dynamics.
    std::vector<Eigen::MatrixXd> A_dyn, B_dyn;
    computeLinearizedDynamics(X, U, A_dyn, B_dyn);

    const int n_states = (N+1)*nx;
    const int n_controls = N*nu;
    const int n_dec = n_states + n_controls;

    // Reference for cost (using objective's reference state).
    Eigen::VectorXd x_ref = objective_->getReferenceState(); 
    Eigen::VectorXd u_ref = Eigen::VectorXd::Zero(nu); 
    
    // Build quadratic cost.
    std::vector<Eigen::Triplet<double>> H_triplets;
    H_triplets.reserve(n_dec * (nx+nu));
    g = Eigen::VectorXd::Zero(n_dec);

    // Running cost for states.
    for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd Q = objective_->getRunningCostStateHessian(X[t], U[t], t);
        Q *= 2.0;
        Eigen::VectorXd q_x = -Q * x_ref;
        int x_idx = t*nx;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                double val = Q(i,j);
                if (i == j) val += reg;
                if (std::fabs(val) > 1e-12) {
                    H_triplets.push_back({x_idx + i, x_idx + j, val});
                }
            }
        }
        g.segment(x_idx, nx) += q_x;
    }

    // Terminal cost.
    {
        Eigen::MatrixXd QN = objective_->getFinalCostHessian(X[N]);
        QN *= 2.0;
        Eigen::VectorXd q_xN = -QN * x_ref;
        int x_idx = N*nx;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                double val = QN(i,j);
                if (i == j) val += reg;
                if (std::fabs(val) > 1e-12) {
                    H_triplets.push_back({x_idx + i, x_idx + j, val});
                }
            }
        }
        g.segment(x_idx, nx) += q_xN;
    }

    // Control cost.
    for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd R = objective_->getRunningCostControlHessian(X[t], U[t], t);
        R *= 2.0;
        Eigen::VectorXd q_u = -R * u_ref;
        int u_idx_start = n_states + t*nu;
        for (int i = 0; i < nu; ++i) {
            for (int j = 0; j < nu; ++j) {
                double val = R(i,j);
                if (i == j) val += reg;
                if (std::fabs(val) > 1e-12) {
                    H_triplets.push_back({u_idx_start + i, u_idx_start + j, val});
                }
            }
        }
        g.segment(u_idx_start, nu) += q_u;
    }

    // Build equality constraints for dynamics.
    std::vector<Eigen::Triplet<double>> A_triplets;
    int n_eq = (N+1)*nx;  // dynamics and initial state constraints.
    // Initial state constraint: x0 = initial_state_.
    for (int i = 0; i < nx; ++i) {
        A_triplets.push_back({i, i, 1.0});
    }
    int eq_row = nx;
    for (int t = 0; t < N; ++t) {
        for (int i = 0; i < nx; ++i) {
            // x_{t+1} term.
            A_triplets.push_back({eq_row + i, (t+1)*nx + i, 1.0});
            // -A_dyn[t]*x_t.
            for (int j = 0; j < nx; ++j) {
                double val = -A_dyn[t](i,j);
                if (std::fabs(val) > 1e-12) {
                    A_triplets.push_back({eq_row + i, t*nx + j, val});
                }
            }
            // -B_dyn[t]*u_t.
            for (int j = 0; j < nu; ++j) {
                double val = -B_dyn[t](i,j);
                if (std::fabs(val) > 1e-12) {
                    A_triplets.push_back({eq_row + i, n_states + t*nu + j, val});
                }
            }
        }
        eq_row += nx;
    }

    // Build inequality constraints (box constraints).
    // Here we impose trust region bounds on states and bounds on controls.
    auto control_box_constraint = getConstraint<ControlBoxConstraint>("ControlBoxConstraint");
    Eigen::VectorXd xmin = Eigen::VectorXd::Constant(nx, -trust_region_radius);
    Eigen::VectorXd xmax = Eigen::VectorXd::Constant(nx, trust_region_radius);
    Eigen::VectorXd umin = control_box_constraint->getLowerBound();
    Eigen::VectorXd umax = control_box_constraint->getUpperBound();

    int n_ineq = (N+1)*nx + N*nu;
    int n_con = n_eq + n_ineq; // initial total constraints.
    l.resize(n_con);
    u.resize(n_con);

    // Equality constraints bounds.
    for (int i = 0; i < nx; ++i) {
        l(i) = initial_state_(i);
        u(i) = initial_state_(i);
    }
    for (int i = nx; i < n_eq; ++i) {
        l(i) = 0.0;
        u(i) = 0.0;
    }

    // Inequality constraints: add identity rows.
    int ineq_start = n_eq;
    for (int i = 0; i < n_ineq; ++i) {
        A_triplets.push_back({ineq_start + i, i, 1.0});
    }
    // State bounds for t = 0..N.
    for (int t = 0; t <= N; ++t) {
        for (int i = 0; i < nx; ++i) {
            l(ineq_start + t*nx + i) = xmin(i);
            u(ineq_start + t*nx + i) = xmax(i);
        }
    }
    // Control bounds for t = 0..N-1.
    for (int t = 0; t < N; ++t) {
        for (int i = 0; i < nu; ++i) {
            int idx = ineq_start + (N+1)*nx + t*nu + i;
            l(idx) = umin(i);
            u(idx) = umax(i);
        }
    }

    // ----- Add Terminal State Equality Constraint -----
    // Enforce x_N = reference_state_ (goal state).
    int n_term = nx;  // one constraint per state variable.
    int new_total_con = n_con + n_term;
    l.conservativeResize(new_total_con);
    u.conservativeResize(new_total_con);
    for (int i = 0; i < nx; ++i) {
        // x_N is at indices [N*nx, (N+1)*nx-1] in the decision vector.
        A_triplets.push_back({n_con + i, N*nx + i, 1.0});
        l(n_con + i) = reference_state_(i);
        u(n_con + i) = reference_state_(i);
    }
    n_con = new_total_con; // update total number of constraints.

    // Build final sparse matrices.
    H.resize(n_dec, n_dec);
    A.resize(n_con, n_dec);
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());
    // Ensure H is symmetric.
    Eigen::SparseMatrix<double> H_t = H.transpose();
    H = 0.5 * (H + H_t);
}

void SQPSolver::extractUpdates(const Eigen::VectorXd& qp_solution,
                               std::vector<Eigen::VectorXd>& dX,
                               std::vector<Eigen::VectorXd>& dU) {
    const int nx = system_->getStateDim();
    const int nu = system_->getControlDim();
    const int N  = static_cast<int>(U_.size());
    dX.resize(N+1);
    dU.resize(N);
    const int n_states = (N+1)*nx;
    const int n_controls = N*nu;
    if (qp_solution.size() != n_states + n_controls) {
        throw std::runtime_error("QP solution size does not match expected dimensions.");
    }
    for (int t = 0; t <= N; ++t) {
        dX[t] = qp_solution.segment(t * nx, nx);
    }
    for (int t = 0; t < N; ++t) {
        dU[t] = qp_solution.segment(n_states + t * nu, nu);
    }
}

double SQPSolver::computeConstraintViolation(const std::vector<Eigen::VectorXd>& X,
                                               const std::vector<Eigen::VectorXd>& U) const {
    double total_violation = 0.0;
    const int N = static_cast<int>(U.size());
    const int nx = system_->getStateDim();
    const int nu = system_->getControlDim();
    for (int t = 0; t <= N; ++t) {
        const Eigen::VectorXd& x_t = X[t];
        Eigen::VectorXd u_t = (t < N) ? U[t] : Eigen::VectorXd::Zero(nu);
        auto control_box_constraint = getConstraint<ControlBoxConstraint>("ControlBoxConstraint");
        Eigen::VectorXd g_val = control_box_constraint->evaluate(x_t, u_t);
        Eigen::VectorXd lb = control_box_constraint->getLowerBound();
        Eigen::VectorXd ub = control_box_constraint->getUpperBound();
        for (int i = 0; i < g_val.size(); ++i) {
            double lower_diff = lb(i) - g_val(i);
            double upper_diff = g_val(i) - ub(i);
            if (lower_diff > 0) total_violation += lower_diff;
            if (upper_diff > 0) total_violation += upper_diff;
        }
    }
    return total_violation;
}

double SQPSolver::computeMeritFunction(const std::vector<Eigen::VectorXd>& X,
                                       const std::vector<Eigen::VectorXd>& U,
                                       double penalty) const {
    double cost = objective_->evaluate(X, U);
    double viol = computeConstraintViolation(X, U);
    return cost + penalty * viol;
}

double SQPSolver::lineSearch(const std::vector<Eigen::VectorXd>& X,
                             const std::vector<Eigen::VectorXd>& U,
                             const std::vector<Eigen::VectorXd>& dX,
                             const std::vector<Eigen::VectorXd>& dU) {
    double alpha = 1.0;
    const double c = 1e-4;  // Armijo constant.
    double merit_curr = computeMeritFunction(X, U, options_.merit_penalty);
    // Backtrack until the merit function decreases sufficiently.
    for (int i = 0; i < options_.line_search_max_iterations; ++i) {
        std::vector<Eigen::VectorXd> X_candidate, U_candidate;
        updateTrajectories(X, U, dX, dU, X_candidate, U_candidate, alpha);
        double merit_candidate = computeMeritFunction(X_candidate, U_candidate, options_.merit_penalty);
        if (merit_candidate < merit_curr - c * alpha) {
            break;
        }
        alpha *= options_.tau;
    }
    return alpha;
}

void SQPSolver::updateTrajectories(const std::vector<Eigen::VectorXd>& X,
                                   const std::vector<Eigen::VectorXd>& U,
                                   const std::vector<Eigen::VectorXd>& dX,
                                   const std::vector<Eigen::VectorXd>& dU,
                                   std::vector<Eigen::VectorXd>& X_new,
                                   std::vector<Eigen::VectorXd>& U_new,
                                   double alpha) {
    const int N = U.size();
    X_new.resize(N + 1);
    U_new.resize(N);
    X_new[0] = initial_state_;
    auto control_box_constraint = getConstraint<ControlBoxConstraint>("ControlBoxConstraint");
    for (int t = 0; t < N; ++t) {
        U_new[t] = U[t] + alpha * dU[t];
        U_new[t] = control_box_constraint->clamp(U_new[t]);
        X_new[t+1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
    }
}

} // namespace cddp
