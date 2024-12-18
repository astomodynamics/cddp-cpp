/*
 * Implementation of Sequential Quadratic Programming Solver
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

    // // Initialize OSQP settings with defaults
    // osqp_settings_.eps_abs = options_.osqp_eps_abs;
    // osqp_settings_.eps_rel = options_.osqp_eps_rel;
    // osqp_settings_.max_iter = options_.osqp_max_iter;
    // osqp_settings_.verbose = options_.osqp_verbose;
    // osqp_settings_.warm_start = options_.warm_start;
    // osqp_settings_.adaptive_rho = true;

    // Initialize storage
    initializeSQP();
}

void SQPSolver::initializeSQP() {
    if (!system_) {
        return;  // Wait until system is set to know dimensions
    }

    const int state_dim = system_->getStateDim();
    const int control_dim = system_->getControlDim();

    // Initialize trajectories
    X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(state_dim));
    U_.resize(horizon_, Eigen::VectorXd::Zero(control_dim));

    // Set initial and final states
    X_[0] = initial_state_;
    // X_.back() = reference_state_;
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

    // Forward propagate dynamics
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
        // Discretize and store
        A[t] = (At * timestep_ + Eigen::MatrixXd::Identity(At.rows(), At.cols()));
        B[t] = Bt * timestep_;
    }
}

SQPResult SQPSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SQPResult result;
    result.success = false;
    result.iterations = 0;

    // Check if system and objective are set
    if (!system_ || !objective_) {
        throw std::runtime_error("System or objective not set");
    }

    // Initial propagation
    propagateDynamics(initial_state_, U_);

    // Initialize trust region radius
    double trust_region_radius = options_.trust_region_radius;

    // Current trajectory
    std::vector<Eigen::VectorXd> X_curr = X_;
    std::vector<Eigen::VectorXd> U_curr = U_;

    // Initialize J_curr and viol_curr
    double J_curr = objective_->evaluate(X_curr, U_curr);
    double viol_curr = computeConstraintViolation(X_curr, U_curr);

    if (options_.verbose) {
        std::cout << "Initial cost: " << J_curr << std::endl;
        std::cout << "Initial constraint violation: " << viol_curr << std::endl;
    }

    // Set up OSQP solver instance
    osqp::OsqpInstance instance;
    osqp::OsqpSolver solver;

    // Main SQP loop
    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        result.iterations++;

        // 1. Form QP subproblem
        Eigen::SparseMatrix<double> H, A;
        Eigen::VectorXd g, l, u;
        formQPSubproblem(X_curr, U_curr, trust_region_radius, H, g, A, l, u);

        // 2. Set up and solve QP
        instance.objective_matrix = H;
        instance.objective_vector = g;
        instance.constraint_matrix = A;
        instance.lower_bounds = l;
        instance.upper_bounds = u;

        if (iter == 0) {
            // Initialize solver
            auto status = solver.Init(instance, osqp_settings_);
            if (!status.ok()) {
                if (options_.verbose) {
                    std::cout << "OSQP initialization failed: " << status.message() << std::endl;
                }
                return result;
            }
        } else {
            // Update existing solver
            solver.UpdateObjectiveMatrix(H);
            solver.SetObjectiveVector(g);
            solver.UpdateConstraintMatrix(A);
            solver.SetBounds(l, u);
        }

        // Solve QP
        auto exit_code = solver.Solve();
        if (exit_code != osqp::OsqpExitCode::kOptimal) {
            if (options_.verbose) {
                std::cout << "QP solve failed at iteration " << iter << std::endl;
            }
            break;
        }

        // 3. Extract solution updates
        std::vector<Eigen::VectorXd> dX, dU;
        extractUpdates(solver.primal_solution(), dX, dU);
        
        // 4. Update trajectories
        std::vector<Eigen::VectorXd> X_new, U_new;
        updateTrajectories(X_curr, U_curr, dX, dU, X_new, U_new);

        // 5. Evaluate merit function and update trust region
        double J_new = objective_->evaluate(X_new, U_new);
        double viol_new = computeConstraintViolation(X_new, U_new);

        // Compute the predicted reduction (from the QP subproblem)
        double predicted_reduction = computeMeritFunction(X_curr, U_curr, options_.merit_penalty) - 
                                     computeMeritFunction(X_new, U_new, options_.merit_penalty);

        // Ensure the predicted reduction is non-negative (or very small) to avoid issues with the ratio
        predicted_reduction = std::max(predicted_reduction, 1e-12); 

        double rho = (J_curr - J_new) / predicted_reduction;

        // Trust region update
        if (rho > options_.trust_region_eta1) {
            // Accept the step
            X_curr = X_new;
            U_curr = U_new;
            J_curr = J_new;
            viol_curr = viol_new;

            // Increase trust region radius
            trust_region_radius = std::min(options_.trust_region_radius_max, trust_region_radius * options_.trust_region_gamma1); 
        } else if (rho > options_.trust_region_eta2) {
            // Accept the step
            X_curr = X_new;
            U_curr = U_new;
            J_curr = J_new;
            viol_curr = viol_new;

            // Keep trust region radius the same
        } else {
            // Reject the step
            // Keep X_curr and U_curr unchanged

            // Decrease trust region radius
            trust_region_radius = trust_region_radius * options_.trust_region_gamma2;
        }
        
        result.obj_history.push_back(J_new);
        result.viol_history.push_back(viol_new);

        // Compute relative changes
        double dJ = std::abs(J_new - J_curr) / (std::abs(J_curr) + 1e-10);
        double max_step = 0.0;
        for (size_t i = 0; i < U_new.size(); ++i) {
            max_step = std::max(max_step, (U_new[i] - U_curr[i]).lpNorm<Eigen::Infinity>());
        }

        // Check convergence criteria
        if (dJ < options_.ftol && max_step < options_.xtol && viol_new < options_.gtol && iter >= options_.min_iterations) {
            result.success = true;
            if (options_.verbose) {
                std::cout << "SQP converged at iteration " << iter << std::endl;
                std::cout << "Final cost: " << J_new << ", violation: " << viol_new << std::endl;
            }
            J_curr = J_new;
            viol_curr = viol_new;
            break;
        }

        // // Update current solution
        // X_curr = X_new;
        // U_curr = U_new;
        // J_curr = J_new;
        // viol_curr = viol_new;

        if (options_.verbose) {
            std::cout << "Iter " << iter 
                      << ": cost=" << J_curr 
                      << ", viol=" << viol_curr 
                      << std::endl;
        }
    }

    // Store final results
    result.X = X_curr;
    result.U = U_curr;
    result.objective_value = J_curr;
    result.constraint_violation = viol_curr;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.solve_time = std::chrono::duration<double>(end_time - start_time).count();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (options_.verbose) {
        std::cout << "\nSQP finished in " << result.iterations << " iterations" << std::endl;
        std::cout << "Solve time: " << result.solve_time << " seconds" << std::endl;
        std::cout << "          : " << duration.count() << " microseconds" << std::endl;
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
    const int N  = U.size();  // horizon

    // Compute linearized dynamics
    std::vector<Eigen::MatrixXd> A_dyn, B_dyn;
    computeLinearizedDynamics(X, U, A_dyn, B_dyn);

    // Decision variables: z = [x_0; x_1; ...; x_N; u_0; ...; u_{N-1}]
    const int n_states = (N+1)*nx;
    const int n_controls = N*nu;
    const int n_dec = n_states + n_controls;

    // Initialize reference states and controls
    Eigen::VectorXd x_ref = objective_->getReferenceState(); 
    Eigen::VectorXd u_ref = Eigen::VectorXd::Zero(nu); 
    
    std::vector<Eigen::Triplet<double>> H_triplets;
    H_triplets.reserve(n_dec * (nx+nu));

    g = Eigen::VectorXd::Zero(n_dec);

    // Build quadratic cost for states and controls
    // State costs for t=0,...,N-1
    for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd Q = objective_->getRunningCostStateHessian(X[t], U[t], t);
        Q *= 2.0;
        // Gradient w.r.t x_t: g_x_t = -2Q x_ref
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

    // Terminal cost at x_N
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

    // Control costs
    for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd R = objective_->getRunningCostControlHessian(X[t], U[t], t);
        R *= 2.0;  // For QP form
        // Gradient w.r.t u_t: g_u_t = -2 R u_ref
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

    // Build constraints
    // Equality constraints:
    // x_0 = initial_state_
    // For t=0..N-1: x_{t+1} = A_dyn[t]*x_t + B_dyn[t]*u_t

    std::vector<Eigen::Triplet<double>> A_triplets;

    // Total eq constraints: (N+1)*nx
    int n_eq = (N+1)*nx;

    // Initial state constraint:
    for (int i = 0; i < nx; ++i) {
        A_triplets.push_back({i, i, 1.0}); // x_0(i) = initial_state_(i)
    }

    int eq_row = nx;
    for (int t = 0; t < N; ++t) {
        for (int i = 0; i < nx; ++i) {
            // x_{t+1}
            A_triplets.push_back({eq_row + i, (t+1)*nx + i, 1.0});
            // -A_dyn[t]*x_t
            for (int j = 0; j < nx; ++j) {
                double val = -A_dyn[t](i,j);
                if (std::fabs(val) > 1e-12) {
                    A_triplets.push_back({eq_row + i, t*nx + j, val});
                }
            }
            // -B_dyn[t]*u_t
            for (int j = 0; j < nu; ++j) {
                double val = -B_dyn[t](i,j);
                if (std::fabs(val) > 1e-12) {
                    A_triplets.push_back({eq_row + i, n_states + t*nu + j, val});
                }
            }
        }
        eq_row += nx;
    }

    // Inequality constraints:
    // z = [x_0; ...; x_N; u_0; ...; u_{N-1}]
    // Suppose we have state bounds xmin, xmax and control bounds umin, umax
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");
    Eigen::VectorXd xmin, xmax, umin, umax;
    // Obtain these from your system or constraints:
    xmin = Eigen::VectorXd::Constant(nx, -trust_region_radius); 
    xmax = Eigen::VectorXd::Constant(nx, trust_region_radius);  
    umin = control_box_constraint->getLowerBound();
    umax = control_box_constraint->getUpperBound();

    // Number of inequality constraints = (N+1)*nx + N*nu
    // We add them as Aineq = I * z
    int n_ineq = (N+1)*nx + N*nu;
    int n_con = n_eq + n_ineq;

    l.resize(n_con);
    u.resize(n_con);

    // Equality constraints bounds:
    // initial state: x_0 = initial_state_
    for (int i = 0; i < nx; ++i) {
        l(i) = initial_state_(i);
        u(i) = initial_state_(i);
    }

    // dynamics: 0 = x_{t+1} - A_dyn[t]x_t - B_dyn[t]u_t
    for (int i = nx; i < n_eq; ++i) {
        l(i) = 0.0;
        u(i) = 0.0;
    }

    // Inequality constraints (Aineq = I):
    int ineq_start = n_eq;
    for (int i = 0; i < n_ineq; ++i) {
        A_triplets.push_back({ineq_start + i, i, 1.0});
    }

    // Fill in state and control bounds // Trust region bounds
    // States: for t=0..N TODO: Check if this is correct
    for (int t = 0; t <= N; ++t) {
        for (int i = 0; i < nx; ++i) {
            l(ineq_start + t*nx + i) = xmin(i);
            u(ineq_start + t*nx + i) = xmax(i);
        }
    }

    // Controls: for t=0..N-1 
    for (int t = 0; t < N; ++t) {
        for (int i = 0; i < nu; ++i) {
            int idx = ineq_start + (N+1)*nx + t*nu + i;
            l(idx) = umin(i);
            u(idx) = umax(i);
        }
    }

    // Build final sparse matrices
    H.resize(n_dec, n_dec);
    A.resize(n_con, n_dec);

    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    A.setFromTriplets(A_triplets.begin(), A_triplets.end());

    // Symmetrize H
    Eigen::SparseMatrix<double> H_t = H.transpose();
    H = 0.5 * (H + H_t);
}

void SQPSolver::extractUpdates(const Eigen::VectorXd& qp_solution,
                               std::vector<Eigen::VectorXd>& dX,
                               std::vector<Eigen::VectorXd>& dU) {
    const int nx = system_->getStateDim();
    const int nu = system_->getControlDim();
    const int N  = static_cast<int>(U_.size());  // horizon

    // Resize output vectors
    dX.resize(N+1);
    dU.resize(N);

    // Number of state decision variables
    const int n_states = (N+1)*nx;
    // Number of control decision variables
    const int n_controls = N*nu;
    // Total decision variables
    const int n_dec = n_states + n_controls;

    if (qp_solution.size() != n_dec) {
        throw std::runtime_error("QP solution size does not match expected dimensions.");
    }

    // Extract dX
    // x_t is located at indices [t*nx : (t+1)*nx - 1]
    for (int t = 0; t <= N; ++t) {
        dX[t] = qp_solution.segment(t * nx, nx);
    }

    // Extract dU
    // u_t is located at indices [n_states + t*nu : n_states + (t+1)*nu - 1]
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
        
        // For the last time step, there may be no control input
        Eigen::VectorXd u_t;
        if (t < N) {
            u_t = U[t];
        } else {
            u_t = Eigen::VectorXd::Zero(nu);
        }

        auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");
        Eigen::VectorXd g_val = control_box_constraint->evaluate(x_t, u_t);
        Eigen::VectorXd lb = control_box_constraint->getLowerBound();
        Eigen::VectorXd ub = control_box_constraint->getUpperBound();

        // Compute the violation for each element of the constraint
        for (int i = 0; i < g_val.size(); ++i) {
            double val = g_val(i);

            double lower_diff = lb(i) - val;
            double upper_diff = val - ub(i);

            if (lower_diff > 0) {
                total_violation += lower_diff;
            }
            if (upper_diff > 0) {
                total_violation += upper_diff;
            }
        }
        
    }

    return total_violation;
}



double SQPSolver::computeMeritFunction(const std::vector<Eigen::VectorXd>& X,
                                       const std::vector<Eigen::VectorXd>& U,
                                       double eta) const {
    // Evaluate the objective (cost)
    double cost = objective_->evaluate(X, U);

    // Compute constraint violation measure
    double viol = computeConstraintViolation(X, U);

    // Merit function: cost + eta * violation
    double merit = cost + eta * viol;

    return merit;
}


void SQPSolver::updateTrajectories(const std::vector<Eigen::VectorXd>& X,
                                   const std::vector<Eigen::VectorXd>& U,
                                   const std::vector<Eigen::VectorXd>& dX,
                                   const std::vector<Eigen::VectorXd>& dU,
                                   std::vector<Eigen::VectorXd>& X_new,
                                   std::vector<Eigen::VectorXd>& U_new) {
    const int N = static_cast<int>(U.size()); // Horizon length

    // Resize output trajectories
    X_new.resize(N + 1);
    U_new.resize(N);
    X_new[0] = initial_state_;

    // // Update states
    // for (int t = 0; t <= N; ++t) {
    //     X_new[t] = X[t] + alpha * dX[t];
    // }

    // // Update controls
    // for (int t = 0; t < N; ++t) {
    //     U_new[t] = U[t] + alpha * dU[t];
    // }
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Compute candidate trajectories
    for (size_t t = 0; t < U.size(); ++t) {
        U_new[t] = U[t] + dU[t];
        // Clamp control input
        U_new[t] = control_box_constraint->clamp(U_new[t]);
        X_new[t + 1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
    }

}

} // namespace cddp