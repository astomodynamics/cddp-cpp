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

    // Current trajectory
    std::vector<Eigen::VectorXd> X_curr = X_;
    std::vector<Eigen::VectorXd> U_curr = U_;

    // Set up OSQP solver instance
    osqp::OsqpInstance instance;
    osqp::OsqpSolver solver;

    // Main SQP loop
    for (int iter = 0; iter < 1; ++iter) {
        result.iterations++;
        // TODO: Implement full SQP iteration
        // 1. Form QP subproblem
        Eigen::SparseMatrix<double> H, A;
        Eigen::VectorXd g, l, u;
        formQPSubproblem(X_curr, U_curr, H, g, A, l, u);

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

        // // 3. Extract solution updates
        // std::vector<Eigen::VectorXd> dX, dU;
        // extractUpdates(solver.primal_solution(), dX, dU);

        // // 4. Line search
        // double alpha = lineSearch(X_curr, U_curr, dX, dU);
        // if (alpha < options_.xtol) {
        //     if (options_.verbose) {
        //         std::cout << "Line search failed at iteration " << iter << std::endl;
        //     }
        //     break;
        // }

        // // 5. Update trajectories
        // std::vector<Eigen::VectorXd> X_new, U_new;
        // updateTrajectories(X_curr, U_curr, dX, dU, alpha, X_new, U_new);

        // // 6. Check for convergence
        // double J_new = objective_->evaluate(X_new, U_new);
        // double viol_new = computeConstraintViolation(X_new);
        
        // result.obj_history.push_back(J_new);
        // result.viol_history.push_back(viol_new);

        // // Compute relative changes
        // double dJ = std::abs(J_new - J_curr) / (std::abs(J_curr) + 1e-10);
        // double max_step = 0.0;
        // for (size_t i = 0; i < U_new.size(); ++i) {
        //     max_step = std::max(max_step, (U_new[i] - U_curr[i]).lpNorm<Eigen::Infinity>());
        // }

        // // Check convergence criteria
        // if (dJ < options_.ftol && max_step < options_.xtol && viol_new < options_.gtol) {
        //     result.success = true;
        //     if (options_.verbose) {
        //         std::cout << "SQP converged at iteration " << iter << std::endl;
        //         std::cout << "Final cost: " << J_new << ", violation: " << viol_new << std::endl;
        //     }
        //     break;
        // }

        // // Update current solution
        // X_curr = X_new;
        // U_curr = U_new;
        // J_curr = J_new;
        // viol_curr = viol_new;

        // if (options_.verbose) {
        //     std::cout << "Iter " << iter 
        //              << ": cost=" << J_curr 
        //              << ", viol=" << viol_curr 
        //              << ", step=" << alpha 
        //              << std::endl;
        // }
    }

    // // Store final results
    // result.X = X_curr;
    // result.U = U_curr;
    // result.objective_value = J_curr;
    // result.constraint_violation = viol_curr;

    // auto end_time = std::chrono::high_resolution_clock::now();
    // result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    // if (options_.verbose) {
    //     std::cout << "\nSQP finished in " << result.iterations << " iterations" << std::endl;
    //     std::cout << "Solve time: " << result.solve_time << " seconds" << std::endl;
    //     std::cout << "Final cost: " << result.objective_value << std::endl;
    //     std::cout << "Final constraint violation: " << result.constraint_violation << std::endl;
    //     std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
    // }

    return result;
}

void SQPSolver::formQPSubproblem(const std::vector<Eigen::VectorXd>& X,
                                 const std::vector<Eigen::VectorXd>& U,
                                 Eigen::SparseMatrix<double>& H,
                                 Eigen::VectorXd& g,
                                 Eigen::SparseMatrix<double>& A,
                                 Eigen::VectorXd& l,
                                 Eigen::VectorXd& u) {
    // Regularization
    const double reg = 1e-6; 

    // Get problem dimensions
    const int nx = system_->getStateDim();
    const int nu = system_->getControlDim();
    const int N  = U.size();  // horizon

    std::vector<Eigen::MatrixXd> A_dyn, B_dyn;
    computeLinearizedDynamics(X, U, A_dyn, B_dyn);

    // Decision variables: z = [x_0; x_1; ...; x_N; u_0; ...; u_{N-1}]
    // Dimensions
    const int n_states = (N+1)*nx;
    const int n_controls = N*nu;
    const int n_dec = n_states + n_controls;

    Eigen::VectorXd x_ref = objective_->getReferenceState(); 

    std::vector<Eigen::Triplet<double>> H_triplets;
    H_triplets.reserve((n_states*nx + n_controls*nu)); // Rough reservation

    g = Eigen::VectorXd::Zero(n_dec);

    // State cost blocks: Q for each x_t
    for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd Q = objective_->getRunningCostStateHessian(X[t], U[t], t);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                double val = Q(i,j);
                if (i == j) val += reg;
                if (std::fabs(val) > 1e-12) {
                    H_triplets.push_back({t*nx + i, t*nx + j, val});
                }
            }
        }
        Eigen::VectorXd q_x = -Q * x_ref;
        g.segment(t*nx, nx) += q_x;
    }

    // Terminal cost QN for x_N
    {
        int t = N;
        Eigen::MatrixXd QN = objective_->getFinalCostHessian(X[t]);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nx; ++j) {
                double val = QN(i,j);
                if (i == j) val += reg;
                if (std::fabs(val) > 1e-12) {
                    H_triplets.push_back({t*nx + i, t*nx + j, val});
                }
            }
        }
        Eigen::VectorXd q_xN = -QN * x_ref;
        g.segment(N*nx, nx) += q_xN;
    }

    // Control cost blocks: R for each u_t
    for (int t = 0; t < N; ++t) {
        Eigen::MatrixXd R = objective_->getRunningCostControlHessian(X[t], U[t], t);
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
    }

    // Build equality constraints
    std::vector<Eigen::Triplet<double>> A_triplets;

    int eq_row = 0;
    for (int t = 0; t < N; ++t) {
        // x_{t+1} part
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

    for (int i = 0; i < nx; ++i) {
        A_triplets.push_back({i, i, 1.0}); // x_0(i)
    }

    // Clear eq_row logic and rebuild:
    A_triplets.clear();

    // Initial state constraint:
    for (int i = 0; i < nx; ++i) {
        A_triplets.push_back({i, i, 1.0}); // x_0(i)
    }

    eq_row = nx; // next set of eq constraints start at row nx

    for (int t = 0; t < N; ++t) {
        for (int i = 0; i < nx; ++i) {
            A_triplets.push_back({eq_row + i, (t+1)*nx + i, 1.0});
            for (int j = 0; j < nx; ++j) {
                double val = -A_dyn[t](i,j);
                if (std::fabs(val) > 1e-12) {
                    A_triplets.push_back({eq_row + i, t*nx + j, val});
                }
            }
            for (int j = 0; j < nu; ++j) {
                double val = -B_dyn[t](i,j);
                if (std::fabs(val) > 1e-12) {
                    A_triplets.push_back({eq_row + i, n_states + t*nu + j, val});
                }
            }
        }
        eq_row += nx;
    }

    // Total eq constraints: (N+1)*nx
    int n_eq = (N+1)*nx;
    int n_ineq = (N+1)*nx + N*nu; 

    // So total constraints: n_con = n_eq + n_ineq
    int n_con = n_eq + ( (N+1)*nx + N*nu );

    // Resize l, u
    l.resize(n_con);
    u.resize(n_con);

    // Set equality part: first n_eq rows
    // For initial state: x_0 - initial_state_ = 0
    for (int i = 0; i < nx; ++i) {
        l(i) = initial_state_(i);
        u(i) = initial_state_(i);
    }

    // For dynamics: they must be zero
    for (int i = nx; i < n_eq; ++i) {
        l(i) = 0.0;
        u(i) = 0.0;
    }

    // Now add Aineq = I
    int ineq_start = n_eq;
    for (int i = 0; i < n_ineq; ++i) {
        A_triplets.push_back({ineq_start + i, i, 1.0});
    }
    
    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");
    Eigen::VectorXd xmin, xmax, umin, umax;
    xmin = Eigen::VectorXd::Constant(nx, -1e6);
    xmax = Eigen::VectorXd::Constant(nx, 1e6);
    umin = control_box_constraint->getLowerBound();
    umax =  control_box_constraint->getUpperBound();

    // Fill in l and u for Aineq * z <= u and Aineq * z >= l
    // Aineq = I, so l(i), u(i) are just the bounds on z(i).
    // z = [x_0; ...; x_N; u_0; ...; u_{N-1}]
    // For states:
    for (int t = 0; t <= N; ++t) {
        for (int i = 0; i < nx; ++i) {
            l(ineq_start + t*nx + i) = xmin(i);
            u(ineq_start + t*nx + i) = xmax(i);
        }
    }
    // For controls:
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



double SQPSolver::computeConstraintViolation(const std::vector<Eigen::VectorXd>& constraint_values) const {
    // TODO: Implement constraint violation computation
    return 0.0;
}

double SQPSolver::computeMeritFunction(const std::vector<Eigen::VectorXd>& X,
                                     const std::vector<Eigen::VectorXd>& U,
                                     double eta) const {
    // TODO: Implement merit function
    return 0.0;
}

double SQPSolver::lineSearch(const std::vector<Eigen::VectorXd>& X,
                           const std::vector<Eigen::VectorXd>& U,
                           const std::vector<Eigen::VectorXd>& dX,
                           const std::vector<Eigen::VectorXd>& dU) {
    // TODO: Implement line search
    return 1.0;
}

void SQPSolver::extractUpdates(const Eigen::VectorXd& qp_solution,
                             std::vector<Eigen::VectorXd>& dX,
                             std::vector<Eigen::VectorXd>& dU) {
    // TODO: Implement update extraction
}

void SQPSolver::updateTrajectories(const std::vector<Eigen::VectorXd>& X,
                                 const std::vector<Eigen::VectorXd>& U,
                                 const std::vector<Eigen::VectorXd>& dX,
                                 const std::vector<Eigen::VectorXd>& dU,
                                 double alpha,
                                 std::vector<Eigen::VectorXd>& X_new,
                                 std::vector<Eigen::VectorXd>& U_new) {
    // TODO: Implement trajectory update
}

} // namespace cddp