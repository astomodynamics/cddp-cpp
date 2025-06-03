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
#include "sqp_core/sqp_core.hpp"
#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

namespace cddp {

SCPSolver::SCPSolver(const Eigen::VectorXd& initial_state,
                               const Eigen::VectorXd& reference_state,
                               int horizon,
                               double timestep)
    : initial_state_(initial_state),
      reference_state_(reference_state),
      horizon_(horizon),
      timestep_(timestep)
{
    // Initialize trajectory estimates (if not provided later).
    initializeSCP();
}

void SCPSolver::setInitialTrajectory(const std::vector<Eigen::VectorXd>& X,
                                            const std::vector<Eigen::VectorXd>& U) {
    X_ = X;
    U_ = U;
}

//
// If trajectories have not been provided, initialize by linear interpolation.
//
void SCPSolver::initializeSCP() {
    int state_dim = initial_state_.size();
    // Initialize X_ with horizon_+1 states via linear interpolation.
    if (X_.empty()) {
        X_.resize(horizon_ + 1);
        for (int t = 0; t <= horizon_; ++t) {
            double alpha = static_cast<double>(t) / horizon_;
            X_[t] = initial_state_ + alpha * (reference_state_ - initial_state_);
        }
        X_[0] = initial_state_;
    }
    // Initialize U_ (if not provided) with zeros.
    if (U_.empty()) {
        int control_dim = 2;
        if (system_) {
            control_dim = system_->getControlDim();
        }
        U_.resize(horizon_);
        for (int t = 0; t < horizon_; ++t) {
            U_[t] = Eigen::VectorXd::Zero(control_dim);
        }
    }
}


void SCPSolver::computeLinearizedDynamics(const std::vector<Eigen::VectorXd>& X,
                                               const std::vector<Eigen::VectorXd>& U,
                                               std::vector<Eigen::MatrixXd>& A,
                                               std::vector<Eigen::MatrixXd>& B) const {
    int state_dim = initial_state_.size();
    int control_dim = U[0].size();
    double eps = 1e-5;
    A.resize(horizon_);
    B.resize(horizon_);
    for (int t = 0; t < horizon_; ++t) {
        Eigen::VectorXd x = X[t];
        Eigen::VectorXd u = U[t];
        Eigen::VectorXd f0 = system_->getDiscreteDynamics(x, u, t * timestep_);
        A[t] = Eigen::MatrixXd::Zero(state_dim, state_dim);
        B[t] = Eigen::MatrixXd::Zero(state_dim, control_dim);
        
        auto [fx, fu] = system_->getJacobians(x, u, t * timestep_);
        A[t] = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * fx;
        B[t] = timestep_ * fu;
    }
}

SCPResult SCPSolver::solve() {
    using namespace casadi;
    SCPResult result;
    
    // --- Parameters (adjust as needed) ---
    double Delta = options_.trust_region_radius;  // initial trust-region radius (Δ₀)
    double convergence_threshold = 1e-3;            // convergence threshold
    int max_it = options_.max_iterations;           // maximum iterations

    // --- Dimensions and Initialization ---
    int state_dim = initial_state_.size();
    int control_dim = U_[0].size();
    int N = horizon_;  // There are N control intervals, so the state trajectory has N+1 points

    // Use the current trajectory estimates stored in X_ and U_
    std::vector<Eigen::VectorXd> X = X_;  // our state trajectory
    std::vector<Eigen::VectorXd> U = U_;

    // If no initial trajectory is provided, you might consider initializing X by linear interpolation:
    // for (int t = 0; t <= N; ++t) {
    //     double alpha = static_cast<double>(t) / N;
    //     X[t] = initial_state_ + alpha * (reference_state_ - initial_state_);
    // }

    // Store previous iterate for linearization and convergence checking.
    std::vector<Eigen::VectorXd> X_prev = X;
    std::vector<Eigen::VectorXd> U_prev = U;

    // Save history of iterates.
    std::vector< std::vector<Eigen::VectorXd> > X_all;
    std::vector< std::vector<Eigen::VectorXd> > U_all;
    X_all.push_back(X);
    U_all.push_back(U);

    bool success = false;
    int it = 1;

    // Start timing.
    auto start = std::chrono::high_resolution_clock::now();

    // --- Main SCP Loop ---
    while (it < max_it) {
        // Compute convergence metric: sum of norm differences between current and previous iterates.
        double conv_metric = 0.0;
        for (size_t t = 0; t < X.size(); ++t)
            conv_metric += (X[t] - X_prev[t]).norm();
        for (size_t t = 0; t < U.size(); ++t)
            conv_metric += (U[t] - U_prev[t]).norm();

        // After at least 3 iterations, check convergence.
        if (it > 2 && conv_metric < convergence_threshold) {
            success = true;
            break;
        }

        // Save the current iterate as previous.
        X_prev = X;
        U_prev = U;

        std::vector<MX> x_vars, u_vars;
        for (int t = 0; t <= N; ++t)
            x_vars.push_back(MX::sym("x_" + std::to_string(t), state_dim));
        for (int t = 0; t < N; ++t)
            u_vars.push_back(MX::sym("u_" + std::to_string(t), control_dim));

        // --- Define the Objective ---
        MX cost = 0;
        for (int t = 0; t < N; ++t) {
            cost += dot(u_vars[t], u_vars[t]);
        }
        DM ref_dm = DM(std::vector<double>(reference_state_.data(),
                                             reference_state_.data() + state_dim));
        MX terminal_error = x_vars[N] - ref_dm;
        cost += 1e6 * dot(terminal_error, terminal_error);

        // --- Define Constraints ---
        std::vector<MX> g;
        // (1) Initial state constraint: x₀ == initial_state.
        DM init_dm = DM(std::vector<double>(initial_state_.data(),
                                              initial_state_.data() + state_dim));
        g.push_back(x_vars[0] - init_dm);

        // (2) Dynamics constraints: linearize the dynamics around (X_prev, U_prev).
        std::vector<Eigen::MatrixXd> A, B;
        computeLinearizedDynamics(X_prev, U_prev, A, B);
        for (int t = 0; t < N; ++t) {
            Eigen::VectorXd f_nom = system_->getDiscreteDynamics(X_prev[t], U_prev[t], t * timestep_);
            DM f_nom_dm = DM(std::vector<double>(f_nom.data(),
                                                   f_nom.data() + state_dim));
            DM xbar = DM(std::vector<double>(X_prev[t].data(),
                                               X_prev[t].data() + state_dim));
            DM ubar = DM(std::vector<double>(U_prev[t].data(),
                                               U_prev[t].data() + control_dim));
            // Convert A[t] and B[t] to CasADi DM.
            DM A_dm = DM::zeros(state_dim, state_dim);
            DM B_dm = DM::zeros(state_dim, control_dim);
            for (int i = 0; i < state_dim; ++i)
                for (int j = 0; j < state_dim; ++j)
                    A_dm(i, j) = A[t](i, j);
            for (int i = 0; i < state_dim; ++i)
                for (int j = 0; j < control_dim; ++j)
                    B_dm(i, j) = B[t](i, j);
            // Linearized dynamics constraint: 
            //   x_{t+1} ≈ f_nom + A*(x_t - xbar) + B*(u_t - ubar)
            MX dyn_lin = f_nom_dm + mtimes(A_dm, (x_vars[t] - xbar))
                                   + mtimes(B_dm, (u_vars[t] - ubar));
            g.push_back(x_vars[t+1] - dyn_lin);
        }

        // (3) Trust-region constraints: force the new decision variables to stay within [-Delta, Delta] of the previous iterate.
        for (int t = 0; t <= N; ++t) {
            g.push_back(x_vars[t] - DM(std::vector<double>(X_prev[t].data(),
                                                             X_prev[t].data() + state_dim)));
        }
        for (int t = 0; t < N; ++t) {
            g.push_back(u_vars[t] - DM(std::vector<double>(U_prev[t].data(),
                                                             U_prev[t].data() + control_dim)));
        }
        MX g_all = vertcat(g);

        // --- Concatenate Decision Variables ---
        std::vector<MX> w_list;
        for (int t = 0; t <= N; ++t)
            w_list.push_back(x_vars[t]);
        for (int t = 0; t < N; ++t)
            w_list.push_back(u_vars[t]);
        MX w = vertcat(w_list);
        int n_w = w.size1();

        // --- Set Variable and Constraint Bounds ---
        std::vector<double> lbw(n_w, -1e20), ubw(n_w, 1e20);
        // For constraints:
        int num_eq = state_dim + N * state_dim;
        int num_trust = (N + 1) * state_dim + N * control_dim;
        std::vector<double> lbg, ubg;
        for (int i = 0; i < num_eq; ++i) {
            lbg.push_back(0.0);
            ubg.push_back(0.0);
        }
        for (int i = 0; i < num_trust; ++i) {
            lbg.push_back(-Delta);
            ubg.push_back(Delta);
        }

        // --- Formulate and Solve the NLP or QP ---
        MXDict nlp = {{"x", w}, {"f", cost}, {"g", g_all}};
        // Dict solver_opts;
        // solver_opts["ipopt.max_iter"] = options_.ipopt_max_iter;
        // solver_opts["ipopt.tol"] = options_.ipopt_tol;
        Function solver_fun = qpsol("solver", "qrqp", nlp);

        // Use the previous iterate as the initial guess.
        std::vector<double> w0(n_w, 0.0);
        int offset = 0;
        for (int t = 0; t <= N; ++t) {
            for (int i = 0; i < state_dim; ++i)
                w0[offset++] = X_prev[t](i);
        }
        for (int t = 0; t < N; ++t) {
            for (int i = 0; i < control_dim; ++i)
                w0[offset++] = U_prev[t](i);
        }

        DMDict solver_args = {{"x0", w0}, {"lbx", lbw}, {"ubx", ubw},
                              {"lbg", lbg}, {"ubg", ubg}};
        DMDict solver_res = solver_fun(solver_args);
        std::vector<double> sol = std::vector<double>(solver_res.at("x"));

        // --- Extract the New Trajectory ---
        std::vector<Eigen::VectorXd> X_new(N + 1, Eigen::VectorXd(state_dim));
        std::vector<Eigen::VectorXd> U_new(N, Eigen::VectorXd(control_dim));
        offset = 0;
        for (int t = 0; t <= N; ++t) {
            Eigen::VectorXd x_new(state_dim);
            for (int i = 0; i < state_dim; ++i)
                x_new(i) = sol[offset++];
            X_new[t] = x_new;
        }
        for (int t = 0; t < N; ++t) {
            Eigen::VectorXd u_new(control_dim);
            for (int i = 0; i < control_dim; ++i)
                u_new(i) = sol[offset++];
            U_new[t] = u_new;
        }

        // --- Update the Current Trajectory ---
        X = X_new;
        U = U_new;
        X_all.push_back(X);
        U_all.push_back(U);

        // --- Adjust Trust-Region Radius ---
        double step_norm = 0.0;
        for (int t = 0; t <= N; ++t)
            step_norm += (X[t] - X_prev[t]).norm();
        for (int t = 0; t < N; ++t)
            step_norm += (U[t] - U_prev[t]).norm();
        if (step_norm > 0.9 * Delta)
            Delta = std::min(Delta * 2.0, 1e6);
        else
            Delta *= 0.5;

        if (options_.verbose)
            std::cout << "[SCP] Iteration " << it << ", convergence metric: " << conv_metric << std::endl;

        it++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.solve_time = std::chrono::duration<double>(end - start).count();
    result.iterations = it;
    result.success = (it < max_it);
    result.X = X;
    result.U = U;
    result.objective_value = 0.0;         // (Not computed in this simplified version)
    result.constraint_violation = 0.0;      // (Not computed in this simplified version)

    // Optionally, check trust-region satisfaction.
    bool B_trust_satisfied = satisfies_trust_region_constraints(X, X_all[X_all.size()-2], Delta);
    if (options_.verbose)
        std::cout << "[SCP] Trust region satisfied: " << (B_trust_satisfied ? "true" : "false") << std::endl;

    return result;
}
} // namespace cddp
