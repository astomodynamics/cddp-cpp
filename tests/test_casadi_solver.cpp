#include <iostream>
#include <vector>
#include <chrono>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include "gtest/gtest.h"

TEST(CasadiSolverTest, SolveDubinsCar) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;

    // Define initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Define weighting matrices Q, R, Qf
    Eigen::MatrixXd Q = 5 * Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 1200.0, 0.0, 0.0,
          0.0, 1200.0, 0.0,
          0.0, 0.0, 700.0;

    // Convert Eigen to CasADi DM
    casadi::DM Q_dm(Q.rows(), Q.cols());
    for (int i=0; i<Q.rows(); i++) {
        for (int j=0; j<Q.cols(); j++) {
            Q_dm(i,j) = Q(i,j);
        }
    }

    casadi::DM R_dm(R.rows(), R.cols());
    for (int i=0; i<R.rows(); i++) {
        for (int j=0; j<R.cols(); j++) {
            R_dm(i,j) = R(i,j);
        }
    }

    casadi::DM Qf_dm(Qf.rows(), Qf.cols());
    for (int i=0; i<Qf.rows(); i++) {
        for (int j=0; j<Qf.cols(); j++) {
            Qf_dm(i,j) = Qf(i,j);
        }
    }

    // Control bounds
    Eigen::VectorXd control_lower_bound(control_dim), control_upper_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    control_upper_bound << 1.0, M_PI;

    // Decision variables: Z = [X_0,...,X_N, U_0,...,U_{N-1}]
    int n_states = (horizon+1)*state_dim;
    int n_controls = horizon*control_dim;
    int n_dec = n_states + n_controls;

    // CasADi symbolic variables
    casadi::MX x_var = casadi::MX::sym("X", n_states);
    casadi::MX u_var = casadi::MX::sym("U", n_controls);
    casadi::MX z = casadi::MX::vertcat({x_var, u_var});

    // Helper lambda for indexing
    auto X_t = [&](int t) {
        return x_var(casadi::Slice(t*state_dim, (t+1)*state_dim));
    };
    auto U_t = [&](int t) {
        return u_var(casadi::Slice(t*control_dim, (t+1)*control_dim));
    };

    // Dubins Car discrete dynamics
    auto dubins_dynamics = [&](casadi::MX x, casadi::MX u) {
        casadi::MX x_next(3,1);
        casadi::MX v = u(0);
        casadi::MX omega = u(1);
        casadi::MX theta = x(2);
        using casadi::cos;
        using casadi::sin;

        casadi::MX ctheta = cos(theta);
        casadi::MX stheta = sin(theta);

        x_next(0) = x(0) + v * ctheta * timestep;
        x_next(1) = x(1) + v * stheta * timestep;
        x_next(2) = x(2) + omega*timestep;
        return x_next;
    };

    // Build constraints
    casadi::MX g;   
    // Initial state constraint: X_0 = initial_state
    casadi::DM init_state_dm(std::vector<double>(initial_state.data(), initial_state.data()+state_dim));
    g = casadi::MX::vertcat({g, X_t(0) - init_state_dm});

    // Dynamics constraints: for t=0..N-1
    for (int t=0; t<horizon; t++) {
        casadi::MX x_next_expr = dubins_dynamics(X_t(t), U_t(t));
        g = casadi::MX::vertcat({g, X_t(t+1) - x_next_expr});
    }

    // Cost
    casadi::DM goal_dm(std::vector<double>(goal_state.data(), goal_state.data()+state_dim));
    casadi::MX cost = casadi::MX::zeros(1,1);

    for (int t=0; t<horizon; t++) {
        casadi::MX x_diff = X_t(t) - goal_dm;
        casadi::MX u_diff = U_t(t);
        casadi::MX state_cost = casadi::MX::mtimes({x_diff.T(), Q_dm, x_diff});
        casadi::MX control_cost = casadi::MX::mtimes({u_diff.T(), R_dm, u_diff});
        cost = cost + state_cost + control_cost;
    }

    // Terminal cost
    casadi::MX x_diff_final = X_t(horizon) - goal_dm;
    casadi::MX terminal_cost = casadi::MX::mtimes({x_diff_final.T(), Qf_dm, x_diff_final});
    cost = cost + terminal_cost;

    // Bounds on decision variables (use lbx, ubx)
    std::vector<double> lbx(n_dec, -1e9);
    std::vector<double> ubx(n_dec, 1e9);

    // Apply control bounds
    for (int t=0; t<horizon; t++) {
        for (int i=0; i<control_dim; i++) {
            lbx[n_states + t*control_dim + i] = control_lower_bound(i);
            ubx[n_states + t*control_dim + i] = control_upper_bound(i);
        }
    }

    // Constraints are equalities
    int n_con = (int)g.numel();
    std::vector<double> lbg(n_con, 0.0);
    std::vector<double> ubg(n_con, 0.0);

    // Initial guess
    std::vector<double> x0_guess(n_dec, 0.0);
    for (int i=0; i<state_dim; i++) {
        x0_guess[i] = initial_state(i);
    }

    // Define NLP
    std::map<std::string, casadi::MX> nlp;
    nlp["x"] = z;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict solver_opts;
    solver_opts["print_time"] = true;
    solver_opts["ipopt.print_level"] = 5;
    solver_opts["ipopt.max_iter"] = 200;
    solver_opts["ipopt.tol"] = 1e-6;

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Convert vectors to DM
    casadi::DM x0_dm(x0_guess);
    casadi::DM lbg_dm(lbg);
    casadi::DM ubg_dm(ubg);
    casadi::DM lbx_dm(lbx);
    casadi::DM ubx_dm(ubx);

    casadi::DMDict arg({
        {"x0", x0_dm},
        {"lbx", lbx_dm},
        {"ubx", ubx_dm},
        {"lbg", lbg_dm},
        {"ubg", ubg_dm}
    });

    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(arg);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() * 1e6 << " us" << std::endl;

    // Extract solution
    std::vector<double> sol = (std::vector<double>)res["x"];

    // Print final cost
    double obj_val = static_cast<double>(res["f"]);
    std::cout << "Final objective: " << obj_val << std::endl;

    // Extract trajectories
    std::vector<Eigen::VectorXd> X_sol(horizon+1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));

    for (int t=0; t<=horizon; t++) {
        Eigen::VectorXd x_t(state_dim);
        for (int i=0; i<state_dim; i++) {
            x_t(i) = sol[t*state_dim + i];
        }
        X_sol[t] = x_t;
    }

    for (int t=0; t<horizon; t++) {
        Eigen::VectorXd u_t(control_dim);
        for (int i=0; i<control_dim; i++) {
            u_t(i) = sol[n_states + t*control_dim + i];
        }
        U_sol[t] = u_t;
    }

    // Print final state
    std::cout << "Final state: " << X_sol.back().transpose() << std::endl;
    std::cout << "Goal state : " << goal_state.transpose() << std::endl;

    ASSERT_NEAR((X_sol.back() - goal_state).norm(), 0.0, 1e-1);
}
