/**
 * ipopt_unicycle.cpp
 *
 * A CasADi/IPOPT script for solving an optimal control problem for a unicycle model,
 * with a post-solution plot of the generated trajectory.
 *
 * The unicycle dynamics are:
 *     xₖ₊₁ = xₖ + vₖ cos(θₖ) * Δt
 *     yₖ₊₁ = yₖ + vₖ sin(θₖ) * Δt
 *     θₖ₊₁ = θₖ + ωₖ * Δt
 *
 * where the state is [x, y, θ] and the control is [v, ω].
 *
 * A terminal condition is enforced by adding a constraint
 * that the final state equals the goal state.
 *
 * Compilation example (paths may need to be adjusted):
 *   g++ ipopt_unicycle.cpp -o ipopt_unicycle -I/path/to/casadi/include -I/path/to/eigen -I/path/to/matplotlib-cpp
 *
 * Author: Your Name
 * Date: February 2025
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

// Include matplotlib-cpp header (ensure you have this library installed)
#include "matplotlibcpp.hpp"
namespace plt = matplotlibcpp;

int main() {
    ////////// Problem Setup //////////
    const int state_dim   = 3;   // [x, y, theta]
    const int control_dim = 2;   // [v, omega]
    const int horizon     = 100; // Number of control intervals
    const double timestep = 0.03; // Time step

    // Define initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;         // starting at the origin, heading 45° rad

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;            // goal position with heading 90° rad

    // Define cost weighting matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    // (Here you might set Q if you want to penalize state deviations during the trajectory.)

    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    // Terminal cost weight (optional if using terminal constraint)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0, 0) = 100.0;  // x position
    Qf(1, 1) = 100.0;  // y position
    Qf(2, 2) = 100.0;  // heading

    // Convert Eigen matrices to CasADi DM using a nested loop conversion.
    casadi::DM Q_dm(Q.rows(), Q.cols());
    for (int i = 0; i < Q.rows(); i++) {
        for (int j = 0; j < Q.cols(); j++) {
            Q_dm(i, j) = Q(i, j);
        }
    }
    casadi::DM R_dm(R.rows(), R.cols());
    for (int i = 0; i < R.rows(); i++) {
        for (int j = 0; j < R.cols(); j++) {
            R_dm(i, j) = R(i, j);
        }
    }
    casadi::DM Qf_dm(Qf.rows(), Qf.cols());
    for (int i = 0; i < Qf.rows(); i++) {
        for (int j = 0; j < Qf.cols(); j++) {
            Qf_dm(i, j) = Qf(i, j);
        }
    }

    // Define control bounds (for example, v ∈ [-1, 1] and ω ∈ [-π, π])
    Eigen::VectorXd u_min(control_dim), u_max(control_dim);
    u_min << -1.0, -M_PI;
    u_max <<  1.0,  M_PI;

    ////////// Decision Variables //////////
    // The decision vector z contains the state trajectory and control trajectory:
    //    z = [X₀, X₁, ..., X_N, U₀, U₁, ..., U_{N-1}]
    const int n_states   = (horizon + 1) * state_dim;
    const int n_controls = horizon * control_dim;
    const int n_dec      = n_states + n_controls;

    // Define symbolic variables for states and controls
    casadi::MX X = casadi::MX::sym("X", n_states);
    casadi::MX U = casadi::MX::sym("U", n_controls);
    casadi::MX z = casadi::MX::vertcat({X, U});

    // Helper lambdas to extract the state and control at time step t
    auto X_t = [=](int t) -> casadi::MX {
        return X(casadi::Slice(t * state_dim, (t + 1) * state_dim));
    };
    auto U_t = [=](int t) -> casadi::MX {
        return U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
    };

    ////////// Unicycle Dynamics //////////
    // Discrete-time unicycle dynamics using Euler integration.
    // State: [x, y, theta], Control: [v, omega]
    auto unicycle_dynamics = [=](casadi::MX x, casadi::MX u) -> casadi::MX {
        casadi::MX x_next = casadi::MX::zeros(state_dim, 1);
        casadi::MX theta = x(2);
        casadi::MX v     = u(0);
        casadi::MX omega = u(1);

        // Use casadi's trigonometric functions
        using casadi::cos;
        using casadi::sin;
        casadi::MX ctheta = cos(theta);
        casadi::MX stheta = sin(theta);
        // Euler integration discretization
        x_next(0) = x(0) + v * ctheta * timestep;
        x_next(1) = x(1) + v * stheta * timestep;
        x_next(2) = x(2) + omega * timestep;
        return x_next;
    };

    ////////// Constraints //////////
    // We build constraints for the initial condition and the dynamics.
    casadi::MX g; 

    // Initial state constraint: X₀ = initial_state
    casadi::DM init_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + state_dim));
    g = casadi::MX::vertcat({g, X_t(0) - init_state_dm});

    // Dynamics constraints: for t = 0, …, horizon-1, enforce Xₜ₊₁ = f(Xₜ, Uₜ)
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_next_expr = unicycle_dynamics(X_t(t), U_t(t));
        g = casadi::MX::vertcat({g, X_t(t + 1) - x_next_expr});
    }

    // --- Terminal Condition Constraint ---
    // Enforce that the terminal state equals the goal state:
    casadi::DM goal_dm(std::vector<double>(goal_state.data(), goal_state.data() + state_dim));
    casadi::MX terminal_constr = X_t(horizon) - goal_dm;
    g = casadi::MX::vertcat({g, terminal_constr});
    
    ////////// Cost Function //////////
    // Build a cost that penalizes deviations along the trajectory and control effort.
    casadi::MX cost = casadi::MX::zeros(1, 1);
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_diff = X_t(t) - goal_dm;
        casadi::MX u_diff = U_t(t);
        casadi::MX state_cost = casadi::MX::mtimes({x_diff.T(), Q_dm, x_diff});
        casadi::MX control_cost = casadi::MX::mtimes({u_diff.T(), R_dm, u_diff});
        cost = cost + state_cost + control_cost;
    }
    // Terminal cost (optional if using a terminal constraint; here it adds extra penalty)
    casadi::MX x_diff_final = X_t(horizon) - goal_dm;
    casadi::MX terminal_cost = casadi::MX::mtimes({x_diff_final.T(), Qf_dm, x_diff_final});
    cost = cost + terminal_cost;

    ////////// Variable Bounds and Initial Guess //////////
    // Set bounds on the decision vector.
    std::vector<double> lbx(n_dec, -1e20);
    std::vector<double> ubx(n_dec,  1e20);
    // Apply control bounds for the control segments.
    for (int t = 0; t < horizon; t++) {
        for (int i = 0; i < control_dim; i++) {
            lbx[n_states + t * control_dim + i] = u_min(i);
            ubx[n_states + t * control_dim + i] = u_max(i);
        }
    }

    // The complete set of constraints (g) must be equal to zero.
    const int n_g = static_cast<int>(g.size1());
    std::vector<double> lbg(n_g, 0.0);
    std::vector<double> ubg(n_g, 0.0);

    // Provide an initial guess for the decision vector.
    std::vector<double> x0(n_dec, 0.0);
    // Set the initial state portion.
    for (int i = 0; i < state_dim; i++) {
        x0[i] = initial_state(i);
    }
    // Linearly interpolate the state trajectory from initial_state to goal_state.
    for (int t = 1; t <= horizon; t++) {
        for (int i = 0; i < state_dim; i++) {
            x0[t * state_dim + i] = initial_state(i) + (goal_state(i) - initial_state(i)) * (double)t / horizon;
        }
    }
    // The control part of the initial guess remains zero.

    ////////// NLP Definition and IPOPT Solver Setup //////////
    std::map<std::string, casadi::MX> nlp;
    nlp["x"] = z;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict solver_opts;
    solver_opts["print_time"]       = true;
    solver_opts["ipopt.print_level"]  = 5;
    solver_opts["ipopt.max_iter"]     = 500;
    solver_opts["ipopt.tol"]          = 1e-6;

    // Create the NLP solver instance using IPOPT.
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Convert the initial guess and bounds into DM objects.
    casadi::DM x0_dm = casadi::DM(x0);
    casadi::DM lbx_dm = casadi::DM(lbx);
    casadi::DM ubx_dm = casadi::DM(ubx);
    casadi::DM lbg_dm = casadi::DM(lbg);
    casadi::DM ubg_dm = casadi::DM(ubg);

    casadi::DMDict arg({
        {"x0", x0_dm},
        {"lbx", lbx_dm},
        {"ubx", ubx_dm},
        {"lbg", lbg_dm},
        {"ubg", ubg_dm}
    });

    ////////// Solve the NLP //////////
    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(arg);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Solver elapsed time: " << elapsed.count() << " s" << std::endl;

    ////////// Extract and Display the Solution //////////
    // The result 'res["x"]' is a DM vector with the optimized decision variables.
    std::vector<double> sol = std::vector<double>(res.at("x"));

    // Extract the state trajectory for plotting: collect x and y coordinates.
    std::vector<double> x_hist, y_hist;
    for (int t = 0; t <= horizon; t++) {
        double x_val = sol[t * state_dim + 0];
        double y_val = sol[t * state_dim + 1];
        x_hist.push_back(x_val);
        y_hist.push_back(y_val);
    }

    // Print the final state and objective value.
    std::vector<double> final_state(state_dim);
    for (int i = 0; i < state_dim; i++) {
        final_state[i] = sol[horizon * state_dim + i];
    }
    std::cout << "Final state:";
    for (double val : final_state)
        std::cout << " " << val;
    std::cout << std::endl;

    double final_obj = static_cast<double>(res.at("f"));
    std::cout << "Final objective: " << final_obj << std::endl;

    ////////// Plotting the Trajectory //////////
    plt::figure();
    // Plot the trajectory as a blue line.
    plt::plot(x_hist, y_hist, "b-");
    // Mark the initial state in green and the goal state in red.
    plt::scatter(std::vector<double>{initial_state(0)}, std::vector<double>{initial_state(1)}, 100, {{"color", "green"}, {"label", "Start"}});
    plt::scatter(std::vector<double>{goal_state(0)}, std::vector<double>{goal_state(1)}, 100, {{"color", "red"}, {"label", "Goal"}});
    plt::xlabel("x");
    plt::ylabel("y");
    plt::title("Unicycle Trajectory");
    plt::legend();
    plt::grid(true);
    plt::show();

    return 0;
}
