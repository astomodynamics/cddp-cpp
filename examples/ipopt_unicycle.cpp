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

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <filesystem>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

int main() {
    ////////// Problem Setup //////////
    const int state_dim   = 3;   // [x, y, theta]
    const int control_dim = 2;   // [v, omega]
    const int horizon     = 100; // Number of control intervals
    const double timestep = 0.03; // Time step

    // Define initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;         

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;          
    // Define cost weighting matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);

    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    // Terminal cost weight (optional if using terminal constraint)
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0, 0) = 100.0;  // x position
    Qf(1, 1) = 100.0;  // y position
    Qf(2, 2) = 100.0;  // heading


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

    casadi::MX g; 

    // Initial state constraint: X₀ = initial_state
    casadi::DM init_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + state_dim));
    g = casadi::MX::vertcat({g, X_t(0) - init_state_dm});

    // Dynamics constraints: 
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_next_expr = unicycle_dynamics(X_t(t), U_t(t));
        g = casadi::MX::vertcat({g, X_t(t + 1) - x_next_expr});
    }

    // --- Terminal Condition Constraint ---
    casadi::DM goal_dm(std::vector<double>(goal_state.data(), goal_state.data() + state_dim));
    casadi::MX terminal_constr = X_t(horizon) - goal_dm;
    g = casadi::MX::vertcat({g, terminal_constr});
    
    ////////// Cost Function //////////
    casadi::MX cost = casadi::MX::zeros(1, 1);
    for (int t = 0; t < horizon; t++) {
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

    ////////// Variable Bounds and Initial Guess //////////
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

    // Convert to state and control trajectories
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));
    std::vector<double> t_sol(horizon + 1);
    for (int t = 0; t <= horizon; t++) {
        t_sol[t] = t * timestep;
    }

    for (int t = 0; t <= horizon; t++)
    {
        for (int i = 0; i < state_dim; i++)
        {
            X_sol[t](i) = sol[t * state_dim + i];
        }
    }

    for (int t = 0; t < horizon; t++)
    {
        for (int i = 0; i < control_dim; i++)
        {
            U_sol[t](i) = sol[n_states + t * control_dim + i];
        }
    }

    // Create directory for saving plot (if it doesn't exist)
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Plot the solution (x-y plane)
    std::vector<double> x_arr, y_arr, theta_arr;
    for (const auto& x : X_sol) {
        x_arr.push_back(x(0));
        y_arr.push_back(x(1));
        theta_arr.push_back(x(2));
    }

    // Plot the solution (control inputs)
    std::vector<double> v_arr, omega_arr;
    for (const auto& u : U_sol) {
        v_arr.push_back(u(0));
        omega_arr.push_back(u(1));
    }

    // -----------------------------
    // Plot states and controls
    // -----------------------------
    auto f1 = figure();
    f1->size(1200, 800);

    // First subplot: Position Trajectory
    auto ax1 = subplot(3, 1, 0);
    auto plot_handle = plot(ax1, x_arr, y_arr, "-b");
    plot_handle->line_width(3);
    title(ax1, "Position Trajectory");
    xlabel(ax1, "x [m]");
    ylabel(ax1, "y [m]");

    // Second subplot: Heading Angle vs Time
    auto ax2 = subplot(3, 1, 1);
    auto heading_plot_handle = plot(ax2, t_sol, theta_arr);
    heading_plot_handle->line_width(3);
    title(ax2, "Heading Angle");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "theta [rad]");

    // Fourth subplot: Control Inputs
    auto ax4 = subplot(3, 1, 2);
    auto p1 = plot(ax4, v_arr, "--b");
    p1->line_width(3);
    p1->display_name("Acceleration");

    hold(ax4, true);
    auto p2 = plot(ax4, omega_arr, "--r");
    p2->line_width(3);
    p2->display_name("Steering");

    title(ax4, "Control Inputs");
    xlabel(ax4, "Step");
    ylabel(ax4, "Control");
    matplot::legend(ax4);

    f1->draw();
    f1->save(plotDirectory + "/unicycle_ipopt_results.png");

    // -----------------------------
    // Animation: unicycle Trajectory
    // -----------------------------
    auto f2 = figure();
    f2->size(800, 600);
    auto ax_anim = f2->current_axes();
    if (!ax_anim)
    {
        ax_anim = axes();
    }

    double car_length = 0.35;
    double car_width = 0.15;

    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        if (i % 10 == 0)
        {
            ax_anim->clear();
            hold(ax_anim, true);

            double x = x_arr[i];
            double y = y_arr[i];
            double theta = theta_arr[i];

            // Compute unicycle rectangle corners
            std::vector<double> car_x(5), car_y(5);
            car_x[0] = x + car_length / 2 * cos(theta) - car_width / 2 * sin(theta);
            car_y[0] = y + car_length / 2 * sin(theta) + car_width / 2 * cos(theta);
            car_x[1] = x + car_length / 2 * cos(theta) + car_width / 2 * sin(theta);
            car_y[1] = y + car_length / 2 * sin(theta) - car_width / 2 * cos(theta);
            car_x[2] = x - car_length / 2 * cos(theta) + car_width / 2 * sin(theta);
            car_y[2] = y - car_length / 2 * sin(theta) - car_width / 2 * cos(theta);
            car_x[3] = x - car_length / 2 * cos(theta) - car_width / 2 * sin(theta);
            car_y[3] = y - car_length / 2 * sin(theta) + car_width / 2 * cos(theta);
            car_x[4] = car_x[0];
            car_y[4] = car_y[0];

            auto car_line = plot(ax_anim, car_x, car_y);
            car_line->color("black");
            car_line->line_style("solid");
            car_line->line_width(2);
            car_line->display_name("Car");

            // Plot trajectory up to current frame
            std::vector<double> traj_x(x_arr.begin(), x_arr.begin() + i + 1);
            std::vector<double> traj_y(y_arr.begin(), y_arr.begin() + i + 1);
            auto traj_line = plot(ax_anim, traj_x, traj_y);
            traj_line->color("blue");
            traj_line->line_style("solid");
            traj_line->line_width(1.5);
            traj_line->display_name("Trajectory");

            title(ax_anim, "unicycle Trajectory");
            xlabel(ax_anim, "x [m]");
            ylabel(ax_anim, "y [m]");
            xlim(ax_anim, {-1, 2.2});
            ylim(ax_anim, {-1, 2.2});
            // legend(ax_anim);

            std::string filename = plotDirectory + "/unicycle_frame_" + std::to_string(i) + ".png";
            f2->draw();
            f2->save(filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 30 " + plotDirectory + "/unicycle_frame_*.png " + plotDirectory + "/unicycle_ipopt.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/unicycle_frame_*.png";
    std::system(cleanup_command.c_str());


    std::cout << "GIF animation created successfully: " << plotDirectory + "/unicycle_ipopt.gif" << std::endl;

    return 0;
}

// :~/github/cddp-cpp/build$ ./examples/ipopt_unicycle 

// ******************************************************************************
// This program contains Ipopt, a library for large-scale nonlinear optimization.
//  Ipopt is released as open source code under the Eclipse Public License (EPL).
//          For more information visit http://projects.coin-or.org/Ipopt
// ******************************************************************************

// This is Ipopt version 3.11.9, running with linear solver mumps.
// NOTE: Other linear solvers might be more efficient (see Ipopt documentation).

// Number of nonzeros in equality constraint Jacobian...:     1106
// Number of nonzeros in inequality constraint Jacobian.:        0
// Number of nonzeros in Lagrangian Hessian.............:      506

// Total number of variables............................:      503
//                      variables with only lower bounds:        0
//                 variables with lower and upper bounds:      200
//                      variables with only upper bounds:        0
// Total number of equality constraints.................:      306
// Total number of inequality constraints...............:        0
//         inequality constraints with only lower bounds:        0
//    inequality constraints with lower and upper bounds:        0
//         inequality constraints with only upper bounds:        0

// iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
//    0  0.0000000e+00 2.00e-02 0.00e+00  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
//    1  1.3213637e+01 1.45e-02 5.18e+00  -1.0 3.57e+00    -  2.22e-01 2.77e-01h  1
//    2  4.8004394e+01 2.56e-02 3.60e+00  -1.0 2.64e+00   0.0 2.24e-01 3.65e-01h  1
//    3  6.2828049e+01 2.00e-02 3.08e+00  -1.0 3.97e+00    -  1.91e-01 2.19e-01h  1
//    4  5.3377401e+01 1.25e-02 1.56e+00  -1.0 1.01e+00    -  3.34e-01 3.98e-01f  1
//    5  5.6693670e+01 3.61e-03 6.71e-01  -1.0 1.78e+00    -  9.31e-01 7.28e-01h  1
//    6  6.0580150e+01 2.37e-04 4.35e-02  -1.7 1.95e-01    -  9.48e-01 1.00e+00h  1
//    7  6.0448863e+01 2.28e-05 2.20e-01  -2.5 1.36e-01    -  6.39e-01 1.00e+00f  1
//    8  6.0165740e+01 1.35e-05 6.29e-02  -2.5 1.28e-01    -  7.42e-01 1.00e+00f  1
//    9  6.0066412e+01 4.10e-06 5.07e-04  -2.5 5.72e-02    -  1.00e+00 1.00e+00f  1
// iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
//   10  5.9914306e+01 9.25e-06 2.97e-02  -3.8 5.45e-02    -  6.88e-01 1.00e+00f  1
//   11  5.9875609e+01 1.08e-06 3.38e-03  -3.8 2.49e-02    -  9.21e-01 1.00e+00f  1
//   12  5.9869812e+01 6.18e-08 7.97e-06  -3.8 7.32e-03    -  1.00e+00 1.00e+00h  1
//   13  5.9857746e+01 9.70e-08 5.52e-04  -5.7 5.35e-03    -  9.49e-01 1.00e+00f  1
//   14  5.9857287e+01 2.38e-09 5.54e-07  -5.7 2.47e-03    -  1.00e+00 1.00e+00h  1
//   15  5.9857127e+01 3.02e-10 6.60e-08  -7.0 9.07e-04    -  1.00e+00 1.00e+00h  1
//   16  5.9857126e+01 1.05e-12 3.89e-10  -7.0 1.04e-04    -  1.00e+00 1.00e+00h  1

// Number of Iterations....: 16

//                                    (scaled)                 (unscaled)
// Objective...............:   5.9857125548898239e+01    5.9857125548898239e+01
// Dual infeasibility......:   3.8946268432482611e-10    3.8946268432482611e-10
// Constraint violation....:   1.0502709812953981e-12    1.0502709812953981e-12
// Complementarity.........:   1.0259906103085439e-07    1.0259906103085439e-07
// Overall NLP error.......:   1.0259906103085439e-07    1.0259906103085439e-07


// Number of objective function evaluations             = 17
// Number of objective gradient evaluations             = 17
// Number of equality constraint evaluations            = 17
// Number of inequality constraint evaluations          = 0
// Number of equality constraint Jacobian evaluations   = 17
// Number of inequality constraint Jacobian evaluations = 0
// Number of Lagrangian Hessian evaluations             = 16
// Total CPU secs in IPOPT (w/o function evaluations)   =      0.017
// Total CPU secs in NLP function evaluations           =      0.004

// EXIT: Optimal Solution Found.
//       solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
//        nlp_f  | 242.00us ( 14.24us) 241.34us ( 14.20us)        17
//        nlp_g  | 503.00us ( 29.59us) 503.62us ( 29.62us)        17
//   nlp_grad_f  | 493.00us ( 27.39us) 490.74us ( 27.26us)        18
//   nlp_hess_l  |   1.71ms (106.69us)   1.71ms (106.86us)        16
//    nlp_jac_g  |   1.85ms (102.56us)   1.85ms (102.81us)        18
//        total  |  23.72ms ( 23.72ms)  23.73ms ( 23.73ms)         1
// Solver elapsed time: 0.0238267 s
// GIF animation created successfully: ../results/tests/unicycle_ipopt.gif