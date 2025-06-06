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

    ////////// NLP Definition and SNOPT Solver Setup //////////
    std::map<std::string, casadi::MX> nlp;
    nlp["x"] = z;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict solver_opts;
    solver_opts["print_time"] = true;
    // SNOPT-specific options
    solver_opts["snopt.print_level"] = 1;
    solver_opts["snopt.major_iterations_limit"] = 500;
    solver_opts["snopt.minor_iterations_limit"] = 500;
    // solver_opts["snopt.major_optimality_tolerance"] = 1e-6;
    // solver_opts["snopt.major_feasibility_tolerance"] = 1e-6;
    // solver_opts["snopt.minor_feasibility_tolerance"] = 1e-6;
    // solver_opts["snopt.verify_level"] = 0;  // 0 = no verification, -1 = cheap test, 1 = individual gradients
    // solver_opts["start"] = "cold";    // cold or warm start

    // Create the NLP solver instance using SNOPT.
    casadi::Function solver = casadi::nlpsol("solver", "snopt", nlp, solver_opts);

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
    std::cout << "SNOPT Solver elapsed time: " << elapsed.count() << " s" << std::endl;

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
    title(ax1, "Position Trajectory (SNOPT)");
    xlabel(ax1, "x [m]");
    ylabel(ax1, "y [m]");

    // Second subplot: Heading Angle vs Time
    auto ax2 = subplot(3, 1, 1);
    auto heading_plot_handle = plot(ax2, t_sol, theta_arr);
    heading_plot_handle->line_width(3);
    title(ax2, "Heading Angle (SNOPT)");
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

    title(ax4, "Control Inputs (SNOPT)");
    xlabel(ax4, "Step");
    ylabel(ax4, "Control");
    matplot::legend(ax4);

    f1->draw();
    f1->save(plotDirectory + "/unicycle_snopt_results.png");

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

            title(ax_anim, "Unicycle Trajectory (SNOPT)");
            xlabel(ax_anim, "x [m]");
            ylabel(ax_anim, "y [m]");
            xlim(ax_anim, {-1, 2.2});
            ylim(ax_anim, {-1, 2.2});
            // legend(ax_anim);

            std::string filename = plotDirectory + "/unicycle_snopt_frame_" + std::to_string(i) + ".png";
            f2->draw();
            f2->save(filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 30 " + plotDirectory + "/unicycle_snopt_frame_*.png " + plotDirectory + "/unicycle_snopt.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/unicycle_snopt_frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "GIF animation created successfully: " << plotDirectory + "/unicycle_snopt.gif" << std::endl;

    return 0;
}

// :~/github/cddp-cpp/build$ ./examples/snopt_unicycle 
//  ==============================
//     SNOPT  C interface  2.2.0   
//  ==============================
//  S N O P T  7.7.7    (Feb 2021)
//  ==============================

//  SNMEMB EXIT 100 -- finished successfully
//  SNMEMB INFO 104 -- memory requirements estimated

//  Trial version of SNOPT -- for evaluation or academic purposes only



//  Nonlinear constraints     306     Linear constraints       0
//  Nonlinear variables       503     Linear variables         0
//  Jacobian  variables       503     Objective variables    503
//  Total constraints         306     Total variables        503



//  The user has defined    1106   out of    1106   constraint gradients.
//  The user has defined     503   out of     503   objective  gradients.


//         Minor NumInf  FP mult  FP step   rgNorm         SumInf     nS
//           100      1  1.3E-01  3.2E-01           5.1588740E+01

//         Minor NonOpt  QP mult  QP step   rgNorm Elastic QP obj     nS
//           200    102 -4.1E-01  1.0E+00  2.9E-11  9.0584333E+04     72

//  Major Minors     Step   nCon Feasible  Optimal  MeritFunction     nS Penalty
//      0    227               1  1.0E-02  2.2E-02  0.0000000E+00     99           r iT
//      1     64  1.0E+00      2  8.2E-02  2.7E+00  9.1086436E+04    109 4.8E+07   rl
//      2      4  8.9E-01      3  8.3E-03  1.4E+00  1.0615859E+04    108 2.7E+04 s  l
//      3     56  1.0E+00      4  7.7E-04  3.9E+00  6.7037234E+01    163 9.4E+02
//      4     24  1.0E+00      5 (1.4E-07) 1.3E-02  6.1572688E+01    140 1.8E+02
//      5     27  1.0E+00      6 (5.3E-08) 2.5E-02  6.1489200E+01    114 7.8E+01
//      6      3  1.0E+00      7 (7.0E-07) 2.9E-02  6.1150122E+01    112 7.8E+01
//      7      1  1.0E+00      8  1.3E-05  6.1E-02  6.0022814E+01    112 7.8E+01
//      8      2  1.0E+00      9  3.8E-06  1.8E-02  5.9907766E+01    113 7.8E+01
//      9      1  1.0E+00     10 (2.7E-09) 7.3E-03  5.9906665E+01    113 7.8E+01

//  Major Minors     Step   nCon Feasible  Optimal  MeritFunction     nS Penalty
//     10      1  1.0E+00     11 (9.1E-07) 4.3E-03  5.9880339E+01    113 7.8E+01
//     11      1  1.0E+00     12  3.7E-06  1.7E-02  5.9858922E+01    113 7.8E+01
//     12      1  1.0E+00     13 (4.0E-09) 2.8E-04  5.9858653E+01    113 7.8E+01   R
//     13      2  1.0E+00     14 (4.9E-11) 3.9E-04  5.9858608E+01    112 7.8E+01 s
//     14      1  1.0E+00     15 (3.1E-07) 4.7E-06  5.9857118E+01    112 7.8E+01
//     15      1  1.0E+00     16 (1.2E-11)(7.7E-07) 5.9857118E+01    112 7.8E+01

//  SNOPTC EXIT   0 -- finished successfully
//  SNOPTC INFO   1 -- optimality conditions satisfied

//  Problem name                 solver
//  No. of iterations                 416   Objective            5.9857118027E+01
//  No. of major iterations            15   Linear    obj. term  0.0000000000E+00
//  Penalty parameter           7.838E+01   Nonlinear obj. term  5.9857118027E+01
//  User function calls (total)        17
//  No. of superbasics                112   No. of basic nonlinears           306
//  No. of degenerate steps             1   Percentage                       0.24
//  Max x                     301 2.0E+00   Max pi                    250 4.2E+01
//  Max Primal infeas         786 2.5E-11   Max Dual infeas           501 3.3E-05
//  Nonlinear constraint violn    2.5E-11



//  Solution printed on file  10

//  Time for MPS input                             0.00 seconds
//  Time for solving problem                       0.03 seconds
//  Time for solution output                       0.00 seconds
//  Time for constraint functions                  0.00 seconds
//  Time for objective function                    0.00 seconds
//       solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
//    nlp_jac_f  | 627.00us ( 33.00us) 625.49us ( 32.92us)        19
//    nlp_jac_g  |   2.19ms (115.26us)   2.19ms (115.40us)        19
//        total  |  31.82ms ( 31.82ms)  31.82ms ( 31.82ms)         1
// SNOPT Solver elapsed time: 0.0319228 s
// GIF animation created successfully: ../results/tests/unicycle_snopt.gif