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
#include <filesystem>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

int main() {
    // Dimensions and time parameters
    const int state_dim   = 4;   // [x, θ, x_dot, θ_dot]
    const int control_dim = 1;   // [force]
    const int horizon     = 100; // Number of time steps
    const double timestep = 0.05; // Time discretization

    // Cartpole physical parameters
    double cart_mass   = 1.0;    // mass of the cart (kg)
    double pole_mass   = 0.2;    // mass of the pole (kg)
    double pole_length = 0.5;    // length of the pole (m)
    double gravity     = 9.81;   // gravitational acceleration (m/s²)

    // Define initial and goal states.
    // State order: [x, θ, x_dot, θ_dot]
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, 0.0, 0.0;    // Cart at origin; pole hanging downward (θ = 0)
    
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, M_PI, 0.0, 0.0;        // Cart at origin; pole upright (θ = π)

    // Cost weighting matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0,0) = 100.0;  // Cart position
    Qf(1,1) = 100.0;  // Pole angle
    Qf(2,2) = 100.0;  // Cart velocity
    Qf(3,3) = 100.0;  // Pole angular velocity

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

    // Control bounds (e.g., force limits)
    Eigen::VectorXd control_lower_bound(control_dim);
    Eigen::VectorXd control_upper_bound(control_dim);
    control_lower_bound << -10.0;
    control_upper_bound <<  10.0;

    ////////// Decision Variables //////////
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

    auto cartpole_dynamics = [=](casadi::MX state, casadi::MX control) -> casadi::MX {
        // Extract state components
        casadi::MX pos       = state(0);  // cart position
        casadi::MX theta     = state(1);  // pole angle
        casadi::MX x_dot     = state(2);
        casadi::MX theta_dot = state(3);
        casadi::MX force     = control(0);

        // Use casadi's trigonometric functions
        using casadi::cos;
        using casadi::sin;
        casadi::MX sin_theta = sin(theta);
        casadi::MX cos_theta = cos(theta);
        casadi::MX total_mass = cart_mass + pole_mass;
        casadi::MX den = cart_mass + pole_mass * sin_theta * sin_theta;

        casadi::MX x_ddot = (force + pole_mass * sin_theta * (pole_length * theta_dot * theta_dot + gravity * cos_theta)) / den;
        casadi::MX theta_ddot = (-force * cos_theta 
                                 - pole_mass * pole_length * theta_dot * theta_dot * cos_theta * sin_theta
                                 - total_mass * gravity * sin_theta) / (pole_length * den);

        // Euler discretization:
        casadi::MX next_state = casadi::MX::vertcat({
            pos       + timestep * x_dot,
            theta     + timestep * theta_dot,
            x_dot     + timestep * x_ddot,
            theta_dot + timestep * theta_ddot
        });
        return next_state;
    };

    ////////// Constraints //////////
    casadi::MX g; 

    // Initial state constraint: 
    casadi::DM init_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + state_dim));
    g = casadi::MX::vertcat({g, X_t(0) - init_state_dm});

    // Dynamics constraints: 
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_next_expr = cartpole_dynamics(X_t(t), U_t(t));
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

    std::vector<double> lbx(n_dec, -1e20), ubx(n_dec, 1e20);
    // Apply control bounds to the control part of the decision vector.
    for (int t = 0; t < horizon; t++) {
        for (int i = 0; i < control_dim; i++) {
            lbx[n_states + t * control_dim + i] = control_lower_bound(i);
            ubx[n_states + t * control_dim + i] = control_upper_bound(i);
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

    std::map<std::string, casadi::MX> nlp;
    nlp["x"] = z;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict solver_opts;
    solver_opts["print_time"]       = true;
    solver_opts["ipopt.print_level"]  = 5;
    solver_opts["ipopt.max_iter"]     = 500;
    solver_opts["ipopt.tol"]          = 1e-6;

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Convert initial guess and bounds into DM objects.
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

    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(arg);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Solver elapsed time: " << elapsed.count() << " s" << std::endl;

    std::vector<double> sol = std::vector<double>(res.at("x"));

    // Recover state and control trajectories.
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));
    for (int t = 0; t <= horizon; t++) {
        Eigen::VectorXd x_t(state_dim);
        for (int i = 0; i < state_dim; i++) {
            x_t(i) = sol[t * state_dim + i];
        }
        X_sol[t] = x_t;
    }
    for (int t = 0; t < horizon; t++) {
        Eigen::VectorXd u_t(control_dim);
        for (int i = 0; i < control_dim; i++) {
            u_t(i) = sol[n_states + t * control_dim + i];
        }
        U_sol[t] = u_t;
    }

    // Create plot directory.
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Create a time vector.
    std::vector<double> t_sol(horizon + 1);
    for (int t = 0; t <= horizon; t++) {
        t_sol[t] = t * timestep;
    }
    
    // Extract solution data.
    std::vector<double> x_arr, x_dot_arr, theta_arr, theta_dot_arr, force_arr, time_arr, time_arr2;
    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        theta_arr.push_back(X_sol[i](1));
        x_dot_arr.push_back(X_sol[i](2));
        theta_dot_arr.push_back(X_sol[i](3));
    }
    for (size_t i = 0; i < U_sol.size(); ++i) {
        force_arr.push_back(U_sol[i](0));
        time_arr2.push_back(t_sol[i]);
    }

    // --- Plot static results (2x2 plots for state trajectories) ---
    auto fig1 = figure();
    fig1->size(1200, 800);

    auto ax1 = subplot(2, 2, 1);
    title(ax1, "Cart Position");
    plot(ax1, time_arr, x_arr)->line_style("b-");
    xlabel(ax1, "Time [s]");
    ylabel(ax1, "Position [m]");
    grid(ax1, true);

    auto ax2 = subplot(2, 2, 2);
    title(ax2, "Cart Velocity");
    plot(ax2, time_arr, x_dot_arr)->line_style("b-");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Velocity [m/s]");
    grid(ax2, true);

    auto ax3 = subplot(2, 2, 3);
    title(ax3, "Pole Angle");
    plot(ax3, time_arr, theta_arr)->line_style("b-");
    xlabel(ax3, "Time [s]");
    ylabel(ax3, "Angle [rad]");
    grid(ax3, true);

    auto ax4 = subplot(2, 2, 4);
    title(ax4, "Pole Angular Velocity");
    plot(ax4, time_arr, theta_dot_arr)->line_style("b-");
    xlabel(ax4, "Time [s]");
    ylabel(ax4, "Angular Velocity [rad/s]");
    grid(ax4, true);

    fig1->save(plotDirectory + "/cartpole_results.png");

    // --- Plot control inputs ---
    auto fig2 = figure();
    fig2->size(800, 600);
    title("Control Inputs");
    plot(time_arr2, force_arr)->line_style("b-");
    xlabel("Time [s]");
    ylabel("Force [N]");
    grid(true);
    fig2->save(plotDirectory + "/cartpole_control_inputs.png");

    // --- Animation ---
    auto fig3 = figure();
    auto ax_fig3 = fig3->current_axes();
    fig3->size(800, 600);
    title("CartPole Animation");
    xlabel("x");
    ylabel("y");

    double cart_width = 0.3;
    double cart_height = 0.2;
    double pole_width = 0.05;

    // Loop over the solution states to generate animation frames.
    for (size_t i = 0; i < X_sol.size(); ++i) {
        if (i % 5 == 0) {
            // Clear previous content.
            cla(ax_fig3);
            hold(ax_fig3, true);

            // Current state.
            double x = x_arr[i];
            double theta = theta_arr[i];

            // Plot the cart as a rectangle centered at (x, 0).
            std::vector<double> cart_x = { x - cart_width/2, x + cart_width/2,
                                           x + cart_width/2, x - cart_width/2,
                                           x - cart_width/2 };
            std::vector<double> cart_y = { -cart_height/2, -cart_height/2,
                                           cart_height/2, cart_height/2,
                                           -cart_height/2 };
            plot(cart_x, cart_y)->line_style("k-");

            // Plot the pole as a line from the top center of the cart.
            double pole_end_x = x + pole_length * std::sin(theta);
            double pole_end_y = cart_height/2 - pole_length * std::cos(theta);
            std::vector<double> pole_x = { x, pole_end_x };
            std::vector<double> pole_y = { cart_height/2, pole_end_y };
            plot(pole_x, pole_y)->line_style("b-");

            // Plot the pole bob as a circle.
            std::vector<double> circle_x, circle_y;
            int num_points = 20;
            for (int j = 0; j <= num_points; ++j) {
                double t = 2 * M_PI * j / num_points;
                circle_x.push_back(pole_end_x + pole_width * std::cos(t));
                circle_y.push_back(pole_end_y + pole_width * std::sin(t));
            }
            plot(circle_x, circle_y)->line_style("b-");

            // Set fixed axis limits for stable animation.
            xlim({-2.0, 2.0});
            ylim({-1.5, 1.5});

            // Save the frame.
            std::string filename = plotDirectory + "/frame_" + std::to_string(i) + ".png";
            fig3->save(filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // Combine all saved frames into a GIF using ImageMagick's convert tool.
    std::string command = "convert -delay 30 " + plotDirectory + "/frame_*.png " + plotDirectory + "/cartpole_ipopt.gif";
    std::system(command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "Animation saved as cartpole_ipopt.gif" << std::endl;
    
    return 0;
}
