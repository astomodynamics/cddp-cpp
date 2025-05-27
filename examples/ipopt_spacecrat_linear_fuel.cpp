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
#include <filesystem>
#include <cmath>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

int main()
{
    // Problem parameters
    int state_dim = 8;   // [x y z vx vy vz mass accumulated_control_effort]
    int control_dim = 3; // [ux uy uz]
    int horizon = 400;
    double time_horizon = 400.0;
    double timestep = time_horizon / horizon;

    // HCW parameters
    double mean_motion = 0.001107;
    double initial_mass = 100.0;
    double isp = 300.0;
    double g0 = 9.80665;
    double nominal_radius = 50.0;
    double u_max = 10.0;
    double u_min = -10.0;
    double u_max_norm = 10.0;
    double u_min_norm = 0.0;

    // Initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << nominal_radius, 0.0, 0.0, 0.0, -2.0*mean_motion*nominal_radius, 0.0, initial_mass, 0.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, initial_mass, 0.0;

    // Define variables
    casadi::MX x = casadi::MX::sym("x", state_dim);
    casadi::MX u = casadi::MX::sym("u", control_dim);

    
    casadi::MX vx = x(3);
    casadi::MX vy = x(4);
    casadi::MX vz = x(5);
    casadi::MX mass = x(6);
    casadi::MX accumulated_control_effort = x(7);
    casadi::MX ux = u(0);
    casadi::MX uy = u(1);
    casadi::MX uz = u(2);

    // Continuous dynamics (HCW)
    casadi::MX dx = casadi::MX::zeros(state_dim);

    dx(0) = vx;
    dx(1) = vy;
    dx(2) = vz;
    dx(3) = -2.0*mean_motion*vx + ux / mass;
    dx(4) = -2.0*mean_motion*vy + uy / mass;
    dx(5) = -2.0*mean_motion*vz + uz / mass;
    dx(6) = -ux*ux - uy*uy - uz*uz / (mass*mass);
    dx(7) = ux*ux + uy*uy + uz*uz;

    // Discrete dynamics using Euler integration
    casadi::MX f = x + timestep * dx;
    casadi::Function F("F", {x, u}, {f});

    // Decision variables
    int n_states = (horizon + 1) * state_dim;
    int n_controls = horizon * control_dim;
    casadi::MX X = casadi::MX::sym("X", n_states);
    casadi::MX U = casadi::MX::sym("U", n_controls);
    casadi::MX z = casadi::MX::vertcat({X, U});

    // Objective function terms
    auto running_cost = [&](casadi::MX x, casadi::MX u)
    {
        return 0.0;
    };

    auto terminal_cost = [&](casadi::MX x)
    {
        // Final cost is the terminal mass.
        casadi::MX cost =-x(6);
        return cost;
    };

    // Build objective (general)
    casadi::MX J = 0;
    for (int t = 0; t < horizon; t++)
    {
        casadi::MX x_t = X(casadi::Slice(t * state_dim, (t + 1) * state_dim));
        casadi::MX u_t = U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
        J += running_cost(x_t, u_t);
    }
    J += terminal_cost(X(casadi::Slice(horizon * state_dim, (horizon + 1) * state_dim)));

    // Constraints
    casadi::MX g;

    // Initial condition (general)
    g = X(casadi::Slice(0, state_dim)) -
        casadi::DM(std::vector<double>(initial_state.data(), initial_state.data() + state_dim));

    // Dynamics constraints (general)
    for (int t = 0; t < horizon; t++)
    {
        casadi::MX x_t = X(casadi::Slice(t * state_dim, (t + 1) * state_dim));
        casadi::MX u_t = U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
        casadi::MX x_next = F(casadi::MXVector{x_t, u_t})[0];
        casadi::MX x_next_sym = X(casadi::Slice((t + 1) * state_dim, (t + 2) * state_dim));
        g = casadi::MX::vertcat({g, x_next_sym - x_next});
    }

    // Terminal condition (HCW specific)
    // x, y, z, vx, vy, vz are zero
    casadi::MX x_T = X(casadi::Slice(horizon * state_dim, (horizon + 1) * state_dim));
    g = casadi::MX::vertcat({g, x_T(0)}); // x_terminal = 0
    g = casadi::MX::vertcat({g, x_T(1)}); // y_terminal = 0
    g = casadi::MX::vertcat({g, x_T(2)}); // z_terminal = 0
    g = casadi::MX::vertcat({g, x_T(3)}); // vx_terminal = 0
    g = casadi::MX::vertcat({g, x_T(4)}); // vy_terminal = 0
    g = casadi::MX::vertcat({g, x_T(5)}); // vz_terminal = 0

    // Control constraints (HCW specific)
    // ux, uy, uz are bounded by u_max and u_min (this is defined later)
    // Also, the thrust magnitude is bounded by u_max_norm and u_min_norm
    for (int t = 0; t < horizon; t++)
    {
        casadi::MX u_t = U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
        g = casadi::MX::vertcat({g, casadi::MX::norm_2(u_t) - u_max_norm});
        g = casadi::MX::vertcat({g, casadi::MX::norm_2(u_t) - u_min_norm});
    }

    // NLP
    casadi::MXDict nlp = {
        {"x", z},
        {"f", J},
        {"g", g}};

    // Solver options
    casadi::Dict solver_opts;
    solver_opts["ipopt.max_iter"] = 1000;
    solver_opts["ipopt.print_level"] = 5;
    solver_opts["print_time"] = true;
    solver_opts["ipopt.tol"] = 1e-6;

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Bounds
    std::vector<double> lbx(n_states + n_controls);
    std::vector<double> ubx(n_states + n_controls);
    std::vector<double> lbg(g.size1());
    std::vector<double> ubg(g.size1());

    // State bounds
    for (int t = 0; t <= horizon; t++)
    {
        for (int i = 0; i < state_dim; i++)
        {
            lbx[t * state_dim + i] = -1e20;
            ubx[t * state_dim + i] = 1e20;
        }
    }

    // Control bounds
    for (int t = 0; t < horizon; t++)
    {
        // Acceleration bounds
        lbx[n_states + t * control_dim + 0] = -u_max;
        ubx[n_states + t * control_dim + 0] = u_max;
        lbx[n_states + t * control_dim + 1] = -u_max;
        ubx[n_states + t * control_dim + 1] = u_max;
        lbx[n_states + t * control_dim + 2] = -u_max;
        ubx[n_states + t * control_dim + 2] = u_max;
    }

    // Constraint bounds (all equality constraints)
    for (int i = 0; i < g.size1(); i++)
    {
        lbg[i] = 0;
        ubg[i] = 0;
    }

    // Initial guess
    std::vector<double> x0(n_states + n_controls, 0.0);

    // Initial state
    for (int i = 0; i < state_dim; i++)
    {
        x0[i] = initial_state(i);
    }

    // Linear interpolation for states
    for (int t = 1; t <= horizon; t++)
    {
        for (int i = 0; i < state_dim; i++)
        {
            x0[t * state_dim + i] = initial_state(i) +
                                    (goal_state(i) - initial_state(i)) * t / horizon;
        }
    }

    // Call the solver
    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(casadi::DMDict{
        {"x0", x0},
        {"lbx", lbx},
        {"ubx", ubx},
        {"lbg", lbg},
        {"ubg", ubg}});
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Solve time: " << duration.count() << " microseconds" << std::endl;

    // Extract solution
    std::vector<double> sol = std::vector<double>(res.at("x"));

    // Convert to state and control trajectories
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));

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


    return 0;
}