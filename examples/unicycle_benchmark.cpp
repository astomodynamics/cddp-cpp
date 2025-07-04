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
#include <cmath>
#include <filesystem>
#include <memory>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <casadi/casadi.hpp>

#include "cddp.hpp" 
#include "matplot/matplot.h"

namespace fs = std::filesystem;
using namespace matplot;

int main() {
    // --------------------------
    // 1. Shared problem setup
    // --------------------------
    const int state_dim = 3;    // [x, y, theta]
    const int control_dim = 2;  // [v, omega]
    const int horizon = 100;
    const double timestep = 0.03;
    const std::string integration_type = "euler";

    // Quadratic cost
    Eigen::MatrixXd Q = 0.01 * Eigen::MatrixXd::Identity(state_dim, state_dim);
    Q(2, 2) = 0.0;
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0,  0.0,   0.0,
           0.0, 100.0,   0.0,
           0.0,   0.0, 100.0;

    // Goal state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 3.0, 3.0, M_PI / 2.0;

    // Empty reference states
    std::vector<Eigen::VectorXd> empty_ref;

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 2.0;

    // Constraint parameters
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 2.0, M_PI;
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -2.0, -M_PI;
    double radius = 0.4;
    Eigen::Vector2d center(1.0, 1.0);
    double radius2 = 0.4;
    Eigen::Vector2d center2(1.5, 2.5);

    // Control bias
    Eigen::VectorXd control_bias(control_dim);
    control_bias << 0.0, 0.0;

    // Create a directory for saving plots
    const std::string plotDirectory = "../results/benchmark";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Helper function to create initial trajectory
    auto createInitialTrajectory = [&]() {
        std::vector<Eigen::VectorXd> X_init(horizon + 1, initial_state);
        std::vector<Eigen::VectorXd> U_init(horizon, Eigen::VectorXd::Zero(control_dim));
        
        auto dyn_system = std::make_unique<cddp::Unicycle>(timestep, integration_type);
        X_init[0] = initial_state;
        for (int i = 0; i < horizon; ++i) {
            U_init[i] = control_bias;
            X_init[i + 1] = dyn_system->getDiscreteDynamics(X_init[i], U_init[i], i * timestep);
        }
        return std::make_pair(X_init, U_init);
    };

    auto [X_init, U_init] = createInitialTrajectory();

    // --------------------------------------------------------
    // 2. Baseline #1: ASDDP (requires higher regularization)
    // --------------------------------------------------------
    std::cout << "Solving with ASDDP..." << std::endl;
    
    cddp::CDDPOptions options_asddp;
    options_asddp.max_iterations = 200;  
    options_asddp.verbose = true;
    options_asddp.debug = false; 
    options_asddp.enable_parallel = false;
    options_asddp.num_threads = 1;
    options_asddp.tolerance = 1e-4;  
    options_asddp.acceptable_tolerance = 1e-5;  
    options_asddp.regularization.initial_value = 1e-2; 

    cddp::CDDP solver_asddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
        options_asddp
    );
    
    solver_asddp.setInitialTrajectory(X_init, U_init);

    // Add constraints (consistent with other solvers)
    solver_asddp.addPathConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    solver_asddp.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_asddp.addPathConstraint("BallConstraint2",
        std::make_unique<cddp::BallConstraint>(radius2, center2));

    // Solve for baseline #1
    auto start_time_asddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_asddp = solver_asddp.solve(cddp::SolverType::ASDDP);
    auto end_time_asddp = std::chrono::high_resolution_clock::now();
    auto solve_time_asddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_asddp - start_time_asddp).count();
    
    auto X_asddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_asddp.at("state_trajectory"));
    auto U_asddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_asddp.at("control_trajectory"));
    double cost_asddp = std::any_cast<double>(sol_asddp.at("final_objective"));
    std::cout << "ASDDP Optimal Cost: " << cost_asddp << std::endl;

    // Extract X and Y coordinates for ASDDP
    std::vector<double> x_asddp, y_asddp;
    for (const auto& state : X_asddp_sol) {
        x_asddp.push_back(state(0));
        y_asddp.push_back(state(1));
    }

    // --------------------------------------------------------
    // 3. Baseline #2: LogDDP    
    // --------------------------------------------------------
    std::cout << "Solving with LogDDP..." << std::endl;
    
    cddp::CDDPOptions options_logddp;
    options_logddp.max_iterations = 10000;
    options_logddp.verbose = true;
    options_logddp.debug = false;
    options_logddp.tolerance = 1e-5;
    options_logddp.acceptable_tolerance = 1e-4;
    options_logddp.regularization.initial_value = 1e-4;
    options_logddp.log_barrier.barrier.mu_initial = 1e-0;
    options_logddp.log_barrier.barrier.mu_update_factor = 0.2;
    options_logddp.log_barrier.relaxed_log_barrier_delta = 1e-12;
    options_logddp.log_barrier.segment_length = horizon;
    options_logddp.log_barrier.use_relaxed_log_barrier_penalty = true;
    options_logddp.log_barrier.rollout_type = "nonlinear";
    options_logddp.log_barrier.use_controlled_rollout = true;

    cddp::CDDP solver_logddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
        options_logddp
    );

    solver_logddp.setInitialTrajectory(X_init, U_init);

    // Add constraints for LogDDP
    solver_logddp.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_logddp.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_logddp.addPathConstraint("BallConstraint2",
        std::make_unique<cddp::BallConstraint>(radius2, center2));

    // Solve for baseline #2: LogDDP
    auto start_time_logddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_logddp = solver_logddp.solve("LogDDP");
    auto end_time_logddp = std::chrono::high_resolution_clock::now();
    auto solve_time_logddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_logddp - start_time_logddp).count();
    
    auto X_logddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_logddp.at("state_trajectory"));
    auto U_logddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_logddp.at("control_trajectory"));
    double cost_logddp = std::any_cast<double>(sol_logddp.at("final_objective"));
    std::cout << "LogDDP Optimal Cost: " << cost_logddp << std::endl;

    // Extract X and Y coordinates for LogDDP
    std::vector<double> x_logddp, y_logddp;
    for (const auto& state : X_logddp_sol) {
        x_logddp.push_back(state(0));
        y_logddp.push_back(state(1));
    }

    // --------------------------------------------------------
    // 4. Baseline #3: IPDDP    
    // --------------------------------------------------------
    std::cout << "Solving with IPDDP..." << std::endl;
    
    cddp::CDDPOptions options_ipddp;
    options_ipddp.max_iterations = 200;
    options_ipddp.verbose = true;
    options_ipddp.debug = false;
    options_ipddp.tolerance = 1e-5;
    options_ipddp.acceptable_tolerance = 1e-6;
    options_ipddp.ipddp.barrier.mu_initial = 1e-0;
    options_ipddp.ipddp.barrier.mu_update_factor = 0.5;
    options_ipddp.ipddp.barrier.mu_update_power = 1.2;
    
    cddp::CDDP solver_ipddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
        options_ipddp
    );

    solver_ipddp.setInitialTrajectory(X_init, U_init);

    // Add constraints for IPDDP
    solver_ipddp.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_ipddp.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_ipddp.addPathConstraint("BallConstraint2",
        std::make_unique<cddp::BallConstraint>(radius2, center2));
    
    // Solve for baseline #3: IPDDP
    auto start_time_ipddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_ipddp = solver_ipddp.solve(cddp::SolverType::IPDDP);
    auto end_time_ipddp = std::chrono::high_resolution_clock::now();
    auto solve_time_ipddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ipddp - start_time_ipddp).count();
    
    auto X_ipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp.at("state_trajectory"));
    auto U_ipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp.at("control_trajectory"));
    double cost_ipddp = std::any_cast<double>(sol_ipddp.at("final_objective"));
    std::cout << "IPDDP Optimal Cost: " << cost_ipddp << std::endl;

    // Extract X and Y coordinates for IPDDP
    std::vector<double> x_ipddp, y_ipddp;
    for (const auto& state : X_ipddp_sol) {
        x_ipddp.push_back(state(0));
        y_ipddp.push_back(state(1));
    }

    // --------------------------------------------------------
    // 5. Baseline #4: MSIPDDP    
    // --------------------------------------------------------
    std::cout << "Solving with MSIPDDP..." << std::endl;
    
    cddp::CDDPOptions options_msipddp;
    options_msipddp.max_iterations = 200;
    options_msipddp.verbose = true;
    options_msipddp.debug = false;
    options_msipddp.tolerance = 1e-5;
    options_msipddp.acceptable_tolerance = 1e-6;
    options_msipddp.msipddp.segment_length = 5;
    options_msipddp.msipddp.rollout_type = "nonlinear";
    options_msipddp.msipddp.use_controlled_rollout = false;
    options_msipddp.msipddp.barrier.mu_initial = 1e-0;
    options_msipddp.msipddp.barrier.mu_update_factor = 0.5;
    options_msipddp.msipddp.barrier.mu_update_power = 1.2;
    
    cddp::CDDP solver_msipddp(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
        options_msipddp
    );

    solver_msipddp.setInitialTrajectory(X_init, U_init);

    // Add constraints for MSIPDDP
    solver_msipddp.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_msipddp.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_msipddp.addPathConstraint("BallConstraint2",
        std::make_unique<cddp::BallConstraint>(radius2, center2));

    // Solve for baseline #4: MSIPDDP
    auto start_time_msipddp = std::chrono::high_resolution_clock::now();
    cddp::CDDPSolution sol_msipddp = solver_msipddp.solve(cddp::SolverType::MSIPDDP);
    auto end_time_msipddp = std::chrono::high_resolution_clock::now();
    auto solve_time_msipddp = std::chrono::duration_cast<std::chrono::microseconds>(end_time_msipddp - start_time_msipddp).count();
    
    auto X_msipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_msipddp.at("state_trajectory"));
    auto U_msipddp_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_msipddp.at("control_trajectory"));
    double cost_msipddp = std::any_cast<double>(sol_msipddp.at("final_objective"));
    std::cout << "MSIPDDP Optimal Cost: " << cost_msipddp << std::endl;

    // Extract X and Y coordinates for MSIPDDP
    std::vector<double> x_msipddp, y_msipddp;
    for (const auto& state : X_msipddp_sol) {
        x_msipddp.push_back(state(0));
        y_msipddp.push_back(state(1));
    }   

    // --------------------------------------------------------
    // 5. Baseline #5 & #6: IPOPT and SNOPT (using CasADi)
    // --------------------------------------------------------
    // NOTE: Both solvers reuse the same NLP problem definition to avoid duplication
    std::cout << "Solving with IPOPT..." << std::endl;
    
    std::vector<Eigen::VectorXd> X_ipopt_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_ipopt_sol(horizon, Eigen::VectorXd(control_dim));
    std::vector<double> x_ipopt, y_ipopt;
    double solve_time_ipopt_numeric = 0.0;
    double cost_ipopt = 0.0;

    { // IPOPT specific scope
        const int n_s = state_dim;    // Renaming for clarity in CasADi context
        const int n_c = control_dim;  // Renaming for clarity in CasADi context
        const int H = horizon;        // Renaming for clarity

        // Define symbolic variables for states and controls
        casadi::MX X_casadi = casadi::MX::sym("X_casadi", (H + 1) * n_s);
        casadi::MX U_casadi = casadi::MX::sym("U_casadi", H * n_c);
        casadi::MX Z_casadi = casadi::MX::vertcat({X_casadi, U_casadi});

        // Helper lambdas to extract the state and control at time step t
        auto X_t = [&](int t) -> casadi::MX {
            return X_casadi(casadi::Slice(t * n_s, (t + 1) * n_s));
        };
        auto U_t = [&](int t) -> casadi::MX {
            return U_casadi(casadi::Slice(t * n_c, (t + 1) * n_c));
        };
        using casadi::cos;
        using casadi::sin;
        
        // Unicycle dynamics function for CasADi
        auto unicycle_dynamics_casadi = [&](casadi::MX x, casadi::MX u) -> casadi::MX {
            casadi::MX x_next = casadi::MX::zeros(n_s, 1);
            casadi::MX theta = x(2);
            casadi::MX v     = u(0);
            casadi::MX omega = u(1);
            casadi::MX ctheta = cos(theta);
            casadi::MX stheta = sin(theta);
            x_next(0) = x(0) + v * ctheta * timestep;
            x_next(1) = x(1) + v * stheta * timestep;
            x_next(2) = x(2) + omega * timestep;
            return x_next;
        };

        // Cost function
        casadi::MX cost_casadi = 0;
        casadi::DM Q_dm = casadi::DM::zeros(n_s, n_s);
        casadi::DM R_dm = casadi::DM::zeros(n_c, n_c);
        casadi::DM Qf_dm = casadi::DM::zeros(n_s, n_s);

        for(int i=0; i<n_s; ++i) for(int j=0; j<n_s; ++j) Q_dm(i,j) = Q(i,j) * timestep;
        for(int i=0; i<n_c; ++i) for(int j=0; j<n_c; ++j) R_dm(i,j) = R(i,j) * timestep;
        for(int i=0; i<n_s; ++i) for(int j=0; j<n_s; ++j) Qf_dm(i,j) = Qf(i,j);
        
        casadi::DM goal_state_dm(std::vector<double>(goal_state.data(), goal_state.data() + n_s));

        for (int t = 0; t < H; ++t) {
            casadi::MX x_diff = X_t(t) - goal_state_dm;
            casadi::MX u_curr = U_t(t);
            cost_casadi += casadi::MX::mtimes({x_diff.T(), Q_dm, x_diff});
            cost_casadi += casadi::MX::mtimes({u_curr.T(), R_dm, u_curr});
        }
        casadi::MX x_final_diff = X_t(H) - goal_state_dm;
        cost_casadi += casadi::MX::mtimes({x_final_diff.T(), Qf_dm, x_final_diff});

        // Constraints
        casadi::MX g_casadi;
        casadi::DM initial_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + n_s));
        g_casadi = casadi::MX::vertcat({g_casadi, X_t(0) - initial_state_dm});

        for (int t = 0; t < H; ++t) {
            g_casadi = casadi::MX::vertcat({g_casadi, X_t(t+1) - unicycle_dynamics_casadi(X_t(t), U_t(t))});
        }

        // Store the number of equality constraints (initial state + dynamics)
        int num_equality_constraints = static_cast<int>(g_casadi.size1());

        // Add Ball Constraints (inequality constraints)
        for (int t = 0; t <= H; ++t) {
            casadi::MX x_coord = X_t(t)(0);
            casadi::MX y_coord = X_t(t)(1);

            // Ball Constraint 1
            casadi::MX term1_c1 = x_coord - center(0);
            casadi::MX term2_c1 = y_coord - center(1);
            casadi::MX ball_constraint1 = term1_c1 * term1_c1 + 
                                          term2_c1 * term2_c1 - 
                                          radius * radius;
            g_casadi = casadi::MX::vertcat({g_casadi, ball_constraint1});

            // Ball Constraint 2
            casadi::MX term1_c2 = x_coord - center2(0);
            casadi::MX term2_c2 = y_coord - center2(1);
            casadi::MX ball_constraint2 = term1_c2 * term1_c2 + 
                                          term2_c2 * term2_c2 - 
                                          radius2 * radius2;
            g_casadi = casadi::MX::vertcat({g_casadi, ball_constraint2});
        }

        // NLP definition
        casadi::MXDict nlp_casadi = {{"x", Z_casadi}, {"f", cost_casadi}, {"g", g_casadi}};
        casadi::Dict solver_opts;
        solver_opts["ipopt.print_level"] = 0;
        solver_opts["print_time"] = true;
        solver_opts["ipopt.tol"] = 1e-6;
        solver_opts["ipopt.acceptable_tol"] = 1e-5;
        casadi::Function solver_ipopt = casadi::nlpsol("solver_ipopt", "ipopt", nlp_casadi, solver_opts);

        // Bounds
        std::vector<double> lbx_casadi((H+1)*n_s + H*n_c, -casadi::inf);
        std::vector<double> ubx_casadi((H+1)*n_s + H*n_c, casadi::inf);

        for (int t = 0; t < H; ++t) {
            for (int i = 0; i < n_c; ++i) {
                lbx_casadi[(H+1)*n_s + t*n_c + i] = control_lower_bound(i);
                ubx_casadi[(H+1)*n_s + t*n_c + i] = control_upper_bound(i);
            }
        }
        
        int n_g_casadi = static_cast<int>(g_casadi.size1());
        std::vector<double> lbg_casadi_vec(n_g_casadi);
        std::vector<double> ubg_casadi_vec(n_g_casadi);

        // Bounds for equality constraints
        for (int i = 0; i < num_equality_constraints; ++i) {
            lbg_casadi_vec[i] = 0.0;
            ubg_casadi_vec[i] = 0.0;
        }

        // Bounds for ball constraints (inequality)
        for (int i = num_equality_constraints; i < n_g_casadi; ++i) {
            lbg_casadi_vec[i] = 0.0;
            ubg_casadi_vec[i] = casadi::inf;
        }

        // Initial guess
        std::vector<double> x0_casadi_vec((H+1)*n_s + H*n_c, 0.0);
        for (int i = 0; i < n_s; ++i) x0_casadi_vec[i] = initial_state(i);
        for (int t = 1; t <= H; ++t) {
            for (int i = 0; i < n_s; ++i) {
                x0_casadi_vec[t*n_s + i] = initial_state(i);
            }
        }

        casadi::DMDict arg_ipopt;
        arg_ipopt["lbx"] = casadi::DM(lbx_casadi);
        arg_ipopt["ubx"] = casadi::DM(ubx_casadi);
        arg_ipopt["lbg"] = casadi::DM(lbg_casadi_vec);
        arg_ipopt["ubg"] = casadi::DM(ubg_casadi_vec);
        arg_ipopt["x0"] = casadi::DM(x0_casadi_vec);
        
        auto start_time_ipopt = std::chrono::high_resolution_clock::now();
        casadi::DMDict res_ipopt = solver_ipopt(arg_ipopt);
        auto end_time_ipopt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_ipopt = end_time_ipopt - start_time_ipopt;
        solve_time_ipopt_numeric = duration_ipopt.count();

        std::vector<double> sol_ipopt_vec = std::vector<double>(res_ipopt.at("x"));

        if (res_ipopt.count("f")) {
            double optimal_cost = static_cast<double>(casadi::DM(res_ipopt.at("f")));
            std::cout << "IPOPT Optimal Cost: " << optimal_cost << std::endl;
            cost_ipopt = optimal_cost;
        }

        for (int t = 0; t <= H; ++t) {
            for (int i = 0; i < n_s; ++i) X_ipopt_sol[t](i) = sol_ipopt_vec[t*n_s + i];
        }
        for (int t = 0; t < H; ++t) {
            for (int i = 0; i < n_c; ++i) U_ipopt_sol[t](i) = sol_ipopt_vec[(H+1)*n_s + t*n_c + i];
        }
        
        for (const auto& state : X_ipopt_sol) {
            x_ipopt.push_back(state(0));
            y_ipopt.push_back(state(1));
        }
        std::cout << "IPOPT solve time: " << solve_time_ipopt_numeric << " seconds" << std::endl; 
    }

    // --------------------------------------------------------
    // SNOPT: Reusing the same NLP definition from IPOPT above
    // --------------------------------------------------------
    std::cout << "Solving with SNOPT..." << std::endl;
    
    std::vector<Eigen::VectorXd> X_snopt_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_snopt_sol(horizon, Eigen::VectorXd(control_dim));
    std::vector<double> x_snopt, y_snopt;
    double solve_time_snopt_numeric = 0.0;
    double cost_snopt = 0.0;

    { // SNOPT specific scope
        const int n_s = state_dim;    // Renaming for clarity in CasADi context
        const int n_c = control_dim;  // Renaming for clarity in CasADi context
        const int H = horizon;        // Renaming for clarity

        // Define symbolic variables for states and controls
        casadi::MX X_casadi = casadi::MX::sym("X_casadi", (H + 1) * n_s);
        casadi::MX U_casadi = casadi::MX::sym("U_casadi", H * n_c);
        casadi::MX Z_casadi = casadi::MX::vertcat({X_casadi, U_casadi});

        // Helper lambdas to extract the state and control at time step t
        auto X_t = [&](int t) -> casadi::MX {
            return X_casadi(casadi::Slice(t * n_s, (t + 1) * n_s));
        };
        auto U_t = [&](int t) -> casadi::MX {
            return U_casadi(casadi::Slice(t * n_c, (t + 1) * n_c));
        };
        using casadi::cos;
        using casadi::sin;
        
        // Unicycle dynamics function for CasADi
        auto unicycle_dynamics_casadi = [&](casadi::MX x, casadi::MX u) -> casadi::MX {
            casadi::MX x_next = casadi::MX::zeros(n_s, 1);
            casadi::MX theta = x(2);
            casadi::MX v     = u(0);
            casadi::MX omega = u(1);
            casadi::MX ctheta = cos(theta);
            casadi::MX stheta = sin(theta);
            x_next(0) = x(0) + v * ctheta * timestep;
            x_next(1) = x(1) + v * stheta * timestep;
            x_next(2) = x(2) + omega * timestep;
            return x_next;
        };

        // Cost function
        casadi::MX cost_casadi = 0;
        casadi::DM Q_dm = casadi::DM::zeros(n_s, n_s);
        casadi::DM R_dm = casadi::DM::zeros(n_c, n_c);
        casadi::DM Qf_dm = casadi::DM::zeros(n_s, n_s);

        for(int i=0; i<n_s; ++i) for(int j=0; j<n_s; ++j) Q_dm(i,j) = Q(i,j) * timestep;
        for(int i=0; i<n_c; ++i) for(int j=0; j<n_c; ++j) R_dm(i,j) = R(i,j) * timestep;
        for(int i=0; i<n_s; ++i) for(int j=0; j<n_s; ++j) Qf_dm(i,j) = Qf(i,j);
        
        casadi::DM goal_state_dm(std::vector<double>(goal_state.data(), goal_state.data() + n_s));

        for (int t = 0; t < H; ++t) {
            casadi::MX x_diff = X_t(t) - goal_state_dm;
            casadi::MX u_curr = U_t(t);
            cost_casadi += casadi::MX::mtimes({x_diff.T(), Q_dm, x_diff});
            cost_casadi += casadi::MX::mtimes({u_curr.T(), R_dm, u_curr});
        }
        casadi::MX x_final_diff = X_t(H) - goal_state_dm;
        cost_casadi += casadi::MX::mtimes({x_final_diff.T(), Qf_dm, x_final_diff});

        // Constraints
        casadi::MX g_casadi;
        casadi::DM initial_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + n_s));
        g_casadi = casadi::MX::vertcat({g_casadi, X_t(0) - initial_state_dm});

        for (int t = 0; t < H; ++t) {
            g_casadi = casadi::MX::vertcat({g_casadi, X_t(t+1) - unicycle_dynamics_casadi(X_t(t), U_t(t))});
        }

        // Store the number of equality constraints (initial state + dynamics)
        int num_equality_constraints = static_cast<int>(g_casadi.size1());

        // Add Ball Constraints (inequality constraints)
        for (int t = 0; t <= H; ++t) {
            casadi::MX x_coord = X_t(t)(0);
            casadi::MX y_coord = X_t(t)(1);

            // Ball Constraint 1
            casadi::MX term1_c1 = x_coord - center(0);
            casadi::MX term2_c1 = y_coord - center(1);
            casadi::MX ball_constraint1 = term1_c1 * term1_c1 + 
                                          term2_c1 * term2_c1 - 
                                          radius * radius;
            g_casadi = casadi::MX::vertcat({g_casadi, ball_constraint1});

            // Ball Constraint 2
            casadi::MX term1_c2 = x_coord - center2(0);
            casadi::MX term2_c2 = y_coord - center2(1);
            casadi::MX ball_constraint2 = term1_c2 * term1_c2 + 
                                          term2_c2 * term2_c2 - 
                                          radius2 * radius2;
            g_casadi = casadi::MX::vertcat({g_casadi, ball_constraint2});
        }

        // NLP definition
        casadi::MXDict nlp_casadi = {{"x", Z_casadi}, {"f", cost_casadi}, {"g", g_casadi}};
        casadi::Dict solver_opts;
        solver_opts["snopt.print_level"] = 1;
        solver_opts["print_time"] = true;
        solver_opts["snopt.major_iterations_limit"] = 500;
        solver_opts["snopt.minor_iterations_limit"] = 500;
        casadi::Function solver_snopt = casadi::nlpsol("solver_snopt", "snopt", nlp_casadi, solver_opts);

        // Bounds
        std::vector<double> lbx_casadi((H+1)*n_s + H*n_c, -casadi::inf);
        std::vector<double> ubx_casadi((H+1)*n_s + H*n_c, casadi::inf);

        for (int t = 0; t < H; ++t) {
            for (int i = 0; i < n_c; ++i) {
                lbx_casadi[(H+1)*n_s + t*n_c + i] = control_lower_bound(i);
                ubx_casadi[(H+1)*n_s + t*n_c + i] = control_upper_bound(i);
            }
        }
        
        int n_g_casadi = static_cast<int>(g_casadi.size1());
        std::vector<double> lbg_casadi_vec(n_g_casadi);
        std::vector<double> ubg_casadi_vec(n_g_casadi);

        // Bounds for equality constraints
        for (int i = 0; i < num_equality_constraints; ++i) {
            lbg_casadi_vec[i] = 0.0;
            ubg_casadi_vec[i] = 0.0;
        }

        // Bounds for ball constraints (inequality)
        for (int i = num_equality_constraints; i < n_g_casadi; ++i) {
            lbg_casadi_vec[i] = 0.0;
            ubg_casadi_vec[i] = casadi::inf;
        }

        // Initial guess
        std::vector<double> x0_casadi_vec((H+1)*n_s + H*n_c, 0.0);
        for (int i = 0; i < n_s; ++i) x0_casadi_vec[i] = initial_state(i);
        for (int t = 1; t <= H; ++t) {
            for (int i = 0; i < n_s; ++i) {
                x0_casadi_vec[t*n_s + i] = initial_state(i);
            }
        }

        casadi::DMDict arg_snopt;
        arg_snopt["lbx"] = casadi::DM(lbx_casadi);
        arg_snopt["ubx"] = casadi::DM(ubx_casadi);
        arg_snopt["lbg"] = casadi::DM(lbg_casadi_vec);
        arg_snopt["ubg"] = casadi::DM(ubg_casadi_vec);
        arg_snopt["x0"] = casadi::DM(x0_casadi_vec);
        
        auto start_time_snopt = std::chrono::high_resolution_clock::now();
        casadi::DMDict res_snopt = solver_snopt(arg_snopt);
        auto end_time_snopt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_snopt = end_time_snopt - start_time_snopt;
        solve_time_snopt_numeric = duration_snopt.count();

        std::vector<double> sol_snopt_vec = std::vector<double>(res_snopt.at("x"));

        if (res_snopt.count("f")) {
            double optimal_cost = static_cast<double>(casadi::DM(res_snopt.at("f")));
            std::cout << "SNOPT Optimal Cost: " << optimal_cost << std::endl;
            cost_snopt = optimal_cost;
        }

        for (int t = 0; t <= H; ++t) {
            for (int i = 0; i < n_s; ++i) X_snopt_sol[t](i) = sol_snopt_vec[t*n_s + i];
        }
        for (int t = 0; t < H; ++t) {
            for (int i = 0; i < n_c; ++i) U_snopt_sol[t](i) = sol_snopt_vec[(H+1)*n_s + t*n_c + i];
        }
        
        for (const auto& state : X_snopt_sol) {
            x_snopt.push_back(state(0));
            y_snopt.push_back(state(1));
        }
        std::cout << "SNOPT solve time: " << solve_time_snopt_numeric << " seconds" << std::endl; 
    }

    // --------------------------------------------------------
    // 7. Plot all trajectories in one figure
    // --------------------------------------------------------
    auto main_figure = figure(true);
    main_figure->size(3600, 500);

    // Initial guess data
    std::vector<double> x_init_plot, y_init_plot;
    for (int i = 0; i < horizon + 1; ++i) {
        x_init_plot.push_back(X_init[i](0));
        y_init_plot.push_back(X_init[i](1));
    }

    std::vector<double> cx, cy;
    for (double th = 0.0; th < 2.0 * M_PI; th += 0.01) {
        cx.push_back(center(0) + radius * std::cos(th));
        cy.push_back(center(1) + radius * std::sin(th));
    }

    std::vector<double> cx2, cy2;
    for (double th = 0.0; th < 2.0 * M_PI; th += 0.01) {
        cx2.push_back(center2(0) + radius2 * std::cos(th));
        cy2.push_back(center2(1) + radius2 * std::sin(th));
    }

    // --- Subplot 1: ASDDP ---
    auto ax_asddp = subplot(1, 6, 0);

    auto l_asddp = plot(ax_asddp, x_asddp, y_asddp, "-r");
    l_asddp->display_name("ASDDP Solution");
    l_asddp->line_width(2);

    hold(ax_asddp, true);

    auto cplot_asddp = plot(ax_asddp, cx, cy, "--g");
    cplot_asddp->display_name("Ball Constraint");
    cplot_asddp->line_width(2);

    auto cplot_asddp2 = plot(ax_asddp, cx2, cy2, "--g");
    cplot_asddp2->display_name("Ball Constraint 2");
    cplot_asddp2->line_width(2);

    auto l_asddp_init = plot(ax_asddp, x_init_plot, y_init_plot, "-k");
    l_asddp_init->display_name("Initial Guess");
    l_asddp_init->line_width(2);

    title(ax_asddp, "ASDDP Trajectory");
    xlabel(ax_asddp, "x [m]");
    ylabel(ax_asddp, "y [m]");
    xlim(ax_asddp, {-0.2, 3.2});
    ylim(ax_asddp, {-0.2, 3.2});
    auto leg_asddp = matplot::legend(ax_asddp);
    leg_asddp->location(legend::general_alignment::topleft);
    grid(ax_asddp, true);

    // --- Subplot 2: LogDDP ---
    auto ax_logddp = subplot(1, 6, 1);

    auto l_logddp = plot(ax_logddp, x_logddp, y_logddp, "-b");
    l_logddp->display_name("LogDDP Solution");
    l_logddp->line_width(2);

    hold(ax_logddp, true);

    auto cplot_logddp = plot(ax_logddp, cx, cy, "--g");
    cplot_logddp->display_name("Ball Constraint");
    cplot_logddp->line_width(2);

    auto cplot_logddp2 = plot(ax_logddp, cx2, cy2, "--g");
    cplot_logddp2->display_name("Ball Constraint 2");
    cplot_logddp2->line_width(2);

    auto l_logddp_init = plot(ax_logddp, x_init_plot, y_init_plot, "-k");
    l_logddp_init->display_name("Initial Guess");
    l_logddp_init->line_width(2);

    title(ax_logddp, "LogDDP Trajectory");
    xlabel(ax_logddp, "x [m]");
    ylabel(ax_logddp, "y [m]");
    xlim(ax_logddp, {-0.2, 3.2});
    ylim(ax_logddp, {-0.2, 3.2});
    auto leg_logddp = matplot::legend(ax_logddp);
    leg_logddp->location(legend::general_alignment::topleft);
    grid(ax_logddp, true);

    // --- Subplot 3: IPDDP ---
    auto ax_ipddp = subplot(1, 6, 2);

    auto l_ipddp = plot(ax_ipddp, x_ipddp, y_ipddp, "-g");
    l_ipddp->display_name("IPDDP Solution");
    l_ipddp->line_width(2);

    hold(ax_ipddp, true);

    auto cplot_ipddp = plot(ax_ipddp, cx, cy, "--g");
    cplot_ipddp->display_name("Ball Constraint");
    cplot_ipddp->line_width(2);

    auto cplot_ipddp2 = plot(ax_ipddp, cx2, cy2, "--g");
    cplot_ipddp2->display_name("Ball Constraint 2");
    cplot_ipddp2->line_width(2);

    auto l_ipddp_init = plot(ax_ipddp, x_init_plot, y_init_plot, "-k");
    l_ipddp_init->display_name("Initial Guess");
    l_ipddp_init->line_width(2);

    title(ax_ipddp, "IPDDP Trajectory");
    xlabel(ax_ipddp, "x [m]");
    ylabel(ax_ipddp, "y [m]");
    xlim(ax_ipddp, {-0.2, 3.2});
    ylim(ax_ipddp, {-0.2, 3.2});
    auto leg_ipddp = matplot::legend(ax_ipddp);
    leg_ipddp->location(legend::general_alignment::topleft);
    grid(ax_ipddp, true);

    // --- Subplot 4: MSIPDDP ---
    auto ax_msipddp = subplot(1, 6, 3);

    auto l_msipddp = plot(ax_msipddp, x_msipddp, y_msipddp, "-c");
    l_msipddp->display_name("MSIPDDP Solution");
    l_msipddp->line_width(2);

    hold(ax_msipddp, true);

    auto cplot_msipddp = plot(ax_msipddp, cx, cy, "--g");
    cplot_msipddp->display_name("Ball Constraint");
    cplot_msipddp->line_width(2);

    auto cplot_msipddp2 = plot(ax_msipddp, cx2, cy2, "--g");
    cplot_msipddp2->display_name("Ball Constraint 2");
    cplot_msipddp2->line_width(2);

    auto l_msipddp_init = plot(ax_msipddp, x_init_plot, y_init_plot, "-k");
    l_msipddp_init->display_name("Initial Guess");
    l_msipddp_init->line_width(2);

    title(ax_msipddp, "MSIPDDP Trajectory");
    xlabel(ax_msipddp, "x [m]");
    ylabel(ax_msipddp, "y [m]");
    xlim(ax_msipddp, {-0.2, 3.2});
    ylim(ax_msipddp, {-0.2, 3.2});
    auto leg_msipddp = matplot::legend(ax_msipddp);
    leg_msipddp->location(legend::general_alignment::topleft);
    grid(ax_msipddp, true);

    // --- Subplot 5: IPOPT ---
    auto ax_ipopt = subplot(1, 6, 4);
    auto l_ipopt_sol = plot(ax_ipopt, x_ipopt, y_ipopt, "-m");
    l_ipopt_sol->display_name("IPOPT Solution");
    l_ipopt_sol->line_width(2);

    hold(ax_ipopt, true);

    auto cplot_ipopt = plot(ax_ipopt, cx, cy, "--g");
    cplot_ipopt->display_name("Ball Constraint");
    cplot_ipopt->line_width(2);
    
    auto cplot_ipopt2 = plot(ax_ipopt, cx2, cy2, "--g");
    cplot_ipopt2->display_name("Ball Constraint 2");
    cplot_ipopt2->line_width(2);

    auto l_ipopt_init = plot(ax_ipopt, x_init_plot, y_init_plot, "-k");
    l_ipopt_init->display_name("Initial Guess");
    l_ipopt_init->line_width(2);
    
    title(ax_ipopt, "IPOPT Trajectory");
    xlabel(ax_ipopt, "x [m]");
    ylabel(ax_ipopt, "y [m]");
    xlim(ax_ipopt, {-0.2, 3.2});
    ylim(ax_ipopt, {-0.2, 3.2});
    auto leg_ipopt = matplot::legend(ax_ipopt);
    leg_ipopt->location(legend::general_alignment::topleft);
    grid(ax_ipopt, true);

    // --- Subplot 6: SNOPT ---
    auto ax_snopt = subplot(1, 6, 5);
    auto l_snopt_sol = plot(ax_snopt, x_snopt, y_snopt, "-y");
    l_snopt_sol->display_name("SNOPT Solution");
    l_snopt_sol->line_width(2);

    hold(ax_snopt, true);

    auto cplot_snopt = plot(ax_snopt, cx, cy, "--g");
    cplot_snopt->display_name("Ball Constraint");
    cplot_snopt->line_width(2);
    
    auto cplot_snopt2 = plot(ax_snopt, cx2, cy2, "--g");
    cplot_snopt2->display_name("Ball Constraint 2");
    cplot_snopt2->line_width(2);

    auto l_snopt_init = plot(ax_snopt, x_init_plot, y_init_plot, "-k");
    l_snopt_init->display_name("Initial Guess");
    l_snopt_init->line_width(2);
    
    title(ax_snopt, "SNOPT Trajectory");
    xlabel(ax_snopt, "x [m]");
    ylabel(ax_snopt, "y [m]");
    xlim(ax_snopt, {-0.2, 3.2});
    ylim(ax_snopt, {-0.2, 3.2});
    auto leg_snopt = matplot::legend(ax_snopt);
    leg_snopt->location(legend::general_alignment::topleft);
    grid(ax_snopt, true);

    main_figure->draw();
    main_figure->save(plotDirectory + "/unicycle_baseline_trajectories_comparison.png");
    std::cout << "Saved combined trajectory plot to "
              << (plotDirectory + "/unicycle_baseline_trajectories_comparison.png") << std::endl;

    // --------------------------------------------------------
    // 8. Plot computation times
    // --------------------------------------------------------
    auto time_figure = figure(true);
    time_figure->size(1200, 600);

    std::vector<double> solve_times = {
        solve_time_asddp / 1000000.0,      // Convert to seconds
        solve_time_logddp / 1000000.0,     // Convert to seconds
        solve_time_ipddp / 1000000.0,      // Convert to seconds
        solve_time_msipddp / 1000000.0,    // Convert to seconds
        solve_time_ipopt_numeric,
        solve_time_snopt_numeric
    };

    std::vector<std::string> solver_names = {
        "ASDDP", "LogDDP", "IPDDP", "MSIPDDP", "IPOPT", "SNOPT"
    };

    auto ax_times = time_figure->current_axes();
    auto b = bar(ax_times, solve_times);
    ax_times->xticks(matplot::iota(1.0, static_cast<double>(solver_names.size())));
    ax_times->xticklabels(solver_names);
    title(ax_times, "Solver Computation Time Comparison");
    xlabel(ax_times, "Solver");
    ylabel(ax_times, "Solve Time (seconds)");
    grid(ax_times, true);

    time_figure->draw();
    time_figure->save(plotDirectory + "/unicycle_computation_time_comparison.png");
    std::cout << "Saved computation time plot to "
              << (plotDirectory + "/unicycle_computation_time_comparison.png") << std::endl;

    // --------------------------------------------------------
    // 9. Print summary
    // --------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "       Unicycle Benchmark Summary\n";
    std::cout << "========================================\n";
    std::cout << "Solver    | Final Cost | Solve Time (s)\n";
    std::cout << "----------|------------|---------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "ASDDP     | " << std::setw(10) << cost_asddp 
              << " | " << std::setw(13) << solve_time_asddp / 1000000.0 << "\n";
    std::cout << "LogDDP    | " << std::setw(10) << cost_logddp 
              << " | " << std::setw(13) << solve_time_logddp / 1000000.0 << "\n";
    std::cout << "IPDDP     | " << std::setw(10) << cost_ipddp 
              << " | " << std::setw(13) << solve_time_ipddp / 1000000.0 << "\n";
    std::cout << "MSIPDDP   | " << std::setw(10) << cost_msipddp 
              << " | " << std::setw(13) << solve_time_msipddp / 1000000.0 << "\n";
    std::cout << "IPOPT     | " << std::setw(10) << cost_ipopt 
              << " | " << std::setw(13) << solve_time_ipopt_numeric << "\n";
    std::cout << "SNOPT     | " << std::setw(10) << cost_snopt 
              << " | " << std::setw(13) << solve_time_snopt_numeric << "\n";
    std::cout << "========================================\n\n";

    return 0;
}
