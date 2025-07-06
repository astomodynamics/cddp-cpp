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

    // Create a unicycle instance
    auto dyn_system = std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Quadratic cost
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0,  0.0,   0.0,
           0.0, 100.0,   0.0,
           0.0,   0.0, 100.0;

    // Goal state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI / 2.0;

    // Empty reference states
    std::vector<Eigen::VectorXd> empty_ref;

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    // Options for the baseline #2 solver (10 iterations)
    cddp::CDDPOptions options_10;
    options_10.max_iterations = 10;
    options_10.verbose = true;
    options_10.debug = false;
    options_10.enable_parallel = false;
    options_10.num_threads = 1;
    options_10.tolerance = 1e-5;
    options_10.acceptable_tolerance = 1e-4;
    options_10.regularization.initial_value = 1e-2;
    options_10.ipddp.barrier.mu_initial = 1e-1;

    cddp::CDDPOptions options_ipddp;
    options_ipddp.max_iterations = 1000;
    options_ipddp.verbose = true;
    options_ipddp.debug = false;
    options_ipddp.enable_parallel = false;
    options_ipddp.num_threads = 1;
    options_ipddp.tolerance = 1e-5;
    options_ipddp.acceptable_tolerance = 1e-4;
    options_ipddp.regularization.initial_value = 1e-4;
    options_ipddp.ipddp.barrier.mu_initial = 1e-1;

    cddp::CDDPOptions options_asddp;
    options_asddp.max_iterations = 100;
    options_asddp.verbose = true;
    options_asddp.debug = false;
    options_asddp.enable_parallel = false;
    options_asddp.num_threads = 1;
    options_asddp.tolerance = 1e-5;
    options_asddp.acceptable_tolerance = 1e-4;
    options_asddp.regularization.initial_value = 1e-5;

    // Constraint parameters
    // (Used only by baseline #2 and the subsequent 4 solutions,
    //  but not by the unconstrained solver.)
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 2.0, M_PI;
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -2.0, -M_PI;
    double radius = 0.4;
    Eigen::Vector2d center(1.0, 1.0);

    // Create a directory for saving plots
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // --------------------------------------------------------
    // 2. Baseline #1: Unconstrained (no ball, no control constraint)
    // --------------------------------------------------------
    cddp::CDDP solver_unconstrained(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
        options_10  // We can also reuse the same options, or set different ones
    );
    // We do NOT add any constraint here
    // Simple initial guess
    std::vector<Eigen::VectorXd> X_unconstrained_init(horizon + 1, initial_state);
    std::vector<Eigen::VectorXd> U_unconstrained_init(horizon, Eigen::VectorXd::Zero(control_dim));
    solver_unconstrained.setInitialTrajectory(X_unconstrained_init, U_unconstrained_init);

    // Solve for baseline #1
    cddp::CDDPSolution sol_unconstrained = solver_unconstrained.solve(cddp::SolverType::IPDDP);
    auto X_unconstrained_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_unconstrained.at("state_trajectory"));
    auto U_unconstrained_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_unconstrained.at("control_trajectory"));

    // --------------------------------------------------------
    // 3. Baseline #2: IPDDP with constraints (10 iterations)
    //    => "BallConstraint" + "ControlConstraint"
    // --------------------------------------------------------
    cddp::CDDP solver_ipddp_10(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
        options_10
    );
    // Add constraints
    solver_ipddp_10.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_ipddp_10.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));

    // Simple initial guess
    std::vector<Eigen::VectorXd> X_ipddp10_init(horizon + 1, initial_state);
    std::vector<Eigen::VectorXd> U_ipddp10_init(horizon, Eigen::VectorXd::Zero(control_dim));
    solver_ipddp_10.setInitialTrajectory(X_unconstrained_sol, U_unconstrained_sol);

    // Solve for baseline #2
    cddp::CDDPSolution sol_ipddp_10 = solver_ipddp_10.solve(cddp::SolverType::IPDDP);
    auto X_ipddp10_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp_10.at("state_trajectory"));
    auto U_ipddp10_sol = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp_10.at("control_trajectory"));

    // --------------------------------------------------------
    // 4. IPDDP and ASDDP (with constraints) using
    //    each baseline solution as the initial guess
    // --------------------------------------------------------
    //
    //  We'll create 4 new solvers:
    //    A) IPDDP from unconstrained
    //    B) ASDDP from unconstrained
    //    C) IPDDP from ipddp_10 baseline
    //    D) ASDDP from ipddp_10 baseline
    //
    // --------------------------------------------------------
    auto makeConstrainedSolver = [&](const std::string &method_name, bool use_baseline = false) {
        if (use_baseline) {
            std::cout << "Using baseline solution as initial guess." << std::endl;
        } else {
            std::cout << "Using unconstrained solution as initial guess." << std::endl;
        }

        if (use_baseline) {
            return cddp::CDDP(
                initial_state,
                goal_state,
                horizon,
                timestep,
                std::make_unique<cddp::Unicycle>(timestep, integration_type),
                std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
                options_10
            );
        } else {
            if (method_name == "IPDDP") {
                std::cout << "Using IPDDP method." << std::endl;
                return cddp::CDDP(
                    initial_state,
                    goal_state,
                    horizon,
                    timestep,
                    std::make_unique<cddp::Unicycle>(timestep, integration_type),
                    std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
                    options_ipddp
                );
            } else if (method_name == "ASDDP") {
                std::cout << "Using ASDDP method." << std::endl;
                return cddp::CDDP(
                    initial_state,
                    goal_state,
                    horizon,
                    timestep,
                    std::make_unique<cddp::Unicycle>(timestep, integration_type),
                    std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_ref, timestep),
                    options_asddp
                );
            }
            
        }
    };

    // (A) IPDDP from unconstrained
    cddp::CDDP solver_ipddp_from_unconstrained = makeConstrainedSolver("IPDDP");
    solver_ipddp_from_unconstrained.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_ipddp_from_unconstrained.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_ipddp_from_unconstrained.setInitialTrajectory(
        X_unconstrained_sol, U_unconstrained_sol
    );
    auto sol_ipddp_from_uncon = solver_ipddp_from_unconstrained.solve(cddp::SolverType::IPDDP);
    auto X_ipddp_from_uncon = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp_from_uncon.at("state_trajectory"));

    // (B) ASDDP from unconstrained
    cddp::CDDP solver_asddp_from_unconstrained = makeConstrainedSolver("ASDDP");
    solver_asddp_from_unconstrained.addPathConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound,
            control_upper_bound));
    solver_asddp_from_unconstrained.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_asddp_from_unconstrained.setInitialTrajectory(
        X_unconstrained_sol, U_unconstrained_sol
    );
    auto sol_asddp_from_uncon = solver_asddp_from_unconstrained.solve(cddp::SolverType::ASDDP);
    auto X_asddp_from_uncon = std::any_cast<std::vector<Eigen::VectorXd>>(sol_asddp_from_uncon.at("state_trajectory"));

    // (C) IPDDP from ipddp_10 baseline
    cddp::CDDP solver_ipddp_from_ipddp10 = makeConstrainedSolver("IPDDP");
    solver_ipddp_from_ipddp10.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    solver_ipddp_from_ipddp10.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_ipddp_from_ipddp10.setInitialTrajectory(
        X_ipddp10_sol, U_ipddp10_sol
    );
    auto sol_ipddp_from_ipddp10 = solver_ipddp_from_ipddp10.solve(cddp::SolverType::IPDDP);
    auto X_ipddp_from_ipddp10 = std::any_cast<std::vector<Eigen::VectorXd>>(sol_ipddp_from_ipddp10.at("state_trajectory"));

    // (D) ASDDP from ipddp_10 baseline
    cddp::CDDP solver_asddp_from_ipddp10 = makeConstrainedSolver("ASDDP");
    solver_asddp_from_ipddp10.addPathConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound,
            control_upper_bound));
    solver_asddp_from_ipddp10.addPathConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));
    solver_asddp_from_ipddp10.setInitialTrajectory(
        X_ipddp10_sol, U_ipddp10_sol
    );
    auto sol_asddp_from_ipddp10 = solver_asddp_from_ipddp10.solve(cddp::SolverType::ASDDP);
    auto X_asddp_from_ipddp10 = std::any_cast<std::vector<Eigen::VectorXd>>(sol_asddp_from_ipddp10.at("state_trajectory"));

    // --------------------------------------------------------
    // 5. Convert all 6 solutions to (x,y) data for a single plot
    // --------------------------------------------------------
    auto stateSeqToXY = [&](const std::vector<Eigen::VectorXd> &X_seq) {
        std::vector<double> xv, yv;
        xv.reserve(X_seq.size());
        yv.reserve(X_seq.size());
        for (auto &st : X_seq) {
            xv.push_back(st(0));
            yv.push_back(st(1));
        }
        return std::make_pair(xv, yv);
    };

    // Baseline #1: unconstrained
    auto [x_uncon, y_uncon] = stateSeqToXY(X_unconstrained_sol);
    // Baseline #2: ipddp_10
    auto [x_ipddp10, y_ipddp10] = stateSeqToXY(X_ipddp10_sol);

    // IPDDP & ASDDP from unconstrained
    auto [x_ipddp_from_un, y_ipddp_from_un] = stateSeqToXY(X_ipddp_from_uncon);
    auto [x_asddp_from_un, y_asddp_from_un] = stateSeqToXY(X_asddp_from_uncon);

    // IPDDP & ASDDP from ipddp_10
    auto [x_ipddp_from_ip10, y_ipddp_from_ip10] = stateSeqToXY(X_ipddp_from_ipddp10);
    auto [x_asddp_from_ip10, y_asddp_from_ip10] = stateSeqToXY(X_asddp_from_ipddp10);

    // --------------------------------------------------------
    // 6. Plot all 6 lines in one figure
    // --------------------------------------------------------
    auto f1 = figure(true);
    f1->size(1000, 800);
    auto ax = f1->current_axes();

    // Plot baseline #1: unconstrained
    auto l0 = plot(ax, x_uncon, y_uncon, "-k");
    l0->display_name("Baseline Unconstrained");
    l0->line_width(2);

    hold(ax, true);

    // Plot baseline #2: ipddp_10
    auto l1 = plot(ax, x_ipddp10, y_ipddp10, "-b");
    l1->display_name("Baseline IPDDP(10)");
    l1->line_width(2);

    // Plot IPDDP from unconstrained
    auto l2 = plot(ax, x_ipddp_from_un, y_ipddp_from_un, "-r");
    l2->display_name("IPDDP from Unconstrained");
    l2->line_width(2);

    // Plot ASDDP from unconstrained
    auto l3 = plot(ax, x_asddp_from_un, y_asddp_from_un, "--r");
    l3->display_name("ASDDP from Unconstrained");
    l3->line_width(2);

    // Plot IPDDP from ipddp_10
    auto l4 = plot(ax, x_ipddp_from_ip10, y_ipddp_from_ip10, "-m");
    l4->display_name("IPDDP from IPDDP(10)");
    l4->line_width(2);

    // Plot ASDDP from ipddp_10
    auto l5 = plot(ax, x_asddp_from_ip10, y_asddp_from_ip10, "--m");
    l5->display_name("ASDDP from IPDDP(10)");
    l5->line_width(2);

    // Plot the Ball constraint circle
    // (just for reference — it won't show up on the unconstrained solver)
    std::vector<double> cx, cy;
    for (double th = 0.0; th < 2.0 * M_PI; th += 0.01) {
        cx.push_back(center(0) + radius * std::cos(th));
        cy.push_back(center(1) + radius * std::sin(th));
    }
    auto cplot = plot(ax, cx, cy, "--g");
    cplot->display_name("Ball Constraint");
    cplot->line_width(2);

    title(ax, "Comparison of 6 Trajectories");
    xlabel(ax, "x [m]");
    ylabel(ax, "y [m]");
    xlim(ax, {-0.2, 2.5});
    ylim(ax, {-0.2, 2.5});
    auto l = matplot::legend(ax);
    l->location(legend::general_alignment::topleft);
    grid(ax, true);

    f1->draw();
    f1->save(plotDirectory + "/unicycle_six_trajectories_comparison.png");
    std::cout << "Saved figure with 6 trajectories to "
              << (plotDirectory + "/unicycle_six_trajectories_comparison.png") << std::endl;

    return 0;
}
