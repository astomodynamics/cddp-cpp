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

#include "cddp.hpp"

// Include matplot
#include "matplot/matplot.h"

namespace fs = std::filesystem;
using namespace matplot;

int main() {
    // -------------------------------------------------
    // 1. Problem Setup
    // -------------------------------------------------
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 300;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle dynamical system
    auto dyn_system = std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0,  0.0,  0.0,
           0.0, 100.0,  0.0,
           0.0,   0.0, 100.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 3.0, 3.0, M_PI / 2.0;

    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    // CDDP options
    cddp::CDDPOptions options;
    options.max_iterations = 1000;
    options.verbose = true;
    options.debug = false;
    options.use_parallel = false;
    options.num_threads = 1;
    options.cost_tolerance = 1e-5;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "both";
    options.regularization_control = 1e-2;
    options.regularization_state = 1e-3;
    options.barrier_coeff = 1e-1;

    // Control constraint
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.1, M_PI;

    // -------------------------------------------------
    // 2. Configure and Solve IPDDP with Two BallConstraints
    // -------------------------------------------------
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );

    // Set dynamical system & objective explicitly
    cddp_solver.setDynamicalSystem(std::make_unique<cddp::Unicycle>(timestep, integration_type));
    cddp_solver.setObjective(std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    ));

    // Add constraints
    cddp_solver.addConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // First ball constraint
    double radius1 = 0.4;
    Eigen::Vector2d center1(1.0, 1.0);
    cddp_solver.addConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius1, center1));

    // Second ball constraint
    double radius2 = 0.4;
    Eigen::Vector2d center2(1.5, 2.5);
    cddp_solver.addConstraint("BallConstraint2",
        std::make_unique<cddp::BallConstraint>(radius2, center2));

    // Initial trajectory guess
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon + 1; ++i) {
        X_sol[i] = initial_state;
    }
    cddp_solver.setInitialTrajectory(X_sol, U_sol);

    // Solve
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");
    X_sol = solution.state_sequence; 
    U_sol = solution.control_sequence;

    // -------------------------------------------------
    // 3. Prepare Data for Plotting
    // -------------------------------------------------
    std::vector<double> x_sol_plot, y_sol_plot;
    x_sol_plot.reserve(X_sol.size());
    y_sol_plot.reserve(X_sol.size());

    for (const auto &state : X_sol) {
        x_sol_plot.push_back(state(0));
        y_sol_plot.push_back(state(1));
    }

    // Create circle points for the two ball constraints
    std::vector<double> x_ball1, y_ball1;
    std::vector<double> x_ball2, y_ball2;
    for (double t = 0.0; t < 2 * M_PI; t += 0.01) {
        x_ball1.push_back(center1(0) + radius1 * std::cos(t));
        y_ball1.push_back(center1(1) + radius1 * std::sin(t));
        x_ball2.push_back(center2(0) + radius2 * std::cos(t));
        y_ball2.push_back(center2(1) + radius2 * std::sin(t));
    }

    // -------------------------------------------------
    // 4. Plot with matplot
    // -------------------------------------------------
    auto f1 = figure(true);
    f1->size(800, 600);

    auto ax = f1->current_axes();
    auto traj_line = plot(ax, x_sol_plot, y_sol_plot);
    traj_line->color("blue");
    traj_line->line_width(2);
    traj_line->display_name("IPDDP");

    // We want multiple lines on same plot:
    hold(ax, true);

    // Ball constraints (two circles)
    auto ball1 = plot(ax, x_ball1, y_ball1);
    ball1->line_style("--");
    ball1->line_width(2);
    ball1->color("green");
    ball1->display_name("Ball Constraint 1");

    auto ball2 = plot(ax, x_ball2, y_ball2);
    ball2->line_style("--");
    ball2->line_width(2);
    ball2->color("green");
    ball2->display_name("Ball Constraint 2");

    // Turn on grid, label, etc.
    grid(ax, true);
    xlabel(ax, "x");
    ylabel(ax, "y");
    xlim(ax, {0.0, 4.0});
    ylim(ax, {0.0, 4.0});
    title(ax, "IPDDP Safe Trajectory with Two Ball Constraints");
    legend(ax);

    // -------------------------------------------------
    // 5. Save Plot
    // -------------------------------------------------
    std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    f1->draw();
    f1->save(plotDirectory + "/trajectory_comparison_ipddp_v2_matplot.png");
    std::cout << "Trajectory comparison saved to "
              << (plotDirectory + "/trajectory_comparison_ipddp_v2_matplot.png")
              << std::endl;

    return 0;
}
