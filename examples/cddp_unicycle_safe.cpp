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
    // -------------------------------------------------------
    // 1. Problem Setup
    // -------------------------------------------------------
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle dynamical system instance
    std::unique_ptr<cddp::DynamicalSystem> dyn_system =
        std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0,  0.0,  0.0,
           0.0, 100.0,  0.0,
           0.0,   0.0, 100.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI / 2.0;

    // Empty reference states (if needed)
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    );

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    // CDDP options
    cddp::CDDPOptions options;
    options.max_iterations = 100;
    options.verbose = true;
    options.debug = false;
    options.cost_tolerance = 1e-5;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "control";
    options.regularization_control = 1e-5;

    // Define control box constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -2.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 2.0, M_PI;

    // -------------------------------------------------------
    // 2. Solve the CDDP Problem (No Ball Constraint)
    // -------------------------------------------------------
    cddp::CDDP solver_baseline(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );

    // Set system, objective, constraints, and initial guess
    solver_baseline.setDynamicalSystem(std::make_unique<cddp::Unicycle>(timestep, integration_type));
    solver_baseline.setObjective(std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    ));
    solver_baseline.addConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Simple initial guess: states = initial_state, controls = zeros
    std::vector<Eigen::VectorXd> X_baseline(horizon + 1, initial_state);
    std::vector<Eigen::VectorXd> U_baseline(horizon, Eigen::VectorXd::Zero(control_dim));
    solver_baseline.setInitialTrajectory(X_baseline, U_baseline);

    // Solve
    cddp::CDDPSolution solution_baseline = solver_baseline.solve("ASCDDP");
    auto X_baseline_sol = solution_baseline.state_sequence; // size horizon + 1

    // -------------------------------------------------------
    // 3. Solve with BallConstraint
    // -------------------------------------------------------
    cddp::CDDP solver_ball(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    solver_ball.setDynamicalSystem(std::make_unique<cddp::Unicycle>(timestep, integration_type));
    solver_ball.setObjective(std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    ));
    solver_ball.addConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Add the BallConstraint
    double radius = 0.4;
    Eigen::Vector2d center(1.0, 1.0);
    solver_ball.addConstraint("BallConstraint",
        std::make_unique<cddp::BallConstraint>(radius, center));

    // Initial trajectory for the ball-constrained solver
    std::vector<Eigen::VectorXd> X_ball(horizon + 1, initial_state);
    std::vector<Eigen::VectorXd> U_ball(horizon, Eigen::VectorXd::Zero(control_dim));
    solver_ball.setInitialTrajectory(X_ball, U_ball);

    // Solve
    cddp::CDDPSolution solution_ball = solver_ball.solve("ASCDDP");
    auto X_ball_sol = solution_ball.state_sequence;  // horizon+1

    // -------------------------------------------------------
    // 4. Prepare Data for Plotting
    // -------------------------------------------------------
    std::vector<double> x_baseline, y_baseline;
    std::vector<double> x_ball_vec, y_ball_vec;
    x_baseline.reserve(X_baseline_sol.size());
    y_baseline.reserve(X_baseline_sol.size());
    x_ball_vec.reserve(X_ball_sol.size());
    y_ball_vec.reserve(X_ball_sol.size());

    // Convert baseline solution states
    for (const auto &state : X_baseline_sol) {
        x_baseline.push_back(state(0));
        y_baseline.push_back(state(1));
    }
    // Convert ball-constrained solution states
    for (const auto &state : X_ball_sol) {
        x_ball_vec.push_back(state(0));
        y_ball_vec.push_back(state(1));
    }

    // Prepare circle points for the ball constraint
    std::vector<double> x_circle, y_circle;
    x_circle.reserve(630); // ~ (2*pi / 0.01)
    y_circle.reserve(630);

    for (double t = 0.0; t < 2.0 * M_PI; t += 0.01) {
        x_circle.push_back(center(0) + radius * std::cos(t));
        y_circle.push_back(center(1) + radius * std::sin(t));
    }

    // -------------------------------------------------------
    // 5. Plot with matplot
    // -------------------------------------------------------
    // Create the figure window
    auto f1 = figure(true); // 'true' => make it visible (depending on your system)
    f1->size(800, 600);

    // Create an axes object
    auto ax = f1->current_axes();

    // Plot the baseline solution (blue)
    auto h_baseline = plot(ax, x_baseline, y_baseline);
    h_baseline->line_width(2);
    h_baseline->display_name("Without Ball Constraint");
    h_baseline->color("blue");

    // Plot the ball-constrained solution (red)
    hold(ax, true);
    auto h_ball = plot(ax, x_ball_vec, y_ball_vec);
    h_ball->line_width(2);
    h_ball->display_name("With Ball Constraint");
    h_ball->color("red");

    // Plot the constraint circle (green, dashed)
    auto h_circle = plot(ax, x_circle, y_circle);
    h_circle->line_style("--");
    h_circle->line_width(2);
    h_circle->color("green");
    h_circle->display_name("Ball Constraint Region");

    // Add title, labels, legend
    title(ax, "Trajectory Comparison: With vs. Without BallConstraint");
    xlabel(ax, "x");
    ylabel(ax, "y");
    xlim(ax, {-0.5, 2.5});
    ylim(ax, {-0.5, 2.5});
    legend(ax);

    // Create directory for saving (if not existing)
    std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Save figure
    f1->draw();
    f1->save(plotDirectory + "/trajectory_comparison_matplot.png");
    std::cout << "Trajectory comparison saved to "
              << plotDirectory + "/trajectory_comparison_matplot.png" << std::endl;

    return 0;
}
