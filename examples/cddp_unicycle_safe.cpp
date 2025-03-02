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

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main() {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a unicycle dynamical system instance
    std::unique_ptr<cddp::DynamicalSystem> dyn_system = std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.05 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0, 0.0, 0.0,
          0.0, 100.0, 0.0,
          0.0, 0.0, 100.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Create an empty vector for reference states
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Define initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;

    // Set up common CDDP options
    cddp::CDDPOptions options;
    options.max_iterations = 20;
    options.verbose = false;
    options.debug = false;
    options.cost_tolerance = 1e-5;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "none";

    // Define control box constraint bounds
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -2.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 2.0, M_PI;

    // --------------------------
    // Solve the CDDP problem without the ball constraint
    cddp::CDDP solver_baseline(
        initial_state, 
        goal_state, 
        horizon, 
        timestep, 
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    solver_baseline.setDynamicalSystem(std::make_unique<cddp::Unicycle>(timestep, integration_type));
    solver_baseline.setObjective(std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep));
    solver_baseline.addConstraint("ControlBoxConstraint", std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Set an initial trajectory (all states equal to the initial state)
    std::vector<Eigen::VectorXd> X_baseline(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_baseline(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon + 1; ++i) {
        X_baseline[i] = initial_state;
    }
    solver_baseline.setInitialTrajectory(X_baseline, U_baseline);

    cddp::CDDPSolution solution_baseline = solver_baseline.solve("ASCDDP");
    auto X_baseline_sol = solution_baseline.state_sequence;

    // --------------------------
    // 2. Solve with BallConstraint
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
    solver_ball.setObjective(std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep));
    solver_ball.addConstraint("ControlBoxConstraint", std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    double radius = 0.4;
    Eigen::Vector2d center(1.0, 1.0);
    solver_ball.addConstraint("BallConstraint", std::make_unique<cddp::BallConstraint>(radius, center));

    // Set an initial trajectory for the ball-constrained solver
    std::vector<Eigen::VectorXd> X_ball(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_ball(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon + 1; ++i) {
        X_ball[i] = initial_state;
    }
    solver_ball.setInitialTrajectory(X_ball, U_ball);

    cddp::CDDPSolution solution_ball = solver_ball.solve("ASCDDP");
    auto X_ball_sol = solution_ball.state_sequence;

    // --------------------------
    // Plot the trajectories
    std::vector<double> x_baseline, y_baseline;
    std::vector<double> x_ball, y_ball;
    for (const auto &state : X_baseline_sol) {
        x_baseline.push_back(state(0));
        y_baseline.push_back(state(1));
    }
    for (const auto &state : X_ball_sol) {
        x_ball.push_back(state(0));
        y_ball.push_back(state(1));
    }

    // Plot trajectories on the same figure
    plt::figure();
    plt::plot(x_baseline, y_baseline, {{"color", "b"}, {"linestyle", "-"}, {"label", "Without Ball Constraint"}});
    plt::plot(x_ball, y_ball, {{"color", "r"}, {"linestyle", "-"}, {"label", "With Ball Constraint"}});

    
    
    // Also plot the ball for reference
    std::vector<double> t_ball, x_ball_circle, y_ball_circle;
    for (double t = 0.0; t < 2 * M_PI; t += 0.01) {
        t_ball.push_back(t);
        x_ball_circle.push_back(center(0) + radius * cos(t));
        y_ball_circle.push_back(center(1) + radius * sin(t));
    }plt::plot(x_ball_circle, y_ball_circle, {{"color", "g"}, {"linestyle", "--"}, {"label", "Ball Constraint"}});

    plt::xlabel("x");
    plt::ylabel("y");
    plt::title("Trajectory Comparison: With vs. Without BallConstraint");
    plt::legend();
    
    // Save the comparison plot
    std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }
    plt::save(plotDirectory + "/trajectory_comparison.png");
    std::cout << "Trajectory comparison saved to " << plotDirectory + "/trajectory_comparison.png" << std::endl;

    return 0;
}
