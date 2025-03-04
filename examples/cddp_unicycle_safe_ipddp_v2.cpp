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
    int horizon = 300;
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
    goal_state << 3.0, 3.0, M_PI/2.0;

    // Create an empty vector for reference states
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Define initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0;

    // Set up common CDDP options
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

    // Define control box constraint bounds
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.1, M_PI;

    // --------------------------
    cddp::CDDP cddp_solver(
        initial_state, 
        goal_state, 
        horizon, 
        timestep, 
        std::make_unique<cddp::Unicycle>(timestep, integration_type),
        std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep),
        options
    );
    cddp_solver.setDynamicalSystem(std::make_unique<cddp::Unicycle>(timestep, integration_type));
    cddp_solver.setObjective(std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep));
    cddp_solver.addConstraint("ControlConstraint", std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    double radius1 = 0.4;
    Eigen::Vector2d center1(1.0, 1.0);
    cddp_solver.addConstraint("BallConstraint", std::make_unique<cddp::BallConstraint>(radius1, center1));
    double radius2 = 0.4;
    Eigen::Vector2d center2(1.5, 2.5);
    cddp_solver.addConstraint("BallConstraint2", std::make_unique<cddp::BallConstraint>(radius2, center2));

    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon + 1; ++i) {
        X_sol[i] = initial_state;
    }
    cddp_solver.setInitialTrajectory(X_sol, U_sol);

    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");
    X_sol = solution.state_sequence;
    U_sol = solution.control_sequence;

    // --------------------------
    // Plot the trajectories for comparison
    std::vector<double> x_sol_plot, y_sol_plot;

    for (const auto &state : X_sol) {
        x_sol_plot.push_back(state(0));
        y_sol_plot.push_back(state(1));
    }

    plt::figure();
    plt::plot(x_sol_plot, y_sol_plot, {{"color", "b"}, {"linestyle", "-"}, {"label", "IPDDP"}});
    
    // Also plot the ball for reference
    std::vector<double> t_ball, x_ball_circle, y_ball_circle;
    std::vector<double> t_ball2, x_ball_circle2, y_ball_circle2;
    for (double t = 0.0; t < 2 * M_PI; t += 0.01) {
        t_ball.push_back(t);
        x_ball_circle.push_back(center1(0) + radius1 * cos(t));
        y_ball_circle.push_back(center1(1) + radius1 * sin(t));
        x_ball_circle2.push_back(center2(0) + radius2 * cos(t));
        y_ball_circle2.push_back(center2(1) + radius2 * sin(t));      
    }
    plt::plot(x_ball_circle, y_ball_circle, {{"color", "g"}, {"linestyle", "--"}, {"label", "Ball Constraint"}});
    plt::plot(x_ball_circle2, y_ball_circle2, {{"color", "g"}, {"linestyle", "--"}});
    plt::grid(true);
    plt::xlabel("x");
    plt::ylabel("y");
    plt::title("IPDDP safe trajectory");
    plt::legend();
    
    // Save the comparison plot
    std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }
    plt::save(plotDirectory + "/trajectory_comparison_ipddp_v2.png");
    std::cout << "Trajectory comparison saved to " << plotDirectory + "/trajectory_comparison_ipddp_v2.png" << std::endl;

    return 0;
}
