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
#include <memory>
#include <Eigen/Dense>

#include "matplotlibcpp.hpp"
#include "cddp.hpp"
#include "sqp_core/sqp_core.hpp"

namespace plt = matplotlibcpp;

int main() {
    ////////// Problem Setup //////////
    const int state_dim   = 3;   // [x, y, theta]
    const int control_dim = 2;   // [v, omega]
    const int horizon     = 100; // Number of control intervals
    const double timestep = 0.03; // Time step
    const std::string integration_type = "euler";

    // Define initial and goal states.
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI / 2.0;

    // Create the unicycle dynamical system.
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Unicycle>(timestep, integration_type);

    // Define cost weighting matrices.
    // Running state cost (zero in this example).
    Eigen::MatrixXd Q = 5 * Eigen::MatrixXd::Zero(state_dim, state_dim);
    // Control cost.
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    // Terminal state cost.
    Eigen::MatrixXd Qf(state_dim, state_dim);
    Qf << 1200.0,   0.0,    0.0,
            0.0, 1200.0,    0.0,
            0.0,    0.0,  700.0;

    // Create a (empty) reference trajectory.
    std::vector<Eigen::VectorXd> empty_reference_states;

    // Create the quadratic objective.
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep
    );

    // Set up SQP (SCP) options.
    cddp::SCPOptions options;
    options.max_iterations     = 5;
    options.min_iterations     = 3;
    options.ftol               = 1e-6;
    options.xtol               = 1e-6;
    options.gtol               = 1e-6;
    options.merit_penalty      = 100.0;
    options.verbose            = true;
    options.trust_region_radius= 100.0;
    options.ipopt_print_level  = 5;

    // Create the SQP solver.
    cddp::SCPSolver sqp_solver(initial_state, goal_state, horizon, timestep);
    sqp_solver.setDynamicalSystem(std::move(system));
    sqp_solver.setObjective(std::move(objective));
    sqp_solver.setOptions(options);

    // Define control bounds (v ∈ [-1, 1] and ω ∈ [-π, π]).
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound <<  1.0,  M_PI;

    // Add the control box constraint.
    sqp_solver.addConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound)
    );

    // Set an initial trajectory.
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon,   Eigen::VectorXd::Zero(control_dim));

    X[0] = initial_state;
    for (int t = 1; t <= horizon; ++t) {
        X[t] = initial_state + (goal_state - initial_state) * (static_cast<double>(t) / horizon);
    }
    sqp_solver.setInitialTrajectory(X, U);

    ////////// Solve the Problem //////////
    auto start_time = std::chrono::high_resolution_clock::now();
    cddp::SCPResult solution = sqp_solver.solve();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "SQP solver elapsed time: " << elapsed.count() << " s" << std::endl;

    ////////// Extract and Display the Results //////////
    // Extract the state trajectory for plotting.
    std::vector<double> x_hist, y_hist;
    for (const auto& state : solution.X) {
        x_hist.push_back(state(0));
        y_hist.push_back(state(1));
    }

    // Print summary results.
    std::cout << "Initial state: " << solution.X.front().transpose() << std::endl;
    std::cout << "Final state:   " << solution.X.back().transpose() << std::endl;
    std::cout << "Goal state:    " << goal_state.transpose() << std::endl;
    std::cout << "Total iterations: " << solution.iterations << std::endl;
    std::cout << "Solve time (from SCPResult): " << solution.solve_time << " s" << std::endl;

    ////////// Plot the Trajectory //////////
    plt::figure();
    plt::plot(x_hist, y_hist, "b-");
    plt::scatter(std::vector<double>{initial_state(0)}, std::vector<double>{initial_state(1)},
                 100, {{"color", "green"}, {"label", "Start"}});
    plt::scatter(std::vector<double>{goal_state(0)}, std::vector<double>{goal_state(1)},
                 100, {{"color", "red"}, {"label", "Goal"}});
    plt::xlabel("x");
    plt::ylabel("y");
    plt::title("Unicycle Trajectory (SQP)");
    plt::legend();
    plt::grid(true);
    plt::show();

    return 0;
}
