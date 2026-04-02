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

#include "cddp.hpp"
#include "cddp_example_utils.hpp"

int main() {
    constexpr double kPi = 3.14159265358979323846;
    constexpr int state_dim = 4;
    constexpr int control_dim = 1;
    constexpr int horizon = 100;
    constexpr double timestep = 0.05;

    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, 0.0, 0.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, kPi, 0.0, 0.0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

    cddp::CDDPOptions options;
    options.max_iterations = 80;
    options.tolerance = 1e-6;
    options.acceptable_tolerance = 1e-5;
    options.regularization.initial_value = 1e-5;

    cddp::CDDP solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::CartPole>(
            timestep, "rk4", 1.0, 0.2, 0.5, 9.81, 0.0),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, std::vector<Eigen::VectorXd>{}, timestep),
        options);

    Eigen::VectorXd lower(control_dim);
    lower << -5.0;
    Eigen::VectorXd upper(control_dim);
    upper << 5.0;
    solver.addPathConstraint("ControlConstraint",
                             std::make_unique<cddp::ControlConstraint>(lower, upper));

    auto [X, U] =
        cddp::example::makeInitialTrajectory(initial_state, horizon, control_dim);
    solver.setInitialTrajectory(X, U);

    const cddp::CDDPSolution solution = solver.solve(cddp::SolverType::IPDDP);
    if (solution.state_trajectory.empty()) {
        std::cerr << "Cart-pole example failed: empty trajectory" << std::endl;
        return 1;
    }

    const Eigen::VectorXd final_error = solution.state_trajectory.back() - goal_state;
    std::cout << "Cart-pole example completed with status: "
              << solution.status_message << '\n'
              << "Final objective: " << solution.final_objective << '\n'
              << "Final state error norm: " << final_error.norm() << std::endl;
    return 0;
}
