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

int main() {
    constexpr double kPi = 3.14159265358979323846;
    constexpr int state_dim = 6;
    constexpr int control_dim = 3;
    constexpr int horizon = 160;
    constexpr double timestep = 0.01;

    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, -kPi / 2.0, kPi, 0.0, 0.0, 0.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << kPi, -kPi / 6.0, -kPi / 3.0, 0.0, 0.0, 0.0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q.diagonal().segment(0, 3) = Eigen::Vector3d::Ones();
    Q.diagonal().segment(3, 3) = 0.1 * Eigen::Vector3d::Ones();
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 100.0 * Q;

    cddp::CDDPOptions options;
    options.max_iterations = 80;
    options.line_search.max_iterations = 20;

    cddp::CDDP solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Manipulator>(timestep, "rk4"),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, std::vector<Eigen::VectorXd>{}, timestep),
        options);

    const Eigen::VectorXd lower =
        -50.0 * Eigen::VectorXd::Ones(control_dim);
    const Eigen::VectorXd upper =
        50.0 * Eigen::VectorXd::Ones(control_dim);
    solver.addPathConstraint("ControlConstraint",
                             std::make_unique<cddp::ControlConstraint>(lower, upper));

    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i <= horizon; ++i) {
        const double alpha = static_cast<double>(i) / static_cast<double>(horizon);
        X[i] = (1.0 - alpha) * initial_state + alpha * goal_state;
    }
    solver.setInitialTrajectory(X, U);

    const cddp::CDDPSolution solution = solver.solve(cddp::SolverType::CLDDP);
    if (solution.state_trajectory.empty()) {
        std::cerr << "Manipulator example failed: empty trajectory" << std::endl;
        return 1;
    }

    const Eigen::VectorXd final_error = solution.state_trajectory.back() - goal_state;
    std::cout << "Manipulator example completed with status: "
              << solution.status_message << '\n'
              << "Final objective: " << solution.final_objective << '\n'
              << "Final state error norm: " << final_error.norm() << std::endl;
    return 0;
}
