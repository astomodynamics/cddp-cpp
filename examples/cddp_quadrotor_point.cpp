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
    constexpr int state_dim = 13;
    constexpr int control_dim = 4;
    constexpr int horizon = 120;
    constexpr double timestep = 0.02;

    const double mass = 1.0;
    const double arm_length = 0.2;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();
    inertia(0, 0) = 0.01;
    inertia(1, 1) = 0.01;
    inertia(2, 2) = 0.02;

    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state(3) = 1.0;

    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state(0) = 3.0;
    goal_state(2) = 2.0;
    goal_state(3) = 1.0;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(4, 4) = 0.1;
    Q(5, 5) = 0.1;
    Q(6, 6) = 0.1;

    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 500.0;
    Qf(1, 1) = 500.0;
    Qf(2, 2) = 500.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;
    Qf(7, 7) = 10.0;
    Qf(8, 8) = 10.0;
    Qf(9, 9) = 10.0;

    cddp::CDDPOptions options;
    options.max_iterations = 120;
    options.line_search.max_iterations = 15;
    options.regularization.initial_value = 1e-4;

    cddp::CDDP solver(
        initial_state,
        goal_state,
        horizon,
        timestep,
        std::make_unique<cddp::Quadrotor>(
            timestep, mass, inertia, arm_length, "rk4"),
        std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, goal_state, std::vector<Eigen::VectorXd>{}, timestep),
        options);

    const Eigen::VectorXd lower =
        Eigen::VectorXd::Zero(control_dim);
    const Eigen::VectorXd upper =
        5.0 * Eigen::VectorXd::Ones(control_dim);
    solver.addPathConstraint("ControlConstraint",
                             std::make_unique<cddp::ControlConstraint>(lower, upper));

    cddp::Quadrotor propagation_model(timestep, mass, inertia, arm_length, "rk4");
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    const double hover_thrust = mass * 9.81 / 4.0;
    for (auto &u : U) {
        u = hover_thrust * Eigen::VectorXd::Ones(control_dim);
    }
    for (int i = 0; i < horizon; ++i) {
        X[i + 1] = propagation_model.getDiscreteDynamics(X[i], U[i], i * timestep);
    }
    solver.setInitialTrajectory(X, U);

    const cddp::CDDPSolution solution = solver.solve(cddp::SolverType::IPDDP);
    if (solution.state_trajectory.empty()) {
        std::cerr << "Quadrotor example failed: empty trajectory" << std::endl;
        return 1;
    }

    const Eigen::VectorXd final_error = solution.state_trajectory.back() - goal_state;
    std::cout << "Quadrotor example completed with status: "
              << solution.status_message << '\n'
              << "Final objective: " << solution.final_objective << '\n'
              << "Final position error norm: " << final_error.head(3).norm()
              << std::endl;
    return 0;
}
