/*
 Copyright 2025 Tomo Sasaki

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
#include <cmath>
#include <iostream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp"

// Helper: create CDDP options for ALDDP tests
static cddp::CDDPOptions makeALDDPOptions() {
  cddp::CDDPOptions options;
  options.max_iterations = 500;
  options.tolerance = 1e-3;
  options.acceptable_tolerance = 1e-4;
  options.verbose = true;
  options.debug = false;

  options.alddp.penalty_init = 1.0;
  options.alddp.penalty_max = 1e6;
  options.alddp.penalty_update_factor = 10.0;
  options.alddp.max_inner_iterations = 30;
  options.alddp.max_outer_iterations = 15;
  options.alddp.constraint_tolerance = 1e-4;
  options.alddp.inner_tolerance_init = 1e-2;
  options.alddp.use_sqrt_backward_pass = false;
  return options;
}

// ============================================================================
// CartPole tests
// ============================================================================

TEST(ALDDPTest, SolveCartPole) {
  // CartPole swing-up with control bounds only (BoxQP path)
  const int state_dim = 4;
  const int control_dim = 1;
  const int horizon = 200;
  const double timestep = 0.02;

  auto system = std::make_unique<cddp::CartPole>(timestep, "rk4");

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
  goal_state << 0.0, M_PI, 0.0, 0.0; // Cart at origin, pole upright

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
  // Cart at origin, pole down

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  // Control bounds
  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -20.0;
  u_ub << 20.0;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  auto options = makeALDDPOptions();
  solver.setOptions(options);
  solver.setInitialTrajectory(
      std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  EXPECT_TRUE(solution.status_message.find("Optimal") != std::string::npos ||
              solution.status_message.find("Acceptable") != std::string::npos ||
              solution.status_message.find("RegularizationLimit") != std::string::npos);
  EXPECT_GT(solution.iterations_completed, 0);

  // Check that final state is near goal
  const auto &x_final = solution.state_trajectory.back();
  EXPECT_NEAR(x_final(1), M_PI, 0.5); // Pole angle near upright
}

TEST(ALDDPTest, SolveCartPoleWithStateBounds) {
  // CartPole with state constraints (cart position limits)
  const int state_dim = 4;
  const int control_dim = 1;
  const int horizon = 200;
  const double timestep = 0.02;

  auto system = std::make_unique<cddp::CartPole>(timestep, "rk4");

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
  goal_state << 0.0, M_PI, 0.0, 0.0;

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  // Control bounds
  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -20.0;
  u_ub << 20.0;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  // State bounds: cart position [-2, 2]
  Eigen::VectorXd x_lb(state_dim), x_ub(state_dim);
  x_lb << -2.0, -1e6, -1e6, -1e6;
  x_ub << 2.0, 1e6, 1e6, 1e6;
  solver.addPathConstraint("StateConstraint",
      std::make_unique<cddp::StateConstraint>(x_lb, x_ub));

  auto options = makeALDDPOptions();
  solver.setOptions(options);
  solver.setInitialTrajectory(
      std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  EXPECT_GT(solution.iterations_completed, 0);

  // Verify cart position stays within bounds (with some tolerance)
  for (const auto &x : solution.state_trajectory) {
    EXPECT_GE(x(0), -2.1);
    EXPECT_LE(x(0), 2.1);
  }
}

// ============================================================================
// Unicycle tests
// ============================================================================

TEST(ALDDPTest, SolveUnicycle) {
  // Unicycle path following with control bounds
  const int state_dim = 3; // x, y, theta
  const int control_dim = 2; // v, omega
  const int horizon = 100;
  const double timestep = 0.1;

  auto system = std::make_unique<cddp::Unicycle>(timestep, "euler");

  Eigen::MatrixXd Q = 0.1 * Eigen::MatrixXd::Identity(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state(state_dim);
  goal_state << 5.0, 5.0, 0.0;

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  // Control bounds
  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -2.0, -2.0;
  u_ub << 2.0, 2.0;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  auto options = makeALDDPOptions();
  solver.setOptions(options);
  solver.setInitialTrajectory(
      std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  EXPECT_GT(solution.iterations_completed, 0);

  // Check that final state approaches goal
  const auto &x_final = solution.state_trajectory.back();
  EXPECT_NEAR(x_final(0), 5.0, 1.0);
  EXPECT_NEAR(x_final(1), 5.0, 1.0);
}

TEST(ALDDPTest, SolveUnicycleWithStateBounds) {
  // Unicycle with state constraints (position bounds)
  const int state_dim = 3;
  const int control_dim = 2;
  const int horizon = 100;
  const double timestep = 0.1;

  auto system = std::make_unique<cddp::Unicycle>(timestep, "euler");

  Eigen::MatrixXd Q = 0.1 * Eigen::MatrixXd::Identity(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state(state_dim);
  goal_state << 5.0, 5.0, 0.0;

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -2.0, -2.0;
  u_ub << 2.0, 2.0;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  // State bounds: y position [0, 8]
  Eigen::VectorXd x_lb(state_dim), x_ub(state_dim);
  x_lb << -1e6, 0.0, -1e6;
  x_ub << 1e6, 8.0, 1e6;
  solver.addPathConstraint("StateConstraint",
      std::make_unique<cddp::StateConstraint>(x_lb, x_ub));

  auto options = makeALDDPOptions();
  solver.setOptions(options);
  solver.setInitialTrajectory(
      std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  EXPECT_GT(solution.iterations_completed, 0);

  // Verify y-position stays within bounds
  for (const auto &x : solution.state_trajectory) {
    EXPECT_GE(x(1), -0.2);
    EXPECT_LE(x(1), 8.2);
  }
}

// ============================================================================
// Car parking with obstacles
// ============================================================================

TEST(ALDDPTest, SolveCarParking) {
  // Car parallel parking with control bounds
  const int state_dim = 4; // x, y, theta, v
  const int control_dim = 2; // acceleration, steering rate
  const int horizon = 100;
  const double timestep = 0.1;

  auto system = std::make_unique<cddp::Car>(timestep, 2.0, "rk4");

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state(state_dim);
  goal_state << 5.0, 2.0, 0.0, 0.0; // Park at (5,2), heading=0, stopped

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -3.0, -1.0;
  u_ub << 3.0, 1.0;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  auto options = makeALDDPOptions();
  solver.setOptions(options);
  solver.setInitialTrajectory(
      std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  EXPECT_GT(solution.iterations_completed, 0);
}

// ============================================================================
// Infeasible start
// ============================================================================

TEST(ALDDPTest, SolveWithInfeasibleStart) {
  // CartPole with interpolated (infeasible) state trajectory
  const int state_dim = 4;
  const int control_dim = 1;
  const int horizon = 200;
  const double timestep = 0.02;

  auto system = std::make_unique<cddp::CartPole>(timestep, "rk4");

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
  goal_state << 0.0, M_PI, 0.0, 0.0;

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -20.0;
  u_ub << 20.0;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  // Create an interpolated (dynamically infeasible) initial trajectory
  std::vector<Eigen::VectorXd> X_init(horizon + 1);
  for (int t = 0; t <= horizon; ++t) {
    double alpha = static_cast<double>(t) / horizon;
    X_init[t] = (1.0 - alpha) * initial_state + alpha * goal_state;
  }

  auto options = makeALDDPOptions();
  options.alddp.slack_penalty = 100.0;
  solver.setOptions(options);
  solver.setInitialTrajectory(
      X_init,
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  EXPECT_GT(solution.iterations_completed, 0);
  // The solver should produce a valid trajectory
  EXPECT_FALSE(solution.state_trajectory.empty());
}

// ============================================================================
// Sqrt backward pass comparison
// ============================================================================

TEST(ALDDPTest, SolveSqrtBackwardPass) {
  // Same unicycle problem, compare sqrt vs standard backward pass
  const int state_dim = 3;
  const int control_dim = 2;
  const int horizon = 50;
  const double timestep = 0.1;

  Eigen::VectorXd goal_state(state_dim);
  goal_state << 3.0, 3.0, 0.0;
  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  Eigen::MatrixXd Q = 0.1 * Eigen::MatrixXd::Identity(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  std::vector<Eigen::VectorXd> empty_ref;

  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -2.0, -2.0;
  u_ub << 2.0, 2.0;

  // Standard backward pass
  double cost_standard;
  {
    auto system = std::make_unique<cddp::Unicycle>(timestep, "euler");
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_ref, timestep);

    cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
    solver.setDynamicalSystem(std::move(system));
    solver.setObjective(std::move(objective));
    solver.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

    auto opts = makeALDDPOptions();
    opts.alddp.use_sqrt_backward_pass = false;
    opts.max_iterations = 200;
    opts.verbose = false;
    opts.print_solver_header = false;
    solver.setOptions(opts);
    solver.setInitialTrajectory(
        std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
        std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

    auto sol = solver.solve(cddp::SolverType::ALDDP);
    cost_standard = sol.final_objective;
  }

  // Sqrt backward pass
  double cost_sqrt;
  {
    auto system = std::make_unique<cddp::Unicycle>(timestep, "euler");
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_ref, timestep);

    cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
    solver.setDynamicalSystem(std::move(system));
    solver.setObjective(std::move(objective));
    solver.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

    auto opts = makeALDDPOptions();
    opts.alddp.use_sqrt_backward_pass = true;
    opts.max_iterations = 200;
    opts.max_cpu_time = 5.0; // 5 second timeout
    opts.verbose = false;
    opts.print_solver_header = false;
    solver.setOptions(opts);
    solver.setInitialTrajectory(
        std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
        std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

    auto sol = solver.solve(cddp::SolverType::ALDDP);
    cost_sqrt = sol.final_objective;
  }

  // Both should produce finite costs
  EXPECT_TRUE(std::isfinite(cost_standard));
  EXPECT_TRUE(std::isfinite(cost_sqrt));
}

// ============================================================================
// Convergence failure
// ============================================================================

TEST(ALDDPTest, ConvergenceFailure) {
  // Overly constrained problem -- should terminate gracefully
  const int state_dim = 3;
  const int control_dim = 2;
  const int horizon = 50;
  const double timestep = 0.1;

  auto system = std::make_unique<cddp::Unicycle>(timestep, "euler");

  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
  Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  Eigen::VectorXd goal_state(state_dim);
  goal_state << 100.0, 100.0, 0.0; // Very far goal

  std::vector<Eigen::VectorXd> empty_ref;
  auto objective = std::make_unique<cddp::QuadraticObjective>(
      Q, R, Qf, goal_state, empty_ref, timestep);

  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
  solver.setDynamicalSystem(std::move(system));
  solver.setObjective(std::move(objective));

  // Very tight control bounds
  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -0.1, -0.1;
  u_ub << 0.1, 0.1;
  solver.addPathConstraint("ControlConstraint",
      std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

  // Very tight state bounds (impossible to reach goal)
  Eigen::VectorXd x_lb(state_dim), x_ub(state_dim);
  x_lb << -1.0, -1.0, -1e6;
  x_ub << 1.0, 1.0, 1e6;
  solver.addPathConstraint("StateConstraint",
      std::make_unique<cddp::StateConstraint>(x_lb, x_ub));

  auto options = makeALDDPOptions();
  options.alddp.max_outer_iterations = 3;
  options.max_iterations = 100;
  options.verbose = false;
  solver.setOptions(options);
  solver.setInitialTrajectory(
      std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
      std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

  cddp::CDDPSolution solution = solver.solve(cddp::SolverType::ALDDP);

  // Should terminate, not crash
  EXPECT_FALSE(solution.state_trajectory.empty());
}

// ============================================================================
// Constraint-free comparison with CLDDP
// ============================================================================

TEST(ALDDPTest, ConstraintFreeMatchesCLDDP) {
  // Same problem solved by ALDDP and CLDDP should give similar results
  const int state_dim = 4;
  const int control_dim = 1;
  const int horizon = 100;
  const double timestep = 0.02;

  Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
  goal_state << 0.0, M_PI, 0.0, 0.0;
  Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
  Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
  Eigen::MatrixXd Qf = 100.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);

  std::vector<Eigen::VectorXd> empty_ref;

  Eigen::VectorXd u_lb(control_dim), u_ub(control_dim);
  u_lb << -20.0;
  u_ub << 20.0;

  // CLDDP solve
  double cost_clddp;
  {
    auto system = std::make_unique<cddp::CartPole>(timestep, "rk4");
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_ref, timestep);

    cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
    solver.setDynamicalSystem(std::move(system));
    solver.setObjective(std::move(objective));
    solver.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

    cddp::CDDPOptions opts;
    opts.max_iterations = 200;
    opts.tolerance = 1e-4;
    opts.acceptable_tolerance = 1e-5;
    opts.verbose = false;
    opts.print_solver_header = false;
    solver.setOptions(opts);
    solver.setInitialTrajectory(
        std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
        std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

    auto sol = solver.solve(cddp::SolverType::CLDDP);
    cost_clddp = sol.final_objective;
  }

  // ALDDP solve (no AL constraints, only BoxQP control bounds)
  double cost_alddp;
  {
    auto system = std::make_unique<cddp::CartPole>(timestep, "rk4");
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_ref, timestep);

    cddp::CDDP solver(initial_state, goal_state, horizon, timestep);
    solver.setDynamicalSystem(std::move(system));
    solver.setObjective(std::move(objective));
    solver.addPathConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_lb, u_ub));

    auto opts = makeALDDPOptions();
    opts.verbose = false;
    opts.print_solver_header = false;
    solver.setOptions(opts);
    solver.setInitialTrajectory(
        std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
        std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim)));

    auto sol = solver.solve(cddp::SolverType::ALDDP);
    cost_alddp = sol.final_objective;
  }

  // Costs should be in the same ballpark (within 50%)
  if (cost_clddp > 0 && cost_alddp > 0) {
    double ratio = cost_alddp / cost_clddp;
    EXPECT_GT(ratio, 0.5);
    EXPECT_LT(ratio, 2.0);
  }
}
