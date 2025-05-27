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
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp"

TEST(IPDDPTest, Solve) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a Unicycle instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(timestep, integration_type); // Create unique_ptr

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.05 / timestep * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0, 0.0, 0.0,
          0.0, 100.0, 0.0,
          0.0, 0.0, 100.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;

    // Create an empty vector of Eigen::VectorXd
    std::vector<Eigen::VectorXd> empty_reference_states; 
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial and target states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI/4.0; 

    // Create CDDP Options
    cddp::CDDPOptions options;
    options.max_iterations = 60;
    options.cost_tolerance = 1e-7;
    options.use_parallel = false;
    options.num_threads = 1;
    options.verbose = true;
    options.debug = false;
    options.barrier_coeff = 1e-1;

    // Create CDDP solver
    cddp::CDDP cddp_solver(
      initial_state, 
      goal_state, 
      horizon, 
      timestep, 
      std::make_unique<cddp::Unicycle>(timestep, integration_type), 
      std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep), 
      options);
    // cddp_solver.setDynamicalSystem(std::move(system));
    // cddp_solver.setObjective(std::move(objective));

    // Define constraints
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.1, 1*M_PI;
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound = -control_upper_bound;
    
    
    // Add the constraint to the solver
    cddp_solver.addConstraint(std::string("ControlConstraint"), std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    auto constraint = cddp_solver.getConstraint<cddp::ControlConstraint>("ControlConstraint");

    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    for (int i = 0; i < horizon; ++i) {
      U[i] << 0.01, 0.01;
      X[i+1] = system->getDiscreteDynamics(X[i], U[i]);
    }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    ASSERT_TRUE(solution.converged);

    // Extract solution
    auto X_sol = solution.state_sequence; // size: horizon + 1
    auto U_sol = solution.control_sequence; // size: horizon
    auto t_sol = solution.time_sequence; // size: horizon + 1
}


/*
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from CDDPTest
[ RUN      ] CDDPTest.Solve
QuadraticObjective: Using single reference state
QuadraticObjective: Using single reference state

+---------------------------------------------------------+
|    ____ ____  ____  ____    _          ____             |
|   / ___|  _ \|  _ \|  _ \  (_)_ __    / ___| _     _    |
|  | |   | | | | | | | |_) | | | '_ \  | |   _| |_ _| |_  |
|  | |___| |_| | |_| |  __/  | | | | | | |__|_   _|_   _| |
|   \____|____/|____/|_|     |_|_| |_|  \____||_|   |_|   |
+---------------------------------------------------------+

Constrained Differential Dynamic Programming
Author: Tomo Sasaki (@astomodynamics)
----------------------------------------------------------


========================================
           CDDP Options
========================================
Cost Tolerance:       0.01
Grad Tolerance:     0.0001
Max Iterations:         40
Max CPU Time:          0

Line Search:
  Max Iterations:    11
  Backtracking Coeff:     1
  Backtracking Min:   0.5
  Backtracking Factor: 0.501187

Log-Barrier:
  Barrier Coeff: 1e-06
  Barrier Factor:   0.1
  Barrier Tolerance: 1e-08
  Relaxation Coeff:     1
  Barrier Order:     2
  Filter Acceptance: 1e-08
  Constraint Tolerance: 1e-12

Regularization:
  Regularization Type: control
  Regularization State: 1e-06
  Regularization State Step:     1
  Regularization State Max: 10000
  Regularization State Min: 1e-08
  Regularization State Factor:    10
  Regularization Control: 1e-06
  Regularization Control Step:     1
  Regularization Control Max: 10000
  Regularization Control Min: 1e-08
  Regularization Control Factor:    10

Other:
  Print Iterations: Yes
  iLQR: Yes
  Use Parallel: Yes
  Num Threads: 10
  Relaxed Log-Barrier: No
  Early Termination: Yes

BoxQP:
  BoxQP Max Iterations: 100
  BoxQP Min Grad: 1e-08
  BoxQP Min Rel Improve: 1e-08
  BoxQP Step Dec: 0.6
  BoxQP Min Step: 1e-22
  BoxQP Armijo: 0.1
  BoxQP Verbose: No
========================================

ControlBoxConstraint is set
 Iter        Cost        Lagr      Grad      Step      RegS      RegC        Mu      Viol
-----------------------------------------------------------------------------------------
    0   2.123e+02   2.123e+02  1.00e+10     1.000  0.00e+00  1.00e-06  1.00e-06  0.00e+00
    1   1.064e+02   1.064e+02  3.14e+00     0.501  0.00e+00  1.00e-07  1.00e-06  0.00e+00
    2   7.413e+01   7.413e+01  3.14e+00     0.251  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    3   5.256e+01   5.256e+01  3.19e+00     0.251  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    4   2.596e+01   2.596e+01  3.25e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    5   1.438e+01   1.438e+01  3.36e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    6   8.929e+00   8.929e+00  3.46e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    7   6.145e+00   6.145e+00  3.56e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    8   4.610e+00   4.610e+00  3.64e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
    9   3.703e+00   3.703e+00  3.71e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
 Iter        Cost        Lagr      Grad      Step      RegS      RegC        Mu      Viol
-----------------------------------------------------------------------------------------
   10   3.136e+00   3.136e+00  3.77e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   11   2.764e+00   2.764e+00  3.83e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   12   2.510e+00   2.510e+00  3.88e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   13   2.330e+00   2.330e+00  3.92e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   14   2.200e+00   2.200e+00  3.97e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   15   2.104e+00   2.104e+00  4.00e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   16   2.031e+00   2.031e+00  4.03e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   17   1.976e+00   1.976e+00  4.06e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   18   1.932e+00   1.932e+00  4.09e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   19   1.898e+00   1.898e+00  4.11e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
 Iter        Cost        Lagr      Grad      Step      RegS      RegC        Mu      Viol
-----------------------------------------------------------------------------------------
   20   1.871e+00   1.871e+00  4.13e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   21   1.850e+00   1.850e+00  4.15e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   22   1.833e+00   1.833e+00  4.16e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   23   1.819e+00   1.819e+00  4.18e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00
   24   1.807e+00   1.807e+00  4.19e+00     0.501  0.00e+00  1.00e-08  1.00e-06  0.00e+00

========================================
           CDDP Solution
========================================
Converged: Yes
Iterations: 25
Solve Time: 2.5739e+04 micro sec
Final Cost: 1.797722e+00
========================================

[       OK ] CDDPTest.Solve (612 ms)
[----------] 1 test from CDDPTest (612 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (612 ms total)
[  PASSED  ] 1 test.
*/