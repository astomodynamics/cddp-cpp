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

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

TEST(CDDPTest, Solve) {
    // Problem parameters
    int state_dim = 3;
    int control_dim = 2;
    int horizon = 100;
    double timestep = 0.03;
    std::string integration_type = "euler";

    // Create a dubins car instance 
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Unicycle>(timestep, integration_type); // Create unique_ptr

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 50.0, 0.0, 0.0,
          0.0, 50.0, 0.0,
          0.0, 0.0, 10.0;
    Qf = 0.5 * Qf;
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
    options.max_iterations = 20;
    options.cost_tolerance = 1e-2;
    options.use_parallel = true;
    options.num_threads = 10;
    options.verbose = true;
    options.debug = false;

    // Create CDDP solver
    cddp::CDDP cddp_solver(
      initial_state, 
      goal_state, 
      horizon, 
      timestep, 
      std::make_unique<cddp::Unicycle>(timestep, integration_type), 
      std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep), 
      options);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.0, -M_PI;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.0, M_PI;
    
    // Add the constraint to the solver
    cddp_solver.addConstraint(std::string("ControlBoxConstraint"), std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Set options
    cddp_solver.setOptions(options);

    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve();
    // cddp::CDDPSolution solution = cddp_solver.solveCLDDP();

    ASSERT_TRUE(solution.converged);

    // Extract solution
    auto X_sol = solution.state_sequence; // size: horizon + 1
    auto U_sol = solution.control_sequence; // size: horizon
    auto t_sol = solution.time_sequence; // size: horizon + 1

    // // Create directory for saving plot (if it doesn't exist)
    // const std::string plotDirectory = "../results/tests";
    // if (!fs::exists(plotDirectory)) {
    //     fs::create_directory(plotDirectory);
    // }

    // // Plot the solution (x-y plane)
    // std::vector<double> x_arr, y_arr, theta_arr;
    // for (const auto& x : X_sol) {
    //     x_arr.push_back(x(0));
    //     y_arr.push_back(x(1));
    //     theta_arr.push_back(x(2));
    // }

    // // Plot the solution (control inputs)
    // std::vector<double> v_arr, omega_arr;
    // for (const auto& u : U_sol) {
    //     v_arr.push_back(u(0));
    //     omega_arr.push_back(u(1));
    // }

    // // Plot the solution by subplots
    // plt::subplot(2, 1, 1);
    // plt::plot(x_arr, y_arr);
    // plt::title("State Trajectory");
    // plt::xlabel("x");
    // plt::ylabel("y");

    // plt::subplot(2, 1, 2);
    // plt::plot(v_arr);
    // plt::plot(omega_arr);
    // plt::plot(std::vector<double>(U_sol.size(), -1.0), "r--");
    // plt::plot(std::vector<double>(U_sol.size(), 1.0), "r--");
    // plt::title("Control Inputs");
    // plt::save(plotDirectory + "/dubincs_car_cddp_test.png");

    // // Create figure and axes
    // plt::figure_size(800, 600);
    // plt::title("Dubins Car Trajectory");
    // plt::xlabel("x");
    // plt::ylabel("y");
    // plt::xlim(-1, 3); // Adjust limits as needed
    // plt::ylim(-1, 3); // Adjust limits as needed

    // // Car dimensions
    // double car_length = 0.2;
    // double car_width = 0.1;

    // // Animation function
    // for (int i = 0; i < X_sol.size(); ++i) {

    //     if (i % 5 == 0) {
    //         // Clear previous plot
	// 		plt::clf();

    //         // Plot car as a box with heading
    //         double x = x_arr[i];
    //         double y = y_arr[i];
    //         double theta = theta_arr[i];
            
    //         // Calculate car corner points
    //         std::vector<double> car_x(5);
    //         std::vector<double> car_y(5);

    //         car_x[0] = x + car_length/2 * cos(theta) - car_width/2 * sin(theta);
    //         car_y[0] = y + car_length/2 * sin(theta) + car_width/2 * cos(theta);

    //         car_x[1] = x + car_length/2 * cos(theta) + car_width/2 * sin(theta);
    //         car_y[1] = y + car_length/2 * sin(theta) - car_width/2 * cos(theta);

    //         car_x[2] = x - car_length/2 * cos(theta) + car_width/2 * sin(theta);
    //         car_y[2] = y - car_length/2 * sin(theta) - car_width/2 * cos(theta);

    //         car_x[3] = x - car_length/2 * cos(theta) - car_width/2 * sin(theta);
    //         car_y[3] = y - car_length/2 * sin(theta) + car_width/2 * cos(theta);

    //         car_x[4] = car_x[0]; // Close the shape
    //         car_y[4] = car_y[0];

    //         // Plot the car
    //         plt::plot(car_x, car_y, "k-");        

    //         // Plot trajectory up to current point
    //         plt::plot(std::vector<double>(x_arr.begin(), x_arr.begin() + i + 1), 
    //                 std::vector<double>(y_arr.begin(), y_arr.begin() + i + 1), "b-");

    //         // Add plot title
    //         plt::title("Dubins Car Trajectory");

    //         // Set labels
    //         plt::xlabel("x");
    //         plt::ylabel("y");

    //         // Adjust limits as needed
    //         plt::xlim(-1, 3); 
    //         plt::ylim(-1, 3); 

    //         // Enable legend
    //         plt::legend();

    //         // Save current frame as an image
    //         std::string filename = plotDirectory + "/dubins_car_" + std::to_string(i) + ".png";
    //         plt::save(filename);
            
    //         // Display plot continuously
    //         plt::pause(0.01); // Pause for a short time

    //     }
    // };
}

// Create gif from images using ImageMagick
// Installation:
// $ sudo apt-get install imagemagick

// convert -delay 100 ../results/tests/dubins_car_*.png ../results/tests/dubins_car.gif 


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
Max Iterations:         20
Max CPU Time:          0

Line Search:
  Max Iterations:    11
  Backtracking Coeff:     1
  Backtracking Min:   0.5
  Backtracking Factor: 0.501187

Log-Barrier:
  Barrier Coeff:  0.01
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
    0   2.123e+02  7.905e-323  1.00e+10     1.000  0.00e+00  1.00e-06  1.00e-02  0.00e+00
    1   2.800e+01   2.800e+01  3.00e+00     1.000  0.00e+00  1.00e-07  1.00e-02  0.00e+00
    2   4.297e+00   4.297e+00  1.47e+00     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00
    3   1.903e+00   1.903e+00  1.45e-01     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00
    4   1.763e+00   1.763e+00  9.74e-02     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00
    5   1.741e+00   1.741e+00  1.91e-02     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00
    6   1.724e+00   1.724e+00  1.46e-02     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00

========================================
           CDDP Solution
========================================
Converged: Yes
Iterations: 7
Solve Time: 6.7100e+03 micro sec
Final Cost: 1.722832e+00
========================================

[       OK ] CDDPTest.Solve (6 ms)
[----------] 1 test from CDDPTest (6 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (6 ms total)
[  PASSED  ] 1 test.
*/



/* Unconstrained
========================================
           CDDP Options
========================================
Cost Tolerance:       0.01
Grad Tolerance:     0.0001
Max Iterations:         20
Max CPU Time:          0

Line Search:
  Max Iterations:    11
  Backtracking Coeff:     1
  Backtracking Min:   0.5
  Backtracking Factor: 0.501187

Log-Barrier:
  Barrier Coeff:  0.01
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

 Iter        Cost        Lagr      Grad      Step      RegS      RegC        Mu      Viol
-----------------------------------------------------------------------------------------
    0   2.123e+02   0.000e+00  1.00e+10     1.000  0.00e+00  1.00e-06  1.00e-02  0.00e+00
    1   5.653e+01   5.653e+01  3.00e+00     0.501  0.00e+00  1.00e-07  1.00e-02  0.00e+00
    2   1.965e+00   1.965e+00  2.15e+00     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00
    3   1.728e+00   1.728e+00  7.53e-02     1.000  0.00e+00  1.00e-08  1.00e-02  0.00e+00

========================================
           CDDP Solution
========================================
Converged: Yes
Iterations: 4
Solve Time: 2.3630e+03 micro sec
Final Cost: 1.720145e+00
========================================

[       OK ] CDDPTest.Solve (2 ms)
[----------] 1 test from CDDPTest (2 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (2 ms total)
[  PASSED  ] 1 test.

*/