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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>

#include "cddp_core/qp_solver.hpp"
#include "cddp_core/boxqp.hpp"

using namespace std;
using namespace Eigen;
using namespace cddp;

// Helper function to print solver results 
void printResults(const std::string& solver_name,
                 const Eigen::VectorXd& solution,
                 double objective_value,
                 double elapsed_time,
                 int status,
                 double constraint_violation = 0.0) {
    std::cout << "\n>>> Results from " << solver_name << ":" << std::endl;
    std::cout << "optimal sol: " << solution.transpose() << std::endl;
    std::cout << "optimal obj: " << objective_value << std::endl;
    std::cout << "elapsed time: " << elapsed_time << "s" << std::endl;
    std::cout << "status: " << status << std::endl;
    if (constraint_violation != 0.0) {
        std::cout << "constraint violation: " << constraint_violation << std::endl;
    }
    std::cout << ">>> End of " << solver_name << " test" << std::endl;
}

TEST(QPSolver, ComparisonTest) {     
    // Problem setup
    const int n = 5;  // variables
    const int m = 10;  // constraints

    // Define QP problem
    Matrix<double, 5, 5> Q;
    Matrix<double, 5, 1> q;
    Matrix<double, -1, 5> A(m, 5);
    VectorXd b(m);
    VectorXd lb(m), ub(m);

    // Setup quadratic term and linear term
    Q << 4.0, 1.0, 0.0, 0.0, 0.0,
         1.0, 2.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 3.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 2.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 2.0;
    q << -1.0, -4.0, 0.0, -3.0, -2.0;

    // Setup constraints with both lower and upper bounds
    A << 1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 1.0,
         -1.0, 0.0, 0.0, 0.0, 0.0,
         0.0, -1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, -1.0, 0.0, 0.0,
         0.0, 0.0, 0.0, -1.0, 0.0,
         0.0, 0.0, 0.0, 0.0, -1.0;
    b << 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    lb = VectorXd::Constant(m/2, 0.0);
    ub = VectorXd::Constant(m/2, 2.0);

    std::cout << "\n====== Comparing QP Solvers ======\n" << std::endl;

    // 3. Test BoxQP for comparison - only uses simple bounds
    {
        BoxQPOptions options;
        options.verbose = false;
        BoxQPSolver solver(options);

        VectorXd box_lb = VectorXd::Constant(n, 0.0);
        VectorXd box_ub = VectorXd::Constant(n, 2.0);

        auto start_time = std::chrono::high_resolution_clock::now();
        BoxQPResult result = solver.solve(Q, q, box_lb, box_ub);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        printResults("BoxQP", result.x, result.final_value, 
                    elapsed.count(), static_cast<int>(result.status));
    }

    // 5. Test our QP solver 
    {
        QPSolverOptions options;
        options.verbose = false;
        QPSolver solver(options);

        solver.setDimensions(n, m);
        solver.setHessian(Q);
        solver.setGradient(q);
        solver.setConstraints(A, b);

        auto start_time = std::chrono::high_resolution_clock::now();
        QPResult result = solver.solve();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        printResults("QPSolver", result.x, result.objective_value, 
                    elapsed.count(), static_cast<int>(result.status));

    }
}

/* $ ./tests/test_qp_solvers
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from QPSolver
[ RUN      ] QPSolver.ComparisonTest

====== Comparing QP Solvers ======


>>> Results from OSQP:
optimal sol: -8.58736e-05      1.66673            0     0.666634            1
optimal obj: -5.33339
elapsed time: 5.712e-06s
status: 0
>>> End of OSQP test

>>> Results from SDQP:
optimal sol:        0  1.66667        0 0.666667        1
optimal obj: -5.33333
elapsed time: 1.0261e-05s
status: 0
>>> End of SDQP test

>>> Results from BoxQP:
optimal sol:        0  1.66667        0 0.666667        1
optimal obj: -5.33333
elapsed time: 6.885e-06s
status: 5
>>> End of BoxQP test
Set parameter Username
Academic license - for non-commercial use only - expires 2025-09-25

>>> Results from Gurobi:
optimal sol: 0 0 0 0 0
optimal obj: -5.33333
elapsed time: 0.000398952s
status: 2
>>> End of Gurobi test

>>> Results from QPSolver:
optimal sol:        0  1.66667       -0 0.666667        1
optimal obj: -5.33333
elapsed time: 6.922e-06s
status: 0
>>> End of QPSolver test
[       OK ] QPSolver.ComparisonTest (1 ms)
[----------] 1 test from QPSolver (1 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1 ms total)
[  PASSED  ] 1 test.
*/
