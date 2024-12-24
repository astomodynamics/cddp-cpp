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

TEST(BoxQPSolver, LargeDimensionTest) {
    // Problem setup
    const int n = 15;  // variables
    VectorXd box_lb = VectorXd::Constant(n, -2.0);
    VectorXd box_ub = VectorXd::Constant(n, 2.0);
    Eigen::Matrix<double, 15, 15> Q;
    Q << 
      // Row 0
       4.12533748e+00,  1.97764831e-01, -3.61210691e-01,  8.00623610e-01,  7.62345888e-01, -1.06898343e-01,
       1.30339351e-01, -1.13259907e-01, -6.96950858e-01,  5.34670445e-01,  5.09630616e-01,  1.60610297e+00,
       4.62892147e-01, -4.16042523e-02,  1.15372788e+00,

      // Row 1
       1.97764831e-01,  6.09416699e+00,  1.24916711e-01,  3.03114135e-01,  8.33526744e-02, -7.50650156e-01,
       3.32367503e-01, -8.07693384e-01, -3.63386713e-01,  5.97739905e-02,  1.12281290e-01,  7.82962364e-01,
      -4.31227827e-01, -7.78651128e-01,  3.53467184e-01,

      // Row 2
      -3.61210691e-01,  1.24916711e-01,  5.50001670e+00, -1.15999299e+00,  2.52935791e-02,  6.78993972e-01,
      -4.26688352e-01,  2.79368593e-01, -1.20240548e+00,  5.81617495e-03, -6.60420998e-01, -8.52349453e-01,
      -2.21853251e-01,  3.25270059e-01, -4.93159468e-01,

      // Row 3
       8.00623610e-01,  3.03114135e-01, -1.15999299e+00,  4.77842560e+00,  2.32184776e-04, -7.16724074e-01,
       1.17921496e+00, -2.44985365e-01, -6.36689772e-01, -1.18475001e+00, -1.06986243e+00,  1.15671298e+00,
       3.28904560e-01, -1.99200945e-02, -6.51702604e-01,

      // Row 4
       7.62345888e-01,  8.33526744e-02,  2.52935791e-02,  2.32184776e-04,  4.40332914e+00,  4.66324836e-01,
      -7.95405086e-01, -2.58389526e-01, -2.64881040e-01, -9.80426131e-01,  7.14939428e-01, -8.29804991e-02,
      -9.83858675e-01, -1.10025256e-01, -4.81364585e-01,

      // Row 5
      -1.06898343e-01, -7.50650156e-01,  6.78993972e-01, -7.16724074e-01,  4.66324836e-01,  3.33222011e+00,
      -1.74735819e-01,  7.60206374e-01,  1.38069876e-01, -1.14377493e+00,  2.49459813e-01, -3.03438457e-01,
       4.87653948e-01, -1.41316602e+00, -5.60885984e-01,

      // Row 6
       1.30339351e-01,  3.32367503e-01, -4.26688352e-01,  1.17921496e+00, -7.95405086e-01, -1.74735819e-01,
       4.67874106e+00, -2.86394759e-01, -3.00259011e-01,  2.58571169e-01,  2.28293192e-01, -1.12559618e-01,
      -2.78415986e-01, -2.69818813e-01, -1.16263409e+00,

      // Row 7
      -1.13259907e-01, -8.07693384e-01,  2.79368593e-01, -2.44985365e-01, -2.58389526e-01,  7.60206374e-01,
      -2.86394759e-01,  5.41952411e+00,  4.44979488e-01,  9.16500690e-01,  8.42587916e-02,  2.27753618e-01,
       1.52053204e+00,  2.93800655e-01,  1.77636682e-01,

      // Row 8
      -6.96950858e-01, -3.63386713e-01, -1.20240548e+00, -6.36689772e-01, -2.64881040e-01,  1.38069876e-01,
      -3.00259011e-01,  4.44979488e-01,  5.07267790e+00, -1.62937574e-01,  1.36247527e-01, -1.82020551e-01,
       1.15182717e+00, -7.22601223e-01,  1.48759838e+00,

      // Row 9
       5.34670445e-01,  5.97739905e-02,  5.81617495e-03, -1.18475001e+00, -9.80426131e-01, -1.14377493e+00,
       2.58571169e-01,  9.16500690e-01, -1.62937574e-01,  5.46146888e+00, -2.05325482e-01, -9.36657857e-02,
      -1.71213726e-02,  5.30645010e-01,  2.73102921e-01,

      // Row 10
       5.09630616e-01,  1.12281290e-01, -6.60420998e-01, -1.06986243e+00,  7.14939428e-01,  2.49459813e-01,
       2.28293192e-01,  8.42587916e-02,  1.36247527e-01, -2.05325482e-01,  5.23868157e+00, -5.17103373e-01,
      -8.16166203e-01, -4.75320773e-01,  7.70793551e-02,

      // Row 11
       1.60610297e+00,  7.82962364e-01, -8.52349453e-01,  1.15671298e+00, -8.29804991e-02, -3.03438457e-01,
      -1.12559618e-01,  2.27753618e-01, -1.82020551e-01, -9.36657857e-02, -5.17103373e-01,  4.34814350e+00,
      -1.42655295e+00, -1.51741483e-01,  5.85827374e-01,

      // Row 12
       4.62892147e-01, -4.31227827e-01, -2.21853251e-01,  3.28904560e-01, -9.83858675e-01,  4.87653948e-01,
      -2.78415986e-01,  1.52053204e+00,  1.15182717e+00, -1.71213726e-02, -8.16166203e-01, -1.42655295e+00,
       6.06160017e+00, -5.47030759e-01, -1.17969147e+00,

      // Row 13
      -4.16042523e-02, -7.78651128e-01,  3.25270059e-01, -1.99200945e-02, -1.10025256e-01, -1.41316602e+00,
      -2.69818813e-01,  2.93800655e-01, -7.22601223e-01,  5.30645010e-01, -4.75320773e-01, -1.51741483e-01,
      -5.47030759e-01,  4.51860738e+00,  1.44827840e+00,

      // Row 14
       1.15372788e+00,  3.53467184e-01, -4.93159468e-01, -6.51702604e-01, -4.81364585e-01, -5.60885984e-01,
      -1.16263409e+00,  1.77636682e-01,  1.48759838e+00,  2.73102921e-01,  7.70793551e-02,  5.85827374e-01,
      -1.17969147e+00,  1.44827840e+00,  5.10583464e+00;

      Eigen::Matrix<double, 15, 1> q;
        q << -0.86050975,
            -1.63121951,
            -0.30147231,
            -0.25623270,
            0.85766191,
            -0.11059050,
            -0.43243198,
            1.07703747,
            -0.22482656,
            -0.57624182,
            0.57460892,
            -0.48982822,
            0.65880214,
            -0.59691711,
            -0.22295918;

        BoxQPOptions options;
        options.verbose = false;
        BoxQPSolver solver(options);

        auto start_time = std::chrono::high_resolution_clock::now();
        BoxQPResult result = solver.solve(Q, q, box_lb, box_ub);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        printResults("BoxQP", result.x, result.final_value, 
                    elapsed.count(), static_cast<int>(result.status));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
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
