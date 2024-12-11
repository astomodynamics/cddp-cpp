#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>

#include "cddp_core/helper.hpp"
#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"
#include "gurobi_c++.h"

using namespace std;
using namespace Eigen;
using namespace cddp;

TEST(BoxQP, ComparisonTest) {   
    // Problem setup
    const int n = 5;  // variables
    const int m = 10;  // constraints

    // Define QP problem
    Matrix<double, 5, 5> Q;
    Matrix<double, 5, 1> c;
    Matrix<double, -1, 5> A(m, 5);
    VectorXd b(m);

    // Setup quadratic term and linear term
    Q << 4.0, 1.0, 0.0, 0.0, 0.0,
         1.0, 2.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 3.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 2.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 2.0;
    c << -1.0, -4.0, 0.0, -3.0, -2.0;

    // Setup box constraints (bounds of the form lower ≤ A x ≤ upper => lower ≤ x ≤ upper)
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

    std::cout << "\n====== Comparing QP Solvers ======\n" << std::endl;

    // 1. Test BoxQP
    {
        std::cout << "\n>>> Testing BoxQP:" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        BoxQPOptions options;
        options.verbose = false;
        BoxQPSolver solver(options);

        // Set box constraints directly: 0 ≤ x_i ≤ 2
        VectorXd lower = VectorXd::Constant(n, 0.0);
        VectorXd upper = VectorXd::Constant(n, 2.0);

        BoxQPResult result = solver.solve(Q, c, lower, upper);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "optimal sol: " << result.x.transpose() << std::endl;
        std::cout << "optimal obj: " << result.final_value << std::endl;
        std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;
        std::cout << "status: " << static_cast<int>(result.status) << std::endl;

        std::cout << ">>> End of BoxQP test" << std::endl;
    }

    // 2. Test SDQP
    {
        std::cout << "\n>>> Testing SDQP:" << std::endl;
        VectorXd x(n);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        double minobj = sdqp::sdqp(Q, c, A, b, x);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "optimal sol: " << x.transpose() << std::endl;
        std::cout << "optimal obj: " << minobj << std::endl;
        std::cout << "cons precision: " << (A * x - b).maxCoeff() << std::endl;
        std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;

        std::cout << ">>> End of SDQP test" << std::endl;
    }

    // 3. Test OSQP
    {
        std::cout << "\n>>> Testing OSQP:" << std::endl;

        // Convert to sparse matrices
        SparseMatrix<double> P(n, n);
        VectorXd q = c;
        SparseMatrix<double> G(m/2, n);
        // The first half of b is the upper bound, the second half is the lower bound
        VectorXd uppper = b.head(m/2);
        VectorXd lower = b.tail(m/2);

        // Convert Q to sparse
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Q(i,j) != 0) {
                    P.insert(i, j) = Q(i,j);
                }
            }
        }
        P.makeCompressed();

        // Convert A to sparse
        for (int i = 0; i < m / 2; i++) {
            for (int j = 0; j < n; j++) {
                if (A(i,j) != 0) {
                    G.insert(i, j) = A(i,j);
                }
            }
        }
        G.makeCompressed();

        osqp::OsqpSolver solver;
        osqp::OsqpInstance instance;
        osqp::OsqpSettings settings;

        instance.objective_matrix = P;
        instance.objective_vector = q;
        instance.constraint_matrix = G;
        instance.upper_bounds = uppper;
        instance.lower_bounds = lower;

        settings.verbose = false;

        solver.Init(instance, settings);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto status = solver.Solve();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "optimal sol: " << solver.primal_solution().transpose() << std::endl;
        std::cout << "optimal obj: " << solver.objective_value() << std::endl;
        std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;
        std::cout << "status: " << static_cast<int>(status) << std::endl;

        std::cout << ">>> End of OSQP test" << std::endl;
    }

    // 4. Test Gurobi
    {
        std::cout << "\n>>> Testing Gurobi:" << std::endl;
        
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        // Create 5 variables with box constraints: 0 ≤ x ≤ 2
        GRBVar x[5];
        for (int i = 0; i < n; ++i) {
            x[i] = model.addVar(0.0, 2.0, 0.0, GRB_CONTINUOUS, "x" + std::to_string(i+1));
        }

        // Set objective:
        GRBQuadExpr obj = 0.0;
        // Add quadratic terms
        for (int i = 0; i < n; i++) {
            // diagonal terms
            obj += 0.5 * Q(i,i) * x[i] * x[i];
            // off-diagonal terms (take only j > i to avoid double-counting)
            for (int j = i+1; j < n; j++) {
                if (Q(i,j) != 0.0) {
                    // Add (Q(i,j) + Q(j,i))/2 * x_i * x_j because Q is symmetric
                    double val = 0.5 * (Q(i,j) + Q(j,i));
                    obj += val * x[i] * x[j];
                }
            }
        }

        // Add linear terms
        for (int i = 0; i < n; i++) {
            if (c(i) != 0.0) {
                obj += c(i) * x[i];
            }
        }
        model.setObjective(obj, GRB_MINIMIZE);

        auto start_time = std::chrono::high_resolution_clock::now();
        model.optimize();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "optimal sol: ";
        for (int i = 0; i < n; i++) {
            std::cout << x[i].get(GRB_DoubleAttr_X) << (i == n-1 ? "\n" : " ");
        }

        std::cout << "optimal obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
        std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;
        std::cout << "status: " << model.get(GRB_IntAttr_Status) << std::endl;

        std::cout << ">>> End of Gurobi test" << std::endl;
    }
}

// $ ./tests/test_boxqp 
// [==========] Running 1 test from 1 test suite.
// [----------] Global test environment set-up.
// [----------] 1 test from BoxQP
// [ RUN      ] BoxQP.ComparisonTest

// ====== Comparing QP Solvers ======


// >>> Testing BoxQP:
// optimal sol:        0  1.66667        0 0.666667        1
// optimal obj: -5.33333
// elapsed time: 9.042e-06s
// status: 5
// >>> End of BoxQP test

// >>> Testing SDQP:
// optimal sol:        0  1.66667        0 0.666667        1
// optimal obj: -5.33333
// cons precision: 0
// elapsed time: 1.5736e-05s
// >>> End of SDQP test

// >>> Testing OSQP:
// optimal sol: -8.58736e-05      1.66673            0     0.666634            1
// optimal obj: -5.33339
// elapsed time: 5.986e-06s
// status: 0
// >>> End of OSQP test

// >>> Testing Gurobi:
// Set parameter Username
// Academic license - for non-commercial use only - expires 2025-09-25
// Gurobi Optimizer version 11.0.3 build v11.0.3rc0 (linux64 - "Ubuntu 22.04.5 LTS")

// CPU model: 13th Gen Intel(R) Core(TM) i7-13620H, instruction set [SSE2|AVX|AVX2]
// Thread count: 16 physical cores, 16 logical processors, using up to 16 threads

// Optimize a model with 0 rows, 5 columns and 0 nonzeros
// Model fingerprint: 0xc94a4dee
// Model has 7 quadratic objective terms
// Coefficient statistics:
//   Matrix range     [0e+00, 0e+00]
//   Objective range  [1e+00, 4e+00]
//   QObjective range [2e+00, 4e+00]
//   Bounds range     [2e+00, 2e+00]
//   RHS range        [0e+00, 0e+00]
// Presolve removed 0 rows and 2 columns
// Presolve time: 0.00s
// Presolved: 0 rows, 3 columns, 0 nonzeros
// Presolved model has 5 quadratic objective terms
// Ordering time: 0.00s

// Barrier statistics:
//  Free vars  : 2
//  AA' NZ     : 1.000e+00
//  Factor NZ  : 3.000e+00
//  Factor Ops : 5.000e+00 (less than 1 second per iteration)
//  Threads    : 1

//                   Objective                Residual
// Iter       Primal          Dual         Primal    Dual     Compl     Time
//    0   1.72657061e+05 -1.87160686e+05  1.29e+03 3.56e+02  1.00e+06     0s
//    1  -7.20718056e+00 -7.97784134e+03  6.88e-01 1.90e-01  1.86e+03     0s
//    2  -3.97508523e+00 -7.28536444e+02  6.88e-07 1.90e-07  1.21e+02     0s
//    3  -4.02593688e+00 -8.34733151e+00  2.29e-09 6.34e-10  7.20e-01     0s
//    4  -5.10505253e+00 -5.77644852e+00  2.44e-15 7.77e-16  1.12e-01     0s
//    5  -5.31276160e+00 -5.35788488e+00  5.55e-17 2.64e-16  7.52e-03     0s
//    6  -5.33313779e+00 -5.33379628e+00  4.44e-16 2.22e-16  1.10e-04     0s
//    7  -5.33333314e+00 -5.33333380e+00  0.00e+00 7.10e-17  1.10e-07     0s
//    8  -5.33333333e+00 -5.33333333e+00  0.00e+00 2.22e-16  1.10e-10     0s

// Barrier solved model in 8 iterations and 0.00 seconds (0.00 work units)
// Optimal objective -5.33333333e+00

// optimal sol: 2.94244e-10 1.66667 0 0.666667 1
// optimal obj: -5.33333
// elapsed time: 0.000480111s
// status: 2
// >>> End of Gurobi test
// [       OK ] BoxQP.ComparisonTest (1 ms)
// [----------] 1 test from BoxQP (1 ms total)

// [----------] Global test environment tear-down
// [==========] 1 test from 1 test suite ran. (1 ms total)
// [  PASSED  ] 1 test.