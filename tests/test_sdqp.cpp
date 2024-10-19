#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"
#include "gurobi_c++.h"
#include <chrono>

using namespace std;
using namespace Eigen;

TEST(SDQP, test_sdqp) {   
    // Test for sdqp
    int m = 7;
    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Matrix<double, 3, 1> c;
    Eigen::VectorXd x(3);        // decision variables
    Eigen::Matrix<double, -1, 3> A(m, 3); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    Q << 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0;
    c << 1.2, 2.5, -10.0;

    A << 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        -0.7, 0.5, 0.0,
        0.5, -1.0, 0.0,
        0.0, 0.13, -1.0,
        0.1, -3.0, -1.3;
    b << 10.0, 10.0, 10.0, 1.7, -7.1, -3.31, 2.59;

    auto start_time = std::chrono::high_resolution_clock::now();
    double minobj = sdqp::sdqp(Q, c, A, b, x);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << minobj << std::endl;
    std::cout << "cons precision: " << (A * x - b).maxCoeff() << std::endl;
    std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;

    // Test for osqp
    Eigen::SparseMatrix<double> P(3, 3);
    Eigen::VectorXd q(3);
    Eigen::SparseMatrix<double> G(m, 3);
    Eigen::VectorXd h(m);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (Q(i, j) != 0) {
                P.insert(i, j) = Q(i, j);
            }
        }
    }
    P.makeCompressed(); // Important for efficient storage and operations

    for (int i = 0; i < 3; ++i) {
        q(i) = c(i);
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (A(i, j) != 0) {
                G.insert(i, j) = A(i, j);
            }
        }
        h(i) = b(i);
    }
    G.makeCompressed(); // Important for efficient storage and operations



    osqp::OsqpSolver osqp_solver;
    osqp::OsqpInstance osqp_instance;
    osqp::OsqpSettings osqp_settings;

    osqp_instance.objective_matrix = P;
    osqp_instance.objective_vector = q;
    osqp_instance.constraint_matrix = G;
    osqp_instance.upper_bounds = h;
    osqp_instance.lower_bounds = Eigen::VectorXd::Constant(m, -std::numeric_limits<double>::infinity());

    osqp_settings.verbose = true;


    osqp_solver.Init(osqp_instance, osqp_settings);

    start_time = std::chrono::high_resolution_clock::now();
    osqp_solver.Solve();
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;

    std::cout << "optimal sol: " << osqp_solver.primal_solution().transpose() << std::endl;
    std::cout << "optimal obj: " << osqp_solver.objective_value() << std::endl;
    std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;

    // Test for gurobi
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    // Create variables
    GRBVar x1 = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x1");
    GRBVar x2 = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x2");
    GRBVar x3 = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x3");

    // Set objective
    GRBQuadExpr obj = x1 * x1 + x2 * x2 + x3 * x3 + x1 * x2 + x1 * x3 + x2 * x3 + 1.2 * x1 + 2.5 * x2 - 10 * x3;
    model.setObjective(obj);

    // Add constraints
    model.addConstr(x1 <= 10, "c0");
    model.addConstr(x2 <= 10, "c1");
    model.addConstr(x3 <= 10, "c2");
    model.addConstr(-0.7 * x1 + 0.5 * x2 <= 1.7, "c3");
    model.addConstr(0.5 * x1 - x2 <= -7.1, "c4");
    model.addConstr(0.13 * x2 - x3 <= -3.31, "c5");
    model.addConstr(0.1 * x1 - 3 * x2 - 1.3 * x3 <= 2.59, "c6");

    // Optimize model
    start_time = std::chrono::high_resolution_clock::now();
    model.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    elapsed = end_time - start_time;

    std::cout << "optimal sol: " << x1.get(GRB_DoubleAttr_X) << " " 
              << x2.get(GRB_DoubleAttr_X) << " " 
              << x3.get(GRB_DoubleAttr_X) << std::endl;
    std::cout << "optimal obj: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
    std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;

 
}

// Output:
/*
tom:~/github/cddp-cpp/build$ ./tests/test_sdqp 
Running main() from /home/tom/github/cddp-cpp/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from SDQP
[ RUN      ] SDQP.test_sdqp
optimal sol: 4.11111 9.15556 4.50022
optimal obj: 201.14
cons precision: 0
elapsed time: 1.362e-05s
-----------------------------------------------------------------
           OSQP v0.6.0  -  Operator Splitting QP Solver
              (c) Bartolomeo Stellato,  Goran Banjac
        University of Oxford  -  Stanford University 2019
-----------------------------------------------------------------
problem:  variables n = 3, constraints m = 7
          nnz(P) + nnz(A) = 18
settings: linear system solver = qdldl,
          eps_abs = 1.0e-03, eps_rel = 1.0e-03,
          eps_prim_inf = 1.0e-04, eps_dual_inf = 1.0e-04,
          rho = 1.00e-01 (adaptive),
          sigma = 1.00e-06, alpha = 1.60, max_iter = 4000
          check_termination: on (interval 25),
          scaling: on, scaled_termination: off
          warm start: on, polish: off, time_limit: off

iter   objective    pri res    dua res    rho        time
   1  -3.5142e+01   7.71e+00   1.70e+01   1.00e-01   4.73e-05s
  50   2.0056e+02   3.91e-03   3.97e-03   1.00e-01   8.43e-05s

status:               solved
number of iterations: 50
optimal objective:    200.5642
run time:             9.31e-05s
optimal rho estimate: 1.72e-01

optimal sol: 4.09913 9.14661 4.49725
optimal obj: 200.564
elapsed time: 7.3378e-05s
Set parameter Username
Academic license - for non-commercial use only - expires 2025-09-25
Gurobi Optimizer version 11.0.3 build v11.0.3rc0 (linux64 - "Ubuntu 22.04.5 LTS")

CPU model: 13th Gen Intel(R) Core(TM) i7-13620H, instruction set [SSE2|AVX|AVX2]
Thread count: 16 physical cores, 16 logical processors, using up to 16 threads

Optimize a model with 7 rows, 3 columns and 12 nonzeros
Model fingerprint: 0x8e70d5c4
Model has 6 quadratic objective terms
Coefficient statistics:
  Matrix range     [1e-01, 3e+00]
  Objective range  [1e+00, 1e+01]
  QObjective range [4e+00, 4e+00]
  Bounds range     [0e+00, 0e+00]
  RHS range        [2e+00, 1e+01]
Presolve removed 3 rows and 0 columns
Presolve time: 0.00s
Presolved: 4 rows, 3 columns, 9 nonzeros
Presolved model has 6 quadratic objective terms
Ordering time: 0.00s

Barrier statistics:
 Free vars  : 2
 AA' NZ     : 1.500e+01
 Factor NZ  : 2.100e+01
 Factor Ops : 9.100e+01 (less than 1 second per iteration)
 Threads    : 1

                  Objective                Residual
Iter       Primal          Dual         Primal    Dual     Compl     Time
   0   4.53999539e+01 -9.95055472e+04  2.61e+00 1.00e+03  1.00e+06     0s
   1   4.78271120e+04 -1.18614730e+05  7.38e+01 7.48e+01  7.95e+04     0s
   2   2.62571225e+02 -5.84043340e+04  2.18e+00 2.27e+00  4.31e+03     0s
   3   7.03334911e+02 -1.24970817e+04  8.88e-16 2.27e-06  4.71e+02     0s
   4   6.86124700e+02  1.91875199e+02  1.78e-15 5.68e-08  1.77e+01     0s
   5   4.57531707e+02  3.34254847e+02  4.44e-15 6.75e-14  4.40e+00     0s
   6   4.20979293e+02  4.19231361e+02  2.75e-14 7.11e-15  6.24e-02     0s
   7   4.19461307e+02  4.19459545e+02  1.13e-13 4.71e-15  6.30e-05     0s
   8   4.19459774e+02  4.19459773e+02  3.02e-14 2.87e-15  6.30e-08     0s
   9   4.19459773e+02  4.19459773e+02  1.78e-14 5.86e-15  6.30e-11     0s

Barrier solved model in 9 iterations and 0.00 seconds (0.00 work units)
Optimal objective 4.19459773e+02

optimal sol: 4.11111 9.15556 4.50022
optimal obj: 419.46
elapsed time: 0.00133694s
[       OK ] SDQP.test_sdqp (4 ms)
[----------] 1 test from SDQP (4 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (4 ms total)
[  PASSED  ] 1 test.
*/ 
