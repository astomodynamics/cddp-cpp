#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"
#include <chrono>

using namespace std;
using namespace Eigen;

TEST(SDQP, test_sdqp) {   
    // Test for sdqp
    int m = 7;
    Eigen::Matrix<double, 3, 3> Q;
    Eigen::Matrix<double, 3, 1> c;
    Eigen::Matrix<double, 3, 1> x;        // decision variables
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
    double minobj = sdqp::sdqp<3>(Q, c, A, b, x);
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
    
}

// Output:
// optimal sol: 4.11111 9.15556 4.50022
// optimal obj: 201.14
// cons precision: 0
// elapsed time: 2.4469e-05s
// -----------------------------------------------------------------
//            OSQP v0.6.0  -  Operator Splitting QP Solver
//               (c) Bartolomeo Stellato,  Goran Banjac
//         University of Oxford  -  Stanford University 2019
// -----------------------------------------------------------------
// problem:  variables n = 3, constraints m = 7
//           nnz(P) + nnz(A) = 18
// settings: linear system solver = qdldl,
//           eps_abs = 1.0e-03, eps_rel = 1.0e-03,
//           eps_prim_inf = 1.0e-04, eps_dual_inf = 1.0e-04,
//           rho = 1.00e-01 (adaptive),
//           sigma = 1.00e-06, alpha = 1.60, max_iter = 4000
//           check_termination: on (interval 25),
//           scaling: on, scaled_termination: off
//           warm start: on, polish: off, time_limit: off

// iter   objective    pri res    dua res    rho        time
//    1  -3.5142e+01   7.71e+00   1.70e+01   1.00e-01   6.72e-05s
//   50   2.0056e+02   3.91e-03   3.97e-03   1.00e-01   1.15e-04s

// status:               solved
// number of iterations: 50
// optimal objective:    200.5642
// run time:             1.26e-04s
// optimal rho estimate: 1.72e-01

// optimal sol: 4.09913 9.14661 4.49725
// optimal obj: 200.564
// elapsed time: 0.00010869s