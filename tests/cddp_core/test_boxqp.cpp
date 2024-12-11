#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>

#include "cddp_core/helper.hpp"
#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"

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
}
