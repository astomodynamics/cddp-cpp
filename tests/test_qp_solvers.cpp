#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>

#include "cddp_core/boxqp.hpp"
#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"
#include "gurobi_c++.h"

using namespace std;
using namespace Eigen;
using namespace cddp;

#include <iostream>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <chrono>

#include "cddp_core/qp_solver.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp-cpp/sdqp.hpp"
#include "osqp++.h"

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
    // 1. Test OSQP
    {
        // Convert to sparse matrices
        SparseMatrix<double> P(n, n);
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

        osqp::OsqpSolver osqp_solver;
        osqp::OsqpInstance instance;
        osqp::OsqpSettings settings;

        instance.objective_matrix = P;
        instance.objective_vector = q;
        instance.constraint_matrix = G;
        instance.upper_bounds = ub;
        instance.lower_bounds = lb;

        settings.verbose = false;

        osqp_solver.Init(instance, settings);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto status = osqp_solver.Solve();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        printResults("OSQP", osqp_solver.primal_solution(), 
                    osqp_solver.objective_value(), elapsed.count(), 
                    static_cast<int>(status));
    }

    // 2. Test SDQP
    {
        VectorXd x(n);

        auto start_time = std::chrono::high_resolution_clock::now();
        double minobj = sdqp::sdqp(Q, q, A, b, x);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        printResults("SDQP", x, minobj, elapsed.count(), 0);
    }

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

    // 4. Test Gurobi
    {   
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
            if (q(i) != 0.0) {
                obj += q(i) * x[i];
            }
        }
        model.setObjective(obj, GRB_MINIMIZE);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        auto start_time = std::chrono::high_resolution_clock::now();
        model.optimize();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        printResults("Gurobi", VectorXd::Zero(n), model.get(GRB_DoubleAttr_ObjVal), 
                    elapsed.count(), model.get(GRB_IntAttr_Status));
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
