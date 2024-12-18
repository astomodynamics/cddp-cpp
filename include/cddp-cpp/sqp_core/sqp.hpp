/*
 * SQP Solver using OSQP for Constrained Optimization Problems
 */
#ifndef CDDP_SQP_SOLVER_HPP
#define CDDP_SQP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>
#include "osqp++.h"
#include "cddp_core/dynamical_system.hpp"
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"

namespace cddp {

/**
 * @brief Configuration options for SQP solver
 */
struct SQPOptions {
    int max_iterations = 100;           // Maximum number of iterations
    double ftol = 1e-6;                // Function value tolerance
    double xtol = 1e-6;                // Step size tolerance
    double gtol = 1e-6;                // Gradient tolerance
    double eta = 0.1;                  // Merit function parameter
    double tau = 0.5;                  // Line search parameter
    bool verbose = false;              // Print debug info
    
    // OSQP specific options
    double osqp_eps_abs = 1e-5;        // Absolute tolerance
    double osqp_eps_rel = 1e-3;        // Relative tolerance
    int osqp_max_iter = 4000;          // Maximum OSQP iterations
    bool osqp_verbose = false;         // OSQP verbosity
    bool warm_start = true;            // Use warm starting
};

/**
 * @brief Results from SQP optimization
 */
struct SQPResult {
    bool success;                      // Whether optimization succeeded
    int iterations;                    // Number of iterations taken
    double objective_value;            // Final objective value
    double constraint_violation;       // Final constraint violation
    Eigen::VectorXd solution;          // Optimal solution vector
    std::vector<double> obj_history;   // History of objective values
    std::vector<double> viol_history;  // History of constraint violations
};

/**
 * @brief Sequential Quadratic Programming solver using OSQP
 */
class SQPSolver {
public:
    /**
     * @brief Constructor
     * @param options Solver configuration options
     */
    explicit SQPSolver(const SQPOptions& options = SQPOptions());

    /**
     * @brief Set the optimization problem
     * @param system Dynamical system
     * @param objective Objective function
     * @param constraints Vector of constraints
     */
    void setProblem(std::shared_ptr<DynamicalSystem> system,
                   std::shared_ptr<Objective> objective,
                   const std::vector<std::shared_ptr<Constraint>>& constraints);

    /**
     * @brief Solve the optimization problem
     * @param x0 Initial guess
     * @return Solution results
     */
    SQPResult solve(const Eigen::VectorXd& x0);

private:
    // Helper methods
    void formQPSubproblem(const Eigen::VectorXd& x,
                         double f,
                         const Eigen::VectorXd& grad_f,
                         const std::vector<double>& c,
                         const std::vector<Eigen::MatrixXd>& jac_c,
                         Eigen::SparseMatrix<double>& H,
                         Eigen::VectorXd& g,
                         Eigen::SparseMatrix<double>& A,
                         Eigen::VectorXd& l,
                         Eigen::VectorXd& u);
    
    void computeHessianApproximation(const Eigen::VectorXd& x,
                                   std::vector<Eigen::Triplet<double>>& H_triplets);
    
    double computeConstraintViolation(const std::vector<double>& c) const;
    
    double lineSearch(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& dx,
                     double f0,
                     const Eigen::VectorXd& grad_f,
                     const std::vector<double>& c0);

    double evaluateObjective(const Eigen::VectorXd& x) const;
    Eigen::VectorXd evaluateGradient(const Eigen::VectorXd& x) const;
    void evaluateConstraints(const Eigen::VectorXd& x,
                           std::vector<double>& c,
                           std::vector<Eigen::MatrixXd>& jac_c) const;

    // Member variables
    SQPOptions options_;
    osqp::OsqpSettings osqp_settings_;
    
    std::shared_ptr<DynamicalSystem> system_;
    std::shared_ptr<Objective> objective_;
    std::vector<std::shared_ptr<Constraint>> constraints_;
};

} // namespace cddp

#endif // CDDP_SQP_SOLVER_HPP