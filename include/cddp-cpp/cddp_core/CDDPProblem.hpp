#ifndef CDDP_CDDPPROBLEM_HPP
#define CDDP_CDDPPROBLEM_HPP

#include <iostream>
#include <memory>
#include <vector>
#include "Eigen/Dense"
#include "DynamicalSystem.hpp" 
#include "Objective.hpp" 
#include "Constraint.hpp" 
// #include "cddp/cddp_core/Helper.hpp"

namespace cddp {

struct CDDPOptions {
    double cost_tolerance = 1e-4;         // Tolerance for changes in cost function
    double grad_tolerance = 1e-6;         // Tolerance for cost gradient magnitude
    int max_iterations = 100; 
    int max_line_search_iterations = 11; // Maximum iterations for line search
    double backtracking_coeff = 1.0;      // Maximum step size for line search backtracking
    double backtracking_min = 0.5;    // Coefficient for line search backtracking
    double backtracking_factor = std::pow(2, -2);  // Factor for line search backtracking

    // Active Set Method Options
    int active_set_max_iterations = 100;  // Maximum iterations for active set method
    double active_set_tolerance = 1e-6;   // Tolerance for active set
    double active_set_coeff = 1.0;        // Coefficient for active set method
    double trust_region_radius = 1.0;      // Coefficient for trust region method
    double trust_region_factor = 0.90;      // Coefficient for trust region method
    
    // Line search method
    // int line_search_type = 0;             // 0 for Armijo, 1 for Wolfe
    double line_search_coeff = 1e-0;      // Coefficient for line search
    double line_search_factor = 0.5;      // Factor for line search
    int line_search_max_iterations = 11; // Maximum iterations for line search


    int regularization_type = 0;          // 0 or 1 for different regularization types
    double regularization_x = 1e-6;       // Regularization for state
    double regularization_u = 1e-6;       // Regularization for control
    double regularization_tolerance = 1e-6; // Tolerance for regularization
    double regularization_factor = 10.0;  // Factor for regularization
    double regularization_max = 1e6;      // Maximum regularization
    double regularization_min = 1e-6;     // Minimum regularization
    bool print_iterations = true;         // Option for debug printing 
    bool is_ilqr = false;                  // Option for iLQR
};

class CDDPProblem {
public:
    CDDPProblem(const Eigen::VectorXd& initial_state, const Eigen::VectorXd& goal_state, int horizon, double dt);

    // Problem Setup
    void setInitialState(const Eigen::VectorXd& x0);
    void setGoalState(const Eigen::VectorXd& goal); 
    void setHorizon(int T);
    void setTimeStep(double dt);
    void setCostMatrices(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf); 
    void setCostFunction(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf);
    void setOptions(const CDDPOptions& opts);
    void setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U);
    void initializeCost();
    void setDynamicalSystem(std::unique_ptr<DynamicalSystem> dynamics);
    void setObjective(std::unique_ptr<Objective> objective);
    void addConstraint(std::unique_ptr<Constraint> constraint);
    Eigen::VectorXd getInitialState() { return initial_state_; }
    Eigen::VectorXd getGoalState() { return goal_state_; }
    int getHorizon() { return horizon_; }
    double getTimeStep() { return dt_; }
    std::vector<Eigen::VectorXd> getTrajectory() { return X_; }
    std::vector<Eigen::VectorXd> getControlSequence() { return U_; }

    // Solver methods
    std::vector<Eigen::VectorXd> solve(); 
    bool solveForwardPass();
    bool solveBackwardPass();

    // update methods
    // void updateControl();
    // void updateTrajectory();
    // void updateValueFunction();
    // void updateCost();
    // void updateQFunction();
    // void updateValueFunctionDerivative();
    // void updateValueFunctionHessian();
    // void updateCostChange();
    // void updateGradientNorm();
    // void updateLambda();
    // void updateActiveSet();

    // updateControl();

    // Getters
    // Eigen::MatrixXd getTrajectory() { return X; }
    // Eigen::MatrixXd getControlSequence() { return U; }
    // double getCost() { return J; }
    // std::vector<double> getValueFunction() { return V; }
    // std::vector<Eigen::VectorXd> getValueFunctionDerivative() { return V_X; }
    // std::vector<Eigen::MatrixXd> getValueFunctionHessian() { return V_XX; }

private:
    // Problem Data
    std::unique_ptr<DynamicalSystem> dynamics_;
    Eigen::VectorXd initial_state_;
    Eigen::VectorXd goal_state_;
    int horizon_;
    double dt_;  // Time step

    // Intermediate trajectories
    std::vector<Eigen::VectorXd> X_;
    std::vector<Eigen::VectorXd> U_;

    // Intermediate cost
    double J_;

    // Feedforward and feedback gains
    std::vector<Eigen::VectorXd> k_;
    std::vector<Eigen::MatrixXd> K_;

    // Intermediate value function
    std::vector<double> V_;
    std::vector<Eigen::VectorXd> V_X_;
    std::vector<Eigen::MatrixXd> V_XX_;

    // Q-function matrices
    std::vector<Eigen::MatrixXd> Q_UU_;
    std::vector<Eigen::MatrixXd> Q_UX_;
    std::vector<Eigen::VectorXd> Q_U_; 

    // Helper methods
    // void printIteration(int iter, double cost, double grad_norm, double lambda);

    // Options
    CDDPOptions options_;

    std::unique_ptr<Objective> objective_; // Store a cost function
    std::vector<std::unique_ptr<Constraint>> constraint_set_; // Store multiple constraints
};


} // namespace cddp
#endif // CDDP_CDDPPROBLEM_HPP