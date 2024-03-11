#ifndef CDDP_CDDPPROBLEM_HPP
#define CDDP_CDDPPROBLEM_HPP

#include <Eigen/Dense>
#include <vector>

#include "model/DynamicalSystem.hpp" 

namespace cddp {

struct CDDPOptions {
    double cost_tolerance = 1e-4;         // Tolerance for changes in cost function
    double grad_tolerance = 1e-6;         // Tolerance for cost gradient magnitude
    int max_iterations = 100; 
    int max_line_search_iterations = 11; // Maximum iterations for line search
    double backtracking_coeff = 1.0;      // Maximum step size for line search backtracking
    double backtracking_min = 0.5;    // Coefficient for line search backtracking
    double backtracking_factor = std::pow(10, -3);  // Factor for line search backtracking

    // Active Set Method Options
    int active_set_max_iterations = 100;  // Maximum iterations for active set method
    double active_set_tolerance = 1e-6;   // Tolerance for active set
    double initial_lambda = 1.0;          // Initial regularization parameter
    double initial_dlambda = 1.0;         // Initial step for lambda update
    double lambda_scaling_factor = 10.0;  // Factor for scaling lambda up or down
    double max_lambda = 1e10;             // Upper bound for lambda 
    double min_lambda = 1e-6;             // Lower bound for lambda
    int regularization_type = 0;          // 0 or 1 for different regularization types
    bool print_iterations = true;         // Option for debug printing 
};

class CDDPProblem {
public:
    CDDPProblem(DynamicalSystem* system, const Eigen::VectorXd& initialState, int horizon, double dt);

    // Problem Setup
    void setInitialState(const Eigen::VectorXd& x0);
    void setGoalState(const Eigen::VectorXd& goal); 
    void setHorizon(int T);
    void setTimeStep(double dt);
    void setCostMatrices(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf); 
    void setOptions(const CDDPOptions& opts);
    void setInitialTrajectory(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U);

    // Solver methods
    Eigen::MatrixXd solve(); 
    bool solveForwardPass();

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
    DynamicalSystem* dynamics;
    Eigen::VectorXd initial_state;
    Eigen::VectorXd goal_state;
    int horizon;
    double dt;  // Time step
    
    // Cost Matrices
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Qf;

    // Intermediate trajectories
    Eigen::MatrixXd X;
    Eigen::MatrixXd U;

    // Intermediate cost
    double J;

    // Intermediate value function
    std::vector<double> V;
    std::vector<Eigen::VectorXd> V_X;
    std::vector<Eigen::MatrixXd> V_XX;

    // Q-function matrices
    std::vector<Eigen::MatrixXd> Q_UU;
    std::vector<Eigen::MatrixXd> Q_UX;
    std::vector<Eigen::VectorXd> Q_U; 

    // Helper methods
    // void forwardPass();
    // void backwardPass();
    // void updateControlSequence();
    // void updateTrajectory();
    // void updateValueFunction();
    // double calculateCost();
    // double calculateFinalCost();
    // double calculateCostChange();
    // double calculateGradientNorm();
    // void printIteration(int iter, double cost, double grad_norm, double lambda);

    // Options
    CDDPOptions options;
};


} // namespace cddp
#endif // CDDP_CDDPPROBLEM_HPP