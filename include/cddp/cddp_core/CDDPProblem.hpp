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
    double backtracking_range = 0.5;      // Range for line search backtracking (0, 1]
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

    // Solver methods
    std::vector<Eigen::VectorXd> solve(); 

    
    

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

    // Value function matrices
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