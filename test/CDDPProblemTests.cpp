#include <iostream> 
#include <Eigen/Dense>
#include "cddp_core/CDDPProblem.hpp"
#include "model/DubinsCar.hpp"

using namespace cddp;

// Simple Test Function
bool testBasicCDDP() {
    int state_dim = 3; 
    int control_dim = 2; 
    double dt = 0.05;
    int horizon = 100;
    int integration_type = 0; // 0 for Euler, 1 for Heun, 2 for RK3, 3 for RK4

    // Problem Setup
    Eigen::VectorXd initialState(state_dim);
    initialState << 0.0, 0.0, M_PI/4; // Initial state

    DubinsCar system(state_dim, control_dim, dt, integration_type); // Your DoubleIntegrator instance
    CDDPProblem cddp_solver(&system, initialState, horizon, dt);
    
    // Set goal state if needed
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/4;
    cddp_solver.setGoalState(goal_state);

    // Simple Cost Matrices 
    Eigen::MatrixXd Q(state_dim, state_dim);
    Q << 0e-1, 0, 0, 
         0, 0e-1, 0, 
         0, 0, 0e-3;
    Eigen::MatrixXd R(control_dim, control_dim);
    R << 1e+0, 0, 
         0, 1e+0; 
    Eigen::MatrixXd Qf(state_dim, state_dim);
    Qf << 50, 0, 0, 
          0, 50, 0, 
          0, 0, 10; 
    QuadraticCost objective(Q, R, Qf, goal_state, dt);
    cddp_solver.setObjective(std::make_unique<QuadraticCost>(objective));

    CDDPOptions opts;
    // Set options if needed
    opts.max_iterations = 1;
    // opts.cost_tolerance = 1e-6;
    // opts.grad_tolerance = 1e-8;
    // opts.print_iterations = false;
    cddp_solver.setOptions(opts);



    // Set initial trajectory if needed
    std::vector<Eigen::VectorXd> X = std::vector<Eigen::VectorXd>(horizon + 1, initialState);
    std::vector<Eigen::VectorXd> U = std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(state_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve!
    std::vector<Eigen::VectorXd> U_sol = cddp_solver.solve();

    // std::cout << "Optimal control (first step):\n" << optimal_control_seq[0] << std::endl;

    return true; // Replace with proper success/failure logic based on your assertions
}

int main() {
    if (testBasicCDDP()) {
        std::cout << "Basic CDDP Test Passed!" << std::endl;
    } else {
        std::cout << "Basic CDDP Test Failed!" << std::endl;
        return 1; // Indicate failure 
    }
    return 0;
}
