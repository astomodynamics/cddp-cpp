#include <iostream> 
#include <Eigen/Dense>
#include "cddp_core/CDDPProblem.hpp"
#include "model/DoubleIntegrator.hpp"  

using namespace cddp;

// Simple Test Function
bool testBasicCDDP() {
    int state_dim = 4; 
    int control_dim = 2; 
    double dt = 0.1;
    int horizon = 10;
    int integration_type = 0; // 0 for Euler, 1 for Heun, 2 for RK3, 3 for RK4

    // Problem Setup
    Eigen::VectorXd initialState(state_dim);
    initialState << 1, 0, 0.5, -0.1; // Initial state

    DoubleIntegrator system(state_dim, control_dim, dt, integration_type); // Your DoubleIntegrator instance
    CDDPProblem cddp_solver(&system, initialState, horizon, dt);
    

    // Simple Cost Matrices 
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd R =  0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim); 
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim); 
    // cddp_solver.setCostMatrices(Q, R, Qf);

    // QuadraticCost objective(Q, R, Qf, goal_state);

    CDDPOptions opts;
    // Set options if needed
    // opts.max_iterations = 100;
    // opts.cost_tolerance = 1e-6;
    // opts.grad_tolerance = 1e-8;
    // opts.print_iterations = false;
    // cddp_solver.setOptions(opts);

    // Set goal state if needed
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0, 0, 0, 0;
    cddp_solver.setGoalState(goal_state);

    QuadraticCost objective(Q, R, Qf, goal_state);

    // Set initial state if needed
    // cddp_solver.setInitialState(initialState);

    // Set horizon if needed
    // cddp_solver.setHorizon(horizon);
    
    // Set time step if needed
    // cddp_solver.setTimeStep(dt);

    


    // Set initial trajectory if needed
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(state_dim, horizon + 1);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(control_dim, horizon);
    cddp_solver.setInitialTrajectory(X, U);

    // Solve!
    // Eigen::MatrixXd optimal_control_seq = cddp_solver.solve();

    // (Add assertions: Check if the returned sequence has the correct length, etc.)

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
