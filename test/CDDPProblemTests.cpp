#include <iostream> 
#include "Eigen/Dense"
#include "cddp/cddp_core/CDDPProblem.hpp"
#include "cddp/model/DubinsCar.hpp"

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
    initialState << 0.0, 0.0, 0.0; // Initial state

    DubinsCar system(state_dim, control_dim, dt, integration_type); // Your DoubleIntegrator instance
    CDDPProblem cddp_solver(&system, initialState, horizon, dt);
    
    // Set goal state if needed
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 2.0, 2.0, M_PI/2.0;
    cddp_solver.setGoalState(goal_state);

    // Simple Cost Matrices 
    Eigen::MatrixXd Q(state_dim, state_dim);
    Q << 0e-2, 0, 0, 
         0, 0e-2, 0, 
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

    // Add constraints 
    Eigen::VectorXd lower_bound(control_dim);
    lower_bound << -0.1, -M_PI;

    Eigen::VectorXd upper_bound(control_dim);
    upper_bound << 0.1, M_PI;

    ControlBoxConstraint control_constraint(lower_bound, upper_bound);
    cddp_solver.addConstraint(std::make_unique<ControlBoxConstraint>(control_constraint));

    CDDPOptions opts;
    // Set options if needed
    opts.max_iterations = 20;
    // opts.cost_tolerance = 1e-6;
    // opts.grad_tolerance = 1e-8;
    // opts.print_iterations = false;

    cddp_solver.setOptions(opts);


    // Set initial trajectory if needed
    std::vector<Eigen::VectorXd> X = std::vector<Eigen::VectorXd>(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U = std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(control_dim));
    // X.front() = initialState;
    cddp_solver.setInitialTrajectory(X, U);

    // Solve!
    std::vector<Eigen::VectorXd> U_sol = cddp_solver.solve();

    std::vector<Eigen::VectorXd> X_sol = cddp_solver.getTrajectory();

    // Print solution
    std::cout << "Solution: " << std::endl;
    // print last state
    std::cout << "Final State: " << X_sol.back().transpose() << std::endl;
    // print initial control
    std::cout << "Initial Control: " << U_sol[0].transpose() << std::endl;

    for (int i = 0; i < U_sol.size(); i++) {
        std::cout << "Control " << i << ": " << U_sol[i].transpose() << std::endl;
    }

    return true; 
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
