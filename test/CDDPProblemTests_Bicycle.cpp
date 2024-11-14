#include <iostream> 
#include "Eigen/Dense"
// #include "cddp/cddp_core/CDDPProblem.hpp"
#include "CDDP.hpp"
#include "model/DubinsCar.hpp"
#include "matplotlibcpp.hpp"


#include <iostream> 
#include "Eigen/Dense"
// #include "cddp/cddp_core/CDDPProblem.hpp"
#include "CDDP.hpp"
#include "model/Bicycle.hpp"
#include "matplotlibcpp.hpp"

using namespace cddp;
namespace plt = matplotlibcpp;
// Simple Test Function
bool testBasicCDDP() {
    const int STATE_DIM = 4; 
    const int CONTROL_DIM = 2; 
    const double TIMESTEP = 0.1;
    const int HORIZON = 100;
    const double WHEEL_BASE = 1.5;
    const int INTEGRATION_TYPE = 0; // 0 for Euler, 1 for Heun, 2 for RK3, 3 for RK4

    // Problem Setup
    Eigen::VectorXd initial_state(STATE_DIM);
    initial_state << 0.0, 0.0, M_PI/4.0, 0.0; // Initial state

    // Set goal state
    Eigen::VectorXd goal_state(STATE_DIM);
    goal_state << 5.0, 5.0, M_PI/2.0, 0.0;

    Bicycle system(STATE_DIM, CONTROL_DIM, TIMESTEP, WHEEL_BASE, INTEGRATION_TYPE); 
    CDDPProblem cddp_solver(initial_state, goal_state, HORIZON, TIMESTEP);

    // Set dynamical system
    cddp_solver.setDynamicalSystem(std::make_unique<Bicycle>(system));
    
   
    // Simple Cost Matrices 
    Eigen::MatrixXd Q(STATE_DIM, STATE_DIM);
    Q << 0e-2, 0, 0, 0.0, 
         0, 0e-2, 0, 0.0,
         0, 0, 0e-3, 0.0,
            0, 0, 0, 0.0;

    Eigen::MatrixXd R(CONTROL_DIM, CONTROL_DIM);
    R << 1e+0, 0, 
         0, 1e+0; 

    Eigen::MatrixXd Qf(STATE_DIM, STATE_DIM);
    Qf << 50, 0, 0, 0.0,
          0, 50, 0, 0.0,
          0, 0, 10, 0.0,
            0, 0, 0, 10.0;

    QuadraticCost objective(Q, R, Qf, goal_state, TIMESTEP);
    cddp_solver.setObjective(std::make_unique<QuadraticCost>(objective));

    // Add constraints 
    Eigen::VectorXd lower_bound(CONTROL_DIM);
    lower_bound << -10.0, -M_PI/3;

    Eigen::VectorXd upper_bound(CONTROL_DIM);
    upper_bound << 10.0, M_PI/3;

    ControlBoxConstraint control_constraint(lower_bound, upper_bound);
    cddp_solver.addConstraint(std::make_unique<ControlBoxConstraint>(control_constraint));

    CDDPOptions opts;
    // // Set options if needed
    opts.max_iterations = 20;
    // opts.cost_tolerance = 1e-6;
    // opts.grad_tolerance = 1e-8;
    // opts.print_iterations = false;
    cddp_solver.setOptions(opts);


    // Set initial trajectory if needed
    std::vector<Eigen::VectorXd> X = std::vector<Eigen::VectorXd>(HORIZON + 1, Eigen::VectorXd::Zero(STATE_DIM));
    std::vector<Eigen::VectorXd> U = std::vector<Eigen::VectorXd>(HORIZON, Eigen::VectorXd::Zero(CONTROL_DIM));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve!
    std::vector<Eigen::VectorXd> U_sol = cddp_solver.solve();

    std::vector<Eigen::VectorXd> X_sol = cddp_solver.getTrajectory();

    // Print solution
    std::cout << "Solution: "<< std::endl;
    // print last state
    std::cout << "Final State: " << X_sol.back().transpose() << std::endl;
    // print initial control
    std::cout << "Initial Control: " << U_sol[0].transpose() << std::endl;

    // Plotting
    std::vector<double> x, y, theta, v;
    for (int i = 0; i < X_sol.size(); i++) {
        x.push_back(X_sol[i](0));
        y.push_back(X_sol[i](1));
        theta.push_back(X_sol[i](2));
        v.push_back(X_sol[i](3));
    }

    plt::figure();
    plt::plot(x, y);
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Bicycle model car Trajectory");
    plt::show();

    // for (int i = 0; i < U_sol.size(); i++) {
    //     std::cout << "Control " << i << ": " << U_sol[i].transpose() << std::endl;
    // }

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
