// Description: Test the CDDP solver.
#include <iostream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

TEST(CDDPTest, Solve) {
    // Problem parameters
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 30;
    double timestep = 0.1;

    // Create a pendulum instance (no device needed for Eigen)
    double mass = 1.0; 
    double length = 1.0; 
    double gravity = 9.81;
    cddp::Pendulum pendulum(mass, length, gravity, timestep); 

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, Eigen::VectorXd::Zero(0, 0), timestep);

    // Initial and target states
    // Eigen::VectorXd initial_state = Eigen::VectorXd::Ones(state_dim);
    // Eigen::VectorXd target_state = Eigen::VectorXd::Zero(state_dim); 

    // // Create CDDP solver
    // cddp::CDDP cddp_solver(initial_state, target_state, horizon, timestep);
    // cddp_solver.setDynamicalSystem(std::move(system));
    // cddp_solver.setObjective(std::move(objective));

    // // Solve the problem
    // cddp::CDDPSolution solution = cddp_solver.solve();

    // // Assertions
    // ASSERT_TRUE(solution.converged); // Check if the solver converged
    // // Add more assertions based on expected behavior

}