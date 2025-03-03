/*
 * Example code demonstrating the pendulum model with IPDDP
 */
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "cddp.hpp"


int main() {
    // Problem parameters for the pendulum
    const int state_dim = 2;    // [theta, theta_dot]
    const int control_dim = 1;  // [torque]
    const int horizon = 500;
    const double timestep = 0.05;
    const std::string integration_type = "euler";

    // Random number generator for potential random initialization (if desired)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.1);

    // Create a pendulum instance with chosen parameters: length, mass, damping
    double length = 1.0;
    double mass = 1.0;
    double damping = 0.0;
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Pendulum>(timestep, length, mass, damping, integration_type);

    // Set initial and goal states for the swing-up task.
    // Initial state: near the downward equilibrium.
    Eigen::VectorXd initial_state(state_dim);
    initial_state << -M_PI, 0.0;

    // Goal state: upright position (swing up) with zero angular velocity.
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0;

    // Create the quadratic objective for the pendulum swing-up
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.025 / timestep * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 5.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Create the CDDP solver instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Set control constraints
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 0.25;
    cddp_solver.addConstraint(std::string("ControlConstraint"),
                              std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Set solver options
    cddp::CDDPOptions options;
    options.max_iterations = 500;
    options.verbose = true;
    options.cost_tolerance = 1e-7;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "none";
    // options.regularization_control = 1.0; // Uncomment and adjust if needed
    options.debug = true;
    options.use_parallel = false;
    options.num_threads = 1;
    options.barrier_coeff = 1e-1;
    cddp_solver.setOptions(options);

    // Initialize trajectories: states X and controls U
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));

    // Optionally, add random perturbations to the initial controls
    for (auto& u : U) {
        // u(0) = d(gen);
        u << 0.01;
    }
    X[0] = initial_state;

    double J = 0.0;
    // Simulate the initial trajectory
    for (size_t t = 0; t < horizon; t++) {
        J += cddp_solver.getObjective().running_cost(X[t], U[t], t);
        X[t + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[t], U[t]);
    }
    J += cddp_solver.getObjective().terminal_cost(X.back());
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Initial cost: " << J << std::endl;
    std::cout << "Initial state: " << X[0].transpose() << std::endl;
    std::cout << "Final state: " << X.back().transpose() << std::endl;

    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem using IPDDP
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Extract the solution trajectories:
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    std::cout << "Optimized final state: " << X_sol.back().transpose() << std::endl;

    return 0;
}
