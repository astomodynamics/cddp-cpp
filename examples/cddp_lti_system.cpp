#include <iostream>
#include <vector>
#include <filesystem>
#include <random>
#include "cddp.hpp"

#include "matplot/matplot.h"
using namespace matplot;
namespace fs = std::filesystem;

namespace cddp {

/**
 * @brief Quadratic cost objective for LTI system
 */
class LTIObjective : public QuadraticObjective {
public:
    LTIObjective(int state_dim, 
                 int control_dim, 
                 const Eigen::VectorXd& goal_state,
                 double timestep) 
        : QuadraticObjective(
            0.5 * timestep * Eigen::MatrixXd::Identity(state_dim, state_dim),  // Q
            0.5 * 0.1 * timestep * Eigen::MatrixXd::Identity(control_dim, control_dim), // R
            0.5 * timestep * Eigen::MatrixXd::Identity(state_dim, state_dim),  // Qf 
            goal_state) {}
};

} // namespace cddp

template<typename T>
void printVector(const std::string& label, const std::vector<T>& vec) {
    std::cout << label << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << vec[i];
    }
    std::cout << "]\n";
}

void plotStateTrajectories(const std::vector<Eigen::VectorXd>& X_sol, 
                             const std::string& plotDirectory) {
    const int state_dim = X_sol[0].size();
    std::vector<std::vector<double>> state_histories(state_dim);
    std::vector<double> time_history;

    // Extract state histories (using the time index as sample number)
    for (size_t t = 0; t < X_sol.size(); t++) {
        time_history.push_back(static_cast<double>(t));
        for (int i = 0; i < state_dim; i++) {
            state_histories[i].push_back(X_sol[t](i));
        }
    }

    // Create a new figure and set its size
    auto fig = figure(true);
    fig->size(1200, 800);

    // Create a subplot for each state variable
    for (int i = 0; i < state_dim; i++) {
        // Create subplot: (rows, columns, index)
        subplot(state_dim, 1, i + 1);
        plot(time_history, state_histories[i])->line_width(2);
        title("State " + std::to_string(i + 1));
        grid(on);
    }
    
    save(fig, plotDirectory + "/lti_states.png");
}

int main() {
    // Problem parameters
    int state_dim = 4;    // State dimension
    int control_dim = 2;  // Control dimension
    int horizon = 1000;   // Time horizon
    double timestep = 0.01;  // Time step
    std::string integration_type = "euler";

    // Create LTI system instance
    std::unique_ptr<cddp::DynamicalSystem> system = 
        std::make_unique<cddp::LTISystem>(state_dim, control_dim, timestep, integration_type);

    // Initial and goal states
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state << 0.8378, 0.3794, 1.4796, 0.2382;
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);

    // Create cost objective
    Eigen::MatrixXd Q = 0.5 * Eigen::MatrixXd::Identity(state_dim, state_dim); // Q
    Eigen::MatrixXd R = 0.5 * 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim); // R
    Eigen::MatrixXd Qf = 0.5 * timestep * Eigen::MatrixXd::Identity(state_dim, state_dim);  // Qf 
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints
    Eigen::VectorXd control_lower_bound = -0.6 * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_upper_bound = 0.6 * Eigen::VectorXd::Ones(control_dim);

    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(
            control_lower_bound, control_upper_bound));

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 10;
    options.verbose = true;
    options.regularization_type = "control";
    options.num_threads = 11;
    options.use_parallel = true;
    options.debug = false;
    cddp_solver.setOptions(options);

    // Initialize trajectories
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Random initial controls (here, a constant small control is used)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.1);
    for (auto& u : U) {
        u = 0.01 * Eigen::VectorXd::Ones(control_dim);
    }

    X[0] = initial_state;

    // Simulate initial trajectory
    double J = 0.0;
    double cost = 0.0;
    for (size_t t = 0; t < horizon; t++) {
        cost = cddp_solver.getObjective().running_cost(X[t], U[t], t);
        J += cost;
        X[t + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[t], U[t]);
    }
    J += cddp_solver.getObjective().terminal_cost(X.back());
    std::cout << "Initial state: " << X[0].transpose() << std::endl;
    std::cout << "Final state: " << X.back().transpose() << std::endl;
    std::cout << "Initial cost: " << J << std::endl;
    
    cddp_solver.setInitialTrajectory(X, U);

    // Solve using the CDDP solver
    cddp::CDDPSolution solution = cddp_solver.solve();
    // Alternatively: cddp::CDDPSolution solution = cddp_solver.solveLogCDDP();

    // Extract solution trajectories
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create directory for plots if it doesn't exist
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Plot state trajectories using the matplot API
    plotStateTrajectories(X_sol, plotDirectory);

    // Print optimization statistics
    std::cout << "\nOptimization Results:" << std::endl;
    std::cout << "Iterations: " << solution.iterations << std::endl;
    std::cout << "Final cost: " << solution.cost_sequence.back() << std::endl;
    std::cout << "Converged: " << (solution.converged ? "Yes" : "No") << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " microseconds" << std::endl;

    return 0;
}
