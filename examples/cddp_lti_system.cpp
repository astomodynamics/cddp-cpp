/*
 Copyright 2024 Tomo Sasaki

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <iostream>
#include <vector>
#include <filesystem>
#include <random>
#include "cddp.hpp"

namespace plt = matplotlibcpp;
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
            timestep * Eigen::MatrixXd::Identity(state_dim, state_dim),  // Q
            0.1 * timestep * Eigen::MatrixXd::Identity(control_dim, control_dim), // R
            Eigen::MatrixXd::Identity(state_dim, state_dim),  // Qf 
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

    // Extract state histories
    for(size_t t = 0; t < X_sol.size(); t++) {
        time_history.push_back(t);
        for(int i = 0; i < state_dim; i++) {
            state_histories[i].push_back(X_sol[t](i));
        }
    }

    // Create subplot for each state dimension
    plt::figure_size(1200, 800);
    for(int i = 0; i < state_dim; i++) {
        plt::subplot(state_dim, 1, i+1);
        plt::plot(time_history, state_histories[i]);
        plt::title("State " + std::to_string(i+1));
        plt::grid(true);
    }
    
    plt::tight_layout();
    plt::save(plotDirectory + "/lti_states.png");
}

int main() {
    // Problem parameters
    int state_dim = 10;    // State dimension
    int control_dim = 2;   // Control dimension
    int horizon = 1000;    // Time horizon
    double timestep = 0.01;  // Time step
    std::string integration_type = "euler";

    // Create LTI system instance
    std::unique_ptr<cddp::DynamicalSystem> system = 
        std::make_unique<cddp::LTISystem>(state_dim, control_dim, timestep, integration_type);

    // Initial and goal states
    Eigen::VectorXd initial_state = Eigen::VectorXd::Random(state_dim);
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);

    // Create cost objective
    auto objective = std::make_unique<cddp::LTIObjective>(
        state_dim, control_dim, goal_state, timestep);

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
    options.max_iterations = 50;
    options.verbose = true;
    cddp_solver.setOptions(options);

    // Initialize trajectories
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Random initial controls
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.1);
    for(auto& u : U) {
        u = 0.1 * Eigen::VectorXd::Random(control_dim);
    }

    X[0] = initial_state;

    // Simulate initial trajectory
    double J = 0.0;
    for(size_t t = 0; t < horizon; t++) {
        J += cddp_solver.getObjective().running_cost(X[t], U[t], t);
        X[t + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[t], U[t]);
    }
    J += cddp_solver.getObjective().terminal_cost(X.back());
    std::cout << "Initial cost: " << J << std::endl;
    std::cout << "Initial state: " << X[0].transpose() << std::endl;
    std::cout << "Final state: " << X.back().transpose() << std::endl;
    
    cddp_solver.setInitialTrajectory(X, U);

    // Solve using different CDDP variants
    // cddp::CDDPSolution solution = cddp_solver.solve();
    cddp::CDDPSolution solution = cddp_solver.solveCLDDP();
    // cddp::CDDPSolution solution = cddp_solver.solveLogCDDP();

    // Extract solution trajectories
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create directory for plots
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Plot state trajectories
    plotStateTrajectories(X_sol, plotDirectory);

    // Print statistics
    std::cout << "\nOptimization Results:" << std::endl;
    std::cout << "Iterations: " << solution.iterations << std::endl;
    std::cout << "Final cost: " << solution.cost_sequence.back() << std::endl;
    std::cout << "Converged: " << (solution.converged ? "Yes" : "No") << std::endl;
    std::cout << "Solve time: " << solution.solve_time << " microseconds" << std::endl;

    return 0;
}