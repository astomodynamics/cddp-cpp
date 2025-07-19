/*
 Example demonstrating the three barrier update strategies for IPDDP and MSIPDDP
*/

#include <iostream>
#include <memory>
#include <string>
#include <Eigen/Dense>

#include "cddp.hpp"

int main() 
{
    // Problem setup
    const int horizon = 50;
    const double timestep = 0.05;
    const int state_dim = 3;
    const int control_dim = 2;
    
    // Initial and goal states
    Eigen::VectorXd X0(state_dim);
    X0 << 0.0, 0.0, 0.0;  // x, y, theta
    
    Eigen::VectorXd Xg(state_dim);
    Xg << 2.0, 2.0, 0.0;
    
    // Test each barrier strategy
    std::vector<std::string> solvers = {"IPDDP", "MSIPDDP"};
    std::vector<cddp::BarrierStrategy> strategies = {
        cddp::BarrierStrategy::ADAPTIVE,
        cddp::BarrierStrategy::MONOTONIC,
        cddp::BarrierStrategy::IPOPT
    };
    std::vector<std::string> strategy_names = {"ADAPTIVE", "MONOTONIC", "IPOPT"};
    
    for (const auto& solver_name : solvers) 
    {
        std::cout << "\n========================================\n";
        std::cout << "Testing " << solver_name << " Solver\n";
        std::cout << "========================================\n";
        
        for (size_t i = 0; i < strategies.size(); ++i) 
        {
            std::cout << "\n--- Barrier Strategy: " << strategy_names[i] << " ---\n";
            
            // Create CDDP solver first
            cddp::CDDP cddp_solver(X0, Xg, horizon, timestep);
            
            // Create and set dynamics model
            auto dynamics = std::make_unique<cddp::Unicycle>(timestep);
            cddp_solver.setDynamicalSystem(std::move(dynamics));
            
            // Create and set objective
            Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
            Q(0, 0) = 10.0;  // x
            Q(1, 1) = 10.0;  // y
            Q(2, 2) = 1.0;   // theta
            
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
            R(0, 0) = 0.1;  // linear velocity
            R(1, 1) = 0.1;  // angular velocity
            
            Eigen::MatrixXd Qf = 100.0 * Q;
            
            std::vector<Eigen::VectorXd> empty_reference;
            auto objective = std::make_unique<cddp::QuadraticObjective>(
                Q, R, Qf, Xg, empty_reference, timestep);
            cddp_solver.setObjective(std::move(objective));
            
            // Add constraints
            Eigen::VectorXd u_upper(control_dim);
            u_upper << 1.0, 2.0;  // max linear vel, max angular vel
            
            cddp_solver.addPathConstraint("ControlConstraint",
                std::make_unique<cddp::ControlConstraint>(u_upper));
            
            // State bounds (keep robot in a region)
            Eigen::VectorXd x_lower(state_dim), x_upper_state(state_dim);
            x_lower << -0.5, -0.5, -M_PI;
            x_upper_state << 2.5, 2.5, M_PI;
            
            cddp_solver.addPathConstraint("StateConstraint",
                std::make_unique<cddp::StateConstraint>(x_lower, x_upper_state));
            
            // Configure options
            cddp::CDDPOptions opts;
            opts.max_iterations = 100;
            opts.tolerance = 1e-4;
            opts.verbose = false;
            opts.debug = true;  // Enable debug output to see barrier updates
            opts.return_iteration_info = true;
            
            // Set barrier strategy
            if (solver_name == "IPDDP") {
                opts.ipddp.barrier.strategy = strategies[i];
                opts.ipddp.barrier.mu_initial = 1.0;
                opts.ipddp.barrier.mu_update_factor = 0.2;
                opts.ipddp.barrier.mu_update_power = 1.5;
            } else {
                opts.msipddp.barrier.strategy = strategies[i];
                opts.msipddp.barrier.mu_initial = 1.0;
                opts.msipddp.barrier.mu_update_factor = 0.2;
                opts.msipddp.barrier.mu_update_power = 1.5;
            }
            
            cddp_solver.setOptions(opts);
            
            // Initialize with straight line trajectory
            std::vector<Eigen::VectorXd> X_init(horizon + 1);
            std::vector<Eigen::VectorXd> U_init(horizon);
            
            for (int t = 0; t <= horizon; ++t) {
                double alpha = static_cast<double>(t) / horizon;
                X_init[t] = (1.0 - alpha) * X0 + alpha * Xg;
            }
            
            for (int t = 0; t < horizon; ++t) {
                U_init[t] = Eigen::VectorXd::Zero(control_dim);
            }
            
            cddp_solver.setInitialTrajectory(X_init, U_init);
            
            // Solve
            cddp::CDDPSolution solution = cddp_solver.solve(solver_name);
            
            // Print results
            auto status_message = std::any_cast<std::string>(solution.at("status_message"));
            auto iterations = std::any_cast<int>(solution.at("iterations_completed"));
            auto solve_time = std::any_cast<double>(solution.at("solve_time_ms"));
            auto final_cost = std::any_cast<double>(solution.at("final_objective"));
            
            std::cout << "Status: " << status_message << "\n";
            std::cout << "Iterations: " << iterations << "\n";
            std::cout << "Final cost: " << final_cost << "\n";
            std::cout << "Solve time: " << solve_time << " ms\n";
            
            // Extract final barrier parameter
            try {
                double final_mu = std::any_cast<double>(solution.at("final_barrier_parameter_mu"));
                std::cout << "Final barrier μ: " << final_mu << "\n";
            } catch (const std::exception& e) {
                std::cout << "Final barrier μ: Not available\n";
            }
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Test completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}