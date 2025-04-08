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
#include <Eigen/Dense>
#include <matplot/matplot.h>

#include "cddp.hpp"

namespace fs = std::filesystem;
using namespace cddp;

int main() {
    // =========================================================================
    // 1) Parameters for Trajectory Optimization
    // =========================================================================

    // Optimization horizon info
    int    N       = 400;          // Optimization horizon length
    double time_horizon = 400.0; // Time horizon for optimization [s]
    double dt      = time_horizon / N; // Time step for optimization

    // HCW parameters
    double mean_motion = 0.001107; 
    double mass        = 100.0;   
    
    // Initial state 
    Eigen::VectorXd initial_state(6);
    initial_state << 25.0, 25.0 / std::sqrt(3.0), 0.0, 0.0, 0.0, 0.0; 

    // Final (reference/goal) state
    Eigen::VectorXd goal_state(6);
    goal_state.setZero(); // Goal is the origin

    // Input constraints
    double u_max = 1.0;  // for each dimension
    double u_min = -1.0; // for each dimension

    // Cost weighting
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6,6);
    Q.diagonal() << 1e+1, 1e+1, 1e+1, 1e-0, 1e-0, 1e-0;

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3,3);
    R.diagonal() << 1e-1, 1e-1, 1e-1;

    // Terminal cost 
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(6,6);
    // Qf.diagonal() << 1e3, 1e3, 1e3, 1e1, 1e1, 1e1;

    // =========================================================================
    // 2) Setup Solver and Solve the Optimization Problem Once
    // =========================================================================
    
    // Create the HCW system for optimization
    std::unique_ptr<cddp::DynamicalSystem> hcw_system = 
        std::make_unique<HCW>(dt, mean_motion, mass, "euler");

    // Build cost objective
    std::vector<Eigen::VectorXd> empty_reference; // No intermediate reference states needed
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference, dt
    );

    // Setup IPDDP solver options
    cddp::CDDPOptions options;
    options.max_iterations       = 500;    // May need more iterations for one-shot solve
    options.max_line_search_iterations = 21; 
    options.cost_tolerance       = 1e-5;  // Tighter tolerance for final solve
    options.grad_tolerance       = 1e-5;  // Tighter tolerance for final solve
    options.verbose             = true;  // Show solver progress
    options.use_parallel         = false;
    options.num_threads          = 8;
    options.regularization_type  = "both"; 
    options.regularization_state = 1e-5; 
    options.regularization_control = 1e-5;
    options.barrier_coeff        = 1e-1; // Starting barrier coefficient

    // Setup CDDP solver instance (using IPDDP)
    cddp::CDDP cddp_solver(/*initial_state=*/initial_state,
                           /*goal_state=*/goal_state, // Pass goal state (used by objective)
                           /*horizon=*/N,
                           /*timestep=*/dt,
                           /*system=*/std::move(hcw_system), // System ownership transferred
                           /*objective=*/std::move(objective), // Objective ownership transferred
                           /*options=*/options);

    // Add Control  Constraint
    Eigen::VectorXd u_upper = Eigen::VectorXd::Constant(3, u_max);
    cddp_solver.addConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_upper));

    // Add Second Order Cone Constraint
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // Opening along positive Y-axis
    double fov = M_PI / 3.0; // 30 degrees half-angle
    double tan_fov = std::tan(fov);
    double epsilon = 1e-8;
    // cddp_solver.addConstraint("SecondOrderConeConstraint",
    //     std::make_unique<cddp::SecondOrderConeConstraint>(origin, axis, fov, epsilon));

    // Initialize trajectory guess
    std::vector<Eigen::VectorXd> X_init(N + 1, Eigen::VectorXd::Zero(6));
    std::vector<Eigen::VectorXd> U_init(N,     Eigen::VectorXd::Zero(3));

    // Simple linear interpolation guess for states
    for (int k = 0; k < N + 1; ++k) {
        X_init[k] = initial_state + (goal_state - initial_state) * (double(k) / N);
    }

    // Assign the initial trajectory guess to the solver
    cddp_solver.setInitialTrajectory(X_init, U_init);
    
    // Solve the Trajectory Optimization Problem
    std::cout << "Solving the HCW docking trajectory optimization problem using IPDDP..." << std::endl;
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP"); 

    
    // =========================================================================
    // 3) Visualize the Trajectory
    // =========================================================================
    if (!solution.state_sequence.empty()) {
        namespace plt = matplot;

        std::vector<double> x_traj, y_traj, z_traj;
        for (const auto& state : solution.state_sequence) {
            if (state.size() >= 3) { // Ensure state has at least 3 dimensions (x, y, z)
                x_traj.push_back(state(0));
                y_traj.push_back(state(1));
                z_traj.push_back(state(2));
            }
        }

        auto fig = plt::figure();
        plt::plot3(x_traj, y_traj, z_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);

        // Plot start and end points
        plt::plot3(std::vector<double>{initial_state(0)}, std::vector<double>{initial_state(1)}, std::vector<double>{initial_state(2)}, "go")
            ->marker_size(10).display_name("Start");
        plt::plot3(std::vector<double>{goal_state(0)}, std::vector<double>{goal_state(1)}, std::vector<double>{goal_state(2)}, "rx")
            ->marker_size(10).display_name("Goal");

        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::zlabel("Z [m]");
        plt::title("Spacecraft Docking Trajectory");
        plt::legend();
        plt::grid(true);
        plt::axis("equal"); // Keep aspect ratio for better visualization
        plt::hold(false);

        // Define plot directory and create if it doesn't exist
        std::string plotDirectory = "../results/tests"; // Assuming run from build dir
        if (!fs::exists(plotDirectory)) {
            try {
                fs::create_directories(plotDirectory);
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating directory \"" << plotDirectory << "\": " << e.what() << std::endl;
                // Optionally handle the error, e.g., save to current dir instead
                plotDirectory = "."; // Fallback to current directory
            }
        }

        std::string filename = plotDirectory + "/docking_trajectory.png";
        try {
            plt::save(filename);
            std::cout << "Saved trajectory visualization to " << filename << std::endl;
        } catch (const std::runtime_error& e) {
             std::cerr << "Error saving plot to \"" << filename << "\": " << e.what() << std::endl;
        }

        // =========================================================================
        // 4) Visualize the Trajectory (X-Y Plane)
        // =========================================================================
        auto fig_xy = plt::figure();
        plt::plot(x_traj, y_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);

        // Plot start and end points (2D)
        plt::plot({initial_state(0)}, {initial_state(1)}, "go")->marker_size(10).display_name("Start");
        plt::plot({goal_state(0)}, {goal_state(1)}, "rx")->marker_size(10).display_name("Goal");

        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::title("Spacecraft Docking Trajectory (X-Y Plane)");
        plt::legend();
        plt::grid(true);
        plt::axis("equal");
        plt::hold(false);

        std::string filename_xy = plotDirectory + "/docking_trajectory_xy.png";
        try {
            plt::save(fig_xy, filename_xy); // Save the specific figure
            std::cout << "Saved X-Y trajectory visualization to " << filename_xy << std::endl;
        } catch (const std::runtime_error& e) {
             std::cerr << "Error saving X-Y plot to \"" << filename_xy << "\": " << e.what() << std::endl;
        }

    } else {
        std::cerr << "Solution trajectory is empty, cannot visualize." << std::endl;
    }


    std::cout << "Trajectory optimization finished." << std::endl;
    return 0;
} 