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
    initial_state << 25.0, 25.0 / std::sqrt(3.0)-0.1, 0.0, 0.0, 0.0, 0.0; 

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
    R.diagonal() << 1e-0, 1e-0, 1e-0;

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

    // =========================================================================
    // 2.1) Solve Unconstrained Problem for Initial Guess
    // =========================================================================
    std::cout << "Solving unconstrained problem for initial guess using iLQR..." << std::endl;

    // Create a separate system and objective for the initial solve (or clone)
    std::unique_ptr<cddp::DynamicalSystem> hcw_system_init =
        std::make_unique<HCW>(dt, mean_motion, mass, "euler");
    std::vector<Eigen::VectorXd> empty_reference_init; // No intermediate reference states needed
    auto objective_init = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_init, dt
    );

    // Use simpler options for the initial guess solve
    cddp::CDDPOptions options_init = options; // Copy base options
    options_init.max_iterations = 100;      // Fewer iterations might suffice
    options_init.cost_tolerance = 1e-4;       // Looser tolerance
    options_init.grad_tolerance = 1e-4;       // Looser tolerance
    options_init.verbose = false;           // Less verbose for initial solve
    options_init.regularization_type = "control"; // Simpler regularization often works
    options_init.regularization_control = 1e-4;

    cddp::CDDP cddp_solver_init(/*initial_state=*/initial_state,
                                /*goal_state=*/goal_state,
                                /*horizon=*/N,
                                /*timestep=*/dt,
                                /*system=*/std::move(hcw_system_init),
                                /*objective=*/std::move(objective_init),
                                /*options=*/options_init);

    // Simple linear interpolation guess for the initial solve
    std::vector<Eigen::VectorXd> X_linear_init(N + 1, Eigen::VectorXd::Zero(6));
    std::vector<Eigen::VectorXd> U_zero_init(N,     Eigen::VectorXd::Zero(3));
    for (int k = 0; k < N + 1; ++k) {
        X_linear_init[k] = initial_state + (goal_state - initial_state) * (double(k) / N);
    }
    cddp_solver_init.setInitialTrajectory(X_linear_init, U_zero_init);

    // Solve the unconstrained problem (e.g., using IPDDP)
    cddp::CDDPSolution initial_solution = cddp_solver_init.solve("IPDDP");

    if (initial_solution.state_sequence.empty()) {
        std::cerr << "Failed to find an initial guess solution. Exiting." << std::endl;
        return 1;
    }
    
    // =========================================================================
    // 2.2) Setup Constrained Solver (IPDDP) and Solve
    // =========================================================================
    std::cout << "Setting up and solving the constrained problem using IPDDP..." << std::endl;

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

    // Add Linear Constraints for X-Y Plane Cone Boundary (|y| <= x * tan(fov) for x >= 0)
    // Constraint 1: -tan(fov)*x + y <= 0
    // Constraint 2: -tan(fov)*x - y <= 0
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(1.0, 0.0, 0.0); // Opening along positive X-axis
    double fov = M_PI / 3.0; // 30 degrees half-angle
    double tan_fov = std::tan(fov);
    // Eigen::MatrixXd A_cone_xy(2, 6);
    // A_cone_xy << -tan_fov,  1.0, 0.0, 0.0, 0.0, 0.0, // Row 1
    //              -tan_fov, -1.0, 0.0, 0.0, 0.0, 0.0; // Row 2
    // Eigen::VectorXd b_cone_xy = Eigen::VectorXd::Zero(2);
    // cddp_solver.addConstraint("ConeXYLinearConstraint",
    //     std::make_unique<cddp::LinearConstraint>(A_cone_xy, b_cone_xy));

    // Add Second Order Cone Constraint
    cddp_solver.addConstraint("SecondOrderConeConstraint",
        std::make_unique<cddp::SecondOrderConeConstraint>(origin, axis, fov, 1e-6));

    // Initialize trajectory guess
    // Use the solution from the unconstrained solve as the initial guess
    std::vector<Eigen::VectorXd> X_init = initial_solution.state_sequence;
    std::vector<Eigen::VectorXd> U_init = initial_solution.control_sequence;

    // Assign the initial trajectory guess to the solver
    cddp_solver.setInitialTrajectory(X_init, U_init);
    
    // Solve the Trajectory Optimization Problem
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

        // --- Add Cone Visualization (3D) ---
        // Generate cone surface points (opening along +X)
        double cone_x_min_vis = -5.0; // Min x-extent for visualization
        double cone_x_max_vis = goal_state(0) + 5.0; // Max x-extent (a bit beyond goal)
        if (!y_traj.empty()) {
           // Adjust max based on trajectory, but min should start at origin for +Y side visualization
           cone_x_max_vis = std::max(cone_x_max_vis, *std::max_element(x_traj.begin(), x_traj.end()) + 5.0);
        }
        // Start cone visualization from the origin's x-coordinate (for +X side)
        double cone_x_start = origin(0);
        cone_x_max_vis = std::max(cone_x_max_vis, cone_x_start + 1.0); // Ensure some minimum length

        std::vector<double> x_cone_vals = plt::linspace(cone_x_start, cone_x_max_vis, 30);
        std::vector<double> theta_vals = plt::linspace(0, 2 * M_PI, 50);
        auto [X_cone_grid, THETA_cone] = plt::meshgrid(x_cone_vals, theta_vals); // Grid X values

        // Find orthogonal vectors u, v to the axis (now X-axis)
        Eigen::Vector3d u = Eigen::Vector3d(0.0, 1.0, 0.0); // Y-direction
        Eigen::Vector3d v_ortho = Eigen::Vector3d(0.0, 0.0, 1.0); // Z-direction

        // Calculate Y, Z coordinates for the cone surface
        std::vector<std::vector<double>> Y_cone(X_cone_grid.size(), std::vector<double>(X_cone_grid[0].size()));
        std::vector<std::vector<double>> Z_cone(X_cone_grid.size(), std::vector<double>(X_cone_grid[0].size()));

        for (size_t i = 0; i < X_cone_grid.size(); ++i) {
            for (size_t j = 0; j < X_cone_grid[0].size(); ++j) {
                // Cone opening along positive X: radius depends on displacement from origin along axis
                double x_disp = X_cone_grid[i][j] - origin(0); // x_disp will be >= 0 due to linspace start

                // Radius calculation based on positive displacement along the axis
                double radius = x_disp * tan_fov; // No abs needed since x_disp >= 0
                double theta = THETA_cone[i][j];
                Eigen::Vector3d point_on_circle = radius * (std::cos(theta) * u + std::sin(theta) * v_ortho);
                Eigen::Vector3d point = origin + axis * x_disp + point_on_circle; // Point relative to origin based on x_disp
                // X_cone_grid[i][j] already holds the x value
                Y_cone[i][j] = point.y();
                Z_cone[i][j] = point.z();
            }
        }

        // Plot the cone surface
        plt::surf(X_cone_grid, Y_cone, Z_cone)->face_alpha(0.3).edge_color("none").display_name("FOV Cone (+X)");
        // --- End Cone Visualization (3D) ---

        plt::legend();
        plt::grid(true);
        plt::axis("equal"); // Keep aspect ratio for better visualization
        plt::hold(false);
        plt::show();

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

        // --- Add Cone Visualization (2D - X/Y Slice at Z=0) ---
        // For cone opening along X-axis: Boundary lines are y = oy +/- (x - ox) * tan_fov
        auto current_xlim = plt::xlim();
        // Use only the positive side x >= ox for the lines, consistent with 3D plot
        double x_line_start = std::max(current_xlim[0], origin(0));
        double x_line_end = current_xlim[1];
        if (x_line_end <= x_line_start) { // Ensure valid range if xlim is weird
           x_line_end = x_line_start + 1.0;
        }
        std::vector<double> x_line_vals = {x_line_start, x_line_end};

        // Adjust x values relative to origin for radius calculation
        std::vector<double> x_rel_origin = {x_line_vals[0] - origin(0), x_line_vals[1] - origin(0)};

        // Line 1: y = oy + (x - ox) * tan_fov
        std::vector<double> y_line1 = {origin(1) + x_rel_origin[0] * tan_fov, origin(1) + x_rel_origin[1] * tan_fov};
        // Line 2: y = oy - (x - ox) * tan_fov
        std::vector<double> y_line2 = {origin(1) - x_rel_origin[0] * tan_fov, origin(1) - x_rel_origin[1] * tan_fov};

        plt::plot(x_line_vals, y_line1, "k--")->line_width(1.5).display_name("Cone Boundary (X-Y)");
        plt::plot(x_line_vals, y_line2, "k--")->line_width(1.5);
        // Ensure plot limits encompass the lines if needed (axis equal might handle it)
        plt::xlim(current_xlim); // Restore xlim just in case plot adjusted it
        // --- End Cone Visualization (2D) ---

        plt::legend();
        plt::grid(true);
        plt::axis("equal");
        plt::xlim({-10.0, 30.0});
        plt::ylim({-10.0, 30.0});
        plt::hold(false);
        plt::show();

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