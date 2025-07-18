/*
 Copyright 2025 Tomo Sasaki

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
#include <string>
#include <chrono>
#include <thread>
#include <numeric>
#include "cddp.hpp"
#include "dynamics_model/acrobot.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

int main() {
    int state_dim = 4;
    int control_dim = 1;
    int horizon = 100;
    double timestep = 0.05;
    double Tf = timestep * horizon; // Total time

    double l1 = 1.0;
    double l2 = 1.0;
    double m1 = 1.0;
    double m2 = 1.0;
    double J1 = 1.0;
    double J2 = 1.0;
    std::string integration_type = "rk4";
    
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Acrobot>(
        timestep, l1, l2, m1, m2, J1, J2, integration_type);
    
    // Cost matrices 
    Eigen::MatrixXd Q = 10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd R = 1e-2 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = 100.0 * Q;
    
    // Goal state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << M_PI/2.0, 0.0, 0.0, 0.0;
    
    std::vector<Eigen::VectorXd> empty_reference_states;
    
    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << -M_PI/2.0, 0.0, 0.0, 0.0;
    
    // Create objective
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);
    
    // Control constraints 
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -15.0;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 15.0;
    
    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-3;
    options.regularization.initial_value = 1e-4;
    options.use_ilqr = true;
    options.enable_parallel = true;
    options.num_threads = 10;
    options.debug = false;
    options.ipddp.barrier.mu_initial = 1e-1;
    
    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::move(system), std::move(objective), options);
    
    // Add control constraint
    cddp_solver.addPathConstraint("ControlConstraint", 
        std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    
    // Initial trajectory
    std::vector<Eigen::VectorXd> X_init(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_init(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Rollout initial trajectory
    X_init[0] = initial_state;
    auto acrobot = std::make_unique<cddp::Acrobot>(timestep, l1, l2, m1, m2, J1, J2, integration_type);
    for (int t = 0; t < horizon; ++t) {
        X_init[t + 1] = acrobot->getDiscreteDynamics(X_init[t], U_init[t], t * timestep);
    }
    
    cddp_solver.setInitialTrajectory(X_init, U_init);
    
    // Solve
    cddp::CDDPSolution solution = cddp_solver.solve(cddp::SolverType::IPDDP);
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points"));
    
    // Create plot directory
    const std::string plotDirectory = "../results/acrobot";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }
    
    // Extract solution data for animation
    std::vector<double> time_arr, theta1_arr, theta2_arr;
    
    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        theta1_arr.push_back(X_sol[i](0));
        theta2_arr.push_back(X_sol[i](1));
    }
    
    // --- Animation ---
    auto fig = figure();
    auto ax = fig->current_axes();
    fig->size(800, 800);
    
    // Generate animation frames
    for (size_t i = 0; i < X_sol.size(); i += 5) {
        cla(ax);
        hold(ax, true);
        
        // Current state
        double theta1 = theta1_arr[i];
        double theta2 = theta2_arr[i];
        
        // Link 1 endpoint
        double x1 = l1 * std::sin(theta1);
        double y1 = -l1 * std::cos(theta1);
        
        // Link 2 endpoint (relative to link 1)
        double x2 = x1 + l2 * std::sin(theta1 + theta2);
        double y2 = y1 - l2 * std::cos(theta1 + theta2);
        
        // Plot link 1
        std::vector<double> link1_x = {0.0, x1};
        std::vector<double> link1_y = {0.0, y1};
        auto link1 = plot(link1_x, link1_y);
        link1->line_style("b-");
        link1->line_width(5);
        
        // Plot link 2
        std::vector<double> link2_x = {x1, x2};
        std::vector<double> link2_y = {y1, y2};
        auto link2 = plot(link2_x, link2_y);
        link2->line_style("r-");
        link2->line_width(5);
        
        // Plot joints as circles
        std::vector<double> joint0_x = {0.0};
        std::vector<double> joint0_y = {0.0};
        auto j0 = scatter(joint0_x, joint0_y);
        j0->marker_size(10);
        j0->marker_color("black");
        j0->marker_style("o");
        
        std::vector<double> joint1_x = {x1};
        std::vector<double> joint1_y = {y1};
        auto j1 = scatter(joint1_x, joint1_y);
        j1->marker_size(8);
        j1->marker_color("gray");
        j1->marker_style("o");
        
        std::vector<double> joint2_x = {x2};
        std::vector<double> joint2_y = {y2};
        auto j2 = scatter(joint2_x, joint2_y);
        j2->marker_size(6);
        j2->marker_color("red");
        j2->marker_style("o");
        
        // Set axis properties
        xlim({-2.5, 2.5});
        ylim({-2.5, 2.5});
        xlabel("x [m]");
        ylabel("y [m]");
        title("Acrobot Animation - Time: " + std::to_string(time_arr[i]) + " s");
        grid(true);
        
        // Save frame
        std::string filename = plotDirectory + "/frame_" + std::to_string(i/5) + ".png";
        fig->save(filename);
    }
    
    // Combine all saved frames into a GIF using ImageMagick's convert tool
    std::string command = "convert -delay 30 " + plotDirectory + "/frame_*.png " + plotDirectory + "/acrobot.gif";
    std::system(command.c_str());
    
    // Clean up frame files
    std::string cleanup_command = "rm " + plotDirectory + "/frame_*.png";
    std::system(cleanup_command.c_str());
    
    std::cout << "Animation saved as " << plotDirectory << "/acrobot.gif" << std::endl;
    
    // Print final state
    std::cout << "\nFinal state:" << std::endl;
    std::cout << "θ₁ = " << X_sol.back()(0) << " rad" << std::endl;
    std::cout << "θ₂ = " << X_sol.back()(1) << " rad" << std::endl;
    std::cout << "θ̇₁ = " << X_sol.back()(2) << " rad/s" << std::endl;
    std::cout << "θ̇₂ = " << X_sol.back()(3) << " rad/s" << std::endl;
    
    return 0;
}