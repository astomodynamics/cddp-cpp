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
#include <cmath>
#include <filesystem>
#include <memory>
#include <thread>
#include <chrono>
#include <iomanip>
#include <Eigen/Dense>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

// Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw)
Eigen::Vector3d quaternionToEuler(double qw, double qx, double qy, double qz)
{
    // Roll (phi)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    double phi = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (theta)
    double sinp = 2.0 * (qw * qy - qz * qx);
    double theta = (std::abs(sinp) >= 1.0) ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);

    // Yaw (psi)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double psi = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(phi, theta, psi);
}

int main() {
    // --------------------------
    // 1. Problem setup
    // --------------------------
    
    // State: [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    int state_dim = 13;
    int control_dim = 4; // [f1, f2, f3, f4]
    double timestep = 0.02;

    // Quadrotor parameters
    double mass = 1.2;         // 1.2 kg
    double arm_length = 0.165; // 16.5 cm
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 7.782e-3; // Ixx
    inertia_matrix(1, 1) = 7.782e-3; // Iyy
    inertia_matrix(2, 2) = 1.439e-2; // Izz

    std::string integration_type = "rk4";

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(0, 0) = 1.0; // x position
    Q(1, 1) = 1.0; // y position
    Q(2, 2) = 1.0; // z position

    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);

    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Control constraints
    double min_force = 0.0;
    double max_force = 4.0;
    Eigen::VectorXd control_upper_bound = max_force * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_lower_bound = min_force * Eigen::VectorXd::Ones(control_dim);

    // Initial trajectory guess (hover thrust)
    double hover_thrust = mass * 9.81 / 4.0;

    // Create directory for results
    const std::string plotDirectory = "../results/parallel_comparison";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Test different horizon lengths to see parallelization benefit
    std::vector<int> horizon_lengths = {10, 20, 50, 100, 200, 400};
    
    // Store results
    std::vector<double> sequential_times;
    std::vector<double> parallel_times;
    std::vector<double> speedups;
    std::vector<double> sequential_costs;
    std::vector<double> parallel_costs;
    std::vector<int> sequential_iterations;
    std::vector<int> parallel_iterations;

    std::cout << "\n========================================\n";
    std::cout << "   MSIPDDP Parallel vs Sequential Comparison\n";
    std::cout << "========================================\n";
    std::cout << "Hardware: " << std::thread::hardware_concurrency() << " CPU cores available\n\n";

    for (int horizon : horizon_lengths) {
        std::cout << "\n--- Testing horizon = " << horizon << " ---\n";

        // Figure-8 trajectory parameters
        double figure8_scale = 3.0;     // 3m
        double constant_altitude = 2.0; // 2m
        double total_time = horizon * timestep;
        double omega = 2.0 * M_PI / total_time;

        // Generate reference trajectory for the given horizon
        // Note: Shorter horizons will only track a partial figure-8 trajectory
        std::vector<Eigen::VectorXd> figure8_reference_states;
        figure8_reference_states.reserve(horizon + 1);

        for (int i = 0; i <= horizon; ++i) {
            double t = i * timestep;
            double angle = omega * t;

            Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);
            ref_state(0) = figure8_scale * std::cos(angle);
            ref_state(1) = figure8_scale * std::sin(angle) * std::cos(angle);
            ref_state(2) = constant_altitude;
            ref_state(3) = 1.0; // qw
            
            figure8_reference_states.push_back(ref_state);
        }

        // Goal state
        Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
        goal_state(0) = figure8_scale;
        goal_state(2) = constant_altitude;
        goal_state(3) = 1.0;

        // Initial state
        Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
        initial_state(0) = figure8_scale;
        initial_state(2) = constant_altitude;
        initial_state(3) = 1.0;

        // Initial trajectory
        std::vector<Eigen::VectorXd> X_init = figure8_reference_states;
        std::vector<Eigen::VectorXd> U_init(horizon, hover_thrust * Eigen::VectorXd::Ones(control_dim));

        // --------------------------------------------------------
        // Test 1: Sequential MSIPDDP
        // --------------------------------------------------------
        std::cout << "Running Sequential MSIPDDP..." << std::endl;
        
        cddp::CDDPOptions options_sequential;
        options_sequential.max_iterations = 50;
        options_sequential.verbose = false;
        options_sequential.debug = false;
        options_sequential.tolerance = 1e-5;
        options_sequential.acceptable_tolerance = 1e-6;
        options_sequential.regularization.initial_value = 1e-3;
        options_sequential.msipddp.barrier.mu_initial = 1e-1;
        options_sequential.msipddp.barrier.mu_update_factor = 0.5;
        options_sequential.msipddp.barrier.mu_update_power = 1.2;
        options_sequential.msipddp.segment_length = 2;
        options_sequential.msipddp.rollout_type = "nonlinear";
        options_sequential.use_ilqr = true;
        options_sequential.enable_parallel = false; // Disable parallelization
        options_sequential.num_threads = 1;

        cddp::CDDP solver_sequential(
            initial_state,
            goal_state,
            horizon,
            timestep,
            std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
            std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, figure8_reference_states, timestep),
            options_sequential
        );

        solver_sequential.setInitialTrajectory(X_init, U_init);
        solver_sequential.addPathConstraint("ControlConstraint",
            std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));

        auto start_seq = std::chrono::high_resolution_clock::now();
        cddp::CDDPSolution sol_sequential = solver_sequential.solve(cddp::SolverType::MSIPDDP);
        auto end_seq = std::chrono::high_resolution_clock::now();
        
        double time_sequential = std::chrono::duration<double>(end_seq - start_seq).count();
        double cost_sequential = std::any_cast<double>(sol_sequential.at("final_objective"));
        int iter_sequential = std::any_cast<int>(sol_sequential.at("iterations_completed"));
        
        sequential_times.push_back(time_sequential);
        sequential_costs.push_back(cost_sequential);
        sequential_iterations.push_back(iter_sequential);

        std::cout << "Sequential: Time = " << std::fixed << std::setprecision(4) 
                  << time_sequential << "s, Cost = " << cost_sequential 
                  << ", Iterations = " << iter_sequential << std::endl;

        // --------------------------------------------------------
        // Test 2: Parallel MSIPDDP
        // --------------------------------------------------------
        std::cout << "Running Parallel MSIPDDP..." << std::endl;
        
        cddp::CDDPOptions options_parallel;
        options_parallel.max_iterations = 50;
        options_parallel.verbose = false;
        options_parallel.debug = false;
        options_parallel.tolerance = 1e-5;
        options_parallel.acceptable_tolerance = 1e-6;
        options_parallel.regularization.initial_value = 1e-3;
        options_parallel.msipddp.barrier.mu_initial = 1e-1;
        options_parallel.msipddp.barrier.mu_update_factor = 0.5;
        options_parallel.msipddp.barrier.mu_update_power = 1.2;
        options_parallel.msipddp.segment_length = 2;
        options_parallel.msipddp.rollout_type = "nonlinear";
        options_parallel.use_ilqr = true;
        options_parallel.enable_parallel = true; // Enable parallelization
        options_parallel.num_threads = std::thread::hardware_concurrency();

        cddp::CDDP solver_parallel(
            initial_state,
            goal_state,
            horizon,
            timestep,
            std::make_unique<cddp::Quadrotor>(timestep, mass, inertia_matrix, arm_length, integration_type),
            std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, figure8_reference_states, timestep),
            options_parallel
        );

        solver_parallel.setInitialTrajectory(X_init, U_init);
        solver_parallel.addPathConstraint("ControlConstraint",
            std::make_unique<cddp::ControlConstraint>(control_upper_bound, control_lower_bound));

        auto start_par = std::chrono::high_resolution_clock::now();
        cddp::CDDPSolution sol_parallel = solver_parallel.solve(cddp::SolverType::MSIPDDP);
        auto end_par = std::chrono::high_resolution_clock::now();
        
        double time_parallel = std::chrono::duration<double>(end_par - start_par).count();
        double cost_parallel = std::any_cast<double>(sol_parallel.at("final_objective"));
        int iter_parallel = std::any_cast<int>(sol_parallel.at("iterations_completed"));
        
        parallel_times.push_back(time_parallel);
        parallel_costs.push_back(cost_parallel);
        parallel_iterations.push_back(iter_parallel);

        double speedup = time_sequential / time_parallel;
        speedups.push_back(speedup);

        std::cout << "Parallel:   Time = " << std::fixed << std::setprecision(4) 
                  << time_parallel << "s, Cost = " << cost_parallel 
                  << ", Iterations = " << iter_parallel << std::endl;
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        
        // Check if solutions are similar
        double cost_difference = std::abs(cost_sequential - cost_parallel);
        if (cost_difference > 1e-3) {
            std::cout << "WARNING: Cost difference = " << cost_difference << std::endl;
        }
    }

    // --------------------------------------------------------
    // Plot results
    // --------------------------------------------------------
    
    // Convert horizon lengths to doubles for plotting
    std::vector<double> horizons_double;
    for (int h : horizon_lengths) {
        horizons_double.push_back(static_cast<double>(h));
    }

    // Figure 1: Computation time comparison
    auto fig1 = figure(true);
    fig1->size(1400, 600);

    // Subplot 1: Absolute times
    auto ax1 = subplot(1, 2, 0);
    auto p1 = plot(ax1, horizons_double, sequential_times, "-o");
    p1->display_name("Sequential");
    p1->line_width(2);
    hold(ax1, true);
    auto p2 = plot(ax1, horizons_double, parallel_times, "-s");
    p2->display_name("Parallel");
    p2->line_width(2);
    
    xlabel(ax1, "Horizon Length");
    ylabel(ax1, "Computation Time (s)");
    title(ax1, "MSIPDDP Computation Time: Sequential vs Parallel");
    legend(ax1);
    grid(ax1, true);

    // Subplot 2: Speedup
    auto ax2 = subplot(1, 2, 1);
    auto p3 = bar(ax2, horizons_double, speedups);
    
    // Add horizontal line at speedup = 1
    hold(ax2, true);
    auto line = plot(ax2, {horizons_double.front(), horizons_double.back()}, {1.0, 1.0}, "r--");
    line->line_width(1);
    
    xlabel(ax2, "Horizon Length");
    ylabel(ax2, "Speedup Factor");
    title(ax2, "Parallel Speedup (Sequential Time / Parallel Time)");
    grid(ax2, true);
    
    // Add text annotations for speedup values
    for (size_t i = 0; i < horizons_double.size(); ++i) {
        auto t = text(ax2, horizons_double[i], speedups[i] + 0.1, 
                     std::to_string(speedups[i]).substr(0, 4) + "x");
        t->font_size(8);
    }

    fig1->draw();
    fig1->save(plotDirectory + "/msipddp_parallel_comparison.png");
    std::cout << "\nSaved computation time plot to " 
              << plotDirectory + "/msipddp_parallel_comparison.png" << std::endl;

    // Figure 2: Scaling analysis
    auto fig2 = figure(true);
    fig2->size(800, 600);
    
    auto ax3 = fig2->current_axes();
    
    auto p4 = loglog(ax3, horizons_double, sequential_times, "-o");
    p4->display_name("Sequential");
    p4->line_width(2);
    hold(ax3, true);
    auto p5 = loglog(ax3, horizons_double, parallel_times, "-s");
    p5->display_name("Parallel");
    p5->line_width(2);
    
    xlabel(ax3, "Horizon Length (log scale)");
    ylabel(ax3, "Computation Time (s, log scale)");
    title(ax3, "MSIPDDP Scaling Analysis");
    legend(ax3);
    grid(ax3, true);

    fig2->draw();
    fig2->save(plotDirectory + "/msipddp_scaling_analysis.png");
    std::cout << "Saved scaling analysis plot to " 
              << plotDirectory + "/msipddp_scaling_analysis.png" << std::endl;

    // --------------------------------------------------------
    // Print summary table
    // --------------------------------------------------------
    std::cout << "\n========================================\n";
    std::cout << "           Summary Results\n";
    std::cout << "========================================\n";
    std::cout << "Horizon | Seq Time | Par Time | Speedup | Seq Cost | Par Cost | Cost Δ | Seq It | Par It\n";
    std::cout << "--------|----------|----------|---------|----------|----------|--------|--------|--------\n";
    
    for (size_t i = 0; i < horizon_lengths.size(); ++i) {
        double cost_diff = std::abs(parallel_costs[i] - sequential_costs[i]);
        std::cout << std::setw(7) << horizon_lengths[i] << " | "
                  << std::setw(8) << std::fixed << std::setprecision(4) << sequential_times[i] << " | "
                  << std::setw(8) << std::fixed << std::setprecision(4) << parallel_times[i] << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << speedups[i] << " | "
                  << std::setw(8) << std::fixed << std::setprecision(1) << sequential_costs[i] << " | "
                  << std::setw(8) << std::fixed << std::setprecision(1) << parallel_costs[i] << " | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << cost_diff << " | "
                  << std::setw(6) << sequential_iterations[i] << " | "
                  << std::setw(6) << parallel_iterations[i] << "\n";
    }
    
    // Calculate average speedup for horizons >= 20 (where parallel is enabled)
    double avg_speedup = 0.0;
    int count = 0;
    for (size_t i = 0; i < horizon_lengths.size(); ++i) {
        if (horizon_lengths[i] >= 20) {
            avg_speedup += speedups[i];
            count++;
        }
    }
    if (count > 0) {
        avg_speedup /= count;
        std::cout << "\nAverage speedup for horizon >= 20: " 
                  << std::fixed << std::setprecision(2) << avg_speedup << "x\n";
    }
    
    std::cout << "\nNote: Parallel backward pass is only activated for horizon >= 20\n";
    std::cout << "========================================\n\n";

    return 0;
}