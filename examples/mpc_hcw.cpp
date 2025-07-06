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

#include "cddp.hpp"

#include "matplot/matplot.h"
using namespace matplot;
namespace fs = std::filesystem;
using namespace cddp;

int main() {
    // =========================================================================
    // 1) Parameters 
    // =========================================================================

    // Simulation times
    double tf         = 600.0;     // final time for simulation
    int    tN         = 6000;      // number of total time steps for simulation
    double dt_sim     = tf / tN;   // simulation time step (0.1 s)
    double Ts         = 2.5;      // control update period [s]

    // HCW parameters
    double mean_motion = 0.001107; 
    double mass        = 100.0;   
    
    HCW hcw_system(dt_sim, mean_motion, mass, "euler"); // HCW system

    // MPC horizon info
    double time_horizon = 400.0; // time horizon for MPC [s]
    int    N       = 40;          // predictive horizon length
    double dt_mpc  = time_horizon / N; // MPC time step

    // Final (reference) state
    Eigen::VectorXd x_ref(6);
    x_ref << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    // Input constraints
    double u_max = 1.0;  // for each dimension
    double u_min = -1.0; // for each dimension

    // Cost weighting
    Eigen::MatrixXd Q  = Eigen::MatrixXd::Zero(6,6);
    {
        Q(0,0) = 1e+1;  Q(1,1) = 1e+1;  Q(2,2) = 1e+1;
        Q(3,3) = 1e-0;  Q(4,4) = 1e-0;  Q(5,5) = 1e-0;
    }

    Eigen::MatrixXd R  = Eigen::MatrixXd::Zero(3,3);
    {
        R(0,0) = 1e-0;  R(1,1) = 1e-0;  R(2,2) = 1e-0;
    }

    // Terminal cost (can be zero or nonzero).
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(6,6);
    // {
    //     Qf(0,0) = 1e3;  Qf(1,1) = 1e3;  Qf(2,2) = 1e3;
    //     Qf(3,3) = 1e1;  Qf(4,4) = 1e1;  Qf(5,5) = 1e1;
    // }

    // Sample initial conditions (24).
    std::vector<Eigen::VectorXd> initial_conditions;
    initial_conditions.reserve(24);
    auto sqrt3 = std::sqrt(3.0);

    std::vector<std::vector<double>> ics_data = {
        {25.0, 25.0/sqrt3, 0, 0, 0, 0},
        {25.0, 0,          0, 0, 0, 0},
        {25.0,-25.0/sqrt3, 0, 0, 0, 0},
        {50.0, 50.0/sqrt3, 0, 0, 0, 0},
        {50.0, -0,          0, 0, 0, 0},
        {50.0,-50.0/sqrt3, 0, 0, 0, 0},
        {75.0, 75.0/sqrt3, 0, 0, 0, 0},
        {75.0, 0,          0, 0, 0, 0},
        {75.0,-75.0/sqrt3, 0, 0, 0, 0},
        {100.0,100.0/sqrt3,0, 0, 0, 0},
        {100.0,0,           0, 0, 0, 0},
        {100.0,-100.0/sqrt3,0, 0, 0, 0},
        {75.0, 25.0/sqrt3,  0, 0, 0, 0},
        {75.0,-25.0/sqrt3,  0, 0, 0, 0},
        {100.0,25.0/sqrt3,  0, 0, 0, 0},
        {100.0,-25.0/sqrt3, 0, 0, 0, 0},
        {75.0, 50.0/sqrt3,  0, 0, 0, 0},
        {75.0,-50.0/sqrt3,  0, 0, 0, 0},
        {100.0,50.0/sqrt3,  0, 0, 0, 0},
        {100.0,-50.0/sqrt3, 0, 0, 0, 0},
        {75.0, 75.0/sqrt3,  0, 0, 0, 0},
        {75.0,-75.0/sqrt3,  0, 0, 0, 0},
        {100.0,75.0/sqrt3,  0, 0, 0, 0},
        {100.0,-75.0/sqrt3, 0, 0, 0, 0}
    };

    // std::vector<std::vector<double>> ics_data = {
    //     {25.0/sqrt3,  25.0,  0, 0, 0, 0},
    //     {0,           25.0,  0, 0, 0, 0},
    //     {-25.0/sqrt3, 25.0,  0, 0, 0, 0},
    //     {50.0/sqrt3,  50.0,  0, 0, 0, 0},
    //     {0,           50.0,  0, 0, 0, 0},
    //     {-50.0/sqrt3, 50.0,  0, 0, 0, 0},
    //     {75.0/sqrt3,  75.0,  0, 0, 0, 0},
    //     {0,           75.0,  0, 0, 0, 0},
    //     {-75.0/sqrt3, 75.0,  0, 0, 0, 0},
    //     {100.0/sqrt3, 100.0, 0, 0, 0, 0},
    //     {0,           100.0, 0, 0, 0, 0},
    //     {-100.0/sqrt3,100.0, 0, 0, 0, 0},
    //     {25.0/sqrt3,  75.0,  0, 0, 0, 0},
    //     {-25.0/sqrt3, 75.0,  0, 0, 0, 0},
    //     {25.0/sqrt3,  100.0, 0, 0, 0, 0},
    //     {-25.0/sqrt3, 100.0, 0, 0, 0, 0},
    //     {50.0/sqrt3,  75.0,  0, 0, 0, 0},
    //     {-50.0/sqrt3, 75.0,  0, 0, 0, 0},
    //     {50.0/sqrt3,  100.0, 0, 0, 0, 0},
    //     {-50.0/sqrt3, 100.0, 0, 0, 0, 0},
    //     {75.0/sqrt3,  75.0,  0, 0, 0, 0},
    //     {-75.0/sqrt3, 75.0,  0, 0, 0, 0},
    //     {75.0/sqrt3,  100.0, 0, 0, 0, 0},
    //     {-75.0/sqrt3, 100.0, 0, 0, 0, 0}
    // };

    for (auto &data : ics_data) {
        Eigen::VectorXd x0(6);
        x0 << data[0], data[1], data[2], data[3], data[4], data[5];
        initial_conditions.push_back(x0);
    }

    // For plotting/recording data (x, y, z vs time).
    // X_data[ i_sample ][ time_index ] = 6D state
    // U_data[ i_sample ][ time_index ] = 3D control
    const int num_samples = static_cast<int>(initial_conditions.size());
    std::vector<std::vector<Eigen::VectorXd>> X_data(num_samples,
                                                     std::vector<Eigen::VectorXd>(tN));
    std::vector<std::vector<Eigen::VectorXd>> U_data(num_samples,
                                                     std::vector<Eigen::VectorXd>(tN - 1));

    // =========================================================================
    // 2) Loop over each initial condition, run an MPC-like closed-loop sim
    // =========================================================================
    // Optional random generator for initial guess in each MPC solve
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.01);

    for (int i = 0; i < num_samples; ++i) {
        // ---------------------------------------------------------------------
        // (A) Create a new instance of the HCW system
        // ---------------------------------------------------------------------
        std::unique_ptr<cddp::DynamicalSystem> mpc_system = 
            std::make_unique<HCW>(dt_mpc, mean_motion, mass, "euler");

        // ---------------------------------------------------------------------
        // (B) Build cost objective
        // ---------------------------------------------------------------------
        std::vector<Eigen::VectorXd> empty_reference;
        auto objective = std::make_unique<cddp::QuadraticObjective>(
            Q, R, Qf, x_ref, empty_reference, dt_mpc
        );

        // Adjust solver options if desired (similar to cddp_hcw.cpp)
        cddp::CDDPOptions options;
        options.max_iterations       = 50;    
        options.line_search.max_iterations = 21; 
        options.tolerance            = 1e-2;
        options.verbose             = false;
        options.enable_parallel      = false;
        options.num_threads          = 8;
        // Regularization
        options.regularization.initial_value = 1e-5;

        // ---------------------------------------------------------------------
        // (C) Setup CDDP solver
        // ---------------------------------------------------------------------
        cddp::CDDP cddp_solver(/*initial_state=*/initial_conditions[i],
                               /*goal_state=*/x_ref,
                               /*horizon=*/N,
                               /*timestep=*/dt_mpc,
                               /*system=*/std::move(mpc_system),
                               /*objective=*/std::move(objective),
                               /*options=*/options);

        // Control box constraints
        Eigen::VectorXd u_lower(3), u_upper(3);
        u_lower << u_min, u_min, u_min;
        u_upper << u_max, u_max, u_max;
        cddp_solver.addPathConstraint(
            "ControlBoxConstraint",
            std::make_unique<cddp::ControlBoxConstraint>(u_lower, u_upper)
        );

        // ---------------------------------------------------------------------
        // (D) Initialize storage for trajectory
        // ---------------------------------------------------------------------
        for (int k = 0; k < tN; ++k) {
            X_data[i][k] = Eigen::VectorXd::Zero(6);
        }
        for (int k = 0; k < tN - 1; ++k) {
            U_data[i][k] = Eigen::VectorXd::Zero(3);
        }
        X_data[i][0] = initial_conditions[i];

        // ---------------------------------------------------------------------
        // (E) Main time loop for simulation 
        // ---------------------------------------------------------------------
        Eigen::VectorXd current_state = initial_conditions[i];
        Eigen::VectorXd current_u(3);
        current_u.setZero();

        std::vector<Eigen::VectorXd> X_init(N + 1, Eigen::VectorXd::Zero(6));
        std::vector<Eigen::VectorXd> U_init(N,     Eigen::VectorXd::Zero(3));

        X_init[0] = initial_conditions[i];

        for (auto& u : U_init) {
            u << d(gen), d(gen), d(gen);
            // u << 0.0, 0.0, 0.0;  // zero thruster
        }

        // // Propagate the initial guess through the dynamics
        for (int k = 0; k < N; ++k) {
            X_init[k + 1] = hcw_system.getDiscreteDynamics(X_init[k], U_init[k], k * dt_mpc);
        }

        // Assign them to the solver
        cddp_solver.setInitialTrajectory(X_init, U_init);
        
        for (int k = 0; k < tN - 1; ++k) {
            double t = k * dt_sim;

            // Check if the state is close to the goal
            if (current_state.norm() < 1e-2) {
                X_data[i][k + 1] = current_state;
                current_u.setZero();
                U_data[i][k]     = current_u;
                continue;
            }

            // Re-solve MPC every Ts seconds fmod
            if (fmod(t, Ts) < dt_sim) {
                // Update the solver’s current-state as the new “initial state”
                cddp_solver.setInitialState(current_state);

                // Solve the MPC problem with horizon N
                cddp::CDDPSolution sol = cddp_solver.solve();
                
                // Extract the *first* control from that horizon
                auto control_traj = std::any_cast<std::vector<Eigen::VectorXd>>(sol.at("control_trajectory"));
                current_u = control_traj[0];

                auto X_sequence = std::any_cast<std::vector<Eigen::VectorXd>>(sol.at("state_trajectory"));
                auto U_sequence = std::any_cast<std::vector<Eigen::VectorXd>>(sol.at("control_trajectory"));

                cddp_solver.setInitialTrajectory(X_sequence, U_sequence);
            }

            // -----------------------------------------------------------------
            // Advance the state by one simulation step dt_sim
            // -----------------------------------------------------------------
            { 
                current_state = hcw_system.getDiscreteDynamics(current_state, current_u, k * dt_sim);
            }

            // Save data
            X_data[i][k + 1] = current_state;
            U_data[i][k]     = current_u;
        } // end for k
    } // end for i over samples

    // =========================================================================
    // 3) Plot the X-Y trajectories for all samples
    // =========================================================================
    {
        auto fig = figure();
        fig->size(800, 600);
        title("HCW MPC Trajectories (x vs y)");
        for (int i = 0; i < num_samples; ++i) {
            std::vector<double> xvals, yvals;
            xvals.reserve(tN);
            yvals.reserve(tN);
            for (int k = 0; k < tN; ++k) {
                xvals.push_back(X_data[i][k](0)); // rx
                yvals.push_back(X_data[i][k](1)); // ry
            }
            plot(xvals, yvals);
        }
        
        // Plot the initial conditions
        for (int i = 0; i < num_samples; ++i) {
            std::vector<double> xvals, yvals;
            xvals.push_back(initial_conditions[i](0));
            yvals.push_back(initial_conditions[i](1));
            plot(xvals, yvals, "ro");
        }

        xlabel("x [m]");
        ylabel("y [m]");
        // axis limit
        xlim({-10, 110});
        ylim({-100, 100});

        // xlim(-100, 100);
        // ylim(-10, 110);
        grid(true);

        // Create results directory
        const std::string plotDirectory = "../results/simulations";
        if (!fs::exists(plotDirectory)) {
            fs::create_directories(plotDirectory);
        }
        std::string figPath = plotDirectory + "/hcw_mpc_cddp_xaxis_trajectories.png";
        save(figPath);
        // show();
    }

    // Plot control 
    {
        figure()->size(800, 600);
        title("HCW MPC Control Inputs");
        for (int i = 0; i < num_samples; ++i) {
            std::vector<double> time_vals;
            std::vector<double> u1_vals, u2_vals, u3_vals, u_ub, u_lb;
            time_vals.reserve(tN - 1);
            u1_vals.reserve(tN - 1);
            u2_vals.reserve(tN - 1);
            u3_vals.reserve(tN - 1);
            u_ub.reserve(tN - 1);
            u_lb.reserve(tN - 1);
            for (int k = 0; k < tN - 1; ++k) {
                time_vals.push_back(k * dt_sim);
                u1_vals.push_back(U_data[i][k](0));
                u2_vals.push_back(U_data[i][k](1));
                u3_vals.push_back(U_data[i][k](2));
                u_ub.push_back(u_max);
                u_lb.push_back(u_min);
            }
            subplot(3, 1, 1);
            plot(time_vals, u1_vals);
            plot(time_vals, u_ub, "r--");
            plot(time_vals, u_lb, "r--");
            xlabel("Time [s]");
            ylabel("u1");
            grid(true);

            subplot(3, 1, 2);
            plot(time_vals, u2_vals);
            plot(time_vals, u_ub, "r--");
            plot(time_vals, u_lb, "r--");
            xlabel("Time [s]");
            ylabel("u2");
            grid(true);

            subplot(3, 1, 3);
            plot(time_vals, u3_vals);
            plot(time_vals, u_ub, "r--");
            plot(time_vals, u_lb, "r--");
            xlabel("Time [s]");
            ylabel("u3");
            grid(true);
        }
        // Create results directory
        const std::string plotDirectory = "../results/simulations";
        if (!fs::exists(plotDirectory
        )) {
            fs::create_directories(plotDirectory);
        }
        std::string figPath = plotDirectory + "/hcw_mpc_cddp_controls.png";
        save(figPath);
        // show();
    }


    std::cout << "MPC simulation for HCW completed successfully!\n";
    return 0;
}
