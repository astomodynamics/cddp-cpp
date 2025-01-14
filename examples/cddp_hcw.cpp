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

/*
 * Example code demonstrating an HCW (Hill-Clohessy-Wiltshire) rendezvous problem with CDDP
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>

#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

// ----------------------------------------------------------------------------------------
// Main function demonstrating usage
// ----------------------------------------------------------------------------------------
int main() {
    // Random number generator for optional initial control sequence
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 0.01);

    // HCW problem dimensions
    const int STATE_DIM   = 6;  // [x, y, z, vx, vy, vz]
    const int CONTROL_DIM = 3;  // [Fx, Fy, Fz]

    // Set up the time horizon
    int horizon = 50;            // Number of steps
    double timestep = 10.0;        // Timestep in seconds (example)
    std::string integration_type = "euler";  // or "rk4", etc.

    // HCW parameters
    double mean_motion = 0.001107; 
    double mass        = 100.0;   

    // Create the HCW dynamical system
    // (this class is defined in your "spacecraft_linear.hpp")
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::HCW>(timestep, mean_motion, mass, integration_type);

    // Initial state (for example, some offset and small velocity)
    Eigen::VectorXd initial_state(STATE_DIM);
    initial_state << 50.0,  14.0,  0.0,  0.0,  0.0,  0.0;

    // Goal state (origin in relative coordinates, zero velocity)
    Eigen::VectorXd goal_state(STATE_DIM);
    goal_state.setZero();

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    {
        Q(0,0) = 1e3;  Q(1,1) = 1e3;  Q(2,2) = 1e3;
        Q(3,3) = 1e1;  Q(4,4) = 1e1;  Q(5,5) = 1e1;
    }
    
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    {
        R(0,0) = 1e2;  R(1,1) = 1e2;  R(2,2) = 1e2;
    }
    
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    // {
    //     Qf(0,0) = 1e3;
    //     Qf(1,1) = 1e3;
    //     Qf(2,2) = 1e3;
    //     Qf(3,3) = 1e1;
    //     Qf(4,4) = 1e1;
    //     Qf(5,5) = 1e1;
    // }

    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Create the CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Optionally add control constraints (e.g., max thruster force)
    Eigen::VectorXd umin(CONTROL_DIM);
    Eigen::VectorXd umax(CONTROL_DIM);
    // Suppose each axis force is limited to +/- 2 N
    umin << -1.0, -1.0, -1.0;
    umax <<  1.0,  1.0,  1.0;

    cddp_solver.addConstraint("ControlBoxConstraint",
        std::make_unique<cddp::ControlBoxConstraint>(umin, umax));

    // Set solver options
    cddp::CDDPOptions options;
    options.max_iterations    = 300;       // Max number of CDDP iterations
    options.verbose           = true;      // Print progress
    options.cost_tolerance    = 1e-3;      // Stop if improvement below this
    options.grad_tolerance    = 1e-3;      // Stop if gradient below this
    options.regularization_type = "control";  // Common regularization approach
    options.debug             = false;     
    options.use_parallel      = true;      
    options.num_threads       = 8;         // Parallelization
    cddp_solver.setOptions(options);

    // Initialize the trajectory (X,U) with something nontrivial
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(STATE_DIM));
    std::vector<Eigen::VectorXd> U(horizon,     Eigen::VectorXd::Zero(CONTROL_DIM));
    
    // Set the initial state
    X[0] = initial_state;

    // Random or small constant initialization for control
    for (auto& u : U) {
        // u << d(gen), d(gen), d(gen);  // small random thruster
        u << 0.0, 0.0, 0.0;  // zero thruster
    }

    // Compute the initial cost by rolling out the initial controls
    double J_init = 0.0;
    for (int t = 0; t < horizon; t++) {
        J_init += cddp_solver.getObjective().running_cost(X[t], U[t], t);
        X[t+1] = cddp_solver.getSystem().getDiscreteDynamics(X[t], U[t]);
    }
    J_init += cddp_solver.getObjective().terminal_cost(X[horizon]);
    std::cout << "[Info] Initial cost: " << J_init << std::endl;
    std::cout << "[Info] Initial final state: " << X[horizon].transpose() << std::endl;

    // Pass this initial guess to the solver
    cddp_solver.setInitialTrajectory(X, U);

    // Solve
    cddp::CDDPSolution solution = cddp_solver.solve();

    // Extract solution and print result
    double J_final = solution.cost_sequence.back();
    std::cout << "\n[Result] CDDP solved." << std::endl;
    std::cout << "[Result] Final cost: " << J_final << std::endl;
    std::cout << "[Result] Final state: "
              << solution.state_sequence.back().transpose() << std::endl;

    // Plot results
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create plot directory
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Extract solution data
    std::vector<double> x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, time_arr;
    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        y_arr.push_back(X_sol[i](1));
        z_arr.push_back(X_sol[i](2));
        vx_arr.push_back(X_sol[i](3));
        vy_arr.push_back(X_sol[i](4));
        vz_arr.push_back(X_sol[i](5));
    }

    // Plot results
    plt::figure_size(1200, 800);
    plt::subplot(2, 3, 1);
    plt::title("X Position");
    plt::plot(time_arr, x_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::grid(true);

    plt::subplot(2, 3, 2);
    plt::title("Y Position");
    plt::plot(time_arr, y_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::grid(true);

    plt::subplot(2, 3, 3);
    plt::title("Z Position");
    plt::plot(time_arr, z_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::grid(true);

    plt::subplot(2, 3, 4);
    plt::title("X Velocity");
    plt::plot(time_arr, vx_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [m/s]");
    plt::grid(true);    

    plt::subplot(2, 3, 5);
    plt::title("Y Velocity");
    plt::plot(time_arr, vy_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [m/s]");
    plt::grid(true);

    plt::subplot(2, 3, 6);
    plt::title("Z Velocity");
    plt::plot(time_arr, vz_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [m/s]");
    plt::grid(true);

    // plt::tight_layout();
    plt::save(plotDirectory + "/hcw_results.png");
    plt::clf();

    // Plot control inputs
    std::vector<double> fx_arr, fy_arr, fz_arr, time_arr2;
    for (size_t i = 0; i < U_sol.size(); ++i) {
        time_arr2.push_back(t_sol[i]);
        fx_arr.push_back(U_sol[i](0));
        fy_arr.push_back(U_sol[i](1));
        fz_arr.push_back(U_sol[i](2));
    }

    plt::figure_size(800, 600);
    plt::title("Control Inputs");
    plt::plot(time_arr2, fx_arr, "b-");
    plt::plot(time_arr2, fy_arr, "r-");
    plt::plot(time_arr2, fz_arr, "g-");
    // Plot the control limits
    plt::plot(time_arr2, std::vector<double>(time_arr2.size(), umin(0)), "k--");
    plt::plot(time_arr2, std::vector<double>(time_arr2.size(), umax(0)), "k--");
    plt::plot(time_arr2, std::vector<double>(time_arr2.size(), umin(1)), "k--");
    plt::plot(time_arr2, std::vector<double>(time_arr2.size(), umax(1)), "k--");
    plt::plot(time_arr2, std::vector<double>(time_arr2.size(), umin(2)), "k--");
    plt::plot(time_arr2, std::vector<double>(time_arr2.size(), umax(2)), "k--");
    plt::xlabel("Time [s]");
    plt::ylabel("Force [N]");
    plt::grid(true);

    plt::save(plotDirectory + "/hcw_control_inputs.png");
    plt::clf();


    return 0;
}
