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
#include <stdexcept>
#include <cmath> // For std::sqrt

#include "cddp.hpp"
#include "cddp_core/cddp_core.hpp"
#include "cddp_core/objective.hpp"
#include "dynamics_model/spacecraft_linear_fuel.hpp"

namespace cddp
{
    class MinimizeFuelObjective : public NonlinearObjective
    {
    public:
        MinimizeFuelObjective(
            double weight_terminal_fuel,
            const Eigen::VectorXd& goal_state_pos_vel, // Should be size 6: [x,y,z,vx,vy,vz]
            double weight_terminal_pos, 
            double weight_terminal_vel,
            double timestep)
            : NonlinearObjective(timestep),
              goal_state_pos_vel_(goal_state_pos_vel),
              weight_terminal_fuel_(weight_terminal_fuel),
              weight_terminal_pos_(weight_terminal_pos),
              weight_terminal_vel_(weight_terminal_vel)
        {
        }

        double running_cost(const Eigen::VectorXd& /*state*/,
                            const Eigen::VectorXd& /*control*/,
                            int /*index*/) const override
        {
            return 0.0; 
        }

        double terminal_cost(const Eigen::VectorXd& final_state) const override
        {
            
            double cost = -weight_terminal_fuel_ * final_state(7) * final_state(7);

            // Position cost
            for (int i = 0; i < 3; ++i) {
                double error = final_state(i) - goal_state_pos_vel_(i);
                cost += 0.5 * weight_terminal_pos_ * error * error;
            }

            // Velocity cost
            for (int i = 0; i < 3; ++i) {
                double error = final_state(i + 3) - goal_state_pos_vel_(i + 3);
                cost += 0.5 * weight_terminal_vel_ * error * error;
            }
            return cost;
        }

        Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& /*control*/, int /*index*/) const override
        {
            return Eigen::VectorXd::Zero(state.size());
        }

        Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& /*state*/, const Eigen::VectorXd& control, int /*index*/) const override
        {
            return Eigen::VectorXd::Zero(control.size());
        }

        Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const override
        {
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(final_state.size());
            grad(7) = weight_terminal_fuel_ * 2.0 * final_state(7);

            // Position gradient
            for (int i = 0; i < 3; ++i) {
                grad(i) = weight_terminal_pos_ * (final_state(i) - goal_state_pos_vel_(i));
            }

            // Velocity gradient
            for (int i = 0; i < 3; ++i) {
                grad(i + 3) = weight_terminal_vel_ * (final_state(i + 3) - goal_state_pos_vel_(i + 3));
            }
            return grad;
        }

        Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& /*control*/, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(state.size(), state.size());
        }

        Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& /*state*/, const Eigen::VectorXd& control, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(control.size(), control.size());
        }

        Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(control.size(), state.size());
        }

        Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const override
        {
            Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(final_state.size(), final_state.size());
            // Hessian for fuel term is zero

            hess(7, 7) = -weight_terminal_fuel_;

            // Position Hessian
            for (int i = 0; i < 3; ++i) {
                hess(i, i) = weight_terminal_pos_;
            }

            // Velocity Hessian
            for (int i = 0; i < 3; ++i) {
                hess(i + 3, i + 3) = weight_terminal_vel_;
            }
            return hess;
        }

    private:
        double weight_terminal_fuel_;
        Eigen::VectorXd goal_state_pos_vel_; // For x, y, z, vx, vy, vz, mass, accumulated_control_effort
        double weight_terminal_pos_;
        double weight_terminal_vel_;
    };
} // namespace cddp

namespace fs = std::filesystem;
using namespace cddp;

int main()
{
    // Optimization horizon info
    int horizon = 200;                  // Optimization horizon length
    double time_horizon = 200.0;        // Time horizon for optimization [s]
    double dt = time_horizon / horizon; // Time step for optimization
    int state_dim = 8;
    int control_dim = 3;

    // HCW parameters
    double mean_motion = 0.001107;
    double mass = 100.0;
    double isp = 300.0;
    double g0 = 9.80665;
    double nominal_radius = 50.0;

    // Initial state
    Eigen::VectorXd initial_state(8);
    initial_state << nominal_radius, 0.0, 0.0, 0.0, -2.0*mean_motion*nominal_radius, 0.0, mass, 0.0;

    // Final (reference/goal) state
    Eigen::VectorXd goal_state(8);
    goal_state << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mass, 0.0; // Goal is the origin

    // Input constraints
    double u_max = 1.0;  // for each dimension
    double u_min = -1.0; // for each dimension
    double u_min_norm = 0.0; // Minimum thrust magnitude
    double u_max_norm = 1.0; // Maximum thrust magnitude, consistent with previous u_max

    // Create the HCW system for optimization
    std::unique_ptr<cddp::DynamicalSystem> hcw_system =
        std::make_unique<SpacecraftLinearFuel>(dt, mean_motion, isp, g0, "euler");

    // Cost weighting
    double weight_terminal_fuel = 1.0;
    double weight_terminal_pos = 1000.0;
    double weight_terminal_vel = 1000.0;

    // Build cost objective
    auto objective = std::make_unique<cddp::MinimizeFuelObjective>(
        weight_terminal_fuel,
        goal_state,
        weight_terminal_pos,
        weight_terminal_vel,
        dt);

    // Setup IPDDP solver options
    cddp::CDDPOptions options;
    options.max_iterations = 500; // May need more iterations for one-shot solve
    options.max_line_search_iterations = 21;
    options.cost_tolerance = 1e-7; // Tighter tolerance for final solve
    options.grad_tolerance = 1e-7; // Tighter tolerance for final solve
    options.verbose = true;        // Show solver progress
    options.use_parallel = false;
    options.num_threads = 1;
    options.regularization_type = "control";
    options.regularization_control = 1e-3;
    options.barrier_coeff = 1e-1; // Starting barrier coefficient
    options.is_ilqr = true;
    options.debug = true;
    options.ms_segment_length = horizon / 10;
    options.ms_rollout_type = "nonlinear";

    // Initial trajectory.
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    // Generate initial trajectory by constant initial state
    for (int i = 0; i < horizon; ++i)
    {
        X[i + 1] = hcw_system->getDiscreteDynamics(X[i], U[i], i * dt);
        // X[i] = initial_state;
    }

    // Create CDDP solver.
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        dt,
        std::move(hcw_system),
        std::move(objective),
        options);
    cddp_solver.setInitialTrajectory(X, U);

    // Add Control Constraint
    Eigen::VectorXd u_upper = Eigen::VectorXd::Constant(3, u_max);
    cddp_solver.addConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_upper));
    // Add Thrust Magnitude Constraint
    cddp_solver.addConstraint("ThrustMagnitudeConstraint",
        std::make_unique<cddp::ThrustMagnitudeConstraint>(u_min_norm, u_max_norm));

    // Solve the Trajectory Optimization Problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Extract the solution
    std::vector<Eigen::VectorXd> X_solution = solution.state_sequence;
    std::vector<Eigen::VectorXd> U_solution = solution.control_sequence;

    if (!X_solution.empty() && !U_solution.empty())
    {
        namespace plt = matplot;

        // Create time vectors
        std::vector<double> t_states(horizon + 1);
        for(int i = 0; i <= horizon; ++i) t_states[i] = i * dt;

        std::vector<double> t_controls(horizon);
        for(int i = 0; i < horizon; ++i) t_controls[i] = i * dt;

        // Extract individual state and control trajectories
        std::vector<double> x_pos(horizon + 1), y_pos(horizon + 1), z_pos(horizon + 1);
        std::vector<double> vx(horizon + 1), vy(horizon + 1), vz(horizon + 1);
        std::vector<double> mass_traj(horizon + 1), acc_effort_traj(horizon + 1);

        for (size_t i = 0; i < X_solution.size(); ++i) {
            const auto& state = X_solution[i];
            if (state.size() >= 8) { // Ensure state has at least 8 dimensions
                x_pos[i] = state(0);
                y_pos[i] = state(1);
                z_pos[i] = state(2);
                vx[i] = state(3);
                vy[i] = state(4);
                vz[i] = state(5);
                mass_traj[i] = state(6);
                acc_effort_traj[i] = state(7);
            }
        }

        std::vector<double> ux(horizon), uy(horizon), uz(horizon);
        std::vector<double> thrust_magnitude(horizon);

        for (size_t i = 0; i < U_solution.size(); ++i) {
            const auto& control = U_solution[i];
            if (control.size() >= 3) { // Ensure control has at least 3 dimensions
                ux[i] = control(0);
                uy[i] = control(1);
                uz[i] = control(2);
                thrust_magnitude[i] = control.norm();
            }
        }

        // --- Generate Plots ---
        // plt::figure_size(1200, 800); // Removed due to compilation error

        // 1. Position Trajectories
        plt::figure();
        plt::plot(t_states, x_pos)->line_width(2).display_name("x (pos)");
        plt::hold(true);
        plt::plot(t_states, y_pos)->line_width(2).display_name("y (pos)");
        plt::plot(t_states, z_pos)->line_width(2).display_name("z (pos)");
        plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Position (m)");
        plt::legend();
        plt::title("Position vs. Time");

        // 2. Velocity Trajectories
        plt::figure();
        plt::plot(t_states, vx)->line_width(2).display_name("vx");
        plt::hold(true);
        plt::plot(t_states, vy)->line_width(2).display_name("vy");
        plt::plot(t_states, vz)->line_width(2).display_name("vz");
        plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Velocity (m/s)");
        plt::legend();
        plt::title("Velocity vs. Time");

        // 3. Mass Trajectory
        plt::figure();
        plt::plot(t_states, mass_traj)->line_width(2).display_name("Mass (kg)");
        plt::xlabel("Time (s)");
        plt::ylabel("Mass (kg)");
        plt::legend();
        plt::title("Mass vs. Time");

        // 4. Accumulated Control Effort
        plt::figure();
        plt::plot(t_states, acc_effort_traj)->line_width(2).display_name("Accum. Effort");
        plt::xlabel("Time (s)");
        plt::ylabel("Effort");
        plt::legend();
        plt::title("Accumulated Control Effort vs. Time");

        // 5. Control Inputs (Zeroth-Order Hold)
        plt::figure();
        plt::stairs(t_controls, ux)->line_width(2).display_name("ux");
        plt::hold(true);
        plt::stairs(t_controls, uy)->line_width(2).display_name("uy");
        plt::stairs(t_controls, uz)->line_width(2).display_name("uz");
        plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Control Input");
        plt::legend();
        plt::title("Control Inputs (ZOH) vs. Time");

        // 6. Thrust Magnitude (Zeroth-Order Hold)
        plt::figure();
        plt::stairs(t_controls, thrust_magnitude)->line_width(2).display_name("||Thrust|| (ZOH)");
        // Add norm constraint lines if available
        // plt::hold(true);
        // plt::plot(t_controls, std::vector<double>(horizon, u_max_norm), "--r")->display_name("Max Norm");
        // if (u_min_norm > 0) plt::plot(t_controls, std::vector<double>(horizon, u_min_norm), "--y")->display_name("Min Norm");
        // plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Thrust Magnitude");
        plt::legend();
        plt::title("Thrust Magnitude (ZOH) vs. Time");

        // 7. 3D Trajectory
        plt::figure();
        plt::plot3(x_pos, y_pos, z_pos, "-o")->line_width(2).marker_size(4).display_name("Trajectory");
        plt::hold(true);
        // Plot Start and End points
        if (!x_pos.empty()){ // Check if trajectories are not empty
             plt::scatter3(std::vector<double>{x_pos.front()}, std::vector<double>{y_pos.front()}, std::vector<double>{z_pos.front()})
                ->marker_color("g").marker_style("o").marker_size(10).display_name("Start");
             plt::scatter3(std::vector<double>{x_pos.back()}, std::vector<double>{y_pos.back()}, std::vector<double>{z_pos.back()})
                ->marker_color("r").marker_style("x").marker_size(10).display_name("End");
        }
        plt::hold(false);
        plt::xlabel("x (m)");
        plt::ylabel("y (m)");
        plt::zlabel("z (m)");
        plt::legend();
        plt::title("3D Trajectory");
        plt::axis(plt::equal); // For aspect_ratio=:equal


        // Show all plots
        plt::show();
        std::cout << "Final mass: " << mass_traj.back() << std::endl;
        std::cout << "Plotting complete." << std::endl;
    }
    else
    {
        std::cout << "Solver did not find a solution, or solution variables are not available. Skipping plots." << std::endl;
    }

    return 0;
}