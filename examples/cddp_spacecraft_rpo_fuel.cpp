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
            if (weight_terminal_fuel_ < 0.0) {
                 throw std::invalid_argument("MinimizeFuelObjective: weight_terminal_fuel must be non-negative.");
            }
            if (weight_terminal_pos_ < 0.0) {
                 throw std::invalid_argument("MinimizeFuelObjective: weight_terminal_pos must be non-negative.");
            }
            if (weight_terminal_vel_ < 0.0) {
                 throw std::invalid_argument("MinimizeFuelObjective: weight_terminal_vel must be non-negative.");
            }
            if (goal_state_pos_vel_.size() != 8) {
                throw std::invalid_argument("MinimizeFuelObjective: goal_state_pos_vel must be of size 8.");
            }
        }

        double running_cost(const Eigen::VectorXd& /*state*/,
                            const Eigen::VectorXd& /*control*/,
                            int /*index*/) const override
        {
            return 0.0; 
        }

        double terminal_cost(const Eigen::VectorXd& final_state) const override
        {
            
            double cost = weight_terminal_fuel_ * final_state(7) * final_state(7);

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

            hess(7, 7) = weight_terminal_fuel_;

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
    int horizon = 400;                  // Optimization horizon length
    double time_horizon = 400.0;        // Time horizon for optimization [s]
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
    double u_max = 10.0;  // for each dimension
    double u_min = -10.0; // for each dimension
    double u_min_norm = 0.0; // Minimum thrust magnitude
    double u_max_norm = 10.0; // Maximum thrust magnitude, consistent with previous u_max

    // Create the HCW system for optimization
    std::unique_ptr<cddp::DynamicalSystem> hcw_system =
        std::make_unique<SpacecraftLinearFuel>(dt, mean_motion, isp, g0, "rk4");

    // Cost weighting
    double weight_terminal_fuel = 1.0;
    double weight_terminal_pos = 100.0;
    double weight_terminal_vel = 100.0;

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
    options.barrier_coeff = 1e-2; // Starting barrier coefficient
    options.is_ilqr = true;
    options.debug = true;
    options.defect_violation_penalty_initial = 1e-0;
    options.ms_segment_length = horizon;
    options.ms_rollout_type = "nonlinear";

    // Initial trajectory.
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    // Generate initial trajectory by constant initial state
    for (int i = 0; i < horizon; ++i)
    {
        X[i + 1] = hcw_system->getDiscreteDynamics(X[i], U[i]);
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
    cddp::CDDPSolution solution = cddp_solver.solve("MSIPDDP");

    // Extract the solution
    std::vector<Eigen::VectorXd> X_solution = solution.state_sequence;
    std::vector<Eigen::VectorXd> U_solution = solution.control_sequence;

    if (!X_solution.empty())
    {
        namespace plt = matplot;

        std::vector<double> x_traj, y_traj, z_traj;
        std::vector<double> ux_traj, uy_traj, uz_traj;
        std::vector<double> thrust_mag_traj;
        for (const auto &state : solution.state_sequence)
        {
            if (state.size() >= 3)
            { // Ensure state has at least 3 dimensions (x, y, z)
                x_traj.push_back(state(0));
                y_traj.push_back(state(1));
                z_traj.push_back(state(2));
            }
        }
        for (const auto &control : solution.control_sequence)
        {
            if (control.size() >= 3)
            {
                ux_traj.push_back(control(0));
                uy_traj.push_back(control(1));
                uz_traj.push_back(control(2));
                thrust_mag_traj.push_back(control.norm());
            }
        }
        // Plot 3d trajectory
        auto fig = plt::figure();
        plt::plot3(x_traj, y_traj, z_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);

        // Plot 2d trajectory
        auto fig_xy = plt::figure();
        plt::plot(x_traj, y_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);
        // Plot start and end points
        plt::plot({initial_state(0)}, {initial_state(1)}, "go")->marker_size(10).display_name("Start");

        // Plot control as subplot
        auto fig_control = plt::figure();
        plt::subplot(2, 1, 1);
        plt::plot(ux_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);
        plt::subplot(2, 1, 2);
        plt::plot(uy_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);
        plt::subplot(2, 1, 3);
        plt::plot(uz_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);
        
        // Plot thrust magnitude
        auto fig_thrust = plt::figure();
        plt::plot(thrust_mag_traj, "-o")->line_width(2).marker_size(4);
        plt::hold(true);

        plt::show();
    }

    return 0;
}