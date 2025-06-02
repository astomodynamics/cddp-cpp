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

namespace cddp
{
    class SumOfTwoNormObjective : public NonlinearObjective
    {
    public:
        SumOfTwoNormObjective(const Eigen::VectorXd &goal_state,
                              double weight_running_control,
                              double weight_terminal_state,
                              double timestep)
            : NonlinearObjective(timestep),
              goal_state_(goal_state),
              weight_running_control_(weight_running_control),
              weight_terminal_state_(weight_terminal_state)
        {
            Qf_ = Eigen::MatrixXd::Identity(goal_state.size(), goal_state.size());
            Qf_(6, 6) = 0.0;
            Qf_(7, 7) = 0.0;
            Qf_(8, 8) = 0.0;
            Qf_(9, 9) = 0.0;
        }

        double running_cost(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control,
                            int /*index*/) const override
        {
            double control_squared = control.squaredNorm();
            double control_norm = std::sqrt(control_squared + epsilon_);
            return weight_running_control_ * control_norm;
        }

        double terminal_cost(const Eigen::VectorXd &final_state) const override
        {
            Eigen::VectorXd state_error = final_state - goal_state_;
            return weight_terminal_state_ * state_error.transpose() * Qf_ * state_error;
        }

        // Analytical Gradients
        Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd & state, const Eigen::VectorXd & /*control*/, int /*index*/) const override
        {
            return Eigen::VectorXd::Zero(state.size()); // Or handle error appropriately if size unknown
        }

        Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd & /*state*/, const Eigen::VectorXd &control, int /*index*/) const override
        {
            double norm_u = std::sqrt(control.squaredNorm() + epsilon_);
            if (norm_u < 1e-8)
            {
                return Eigen::VectorXd::Zero(control.size());
            }
            return weight_running_control_ * control / norm_u;
        }

        Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd &final_state) const override
        {
            Eigen::VectorXd state_error = final_state - goal_state_;
            return 2.0 * weight_terminal_state_ * Qf_ * state_error;
        }

        // Analytical Hessians
        Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd & state, const Eigen::VectorXd & /*control*/, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(state.size(), state.size());
        }

        Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd & /*state*/, const Eigen::VectorXd &control, int /*index*/) const override
        {   
            double norm_u = std::sqrt(control.squaredNorm() + epsilon_);
            int control_dim = control.size();
            if (norm_u < 1e-8)
            {
                return weight_running_control_ * Eigen::MatrixXd::Identity(control_dim, control_dim) / 1e-8; // or a small value like 1.0
            }
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(control_dim, control_dim);
            Eigen::MatrixXd u_ut = control * control.transpose();
            return weight_running_control_ * (I / norm_u - u_ut / (norm_u * norm_u * norm_u));
        }

        Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(control.size(), state.size());
        }

        Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd & /*final_state*/) const override
        {
            int state_dim = goal_state_.size();
            return 2.0 * weight_terminal_state_ * Qf_;
        }

    private:
        Eigen::VectorXd goal_state_;
        double weight_running_control_;
        double weight_terminal_state_;
        double epsilon_ = 1e-5;
        Eigen::MatrixXd Qf_;
    };
} // namespace cddp

namespace fs = std::filesystem;
using namespace cddp;

int main()
{
    // Optimization horizon info
    int horizon = 500;                  // Optimization horizon length
    double time_horizon = 10000.0;        // Time horizon for optimization [s]
    double dt = time_horizon / horizon; // Time step for optimization
    int state_dim = 10;
    int control_dim = 3;

    // HCW parameters
    double gravitational_parameter = 3.9860044e14;
    double ref_radius = (6371.0 + 500.0) * 1e3;
    double ref_period = 2 * M_PI * sqrt(pow(ref_radius, 3) / 3.9860044e14);
    double ref_mean_motion = 2 * M_PI / ref_period;
    double mass = 100.0;
    double nominal_radius = 50.0;
    std::string integration_type = "rk4";

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0, ref_radius, 0.0, 0.0, ref_mean_motion;

    // Final (reference/goal) state
    Eigen::VectorXd goal_state(state_dim);
    goal_state << -100.0, 0.0, 0.0, 0.0, 2*ref_mean_motion*100.0, 0.0, ref_radius, 0.0, 0.0, ref_mean_motion; 

    // Input constraints
    double u_max = 0.05;  // for each dimension
    double u_min = -0.05; // for each dimension
    double u_min_norm = 0.0; // Minimum thrust magnitude
    double u_max_norm = 0.05; // Maximum thrust magnitude, consistent with previous u_max

    // Cost weighting for SumOfTwoNormObjective
    double weight_running_control = 1.0;  // Example value
    double weight_terminal_state = 1000.0; // Example value

    // Create the SpacecraftNonlinear system for optimization
    std::unique_ptr<cddp::DynamicalSystem> spacecraft_system =
        std::make_unique<SpacecraftNonlinear>(dt, integration_type, mass, 1.0, 1.0, gravitational_parameter);

    // Build cost objective
    auto objective = std::make_unique<cddp::SumOfTwoNormObjective>(
        goal_state,
        weight_running_control,
        weight_terminal_state,
        dt);

    // Setup IPDDP solver options
    cddp::CDDPOptions options;
    options.max_iterations = 100; // May need more iterations for one-shot solve
    options.max_line_search_iterations = 21;
    options.cost_tolerance = 1e-7; // Tighter tolerance for final solve
    options.grad_tolerance = 1e-7; // Tighter tolerance for final solve
    options.verbose = true;        // Show solver progress
    options.use_parallel = false;
    options.num_threads = 8;
    options.regularization_type = "control";
    options.regularization_control = 1e-3;
    options.barrier_coeff = 1e-0; // Starting barrier coefficient
    options.is_ilqr = true;
    options.debug = false;
    options.ms_segment_length = horizon;
    options.ms_rollout_type = "nonlinear";

    // Initial trajectory.
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    // Generate initial trajectory by constant initial state
    for (int i = 0; i < horizon; ++i)
    {
        // X[i + 1] = hcw_system->getDiscreteDynamics(X[i], U[i]);
        X[i+1] = initial_state;
    }

    // Create CDDP solver.
    cddp::CDDP cddp_solver(
        initial_state,
        goal_state,
        horizon,
        dt,
        std::move(spacecraft_system),
        std::move(objective),
        options);
    cddp_solver.setInitialTrajectory(X, U);

    // Add Control Constraint
    Eigen::VectorXd u_upper = Eigen::VectorXd::Constant(3, u_max);
    cddp_solver.addConstraint("ControlConstraint",
        std::make_unique<cddp::ControlConstraint>(u_upper));

    // // Add Thrust Magnitude Constraint
    // cddp_solver.addConstraint("MaxThrustMagnitudeConstraint",
    //     std::make_unique<cddp::MaxThrustMagnitudeConstraint>(u_max_norm));

     // Add Ball Constraint (for keep-out zone)
    double radius = 90.0;
    Eigen::Vector2d center(0.0, 0.0);
    // cddp_solver.addConstraint("BallConstraint",
    //     std::make_unique<cddp::BallConstraint>(radius, center, 0.1));

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

        for (size_t i = 0; i < X_solution.size(); ++i) {
            const auto& state = X_solution[i];
            if (state.size() >= state_dim) { // Ensure state has at least 8 dimensions
                x_pos[i] = state(0);
                y_pos[i] = state(1);
                z_pos[i] = state(2);
                vx[i] = state(3);
                vy[i] = state(4);
                vz[i] = state(5);
            }
        }

        std::vector<double> ux(horizon), uy(horizon), uz(horizon);
        std::vector<double> thrust_magnitude(horizon);

        for (size_t i = 0; i < U_solution.size(); ++i) {
            const auto& control = U_solution[i];
            if (control.size() >= control_dim) { // Ensure control has at least 3 dimensions
                ux[i] = control(0);
                uy[i] = control(1);
                uz[i] = control(2);
                thrust_magnitude[i] = control.norm();
            }
        }

        // Circle data for plotting
        std::vector<double> circle_x, circle_y;
        for (double theta = 0; theta <= 2*M_PI; theta += 0.01) {
            circle_x.push_back(radius * std::cos(theta));
            circle_y.push_back(radius * std::sin(theta));
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

        // 7. X-Y plane trajectory (x-axis vertical to the top, y-axis horizontal to the left)
        plt::figure();
        plt::plot(y_pos, x_pos)->line_width(2).display_name("Trajectory");
        plt::hold(true);
        // Plot Start and End points
        if (!x_pos.empty() && !y_pos.empty()){ // Check if trajectories are not empty
             plt::scatter(std::vector<double>{y_pos.front()}, std::vector<double>{x_pos.front()})
                ->marker_color("g").marker_style("o").marker_size(10).display_name("Start");
             plt::scatter(std::vector<double>{y_pos.back()}, std::vector<double>{x_pos.back()})
                ->marker_color("r").marker_style("x").marker_size(10).display_name("End");
        }
        // Plot Ball Constraint
        plt::hold(true);
        plt::plot(circle_x, circle_y)->line_width(2).display_name("Ball Constraint");
        plt::hold(false);
        plt::xlabel("y (m)");
        plt::ylabel("x (m)");
        plt::legend();
        plt::title("X-Y Plane Trajectory");
        plt::axis(plt::equal); // For aspect_ratio=:equal
        plt::gca()->x_axis().reverse(true); // Make y-axis (horizontal on plot) increase to the left

        // 8. 3D Trajectory
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
        std::cout << "Plotting complete." << std::endl;
    }
    else
    {
        std::cout << "Solver did not find a solution, or solution variables are not available. Skipping plots." << std::endl;
    }
        return 0;
    }