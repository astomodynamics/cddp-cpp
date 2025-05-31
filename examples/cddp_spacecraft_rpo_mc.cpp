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
            return weight_terminal_state_ * state_error.squaredNorm();
        }

        // Analytical Gradients
        Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd &state, const Eigen::VectorXd & /*control*/, int /*index*/) const override
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
            return 2.0 * weight_terminal_state_ * state_error;
        }

        // Analytical Hessians
        Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd &state, const Eigen::VectorXd & /*control*/, int /*index*/) const override
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
            return 2.0 * weight_terminal_state_ * Eigen::MatrixXd::Identity(state_dim, state_dim);
        }

    private:
        Eigen::VectorXd goal_state_;
        double weight_running_control_;
        double weight_terminal_state_;
        double epsilon_ = 1e-5;
    };
} // namespace cddp

namespace fs = std::filesystem;
using namespace cddp;

int main()
{
    // Monte Carlo parameters
    int num_mc_runs = 10;
    int successful_runs = 0;
    std::vector<Eigen::VectorXd> first_successful_X_solution;
    std::vector<Eigen::VectorXd> first_successful_U_solution;

    // Setup random number generation for initial state perturbation
    std::mt19937 gen(42); // Fixed seed for reproducibility
    // Define distributions for perturbations
    
    std::normal_distribution<> pos_perturb_dist(0.0, 0.05);
    std::normal_distribution<> vel_perturb_dist(0.0, 0.005);

    for (int mc_run = 0; mc_run < num_mc_runs; ++mc_run)
    {
        std::cout << "\n--- Monte Carlo Run: " << mc_run + 1 << "/" << num_mc_runs << " ---" << std::endl;

        // Optimization horizon info
        int horizon = 200;                  // Optimization horizon length
        double time_horizon = 200.0;        // Time horizon for optimization [s]
        double dt = time_horizon / horizon; // Time step for optimization
        int state_dim = 6;
        int control_dim = 3;

        // HCW parameters
        double mean_motion = 0.001107;
        double mass = 100.0;
        double nominal_radius = 50.0;

        // Initial state (nominal)
        Eigen::VectorXd nominal_initial_state(state_dim);
        nominal_initial_state << nominal_radius, 0.0, 0.0, 0.0, -2.0 * mean_motion * nominal_radius, 0.0;

        // Perturb initial state
        Eigen::VectorXd initial_state = nominal_initial_state;
        initial_state(0) += nominal_radius * pos_perturb_dist(gen);                     // Perturb x
        initial_state(1) += nominal_radius * pos_perturb_dist(gen);                     // Perturb y
        initial_state(2) += nominal_radius * pos_perturb_dist(gen);                     // Perturb z
        initial_state(3) += std::abs(nominal_initial_state(4)) * vel_perturb_dist(gen); // Perturb vx (use initial vy magnitude for scaling)
        initial_state(4) += std::abs(nominal_initial_state(4)) * vel_perturb_dist(gen); // Perturb vy
        initial_state(5) += std::abs(nominal_initial_state(4)) * vel_perturb_dist(gen); // Perturb vz

        // Final (reference/goal) state
        Eigen::VectorXd goal_state(state_dim);
        goal_state.setZero(); // Goal is the origin

        // Input constraints
        double u_max = 1.0;      // for each dimension
        double u_min = -1.0;     // for each dimension
        double u_min_norm = 0.0; // Minimum thrust magnitude
        double u_max_norm = 1.0; // Maximum thrust magnitude, consistent with previous u_max

        // Cost weighting for SumOfTwoNormObjective
        double weight_running_control = 1.0;   // Example value
        double weight_terminal_state = 1000.0; // Example value

        // Create the HCW system for optimization
        std::unique_ptr<cddp::DynamicalSystem> hcw_system =
            std::make_unique<HCW>(dt, mean_motion, mass, "euler");

        // Build cost objective
        auto objective = std::make_unique<cddp::SumOfTwoNormObjective>(
            goal_state,
            weight_running_control,
            weight_terminal_state,
            dt);

        // Setup IPDDP solver options
        cddp::CDDPOptions options;
        options.max_iterations = 1000;
        options.max_line_search_iterations = 21;
        options.cost_tolerance = 1e-7;
        options.grad_tolerance = 1e-7;
        options.verbose = false;
        options.debug = false;
        options.header_and_footer = false;
        options.use_parallel = false;
        options.num_threads = 1;
        options.regularization_type = "control";
        options.regularization_control = 1e-5;
        options.barrier_coeff = 1e-1;
        options.is_ilqr = true;
        options.ms_segment_length = horizon / 10;
        options.ms_rollout_type = "nonlinear";

        // Initial trajectory.
        std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
        std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
        X[0] = initial_state;
        for (int i = 0; i < horizon; ++i)
        {
            X[i + 1] = initial_state;
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

        // Solve the Trajectory Optimization Problem
        cddp::CDDPSolution solution = cddp_solver.solve("MSIPDDP");

        if (!solution.state_sequence.empty() && !solution.control_sequence.empty() && solution.converged)
        {
            std::cout << "Run " << mc_run + 1 << " converged." << std::endl;
            successful_runs++;
            if (first_successful_X_solution.empty())
            {
                first_successful_X_solution = solution.state_sequence;
                first_successful_U_solution = solution.control_sequence;
            }
        }
        else
        {
            std::cout << "Run " << mc_run + 1 << " did NOT converge or solution is empty." << std::endl;
        }
    }

    std::cout << "\n--- Monte Carlo Simulation Summary ---" << std::endl;
    std::cout << "Total runs: " << num_mc_runs << std::endl;
    std::cout << "Successful runs: " << successful_runs << std::endl;
    std::cout << "Success rate: " << (static_cast<double>(successful_runs) / num_mc_runs) * 100.0 << "%" << std::endl;

    // Plotting results from the first successful run
    if (!first_successful_X_solution.empty() && !first_successful_U_solution.empty())
    {
        namespace plt = matplot;

        int horizon = first_successful_U_solution.size();
        double time_horizon = 200.0; // Assuming this is fixed from the problem setup
        double dt = time_horizon / horizon;
        int state_dim = first_successful_X_solution[0].size();
        int control_dim = first_successful_U_solution[0].size();

        // Create time vectors
        std::vector<double> t_states(horizon + 1);
        for (int i = 0; i <= horizon; ++i)
            t_states[i] = i * dt;

        std::vector<double> t_controls(horizon);
        for (int i = 0; i < horizon; ++i)
            t_controls[i] = i * dt;

        // Extract individual state and control trajectories
        std::vector<double> x_pos(horizon + 1), y_pos(horizon + 1), z_pos(horizon + 1);
        std::vector<double> vx(horizon + 1), vy(horizon + 1), vz(horizon + 1);

        for (size_t i = 0; i < first_successful_X_solution.size(); ++i)
        {
            const auto &state = first_successful_X_solution[i];
            if (state.size() >= state_dim)
            {
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

        for (size_t i = 0; i < first_successful_U_solution.size(); ++i)
        {
            const auto &control = first_successful_U_solution[i];
            if (control.size() >= control_dim)
            {
                ux[i] = control(0);
                uy[i] = control(1);
                uz[i] = control(2);
                thrust_magnitude[i] = control.norm();
            }
        }

        // --- Generate Plots ---
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
        plt::title("Position vs. Time (First Successful Run)");

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
        plt::title("Velocity vs. Time (First Successful Run)");

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
        plt::title("Control Inputs (ZOH) vs. Time (First Successful Run)");

        // 6. Thrust Magnitude (Zeroth-Order Hold)
        plt::figure();
        plt::stairs(t_controls, thrust_magnitude)->line_width(2).display_name("||Thrust|| (ZOH)");
        plt::xlabel("Time (s)");
        plt::ylabel("Thrust Magnitude");
        plt::legend();
        plt::title("Thrust Magnitude (ZOH) vs. Time (First Successful Run)");

        // 7. 3D Trajectory
        plt::figure();
        plt::plot3(x_pos, y_pos, z_pos, "-o")->line_width(2).marker_size(4).display_name("Trajectory");
        plt::hold(true);
        if (!x_pos.empty())
        {
            plt::scatter3(std::vector<double>{x_pos.front()}, std::vector<double>{y_pos.front()}, std::vector<double>{z_pos.front()})
                ->marker_color("g")
                .marker_style("o")
                .marker_size(10)
                .display_name("Start");
            plt::scatter3(std::vector<double>{x_pos.back()}, std::vector<double>{y_pos.back()}, std::vector<double>{z_pos.back()})
                ->marker_color("r")
                .marker_style("x")
                .marker_size(10)
                .display_name("End");
        }
        plt::hold(false);
        plt::xlabel("x (m)");
        plt::ylabel("y (m)");
        plt::zlabel("z (m)");
        plt::legend();
        plt::title("3D Trajectory (First Successful Run)");
        plt::axis(plt::equal);

        // Show all plots
        plt::show();
        std::cout << "Plotting complete for the first successful run." << std::endl;
    }
    else
    {
        std::cout << "No successful run to plot, or solution variables are not available." << std::endl;
    }
    return 0;
}
