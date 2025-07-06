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
#include "dynamics_model/spacecraft_roe.hpp" // Added for SpacecraftROE
namespace plt = matplot;

namespace cddp
{
    class SumOfTwoNormObjective : public NonlinearObjective
    {
    public:
        SumOfTwoNormObjective(const Eigen::VectorXd &goal_state_roe, // Renamed for clarity
                              double weight_running_control,
                              double weight_terminal_state,
                              double timestep,
                              double a_ref_m)
            : NonlinearObjective(timestep),
              goal_state_roe_(goal_state_roe), // Renamed for clarity
              weight_running_control_(weight_running_control),
              weight_terminal_state_(weight_terminal_state),
              a_ref_m_(a_ref_m)
        {
        }

        double running_cost(const Eigen::VectorXd &state_roe, // Renamed for clarity
                            const Eigen::VectorXd &control,
                            int /*index*/) const override
        {
            double control_squared = control.squaredNorm();
            if (control_squared < 1e-10)
            {
                return 0.0;
            }
            double control_norm = std::sqrt(control_squared + epsilon_);
            return weight_running_control_ * control_norm;
        }

        double terminal_cost(const Eigen::VectorXd &final_state_roe) const override // Renamed for clarity
        {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(final_state_roe.size(), final_state_roe.size());
            Q(6, 6) = 0.0;
            Eigen::VectorXd state_error = (final_state_roe - goal_state_roe_) * a_ref_m_;
            return weight_terminal_state_ * state_error.transpose() * Q * state_error;
        }

        // Analytical Gradients
        Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd &state_roe, const Eigen::VectorXd & /*control*/, int /*index*/) const override
        {
            return Eigen::VectorXd::Zero(state_roe.size());
        }

        Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd & /*state_roe*/, const Eigen::VectorXd &control, int /*index*/) const override
        {
            double norm_u = std::sqrt(control.squaredNorm() + epsilon_);
            if (norm_u < 1e-8)
            {
                return Eigen::VectorXd::Zero(control.size());
            }
            return weight_running_control_ * control / norm_u;
        }

        Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd &final_state_roe) const override
        {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(final_state_roe.size(), final_state_roe.size());  
            Q(6, 6) = 0.0;
            Eigen::VectorXd state_error = (final_state_roe - goal_state_roe_) * a_ref_m_;
            return 2.0 * weight_terminal_state_ * Q * state_error;
        }

        // Analytical Hessians
        Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd &state_roe, const Eigen::VectorXd & /*control*/, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(state_roe.size(), state_roe.size());
        }

        Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd & /*state_roe*/, const Eigen::VectorXd &control, int /*index*/) const override
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

        Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd &state_roe, const Eigen::VectorXd &control, int /*index*/) const override
        {
            return Eigen::MatrixXd::Zero(control.size(), state_roe.size());
        }

        Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd & /*final_state_roe*/) const override
        {
            int state_dim = goal_state_roe_.size();
            Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
            Q(6, 6) = 0.0;
            return 2.0 * weight_terminal_state_ * Q;
        }

    private:
        Eigen::VectorXd goal_state_roe_; // Stores goal state in ROE coordinates
        double weight_running_control_;
        double weight_terminal_state_;
        double epsilon_ = 1e-6;
        double a_ref_m_;
    };
} // namespace cddp

namespace fs = std::filesystem;
using namespace cddp;

int main()
{
    // Optimization horizon info
    int horizon = 500;                  // Optimization horizon length
    double dt = 20.0; // Time step for optimization
    int state_dim = SpacecraftROE::STATE_DIM; // Use consistent state dimension from SpacecraftROE
    int control_dim = SpacecraftROE::CONTROL_DIM; // Use consistent control dimension

    // SpacecraftROE parameters
    double a_ref_m = (6371.0 + 500.0) * 1e3; // Reference semi-major axis (m) e.g. LEO
    double nu0_rad = 0.0;             // Initial argument of latitude for ROE (rad)
    double mass_kg = 1.0;           // Mass in kg
    double period = 2 * M_PI * sqrt(pow(a_ref_m, 3) / 3.9860044e14); // Orbital period (s)
    double mean_motion = 2 * M_PI / period;

    // Input constraints
    double u_force_max_N = 0.05;

    // Cost weighting for SumOfTwoNormObjective
    double weight_running_control = 1.0;
    double weight_terminal_state = 10.0;

    // Create the SpacecraftROE system for optimization
    std::unique_ptr<cddp::DynamicalSystem> roe_system =
        std::make_unique<SpacecraftROE>(dt, "euler", a_ref_m, nu0_rad);

    // Initial state
    Eigen::VectorXd initial_state_roe(state_dim);
    // initial_state_roe << 0.0, 1000.0, 0.0, 500.0, 0.0, 700.0, 0.0; // Time is the 7th element
    // initial_state_roe /= a_ref_m;
    // initial_state_roe(6) = 0.0;
    Eigen::VectorXd initial_state_hcw(6);
    initial_state_hcw << 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0;
    if (auto* roe_model = dynamic_cast<cddp::SpacecraftROE*>(roe_system.get())) {
        initial_state_roe = roe_model->transformHCWToROE(initial_state_hcw, 0.0);
    } else {
        std::cerr << "Error: roe_system is not a SpacecraftROE model at initial_state_roe." << std::endl;
        return 1; // Or handle error appropriately
    }

    // Final state
    Eigen::VectorXd goal_state_roe(state_dim);
    // goal_state_roe << 0.0, 100.0, 0.0, 300.0, 0.0, 400.0, 0.0; // Time is the 7th element, init to 0.0
    // goal_state_roe /= a_ref_m;
    // goal_state_roe(6) = horizon * dt; // Set target time to the end of the horizon
    Eigen::VectorXd goal_state_hcw(6);
    goal_state_hcw << -100.0, 0.0, 0.0, 0.0, 200.0 * mean_motion, 0.0;
    if (auto* roe_model = dynamic_cast<cddp::SpacecraftROE*>(roe_system.get())) {
        goal_state_roe = roe_model->transformHCWToROE(goal_state_hcw, horizon * dt);
    } else {
        std::cerr << "Error: roe_system is not a SpacecraftROE model at goal_state_roe." << std::endl;
        return 1; // Or handle error appropriately
    }

    // Build cost objective
    auto objective = std::make_unique<cddp::SumOfTwoNormObjective>(
        goal_state_roe,
        weight_running_control,
        weight_terminal_state,
        dt,
        a_ref_m);

    // ======= Initial and final trajectory =======
    int num_steps = static_cast<int>(12.0 * period / dt); // Simulate for 3 orbits
    // Initial and final trajectory
    std::vector<Eigen::VectorXd> X_roe_initial(num_steps + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_accel_initial(num_steps, Eigen::VectorXd::Zero(control_dim));
    X_roe_initial[0] = initial_state_roe;
    for (int i = 0; i < num_steps; ++i) // Iterate up to num_steps
    {
        X_roe_initial[i + 1] = roe_system->getDiscreteDynamics(X_roe_initial[i], U_accel_initial[i], i * dt);
    }
    std::vector<Eigen::VectorXd> X_roe_final(num_steps + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U_accel_final(num_steps, Eigen::VectorXd::Zero(control_dim));
    X_roe_final[0] = goal_state_roe;
    for (int i = 0; i < num_steps; ++i) // Iterate up to num_steps
    {
        X_roe_final[i + 1] = roe_system->getDiscreteDynamics(X_roe_final[i], U_accel_final[i], i * dt);
    }

    // Convert initial and final trajectory to HCW coordinates vector of doubles
    std::vector<std::vector<double>> X_hcw_initial(num_steps + 1);
    std::vector<std::vector<double>> X_hcw_final(num_steps + 1);
    for (int i = 0; i < num_steps + 1; ++i)
    {
        if (auto* roe_model = dynamic_cast<cddp::SpacecraftROE*>(roe_system.get())) {
            Eigen::VectorXd state_hcw_initial = roe_model->transformROEToHCW(X_roe_initial[i].head(6), X_roe_initial[i][6]);
            Eigen::VectorXd state_hcw_final = roe_model->transformROEToHCW(X_roe_final[i].head(6), X_roe_final[i][6]);
            X_hcw_initial[i] = {state_hcw_initial(0), state_hcw_initial(1), state_hcw_initial(2), state_hcw_initial(3), state_hcw_initial(4), state_hcw_initial(5)};
            X_hcw_final[i] = {state_hcw_final(0), state_hcw_final(1), state_hcw_final(2), state_hcw_final(3), state_hcw_final(4), state_hcw_final(5)};
        } else {
            std::cerr << "Error: roe_system is not a SpacecraftROE model during trajectory conversion." << std::endl;
            // Handle error appropriately, maybe clear X_hcw_initial[i] or return
        }
    }

    // Prepare data for 3D plotting
    std::vector<double> initial_x_hcw(num_steps + 1), initial_y_hcw(num_steps + 1), initial_z_hcw(num_steps + 1);
    std::vector<double> initial_t(num_steps + 1);
    std::vector<double> final_x_hcw(num_steps + 1), final_y_hcw(num_steps + 1), final_z_hcw(num_steps + 1);
    std::vector<double> final_t(num_steps + 1);

    for (int i = 0; i < num_steps + 1; ++i) {
        if (X_hcw_initial[i].size() >= 3) { // Ensure there are enough elements
            initial_x_hcw[i] = X_hcw_initial[i][0];
            initial_y_hcw[i] = X_hcw_initial[i][1];
            initial_z_hcw[i] = X_hcw_initial[i][2];
            initial_t[i] = X_roe_initial[i][6];
        }
        if (X_hcw_final[i].size() >= 3) { // Ensure there are enough elements
            final_x_hcw[i] = X_hcw_final[i][0];
            final_y_hcw[i] = X_hcw_final[i][1];
            final_z_hcw[i] = X_hcw_final[i][2];
            final_t[i] = X_roe_final[i][6];
        }
    }
    // ======= End of Initial and final trajectory =======

    // // // Plot 3d initial and final trajectory
    // plt::figure();
    // plt::plot3(initial_x_hcw, initial_y_hcw, initial_z_hcw, "-")->line_width(1).display_name("Initial Trajectory");
    // plt::hold(true);
    // plt::plot3(final_x_hcw, final_y_hcw, final_z_hcw, "-")->line_width(1).display_name("Final Trajectory");
    // plt::hold(false);
    // plt::xlabel("x_hcw (km)");
    // plt::ylabel("y_hcw (km)");
    // plt::zlabel("z_hcw (km)");
    // plt::legend();
    // plt::title("Initial and Final Trajectories (HCW)");
    // plt::axis(plt::equal);

    // // Plot time history
    // plt::figure();
    // plt::plot(initial_t, initial_x_hcw, "-")->line_width(1).display_name("Initial Trajectory");
    // plt::hold(true);
    // plt::plot(final_t, final_x_hcw, "-")->line_width(1).display_name("Final Trajectory");
    // plt::hold(false);
    // plt::xlabel("Time (s)");
    // plt::show();
    
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    for (int i = 0; i < horizon; ++i)
    {
        X[i] = initial_state_roe;
        U[i] = Eigen::VectorXd::Zero(control_dim);
    }
    X[horizon] = initial_state_roe;
    
    // Solver options.
    cddp::CDDPOptions options;
    options.max_iterations = 1000;
    options.tolerance = 1e-6;
    options.regularization.initial_value = 1e-5;
    options.use_ilqr = true;
    options.enable_parallel = true;
    options.num_threads = 12;
    options.debug = false;
    options.ipddp.barrier.mu_initial = 1e-1;
    options.msipddp.segment_length = horizon;
    options.msipddp.rollout_type = "nonlinear";
    options.ipddp.barrier.mu_update_factor = 0.2;
    options.ipddp.barrier.mu_update_power = 1.2;

    // Create CDDP solver.
    cddp::CDDP cddp_solver(
        initial_state_roe,
        goal_state_roe,
        horizon,
        dt,
        std::move(roe_system),
        std::move(objective),
        options);
    cddp_solver.setInitialTrajectory(X, U);

    // goal_state^T * goal_state
    std::cout << "initial_state^T * initial_state: " << initial_state_roe.transpose() * initial_state_roe << std::endl;
    std::cout << "initial_state: " << initial_state_roe.transpose() << std::endl;
    std::cout << "goal_state^T * goal_state: " << goal_state_roe.transpose() * goal_state_roe << std::endl;
    std::cout << "goal_state: " << goal_state_roe.transpose() << std::endl;

    // // Add Control Constraint 
    // Eigen::VectorXd u_upper_accel = Eigen::VectorXd::Constant(control_dim, u_force_max_N);
    // cddp_solver.addPathConstraint("ControlConstraint",
    //     std::make_unique<cddp::ControlConstraint>(u_upper_accel));

    // Solve the Trajectory Optimization Problem
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Extract the solution
    std::vector<Eigen::VectorXd> X_solution_roe = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    std::vector<Eigen::VectorXd> U_solution_accel = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));

    // Print the solution
    // std::cout << "Solution: " << solution.cost << std::endl;
    std::cout << "Solution: " << X_solution_roe.back().transpose() * a_ref_m << std::endl;
    std::cout << "Solution: " << U_solution_accel.back().transpose() << std::endl;

    if (!X_solution_roe.empty() && !U_solution_accel.empty())
    {
        namespace plt = matplot;

        // Create a new SpacecraftROE instance for transformations,
        // as the original roe_system was moved into the solver.
        SpacecraftROE roe_transformer_for_plotting(dt, "euler", a_ref_m, nu0_rad);

        // Create time vectors
        std::vector<double> t_states(horizon + 1);
        for(int i = 0; i <= horizon; ++i) t_states[i] = i * dt;

        std::vector<double> t_controls(horizon);
        for(int i = 0; i < horizon; ++i) t_controls[i] = i * dt;

        // Transform ROE states to HCW states (m) for plotting
        std::vector<double> x_pos_m(horizon + 1), y_pos_m(horizon + 1), z_pos_m(horizon + 1);
        std::vector<double> vx_mps(horizon + 1), vy_mps(horizon + 1), vz_mps(horizon + 1);

        for (size_t i = 0; i < X_solution_roe.size(); ++i) {
            const auto& state_roe_current = X_solution_roe[i];
            if (state_roe_current.size() == state_dim) {
                double current_time = state_roe_current(6);
                Eigen::VectorXd state_hcw_m = roe_transformer_for_plotting.transformROEToHCW(state_roe_current.head(6), current_time);

                x_pos_m[i] = state_hcw_m(0);
                y_pos_m[i] = state_hcw_m(1);
                z_pos_m[i] = state_hcw_m(2);
                vx_mps[i]  = state_hcw_m(3);
                vy_mps[i]  = state_hcw_m(4);
                vz_mps[i]  = state_hcw_m(5);
            }
        }

        std::vector<double> ur_accel(horizon), ut_accel(horizon), un_accel(horizon); // Accelerations in km/s^2
        std::vector<double> accel_magnitude_m_s2(horizon);

        for (size_t i = 0; i < U_solution_accel.size(); ++i) {
            const auto& control_accel = U_solution_accel[i];
            if (control_accel.size() == control_dim) {
                ur_accel[i] = control_accel(0); // m/s^2
                ut_accel[i] = control_accel(1); // m/s^2
                un_accel[i] = control_accel(2); // m/s^2
                accel_magnitude_m_s2[i] = control_accel.norm(); // m/s^2
            }
        }

        // --- Generate Plots ---
        // plt::figure_size(1200, 800);

        // 1. Position Trajectories (HCW in meters)
        plt::figure();
        plt::plot(t_states, x_pos_m)->line_width(2).display_name("x_hcw (m)");
        plt::hold(true);
        plt::plot(t_states, y_pos_m)->line_width(2).display_name("y_hcw (m)");
        plt::plot(t_states, z_pos_m)->line_width(2).display_name("z_hcw (m)");
        plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Position (m)");
        plt::legend();
        plt::title("HCW Position vs. Time");

        // 2. Velocity Trajectories (HCW in m/s)
        plt::figure();
        plt::plot(t_states, vx_mps)->line_width(2).display_name("vx_hcw (m/s)");
        plt::hold(true);
        plt::plot(t_states, vy_mps)->line_width(2).display_name("vy_hcw (m/s)");
        plt::plot(t_states, vz_mps)->line_width(2).display_name("vz_hcw (m/s)");
        plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Velocity (m/s)");
        plt::legend();
        plt::title("HCW Velocity vs. Time");

        // 5. Control Inputs (Accelerations in km/s^2, ZOH)
        plt::figure();
        plt::stairs(t_controls, ur_accel)->line_width(2).display_name("u_r (km/s^2)");
        plt::hold(true);
        plt::stairs(t_controls, ut_accel)->line_width(2).display_name("u_t (km/s^2)");
        plt::stairs(t_controls, un_accel)->line_width(2).display_name("u_n (km/s^2)");
        plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Control Acceleration (km/s^2)");
        plt::legend();
        plt::title("Control Inputs (ZOH) vs. Time");

        // 6. Acceleration Magnitude (km/s^2, ZOH)
        plt::figure();
        plt::stairs(t_controls, accel_magnitude_m_s2)->line_width(2).display_name("||Acceleration|| (m/s^2) (ZOH)");
        // plt::hold(true);
        // plt::plot(t_controls, std::vector<double>(horizon, u_max_norm_accel), "--r")->display_name("Max Norm Accel");
        // if (u_min_norm_accel > 0) plt::plot(t_controls, std::vector<double>(horizon, u_min_norm_accel), "--y")->display_name("Min Norm Accel");
        // plt::hold(false);
        plt::xlabel("Time (s)");
        plt::ylabel("Acceleration Magnitude (m/s^2)");
        plt::legend();
        plt::title("Acceleration Magnitude (ZOH) vs. Time");

        // 7. X-Y plane trajectory (HCW in meters, x-axis vertical, y-axis horizontal to the left)
        plt::figure();
        plt::plot(y_pos_m, x_pos_m)->line_width(2).display_name("Trajectory (HCW)");
        plt::hold(true);
        if (!x_pos_m.empty() && !y_pos_m.empty()){
             plt::scatter(std::vector<double>{y_pos_m.front()}, std::vector<double>{x_pos_m.front()})
                ->marker_color("g").marker_style("o").marker_size(10).display_name("Start");
             plt::scatter(std::vector<double>{y_pos_m.back()}, std::vector<double>{x_pos_m.back()})
                ->marker_color("r").marker_style("x").marker_size(10).display_name("End");
        }
        // plt::plot(circle_y_m, circle_x_m)->line_width(2).display_name("Ball Constraint (HCW)");
        plt::hold(false);
        plt::xlabel("y_hcw (m)");
        plt::ylabel("x_hcw (m)");
        plt::legend();
        plt::title("X-Y Plane Trajectory (HCW)");
        plt::axis(plt::equal);
        plt::gca()->x_axis().reverse(true);

        // 8. 3D Trajectory (HCW in meters)
        plt::figure();
        plt::plot3(x_pos_m, y_pos_m, z_pos_m, "-o")->line_width(2).marker_size(4).display_name("Trajectory (HCW)");
        plt::hold(true);
        if (!x_pos_m.empty()){
             plt::scatter3(std::vector<double>{x_pos_m.front()}, std::vector<double>{y_pos_m.front()}, std::vector<double>{z_pos_m.front()})
                ->marker_color("g").marker_style("o").marker_size(10).display_name("Start");
             plt::scatter3(std::vector<double>{x_pos_m.back()}, std::vector<double>{y_pos_m.back()}, std::vector<double>{z_pos_m.back()})
                ->marker_color("r").marker_style("x").marker_size(10).display_name("End");
        }
        // Could also plot the ball constraint here if it's a sphere.
        plt::hold(false);
        plt::xlabel("x_hcw (m)");
        plt::ylabel("y_hcw (m)");
        plt::zlabel("z_hcw (m)");
        plt::legend();
        plt::title("3D Trajectory (HCW)");
        plt::axis(plt::equal);

        plt::show();
        std::cout << "Plotting complete." << std::endl;
    }
    else
    {
        std::cout << "Solver did not find a solution, or solution variables are not available. Skipping plots." << std::endl;
    }
    return 0;
}
