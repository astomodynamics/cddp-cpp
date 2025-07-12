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
#include <cmath>
#include <map>
#include <string>
#include "cddp.hpp"
#include "dynamics_model/forklift.hpp"
#include "matplot/matplot.h"
#include <random>

using namespace matplot;
namespace fs = std::filesystem;

namespace cddp
{

    class ForkliftParkingObjective : public NonlinearObjective
    {
    public:
        ForkliftParkingObjective(const Eigen::VectorXd &goal_state, double timestep, int horizon)
            : NonlinearObjective(timestep), reference_state_(goal_state), horizon_(horizon)
        {
            // Control cost coefficients 
            // R matrix: cu = [acceleration, steering_rate]
            cu_ = Eigen::Vector2d(1.8, 11.0);  
            // Terminal cost coefficients: Qf = [x, y, theta, v, delta]
            cf_ = Eigen::VectorXd(5);
            cf_ << 1000.0, 1000.0, 5000.0, 16.0, 100.0;  // From README Table 4

            // Stage cost coefficients: Q = [x, y, theta, v, delta]
            cx_full_ = Eigen::VectorXd(5);
            cx_full_ << 1.0, 1.0, 10.0, 4.0, 6.0;  // From README Table 4

            // Time penalty weight
            time_weight_ = 0.05;  // From README
        }

        double running_cost(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control,
                            int index) const override
        {
            // Control cost: 0.5 * u^T * R * u
            double lu = 0.5 * cu_.dot(control.array().square().matrix());

            // State cost: 0.5 * (x - x_tar)^T * Q * (x - x_tar)
            Eigen::VectorXd state_error = state - reference_state_;
            double lx = 0.5 * cx_full_.dot(state_error.array().square().matrix());

            return lu + lx;
        }

        double terminal_cost(const Eigen::VectorXd &final_state) const override
        {
            // Terminal cost: 0.5 * (x - x_tar)^T * Qf * (x - x_tar)
            Eigen::VectorXd state_error = final_state - reference_state_;
            double terminal_cost = 0.5 * cf_.dot(state_error.array().square().matrix());
            
            // Add time penalty: w_T * T (simplified as constant per timestep)
            double time_cost = time_weight_ * horizon_ * timestep_;
            
            return terminal_cost + time_cost;
        }

    private:
        Eigen::VectorXd reference_state_;
        Eigen::Vector2d cu_; // Control cost coefficients (R matrix)
        Eigen::VectorXd cf_; // Terminal cost coefficients (Qf matrix)
        Eigen::VectorXd cx_full_; // Stage cost coefficients (Q matrix)
        double time_weight_; // Time penalty weight
        int horizon_;  // Total horizon length
    };
} // namespace cddp

void plotForkliftBox(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
                     double length, double width, bool rear_steer,
                     const std::string &color, axes_handle ax)
{
    double x = state(0);
    double y = state(1);
    double theta = state(2);
    double steering = state(4);  // Steering angle is now a state

    // Compute the forklift's four corners
    std::vector<double> forklift_x(5), forklift_y(5);

    // Front right
    forklift_x[0] = x + length / 2 * cos(theta) - width / 2 * sin(theta);
    forklift_y[0] = y + length / 2 * sin(theta) + width / 2 * cos(theta);

    // Front left
    forklift_x[1] = x + length / 2 * cos(theta) + width / 2 * sin(theta);
    forklift_y[1] = y + length / 2 * sin(theta) - width / 2 * cos(theta);

    // Rear left
    forklift_x[2] = x - length / 2 * cos(theta) + width / 2 * sin(theta);
    forklift_y[2] = y - length / 2 * sin(theta) - width / 2 * cos(theta);

    // Rear right
    forklift_x[3] = x - length / 2 * cos(theta) - width / 2 * sin(theta);
    forklift_y[3] = y - length / 2 * sin(theta) + width / 2 * cos(theta);

    // Close polygon
    forklift_x[4] = forklift_x[0];
    forklift_y[4] = forklift_y[0];

    // Plot forklift body as a polygon line
    plot(ax, forklift_x, forklift_y, color + "-");

    // Plot base point (center of rear axle) as a red circle
    plot(ax, std::vector<double>{x}, std::vector<double>{y}, "ro");

    // Plot fork position (front of forklift)
    double fork_length = width * 0.8;
    double fork_x_start = x + length / 2 * cos(theta);
    double fork_y_start = y + length / 2 * sin(theta);
    double fork_x_end = fork_x_start + fork_length * cos(theta);
    double fork_y_end = fork_y_start + fork_length * sin(theta);
    
    plot(ax, std::vector<double>{fork_x_start, fork_x_end},
         std::vector<double>{fork_y_start, fork_y_end}, "m-");

    // Plot steering direction
    if (rear_steer) {
        // For rear-steer, show steering at rear axle
        double rear_x = x - length / 2 * cos(theta);
        double rear_y = y - length / 2 * sin(theta);
        double steering_length = width / 2;
        double steering_angle = theta - steering;  // Rear steer is opposite
        double steering_end_x = rear_x + steering_length * cos(steering_angle);
        double steering_end_y = rear_y + steering_length * sin(steering_angle);
        
        plot(ax, std::vector<double>{rear_x, steering_end_x},
             std::vector<double>{rear_y, steering_end_y}, "g-");
    } else {
        // For front-steer, show steering at front axle
        double front_x = x + length / 2 * cos(theta);
        double front_y = y + length / 2 * sin(theta);
        double steering_length = width / 2;
        double steering_angle = theta + steering;
        double steering_end_x = front_x + steering_length * cos(steering_angle);
        double steering_end_y = front_y + steering_length * sin(steering_angle);
        
        plot(ax, std::vector<double>{front_x, steering_end_x},
             std::vector<double>{front_y, steering_end_y}, "g-");
    }
}


int main() {
    // Problem parameters
    const int state_dim = 5;     // [x, y, theta, v, delta]
    const int control_dim = 2;   // [acceleration, steering_rate]
    const int horizon = 600;
    const double timestep = 0.03;
    const std::string integration_type = "rk4";

    // Create a Forklift instance 
    double wheelbase = 1.6; 
    bool rear_steer = true; 
    double max_steering_angle = 0.9;  
    
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Forklift>(timestep, wheelbase, integration_type, 
                                        rear_steer, max_steering_angle);

    // Define initial and goal states for forklift back-in parking
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 2.0, 0.5, M_PI, 0.0, 0.0;  
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0, M_PI/2, 0.0, 0.0;

    // Create the nonlinear objective for forklift parking
    auto objective = std::make_unique<cddp::ForkliftParkingObjective>(goal_state, timestep, horizon);

    // Set solver options for better convergence
    cddp::CDDPOptions options;
    options.max_iterations = 2000; 
    options.verbose = true;  
    options.tolerance = 1e-5;  // Less strict tolerance
    options.acceptable_tolerance = 1e-5;  
    options.regularization.initial_value = 1e-6;
    options.debug = false;
    options.use_ilqr = true;
    options.enable_parallel = true;
    options.num_threads = 10;
    
    options.msipddp.barrier.mu_initial = 1e-1;  // Lower barrier parameter
    options.msipddp.dual_var_init_scale = 1e-1;
    options.msipddp.slack_var_init_scale = 1e-2;
    options.msipddp.segment_length = horizon;  // Longer segments
    options.msipddp.rollout_type = "nonlinear";

    // Create CDDP solver for the forklift model
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::move(system), std::move(objective), options);

    // Define control constraints from README Table 1
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -1.5, -0.4;   // [acceleration, steering_rate]
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 1.5, 0.4;   
    cddp_solver.addPathConstraint("ControlConstraint", 
                                 std::make_unique<cddp::ControlConstraint>(
                                     control_upper_bound, control_lower_bound));

    // State constraints 
    Eigen::VectorXd state_upper_bound(2);
    state_upper_bound << 2.0, max_steering_angle; 
    cddp_solver.addPathConstraint("StateConstraint",
                                 std::make_unique<cddp::StateConstraint>(state_upper_bound));

    // Initialize trajectory for back-in parking maneuver
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    
    for (int i = 0; i < horizon; ++i) {
        // Randomize controls
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.01, 0.01);
        U[i] << dis(gen), dis(gen);
        X[i + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[i], U[i], i * timestep);
    }
    
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem using MSIPDDP
    cddp::CDDPSolution solution = cddp_solver.solve(cddp::SolverType::MSIPDDP);

    // Extract solution trajectories
    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points"));

    // Prepare trajectory data for plotting
    std::vector<double> x_hist, y_hist;
    for (const auto& state : X_sol) {
        x_hist.push_back(state(0));
        y_hist.push_back(state(1));
    }
    
    // Forklift dimensions 
    double forklift_length = 2.5;  // Overall length
    double forklift_width = 1.1;   // Chassis width

    // Create a figure and get current axes
    auto fig = figure(true);
    auto ax = fig->current_axes();

    Eigen::VectorXd empty_control = Eigen::VectorXd::Zero(2);

    // Create directory for saving plots
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory))
    {
        fs::create_directories(plotDirectory);
    }

    // Animation loop: update plot for each time step and save frame
    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        // Skip every 10th frame for smoother animation
        if (i % 10 == 0)
        {
            // Clear previous content
            cla(ax);
            hold(ax, true);

            // Plot the full trajectory
            plot(ax, x_hist, y_hist, "b--");

            // Plot goal configuration
            plotForkliftBox(goal_state, empty_control, forklift_length, 
                           forklift_width, rear_steer, "r", ax);

            // Plot current forklift state
            if (i < U_sol.size())
                plotForkliftBox(X_sol[i], U_sol[i], forklift_length, 
                               forklift_width, rear_steer, "k", ax);
            else
                plotForkliftBox(X_sol[i], empty_control, forklift_length, 
                               forklift_width, rear_steer, "k", ax);

            // Add labels and title
            xlabel(ax, "X [m]");
            ylabel(ax, "Y [m]");
            title(ax, "Forklift Parking with IPDDP - Frame " + std::to_string(i));

            // Set grid and axis limits
            grid(ax, true);
            xlim(ax, {-5, 5});
            ylim(ax, {-5, 5});

            // Update drawing
            fig->draw();

            // Save the frame to a PNG file
            std::string frame_filename = plotDirectory + "/forklift_frame_" + std::to_string(i) + ".png";
            fig->save(frame_filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // Combine all saved frames into a GIF using ImageMagick's convert tool
    std::string command = "convert -delay 10 " + plotDirectory + "/forklift_frame_*.png " 
                         + plotDirectory + "/forklift_parking_ipddp.gif";
    std::system(command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/forklift_frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "Animation saved as forklift_parking_ipddp.gif" << std::endl;

    // Print final state
    std::cout << "\nFinal state:" << std::endl;
    std::cout << "Position: (" << X_sol.back()(0) << ", " << X_sol.back()(1) << ")" << std::endl;
    std::cout << "Heading: " << X_sol.back()(2) * 180.0 / M_PI << " degrees" << std::endl;
    std::cout << "Velocity: " << X_sol.back()(3) << " m/s" << std::endl;
    std::cout << "Steering angle: " << X_sol.back()(4) * 180.0 / M_PI << " degrees" << std::endl;

    return 0;
}