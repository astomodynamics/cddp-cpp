#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include "cddp.hpp"
#include "matplot/matplot.h"

namespace fs = std::filesystem;
using namespace matplot;

int main()
{
    // Problem parameters
    int state_dim = 4;   // [x, y, theta, v]
    int control_dim = 2; // [acceleration, steering_angle]
    int horizon = 100;
    double timestep = 0.05;
    std::string integration_type = "euler";

    // Create a bicycle instance
    double wheelbase = 1.5; // wheelbase length in meters
    std::unique_ptr<cddp::DynamicalSystem> system = std::make_unique<cddp::Bicycle>(timestep, wheelbase, integration_type);

    // Create objective function
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 50.0, 0.0, 0.0, 0.0,
        0.0, 50.0, 0.0, 0.0,
        0.0, 0.0, 10.0, 0.0,
        0.0, 0.0, 0.0, 10.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 5.0, 5.0, M_PI / 2.0, 0.0;

    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Initial state
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, M_PI / 4.0, 0.0;

    // Create CDDP solver and set up the problem
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define and add control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -10.0, -M_PI / 5;
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 10.0, M_PI / 5;
    cddp_solver.addConstraint("ControlBoxConstraint",
                              std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));
    auto constraint = cddp_solver.getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Set options and initial trajectory
    cddp::CDDPOptions options;
    options.max_iterations = 20;
    cddp_solver.setOptions(options);
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem
    cddp::CDDPSolution solution = cddp_solver.solve();

    // Extract solution
    auto X_sol = solution.state_sequence;
    auto U_sol = solution.control_sequence;
    auto t_sol = solution.time_sequence;

    // Create directory for saving plots
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory))
    {
        fs::create_directory(plotDirectory);
    }

    // Create a directory for frame images.
    (void) std::system("mkdir -p frames");

    // Extract trajectory data
    std::vector<double> x_arr, y_arr, theta_arr, v_arr;
    for (const auto &x : X_sol)
    {
        x_arr.push_back(x(0));
        y_arr.push_back(x(1));
        theta_arr.push_back(x(2));
        v_arr.push_back(x(3));
    }

    // Extract control inputs
    std::vector<double> acc_arr, steering_arr;
    for (const auto &u : U_sol)
    {
        acc_arr.push_back(u(0));
        steering_arr.push_back(u(1));
    }

    // -----------------------------
    // Plot states and controls
    // -----------------------------
    auto f1 = figure();
    f1->size(1200, 800);

    // First subplot: Position Trajectory
    auto ax1 = subplot(2, 2, 0);
    auto plot_handle = plot(ax1, x_arr, y_arr, "-b");
    plot_handle->line_width(3);
    title(ax1, "Position Trajectory");
    xlabel(ax1, "x [m]");
    ylabel(ax1, "y [m]");

    // Second subplot: Heading Angle vs Time
    auto ax2 = subplot(2, 2, 1);
    auto heading_plot_handle = plot(ax2, t_sol, theta_arr);
    heading_plot_handle->line_width(3);
    title(ax2, "Heading Angle");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "theta [rad]");

    // Third subplot: Velocity vs Time
    auto ax3 = subplot(2, 2, 2);
    auto velocity_plot_handle = plot(ax3, t_sol, v_arr);
    velocity_plot_handle->line_width(3);
    title(ax3, "Velocity");
    xlabel(ax3, "Time [s]");
    ylabel(ax3, "v [m/s]");

    // Fourth subplot: Control Inputs
    auto ax4 = subplot(2, 2, 3);
    auto p1 = plot(ax4, acc_arr, "--b");
    p1->line_width(3);
    p1->display_name("Acceleration");

    hold(ax4, true);
    auto p2 = plot(ax4, steering_arr, "--r");
    p2->line_width(3);
    p2->display_name("Steering");

    title(ax4, "Control Inputs");
    xlabel(ax4, "Step");
    ylabel(ax4, "Control");
    legend(ax4);

    f1->draw();
    f1->save(plotDirectory + "/bicycle_cddp_results.png");

    // -----------------------------
    // Animation: Bicycle Trajectory
    // -----------------------------
    auto f2 = figure();
    f2->size(800, 600);
    auto ax_anim = f2->current_axes();
    if (!ax_anim)
    {
        ax_anim = axes();
    }

    double car_length = 0.35;
    double car_width = 0.15;

    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        if (i % 10 == 0)
        {
            ax_anim->clear();
            hold(ax_anim, true);

            double x = x_arr[i];
            double y = y_arr[i];
            double theta = theta_arr[i];

            // Compute bicycle rectangle corners
            std::vector<double> car_x(5), car_y(5);
            car_x[0] = x + car_length / 2 * cos(theta) - car_width / 2 * sin(theta);
            car_y[0] = y + car_length / 2 * sin(theta) + car_width / 2 * cos(theta);
            car_x[1] = x + car_length / 2 * cos(theta) + car_width / 2 * sin(theta);
            car_y[1] = y + car_length / 2 * sin(theta) - car_width / 2 * cos(theta);
            car_x[2] = x - car_length / 2 * cos(theta) + car_width / 2 * sin(theta);
            car_y[2] = y - car_length / 2 * sin(theta) - car_width / 2 * cos(theta);
            car_x[3] = x - car_length / 2 * cos(theta) - car_width / 2 * sin(theta);
            car_y[3] = y - car_length / 2 * sin(theta) + car_width / 2 * cos(theta);
            car_x[4] = car_x[0];
            car_y[4] = car_y[0];

            auto car_line = plot(ax_anim, car_x, car_y);
            car_line->color("black");
            car_line->line_style("solid");
            car_line->line_width(2);
            car_line->display_name("Car");

            // Plot the front wheel (steerable)
            double wheel_length = car_width * 0.8;
            double steering_angle = U_sol[std::min(i, U_sol.size() - 1)](1);
            std::vector<double> front_wheel_x = {
                x + car_length / 2 * cos(theta),
                x + car_length / 2 * cos(theta) + wheel_length * cos(theta + steering_angle)};
            std::vector<double> front_wheel_y = {
                y + car_length / 2 * sin(theta),
                y + car_length / 2 * sin(theta) + wheel_length * sin(theta + steering_angle)};
            auto front_wheel_line = plot(ax_anim, front_wheel_x, front_wheel_y);
            front_wheel_line->color("red");
            front_wheel_line->line_style("solid");
            front_wheel_line->line_width(2);
            front_wheel_line->display_name("");

            // Plot trajectory up to current frame
            std::vector<double> traj_x(x_arr.begin(), x_arr.begin() + i + 1);
            std::vector<double> traj_y(y_arr.begin(), y_arr.begin() + i + 1);
            auto traj_line = plot(ax_anim, traj_x, traj_y);
            traj_line->color("blue");
            traj_line->line_style("solid");
            traj_line->line_width(1.5);
            traj_line->display_name("Trajectory");

            title(ax_anim, "Bicycle Trajectory");
            xlabel(ax_anim, "x [m]");
            ylabel(ax_anim, "y [m]");
            xlim(ax_anim, {-1, 6});
            ylim(ax_anim, {-1, 6});
            // legend(ax_anim);

            std::string filename = plotDirectory + "/bicycle_frame_" + std::to_string(i) + ".png";
            f2->draw();
            f2->save(filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 30 " + plotDirectory + "/bicycle_frame_*.png " + plotDirectory + "/bicycle.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/bicycle_frame_*.png";
    std::system(cleanup_command.c_str());

    return 0;
}
