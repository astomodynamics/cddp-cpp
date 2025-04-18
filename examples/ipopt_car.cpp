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
#include <chrono>
#include <filesystem>
#include <cmath>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

void plotCarBox(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
                double length, double width, const std::string &color,
                axes_handle ax)
{
    double x = state(0);
    double y = state(1);
    double theta = state(2);
    double steering = control(1);

    // Compute the car's four corners (and close the polygon)
    std::vector<double> car_x(5), car_y(5);

    // Front right
    car_x[0] = x + length / 2 * cos(theta) - width / 2 * sin(theta);
    car_y[0] = y + length / 2 * sin(theta) + width / 2 * cos(theta);

    // Front left
    car_x[1] = x + length / 2 * cos(theta) + width / 2 * sin(theta);
    car_y[1] = y + length / 2 * sin(theta) - width / 2 * cos(theta);

    // Rear left
    car_x[2] = x - length / 2 * cos(theta) + width / 2 * sin(theta);
    car_y[2] = y - length / 2 * sin(theta) - width / 2 * cos(theta);

    // Rear right
    car_x[3] = x - length / 2 * cos(theta) - width / 2 * sin(theta);
    car_y[3] = y - length / 2 * sin(theta) + width / 2 * cos(theta);

    // Close polygon
    car_x[4] = car_x[0];
    car_y[4] = car_y[0];

    // Plot car body as a polygon line.
    plot(ax, car_x, car_y, color + "-");

    // Plot base point (center of rear axle) as a red circle.
    plot(ax, std::vector<double>{x}, std::vector<double>{y}, "ro");

    // Compute steering direction
    double front_x = x + length / 2 * cos(theta);
    double front_y = y + length / 2 * sin(theta);
    double steering_length = width / 2;
    double steering_angle = theta + steering;
    double steering_end_x = front_x + steering_length * cos(steering_angle);
    double steering_end_y = front_y + steering_length * sin(steering_angle);

    std::vector<double> steer_x = {front_x, steering_end_x};
    std::vector<double> steer_y = {front_y, steering_end_y};
    plot(ax, steer_x, steer_y, "g-");
}

int main()
{
    // Problem parameters
    int state_dim = 4;   // [x y theta v]
    int control_dim = 2; // [steering_angle acceleration]
    int horizon = 500;   // Same as CDDP example
    double timestep = 0.03;
    double wheelbase = 2.0;

    // Initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 1.0, 1.0, 1.5 * M_PI, 0.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0, 0.0, 0.0;

    // Define variables
    casadi::MX x = casadi::MX::sym("x", state_dim);
    casadi::MX u = casadi::MX::sym("u", control_dim);

    // Car dynamics
    casadi::MX theta = x(2);
    casadi::MX v = x(3);
    casadi::MX delta = u(0);
    casadi::MX a = u(1);

    // Continuous dynamics
    casadi::MX dx = casadi::MX::zeros(state_dim);
    using casadi::cos;
    using casadi::sin;
    using casadi::tan;

    dx(0) = v * cos(theta);
    dx(1) = v * sin(theta);
    dx(2) = v * tan(delta) / wheelbase;
    dx(3) = a;

    // Discrete dynamics using Euler integration
    casadi::MX f = x + timestep * dx;
    casadi::Function F("F", {x, u}, {f});

    // Decision variables
    int n_states = (horizon + 1) * state_dim;
    int n_controls = horizon * control_dim;
    casadi::MX X = casadi::MX::sym("X", n_states);
    casadi::MX U = casadi::MX::sym("U", n_controls);
    casadi::MX z = casadi::MX::vertcat({X, U});

    // Objective function terms
    auto running_cost = [&](casadi::MX x, casadi::MX u)
    {
        casadi::MX u_cost = 1e-2 * u(0) * u(0) + 1e-4 * u(1) * u(1);
        casadi::MX xy_norm = x(0) * x(0) + x(1) * x(1);
        casadi::MX x_cost = 1e-3 * (casadi::MX::sqrt(xy_norm / 0.01 + 1) * 0.1 - 0.1);
        return u_cost + x_cost;
    };

    auto terminal_cost = [&](casadi::MX x)
    {
        casadi::MX cost = 0;
        casadi::MX d = x - casadi::DM(std::vector<double>(goal_state.data(), goal_state.data() + state_dim));
        casadi::MX xy_err = d(0) * d(0) + d(1) * d(1);
        cost += 0.1 * (casadi::MX::sqrt(xy_err / 0.01 + 1) * 0.1 - 0.1);
        cost += 1.0 * (casadi::MX::sqrt(d(2) * d(2) / 0.01 + 1) * 0.1 - 0.1);
        cost += 0.3 * (casadi::MX::sqrt(d(3) * d(3) / 1.0 + 1) * 1.0 - 1.0);
        return cost;
    };

    // Build objective
    casadi::MX J = 0;
    for (int t = 0; t < horizon; t++)
    {
        casadi::MX x_t = X(casadi::Slice(t * state_dim, (t + 1) * state_dim));
        casadi::MX u_t = U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
        J += running_cost(x_t, u_t);
    }
    J += terminal_cost(X(casadi::Slice(horizon * state_dim, (horizon + 1) * state_dim)));

    // Constraints
    casadi::MX g;

    // Initial condition
    g = casadi::MX::vertcat({g, X(casadi::Slice(0, state_dim)) -
                                    casadi::DM(std::vector<double>(initial_state.data(), initial_state.data() + state_dim))});

    // Dynamics constraints
    for (int t = 0; t < horizon; t++)
    {
        casadi::MX x_t = X(casadi::Slice(t * state_dim, (t + 1) * state_dim));
        casadi::MX u_t = U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
        casadi::MX x_next = F(casadi::MXVector{x_t, u_t})[0];
        casadi::MX x_next_sym = X(casadi::Slice((t + 1) * state_dim, (t + 2) * state_dim));
        g = casadi::MX::vertcat({g, x_next_sym - x_next});
    }

    // NLP
    casadi::MXDict nlp = {
        {"x", z},
        {"f", J},
        {"g", g}};

    // Solver options
    casadi::Dict solver_opts;
    solver_opts["ipopt.max_iter"] = 1000;
    solver_opts["ipopt.print_level"] = 5;
    solver_opts["print_time"] = true;
    solver_opts["ipopt.tol"] = 1e-6;

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Bounds
    std::vector<double> lbx(n_states + n_controls);
    std::vector<double> ubx(n_states + n_controls);
    std::vector<double> lbg(g.size1());
    std::vector<double> ubg(g.size1());

    // State bounds
    for (int t = 0; t <= horizon; t++)
    {
        for (int i = 0; i < state_dim; i++)
        {
            lbx[t * state_dim + i] = -1e20;
            ubx[t * state_dim + i] = 1e20;
        }
    }

    // Control bounds
    for (int t = 0; t < horizon; t++)
    {
        // Steering angle bounds
        lbx[n_states + t * control_dim] = -0.5;
        ubx[n_states + t * control_dim] = 0.5;
        // Acceleration bounds
        lbx[n_states + t * control_dim + 1] = -2.0;
        ubx[n_states + t * control_dim + 1] = 2.0;
    }

    // Constraint bounds (all equality constraints)
    for (int i = 0; i < g.size1(); i++)
    {
        lbg[i] = 0;
        ubg[i] = 0;
    }

    // Initial guess
    std::vector<double> x0(n_states + n_controls, 0.0);

    // Initial state
    for (int i = 0; i < state_dim; i++)
    {
        x0[i] = initial_state(i);
    }

    // Linear interpolation for states
    for (int t = 1; t <= horizon; t++)
    {
        for (int i = 0; i < state_dim; i++)
        {
            x0[t * state_dim + i] = initial_state(i) +
                                    (goal_state(i) - initial_state(i)) * t / horizon;
        }
    }

    // Call the solver
    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(casadi::DMDict{
        {"x0", x0},
        {"lbx", lbx},
        {"ubx", ubx},
        {"lbg", lbg},
        {"ubg", ubg}});
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "Solve time: " << duration.count() << " microseconds" << std::endl;

    // EXIT: Optimal Solution Found.
    //     solver  :   t_proc      (avg)   t_wall      (avg)    n_eval
    //     nlp_f  |  23.27ms (217.46us)  23.44ms (219.05us)       107
    //     nlp_g  |  27.51ms (257.09us)  27.57ms (257.65us)       107
    // nlp_grad_f  |  39.72ms (374.68us)  39.87ms (376.14us)       106
    // nlp_hess_l  | 244.66ms (  2.35ms) 245.20ms (  2.36ms)       104
    // nlp_jac_g  | 176.39ms (  1.66ms) 176.76ms (  1.67ms)       106
    //     total  |   1.49 s (  1.49 s)   1.49 s (  1.49 s)         1
    // Solve time: 1488881 microseconds

    // Extract solution
    std::vector<double> sol = std::vector<double>(res.at("x"));

    // Convert to state and control trajectories
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));

    for (int t = 0; t <= horizon; t++)
    {
        for (int i = 0; i < state_dim; i++)
        {
            X_sol[t](i) = sol[t * state_dim + i];
        }
    }

    for (int t = 0; t < horizon; t++)
    {
        for (int i = 0; i < control_dim; i++)
        {
            U_sol[t](i) = sol[n_states + t * control_dim + i];
        }
    }

    // Prepare trajectory data
    std::vector<double> x_hist, y_hist;
    for (const auto &x : X_sol)
    {
        x_hist.push_back(x(0));
        y_hist.push_back(x(1));
    }

    // Car dimensions.
    double car_length = 2.1;
    double car_width = 0.9;

    // Create a figure and get current axes.
    auto fig = figure(true);
    auto ax = fig->current_axes();

    Eigen::VectorXd empty_control = Eigen::VectorXd::Zero(2);

    // Create directory for saving plots
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory))
    {
        fs::create_directory(plotDirectory);
    }

    // Create a directory for frame images.
    (void) std::system("mkdir -p frames");

    // Animation loop: update plot for each time step and save frame.
    for (size_t i = 0; i < X_sol.size(); ++i)
    {
        // Skip every 10th frame for smoother animation.
        if (i % 10 == 0)
        {
            // Clear previous content.
            cla(ax);
            hold(ax, true);

            // Plot the full trajectory.
            plot(ax, x_hist, y_hist, "b-");

            // Plot goal configuration.
            plotCarBox(goal_state, empty_control, car_length, car_width, "r", ax);

            // Plot current car state.
            if (i < U_sol.size())
                plotCarBox(X_sol[i], U_sol[i], car_length, car_width, "k", ax);
            else
                plotCarBox(X_sol[i], empty_control, car_length, car_width, "k", ax);

            // Set grid and axis limits.
            grid(ax, true);
            xlim(ax, {-4, 4});
            ylim(ax, {-4, 4});

            // Update drawing.
            fig->draw();

            // Save the frame to a PNG file.
            std::string frame_filename = plotDirectory + "/frame_" + std::to_string(i) + ".png";
            fig->save(frame_filename);
            std::this_thread::sleep_for(std::chrono::milliseconds(80));
        }
    }

    // Combine all saved frames into a GIF using ImageMagick's convert tool.
    std::string command = "convert -delay 15 " + plotDirectory + "/frame_*.png " + plotDirectory + "/car_parking_ipopt.gif";
    std::system(command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "Animation saved as car_parking_ipopt.gif" << std::endl;

    return 0;
}