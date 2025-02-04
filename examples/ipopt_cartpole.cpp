/**
 * ipopt_cartpole.cpp
 *
 * A CasADi/IPOPT script for solving an optimal control problem for a cartpole system,
 * with post-solution plotting and animation.
 *
 * The continuous dynamics of the cartpole are:
 *
 *   ẋ = ẋ  
 *   θ̇ = θ̇  
 *   ẍ = (F + m_p sin(θ)(l θ̇² + g cos(θ))) / (m_c + m_p sin²(θ))
 *   θ̈ = (-F cos(θ) - m_p l θ̇² cos(θ) sin(θ) - (m_c+m_p)g sin(θ))
 *        / (l (m_c + m_p sin²(θ)))
 *
 * where the state vector is [x, θ, ẋ, θ̇] and the control is the applied force F.
 *
 * A terminal condition is enforced by adding a constraint that the final state equals the goal state.
 *
 * Compilation example (adjust include paths as needed):
 *   g++ ipopt_cartpole.cpp -o ipopt_cartpole -I/path/to/casadi/include -I/path/to/eigen -I/path/to/matplotlib-cpp
 *
 * Author: Your Name
 * Date: February 2025
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

// Include matplotlib-cpp header (ensure you have it installed)
#include "matplotlibcpp.hpp"
namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main() {
    ////////////////////////
    // Problem Definition //
    ////////////////////////

    // Dimensions and time parameters
    const int state_dim   = 4;   // [x, θ, x_dot, θ_dot]
    const int control_dim = 1;   // [force]
    const int horizon     = 100; // Number of time steps
    const double timestep = 0.05; // Time discretization

    // Cartpole physical parameters
    double cart_mass   = 1.0;    // mass of the cart (kg)
    double pole_mass   = 0.2;    // mass of the pole (kg)
    double pole_length = 0.5;    // length of the pole (m)
    double gravity     = 9.81;   // gravitational acceleration (m/s²)

    // Define initial and goal states.
    // State order: [x, θ, x_dot, θ_dot]
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 0.0, 0.0, 0.0, 0.0;    // Cart at origin; pole hanging downward (θ = 0)
    
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, M_PI, 0.0, 0.0;        // Cart at origin; pole upright (θ = π)

    // Cost weighting matrices
    // Here we use no running state cost (Q=0) and penalize control effort, with a heavy terminal penalty.
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.5 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf(0,0) = 100.0;  // Cart position
    Qf(1,1) = 100.0;  // Pole angle
    Qf(2,2) = 100.0;  // Cart velocity
    Qf(3,3) = 100.0;  // Pole angular velocity

    // Convert Eigen matrices to CasADi DM using a nested loop conversion.
    casadi::DM Q_dm(Q.rows(), Q.cols());
    for (int i = 0; i < Q.rows(); i++) {
        for (int j = 0; j < Q.cols(); j++) {
            Q_dm(i, j) = Q(i, j);
        }
    }
    casadi::DM R_dm(R.rows(), R.cols());
    for (int i = 0; i < R.rows(); i++) {
        for (int j = 0; j < R.cols(); j++) {
            R_dm(i, j) = R(i, j);
        }
    }
    casadi::DM Qf_dm(Qf.rows(), Qf.cols());
    for (int i = 0; i < Qf.rows(); i++) {
        for (int j = 0; j < Qf.cols(); j++) {
            Qf_dm(i, j) = Qf(i, j);
        }
    }

    // Control bounds (e.g., force limits)
    Eigen::VectorXd control_lower_bound(control_dim);
    Eigen::VectorXd control_upper_bound(control_dim);
    control_lower_bound << -10.0;
    control_upper_bound <<  10.0;

    ////////// Decision Variables //////////
    // The decision vector z contains the state trajectory and control trajectory:
    //    z = [X₀, X₁, ..., X_N, U₀, U₁, ..., U_{N-1}]
    const int n_states   = (horizon + 1) * state_dim;
    const int n_controls = horizon * control_dim;
    const int n_dec      = n_states + n_controls;

    // Define symbolic variables for states and controls
    casadi::MX X = casadi::MX::sym("X", n_states);
    casadi::MX U = casadi::MX::sym("U", n_controls);
    casadi::MX z = casadi::MX::vertcat({X, U});

    // Helper lambdas to extract the state and control at time step t
    auto X_t = [=](int t) -> casadi::MX {
        return X(casadi::Slice(t * state_dim, (t + 1) * state_dim));
    };
    auto U_t = [=](int t) -> casadi::MX {
        return U(casadi::Slice(t * control_dim, (t + 1) * control_dim));
    };

    ////////////////////////////////
    // Discrete Dynamics Function //
    ////////////////////////////////

    // Define the cartpole dynamics using Euler integration.
    // Continuous dynamics:
    //   ẋ = x_dot
    //   θ̇ = theta_dot
    //   ẍ = (F + m_p sin(θ)(l θ̇² + g cos(θ))) / (m_c + m_p sin²(θ))
    //   θ̈ = (-F cos(θ) - m_p l θ̇² cos(θ) sin(θ) - (m_c+m_p)g sin(θ)) / (l (m_c + m_p sin²(θ)))
    auto cartpole_dynamics = [=](casadi::MX state, casadi::MX control) -> casadi::MX {
        // Extract state components
        casadi::MX pos       = state(0);  // cart position
        casadi::MX theta     = state(1);  // pole angle
        casadi::MX x_dot     = state(2);
        casadi::MX theta_dot = state(3);
        casadi::MX force     = control(0);

        // Use casadi's trigonometric functions
        using casadi::cos;
        using casadi::sin;
        casadi::MX sin_theta = sin(theta);
        casadi::MX cos_theta = cos(theta);
        casadi::MX total_mass = cart_mass + pole_mass;
        casadi::MX den = cart_mass + pole_mass * sin_theta * sin_theta;

        casadi::MX x_ddot = (force + pole_mass * sin_theta * (pole_length * theta_dot * theta_dot + gravity * cos_theta)) / den;
        casadi::MX theta_ddot = (-force * cos_theta 
                                 - pole_mass * pole_length * theta_dot * theta_dot * cos_theta * sin_theta
                                 - total_mass * gravity * sin_theta) / (pole_length * den);

        // Euler discretization:
        casadi::MX next_state = casadi::MX::vertcat({
            pos       + timestep * x_dot,
            theta     + timestep * theta_dot,
            x_dot     + timestep * x_ddot,
            theta_dot + timestep * theta_ddot
        });
        return next_state;
    };

    ////////// Constraints //////////
    // We build constraints for the initial condition and the dynamics.
    casadi::MX g; 

    // Initial state constraint: X₀ = initial_state
    casadi::DM init_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + state_dim));
    g = casadi::MX::vertcat({g, X_t(0) - init_state_dm});

    // Dynamics constraints: for t = 0, …, horizon-1, enforce Xₜ₊₁ = f(Xₜ, Uₜ)
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_next_expr = cartpole_dynamics(X_t(t), U_t(t));
        g = casadi::MX::vertcat({g, X_t(t + 1) - x_next_expr});
    }

    // --- Terminal Condition Constraint ---
    // Enforce that the terminal state equals the goal state:
    casadi::DM goal_dm(std::vector<double>(goal_state.data(), goal_state.data() + state_dim));
    casadi::MX terminal_constr = X_t(horizon) - goal_dm;
    g = casadi::MX::vertcat({g, terminal_constr});
    
    ////////// Cost Function //////////
    // Build a cost that penalizes deviations along the trajectory and control effort.
    casadi::MX cost = casadi::MX::zeros(1, 1);
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_diff = X_t(t) - goal_dm;
        casadi::MX u_diff = U_t(t);
        casadi::MX state_cost = casadi::MX::mtimes({x_diff.T(), Q_dm, x_diff});
        casadi::MX control_cost = casadi::MX::mtimes({u_diff.T(), R_dm, u_diff});
        cost = cost + state_cost + control_cost;
    }
    // Terminal cost (optional if using a terminal constraint; here it adds extra penalty)
    casadi::MX x_diff_final = X_t(horizon) - goal_dm;
    casadi::MX terminal_cost = casadi::MX::mtimes({x_diff_final.T(), Qf_dm, x_diff_final});
    cost = cost + terminal_cost;

    /////////////////////////////////
    // Decision Variable Bounds    //
    /////////////////////////////////

    std::vector<double> lbx(n_dec, -1e20), ubx(n_dec, 1e20);
    // Apply control bounds to the control part of the decision vector.
    for (int t = 0; t < horizon; t++) {
        for (int i = 0; i < control_dim; i++) {
            lbx[n_states + t * control_dim + i] = control_lower_bound(i);
            ubx[n_states + t * control_dim + i] = control_upper_bound(i);
        }
    }

    // The complete set of constraints (g) must be equal to zero.
    const int n_g = static_cast<int>(g.size1());
    std::vector<double> lbg(n_g, 0.0);
    std::vector<double> ubg(n_g, 0.0);

    // Provide an initial guess for the decision vector.
    std::vector<double> x0(n_dec, 0.0);
    // Set the initial state portion.
    for (int i = 0; i < state_dim; i++) {
        x0[i] = initial_state(i);
    }
    // Linearly interpolate the state trajectory from initial_state to goal_state.
    for (int t = 1; t <= horizon; t++) {
        for (int i = 0; i < state_dim; i++) {
            x0[t * state_dim + i] = initial_state(i) + (goal_state(i) - initial_state(i)) * (double)t / horizon;
        }
    }
    // The control part of the initial guess remains zero.

    /////////////////////////
    // NLP and IPOPT Setup //
    /////////////////////////

    std::map<std::string, casadi::MX> nlp;
    nlp["x"] = z;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict solver_opts;
    solver_opts["print_time"]       = true;
    solver_opts["ipopt.print_level"]  = 5;
    solver_opts["ipopt.max_iter"]     = 500;
    solver_opts["ipopt.tol"]          = 1e-6;

    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Convert initial guess and bounds into DM objects.
    casadi::DM x0_dm = casadi::DM(x0);
    casadi::DM lbx_dm = casadi::DM(lbx);
    casadi::DM ubx_dm = casadi::DM(ubx);
    casadi::DM lbg_dm = casadi::DM(lbg);
    casadi::DM ubg_dm = casadi::DM(ubg);

    casadi::DMDict arg({
        {"x0", x0_dm},
        {"lbx", lbx_dm},
        {"ubx", ubx_dm},
        {"lbg", lbg_dm},
        {"ubg", ubg_dm}
    });

    /////////////////////////
    // Solve the NLP       //
    /////////////////////////
    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(arg);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Solver elapsed time: " << elapsed.count() << " s" << std::endl;

    /////////////////////////////////
    // Extract the Solution        //
    /////////////////////////////////
    std::vector<double> sol = std::vector<double>(res.at("x"));

    // Recover state and control trajectories.
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));
    for (int t = 0; t <= horizon; t++) {
        Eigen::VectorXd x_t(state_dim);
        for (int i = 0; i < state_dim; i++) {
            x_t(i) = sol[t * state_dim + i];
        }
        X_sol[t] = x_t;
    }
    for (int t = 0; t < horizon; t++) {
        Eigen::VectorXd u_t(control_dim);
        for (int i = 0; i < control_dim; i++) {
            u_t(i) = sol[n_states + t * control_dim + i];
        }
        U_sol[t] = u_t;
    }

    // Create a time vector.
    std::vector<double> t_sol(horizon + 1);
    for (int t = 0; t <= horizon; t++) {
        t_sol[t] = t * timestep;
    }

    /////////////////////////////////
    // Plotting Results            //
    /////////////////////////////////

    // Extract state variables for plotting.
    std::vector<double> x_arr, theta_arr, x_dot_arr, theta_dot_arr;
    for (int t = 0; t <= horizon; t++) {
        x_arr.push_back(X_sol[t](0));
        theta_arr.push_back(X_sol[t](1));
        x_dot_arr.push_back(X_sol[t](2));
        theta_dot_arr.push_back(X_sol[t](3));
    }
    // Extract control input (force)
    std::vector<double> force_arr;
    for (int t = 0; t < horizon; t++) {
        force_arr.push_back(U_sol[t](0));
    }

    // Plot the state trajectories in a 2x2 subplot.
    plt::figure_size(1200, 800);
    plt::subplot(2, 2, 1);
    plt::title("Cart Position");
    plt::plot(t_sol, x_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::grid(true);

    plt::subplot(2, 2, 2);
    plt::title("Pole Angle");
    plt::plot(t_sol, theta_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [rad]");
    plt::grid(true);

    plt::subplot(2, 2, 3);
    plt::title("Cart Velocity");
    plt::plot(t_sol, x_dot_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [m/s]");
    plt::grid(true);

    plt::subplot(2, 2, 4);
    plt::title("Pole Angular Velocity");
    plt::plot(t_sol, theta_dot_arr, "b-");
    plt::xlabel("Time [s]");
    plt::ylabel("Angular Velocity [rad/s]");
    plt::grid(true);

    plt::tight_layout();
    plt::show();

    // Plot control input.
    std::vector<double> t_u(horizon);
    for (int t = 0; t < horizon; t++) {
        t_u[t] = t * timestep;
    }
    plt::figure_size(800, 600);
    plt::title("Control Input (Force)");
    plt::plot(t_u, force_arr, "r-");
    plt::xlabel("Time [s]");
    plt::ylabel("Force [N]");
    plt::grid(true);
    plt::show();

    // /////////////////////////////////
    // // CartPole Animation          //
    // /////////////////////////////////

    // // Create a directory to save animation frames.
    // std::string plotDirectory = "./cartpole_frames";
    // if (!fs::exists(plotDirectory)) {
    //     fs::create_directory(plotDirectory);
    // }

    // // Define cart and pole dimensions for drawing.
    // double cart_width  = 0.3;
    // double cart_height = 0.2;
    // double pole_width  = 0.05;

    // plt::figure();
    // for (int i = 0; i < static_cast<int>(X_sol.size()); i++) {
    //     // Plot every 5th time step.
    //     if (i % 5 == 0) {
    //         plt::clf();

    //         double cart_x = X_sol[i](0);
    //         double theta  = X_sol[i](1);

    //         // Draw the cart as a rectangle (centered at cart_x).
    //         std::vector<double> cart_x_coords = {
    //             cart_x - cart_width / 2, cart_x + cart_width / 2,
    //             cart_x + cart_width / 2, cart_x - cart_width / 2,
    //             cart_x - cart_width / 2
    //         };
    //         // For visualization, set the cart's vertical position near zero.
    //         std::vector<double> cart_y_coords = {
    //             -cart_height / 2, -cart_height / 2,
    //             cart_height / 2, cart_height / 2,
    //             -cart_height / 2
    //         };
    //         plt::plot(cart_x_coords, cart_y_coords, "k-");

    //         // Draw the pole as a line from the top center of the cart.
    //         double pole_base_x = cart_x;
    //         double pole_base_y = cart_height / 2;
    //         double pole_end_x  = pole_base_x + pole_length * std::sin(theta);
    //         double pole_end_y  = pole_base_y - pole_length * std::cos(theta);
    //         std::vector<double> pole_x_coords = {pole_base_x, pole_end_x};
    //         std::vector<double> pole_y_coords = {pole_base_y, pole_end_y};
    //         plt::plot(pole_x_coords, pole_y_coords, "b-", 2);

    //         // Draw a circle at the end of the pole to represent the mass.
    //         std::vector<double> circle_x, circle_y;
    //         int num_points = 20;
    //         for (int j = 0; j <= num_points; j++) {
    //             double angle = 2 * M_PI * j / num_points;
    //             circle_x.push_back(pole_end_x + pole_width * std::cos(angle));
    //             circle_y.push_back(pole_end_y + pole_width * std::sin(angle));
    //         }
    //         plt::plot(circle_x, circle_y, "b-");

    //         plt::xlim(-2, 2);
    //         plt::ylim(-2, 2);
    //         std::string filename = plotDirectory + "/frame_" + std::to_string(i) + ".png";
    //         plt::save(filename);
    //         plt::pause(0.01);
    //     }
    // }

    // std::cout << "Animation frames saved in directory: " << plotDirectory << std::endl;
    
    return 0;
}
