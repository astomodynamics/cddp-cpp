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
#include <cmath>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <filesystem>

#include "cddp.hpp"
#include "matplot/matplot.h"

using namespace matplot;
namespace fs = std::filesystem;

// Convert quaternion [qw, qx, qy, qz] to Euler angles (roll, pitch, yaw)
Eigen::Vector3d quaternionToEuler(double qw, double qx, double qy, double qz)
{
    // Roll (phi)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    double phi = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (theta)
    double sinp = 2.0 * (qw * qy - qz * qx);
    double theta = (std::abs(sinp) >= 1.0) ? std::copysign(M_PI / 2.0, sinp) : std::asin(sinp);

    // Yaw (psi)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double psi = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(phi, theta, psi);
}

// Compute rotation matrix from a unit quaternion [qw, qx, qy, qz]
Eigen::Matrix3d getRotationMatrixFromQuaternion(double qw, double qx, double qy, double qz)
{
    Eigen::Matrix3d R;
    R(0, 0) = 1 - 2 * (qy * qy + qz * qz);
    R(0, 1) = 2 * (qx * qy - qz * qw);
    R(0, 2) = 2 * (qx * qz + qy * qw);

    R(1, 0) = 2 * (qx * qy + qz * qw);
    R(1, 1) = 1 - 2 * (qx * qx + qz * qz);
    R(1, 2) = 2 * (qy * qz - qx * qw);

    R(2, 0) = 2 * (qx * qz - qy * qw);
    R(2, 1) = 2 * (qy * qz + qx * qw);
    R(2, 2) = 1 - 2 * (qx * qx + qy * qy);
    return R;
}

int main() {
    ////////// Problem Setup //////////
    const int state_dim   = 13;  // [x, y, z, qw, qx, qy, qz, vx, vy, vz, omega_x, omega_y, omega_z]
    const int control_dim = 4;   // [f1, f2, f3, f4]
    const int horizon     = 400; // Number of control intervals
    const double timestep = 0.02; // Time step

    // Quadrotor parameters
    const double mass = 1.2;         // kg
    const double arm_length = 0.165; // m
    const double gravity = 9.81;     // m/s^2
    
    // Inertia matrix
    Eigen::Matrix3d inertia_matrix = Eigen::Matrix3d::Zero();
    inertia_matrix(0, 0) = 7.782e-3; // Ixx
    inertia_matrix(1, 1) = 7.782e-3; // Iyy
    inertia_matrix(2, 2) = 1.439e-2; // Izz

    // Figure-8 trajectory parameters
    double figure8_scale = 3.0;     // 3m
    double constant_altitude = 2.0; // 2m
    double total_time = horizon * timestep;
    double omega = 2.0 * M_PI / total_time; // completes 1 cycle over the horizon

    // Define initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << figure8_scale, 0.0, constant_altitude, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    Eigen::VectorXd goal_state(state_dim);
    goal_state << figure8_scale, 0.0, constant_altitude, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    // Generate figure-8 reference trajectory
    std::vector<Eigen::VectorXd> figure8_reference_states;
    figure8_reference_states.reserve(horizon + 1);
    
    for (int i = 0; i <= horizon; ++i) {
        double t = i * timestep;
        double angle = omega * t;

        Eigen::VectorXd ref_state = Eigen::VectorXd::Zero(state_dim);
        ref_state(0) = figure8_scale * std::cos(angle);                    // x
        ref_state(1) = figure8_scale * std::sin(angle) * std::cos(angle);  // y
        ref_state(2) = constant_altitude;                                  // z
        ref_state(3) = 1.0;                                               // qw (identity quaternion)
        ref_state(4) = 0.0;                                               // qx
        ref_state(5) = 0.0;                                               // qy
        ref_state(6) = 0.0;                                               // qz

        figure8_reference_states.push_back(ref_state);
    }

    // Define cost weighting matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Q(0, 0) = 1.0;  // x position
    Q(1, 1) = 1.0;  // y position
    Q(2, 2) = 1.0;  // z position

    Eigen::MatrixXd R = 0.01 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    // Terminal cost weight
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Qf(0, 0) = 1.0;
    Qf(1, 1) = 1.0;
    Qf(2, 2) = 1.0;
    Qf(3, 3) = 1.0;
    Qf(4, 4) = 1.0;
    Qf(5, 5) = 1.0;
    Qf(6, 6) = 1.0;

    // Convert Eigen matrices to CasADi
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

    // Convert inertia matrix to CasADi
    casadi::DM inertia_dm(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            inertia_dm(i, j) = inertia_matrix(i, j);
        }
    }

    // Define control bounds
    Eigen::VectorXd u_min(control_dim), u_max(control_dim);
    u_min << 0.0, 0.0, 0.0, 0.0;
    u_max << 4.0, 4.0, 4.0, 4.0;

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

    // Quadrotor continuous dynamics function (computes derivatives)
    auto quadrotor_derivatives = [=](casadi::MX x, casadi::MX u) -> casadi::MX {
        casadi::MX x_dot = casadi::MX::zeros(state_dim, 1);
        
        // Extract states
        casadi::MX pos = x(casadi::Slice(0, 3));      // [x, y, z]
        casadi::MX quat = x(casadi::Slice(3, 7));     // [qw, qx, qy, qz]
        casadi::MX vel = x(casadi::Slice(7, 10));     // [vx, vy, vz]
        casadi::MX omega = x(casadi::Slice(10, 13));  // [omega_x, omega_y, omega_z]
        
        casadi::MX qw = quat(0), qx = quat(1), qy = quat(2), qz = quat(3);
        casadi::MX omega_x = omega(0), omega_y = omega(1), omega_z = omega(2);
        
        // Normalize quaternion
        casadi::MX q_norm = casadi::MX::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qw = qw / q_norm;
        qx = qx / q_norm;
        qy = qy / q_norm;
        qz = qz / q_norm;
        
        // Extract control inputs (motor forces)
        casadi::MX f1 = u(0), f2 = u(1), f3 = u(2), f4 = u(3);
        
        // Compute total thrust and moments
        casadi::MX thrust = f1 + f2 + f3 + f4;
        casadi::MX tau_x = arm_length * (f1 - f3);
        casadi::MX tau_y = arm_length * (f2 - f4);
        casadi::MX tau_z = 0.1 * (f1 - f2 + f3 - f4);
        
        // Rotation matrix from quaternion
        casadi::MX R11 = 1 - 2 * (qy * qy + qz * qz);
        casadi::MX R12 = 2 * (qx * qy - qz * qw);
        casadi::MX R13 = 2 * (qx * qz + qy * qw);
        casadi::MX R21 = 2 * (qx * qy + qz * qw);
        casadi::MX R22 = 1 - 2 * (qx * qx + qz * qz);
        casadi::MX R23 = 2 * (qy * qz - qx * qw);
        casadi::MX R31 = 2 * (qx * qz - qy * qw);
        casadi::MX R32 = 2 * (qy * qz + qx * qw);
        casadi::MX R33 = 1 - 2 * (qx * qx + qy * qy);
        
        // Position derivative = velocity
        x_dot(casadi::Slice(0, 3)) = vel;
        
        // Quaternion derivative
        x_dot(3) = -0.5 * (qx * omega_x + qy * omega_y + qz * omega_z);  // qw_dot
        x_dot(4) =  0.5 * (qw * omega_x + qy * omega_z - qz * omega_y);  // qx_dot
        x_dot(5) =  0.5 * (qw * omega_y - qx * omega_z + qz * omega_x);  // qy_dot
        x_dot(6) =  0.5 * (qw * omega_z + qx * omega_y - qy * omega_x);  // qz_dot
        
        // Velocity derivative (thrust is applied along body z-axis)
        casadi::MX thrust_world_x = R13 * thrust;
        casadi::MX thrust_world_y = R23 * thrust;
        casadi::MX thrust_world_z = R33 * thrust;
        
        x_dot(7) = thrust_world_x / mass;                    // vx_dot
        x_dot(8) = thrust_world_y / mass;                    // vy_dot
        x_dot(9) = thrust_world_z / mass - gravity;          // vz_dot
        
        // Angular velocity derivative
        casadi::MX inertia_inv = casadi::MX::inv(inertia_dm);
        casadi::MX tau_vec = casadi::MX::vertcat({tau_x, tau_y, tau_z});
        casadi::MX gyroscopic = casadi::MX::cross(omega, casadi::MX::mtimes(inertia_dm, omega));
        casadi::MX angular_acc = casadi::MX::mtimes(inertia_inv, tau_vec - gyroscopic);
        
        x_dot(casadi::Slice(10, 13)) = angular_acc;
        
        return x_dot;
    };

    // RK4 integration function
    auto quadrotor_dynamics = [=](casadi::MX x, casadi::MX u) -> casadi::MX {
        // RK4 integration: k1, k2, k3, k4
        casadi::MX k1 = quadrotor_derivatives(x, u);
        casadi::MX k2 = quadrotor_derivatives(x + timestep/2.0 * k1, u);
        casadi::MX k3 = quadrotor_derivatives(x + timestep/2.0 * k2, u);
        casadi::MX k4 = quadrotor_derivatives(x + timestep * k3, u);
        
        // RK4 final integration step
        return x + timestep/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
    };

    casadi::MX g; 

    // Initial state constraint: X₀ = initial_state
    casadi::DM init_state_dm(std::vector<double>(initial_state.data(), initial_state.data() + state_dim));
    g = casadi::MX::vertcat({g, X_t(0) - init_state_dm});

    // Dynamics constraints
    for (int t = 0; t < horizon; t++) {
        casadi::MX x_next_expr = quadrotor_dynamics(X_t(t), U_t(t));
        g = casadi::MX::vertcat({g, X_t(t + 1) - x_next_expr});
    }

    ////////// Cost Function //////////
    casadi::MX cost = casadi::MX::zeros(1, 1);
    
    // Running cost
    for (int t = 0; t < horizon; t++) {
        // Convert reference state to CasADi
        casadi::DM ref_dm(std::vector<double>(figure8_reference_states[t].data(), 
                                            figure8_reference_states[t].data() + state_dim));
        
        casadi::MX x_diff = X_t(t) - ref_dm;
        casadi::MX u_diff = U_t(t);
        
        casadi::MX state_cost = casadi::MX::mtimes({x_diff.T(), Q_dm, x_diff});
        casadi::MX control_cost = casadi::MX::mtimes({u_diff.T(), R_dm, u_diff});
        cost = cost + state_cost + control_cost;
    }
    
    // Terminal cost
    casadi::DM goal_dm(std::vector<double>(goal_state.data(), goal_state.data() + state_dim));
    casadi::MX x_diff_final = X_t(horizon) - goal_dm;
    casadi::MX terminal_cost = casadi::MX::mtimes({x_diff_final.T(), Qf_dm, x_diff_final});
    cost = cost + terminal_cost;

    ////////// Variable Bounds and Initial Guess //////////
    std::vector<double> lbx(n_dec, -1e20);
    std::vector<double> ubx(n_dec,  1e20);
    
    // Apply control bounds
    for (int t = 0; t < horizon; t++) {
        for (int i = 0; i < control_dim; i++) {
            lbx[n_states + t * control_dim + i] = u_min(i);
            ubx[n_states + t * control_dim + i] = u_max(i);
        }
    }

    // The complete set of constraints (g) must be equal to zero
    const int n_g = static_cast<int>(g.size1());
    std::vector<double> lbg(n_g, 0.0);
    std::vector<double> ubg(n_g, 0.0);

    // Provide an initial guess for the decision vector
    std::vector<double> x0(n_dec, 0.0);
    
    // Set the initial state portion
    for (int i = 0; i < state_dim; i++) {
        x0[i] = initial_state(i);
    }
    
    // Use the reference trajectory as initial guess for states
    for (int t = 1; t <= horizon; t++) {
        for (int i = 0; i < state_dim; i++) {
            x0[t * state_dim + i] = figure8_reference_states[t](i);
        }
    }
    
    // Initial guess for controls (hover thrust)
    double hover_thrust = mass * gravity / 4.0;
    for (int t = 0; t < horizon; t++) {
        for (int i = 0; i < control_dim; i++) {
            x0[n_states + t * control_dim + i] = hover_thrust;
        }
    }

    ////////// NLP Definition and IPOPT Solver Setup //////////
    std::map<std::string, casadi::MX> nlp;
    nlp["x"] = z;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict solver_opts;
    solver_opts["print_time"]         = true;
    solver_opts["ipopt.print_level"]  = 5;
    solver_opts["ipopt.max_iter"]     = 1000;
    solver_opts["ipopt.tol"]          = 1e-6;
    solver_opts["ipopt.acceptable_tol"] = 1e-4;

    // Create the NLP solver instance using IPOPT
    casadi::Function solver = casadi::nlpsol("solver", "ipopt", nlp, solver_opts);

    // Convert the initial guess and bounds into DM objects
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

    ////////// Solve the NLP //////////
    auto start_time = std::chrono::high_resolution_clock::now();
    casadi::DMDict res = solver(arg);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Solver elapsed time: " << elapsed.count() << " s" << std::endl;

    ////////// Extract and Display the Solution //////////
    std::vector<double> sol = std::vector<double>(res.at("x"));

    // Convert to state and control trajectories
    std::vector<Eigen::VectorXd> X_sol(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> U_sol(horizon, Eigen::VectorXd(control_dim));
    std::vector<double> t_sol(horizon + 1);
    for (int t = 0; t <= horizon; t++) {
        t_sol[t] = t * timestep;
    }

    for (int t = 0; t <= horizon; t++) {
        for (int i = 0; i < state_dim; i++) {
            X_sol[t](i) = sol[t * state_dim + i];
        }
    }

    for (int t = 0; t < horizon; t++) {
        for (int i = 0; i < control_dim; i++) {
            U_sol[t](i) = sol[n_states + t * control_dim + i];
        }
    }

    // Create directory for saving plot (if it doesn't exist)
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Extract trajectory data for plotting
    std::vector<double> time_arr, x_arr, y_arr, z_arr;
    std::vector<double> phi_arr, theta_arr, psi_arr;
    std::vector<double> qw_arr, qx_arr, qy_arr, qz_arr, q_norm_arr;

    for (size_t i = 0; i < X_sol.size(); ++i) {
        time_arr.push_back(t_sol[i]);
        x_arr.push_back(X_sol[i](0));
        y_arr.push_back(X_sol[i](1));
        z_arr.push_back(X_sol[i](2));

        double qw = X_sol[i](3);
        double qx = X_sol[i](4);
        double qy = X_sol[i](5);
        double qz = X_sol[i](6);

        // Euler angles
        Eigen::Vector3d euler = quaternionToEuler(qw, qx, qy, qz);
        phi_arr.push_back(euler(0));
        theta_arr.push_back(euler(1));
        psi_arr.push_back(euler(2));

        // Quaternion data
        qw_arr.push_back(qw);
        qx_arr.push_back(qx);
        qy_arr.push_back(qy);
        qz_arr.push_back(qz);
        q_norm_arr.push_back(std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz));
    }

    // Control data
    std::vector<double> time_arr2(time_arr.begin(), time_arr.end() - 1);
    std::vector<double> f1_arr, f2_arr, f3_arr, f4_arr;
    for (const auto& u : U_sol) {
        f1_arr.push_back(u(0));
        f2_arr.push_back(u(1));
        f3_arr.push_back(u(2));
        f4_arr.push_back(u(3));
    }

    // -----------------------------
    // Plot states and controls
    // -----------------------------
    auto f1 = figure();
    f1->size(1200, 900);

    // Position trajectories
    auto ax1 = subplot(4, 1, 0);
    auto plot_handle1 = plot(ax1, time_arr, x_arr, "-r");
    plot_handle1->display_name("x");
    plot_handle1->line_width(2);
    hold(ax1, true);
    auto plot_handle2 = plot(ax1, time_arr, y_arr, "-g");
    plot_handle2->display_name("y");
    plot_handle2->line_width(2);
    auto plot_handle3 = plot(ax1, time_arr, z_arr, "-b");
    plot_handle3->display_name("z");
    plot_handle3->line_width(2);

    title(ax1, "Position Trajectories");
    xlabel(ax1, "Time [s]");
    ylabel(ax1, "Position [m]");
    matplot::legend(ax1);
    grid(ax1, true);

    // Attitude angles
    auto ax2 = subplot(4, 1, 1);
    hold(ax2, true);
    auto plot_handle4 = plot(ax2, time_arr, phi_arr, "-r");
    plot_handle4->display_name("roll");
    plot_handle4->line_width(2);
    auto plot_handle5 = plot(ax2, time_arr, theta_arr, "-g");
    plot_handle5->display_name("pitch");
    plot_handle5->line_width(2);
    auto plot_handle6 = plot(ax2, time_arr, psi_arr, "-b");
    plot_handle6->display_name("yaw");
    plot_handle6->line_width(2);

    title(ax2, "Attitude Angles");
    xlabel(ax2, "Time [s]");
    ylabel(ax2, "Angle [rad]");
    matplot::legend(ax2);
    grid(ax2, true);

    // Motor forces
    auto ax3 = subplot(4, 1, 2);
    auto plot_handle7 = plot(ax3, time_arr2, f1_arr, "-r");
    plot_handle7->display_name("f1");
    plot_handle7->line_width(2);
    hold(ax3, true);
    auto plot_handle8 = plot(ax3, time_arr2, f2_arr, "-g");
    plot_handle8->display_name("f2");
    plot_handle8->line_width(2);
    auto plot_handle9 = plot(ax3, time_arr2, f3_arr, "-b");
    plot_handle9->display_name("f3");
    plot_handle9->line_width(2);
    auto plot_handle10 = plot(ax3, time_arr2, f4_arr, "-k");
    plot_handle10->display_name("f4");
    plot_handle10->line_width(2);

    title(ax3, "Motor Forces");
    xlabel(ax3, "Time [s]");
    ylabel(ax3, "Force [N]");
    matplot::legend(ax3);
    grid(ax3, true);

    // Quaternion components and norm
    auto ax4 = subplot(4, 1, 3);
    hold(ax4, true);

    auto qwh = plot(ax4, time_arr, qw_arr, "-r");
    qwh->display_name("q_w");
    qwh->line_width(2);

    auto qxh = plot(ax4, time_arr, qx_arr, "-g");
    qxh->display_name("q_x");
    qxh->line_width(2);

    auto qyh = plot(ax4, time_arr, qy_arr, "-b");
    qyh->display_name("q_y");
    qyh->line_width(2);

    auto qzh = plot(ax4, time_arr, qz_arr, "-m");
    qzh->display_name("q_z");
    qzh->line_width(2);

    auto qnorm_handle = plot(ax4, time_arr, q_norm_arr, "-k");
    qnorm_handle->display_name("|q|");

    title(ax4, "Quaternion Components and Norm");
    xlabel(ax4, "Time [s]");
    ylabel(ax4, "Value");
    matplot::legend(ax4);
    grid(ax4, true);

    f1->draw();
    f1->save(plotDirectory + "/quadrotor_ipopt_results.png");

    // -----------------------------
    // 3D Trajectory Plot
    // -----------------------------
    auto f2 = figure();
    f2->size(800, 600);
    auto ax_3d = f2->current_axes();
    hold(ax_3d, true);
    
    auto traj3d = plot3(ax_3d, x_arr, y_arr, z_arr);
    traj3d->display_name("Trajectory");
    traj3d->line_style("-");
    traj3d->line_width(2);
    traj3d->color("blue");
    
    // Project trajectory onto x-y plane at z=0
    auto proj_xy = plot3(ax_3d, x_arr, y_arr, std::vector<double>(x_arr.size(), 0.0));
    proj_xy->display_name("X-Y Projection");
    proj_xy->line_style("--");
    proj_xy->line_width(1);
    proj_xy->color("gray");
    
    xlabel(ax_3d, "X [m]");
    ylabel(ax_3d, "Y [m]");
    zlabel(ax_3d, "Z [m]");
    xlim(ax_3d, {-5, 5});
    ylim(ax_3d, {-2, 2});
    zlim(ax_3d, {0, 5});
    title(ax_3d, "3D Trajectory (Figure-8) - IPOPT");
    grid(ax_3d, true);
    
    f2->draw();
    f2->save(plotDirectory + "/quadrotor_ipopt_3d.png");

    // -----------------------------
    // Animation: Quadrotor Trajectory
    // -----------------------------
    auto f_anim = figure();
    f_anim->size(800, 600);
    auto ax_anim = f_anim->current_axes();

    // For collecting the trajectory as we go
    std::vector<double> anim_x, anim_y, anim_z;

    // Render every Nth frame to reduce #images
    int frame_stride = 15;
    double prop_radius = 0.03;

    for (size_t i = 0; i < X_sol.size(); i += frame_stride) {
        ax_anim->clear();
        ax_anim->hold(true);
        ax_anim->grid(true);

        // Current state
        double x = X_sol[i](0);
        double y = X_sol[i](1);
        double z = X_sol[i](2);

        // Accumulate path
        anim_x.push_back(x);
        anim_y.push_back(y);
        anim_z.push_back(z);

        // Plot the partial trajectory so far
        auto path_plot = plot3(anim_x, anim_y, anim_z);
        path_plot->line_width(1.5);
        path_plot->line_style("--");
        path_plot->color("black");

        // Build rotation from quaternion
        Eigen::Vector4d quat(X_sol[i](3), X_sol[i](4), X_sol[i](5), X_sol[i](6));
        Eigen::Matrix3d R = getRotationMatrixFromQuaternion(quat(0), quat(1), quat(2), quat(3));

        // Arm endpoints (front, right, back, left)
        std::vector<Eigen::Vector3d> arm_endpoints;
        arm_endpoints.push_back(Eigen::Vector3d(arm_length, 0, 0));
        arm_endpoints.push_back(Eigen::Vector3d(0, arm_length, 0));
        arm_endpoints.push_back(Eigen::Vector3d(-arm_length, 0, 0));
        arm_endpoints.push_back(Eigen::Vector3d(0, -arm_length, 0));

        // Transform to world coords
        for (auto& pt : arm_endpoints) {
            pt = Eigen::Vector3d(x, y, z) + R * pt;
        }

        // Front-back arm
        std::vector<double> fx = {arm_endpoints[0].x(), arm_endpoints[2].x()};
        std::vector<double> fy = {arm_endpoints[0].y(), arm_endpoints[2].y()};
        std::vector<double> fz = {arm_endpoints[0].z(), arm_endpoints[2].z()};
        auto fb_arm = plot3(fx, fy, fz);
        fb_arm->line_width(2.0);
        fb_arm->color("blue");

        // Right-left arm
        std::vector<double> rx = {arm_endpoints[1].x(), arm_endpoints[3].x()};
        std::vector<double> ry = {arm_endpoints[1].y(), arm_endpoints[3].y()};
        std::vector<double> rz = {arm_endpoints[1].z(), arm_endpoints[3].z()};
        auto rl_arm = plot3(rx, ry, rz);
        rl_arm->line_width(2.0);
        rl_arm->color("red");

        // Motor props as small circles
        auto sphere_points = linspace(0, 2 * M_PI, 15);
        for (const auto& motor_pos : arm_endpoints) {
            std::vector<double> circ_x, circ_y, circ_z;
            for (auto angle : sphere_points) {
                circ_x.push_back(motor_pos.x() + prop_radius * cos(angle));
                circ_y.push_back(motor_pos.y() + prop_radius * sin(angle));
                circ_z.push_back(motor_pos.z());
            }
            auto sphere_plot = plot3(circ_x, circ_y, circ_z);
            sphere_plot->line_style("solid");
            sphere_plot->line_width(1.5);
            sphere_plot->color("cyan");
        }

        title(ax_anim, "Quadrotor Animation - IPOPT");
        xlabel(ax_anim, "X [m]");
        ylabel(ax_anim, "Y [m]");
        zlabel(ax_anim, "Z [m]");
        xlim(ax_anim, {-5, 5});
        ylim(ax_anim, {-5, 5});
        zlim(ax_anim, {0, 5});

        ax_anim->view(30, -30);

        std::string frameFile = plotDirectory + "/quadrotor_ipopt_frame_" + std::to_string(i / frame_stride) + ".png";
        f_anim->draw();
        f_anim->save(frameFile);
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }

    // -----------------------------
    // Generate GIF from frames using ImageMagick
    // -----------------------------
    std::string gif_command = "convert -delay 30 " + plotDirectory + "/quadrotor_ipopt_frame_*.png " + plotDirectory + "/quadrotor_ipopt.gif";
    std::system(gif_command.c_str());

    std::string cleanup_command = "rm " + plotDirectory + "/quadrotor_ipopt_frame_*.png";
    std::system(cleanup_command.c_str());

    std::cout << "GIF animation created successfully: " << plotDirectory + "/quadrotor_ipopt.gif" << std::endl;
    std::cout << "Final state = " << X_sol.back().transpose() << std::endl;

    return 0;
}
