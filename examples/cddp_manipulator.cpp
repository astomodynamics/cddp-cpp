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

#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main() {
    // System dimensions
    int state_dim = 6;   // [q1, q2, q3, dq1, dq2, dq3]
    int control_dim = 3; // [tau1, tau2, tau3]
    int horizon = 200;   // Time horizon for optimization
    double timestep = 0.01;

    // Create manipulator instance
    auto system = std::make_unique<cddp::Manipulator>(timestep, "rk4");

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    // Position costs (joint angles)
    Q.diagonal().segment<3>(0) = 1.0 * Eigen::Vector3d::Ones();
    // Velocity costs
    Q.diagonal().segment<3>(3) = 0.1 * Eigen::Vector3d::Ones();
    
    // Control cost matrix (penalize large torques)
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    
    // Terminal cost matrix (important for reaching target)
    Eigen::MatrixXd Qf = 100.0 * Q;

    // Initial state (current manipulator configuration)
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(state_dim);
    initial_state << 0.0, -M_PI/2, M_PI,  // Initial joint angles
                    0.0, 0.0, 0.0;   // Initial joint velocities

    // Goal state (desired configuration)
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);
    goal_state << M_PI, -M_PI/6, -M_PI/3,  // Target joint angles
                 0.0, 0.0, 0.0;          // Zero final velocities

    // Create objective function with no reference trajectory
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(
        Q, R, Qf, goal_state, empty_reference_states, timestep);

    // Create CDDP solver
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Control constraints (joint torque limits)
    double max_torque = 50.0;
    Eigen::VectorXd control_lower_bound = -max_torque * Eigen::VectorXd::Ones(control_dim);
    Eigen::VectorXd control_upper_bound = max_torque * Eigen::VectorXd::Ones(control_dim);
    
    cddp_solver.addConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));


    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 100;
    options.max_line_search_iterations = 20;
    // options.regularization_type = "control";
    // options.regularization_update_factor = 2.0;
    // options.min_regularization = 1e-6;
    // options.max_regularization = 1e10;
    options.verbose = true;
    cddp_solver.setOptions(options);

    // Initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Linear interpolation for initial trajectory
    for (int i = 0; i <= horizon; ++i) {
        double alpha = static_cast<double>(i) / horizon;
        X[i] = (1.0 - alpha) * initial_state + alpha * goal_state;
    }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the optimal control problem
    cddp::CDDPSolution solution = cddp_solver.solve();

    // Create plot directory
    const std::string plotDirectory = "../results/tests";
    if (!fs::exists(plotDirectory)) {
        fs::create_directory(plotDirectory);
    }

    // Extract trajectories
    std::vector<double> time, time2;
    std::vector<std::vector<double>> joint_angles(3);
    std::vector<std::vector<double>> joint_velocities(3);
    std::vector<std::vector<double>> joint_torques(3);

    for (size_t i = 0; i < solution.time_sequence.size(); ++i) {
        time.push_back(solution.time_sequence[i]);
        
        // State trajectory
        if (i < solution.state_sequence.size()) {
            for (int j = 0; j < 3; ++j) {
                joint_angles[j].push_back(solution.state_sequence[i](j));
                joint_velocities[j].push_back(solution.state_sequence[i](j+3));
            }
        }
        
        // Control trajectory
        if (i < solution.control_sequence.size()) {
            for (int j = 0; j < 3; ++j) {
                joint_torques[j].push_back(solution.control_sequence[i](j));
            }
            time2.push_back(solution.time_sequence[i]);
        }
    }

    // Plot results
    plt::figure_size(1200, 800);
    
    // Joint angles
    plt::subplot(3, 1, 1);
    std::map<std::string, std::string> keywords;
    keywords["linewidth"] = "2";
    
    keywords["label"] = "Joint 1";
    plt::plot(time, joint_angles[0], keywords);
    keywords["label"] = "Joint 2";
    plt::plot(time, joint_angles[1], keywords);
    keywords["label"] = "Joint 3";
    plt::plot(time, joint_angles[2], keywords);
    
    plt::title("Joint Angles");
    plt::xlabel("Time [s]");
    plt::ylabel("Angle [rad]");
    plt::legend();
    plt::grid(true);

    // Joint velocities
    plt::subplot(3, 1, 2);
    keywords["label"] = "Joint 1";
    plt::plot(time, joint_velocities[0], keywords);
    keywords["label"] = "Joint 2";
    plt::plot(time, joint_velocities[1], keywords);
    keywords["label"] = "Joint 3";
    plt::plot(time, joint_velocities[2], keywords);
    
    plt::title("Joint Velocities");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [rad/s]");
    plt::legend();
    plt::grid(true);

    // Control inputs
    plt::subplot(3, 1, 3);
    keywords["label"] = "Joint 1";
    plt::plot(time2, joint_torques[0], keywords);
    keywords["label"] = "Joint 2";
    plt::plot(time2, joint_torques[1], keywords);
    keywords["label"] = "Joint 3";
    plt::plot(time2, joint_torques[2], keywords);
    
    plt::title("Joint Torques");
    plt::xlabel("Time [s]");
    plt::ylabel("Torque [Nm]");
    plt::legend();
    plt::grid(true);

    plt::tight_layout();
    plt::save(plotDirectory + "/manipulator_cddp_results.png");
    // plt::clf();

    cddp::Manipulator manipulator(timestep, "rk4");

    // Generate animation of the solution
    const long fg = plt::figure();
    plt::figure_size(800, 600);

    for (size_t i = 0; i < solution.state_sequence.size(); i += 5) {  // Animate every 5th frame
        plt::clf();

        const auto& state = solution.state_sequence[i];
        
        // Get transformations
        auto transforms = manipulator.getTransformationMatrices(state(0), state(1), state(2));
        Eigen::Matrix4d T01 = transforms[0];
        Eigen::Matrix4d T02 = T01 * transforms[1];
        Eigen::Matrix4d T03 = T02 * transforms[2];
        Eigen::Matrix4d T04 = T03 * transforms[3];

        // Get end-effector position
        Eigen::Vector4d r3;  // End-point wrt Frame 3
        r3 << manipulator.getLinkLength('c'), 0, manipulator.getLinkLength('b'), 1;
        Eigen::Vector4d r0 = T03 * r3;  // Position of end-effector
        // Get elbow position
        Eigen::Vector4d rm;  // Intermediate point between O3 and O4
        rm = T03 * Eigen::Vector4d(0, 0, manipulator.getLinkLength('b'), 1);
    
        // Plot links
        std::vector<double> x = {0, T03(0,3), rm(0), r0(0)};
        std::vector<double> y = {0, T03(1,3), rm(1), r0(1)};
        std::vector<double> z = {0, T03(2,3), rm(2), r0(2)};
        
        std::map<std::string, std::string> link_keywords;
        link_keywords["color"] = "blue";
        link_keywords["linewidth"] = "2";
        plt::plot3(x, y, z, link_keywords, fg);

        // Plot joints
        std::vector<double> joint_x = {0, T03(0,3), rm(0)};
        std::vector<double> joint_y = {0, T03(1,3), rm(1)};
        std::vector<double> joint_z = {0, T03(2,3), rm(2)};
        
        std::map<std::string, std::string> joint_keywords;
        joint_keywords["color"] = "red";
        joint_keywords["marker"] = "o";
        plt::plot3(joint_x, joint_y, joint_z, joint_keywords, fg);

        // Plot end-effector
        std::vector<double> ee_x = {r0(0)};
        std::vector<double> ee_y = {r0(1)};
        std::vector<double> ee_z = {r0(2)};
        
        std::map<std::string, std::string> ee_keywords;
        ee_keywords["color"] = "red";
        ee_keywords["marker"] = "o";
        plt::plot3(ee_x, ee_y, ee_z, ee_keywords, fg);

        plt::xlabel("X [m]");
        plt::ylabel("Y [m]");
        plt::set_zlabel("Z [m]");
        plt::title("Manipulator CDDP Solution");

        plt::xlim(-2, 2);
        plt::ylim(-2, 2);
        plt::zlim(-1, 3);
        plt::grid(true);
        plt::view_init(30, -60);

        plt::save(plotDirectory + "/manipulator_frame_" + std::to_string(i/5) + ".png");
        plt::pause(0.01);
    }
    // plt::show for 3 seconds
    plt::show(3000);

    return 0;
}

// Create gif from images using ImageMagick:
// convert -delay 5 ../results/tests/manipulator_frame_*.png ../results/tests/manipulator.gif