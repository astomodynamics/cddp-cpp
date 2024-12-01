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
#include <chrono>
#include <thread>
#include <cmath>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/spacecraft_linear.hpp"
#include "cddp-cpp/matplotlibcpp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(HCWTest, DiscreteDynamics) {
    // Create an HCW instance
    double timestep = 1.0;  // 1s timestep
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6371e3 + 500e3, 3));  // For 500km orbit
    double mass = 1.0;  // 100 kg spacecraft
    std::string integration_type = "euler";
    
    cddp::HCW hcw(timestep, mean_motion, mass, integration_type);

    // Store states for plotting
    std::vector<double> time_data, x_data, y_data, z_data;
    std::vector<double> vx_data, vy_data, vz_data;

    // Initial state: 
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = -37.59664132226163; 
    state(1) = 27.312455860666148;
    state(2) = 13.656227930333074;
    state(3) = 0.015161970413423813;
    state(4) = 0.08348413138390476;
    state(5) = 0.04174206569195238;

    // No control input initially (free drift)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Simulate for several orbits
    int num_steps = 6000;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        z_data.push_back(state[2]);
        vx_data.push_back(state[3]);
        vy_data.push_back(state[4]);
        vz_data.push_back(state[5]);

        // Compute the next state
        state = hcw.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(hcw.getStateDim(), 6);
    ASSERT_EQ(hcw.getControlDim(), 3);
    ASSERT_DOUBLE_EQ(hcw.getTimestep(), 1);
    ASSERT_EQ(hcw.getIntegrationType(), "euler");

    // Plot the results
    plt::figure();
    
    // Position plots
    plt::subplot(2, 1, 1);
    plt::named_plot("x (radial)", time_data, x_data);
    plt::named_plot("y (along-track)", time_data, y_data);
    plt::named_plot("z (cross-track)", time_data, z_data);
    plt::title("Relative Position");
    plt::xlabel("Time [s]");
    plt::ylabel("Position [m]");
    plt::grid(true);
    plt::legend();

    // Velocity plots
    plt::subplot(2, 1, 2);
    plt::named_plot("vx", time_data, vx_data);
    plt::named_plot("vy", time_data, vy_data);
    plt::named_plot("vz", time_data, vz_data);
    plt::title("Relative Velocity");
    plt::xlabel("Time [s]");
    plt::ylabel("Velocity [m/s]");
    plt::grid(true);
    plt::legend();

    plt::tight_layout();
    plt::save("../results/tests/hcw_discrete_dynamics.png");
}

TEST(HCWTest, ContinuousDynamics) {
    // Create HCW instance
    double timestep = 0.1;
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6.771e6, 3));
    double mass = 100.0;
    cddp::HCW hcw(timestep, mass, mean_motion, "euler");

    // Test hover condition (stationary relative position)
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = 100.0;  // 100m radial offset

    // Calculate required thrust for radial hover
    double hover_thrust = mass * 3.0 * mean_motion * mean_motion * state(0);
    Eigen::VectorXd control(3);
    control << hover_thrust, 0.0, 0.0;  // Radial thrust only

    // Get dynamics
    Eigen::VectorXd state_dot = hcw.getContinuousDynamics(state, control);

    // Test hover maintenance
    // EXPECT_NEAR(state_dot[0], 0.0, 1e-10);  // dx/dt
    // EXPECT_NEAR(state_dot[1], 0.0, 1e-10);  // dy/dt
    // EXPECT_NEAR(state_dot[2], 0.0, 1e-10);  // dz/dt
    // EXPECT_NEAR(state_dot[3], 0.0, 1e-10);  // dvx/dt
    // EXPECT_NEAR(state_dot[4], 0.0, 1e-10);  // dvy/dt
    // EXPECT_NEAR(state_dot[5], 0.0, 1e-10);  // dvz/dt
}

TEST(HCWTest, RelativeTrajectory) {
    plt::figure();
    
    // Create HCW instance
    double timestep = 1.0;
    double mean_motion = std::sqrt(3.986004418e14 / std::pow(6371e3 + 500e3, 3));  // For 500km orbit
    double mass = 1.0;
    cddp::HCW hcw(timestep, mean_motion, mass, "rk4");

    // Initial conditions
    Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
    state(0) = -37.59664132226163; 
    state(1) = 27.312455860666148;
    state(2) = 13.656227930333074;
    state(3) = 0.015161970413423813;
    state(4) = 0.08348413138390476;
    state(5) = 0.04174206569195238;

    // No control input (natural motion)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Store trajectory points
    std::vector<double> x_data, y_data, z_data;

    // Simulate for multiple orbits
    int num_steps = 6000;
    for (int i = 0; i < num_steps; ++i) {
        x_data.push_back(state[0]);
        y_data.push_back(state[1]);
        z_data.push_back(state[2]);
        
        state = hcw.getDiscreteDynamics(state, control);
    }

    // Plot relative motion trajectory
    // XY plot (radial vs along-track)
    plt::subplot(2, 2, 1);
    plt::plot(y_data, x_data);
    plt::title("Radial vs Along-track");
    plt::xlabel("Along-track Y [m]");
    plt::ylabel("Radial X [m]");
    plt::grid(true);

    // XZ plot (radial vs cross-track)
    plt::subplot(2, 2, 2);
    plt::plot(z_data, x_data);
    plt::title("Radial vs Cross-track");
    plt::xlabel("Cross-track Z [m]");
    plt::ylabel("Radial X [m]");
    plt::grid(true);

    // YZ plot (along-track vs cross-track)
    plt::subplot(2, 2, 3);
    plt::plot(z_data, y_data);
    plt::title("Along-track vs Cross-track");
    plt::xlabel("Cross-track Z [m]");
    plt::ylabel("Along-track Y [m]");
    plt::grid(true);

    // 3D plot
    plt::subplot(2, 2, 4);
    plt::plot3(x_data, y_data, z_data);
    plt::title("3D Relative Motion");
    plt::xlabel("Radial X [m]");
    plt::ylabel("Along-track Y [m]");
    plt::set_zlabel("Cross-track Z [m]");
    plt::grid(true);

    plt::tight_layout();
    plt::save("../results/tests/hcw_relative_trajectory.png");
}

// Helper function to create spacecraft marker coordinates
std::vector<std::vector<double>> createSpacecraftMarker(
    const Eigen::Vector3d& position,
    double size = 1.0) {
    
    std::vector<std::vector<double>> marker(3, std::vector<double>());
    
    // Simple cube-like spacecraft shape
    std::vector<double> dx = {-1, 1, 1, -1, -1, -1, 1, 1, -1};
    std::vector<double> dy = {-1, -1, 1, 1, -1, -1, -1, 1, 1};
    std::vector<double> dz = {-1, -1, -1, -1, -1, 1, 1, 1, 1};
    
    for (size_t i = 0; i < dx.size(); ++i) {
        marker[0].push_back(position.x() + size * dx[i]);
        marker[1].push_back(position.y() + size * dy[i]);
        marker[2].push_back(position.z() + size * dz[i]);
    }
    
    return marker;
}

// TEST(HCWTest, AnimateRendezvous) {
//     plt::figure();
    
//     // Create HCW instance
//     double timestep = 0.1;
//     double mean_motion = std::sqrt(3.986004418e14 / std::pow(6.771e6, 3));
//     double mass = 100.0;
//     cddp::HCW hcw(timestep, mean_motion, mass, "rk4");

//     // Initial state: 1km ahead, 100m above
//     Eigen::VectorXd state = Eigen::VectorXd::Zero(6);
//     state << 0.0, 1000.0, 100.0,   // Position
//              0.0, -0.5, 0.0;       // Velocity

//     // Simple proportional control for rendezvous
//     double Kp = 0.001;  // Position gain
//     double Kv = 0.01;   // Velocity gain

//     // Animation parameters
//     int num_steps = 1000;
//     int plot_interval = 10;
//     double spacecraft_size = 10.0;

//     // Store trajectory
//     std::vector<double> x_traj, y_traj, z_traj;

//     for (int i = 0; i < num_steps; ++i) {
//         // Store trajectory point
//         x_traj.push_back(state[0]);
//         y_traj.push_back(state[1]);
//         z_traj.push_back(state[2]);

//         // Compute control (simple PD controller)
//         Eigen::VectorXd control = -Kp * state.segment<3>(0) - Kv * state.segment<3>(3);
//         control *= mass;  // Convert acceleration to force

//         if (i % plot_interval == 0) {
//             plt::clf();
            
//             // Plot trajectory
//             plt::plot3(x_traj, y_traj, z_traj, "k:");

//             // Plot current spacecraft position
//             auto spacecraft = createSpacecraftMarker(
//                 Eigen::Vector3d(state[0], state[1], state[2]),
//                 spacecraft_size
//             );
//             plt::plot3(spacecraft[0], spacecraft[1], spacecraft[2], "b-");

//             // Plot target (origin)
//             auto target = createSpacecraftMarker(
//                 Eigen::Vector3d(0, 0, 0),
//                 spacecraft_size
//             );
//             plt::plot3(target[0], target[1], target[2], "r-");

//             // Plot settings
//             plt::title("Spacecraft Rendezvous Animation");
//             plt::xlabel("Radial X [m]");
//             plt::ylabel("Along-track Y [m]");
//             plt::set_zlabel("Cross-track Z [m]");
//             plt::grid(true);

//             // Set axis limits
//             double max_range = std::max({
//                 *std::max_element(x_traj.begin(), x_traj.end()),
//                 *std::max_element(y_traj.begin(), y_traj.end()),
//                 *std::max_element(z_traj.begin(), z_traj.end())
//             });
//             plt::xlim(-max_range/2, max_range/2);
//             plt::ylim(-max_range/2, max_range/2);
//             plt::zlim(-max_range/2, max_range/2);

//             // Set view angle
//             plt::view_init(30, 45);

//             std::string filename = "../results/tests/hcw_rendezvous_" + 
//                                  std::to_string(i/plot_interval) + ".png";
//             plt::save(filename);
//             plt::pause(0.01);
//         }

//         // Compute next state
//         state = hcw.getDiscreteDynamics(state, control);
//     }
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}