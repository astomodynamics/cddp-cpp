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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;
using namespace cddp;

TEST(SpacecraftLanding2DTest, DiscreteDynamics) {
    double timestep = 0.1;
    std::string integration_type = "rk4";
    SpacecraftLanding2D spacecraft(timestep, integration_type);

    std::vector<double> time_data, x_data, y_data, theta_data, x_vel_data, y_vel_data, w_data;
    std::vector<double> energy_data;

    // Initial state: [x, x_dot, y, y_dot, theta, theta_dot]
    Eigen::VectorXd state(6);
    state << 1000.0,  // x position: 1000m
             0.0,     // x velocity: 0 m/s
             2000.0,  // y position: 2000m 
             -100.0,  // y velocity: -100 m/s (descending)
             0.1,     // theta: slight tilt
             0.0;     // theta_dot: no rotation

    // Control: [thrust_percent, thrust_angle]
    Eigen::VectorXd control(2);
    control << 0.5,   // 50% thrust
              0.0;    // 0 rad thrust angle

    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        time_data.push_back(i * timestep);
        x_data.push_back(state[0]);
        y_data.push_back(state[2]);
        theta_data.push_back(state[4]);
        x_vel_data.push_back(state[1]);
        y_vel_data.push_back(state[3]);
        w_data.push_back(state[5]);
        
        // Calculate total energy (potential + kinetic)
        double potential = spacecraft.getMass() * spacecraft.getGravity() * state[2];  // mgh
        double kinetic = 0.5 * spacecraft.getMass()* (state[1]*state[1] + state[3]*state[3]) +  // translational
                        0.5 * spacecraft.getInertia() * state[5]*state[5];  // rotational
        
        energy_data.push_back(potential + kinetic);

        state = spacecraft.getDiscreteDynamics(state, control);
    }

    // Basic assertions
    ASSERT_EQ(spacecraft.getStateDim(), 6);
    ASSERT_EQ(spacecraft.getControlDim(), 2);
    ASSERT_DOUBLE_EQ(spacecraft.getTimestep(), 0.1);
    ASSERT_EQ(spacecraft.getIntegrationType(), "rk4");

    // // Plot results
    // plt::figure_size(800, 1200);
    
    // // Position plot
    // plt::subplot(4, 1, 1);
    // plt::named_plot("x", time_data, x_data, "b-");
    // plt::named_plot("y", time_data, y_data, "r-");
    // plt::title("Position vs Time");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Position [m]");
    // plt::legend();
    
    // // Velocity plot
    // plt::subplot(4, 1, 2);
    // plt::named_plot("v_x", time_data, x_vel_data, "b-");
    // plt::named_plot("v_y", time_data, y_vel_data, "r-");
    // plt::title("Velocity vs Time");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Velocity [m/s]");
    // plt::legend();

    // // Attitude plot
    // plt::subplot(4, 1, 3);
    // plt::named_plot("theta", time_data, theta_data, "b-");
    // plt::named_plot("omega", time_data, w_data, "r-");
    // plt::title("Attitude vs Time");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Angle [rad], Angular Velocity [rad/s]");
    // plt::legend();

    // // Energy plot
    // plt::subplot(4, 1, 4);
    // plt::plot(time_data, energy_data, "g-");
    // plt::title("Total Energy vs Time");
    // plt::xlabel("Time [s]");
    // plt::ylabel("Energy [J]");

    // // Save plot
    // plt::save("../results/tests/spacecraft_landing2d_test.png");

    // // Animation
    // plt::figure_size(800, 800);
    // plt::title("Spacecraft Landing Animation");
    // plt::xlabel("x [m]");
    // plt::ylabel("y [m]");

    // // Reset state to initial conditions
    // state << 1000.0, 0.0, 2000.0, -100.0, 0.1, 0.0;

    // // Create animation frames
    // for (int i = 0; i < num_steps + 1; ++i) {
    //     if (i % 5 == 0) {  // Create frame every 5 steps
    //         plt::clf();

    //         double x = state[0];
    //         double y = state[2];
    //         double theta = state[4];

    //         // Calculate spacecraft corners
    //         std::vector<double> spacecraft_x, spacecraft_y;
            
    //         // Calculate corners based on length and width
    //         double l = spacecraft.getLength() / 2.0;
    //         double w = spacecraft.getWidth() / 2.0;
            
    //         // Define vertices of spacecraft shape
    //         std::vector<double> vertices_x = {-w, w, w, -w, -w};  // Last point closes the shape
    //         std::vector<double> vertices_y = {-l, -l, l, l, -l};
            
    //         // Transform vertices
    //         spacecraft_x.resize(vertices_x.size());
    //         spacecraft_y.resize(vertices_y.size());
            
    //         for (size_t j = 0; j < vertices_x.size(); ++j) {
    //             // Rotation and translation
    //             spacecraft_x[j] = x + (vertices_x[j] * cos(theta) - vertices_y[j] * sin(theta));
    //             spacecraft_y[j] = y + (vertices_x[j] * sin(theta) + vertices_y[j] * cos(theta));
    //         }

    //         // Plot spacecraft body
    //         plt::plot(spacecraft_x, spacecraft_y, "b-");
            
    //         // Plot thrust vector if thrust is non-zero
    //         if (control[0] > 0) {
    //             double thrust_magnitude = 50.0;  // Visual scale for thrust
    //             double thrust_dir = theta + control[1];
    //             std::vector<double> thrust_x = {x, x - thrust_magnitude * sin(thrust_dir)};
    //             std::vector<double> thrust_y = {y, y - thrust_magnitude * cos(thrust_dir)};
    //             plt::plot(thrust_x, thrust_y, "r-");
    //         }

    //         // Set axis limits with some margin around the trajectory
    //         plt::xlim(0, 1200);
    //         plt::ylim(0, 2200);

    //         // Save frame
    //         std::string filename = "../results/tests/spacecraft_" + std::to_string(i) + ".png";
    //         plt::save(filename);
    //     }

    //     // Update state
    //     state = spacecraft.getDiscreteDynamics(state, control);
    // }
}

// TEST(SpacecraftLanding2DTest, Jacobians) {
//     double timestep = 0.1;
//     SpacecraftLanding2D spacecraft(timestep, "rk4");

//     Eigen::VectorXd state(6);
//     state << 1000.0, 10.0, 2000.0, -100.0, 0.1, 0.1;  // Non-zero state
    
//     Eigen::VectorXd control(2);
//     control << 0.5, 0.1;  // Non-zero control

//     Eigen::MatrixXd A_analytical = spacecraft.getStateJacobian(state, control);
//     Eigen::MatrixXd B_analytical = spacecraft.getControlJacobian(state, control);

//     Eigen::MatrixXd A_numerical = spacecraft.getFiniteDifferenceStateJacobian(state, control);
//     Eigen::MatrixXd B_numerical = spacecraft.getFiniteDifferenceControlJacobian(state, control);

//     ASSERT_EQ(A_analytical.rows(), 6);
//     ASSERT_EQ(A_analytical.cols(), 6);
//     ASSERT_EQ(B_analytical.rows(), 6);
//     ASSERT_EQ(B_analytical.cols(), 2);

//     double tolerance = 1e-5;
//     EXPECT_NEAR((A_analytical - A_numerical).norm(), 0.0, tolerance);
//     EXPECT_NEAR((B_analytical - B_numerical).norm(), 0.0, tolerance);
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}