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

#include <matplot/matplot.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "dynamics_model/spacecraft_nonlinear.hpp"
using namespace cddp;

TEST(SpacecraftNonlinearTest, DiscreteDynamics) {
    // Create a spacecraft instance
    double timestep = 0.01;  // 1s timestep
    double mu = 1.0;  // Earth's gravitational parameter (normalized)
    double mass = 1.0;  // 1 kg spacecraft
    std::string integration_type = "rk4";
    
    SpacecraftNonlinear spacecraft(timestep, integration_type, mass);

    // Store states for plotting
    std::vector<double> time_data, px_data, py_data, pz_data;
    std::vector<double> vx_data, vy_data, vz_data;
    std::vector<double> r0_data, theta_data;

    // Initial state: 
    Eigen::VectorXd state = Eigen::VectorXd::Zero(10);
    state << -0.01127,  
            0.0,      
            0.1,    
            0.02,    
            0.02,      
            0.0,      
            0.9,  
            0.0,      
            0.0,      
            1.22838;  

    // No control input initially (free drift)
    Eigen::VectorXd control = Eigen::VectorXd::Zero(3);

    // Simulate for several orbits
    int num_steps = 3000;
    for (int i = 0; i < num_steps; ++i) {
        // Store data for plotting
        time_data.push_back(i * timestep);
        px_data.push_back(state[0]);
        py_data.push_back(state[1]);
        pz_data.push_back(state[2]);
        vx_data.push_back(state[3]);
        vy_data.push_back(state[4]);
        vz_data.push_back(state[5]);
        r0_data.push_back(state[6]);
        theta_data.push_back(state[7]);

        // Compute the next state
        state = spacecraft.getDiscreteDynamics(state, control);
    }
}

TEST(SpacecraftNonlinearTest, AlfriendExample4_1_RelativeMotion) {
    // Parameters from Example 4.1 of Alfriend et al. "Spacecraft Formation Flying"
    double timestep = 0.01;
    double mu_normalized = 1.0; // Normalized gravitational parameter (mu = 1 in example, a0 = 1)
    double mass = 1.0;          // Mass (typically doesn't affect kinematics if control is acceleration)
    std::string integration_type = "rk4";
    
    // r_scale and v_scale are 1.0 due to the normalization a0=1, mu=1.
    SpacecraftNonlinear spacecraft(timestep, integration_type, mass, 1.0, 1.0, mu_normalized);

    // Initial state from Example 4.1, using x_bar_1(0) = -0.01127
    // State vector: [px, py, pz, vx, vy, vz, r0, theta, dr0, dtheta]
    // (px, py, pz) are (x_bar, y_bar, z_bar)
    // (vx, vy, vz) are (x_bar_prime, y_bar_prime, z_bar_prime)
    // r0 is r0_bar, theta is f0 (true anomaly)
    // dr0 is r0_bar_prime, dtheta is theta0_prime
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(SpacecraftNonlinear::STATE_DIM);
    initial_state << -0.01127,  // px_bar(0)
                      0.0,       // py_bar(0)
                      0.1,       // pz_bar(0)
                      0.02,      // vx_bar(0) (x_bar_prime(0) from eq. 4.38)
                      0.02,      // vy_bar(0) (y_bar_prime(0) from eq. 4.38)
                      0.0,       // vz_bar(0) (z_bar_prime(0) from eq. 4.38)
                      0.9,       // r0_bar(0) (calculated)
                      0.0,       // theta0(0) (f0(0) from eq. 4.38)
                      0.0,       // dr0_bar(0) (r0_bar_prime(0) calculated)
                      1.22838;   // dtheta0(0) (theta0_prime(0) calculated)

    Eigen::VectorXd control = Eigen::VectorXd::Zero(SpacecraftNonlinear::CONTROL_DIM); // No control input (free drift)

    // Chief's orbital period: T = 2*pi / n. With a0=1, mu=1, mean motion n = sqrt(mu/a0^3) = 1.
    // So, normalized period is 2*pi.
    const double chief_orbital_period = 2.0 * M_PI; 
    double simulation_duration = 4.0 * chief_orbital_period; // Simulate for 4 orbital periods as in Fig 4.2
    int num_steps = static_cast<int>(std::round(simulation_duration / timestep));

    std::vector<Eigen::VectorXd> state_history;
    state_history.reserve(num_steps + 1);
    state_history.push_back(initial_state);

    Eigen::VectorXd current_state = initial_state;
    for (int i = 0; i < num_steps; ++i) {
        current_state = spacecraft.getDiscreteDynamics(current_state, control);
        state_history.push_back(current_state);
    }

    // --- Assertions ---

    // 1. Check bounds for relative positions based on Figure 4.2 (allowing for small margins)
    // These checks are performed over the later part of the simulation (e.g., last two periods)
    // to allow any initial numerical transient to settle, though for these ICs it should be stable.
    int steps_per_period = static_cast<int>(std::round(chief_orbital_period / timestep));
    int start_step_for_assertion = num_steps - 2 * steps_per_period;
    if (start_step_for_assertion < 1) start_step_for_assertion = 1; // Avoid checking initial_state directly if it's part of this range

    for (int i = start_step_for_assertion; i <= num_steps; ++i) {
        // Bounds from Fig 4.2 (x/a, y/a, z/a which are px, py, pz here)
        ASSERT_GE(state_history[i](SpacecraftNonlinear::STATE_PX), -0.019) << "px at step " << i << " is too low.";
        ASSERT_LE(state_history[i](SpacecraftNonlinear::STATE_PX),  0.019) << "px at step " << i << " is too high.";
        ASSERT_GE(state_history[i](SpacecraftNonlinear::STATE_PY), -0.101) << "py at step " << i << " is too low.";
        ASSERT_LE(state_history[i](SpacecraftNonlinear::STATE_PY),  0.101) << "py at step " << i << " is too high.";
        ASSERT_GE(state_history[i](SpacecraftNonlinear::STATE_PZ), -0.181) << "pz at step " << i << " is too low.";
        ASSERT_LE(state_history[i](SpacecraftNonlinear::STATE_PZ),  0.181) << "pz at step " << i << " is too high.";
    }
    
    // 2. Check periodicity for relative state components (px, py, pz, vx, vy, vz)
    // Compare state at the end of the 3rd period with the state at the end of the 4th period.
    // (Indices for state_history are 0 to num_steps)
    // State at end of 3rd period: state_history[num_steps - steps_per_period]
    // State at end of 4th period: state_history[num_steps]
    if (num_steps >= steps_per_period) {
        Eigen::VectorXd state_at_3_periods = state_history[num_steps - steps_per_period];
        Eigen::VectorXd state_at_4_periods = state_history[num_steps];

        // Tolerance for periodicity check. Numerical integration errors accumulate.
        // Values are relatively small, so absolute tolerance of 1e-3 to 1e-4 might be appropriate.
        // The example uses x_bar_1(0) = -0.01127 (5dp), theta0_prime = 1.22838 (5dp)
        double tol = 5e-4; 

        ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_PX), state_at_3_periods(SpacecraftNonlinear::STATE_PX), tol);
        ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_PY), state_at_3_periods(SpacecraftNonlinear::STATE_PY), tol);
        ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_PZ), state_at_3_periods(SpacecraftNonlinear::STATE_PZ), tol);
        ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_VX), state_at_3_periods(SpacecraftNonlinear::STATE_VX), tol);
        ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_VY), state_at_3_periods(SpacecraftNonlinear::STATE_VY), tol);
        ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_VZ), state_at_3_periods(SpacecraftNonlinear::STATE_VZ), tol);
    } else {
        FAIL() << "Simulation duration is less than one chief orbital period, cannot check periodicity.";
    }

    namespace plt = matplot;
    std::vector<double> time_plot_data;
    std::vector<double> px_plot_data, py_plot_data, pz_plot_data;

    for(int i=0; i<=num_steps; ++i){
        time_plot_data.push_back(i * timestep);
        px_plot_data.push_back(state_history[i](SpacecraftNonlinear::STATE_PX));
        py_plot_data.push_back(state_history[i](SpacecraftNonlinear::STATE_PY));
        pz_plot_data.push_back(state_history[i](SpacecraftNonlinear::STATE_PZ));
    }

    // // Figure 4.2: Time histories of normalized relative position components
    // plt::figure();
    // plt::subplot(3, 1, 0);
    // plt::plot(time_plot_data, px_plot_data);
    // plt::ylabel("x/a");
    // plt::title("Figure 4.2: Normalized Relative Position vs. Time");

    // plt::subplot(3, 1, 1);
    // plt::plot(time_plot_data, py_plot_data);
    // plt::ylabel("y/a");

    // plt::subplot(3, 1, 2);
    // plt::plot(time_plot_data, pz_plot_data);
    // plt::ylabel("z/a");
    // plt::xlabel("Orbital Periods (Chief)"); // The example image shows this, time is normalized by period
    // // Adjust x-axis to show orbital periods
    // std::vector<double> time_in_periods;
    // for(double t : time_plot_data) time_in_periods.push_back(t / chief_orbital_period);
    // plt::figure(); // Create a new figure for clarity with corrected x-axis
    // plt::subplot(3, 1, 0);
    // plt::plot(time_in_periods, px_plot_data);
    // plt::ylabel("x/a");
    // plt::ylim({-0.02, 0.02});
    // plt::title("Figure 4.2 Style: Normalized Relative Position vs. Chief Orbital Periods");

    // plt::subplot(3, 1, 1);
    // plt::plot(time_in_periods, py_plot_data);
    // plt::ylabel("y/a");
    // plt::ylim({-0.1, 0.1});

    // plt::subplot(3, 1, 2);
    // plt::plot(time_in_periods, pz_plot_data);
    // plt::ylabel("z/a");
    // plt::ylim({-0.2, 0.2});
    // plt::xlabel("Chief Orbital Periods");


    // // Figure 4.3: Relative motion in configuration space
    // plt::figure();
    // // y/a vs x/a
    // plt::subplot(2, 2, 0); // Corresponds to top-left in a 2x2 grid
    // plt::plot(px_plot_data, py_plot_data);
    // plt::xlabel("x/a");
    // plt::ylabel("y/a");
    // plt::axis(plt::equal);
    // plt::title("Figure 4.3 Style: Configuration Space Projections");

    // // z/a vs x/a
    // plt::subplot(2, 2, 1); // Corresponds to top-right
    // plt::plot(px_plot_data, pz_plot_data);
    // plt::xlabel("x/a");
    // plt::ylabel("z/a");
    // plt::axis(plt::equal);

    // // z/a vs y/a (Note: Fig 4.3 shows this plot with y/a on horizontal axis)
    // plt::subplot(2, 2, 2); // Corresponds to bottom-left
    // plt::plot(py_plot_data, pz_plot_data);
    // plt::xlabel("y/a");
    // plt::ylabel("z/a");
    // plt::axis(plt::equal);

    // // 3D plot: x/a, y/a, z/a
    // plt::subplot(2, 2, 3); // Corresponds to bottom-right
    // plt::plot3(px_plot_data, py_plot_data, pz_plot_data);
    // plt::xlabel("x/a");
    // plt::ylabel("y/a");
    // plt::zlabel("z/a");
    // plt::axis(plt::equal);
    // // For view to match Fig 4.3 (approximate)
    // // matplot++ view control is plt::view(azimuth, elevation);
    // plt::view(120, 30); // Adjust these values as needed

    // plt::show();
}

TEST(SpacecraftNonlinearTest, RelativeMotion2) {
    double mu_normalized = 1.0; // Normalized gravitational parameter (mu = 1 in example, a0 = 1)
    double mass = 1.0;          // Mass (typically doesn't affect kinematics if control is acceleration)
    std::string integration_type = "rk4";

    // HCW parameters
    double gravitational_parameter = 3.9860044e14;
    double ref_radius = (6371.0 + 500.0) * 1e3;
    double ref_semimajor_axis = ref_radius;
    double ref_period = 2 * M_PI * sqrt(pow(ref_radius, 3) / 3.9860044e14);
    double ref_mean_motion = 1.0;
    double nominal_radius = 50.0;
    double eccentricity = 0.001;
    double initial_true_anomaly = 0.0;

    double period = 2 * M_PI * std::sqrt(std::pow(ref_semimajor_axis, 3) / gravitational_parameter);
    double time_horizon = 2 * period / period;
    int horizon = 500;
    double timestep = time_horizon / horizon;

    double px0 = -37.59664132226163;
    double py0 = 27.312455860666148;
    double pz0 = 13.656227930333074;
    double vx0 = 0.015161970413423813;
    double vy0 = 0.08348413138390476;
    double vz0 = 0.04174206569195238;
    double r0 = 6371e3 + 500e3;
    double theta0 = 0.0;
    double v_normalized = std::sqrt(gravitational_parameter / ref_semimajor_axis);

    // Initial state from Example 4.1, using x_bar_1(0) = -0.01127
    // State vector: [px, py, pz, vx, vy, vz, r0, theta, dr0, dtheta]
    // (px, py, pz) are (x_bar, y_bar, z_bar)
    // (vx, vy, vz) are (x_bar_prime, y_bar_prime, z_bar_prime)
    // r0 is r0_bar, theta is f0 (true anomaly)
    // dr0 is r0_bar_prime, dtheta is theta0_prime
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(SpacecraftNonlinear::STATE_DIM);
    initial_state << 
        px0 / ref_semimajor_axis,
        py0 / ref_semimajor_axis,
        pz0 / ref_semimajor_axis,
        vx0 / v_normalized,
        vy0 / v_normalized,
        vz0 / v_normalized,
        (1.0 - eccentricity * eccentricity) / (1.0 + eccentricity * std::cos(initial_true_anomaly)),
        theta0,
        eccentricity * std::sin(initial_true_anomaly) * std::sqrt(1.0/(1.0 - eccentricity * eccentricity)),
        std::sqrt(1.0/std::pow(1.0 - eccentricity * eccentricity, 3)) * std::pow(1.0 + eccentricity * std::cos(initial_true_anomaly), 2);


    // r_scale and v_scale are 1.0 due to the normalization a0=1, mu=1.
    SpacecraftNonlinear spacecraft(timestep, integration_type, mass, 1.0, 1.0, mu_normalized);

    Eigen::VectorXd control = Eigen::VectorXd::Zero(SpacecraftNonlinear::CONTROL_DIM); // No control input (free drift)

    // Chief's orbital period: T = 2*pi / n. With a0=1, mu=1, mean motion n = sqrt(mu/a0^3) = 1.
    // So, normalized period is 2*pi.
    const double chief_orbital_period = 2.0 * M_PI; 
    double simulation_duration = 4.0 * chief_orbital_period; // Simulate for 4 orbital periods as in Fig 4.2
    int num_steps = static_cast<int>(std::round(simulation_duration / timestep));

    std::vector<Eigen::VectorXd> state_history;
    state_history.reserve(num_steps + 1);
    state_history.push_back(initial_state);

    Eigen::VectorXd current_state = initial_state;
    for (int i = 0; i < num_steps; ++i) {
        current_state = spacecraft.getDiscreteDynamics(current_state, control);
        state_history.push_back(current_state);
    }

    // --- Assertions ---

    // int steps_per_period = static_cast<int>(std::round(chief_orbital_period / timestep));
    // int start_step_for_assertion = num_steps - 2 * steps_per_period;
    // if (start_step_for_assertion < 1) start_step_for_assertion = 1; 

    // for (int i = start_step_for_assertion; i <= num_steps; ++i) {
    //     // Bounds from Fig 4.2 (x/a, y/a, z/a which are px, py, pz here)
    //     ASSERT_GE(state_history[i](SpacecraftNonlinear::STATE_PX), -0.019) << "px at step " << i << " is too low.";
    //     ASSERT_LE(state_history[i](SpacecraftNonlinear::STATE_PX),  0.019) << "px at step " << i << " is too high.";
    //     ASSERT_GE(state_history[i](SpacecraftNonlinear::STATE_PY), -0.101) << "py at step " << i << " is too low.";
    //     ASSERT_LE(state_history[i](SpacecraftNonlinear::STATE_PY),  0.101) << "py at step " << i << " is too high.";
    //     ASSERT_GE(state_history[i](SpacecraftNonlinear::STATE_PZ), -0.181) << "pz at step " << i << " is too low.";
    //     ASSERT_LE(state_history[i](SpacecraftNonlinear::STATE_PZ),  0.181) << "pz at step " << i << " is too high.";
    // }
    
    // if (num_steps >= steps_per_period) {
    //     Eigen::VectorXd state_at_3_periods = state_history[num_steps - steps_per_period];
    //     Eigen::VectorXd state_at_4_periods = state_history[num_steps];

    //     double tol = 5e-4; 

    //     ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_PX), state_at_3_periods(SpacecraftNonlinear::STATE_PX), tol);
    //     ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_PY), state_at_3_periods(SpacecraftNonlinear::STATE_PY), tol);
    //     ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_PZ), state_at_3_periods(SpacecraftNonlinear::STATE_PZ), tol);
    //     ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_VX), state_at_3_periods(SpacecraftNonlinear::STATE_VX), tol);
    //     ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_VY), state_at_3_periods(SpacecraftNonlinear::STATE_VY), tol);
    //     ASSERT_NEAR(state_at_4_periods(SpacecraftNonlinear::STATE_VZ), state_at_3_periods(SpacecraftNonlinear::STATE_VZ), tol);
    // } else {
    //     FAIL() << "Simulation duration is less than one chief orbital period, cannot check periodicity.";
    // }

    // namespace plt = matplot;
    // std::vector<double> time_plot_data;
    // std::vector<double> px_plot_data, py_plot_data, pz_plot_data;

    // for(int i=0; i<=num_steps; ++i){
    //     time_plot_data.push_back(i * timestep);
    //     px_plot_data.push_back(state_history[i](SpacecraftNonlinear::STATE_PX) * ref_semimajor_axis);
    //     py_plot_data.push_back(state_history[i](SpacecraftNonlinear::STATE_PY) * ref_semimajor_axis);
    //     pz_plot_data.push_back(state_history[i](SpacecraftNonlinear::STATE_PZ) * ref_semimajor_axis);
    // }

    // // Figure 4.2: Time histories of normalized relative position components
    // plt::figure();
    // plt::subplot(3, 1, 0);
    // plt::plot(time_plot_data, px_plot_data);
    // plt::ylabel("x/a");
    // plt::title("Figure 4.2: Normalized Relative Position vs. Time");

    // plt::subplot(3, 1, 1);
    // plt::plot(time_plot_data, py_plot_data);
    // plt::ylabel("y/a");

    // plt::subplot(3, 1, 2);
    // plt::plot(time_plot_data, pz_plot_data);
    // plt::ylabel("z/a");
    // plt::xlabel("Orbital Periods (Chief)"); // The example image shows this, time is normalized by period
    // // Adjust x-axis to show orbital periods
    // std::vector<double> time_in_periods;
    // for(double t : time_plot_data) time_in_periods.push_back(t / chief_orbital_period);
    // plt::figure(); // Create a new figure for clarity with corrected x-axis
    // plt::subplot(3, 1, 0);
    // plt::plot(time_in_periods, px_plot_data);
    // plt::ylabel("x/a");
    // plt::ylim({-0.02, 0.02});
    // plt::title("Figure 4.2 Style: Normalized Relative Position vs. Chief Orbital Periods");

    // plt::subplot(3, 1, 1);
    // plt::plot(time_in_periods, py_plot_data);
    // plt::ylabel("y/a");
    // plt::ylim({-0.1, 0.1});

    // plt::subplot(3, 1, 2);
    // plt::plot(time_in_periods, pz_plot_data);
    // plt::ylabel("z/a");
    // plt::ylim({-0.2, 0.2});
    // plt::xlabel("Chief Orbital Periods");


    // // Figure 4.3: Relative motion in configuration space
    // plt::figure();
    // // y/a vs x/a
    // plt::subplot(2, 2, 0); // Corresponds to top-left in a 2x2 grid
    // plt::plot(px_plot_data, py_plot_data);
    // plt::xlabel("x/a");
    // plt::ylabel("y/a");
    // plt::axis(plt::equal);
    // plt::title("Figure 4.3 Style: Configuration Space Projections");

    // // z/a vs x/a
    // plt::subplot(2, 2, 1); // Corresponds to top-right
    // plt::plot(px_plot_data, pz_plot_data);
    // plt::xlabel("x/a");
    // plt::ylabel("z/a");
    // plt::axis(plt::equal);

    // // z/a vs y/a (Note: Fig 4.3 shows this plot with y/a on horizontal axis)
    // plt::subplot(2, 2, 2); // Corresponds to bottom-left
    // plt::plot(py_plot_data, pz_plot_data);
    // plt::xlabel("y/a");
    // plt::ylabel("z/a");
    // plt::axis(plt::equal);

    // // 3D plot: x/a, y/a, z/a
    // plt::subplot(2, 2, 3); // Corresponds to bottom-right
    // plt::plot3(px_plot_data, py_plot_data, pz_plot_data);
    // plt::xlabel("x/a");
    // plt::ylabel("y/a");
    // plt::zlabel("z/a");
    // plt::axis(plt::equal);
    // // For view to match Fig 4.3 (approximate)
    // // matplot++ view control is plt::view(azimuth, elevation);
    // plt::view(120, 30); // Adjust these values as needed

    // plt::show();
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}