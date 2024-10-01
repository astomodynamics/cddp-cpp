// Description: Test the pendulum dynamics model.
#include <iostream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp-cpp/dynamics_model/pendulum.h" // Assuming you have the Eigen-based Pendulum class
// #include "cddp-cpp/matplotlibcpp.h"

// namespace plt = matplotlibcpp;
using namespace cddp;

TEST(PendulumTest, DiscreteDynamics) {
    // Create a pendulum instance (no device needed for Eigen)
    double mass = 1.0; 
    double length = 1.0; 
    double gravity = 9.81;
    double timestep = 0.05;
    cddp::Pendulum pendulum(mass, length, gravity, timestep); 

    // Store states for plotting
    std::vector<double> time_data, theta_data, theta_dot_data;

    // Initial state and control (use Eigen vectors)
    Eigen::VectorXd state(2);
    state << 0.1, 0.0;  // Start at a small angle, zero velocity
    Eigen::VectorXd control(1);
    control << 0.0; // No torque initially

    // Simulate for a few steps
    int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        // Print the current state
        std::cout << "Step " << i << ": theta = " << state[0]
                  << ", theta_dot = " << state[1] << std::endl;

        // Store data for plotting
        time_data.push_back(i * timestep);
        theta_data.push_back(state[0]);
        theta_dot_data.push_back(state[1]);

        // Compute the next state
        state = pendulum.getDiscreteDynamics(state, control); 
    }

    // Plot the results (same as before)
    // plt::figure();
    // plt::plot(time_data, theta_data, {{"label", "Angle"}});
    // plt::plot(time_data, theta_dot_data, {{"label", "Angular Velocity"}});
    // plt::xlabel("Time");
    // plt::ylabel("State");
    // plt::legend();
    // plt::show();

    // Assertions (adapt to Eigen)
    // No GPU checks needed anymore

    // Assert true if the pendulum has the correct state dimension
    ASSERT_EQ(pendulum.getStateDim(), 2);

    // Assert true if the pendulum has the correct control dimension
    ASSERT_EQ(pendulum.getControlDim(), 1);

    // Assert true if the pendulum has the correct timestep
    ASSERT_DOUBLE_EQ(pendulum.getTimestep(), 0.05);

    // Assert true if the pendulum has the correct integration type
    ASSERT_EQ(pendulum.getIntegrationType(), "rk4"); 
}