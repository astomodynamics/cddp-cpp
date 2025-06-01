/**
 * @file    test_spacecraft_roe.cpp
 * @brief   GTest unit tests for the SpacecraftROE class (QNS-ROE dynamics).
 *
 * Copyright 2024
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file in the project root for more details.
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>
#include <matplot/matplot.h>

#include "dynamics_model/spacecraft_roe.hpp"
using namespace cddp;

TEST(TestSpacecraftROE, BasicQNSROEPropagation)
{
    // 1) Define problem parameters
    double a = (6378.0 + 400.0) * 1e3; // semi-major axis [m]
    double period = 2 * M_PI * sqrt(pow(a, 3) / 3.9860044e14); // orbital period [s]

    // 2) Create SpacecraftROE instance
    double dt = 10.0; // time step
    std::string integration_type = "euler";
    double u0 = 0.0; // initial argument of latitude
    SpacecraftROE qnsRoeModel(dt, integration_type, a, u0);

    // 3) Create an initial relative state in the local RTN or Hill frame (6D)
    Eigen::VectorXd x0_RV(6);
    x0_RV << 495.08966689311296, -200.0,
        6.123233995736766e-15, -0.0,
        -1.1230676194377847, -0.11314009527711179;

    // 4) Convert from R/V to QNSROE
    Eigen::VectorXd x0_roe = qnsRoeModel.transformHCWToROE(x0_RV, 0.0);

    // 5) Integrate from t=0 to t=2*period
    std::vector<double> times;
    std::vector<Eigen::VectorXd> states;
    int num_steps = 2*period / dt; // 3 hours of simulation

    Eigen::VectorXd x_roe = x0_roe;

    for (int i = 0; i < num_steps; ++i)
    {
        double t = i * dt;
        x_roe = qnsRoeModel.getDiscreteDynamics(x_roe, Eigen::VectorXd::Zero(SpacecraftROE::CONTROL_DIM));
        times.push_back(t);
        states.push_back(x_roe);
    }

    // 6) Print 
    std::cout << "[ INFO ] Initial QNS-ROE state:\n"
              << x0_roe.transpose() * a << std::endl;
    std::cout << "[ INFO ] Initial time: " << times.front() << " s" << std::endl;
    std::cout << "[ INFO ] Final time: " << times.back() << " s" << std::endl;
    std::cout << "[ INFO ] Final QNS-ROE state:\n"
              << states.back().transpose() * a << std::endl;

    // 7) Basic sanity checks

    Eigen::VectorXd expected_x0_roe(6), expected_xf_roe(6);
    expected_x0_roe << -4.910333106887314, -200.0, -500.0000000000003, 0.0, -100.0, -6.123233995736765e-15;
    expected_xf_roe << -4.910333106887314, -107.44240150834467, -500.0000000000003, 0.0, -100.0, -6.123233995736765e-15;
    Eigen::VectorXd obtained_x0_roe = states.front().head(6) * a;
    Eigen::VectorXd obtained_xf_roe = states.back().head(6) * a;
    ASSERT_EQ(states.front().size(), SpacecraftROE::STATE_DIM);
    ASSERT_EQ(states.back().size(), SpacecraftROE::STATE_DIM);
    ASSERT_NEAR(obtained_x0_roe.norm(), expected_x0_roe.norm(), 1e-1);
    ASSERT_NEAR(obtained_xf_roe.norm(), expected_xf_roe.norm(), 1e-1);
}

TEST(TestSpacecraftROE, RelativeTrajectory)
{
    // 1) Define problem parameters
    double a = (6378.0 + 400.0) * 1e3; // semi-major axis [m]
    double period = 2 * M_PI * sqrt(pow(a, 3) / 3.9860044e14); // orbital period [s]

    // 2) Create SpacecraftROE instance
    double dt = 10.0; // time step
    std::string integration_type = "euler";
    double u0 = 0.0; // initial argument of latitude
    SpacecraftROE qnsRoeModel(dt, integration_type, a, u0);

    // 3) Initial HCW state
    Eigen::VectorXd x0_hcw(6);
    x0_hcw << -37.59664132226163, 
               27.312455860666148,
               13.656227930333074,
               0.015161970413423813,
               0.08348413138390476,
               0.04174206569195238;

    double mean_motion = sqrt(3.9860044e14 / pow(a, 3));
    // x0_hcw << 100.0, 0.0, 0.0, 0.0, -2.0*mean_motion*100.0, 0.0;

    // 4) Convert initial HCW to ROE
    Eigen::VectorXd x0_roe = qnsRoeModel.transformHCWToROE(x0_hcw, 0.0);

    std::cout << "[ INFO ] Initial HCW state:\n"
              << x0_hcw.transpose() << std::endl;
    std::cout << "[ INFO ] Initial ROE state:\n"
              << x0_roe.transpose() << std::endl;

    // 5) Simulate for a portion of an orbit
    int num_steps = static_cast<int>(3.0 * period / dt); // Simulate for 3 orbits
    Eigen::VectorXd current_roe_state = x0_roe;
    Eigen::VectorXd control = Eigen::VectorXd::Zero(SpacecraftROE::CONTROL_DIM); // No control

    std::vector<Eigen::VectorXd> roe_trajectory;
    std::vector<Eigen::VectorXd> hcw_trajectory;
    roe_trajectory.push_back(current_roe_state);
    hcw_trajectory.push_back(qnsRoeModel.transformROEToHCW(current_roe_state.head(6), 0.0));

    for (int i = 0; i < num_steps; ++i) {
        double t = (i + 1) * dt;
        current_roe_state = qnsRoeModel.getDiscreteDynamics(current_roe_state, control);
        roe_trajectory.push_back(current_roe_state);
        hcw_trajectory.push_back(qnsRoeModel.transformROEToHCW(current_roe_state.head(6), t));
    }

    // Print the final HCW state
    std::cout << "[ INFO ] Final HCW state:\n"
              << hcw_trajectory.back().transpose() << std::endl;
    std::cout << "[ INFO ] Final ROE state:\n"
              << roe_trajectory.back().transpose() * a<< std::endl;

    // 6) Basic Assertions
    ASSERT_EQ(roe_trajectory.size(), num_steps + 1);
    ASSERT_EQ(hcw_trajectory.size(), num_steps + 1);

    // Check that the initial HCW state, when converted to ROE and back to HCW, is consistent.
    Eigen::VectorXd x0_hcw_reconstructed = qnsRoeModel.transformROEToHCW(x0_roe.head(6), 0.0);
    for (int i = 0; i < 6; ++i) {
        ASSERT_NEAR(x0_hcw(i), x0_hcw_reconstructed(i), 1e-9);
    }

    ASSERT_GT(hcw_trajectory.back().norm(), 1e-3); 

    // // 7) Plotting 
    // namespace plt = matplot;
    // std::vector<double> t_states_plot(num_steps + 1);
    // for(int i = 0; i <= num_steps; ++i) t_states_plot[i] = i * dt;

    // std::vector<double> x_pos_plot(num_steps + 1), y_pos_plot(num_steps + 1), z_pos_plot(num_steps + 1);
    // for (size_t i = 0; i < hcw_trajectory.size(); ++i) {
    //     const auto& state_hcw = hcw_trajectory[i];
    //     x_pos_plot[i] = state_hcw(0);
    //     y_pos_plot[i] = state_hcw(1);
    //     z_pos_plot[i] = state_hcw(2);
    // }

    // // X-Y plane trajectory 
    // plt::figure();
    // plt::plot(y_pos_plot, x_pos_plot)->line_width(2).display_name("Trajectory");
    // plt::hold(true);
    // if (!x_pos_plot.empty() && !y_pos_plot.empty()){ 
    //      plt::scatter(std::vector<double>{y_pos_plot.front()}, std::vector<double>{x_pos_plot.front()})
    //         ->marker_color("g").marker_style("o").marker_size(10).display_name("Start");
    //      plt::scatter(std::vector<double>{y_pos_plot.back()}, std::vector<double>{x_pos_plot.back()})
    //         ->marker_color("r").marker_style("x").marker_size(10).display_name("End");
    // }
    // plt::hold(false);
    // plt::xlabel("y (m) [In-track]");
    // plt::ylabel("x (m) [Radial]");
    // plt::legend();
    // plt::title("HCW X-Y Plane Trajectory (from ROE propagation)");
    // plt::axis(plt::equal);
    // plt::gca()->x_axis().reverse(true); 

    // // 3D Trajectory
    // plt::figure();
    // plt::plot3(x_pos_plot, y_pos_plot, z_pos_plot, "-o")->line_width(2).marker_size(4).display_name("Trajectory");
    // plt::hold(true);
    // if (!x_pos_plot.empty()){ 
    //      plt::scatter3(std::vector<double>{x_pos_plot.front()}, std::vector<double>{y_pos_plot.front()}, std::vector<double>{z_pos_plot.front()})
    //         ->marker_color("g").marker_style("o").marker_size(10).display_name("Start");
    //      plt::scatter3(std::vector<double>{x_pos_plot.back()}, std::vector<double>{y_pos_plot.back()}, std::vector<double>{z_pos_plot.back()})
    //         ->marker_color("r").marker_style("x").marker_size(10).display_name("End");
    // }
    // plt::hold(false);
    // plt::xlabel("x (m) [Radial]");
    // plt::ylabel("y (m) [In-track]");
    // plt::zlabel("z (m) [Cross-track]");
    // plt::legend();
    // plt::title("3D HCW Trajectory (from ROE propagation)");
    // plt::axis(plt::equal); 

    // plt::show(); // Show all plots
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
