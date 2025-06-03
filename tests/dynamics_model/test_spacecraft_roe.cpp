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
        x_roe = qnsRoeModel.getDiscreteDynamics(x_roe, Eigen::VectorXd::Zero(SpacecraftROE::CONTROL_DIM), 0.0);
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
    Eigen::VectorXd obtained_x0_roe = states.front() * a;
    Eigen::VectorXd obtained_xf_roe = states.back() * a;   
    ASSERT_EQ(states.front().size(), 6);
    ASSERT_EQ(states.back().size(), 6);
    ASSERT_NEAR(obtained_x0_roe.norm(), expected_x0_roe.norm(), 1e-1);
    ASSERT_NEAR(obtained_xf_roe.norm(), expected_xf_roe.norm(), 1e-1);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
