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
#include <complex>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <unsupported/Eigen/MatrixFunctions>
#include "dynamics_model/lti_system.hpp"
#include <filesystem>

namespace fs = std::filesystem;
using namespace cddp;

TEST(LTISystemTest, SpecifiedInitialization) {
    double timestep = 0.01;
    Eigen::MatrixXd A(2, 2);
    A << 0.9, 0.1,
         -0.1, 0.9;
    Eigen::MatrixXd B(2, 1);
    B << 0.0,
         1.0;
    
    LTISystem sys(A, B, timestep);
    
    // Check if matrices were properly set
    EXPECT_TRUE(sys.getA().isApprox(A));
    EXPECT_TRUE(sys.getB().isApprox(B));
}

TEST(LTISystemTest, DynamicsComputation) {
     double timestep = 0.01;
     Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero(4, 4);
     A_ << 0,    0.2473,   -0.7933,    0.3470,
     -0.2473,         0,   -0.7667,    2.1307,
     0.7933,    0.7667,         0,    0.3154,
     -0.3470,   -2.1307,   -0.3154,         0;

     // Convert to discrete time using matrix exponential
     A_ = (timestep * A_).exp();

     Eigen::MatrixXd B_ = Eigen::MatrixXd::Zero(4, 2);
     B_ << -0.6387,   -0.2026,
     -0.4049,   -0.1975,
     2.3939,    1.5163,
     -0.0496,   -1.7322;

     // Generate random B matrix
     B_ = timestep * B_;
     
     LTISystem sys(A_, B_, timestep);
     
     Eigen::VectorXd state(4);
     state << 0.8378, 0.3794, 1.4796, 0.2382;
     Eigen::VectorXd control(2);
     control << 0.01, 0.01;
    

     Eigen::VectorXd dx = sys.getDiscreteDynamics(state, control, 0.0);
     Eigen::VectorXd expected_dx = A_ * state + B_ * control;
     EXPECT_TRUE(dx.isApprox(expected_dx));

     Eigen::VectorXd x_true = Eigen::VectorXd::Zero(4);
     x_true << 0.8277, 0.3708, 1.4902, 0.2225;
     EXPECT_TRUE((dx - x_true).norm() < 1e-4);
}

TEST(LTISystemTest, Jacobians) {
    double timestep = 0.01;
     Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero(4, 4);
     A_ << 0,    0.2473,   -0.7933,    0.3470,
     -0.2473,         0,   -0.7667,    2.1307,
     0.7933,    0.7667,         0,    0.3154,
     -0.3470,   -2.1307,   -0.3154,         0;

     // Convert to discrete time using matrix exponential
     A_ = (timestep * A_).exp();

     Eigen::MatrixXd B_ = Eigen::MatrixXd::Zero(4, 2);
     B_ << -0.6387,   -0.2026,
     -0.4049,   -0.1975,
     2.3939,    1.5163,
     -0.0496,   -1.7322;

     // Generate random B matrix
     B_ = timestep * B_;
     
     LTISystem sys(A_, B_, timestep);
     
     Eigen::VectorXd state(4);
     state << 0.8378, 0.3794, 1.4796, 0.2382;
     Eigen::VectorXd control(2);
     control << 0.01, 0.01;


     Eigen::MatrixXd A = sys.getStateJacobian(state, control, 0.0);
     A *= timestep;
     A.diagonal().array() += 1.0;
     
     Eigen::MatrixXd B = sys.getControlJacobian(state, control, 0.0);
     B *= timestep;

     Eigen::MatrixXd A_true(4, 4);
     A_true << 1.0000,    0.0024,   -0.0079,    0.0035,
               -0.0025,    0.9997,   -0.0077,    0.0213,
               0.0079,    0.0076,    0.9999,   0.0032,
               -0.0035,   -0.0213,   -0.0031,    0.9998;

     Eigen::MatrixXd B_true(4, 2);
     B_true << -0.0064,  -0.0020,
               -0.0040,   -0.0020,
               0.0239,    0.0152,
               -0.0005,   -0.0173;

     // std::cout << "A: \n" << A << std::endl;
     // std::cout << "B: \n" << B << std::endl;
     // std::cout << "A_true: \n" << A_true << std::endl;
     // std::cout << "B_true: \n" << B_true << std::endl;

     EXPECT_TRUE((A - A_true).norm() < 1e-3);
     EXPECT_TRUE((B - B_true).norm() < 1e-3);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}