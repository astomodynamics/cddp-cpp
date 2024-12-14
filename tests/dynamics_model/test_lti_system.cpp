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
#include "dynamics_model/lti_system.hpp"

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
    Eigen::MatrixXd A(2, 2);
    A << 0.9, 0.1,
         -0.1, 0.9;
    Eigen::MatrixXd B(2, 1);
    B << 0.0,
         1.0;
    
    LTISystem sys(A, B, timestep);
    
    Eigen::VectorXd state(2);
    state << 1.0, 0.0;
    Eigen::VectorXd control(1);
    control << 0.5;
    
    // Test continuous dynamics
    Eigen::VectorXd dx = sys.getContinuousDynamics(state, control);
    Eigen::VectorXd expected_dx = A * state + B * control;
    EXPECT_TRUE(dx.isApprox(expected_dx));
    
    // Test discrete dynamics
    Eigen::VectorXd next_state = sys.getDiscreteDynamics(state, control);
    EXPECT_TRUE(next_state.allFinite());
}

TEST(LTISystemTest, Jacobians) {
    double timestep = 0.01;
    Eigen::MatrixXd A(2, 2);
    A << 0.9, 0.1,
         -0.1, 0.9;
    Eigen::MatrixXd B(2, 1);
    B << 0.0,
         1.0;
    
    LTISystem sys(A, B, timestep);
    
    Eigen::VectorXd state(2);
    state << 1.0, 0.0;
    Eigen::VectorXd control(1);
    control << 0.5;
    
    // Test state Jacobian
    Eigen::MatrixXd state_jac = sys.getStateJacobian(state, control);
    EXPECT_TRUE(state_jac.isApprox(A));
    
    // Test control Jacobian
    Eigen::MatrixXd control_jac = sys.getControlJacobian(state, control);
    EXPECT_TRUE(control_jac.isApprox(B));
    
    // Verify zero Hessians for linear system
    Eigen::MatrixXd state_hess = sys.getStateHessian(state, control);
    Eigen::MatrixXd control_hess = sys.getControlHessian(state, control);
    EXPECT_NEAR(state_hess.norm(), 0.0, 1e-10);
    EXPECT_NEAR(control_hess.norm(), 0.0, 1e-10);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}