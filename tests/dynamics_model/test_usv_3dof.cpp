/*
 Copyright 2024 Tomo Sasaki and The Contributors

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

// Description: Test the Usv3Dof dynamics model.
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath> // Added for M_PI

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp" // Includes core and dynamics models

using namespace cddp;
using ::testing::ElementsAreArray; // For comparing Eigen matrices/vectors

TEST(Usv3DofTest, DynamicsAndDerivatives) {
    // --- Setup ---
    double timestep = 0.05;
    std::string integration_type = "rk4"; // Use RK4 for better accuracy
    Usv3Dof usv_model(timestep, integration_type);

    // Cast to base class pointer to access finite difference methods if needed
    auto system_ptr = std::make_shared<Usv3Dof>(timestep, integration_type);

    // Define a test state and control
    // State: [x, y, psi, u, v, r]
    Eigen::VectorXd x0(usv_model.getStateDim());
    x0 << 1.0, 2.0, M_PI/4.0, 0.5, 0.1, 0.05; // Non-zero initial state

    // Control: [tau_u, tau_v, tau_r]
    Eigen::VectorXd u0(usv_model.getControlDim());
    u0 << 10.0, 1.0, 0.5; // Non-zero control input

    // --- Basic Checks ---
    ASSERT_EQ(usv_model.getStateDim(), 6);
    ASSERT_EQ(usv_model.getControlDim(), 3);
    ASSERT_DOUBLE_EQ(usv_model.getTimestep(), timestep);
    ASSERT_EQ(usv_model.getIntegrationType(), integration_type);

    // --- Dynamics Checks ---
    Eigen::VectorXd x_dot = usv_model.getContinuousDynamics(x0, u0);
    ASSERT_EQ(x_dot.size(), 6);

    Eigen::VectorXd x1 = usv_model.getDiscreteDynamics(x0, u0);
    ASSERT_EQ(x1.size(), 6);
    // Basic sanity check: state should change
    ASSERT_FALSE(x1.isApprox(x0));

    // --- Jacobian Checks ---
    Eigen::MatrixXd A_analytical = usv_model.getStateJacobian(x0, u0);
    ASSERT_EQ(A_analytical.rows(), 6);
    ASSERT_EQ(A_analytical.cols(), 6);

    Eigen::MatrixXd B_analytical = usv_model.getControlJacobian(x0, u0);
    ASSERT_EQ(B_analytical.rows(), 6);
    ASSERT_EQ(B_analytical.cols(), 3);


    // --- Hessian Checks ---
    std::vector<Eigen::MatrixXd> stateHessian = usv_model.getStateHessian(x0, u0);
    ASSERT_EQ(stateHessian.size(), 6);
    for(const auto& Hxx_i : stateHessian) {
        ASSERT_EQ(Hxx_i.rows(), 6);
        ASSERT_EQ(Hxx_i.cols(), 6);
    }

    std::vector<Eigen::MatrixXd> controlHessian = usv_model.getControlHessian(x0, u0);
    ASSERT_EQ(controlHessian.size(), 6); // One matrix for each state dim derivative
    for(const auto& Huu_i : controlHessian) {
        ASSERT_EQ(Huu_i.rows(), 3);
        ASSERT_EQ(Huu_i.cols(), 3);
        // Assert that the control Hessian is zero (or very close) since dynamics are linear in control
        EXPECT_TRUE(Huu_i.isApprox(Eigen::MatrixXd::Zero(3, 3), 1e-9));
    }

     // Optional: Compare Hessians with autodiff results if implemented and desired
} 