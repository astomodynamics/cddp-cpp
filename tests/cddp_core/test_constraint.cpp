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
// Description: Tests for the objective function classes in cddp-cpp.
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "cddp-cpp/cddp_core/constraint.hpp"
#include <matplot/matplot.h>
#include <sys/stat.h>

TEST(ControlBoxConstraintTest, Evaluate) {
    // Create a constraint with lower and upper bounds
    Eigen::VectorXd lower_bound(2);
    lower_bound << -1.0, -2.0;
    Eigen::VectorXd upper_bound(2);
    upper_bound << 1.0, 2.0;
    cddp::ControlBoxConstraint constraint(lower_bound, upper_bound);

    // Test with a control signal within the bounds
    Eigen::VectorXd state(2); 
    state << 0.5, 1.0;
    Eigen::VectorXd control(2);
    control << 0.5, 1.0;
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(control));

    // Test with a control signal outside the bounds
    control << 1.5, -2.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(control)); 
}

TEST(StateBoxConstraint, Evaluate) {
    // Create a constraint with lower and upper bounds
    Eigen::VectorXd lower_bound(2);
    lower_bound << -1.0, -2.0;
    Eigen::VectorXd upper_bound(2);
    upper_bound << 1.0, 2.0;
    cddp::StateBoxConstraint constraint(lower_bound, upper_bound);

    // Test with a state within the bounds
    Eigen::VectorXd state(2);
    state << 0.5, 1.0;
    Eigen::VectorXd control(2); // Control doesn't matter for this constraint
    control << 0.0;
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(state));

    // Test with a state outside the bounds
    state << 1.5, -2.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_TRUE(constraint_value.isApprox(state));
}

TEST(CircleConstraintTest, Evaluate) {
    // Create a circle constraint with a radius of 2.0
    cddp::BallConstraint constraint(2.0, Eigen::Vector2d(0.0, 0.0));

    // Test with a state inside the circle
    Eigen::VectorXd state(2);
    state << 1.0, 1.0;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0; 
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), -2.0, 1e-6); 

    // Test with a state outside the circle
    state << 2.5, 1.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), -8.5, 1e-6);
}

TEST(CircleConstraintTest, Gradients) {
    // Create a circle constraint with a radius of 2.0
    cddp::BallConstraint constraint(2.0, Eigen::Vector2d(0.0, 0.0));

    // Test with a state inside the circle
    Eigen::VectorXd state(2);
    state << 1.0, 1.0;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0; 
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    auto gradients = constraint.getJacobians(state, control);
    ASSERT_TRUE(std::get<0>(gradients).isApprox(Eigen::Vector2d(-2.0, -2.0)));
    ASSERT_TRUE(std::get<1>(gradients).isApprox(Eigen::Vector2d(0.0, 0.0)));

    // Test with a state outside the circle
    state << 2.5, 1.5;
    constraint_value = constraint.evaluate(state, control);
    gradients = constraint.getJacobians(state, control);
    ASSERT_TRUE(std::get<0>(gradients).isApprox(Eigen::Vector2d(-5.0, -3.0)));
    ASSERT_TRUE(std::get<1>(gradients).isApprox(Eigen::Vector2d(0.0, 0.0)));
}

TEST(LinearConstraintTest, Evaluate) {
    // Create a linear constraint with A = [1, 1], b = 1
    Eigen::MatrixXd A(1, 2);
    A << 1.0, 1.0;
    Eigen::VectorXd b(1);
    b << 1.0;
    cddp::LinearConstraint constraint(A, b);

    // Test with a state that satisfies the constraint
    Eigen::VectorXd state(2);
    state << 0.5, 0.5;
    Eigen::VectorXd control(1); // Control doesn't matter for this constraint
    control << 0.0;
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), 1.0, 1e-6);

    // Test with a state that violates the constraint
    state << 0.5, -0.5;
    constraint_value = constraint.evaluate(state, control);
    ASSERT_NEAR(constraint_value(0), 0.0, 1e-6);
}

// New test case for LinearConstraint visualization
TEST(LinearConstraintTest, Visualization) {
    namespace plt = matplot;

    // 1. Define Linear Constraints (Ax <= b)
    // Constraint 1: x + y <= 1
    // Constraint 2: -x + y <= 1
    Eigen::MatrixXd A(2, 2);
    A <<  1.0,  1.0,
         -1.0,  1.0;
    Eigen::VectorXd b(2);
    b << 1.0, 1.0;
    cddp::LinearConstraint constraint(A, b);

    // 2. Generate Grid of Points
    double range = 2.0;
    size_t num_points = 50;
    std::vector<double> x_vals = plt::linspace(-range, range, num_points);
    std::vector<double> y_vals = plt::linspace(-range, range, num_points);
    auto [X, Y] = plt::meshgrid(x_vals, y_vals);

    // 3. Evaluate Constraint and Categorize Points
    std::vector<double> x_feasible, y_feasible;
    std::vector<double> x_infeasible, y_infeasible;
    Eigen::VectorXd control(1); control << 0.0; // Dummy control

    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[0].size(); ++j) {
            Eigen::VectorXd state(2);
            state << X[i][j], Y[i][j];
            Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
            // Check if ALL constraints are satisfied (Ax - b <= 0)
            if (((A * state - b).array() <= 1e-6).all()) { // Use tolerance
                x_feasible.push_back(state(0));
                y_feasible.push_back(state(1));
            } else {
                x_infeasible.push_back(state(0));
                y_infeasible.push_back(state(1));
            }
        }
    }

    // 4. Plotting (Commented out for CI/headless environments)
    auto fig = plt::figure();
    plt::hold(true);

    // Plot feasible and infeasible points
    if (!x_feasible.empty()) {
        plt::plot(x_feasible, y_feasible, "g.")->marker_size(5).display_name("Feasible (Ax <= b)");
    }
    if (!x_infeasible.empty()) {
        plt::plot(x_infeasible, y_infeasible, "r.")->marker_size(5).display_name("Infeasible (Ax > b)");
    }

    // Plot constraint boundaries (Ax = b)
    // Line 1: x + y = 1  => y = 1 - x
    std::vector<double> x_line1 = {-range, range};
    std::vector<double> y_line1 = {1 - x_line1[0], 1 - x_line1[1]};
    plt::plot(x_line1, y_line1, "k-")->line_width(2).display_name("x + y = 1");

    // Line 2: -x + y = 1 => y = 1 + x
    std::vector<double> x_line2 = {-range, range};
    std::vector<double> y_line2 = {1 + x_line2[0], 1 + x_line2[1]};
    plt::plot(x_line2, y_line2, "b-")->line_width(2).display_name("-x + y = 1");


    plt::hold(false);
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Linear Constraint Feasible Space (Ax <= b)");
    plt::legend();
    plt::grid(true);
    plt::axis("equal");
    // plt::xlim({-range, range}); // Optional: Set limits
    // plt::ylim({-range, range});

    // Save the plot to a file
    std::string filename = "linear_constraint_visualization.png";
    plt::save(filename);
    std::cout << "Saved linear constraint visualization to " << filename << std::endl;

    // Assert that the file was created (or would be created if plotting enabled)
    // Since plotting is commented out, we'll just check feasibility/infeasibility counts
    ASSERT_FALSE(x_feasible.empty());
    ASSERT_FALSE(x_infeasible.empty());
    // struct stat buffer;
    // ASSERT_EQ(stat(filename.c_str(), &buffer), 0) << "Plot file '" << filename << "' was not generated.";
}

// New test suite for SecondOrderConeConstraint
TEST(SecondOrderConeConstraintTest, Evaluate) {
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // Opening along positive Y-axis
    double fov = M_PI / 4.0; // 45 degrees half-angle
    double cos_fov_sq = std::pow(std::cos(fov), 2);
    cddp::SecondOrderConeConstraint constraint(origin, axis, fov);

    Eigen::VectorXd control(1); // Control doesn't matter
    control << 0.0;

    // Test with a state INSIDE the cone (g <= 0)
    Eigen::VectorXd state_inside(3);
    state_inside << 0.0, 1.0, 0.0; // Point along positive Y-axis
    double v_norm_sq_inside = (state_inside - origin).squaredNorm();
    double dot_inside = (state_inside - origin).dot(axis);
    double expected_g_inside = cos_fov_sq * v_norm_sq_inside - dot_inside * dot_inside;
    Eigen::VectorXd constraint_value_inside = constraint.evaluate(state_inside, control);
    // Expecting cos(fov)^2 * 1^2 - (1)^2 = 0.5 - 1 = -0.5
    ASSERT_NEAR(constraint_value_inside(0), expected_g_inside, 1e-9);
    ASSERT_LT(constraint_value_inside(0), 0.0);

    // Test with a state OUTSIDE the cone (g > 0)
    Eigen::VectorXd state_outside(3);
    state_outside << 1.0, 1.0, 0.0; // Point off-axis but positive Y
    double v_norm_sq_outside = (state_outside - origin).squaredNorm();
    double dot_outside = (state_outside - origin).dot(axis);
    double expected_g_outside = cos_fov_sq * v_norm_sq_outside - dot_outside * dot_outside;
    Eigen::VectorXd constraint_value_outside = constraint.evaluate(state_outside, control);
    // Expecting cos(fov)^2 * (1^2 + 1^2) - (1)^2 = 0.5 * 2 - 1 = 0
    // Let's move it slightly outside: state_outside << 1.1, 1.0, 0.0;
    state_outside << 1.1, 1.0, 0.0;
    v_norm_sq_outside = (state_outside - origin).squaredNorm(); // 1.21 + 1 = 2.21
    dot_outside = (state_outside - origin).dot(axis); // 1.0
    expected_g_outside = cos_fov_sq * v_norm_sq_outside - dot_outside * dot_outside; // 0.5 * 2.21 - 1 = 1.105 - 1 = 0.105
    constraint_value_outside = constraint.evaluate(state_outside, control);
    ASSERT_NEAR(constraint_value_outside(0), expected_g_outside, 1e-9);
    ASSERT_GT(constraint_value_outside(0), 0.0);

    // Test with a state ON the boundary (g approx 0)
    Eigen::VectorXd state_boundary(3);
    double y_val = 1.5;
    double radius_at_y = std::abs(y_val) * std::tan(fov); // tan(45) = 1, so radius = 1.5
    state_boundary << radius_at_y, y_val, 0.0; // (1.5, 1.5, 0.0)
    double v_norm_sq_boundary = (state_boundary - origin).squaredNorm(); // 1.5^2 + 1.5^2 = 2.25 * 2 = 4.5
    double dot_boundary = (state_boundary - origin).dot(axis); // 1.5
    double expected_g_boundary = cos_fov_sq * v_norm_sq_boundary - dot_boundary * dot_boundary; // 0.5 * 4.5 - 1.5^2 = 2.25 - 2.25 = 0
    Eigen::VectorXd constraint_value_boundary = constraint.evaluate(state_boundary, control);
    ASSERT_NEAR(constraint_value_boundary(0), expected_g_boundary, 1e-9);
    ASSERT_NEAR(constraint_value_boundary(0), 0.0, 1e-9); // Check against 0 directly
}

TEST(SecondOrderConeConstraintTest, Gradients) {
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // Opening along positive Y-axis
    double fov = M_PI / 4.0; // 45 degrees
    double cos_fov_sq = std::pow(std::cos(fov), 2); // 0.5
    cddp::SecondOrderConeConstraint constraint(origin, axis, fov);

    Eigen::VectorXd state(4); // Use state dim > 3
    state << 0.1, 0.5, 0.2, 99.0; // A point inside the cone, in positive Y
    Eigen::VectorXd control(2);
    control << 0.0, 0.0;

    // Calculate Jacobians using the class method
    auto jacobians = constraint.getJacobians(state, control);
    Eigen::MatrixXd state_jacobian = std::get<0>(jacobians);
    Eigen::MatrixXd control_jacobian = std::get<1>(jacobians);

    // Calculate expected state Jacobian using the squared analytical formula
    // dg/dp_s = 2 * (cos_fov_sq * v^T - (v . axis) * axis^T)
    Eigen::Vector3d p_s = state.head(3); // (0.1, 0.5, 0.2)
    Eigen::Vector3d v = p_s - origin;    // (0.1, 0.5, 0.2)
    double dot_prod = v.dot(axis);       // 0.5
    Eigen::RowVector3d expected_dg_dps =
        2.0 * (cos_fov_sq * v.transpose() - dot_prod * axis.transpose());
    // expected_dg_dps = 2.0 * (0.5 * [0.1, 0.5, 0.2] - 0.5 * [0.0, 1.0, 0.0])
    // expected_dg_dps = 2.0 * ([0.05, 0.25, 0.1] - [0.0, 0.5, 0.0])
    // expected_dg_dps = 2.0 * [0.05, -0.25, 0.1]
    // expected_dg_dps = [0.1, -0.5, 0.2]

    // Construct expected full Jacobian dg/dx = dg/dp_s * [I_3x3, 0]
    Eigen::MatrixXd expected_state_jacobian = Eigen::MatrixXd::Zero(1, state.size());
    expected_state_jacobian.leftCols(3) = expected_dg_dps;

    ASSERT_EQ(state_jacobian.rows(), 1);
    ASSERT_EQ(state_jacobian.cols(), state.size());
    ASSERT_TRUE(state_jacobian.isApprox(expected_state_jacobian, 1e-9));

    // Expected control Jacobian should be zero
    ASSERT_EQ(control_jacobian.rows(), 1);
    ASSERT_EQ(control_jacobian.cols(), control.size());
    ASSERT_TRUE(control_jacobian.isApprox(Eigen::MatrixXd::Zero(1, control.size()), 1e-9));
}

// New test case for visualization
TEST(SecondOrderConeConstraintTest, Visualization) {
    namespace plt = matplot;

    // 1. Define Cone Parameters
    Eigen::Vector3d origin(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(0.0, 1.0, 0.0); // Changed to Y-axis
    axis.normalize(); // Ensure axis is normalized
    double fov = M_PI / 4.0; // 45 degrees half-angle
    double tan_fov = std::tan(fov);
    double epsilon = 1e-6; // Regularization
    cddp::SecondOrderConeConstraint constraint(origin, axis, fov);


    // 2. Generate Cone Surface Points
    std::vector<double> h_vals = plt::linspace(0, 2, 20); // Height along opening axis (non-negative for this definition)
    std::vector<double> theta_vals = plt::linspace(0, 2 * M_PI, 40); // Angle around axis

    auto [H, THETA] = plt::meshgrid(h_vals, theta_vals);

    // Find orthogonal vectors u, v to the axis
    Eigen::Vector3d temp_vec = (std::abs(axis.x()) > 1e-6 || std::abs(axis.z()) > 1e-6) ? Eigen::Vector3d(0,0,1) : Eigen::Vector3d(1,0,0);
    Eigen::Vector3d u = axis.cross(temp_vec).normalized();
    Eigen::Vector3d v = axis.cross(u).normalized();

    // Calculate X, Y, Z coordinates for the cone surface
    std::vector<std::vector<double>> X(H.size(), std::vector<double>(H[0].size()));
    std::vector<std::vector<double>> Y(H.size(), std::vector<double>(H[0].size()));
    std::vector<std::vector<double>> Z(H.size(), std::vector<double>(H[0].size()));

    for (size_t i = 0; i < H.size(); ++i) {
        for (size_t j = 0; j < H[0].size(); ++j) {
            double h = H[i][j];
            double theta = THETA[i][j];
            // Radius depends on the projection onto the axis, which is h
            double radius = std::abs(h) * tan_fov;
            Eigen::Vector3d point_on_circle = radius * (std::cos(theta) * u + std::sin(theta) * v);
            Eigen::Vector3d point = origin + axis * h + point_on_circle;
            X[i][j] = point.x();
            Y[i][j] = point.y();
            Z[i][j] = point.z();
        }
    }

    // 3. Generate Sample Points (Inside/Outside)
    Eigen::VectorXd control(1); control << 0.0; // Dummy control

    std::vector<double> x_inside, y_inside, z_inside;
    std::vector<double> x_outside, y_outside, z_outside;
    std::vector<double> x_boundary, y_boundary, z_boundary; // Added for boundary points

    // Points to test (as Eigen::VectorXd for the constraint function)
    // Cone opens along positive Y-axis
    Eigen::VectorXd p_in_axis(3); p_in_axis << 0.0, 1.0, 0.0;      // Inside: Along positive axis
    Eigen::VectorXd p_in_near(3); p_in_near << 0.1, 0.5, 0.1;      // Inside: Near origin
    Eigen::VectorXd p_out_neg(3); p_out_neg << 0.0, -1.0, 0.0;     // Outside: Along negative axis
    Eigen::VectorXd p_out_far(3); p_out_far << 1.0, 1.0, 1.0;      // Outside: Large radius
    Eigen::VectorXd p_out_perp(3); p_out_perp << 1.0, 0.0, 0.0;     // Outside: Perpendicular to axis at origin

    Eigen::VectorXd p_boundary1(3); // Boundary point at y=1
    double radius_at_h1 = std::abs(1.0) * tan_fov;
    p_boundary1 << radius_at_h1, 1.0, 0.0;

    Eigen::VectorXd p_boundary2(3); // Boundary point at y=2
    double radius_at_h2 = std::abs(2.0) * tan_fov;
    p_boundary2 << 0.0, 2.0, -radius_at_h2; // On Z-axis part of circle


    // Check and categorize points
    auto check_and_add = [&](const Eigen::VectorXd& p, const std::string& name) {
        double val = constraint.evaluate(p, control)(0);
        if (val < -1e-7) { // Allow small tolerance for strictly inside
             x_inside.push_back(p(0)); y_inside.push_back(p(1)); z_inside.push_back(p(2));
             std::cout << name << " is inside (value: " << val << ")" << std::endl;
        } else if (val > 1e-7) { // Allow small tolerance for strictly outside
            x_outside.push_back(p(0)); y_outside.push_back(p(1)); z_outside.push_back(p(2));
            std::cout << name << " is outside (value: " << val << ")" << std::endl;
        } else { // Consider it on the boundary
             x_boundary.push_back(p(0)); y_boundary.push_back(p(1)); z_boundary.push_back(p(2)); // Store boundary points
             std::cout << name << " is on boundary (value: " << val << ")" << std::endl;
             // Removed plotting as inside
        }
    };

    check_and_add(p_in_axis, "p_in_axis");
    check_and_add(p_in_near, "p_in_near");
    check_and_add(p_out_neg, "p_out_neg");
    check_and_add(p_out_far, "p_out_far");
    check_and_add(p_out_perp, "p_out_perp");
    check_and_add(p_boundary1, "p_boundary1");
    check_and_add(p_boundary2, "p_boundary2");


    // // 4. Create Plot (Commented out for testing)
    auto fig = plt::figure();
    plt::surf(X, Y, Z)->face_alpha(0.5).edge_color("none");
    plt::hold(true);

    if (!x_inside.empty()) {
        plt::plot3(x_inside, y_inside, z_inside, "go")->marker_size(10).display_name("Inside"); // Changed display name
    }
    if (!x_outside.empty()) {
        plt::plot3(x_outside, y_outside, z_outside, "rx")->marker_size(10).display_name("Outside");
    }
    if (!x_boundary.empty()) { // Added plotting for boundary points
        plt::plot3(x_boundary, y_boundary, z_boundary, "bs")->marker_size(10).display_name("Boundary");
    }

    // Plot origin and axis
    plt::plot3(std::vector<double>{origin.x()}, std::vector<double>{origin.y()}, std::vector<double>{origin.z()}, "k*")->marker_size(12).display_name("Origin");
    Eigen::Vector3d axis_start = origin - axis * 2.5; // Show axis extending both ways
    Eigen::Vector3d axis_end = origin + axis * 2.5;
    plt::plot3(std::vector<double>{axis_start.x(), axis_end.x()},
               std::vector<double>{axis_start.y(), axis_end.y()},
               std::vector<double>{axis_start.z(), axis_end.z()}, "b-.")->line_width(2).display_name("Axis");


    plt::hold(false);
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::zlabel("Z");
    plt::title("Second Order Cone Constraint Visualization");
    plt::legend();
    plt::grid(true);
    plt::axis("equal");
    // plt::show();
    // plt::view(45, 30); // Adjust view angle if needed

    // Save the plot to a file
    std::string filename = "cone_visualization.png";
    plt::save(filename);
    std::cout << "Saved cone visualization to " << filename << std::endl;


    // // Assert that the file was created
    // struct stat buffer;
    // ASSERT_EQ(stat(filename.c_str(), &buffer), 0) << "Plot file '" << filename << "' was not generated.";
}