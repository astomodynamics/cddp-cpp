
/**
 * @file test_autodiff.cpp
 * @brief Test the autodiff library functionality
 */

#include <gtest/gtest.h>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <cddp-cpp/cddp_core/helper.hpp>
#include <Eigen/Dense>

using namespace autodiff;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Simple scalar function: f(x) = x^2 + 3x + 2
dual scalar_function(const dual& x) {
    return x*x + 3*x + 2;
}

// Multivariate function: f(x,y) = x^2 + 2xy + y^2
dual multivariate_function(const VectorXdual& x) {
    return x(0)*x(0) + 2*x(0)*x(1) + x(1)*x(1);
}

// Function to compute with Eigen: f(x) = x^T.A.x + b^T.x
dual quadratic_form(const VectorXdual& x) {
    MatrixXd A = MatrixXd::Identity(2, 2);
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    
    VectorXd b(2);
    b << 3.0, 2.0;
    
    return x.dot(A * x) + b.dot(x);
}

// Equivalent using regular Eigen for numerical diff comparison
double quadratic_form_eigen(const Eigen::VectorXd& x) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    
    Eigen::VectorXd b(2);
    b << 3.0, 2.0;
    
    return x.dot(A * x) + b.dot(x);
}

TEST(AutoDiffTest, ScalarDerivative) {
    dual x = 2.0;
    dual y = scalar_function(x);
    double dydx = derivative(scalar_function, wrt(x), at(x));
    
    // For f(x) = x^2 + 3x + 2, f'(x) = 2x + 3
    // At x = 2, f'(2) = 2*2 + 3 = 7
    EXPECT_DOUBLE_EQ(dydx, 7.0);
}

TEST(AutoDiffTest, MultivariateGradient) {
    VectorXdual x(2);
    x << 1.0, 2.0;
    
    // Compute gradient using autodiff
    VectorXd grad = gradient(multivariate_function, wrt(x), at(x));
    
    // For f(x,y) = x^2 + 2xy + y^2
    // df/dx = 2x + 2y
    // df/dy = 2x + 2y
    // At x=1, y=2: df/dx = 2*1 + 2*2 = 6, df/dy = 2*1 + 2*2 = 6
    EXPECT_DOUBLE_EQ(grad(0), 6.0);
    EXPECT_DOUBLE_EQ(grad(1), 6.0);
}

TEST(AutoDiffTest, QuadraticFormGradient) {
    VectorXdual x(2);
    x << 1.0, 2.0;
    
    // Compute gradient using autodiff
    VectorXd autodiff_grad = gradient(quadratic_form, wrt(x), at(x));
    
    // Compute gradient using finite differences for comparison
    Eigen::VectorXd x_eigen(2);
    x_eigen << 1.0, 2.0;
    Eigen::VectorXd finite_diff_grad = cddp::finite_difference_gradient(quadratic_form_eigen, x_eigen);
    
    // Print results for debugging
    std::cout << "AutoDiff gradient: " << autodiff_grad.transpose() << std::endl;
    std::cout << "Finite diff gradient: " << finite_diff_grad.transpose() << std::endl;
    
    // Check that gradients are close (within a small tolerance)
    EXPECT_NEAR(autodiff_grad(0), finite_diff_grad(0), 1e-5);
    EXPECT_NEAR(autodiff_grad(1), finite_diff_grad(1), 1e-5);
}

TEST(AutoDiffTest, CompareWithFiniteDifference) {
    // Define test point
    Eigen::VectorXd x(2);
    x << 2.0, 3.0;
    
    // Equivalent autodiff point
    VectorXdual x_dual(2);
    x_dual << 2.0, 3.0;
    
    // Compute gradient with autodiff
    VectorXd autodiff_grad = gradient(quadratic_form, wrt(x_dual), at(x_dual));
    
    // Compute using finite difference (with different methods)
    Eigen::VectorXd finite_diff_central = cddp::finite_difference_gradient(quadratic_form_eigen, x, 1e-6, 0);
    Eigen::VectorXd finite_diff_forward = cddp::finite_difference_gradient(quadratic_form_eigen, x, 1e-6, 1);
    Eigen::VectorXd finite_diff_backward = cddp::finite_difference_gradient(quadratic_form_eigen, x, 1e-6, 2);
    
    // Print results
    std::cout << "AutoDiff gradient: " << autodiff_grad.transpose() << std::endl;
    std::cout << "Finite diff (central): " << finite_diff_central.transpose() << std::endl;
    std::cout << "Finite diff (forward): " << finite_diff_forward.transpose() << std::endl;
    std::cout << "Finite diff (backward): " << finite_diff_backward.transpose() << std::endl;
    
    // Compare results (central difference should be closer to autodiff)
    EXPECT_NEAR(autodiff_grad(0), finite_diff_central(0), 1e-5);
    EXPECT_NEAR(autodiff_grad(1), finite_diff_central(1), 1e-5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}