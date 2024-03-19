#include <iostream> 
#include <cassert> 
#include "cddp_core/Objective.hpp"
#include "Eigen/Dense"

namespace cddp { 

// Helper function to compare vectors with tolerance
bool compareVectors(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double tolerance = 1e-6) {
    if (v1.size() != v2.size()) {
        return false;
    }
    return (v1 - v2).array().abs().maxCoeff() <= tolerance;
}

// Helper function to compare matrices with tolerance
bool compareMatrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, double tolerance = 1e-6) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return false;
    }
    return (m1 - m2).array().abs().maxCoeff() <= tolerance;
}

void testQuadraticCost() {
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 5;
    double dt = 0.1;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim) * 0.1; 
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim) * 2.0;
    Eigen::VectorXd goal_state = Eigen::VectorXd::Zero(state_dim);

    // Instantiate QuadraticCost object
    QuadraticCost cost_fn(Q, R, Qf, goal_state, dt);

    // Example state and control
    Eigen::VectorXd x(state_dim);
    x << 1.0, 0.5;
    Eigen::VectorXd u(control_dim);
    u << 0.8;

    // Test running cost 
    double expected_running_cost = ((x - goal_state).transpose() * Q * (x - goal_state))[0] + (u.transpose() * R * u)[0] * dt;
    double calculated_running_cost = cost_fn.calculateRunningCost(x, u);
    assert(std::abs(calculated_running_cost - expected_running_cost) < 1e-6);

    // Test running cost gradient
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> grads = cost_fn.calculateRunningCostGradient(x, u);
    Eigen::VectorXd expected_grad_x = 2 * Q * (x - goal_state) * dt; 
    Eigen::VectorXd expected_grad_u = 2 * R * u * dt; 
    assert(compareVectors(std::get<0>(grads), expected_grad_x));
    assert(compareVectors(std::get<1>(grads), expected_grad_u));
}

}  // namespace cddp (optional)

int main() {
    cddp::testQuadraticCost();
    std::cout << "Objective tests passed!" << std::endl;
    return 0;
}
