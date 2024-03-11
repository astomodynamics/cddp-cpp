
#include <iostream>
#include <cassert> // For basic assertions
#include "cddp/model/DynamicalSystem.hpp"
#include "cddp/model/DoubleIntegrator.hpp"

// Test case for the transition function of the DoubleIntegrator
bool testDoubleIntegratorTransition() {
    cddp::DoubleIntegrator system(4, 2, 0.1, 0);

    Eigen::VectorXd state(4);
    state << 1, 2, 0.5, -0.2; 

    Eigen::VectorXd control(2);
    control << 1.0, -0.5;

    Eigen::VectorXd expected_next_state(4);
    expected_next_state << 1.05, 1.98, 0.6, -0.25; 
    Eigen::VectorXd next_state = system.getDynamics(state, control);

    // Simple assertion: Could be more sophisticated
    if ((next_state - expected_next_state).norm() > 1e-6) {
        std::cout << "Expected: " << expected_next_state.transpose() << std::endl;
        std::cout << "Got: " << next_state.transpose() << std::endl;
        return false; // Test failed
    }
    return true; // Test passed
}

int main() {
    if (testDoubleIntegratorTransition()) {
        std::cout << "DoubleIntegrator transition test passed!" << std::endl;
    } else {
        std::cout << "DoubleIntegrator transition test failed!" << std::endl;
        return 1; // Indicates test failure
    }
    return 0;
}

