#include "dynamics_model/car.hpp"
#include <cmath>

namespace cddp {

Car::Car(double timestep, double wheelbase, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      wheelbase_(wheelbase) {
}

Eigen::VectorXd Car::getDiscreteDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // Extract states
    const double x = state(STATE_X);         // x position
    const double y = state(STATE_Y);         // y position
    const double theta = state(STATE_THETA);  // car angle
    const double v = state(STATE_V);         // velocity

    // Extract controls
    const double delta = control(CONTROL_DELTA);  // steering angle
    const double a = control(CONTROL_A);          // acceleration

    // Constants
    const double d = wheelbase_;  // distance between back and front axles
    const double h = timestep_;   // timestep

    // Compute unit vector in car direction
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    Eigen::Vector2d z(cos_theta, sin_theta);

    // Front wheel rolling distance
    const double f = h * v;

    // Back wheel rolling distance
    // b = d + f*cos(w) - sqrt(d^2 - (f*sin(w))^2)
    const double b = d + f * std::cos(delta) - 
                    std::sqrt(d*d - std::pow(f * std::sin(delta), 2));

    // Change in car angle
    // dtheta = asin(sin(w)*f/d)
    const double dtheta = std::asin(std::sin(delta) * f / d);

    // Compute state change
    Eigen::VectorXd dy = Eigen::VectorXd::Zero(STATE_DIM);
    dy(STATE_X) = b * cos_theta;
    dy(STATE_Y) = b * sin_theta;
    dy(STATE_THETA) = dtheta;
    dy(STATE_V) = h * a;

    // Return next state
    return state + dy;
}

Eigen::MatrixXd Car::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    const int n = STATE_DIM;
    Eigen::MatrixXd J(n, n);
    const double h = 2e-17; // Small perturbation value
    
    // Get nominal output
    Eigen::VectorXd f0 = getDiscreteDynamics(state, control);
    
    // Compute Jacobian column by column using finite differences
    Eigen::VectorXd perturbed_state = state;
    for (int i = 0; i < n; ++i) {
        // Perturb state element
        perturbed_state(i) = state(i) + h;
        
        // Get perturbed output
        Eigen::VectorXd f1 = getDiscreteDynamics(perturbed_state, control);
        
        // Compute derivative using central difference
        J.col(i) = (f1 - f0) / h;
        
        // Reset perturbation
        perturbed_state(i) = state(i);
    }
    
    // Expected output of the state Jacobian is continuous dynamics Jacobian so modify this discrete one
    // Ref: in cddp_core.cpp
    /*
        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx;
        A.diagonal().array() += 1.0; // More efficient way to add identity
        B = timestep_ * Fu;
     */
    J.diagonal().array() -= 1.0;
    J /= timestep_;
    return J;
}

Eigen::MatrixXd Car::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    const int n = STATE_DIM;
    const int m = CONTROL_DIM;
    Eigen::MatrixXd J(n, m);
    const double h = 2e-17; // Small perturbation value
    
    // Get nominal output
    Eigen::VectorXd f0 = getDiscreteDynamics(state, control);
    
    // Compute Jacobian column by column using finite differences
    Eigen::VectorXd perturbed_control = control;
    for (int i = 0; i < m; ++i) {
        // Perturb control element
        perturbed_control(i) = control(i) + h;
        
        // Get perturbed output
        Eigen::VectorXd f1 = getDiscreteDynamics(state, perturbed_control);
        
        // Compute derivative using forward difference
        J.col(i) = (f1 - f0) / h;
        
        // Reset perturbation
        perturbed_control(i) = control(i);
    }

    // Expected output of the control Jacobian is continuous dynamics Jacobian so modify this discrete one
    // Ref: in cddp_core.cpp
    /*
        // Convert continuous dynamics to discrete time
        A = timestep_ * Fx;
        A.diagonal().array() += 1.0; // More efficient way to add identity
        B = timestep_ * Fu;
     */
    J /= timestep_;

    return J;
}

Eigen::MatrixXd Car::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
}

Eigen::MatrixXd Car::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    return Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
}

} // namespace cddp