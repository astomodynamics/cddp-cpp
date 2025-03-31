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
    
    // Use the finite difference Jacobian helper function
    auto dynamics_func = [this, &control](const Eigen::VectorXd& s) {
        return this->getDiscreteDynamics(s, control);
    };
    
    // Get discretized Jacobian 
    Eigen::MatrixXd J = finite_difference_jacobian(dynamics_func, state);

    // Convert discrete Jacobian to continuous time Jacobian
    // A = timestep_ * Fx + I -> Fx = (A - I)/timestep_
    J.diagonal().array() -= 1.0;
    J /= timestep_;
    
    return J;
}

Eigen::MatrixXd Car::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // Use the finite difference Jacobian helper function
    auto dynamics_func = [this, &state](const Eigen::VectorXd& c) {
        return this->getDiscreteDynamics(state, c); 
    };
    
    // Get discretized Jacobian
    Eigen::MatrixXd J = finite_difference_jacobian(dynamics_func, control);

    // Convert discrete Jacobian to continuous time Jacobian
    // B = timestep_ * Fu -> Fu = B/timestep_
    J /= timestep_;
    
    return J;
}

std::vector<Eigen::MatrixXd> Car::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> Car::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}

} // namespace cddp