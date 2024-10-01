#include "cddp-cpp/dynamics_model/pendulum.hpp" 
#include <cmath>
#include <Eigen/Dense>

using namespace cddp;

// Constructor
Pendulum::Pendulum(double mass, double length, double gravity, double timestep)
    : DynamicalSystem(2, 1, timestep, "rk4"),  // 2 states (angle, angular velocity), 1 control (torque)
      mass_(mass), length_(length), gravity_(gravity) {}

// Dynamics: xdot = f(x_t, u_t)
Eigen::VectorXd Pendulum::getContinuousDynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {

    double theta = state[0];   // Angle
    double theta_dot = state[1]; // Angular velocity
    double torque = control[0];   // Control input (torque)

    double theta_ddot = - (gravity_ / length_) * sin(theta) + (1.0 / (mass_ * length_ * length_)) * torque; 

    Eigen::VectorXd xdot(2);
    xdot << theta_dot, theta_ddot;    

    return xdot;
}

// Discrete dynamics: x_{t+1} = f(x_t, u_t)
// We'll use the base class implementation, so no need to redefine it here

Eigen::MatrixXd Pendulum::getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) 
const {
    // TODO: Compute and return the Jacobian matrix df/dx
    return Eigen::MatrixXd::Zero(2, 2); 
}

Eigen::MatrixXd Pendulum::getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control)
const {
    // TODO: Compute and return the Jacobian matrix df/du
    return Eigen::MatrixXd::Zero(2, 1); 
}

Eigen::MatrixXd Pendulum::getStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) 
const {
    // TODO: Compute and return the Hessian tensor d^2f/dx^2 (represented as a matrix)
    return Eigen::MatrixXd::Zero(2*2, 2); 
}

Eigen::MatrixXd Pendulum::getControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control)
const {
    // TODO: Compute and return the Hessian tensor d^2f/du^2 (represented as a matrix)
    return Eigen::MatrixXd::Zero(2*1, 1); 
}