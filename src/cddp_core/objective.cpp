#include <iostream>
#include <Eigen/Dense>

#include "cddp-cpp/cddp_core/objective.hpp"

namespace cddp {
// Quadratic Objective: 0.5 * (x - x_reference)^T Qf (x - x_reference) + sum[0.5 * (x - x_reference)^T Q (x - x_reference) + 0.5 * u^T R u]
// Constructor 
QuadraticObjective::QuadraticObjective(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf, 
                       const Eigen::VectorXd& reference_state)
    : Q_(Q), R_(R), Qf_(Qf), reference_state_(reference_state) {
    // Ensure Q, R, and Qf are positive definite (you might add checks here)
}

// Evaluate the running cost
double QuadraticObjective::evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    Eigen::VectorXd state_error = state - reference_state_;
    return (state_error.transpose() * Q_ * state_error ).value()+ (control.transpose() * R_ * control).value();
}

// Evaluate the final/terminal cost
double QuadraticObjective::terminal_cost(const Eigen::VectorXd& final_state) const {
    Eigen::VectorXd state_error = final_state - reference_state_;
    return (state_error.transpose() * Qf_ * state_error).value();
}

// Gradient of the running cost w.r.t state
Eigen::VectorXd QuadraticObjective::getRunningCostStateGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    Eigen::VectorXd state_error = state - reference_state_;
    return 2.0 * Q_ * state_error;
}

// Gradient of the running cost w.r.t control
Eigen::VectorXd QuadraticObjective::getRunningCostControlGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const  {
    return 2.0 * R_ * control;
}

// Gradient of the final cost w.r.t state
Eigen::VectorXd QuadraticObjective::getFinalCostGradient(const Eigen::VectorXd& final_state) const {
    Eigen::VectorXd state_error = final_state - reference_state_;
    return 2.0 * Qf_ * state_error;
}

// Hessians of the running cost (constant for quadratic objectives)
Eigen::MatrixXd QuadraticObjective::getRunningCostStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const  {
    return 2.0 * Q_;
}

Eigen::MatrixXd QuadraticObjective::getRunningCostControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const  {
    return 2.0 * R_;
}

Eigen::MatrixXd QuadraticObjective::getRunningCostCrossHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const  {
    return Eigen::MatrixXd::Zero(state.size(), control.size());
}

// Hessian of the final cost
Eigen::MatrixXd QuadraticObjective::getFinalCostHessian(const Eigen::VectorXd& final_state) const {
    return 2.0 * Qf_;
}

// TODO: Implement the QuadraticTrackingObjective class


} // namespace cddp