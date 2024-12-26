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
#include "cddp_core/constraint.hpp"
#include "cddp_core/barrier.hpp"
#include <iostream>

namespace cddp
{

LogBarrier::LogBarrier(double barrier_coeff, double relaxation_coeff, int barrier_order, bool is_relaxed_log_barrier)
    : barrier_coeff_(barrier_coeff), relaxation_coeff_(relaxation_coeff), barrier_order_(barrier_order), is_relaxed_log_barrier_(is_relaxed_log_barrier) {}

double LogBarrier::evaluate(const Constraint &constraint, const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
{
    const double eps = barrier_coeff_ * 1e2; // Small value to avoid log(0)
    double barrier_cost = 0.0;
    if (constraint.getName() == "ControlBoxConstraint" || 
        constraint.getName() == "StateBoxConstraint")
    {
        const Eigen::VectorXd& constraint_value = constraint.evaluate(state, control);
        const Eigen::VectorXd& lower_bound = constraint.getLowerBound();
        const Eigen::VectorXd& upper_bound = constraint.getUpperBound();

        Eigen::VectorXd upper = -upper_bound + constraint_value;
        Eigen::VectorXd lower = -constraint_value + lower_bound;

        // Apply bounds to avoid log(0)
        upper = upper.array().max(eps);
        lower = lower.array().max(eps);

        const Eigen::VectorXd& upper_log = upper.array().log();
        const Eigen::VectorXd& lower_log = lower.array().log();

        barrier_cost = barrier_coeff_ * (upper_log.sum() + lower_log.sum());
        
    } else {
        Eigen::VectorXd constraint_value = constraint.evaluate(state, control);

        // Apply bounds to avoid log(0)
        constraint_value = constraint_value.array().max(eps);

        barrier_cost = barrier_coeff_ * std::log(constraint_value.norm());
    }
    return barrier_cost;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> LogBarrier::getGradients(const Constraint &constraint, const Eigen::VectorXd &state, const Eigen::VectorXd &control, double barrier_value) const
{
    const double eps = barrier_coeff_ * 1e2; // Small value to avoid log(0)
    Eigen::VectorXd state_grad = Eigen::VectorXd::Zero(state.size());
    Eigen::VectorXd control_grad = Eigen::VectorXd::Zero(control.size());
    if (constraint.getName() == "ControlBoxConstraint" || 
        constraint.getName() == "StateBoxConstraint") 
    {
        const Eigen::VectorXd& constraint_value = constraint.evaluate(state, control);
        const Eigen::VectorXd& lower_bound = constraint.getLowerBound();
        const Eigen::VectorXd& upper_bound = constraint.getUpperBound();

         Eigen::VectorXd upper_diff = (upper_bound - constraint_value).unaryExpr([eps](double x) { 
            return std::max(x, eps); 
        });
        Eigen::VectorXd lower_diff = (constraint_value - lower_bound).unaryExpr([eps](double x) { 
            return std::max(x, eps); 
        });

        // Then compute gradient using array operations
        Eigen::VectorXd gradient = barrier_coeff_ * (
            upper_diff.array().inverse() - lower_diff.array().inverse()
        ).matrix();

        if (constraint.getName() == "ControlBoxConstraint") {
            control_grad = gradient;
        } else { // StateBoxConstraint
            state_grad = gradient;
        }
    } else{

    }

    return {state_grad, control_grad};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> LogBarrier::getHessians(const Constraint &constraint, const Eigen::VectorXd &state, const Eigen::VectorXd &control, double barrier_value) const
{
    const double eps = barrier_coeff_ * 1e2; // Small value to avoid log(0)
    Eigen::MatrixXd state_hess = Eigen::MatrixXd::Zero(state.size(), state.size());
    Eigen::MatrixXd control_hess = Eigen::MatrixXd::Zero(control.size(), control.size());
    Eigen::MatrixXd cross_hess = Eigen::MatrixXd::Zero(control.size(), state.size());

    if (constraint.getName() == "ControlBoxConstraint" || 
        constraint.getName() == "StateBoxConstraint") 
    {
        const Eigen::VectorXd& constraint_value = constraint.evaluate(state, control);
        const Eigen::VectorXd& lower_bound = constraint.getLowerBound();
        const Eigen::VectorXd& upper_bound = constraint.getUpperBound();

        Eigen::VectorXd upper_diff = (upper_bound - constraint_value).unaryExpr([eps](double x) { 
            return std::max(x, eps); 
        });
        Eigen::VectorXd lower_diff = (constraint_value - lower_bound).unaryExpr([eps](double x) { 
            return std::max(x, eps); 
        });

        // Compute squared reciprocals using array operations
        Eigen::VectorXd hess_diag = barrier_coeff_ * (
            upper_diff.array().square().inverse() + 
            lower_diff.array().square().inverse()
        ).matrix();

        if (constraint.getName() == "ControlBoxConstraint") {
            control_hess = hess_diag.asDiagonal();
        } else { // StateBoxConstraint
            state_hess = hess_diag.asDiagonal();
        }
    } else {

    }

    return {state_hess, control_hess, cross_hess};
}
} // namespace cddp