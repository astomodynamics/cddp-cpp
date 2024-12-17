#include "cddp_core/constraint.hpp"
#include <iostream>

namespace cddp
{

LogBarrier::LogBarrier(double barrier_coeff, double relaxation_coeff, int barrier_order)
    : barrier_coeff_(barrier_coeff), relaxation_coeff_(relaxation_coeff), barrier_order_(barrier_order) {}

double LogBarrier::evaluate(const Constraint &constraint, const Eigen::VectorXd &state, const Eigen::VectorXd &control) const
{
    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    Eigen::VectorXd lower_bound = constraint.getLowerBound();
    Eigen::VectorXd upper_bound = constraint.getUpperBound();

    double barrier_cost = 0.0;
    for (int i = 0; i < constraint_value.size(); ++i)
    {
        if (constraint.getName() == "StateBoxConstraint")
        {
            double upper = -upper_bound(i) + constraint_value(i);
            double lower = -constraint_value(i) + lower_bound(i);

            if (upper > relaxation_coeff_)
            {
                barrier_cost -= std::log(upper);
            }
            else
            {
                barrier_cost += (barrier_order_ - 1) / barrier_order_ * (std::pow((upper - barrier_order_ * relaxation_coeff_) / relaxation_coeff_ * (barrier_order_ - 1), barrier_order_) - 1.0) - std::log(relaxation_coeff_);
            }

            if (lower > relaxation_coeff_)
            {
                barrier_cost -= std::log(lower);
            }
            else
            {
                barrier_cost += (barrier_order_ - 1) / barrier_order_ * (std::pow((lower - barrier_order_ * relaxation_coeff_) / relaxation_coeff_ * (barrier_order_ - 1), barrier_order_) - 1.0) - std::log(relaxation_coeff_);
            }
        }
    }
    return barrier_cost * barrier_coeff_;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> LogBarrier::getGradients(const Constraint &constraint, const Eigen::VectorXd &state, const Eigen::VectorXd &control, double barrier_value) const
{
    if (constraint.getName() == "StateBoxConstraint") {
    
    }
    Eigen::VectorXd state_grad = Eigen::VectorXd::Zero(state.size());
    Eigen::VectorXd control_grad = Eigen::VectorXd::Zero(control.size());

    // Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    // Eigen::VectorXd lower_bound = constraint.getLowerBound();
    // Eigen::VectorXd upper_bound = constraint.getUpperBound();

    // for (int i = 0; i < constraint_value.size(); ++i)
    // {
    //     double upper = upper_bound(i) - constraint_value(i);
    //     double lower = constraint_value(i) - lower_bound(i);
    // }

    return {state_grad * barrier_coeff_, control_grad * barrier_coeff_};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> LogBarrier::getHessians(const Constraint &constraint, const Eigen::VectorXd &state, const Eigen::VectorXd &control, double barrier_value) const
{
    Eigen::MatrixXd state_hess = Eigen::MatrixXd::Zero(state.size(), state.size());
    Eigen::MatrixXd control_hess = Eigen::MatrixXd::Zero(control.size(), control.size());
    Eigen::MatrixXd cross_hess = Eigen::MatrixXd::Zero(control.size(), state.size());

    Eigen::VectorXd constraint_value = constraint.evaluate(state, control);
    Eigen::VectorXd lower_bound = constraint.getLowerBound();
    Eigen::VectorXd upper_bound = constraint.getUpperBound();

    if (constraint.getName() == "StateBoxConstraint")
    {
    
    }

    return {state_hess * barrier_coeff_, control_hess * barrier_coeff_, cross_hess * barrier_coeff_};
}
} // namespace cddp