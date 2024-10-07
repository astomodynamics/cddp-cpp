#ifndef CDDP_CONSTRAINT_HPP
#define CDDP_CONSTRAINT_HPP

#include <Eigen/Dense>

namespace cddp {

class Constraint {
public:
    // Constructor
    Constraint() {}

    // Evaluate the constraint function: g(x, u)
    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Get the lower bound of the constraint
    virtual Eigen::VectorXd getLowerBound() const = 0;

    // Get the upper bound of the constraint
    virtual Eigen::VectorXd getUpperBound() const = 0;

    // Get the Jacobian of the constraint w.r.t the state: dg/dx
    virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Get the Jacobian of the constraint w.r.t the control: dg/du
    virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Get both Jacobians: dg/dx, dg/du
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getJacobians(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
        return {getStateJacobian(state, control), getControlJacobian(state, control)};
    }
};

} // namespace cddp

#endif // CDDP_CONSTRAINT_HPP