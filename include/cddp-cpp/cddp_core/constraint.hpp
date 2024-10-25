#ifndef CDDP_CONSTRAINT_HPP
#define CDDP_CONSTRAINT_HPP

#include <Eigen/Dense>

namespace cddp {

class Constraint {
public:
    // Constructor
    Constraint(const std::string& name) : name_(name) {}

    // Get the name of the constraint
    const std::string& getName() const { return name_; }

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
private:
    std::string name_; // Name of the constraint
};


class ControlBoxConstraint : public Constraint {
public:
    ControlBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) 
        : Constraint("ControlBoxConstraint"), lower_bound_(lower_bound), upper_bound_(upper_bound) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return control; // The constraint is directly on the control
    }

    Eigen::VectorXd getLowerBound() const override {
        return lower_bound_;
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Zero(control.size(), state.size()); 
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Identity(control.size(), control.size()); 
    }

private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

class StateBoxConstraint : public Constraint {
public:
    StateBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) 
        : Constraint("StateBoxConstraint"), lower_bound_(lower_bound), upper_bound_(upper_bound) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return state; // The constraint is directly on the state
    }

    Eigen::VectorXd getLowerBound() const override {
        return lower_bound_;
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Identity(state.size(), state.size()); 
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Zero(state.size(), control.size()); 
    }

private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

// CircleConstraint (assuming the circle is centered at the origin)
class CircleConstraint : public Constraint {
public:
    CircleConstraint(double radius) : Constraint("CircleConstraint"), radius_(radius) {}

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        Eigen::Vector2d position(state(0), state(1)); // Assuming the first two elements of the state are x and y position
        return Eigen::VectorXd::Constant(1, position.squaredNorm()); 
    }

    Eigen::VectorXd getLowerBound() const override {
        return Eigen::VectorXd::Constant(1, 0.0); 
    }

    Eigen::VectorXd getUpperBound() const override {
        return Eigen::VectorXd::Constant(1, radius_ * radius_); 
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        Eigen::MatrixXd jacobian(1, state.size());
        jacobian << 2 * state(0), 2 * state(1), Eigen::RowVectorXd::Zero(state.size() - 2); // Assuming x and y are the first two state elements
        return jacobian;
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return Eigen::MatrixXd::Zero(1, control.size()); 
    }

private:
    double radius_;
};


} // namespace cddp

#endif // CDDP_CONSTRAINT_HPP