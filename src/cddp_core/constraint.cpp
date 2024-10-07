#include "cddp-cpp/cddp_core/constraint.hpp"

namespace cddp {

class ControlBoxConstraint : public Constraint {
public:
    ControlBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) 
        : lower_bound_(lower_bound), upper_bound_(upper_bound) {}

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
        : lower_bound_(lower_bound), upper_bound_(upper_bound) {}

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
    CircleConstraint(double radius) : radius_(radius) {}

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