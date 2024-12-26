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
    
    virtual double computeViolation(const Eigen::VectorXd& state, 
                                  const Eigen::VectorXd& control) const = 0;

    virtual double computeViolationFromValue(const Eigen::VectorXd& g) const = 0;
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

    Eigen::VectorXd clamp(const Eigen::VectorXd& control) const {
        return control.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override {
   
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        return (g - upper_bound_).cwiseMax(0.0).sum() + 
               (lower_bound_ - g).cwiseMax(0.0).sum();
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

    Eigen::VectorXd clamp(const Eigen::VectorXd& state) const {
        return state.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override {
   
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        return (g - upper_bound_).cwiseMax(0.0).sum() + 
               (lower_bound_ - g).cwiseMax(0.0).sum();
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

    double computeViolation(const Eigen::VectorXd& state, 
                       const Eigen::VectorXd& control) const override {
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        return std::max(0.0, g(0) - radius_ * radius_);
    }


private:
    double radius_;
};

} // namespace cddp

#endif // CDDP_CONSTRAINT_HPP