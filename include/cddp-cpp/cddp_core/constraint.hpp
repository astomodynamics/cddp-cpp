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
#include <string>
#include <limits>
#include <tuple>
#include <algorithm>

namespace cddp {

class Constraint {
public:
    // Constructor
    Constraint(const std::string& name) : name_(name) {}

    // Get the name of the constraint
    const std::string& getName() const { return name_; }

    virtual int getDualDim() const { return 0; }

    // Evaluate the constraint function: g(x, u)
    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const = 0;

    // Get the lower bound of the constraint
    virtual Eigen::VectorXd getLowerBound() const = 0;

    // Get the upper bound of the constraint
    virtual Eigen::VectorXd getUpperBound() const = 0;

    // Get the Jacobian of the constraint w.r.t the state: dg/dx
    virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                             const Eigen::VectorXd& control) const = 0;

    // Get the Jacobian of the constraint w.r.t the control: dg/du
    virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                               const Eigen::VectorXd& control) const = 0;

    // Utility: Get both Jacobians: dg/dx, dg/du
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
    getJacobians(const Eigen::VectorXd& state, 
                 const Eigen::VectorXd& control) const 
    {
        return {getStateJacobian(state, control), 
                getControlJacobian(state, control)};
    }
    
    // Compute how far the constraint is violated
    virtual double computeViolation(const Eigen::VectorXd& state, 
                                    const Eigen::VectorXd& control) const = 0;

    // Given g(x,u), compute violation from that vector
    virtual double computeViolationFromValue(const Eigen::VectorXd& g) const = 0;

    // Used for constraints with a center (e.g., ball constraint)
    virtual Eigen::VectorXd getCenter() const {
        throw std::logic_error("This constraint type does not have a center.");
    }

private:
    std::string name_; // Name of the constraint
};

//------------------------------------------------------------------------------

class ControlBoxConstraint : public Constraint {
public:
    ControlBoxConstraint(const Eigen::VectorXd& lower_bound, 
                         const Eigen::VectorXd& upper_bound) 
        : Constraint("ControlBoxConstraint"), 
          lower_bound_(lower_bound), 
          upper_bound_(upper_bound) {}

    int getDualDim() const override {
        return lower_bound_.size() + upper_bound_.size();
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, 
                             const Eigen::VectorXd& control) const override 
    {
        return control; 
    }

    Eigen::VectorXd getLowerBound() const override {
        return lower_bound_;
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Zero(control.size(), control.size()); 
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Identity(control.size(), control.size()); 
    }

    Eigen::VectorXd clamp(const Eigen::VectorXd& control) const {
        return control.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override 
    {
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        // Sum of amounts by which g is above upper_bound or below lower_bound
        return (g - upper_bound_).cwiseMax(0.0).sum() + 
               (lower_bound_ - g).cwiseMax(0.0).sum();
    }

private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

class StateBoxConstraint : public Constraint {
public:
    StateBoxConstraint(const Eigen::VectorXd& lower_bound, 
                       const Eigen::VectorXd& upper_bound) 
        : Constraint("StateBoxConstraint"), 
          lower_bound_(lower_bound), 
          upper_bound_(upper_bound) {}
    
    int getDualDim() const override {
        return lower_bound_.size() + upper_bound_.size();
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, 
                             const Eigen::VectorXd& control) const override 
    {
        return state; 
    }

    Eigen::VectorXd getLowerBound() const override {
        return lower_bound_;
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Identity(state.size(), state.size()); 
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Zero(state.size(), control.size()); 
    }

    Eigen::VectorXd clamp(const Eigen::VectorXd& state) const {
        return state.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override 
    {
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        // Same logic as ControlBoxConstraint but for the state
        return (g - upper_bound_).cwiseMax(0.0).sum() + 
               (lower_bound_ - g).cwiseMax(0.0).sum();
    }

private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
};

class LinearConstraint : public Constraint {
public:
    LinearConstraint(const Eigen::MatrixXd& A, 
                     const Eigen::VectorXd& b,
                     double scale_factor = 1.0)
        : Constraint("LinearConstraint"), 
          A_(A), 
          b_(b),
          scale_factor_(scale_factor) {}
    
    int getDualDim() const override {
        return b_.size();
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, 
                             const Eigen::VectorXd& control) const override 
    {
        return A_ * state;
    }

    Eigen::VectorXd getLowerBound() const override {
        return Eigen::VectorXd::Constant(A_.rows(), -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override {
        return b_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override 
    {
        return A_;
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Zero(A_.rows(), control.size());
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override 
    {
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        return std::max(0.0, (b_ - g).maxCoeff());
    }

private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    double scale_factor_;
};

class ControlConstraint : public Constraint {
public:
    ControlConstraint(const Eigen::VectorXd& upper_bound, 
                      double scale_factor = 1.0)
        : Constraint("ControlConstraint"),
          scale_factor_(scale_factor) {
            // Rescale the upper bound for this constraint class
            upper_bound_.resize(2*upper_bound.size());
            upper_bound_.head(upper_bound.size()) = upper_bound * scale_factor_;
            upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
            dim_ = 2*upper_bound.size();
          }

    int getDualDim() const override {
        return dim_;
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd& /**/, 
                             const Eigen::VectorXd& control) const override 
    {
        // return [-control; control];
        Eigen::VectorXd g(2*control.size());
        g.head(control.size()) = -control;
        g.tail(control.size()) = control;
        return g * scale_factor_;
    }

    Eigen::VectorXd getLowerBound() const override {
        return Eigen::VectorXd::Constant(upper_bound_.size(), -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override {
        return upper_bound_;
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Zero(dim_, state.size());
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override 
    {
        Eigen::MatrixXd jac(2*control.size(), control.size());
        jac.topLeftCorner(control.size(), control.size()) = -Eigen::MatrixXd::Identity(control.size(), control.size());
        jac.bottomRightCorner(control.size(), control.size()) = Eigen::MatrixXd::Identity(control.size(), control.size());
        return jac;
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override 
    {
        Eigen::VectorXd g = evaluate(state, control) - upper_bound_;
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override {
        return (g - upper_bound_).cwiseMax(0.0).sum();
    }
private:    
    int dim_;
    Eigen::VectorXd upper_bound_;
    double scale_factor_;
};

class BallConstraint : public Constraint {
public:
    BallConstraint(double radius, 
                   const Eigen::VectorXd& center, 
                   double scale_factor = 1.0)
      : Constraint("BallConstraint"), 
        radius_(radius), 
        center_(center), 
        scale_factor_(scale_factor)
    {
        dim_ = center.size();
    }

    int getDualDim() const override {
        return 1;
    }

   
    Eigen::VectorXd evaluate(const Eigen::VectorXd& state, 
                             const Eigen::VectorXd& control) const override 
    {
        const Eigen::VectorXd& diff = state.head(dim_) - center_;
        return -Eigen::VectorXd::Constant(1, scale_factor_ * diff.squaredNorm());
    }

    Eigen::VectorXd getLowerBound() const override {
        return -Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity()); 
    }

    Eigen::VectorXd getUpperBound() const override {
        return -Eigen::VectorXd::Constant(1, radius_ * radius_) * scale_factor_;
    }

    double computeViolation(const Eigen::VectorXd& state, 
                            const Eigen::VectorXd& control) const override 
    {
        Eigen::VectorXd g = evaluate(state, control);
        return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd& g) const override 
    {
        double val = g(0);
        double lb = getLowerBound()(0);
        return std::max(0.0, val - lb);
    }
    
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
                                     const Eigen::VectorXd& control) const override 
    {
        const Eigen::VectorXd& diff = state.head(dim_) - center_;
        Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(1, state.size());
        
        for (int i = 0; i < dim_; ++i) {
            jac(0, i) = -2.0 * scale_factor_ * diff(i);
        }

        return jac;
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
                                       const Eigen::VectorXd& control) const override 
    {
        return Eigen::MatrixXd::Zero(1, control.size());
    }

    Eigen::VectorXd getCenter() const { return center_; }
    double getRadius() const { return radius_; }

private:
    double radius_;
    Eigen::VectorXd center_;
    int dim_;
    double scale_factor_;
};

// class QuadraticConstraint : public Constraint {
// public:
//     QuadraticConstraint(const Eigen::MatrixXd& Q, 
//                         const Eigen::VectorXd& q, 
//                         double r, 
//                         double scale_factor = 1.0)
//         : Constraint("QuadraticConstraint"), 
//           Q_(Q), 
//           q_(q), 
//           r_(r),
//           scale_factor_(scale_factor) {}

//     Eigen::VectorXd evaluate(const Eigen::VectorXd& state, 
//                              const Eigen::VectorXd& control) const override 
//     {
//         return 0.5 * state.transpose() * Q_ * state + q_.transpose() * state + r_;
//     }

//     Eigen::VectorXd getLowerBound() const override {
//         return Eigen::VectorXd::Constant(1, -std::numeric_limits<double>::infinity());
//     }

//     Eigen::VectorXd getUpperBound() const override {
//         return Eigen::VectorXd::Zero(1);
//     }

//     Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, 
//                                      const Eigen::VectorXd& control) const override 
//     {
//         return Q_ * state + q_;
//     }

//     Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, 
//                                        const Eigen::VectorXd& control) const override 
//     {
//         return Eigen::MatrixXd::Zero(1, control.size());
//     }

//     double computeViolation(const Eigen::VectorXd& state, 
//                             const Eigen::VectorXd& control) const override 
//     {
//         Eigen::VectorXd g = evaluate(state, control);
//         return computeViolationFromValue(g);
//     }

//     double computeViolationFromValue(const Eigen::VectorXd& g) const override {
//         return std::max(0.0, g(0));
//     }
// private:
//     Eigen::MatrixXd Q_;
//     Eigen::VectorXd q_;
//     double r_;
//     double scale_factor_;
// };
} // namespace cddp

#endif // CDDP_CONSTRAINT_HPP
