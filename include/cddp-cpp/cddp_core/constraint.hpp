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
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath> // For std::acos, std::sqrt, std::max, M_PI
#include <iostream>
#include <limits>
#include <string>
#include <tuple>

namespace cddp
{

  class Constraint
  {
  public:
    // Constructor
    Constraint(const std::string &name) : name_(name) {}

    // Get the name of the constraint
    const std::string &getName() const { return name_; }

    virtual int getDualDim() const { return 0; }

    // Evaluate the constraint function: g(x, u)
    virtual Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control) const = 0;

    // Get the lower bound of the constraint
    virtual Eigen::VectorXd getLowerBound() const = 0;

    // Get the upper bound of the constraint
    virtual Eigen::VectorXd getUpperBound() const = 0;

    // Get the Jacobian of the constraint w.r.t the state: dg/dx
    virtual Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const = 0;

    // Get the Jacobian of the constraint w.r.t the control: dg/du
    virtual Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const = 0;

    // Utility: Get both Jacobians: dg/dx, dg/du
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
    getJacobians(const Eigen::VectorXd &state,
                 const Eigen::VectorXd &control) const
    {
      return {getStateJacobian(state, control),
              getControlJacobian(state, control)};
    }

    // Compute how far the constraint is violated
    virtual double computeViolation(const Eigen::VectorXd &state,
                                    const Eigen::VectorXd &control) const = 0;

    // Given g(x,u), compute violation from that vector
    virtual double computeViolationFromValue(const Eigen::VectorXd &g) const = 0;

    // Used for constraints with a center (e.g., ball constraint)
    virtual Eigen::VectorXd getCenter() const
    {
      throw std::logic_error("This constraint type does not have a center.");
    }

    // Hessian of the constraint w.r.t the state: d^2g/dx^2
    // Returns a vector of matrices, one for each output dimension of g(x,u)
    virtual std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const
    {
      throw std::logic_error(
          "getStateHessian not implemented for this constraint type.");
    }

    // Hessian of the constraint w.r.t the control: d^2g/du^2
    virtual std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const
    {
      throw std::logic_error(
          "getControlHessian not implemented for this constraint type.");
    }

    // Mixed Hessian of the constraint w.r.t state and control: d^2g/dudx
    virtual std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const
    {
      throw std::logic_error(
          "getCrossHessian not implemented for this constraint type.");
    }

    // Utility: Get all Hessians: d^2g/dx^2, d^2g/du^2, d^2g/dudx
    virtual std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>,
                       std::vector<Eigen::MatrixXd>>
    getHessians(const Eigen::VectorXd &state,
                const Eigen::VectorXd &control) const
    {
      return {getStateHessian(state, control), getControlHessian(state, control),
              getCrossHessian(state, control)};
    }

  private:
    std::string name_; // Name of the constraint
  };

  //------------------------------------------------------------------------------

  class ControlBoxConstraint : public Constraint
  {
  public:
    ControlBoxConstraint(const Eigen::VectorXd &lower_bound,
                         const Eigen::VectorXd &upper_bound)
        : Constraint("ControlBoxConstraint"), lower_bound_(lower_bound),
          upper_bound_(upper_bound) {}

    int getDualDim() const override
    {
      return lower_bound_.size() + upper_bound_.size();
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control) const override
    {
      return control;
    }

    Eigen::VectorXd getLowerBound() const override { return lower_bound_; }

    Eigen::VectorXd getUpperBound() const override { return upper_bound_; }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(control.size(), control.size());
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Identity(control.size(), control.size());
    }

    Eigen::VectorXd clamp(const Eigen::VectorXd &control) const
    {
      return control.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      // Sum of amounts by which g is above upper_bound or below lower_bound
      return (g - upper_bound_).cwiseMax(0.0).sum() +
             (lower_bound_ - g).cwiseMax(0.0).sum();
    }

    // Hessians for ControlBoxConstraint are zero
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), control.size())};
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), control.size())};
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), control.size())};
    }

  private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
  };

  class StateBoxConstraint : public Constraint
  {
  public:
    StateBoxConstraint(const Eigen::VectorXd &lower_bound,
                       const Eigen::VectorXd &upper_bound)
        : Constraint("StateBoxConstraint"), lower_bound_(lower_bound),
          upper_bound_(upper_bound) {}

    int getDualDim() const override
    {
      return lower_bound_.size() + upper_bound_.size();
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control) const override
    {
      return state;
    }

    Eigen::VectorXd getLowerBound() const override { return lower_bound_; }

    Eigen::VectorXd getUpperBound() const override { return upper_bound_; }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Identity(state.size(), state.size());
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(state.size(), control.size());
    }

    Eigen::VectorXd clamp(const Eigen::VectorXd &state) const
    {
      return state.cwiseMax(lower_bound_).cwiseMin(upper_bound_);
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      // Same logic as ControlBoxConstraint but for the state
      return (g - upper_bound_).cwiseMax(0.0).sum() +
             (lower_bound_ - g).cwiseMax(0.0).sum();
    }

    // Hessians for StateBoxConstraint are zero
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(state.size(), state.size())};
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(state.size(), control.size())};
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), state.size())};
    }

  private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;
  };

  class LinearConstraint : public Constraint
  {
  public:
    LinearConstraint(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                     double scale_factor = 1.0)
        : Constraint("LinearConstraint"), A_(A), b_(b),
          scale_factor_(scale_factor) {}

    int getDualDim() const override { return b_.size(); }

    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control) const override
    {
      return A_ * state;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return Eigen::VectorXd::Constant(A_.rows(),
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override { return b_; }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const override
    {
      return A_;
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(A_.rows(), control.size());
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      return std::max(0.0, (b_ - g).maxCoeff());
    }

    // Hessians for LinearConstraint are zero
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hxx_list;
      for (int i = 0; i < A_.rows(); ++i)
      {
        Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
      }
      return Hxx_list;
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Huu_list;
      for (int i = 0; i < A_.rows(); ++i)
      {
        Huu_list.push_back(Eigen::MatrixXd::Zero(control.size(), control.size()));
      }
      return Huu_list;
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hux_list;
      for (int i = 0; i < A_.rows(); ++i)
      {
        Hux_list.push_back(Eigen::MatrixXd::Zero(control.size(), state.size()));
      }
      return Hux_list;
    }

  private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
    double scale_factor_;
  };

  class ControlConstraint : public Constraint
  {
  public:
    ControlConstraint(const Eigen::VectorXd &upper_bound,
                      const Eigen::VectorXd &lower_bound = Eigen::VectorXd(),
                      double scale_factor = 1.0)
        : Constraint("ControlConstraint"), scale_factor_(scale_factor)
    {
      // Rescale the upper bound for this constraint class
      upper_bound_.resize(2 * upper_bound.size());
      if (lower_bound.size() == 0)
      {
        upper_bound_.head(upper_bound.size()) = upper_bound * scale_factor_;
        upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
      }
      else
      {
        upper_bound_.head(upper_bound.size()) = -lower_bound * scale_factor_;
        upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
      }
      dim_ = 2 * upper_bound.size();
    }

    int getDualDim() const override { return dim_; }

    Eigen::VectorXd evaluate(const Eigen::VectorXd & /**/,
                             const Eigen::VectorXd &control) const override
    {
      // return [-control; control];
      Eigen::VectorXd g(2 * control.size());
      g.head(control.size()) = -control;
      g.tail(control.size()) = control;
      return g * scale_factor_;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return Eigen::VectorXd::Constant(upper_bound_.size(),
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override { return upper_bound_; }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(dim_, state.size());
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd jac(2 * control.size(), control.size());
      jac.topLeftCorner(control.size(), control.size()) =
          -Eigen::MatrixXd::Identity(control.size(), control.size());
      jac.bottomRightCorner(control.size(), control.size()) =
          Eigen::MatrixXd::Identity(control.size(), control.size());
      return jac;
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control) - upper_bound_;
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      return (g - upper_bound_).cwiseMax(0.0).sum();
    }

    // Hessians for ControlConstraint are zero
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hxx_list;
      for (int i = 0; i < dim_; ++i)
      { // dim_ is 2 * control.size()
        Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
      }
      return Hxx_list;
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Huu_list;
      for (int i = 0; i < dim_; ++i)
      {
        Huu_list.push_back(Eigen::MatrixXd::Zero(control.size(), control.size()));
      }
      return Huu_list;
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hux_list;
      for (int i = 0; i < dim_; ++i)
      {
        Hux_list.push_back(Eigen::MatrixXd::Zero(control.size(), state.size()));
      }
      return Hux_list;
    }

  private:
    int dim_;
    Eigen::VectorXd upper_bound_;
    double scale_factor_;
  };

  class StateConstraint : public Constraint
  {
  public:
    StateConstraint(const Eigen::VectorXd &upper_bound,
                    const Eigen::VectorXd &lower_bound = Eigen::VectorXd(),
                    double scale_factor = 1.0)
        : Constraint("StateConstraint"), scale_factor_(scale_factor)
    {
      // Rescale the upper bound for this constraint class
      upper_bound_.resize(2 * upper_bound.size());
      if (lower_bound.size() == 0)
      {
        upper_bound_.head(upper_bound.size()) = upper_bound * scale_factor_;
        upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
      }
      else
      {
        upper_bound_.head(upper_bound.size()) = -lower_bound * scale_factor_;
        upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
      }
      dim_ = 2 * upper_bound.size();
    }

    int getDualDim() const override { return dim_; }

    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd & /**/) const override
    {
      // return [-control; control];
      Eigen::VectorXd g(2 * state.size());
      g.head(state.size()) = -state;
      g.tail(state.size()) = state;
      return g * scale_factor_;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return Eigen::VectorXd::Constant(upper_bound_.size(),
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override { return upper_bound_; }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(dim_, state.size());
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd jac(2 * state.size(), state.size());
      jac.topLeftCorner(state.size(), state.size()) =
          -Eigen::MatrixXd::Identity(state.size(), state.size());
      jac.bottomRightCorner(state.size(), state.size()) =
          Eigen::MatrixXd::Identity(state.size(), state.size());
      return jac;
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control) - upper_bound_;
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      return (g - upper_bound_).cwiseMax(0.0).sum();
    }

    // Hessians for ControlConstraint are zero
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hxx_list;
      for (int i = 0; i < dim_; ++i)
      { // dim_ is 2 * control.size()
        Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
      }
      return Hxx_list;
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Huu_list;
      for (int i = 0; i < dim_; ++i)
      {
        Huu_list.push_back(Eigen::MatrixXd::Zero(control.size(), control.size()));
      }
      return Huu_list;
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hux_list;
      for (int i = 0; i < dim_; ++i)
      {
        Hux_list.push_back(Eigen::MatrixXd::Zero(control.size(), state.size()));
      }
      return Hux_list;
    }

  private:
    int dim_;
    Eigen::VectorXd upper_bound_;
    double scale_factor_;
  };

  class BallConstraint : public Constraint
  {
  public:
    BallConstraint(double radius, const Eigen::VectorXd &center,
                   double scale_factor = 1.0)
        : Constraint("BallConstraint"), radius_(radius), center_(center),
          scale_factor_(scale_factor)
    {
      dim_ = center.size();
    }

    int getDualDim() const override { return 1; }

    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control) const override
    {
      const Eigen::VectorXd &diff = state.head(dim_) - center_;
      return -Eigen::VectorXd::Constant(1, scale_factor_ * diff.squaredNorm());
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return -Eigen::VectorXd::Constant(1,
                                        std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override
    {
      return -Eigen::VectorXd::Constant(1, radius_ * radius_) * scale_factor_;
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      double val = g(0);
      double lb = getLowerBound()(0);
      return std::max(0.0, val - lb);
    }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd &control) const override
    {
      const Eigen::VectorXd &diff = state.head(dim_) - center_;
      Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(1, state.size());

      for (int i = 0; i < dim_; ++i)
      {
        jac(0, i) = -2.0 * scale_factor_ * diff(i);
      }

      return jac;
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd &state,
                       const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(1, control.size());
    }

    Eigen::VectorXd getCenter() const { return center_; }
    double getRadius() const { return radius_; }

    // Hessians for BallConstraint
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd Hxx = Eigen::MatrixXd::Zero(state.size(), state.size());
      Hxx.topLeftCorner(dim_, dim_) =
          -2.0 * scale_factor_ * Eigen::MatrixXd::Identity(dim_, dim_);
      return {Hxx};
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), control.size())};
    }

    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), state.size())};
    }

  private:
    double radius_;
    Eigen::VectorXd center_;
    int dim_;
    double scale_factor_;
  };

  class PoleConstraint : public Constraint
  {
  public:
    /**
     * Constructor.
     *
     * @param center A 3D vector representing the center of the cylinder.
     * @param direction The axis direction as a character: 'x', 'y', or 'z'.
     * @param radius The radius of the cylinder (pole).
     * @param length The total length of the cylinder (pole). The axial bound will
     * be ±(length/2).
     * @param scale_factor A scaling factor applied to the constraint value.
     */
    PoleConstraint(const Eigen::VectorXd &center, char direction, double radius,
                   double length, double scale_factor = 1.0)
        : Constraint("PoleConstraint"), radius_(radius), length_(length),
          scale_factor_(scale_factor)
    {
      if (center.size() != 3)
      {
        throw std::invalid_argument("Center must be a 3-dimensional vector.");
      }
      center_ = center;
      half_length_ = length_ / 2.0;
      // Set the axis unit vector based on the provided direction.
      if (direction == 'x' || direction == 'X')
      {
        axis_ = Eigen::Vector3d(1.0, 0.0, 0.0);
      }
      else if (direction == 'y' || direction == 'Y')
      {
        axis_ = Eigen::Vector3d(0.0, 1.0, 0.0);
      }
      else if (direction == 'z' || direction == 'Z')
      {
        axis_ = Eigen::Vector3d(0.0, 0.0, 1.0);
      }
      else
      {
        throw std::invalid_argument("Direction must be 'x', 'y', or 'z'.");
      }
    }

    int getDualDim() const override { return 1; }

    Eigen::VectorXd
    evaluate(const Eigen::VectorXd &state,
             const Eigen::VectorXd & /* control */) const override
    {
      if (state.size() < 3)
      {
        throw std::invalid_argument("State must have at least 3 dimensions.");
      }
      Eigen::Vector3d p = state.head(3);
      Eigen::Vector3d diff = p - center_;
      // Axial distance along the cylinder's axis.
      double d_axis = diff.dot(axis_);
      // Radial component (projection of diff onto the plane perpendicular to the
      // axis).
      Eigen::Vector3d radial_vec = diff - d_axis * axis_;
      double d_rad = radial_vec.norm();

      double dx = d_rad - radius_;
      double dy = std::abs(d_axis) - half_length_;
      double signed_distance = 0.0;

      if (dx > 0.0 || dy > 0.0)
      {
        double dx_pos = (dx > 0.0) ? dx : 0.0;
        double dy_pos = (dy > 0.0) ? dy : 0.0;
        signed_distance = std::sqrt(dx_pos * dx_pos + dy_pos * dy_pos);
      }
      else
      {
        signed_distance = std::max(dx, dy); // Both are non-positive.
      }

      Eigen::VectorXd g(1);
      g(0) = -scale_factor_ * signed_distance;
      return g;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return Eigen::VectorXd::Constant(1,
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override
    {
      return Eigen::VectorXd::Zero(1);
    }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd & /* control */) const override
    {
      if (state.size() < 3)
      {
        throw std::invalid_argument("State must have at least 3 dimensions.");
      }
      Eigen::Vector3d p = state.head(3);
      Eigen::Vector3d diff = p - center_;
      double d_axis = diff.dot(axis_);
      Eigen::Vector3d radial_vec = diff - d_axis * axis_;
      double d_rad = radial_vec.norm();

      double dx = d_rad - radius_;
      double dy = std::abs(d_axis) - half_length_;
      double signed_distance = 0.0;
      Eigen::Vector3d d_signed_distance_dp = Eigen::Vector3d::Zero();

      const double eps = 1e-8;
      if (dx > 0.0 || dy > 0.0)
      {
        double dx_pos = (dx > 0.0) ? dx : 0.0;
        double dy_pos = (dy > 0.0) ? dy : 0.0;
        double denom = std::sqrt(dx_pos * dx_pos + dy_pos * dy_pos);
        signed_distance = denom;
        // Compute derivative of d_rad.
        Eigen::Vector3d d_drad_dp;
        if (d_rad > eps)
          d_drad_dp = (radial_vec / d_rad).eval();
        else
          d_drad_dp = Eigen::Vector3d::Zero();

        // Derivative of |d_axis| w.r.t. p.
        double sign_axis = (d_axis >= 0.0) ? 1.0 : -1.0;
        Eigen::Vector3d d_daxis_dp = axis_;

        // Use if/else instead of ternary operator for d_dypos_dp.
        Eigen::Vector3d d_dypos_dp;
        if (dy > 0.0)
          d_dypos_dp = (sign_axis * d_daxis_dp).eval();
        else
          d_dypos_dp = Eigen::Vector3d::Zero();

        // Combine active terms.
        if (signed_distance > eps)
        {
          d_signed_distance_dp =
              (dx_pos * d_drad_dp + dy_pos * d_dypos_dp) / signed_distance;
        }
      }
      else
      {
        if (dx >= dy)
        {
          signed_distance = dx;
          if (d_rad > eps)
            d_signed_distance_dp = (radial_vec / d_rad).eval();
          else
            d_signed_distance_dp = Eigen::Vector3d::Zero();
        }
        else
        {
          signed_distance = dy;
          double sign_axis = (d_axis >= 0.0) ? 1.0 : -1.0;
          d_signed_distance_dp = (sign_axis * axis_).eval();
        }
      }
      // Construct the Jacobian: place derivative for the first three state
      // elements.
      Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, state.size());
      J.block(0, 0, 1, 3) = -scale_factor_ * d_signed_distance_dp.transpose();
      return J;
    }

    // The constraint does not depend on the control input.
    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd & /* state */,
                       const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(1, control.size());
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      // Violation occurs if g(0) > 0.
      return std::max(0.0, g(0));
    }

    // Hessians for PoleConstraint
    // TODO: Implement actual Hessians for PoleConstraint
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      throw std::logic_error(
          "getStateHessian for PoleConstraint not yet implemented.");
      Eigen::MatrixXd Hxx = Eigen::MatrixXd::Zero(state.size(), state.size());
      return {Hxx};
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      throw std::logic_error(
          "getControlHessian for PoleConstraint not yet implemented.");
      return {Eigen::MatrixXd::Zero(control.size(), control.size())};
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      throw std::logic_error(
          "getCrossHessian for PoleConstraint not yet implemented.");
      return {Eigen::MatrixXd::Zero(control.size(), state.size())};
    }

  private:
    Eigen::Vector3d center_; // Center of the cylinder.
    Eigen::Vector3d axis_;   // Unit vector for the cylinder's axis.
    double radius_;          // Cylinder radius.
    double length_;          // Total length of the cylinder.
    double half_length_;     // Precomputed half-length.
    double scale_factor_;    // Scaling factor for the constraint value.
  };

  // Second-order cone constraint: Ensures the state stays inside a cone
  class SecondOrderConeConstraint : public Constraint
  {
  public:
    SecondOrderConeConstraint(
        const Eigen::Vector3d &cone_origin,
        const Eigen::Vector3d
            &opening_direction, // Changed from cone_axis: Represents the
                                // direction the cone opens towards
        double cone_angle_fov,  // In radians [0, PI]
        double regularization_epsilon =
            1e-6, // Small positive number for regularization
        const std::string &name = "SecondOrderConeConstraint")
        : Constraint(name), p_o_(cone_origin),
          axis_(opening_direction.normalized()), // Ensure unit vector, storing
                                                 // the opening direction
          cos_fov_(std::cos(cone_angle_fov)), epsilon_(regularization_epsilon)
    {
      if (cone_angle_fov < 0 || cone_angle_fov > M_PI)
      {
        throw std::invalid_argument(
            "SecondOrderConeConstraint: Cone angle must be between 0 and PI.");
      }
      if (regularization_epsilon <= 0)
      {
        throw std::invalid_argument("SecondOrderConeConstraint: Regularization "
                                    "epsilon must be positive.");
      }
      // Optional: Warn if the provided opening direction was not unit
      if (std::abs(opening_direction.norm() - 1.0) > 1e-6 &&
          opening_direction.norm() != 0.0)
      {
        std::cerr << "Warning: SecondOrderConeConstraint provided "
                     "opening_direction was not a unit vector. Normalizing."
                  << std::endl;
      }
      else if (opening_direction.norm() == 0.0)
      {
        throw std::invalid_argument("SecondOrderConeConstraint: Opening "
                                    "direction cannot be zero vector.");
      }
    }

    int getDualDim() const override
    {
      return 1; // Scalar inequality constraint g(x) <= 0
    }

    // Evaluate g(x) = cos(theta_fov) * sqrt(||p_s - p_o||^2 + epsilon) - (p_s -
    // p_o) . axis
    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd & /*control*/) const override
    {
      if (state.size() < 3)
      {
        throw std::invalid_argument(
            "SecondOrderConeConstraint: State dimension must be at least 3.");
      }
      Eigen::Vector3d p_s = state.head(3);
      Eigen::Vector3d v = p_s - p_o_; // Vector from origin to state point
      double v_squared = v.squaredNorm();
      double reg_norm =
          std::sqrt(v_squared + epsilon_); // Regularized norm ||p_s - p_o||

      double dot_prod = v.dot(axis_); // (p_s - p_o) . axis
      double g_val = reg_norm * cos_fov_ - dot_prod;

      Eigen::VectorXd result(1);
      result(0) = g_val;
      return result;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      // g(x) <= 0
      return Eigen::VectorXd::Constant(1,
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override
    {
      // g(x) <= 0
      return Eigen::VectorXd::Zero(1);
    }

    // Calculate Jacobian dg/dx = dg/dp_s * dp_s/dx
    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd & /*control*/) const override
    {
      if (state.size() < 3)
      {
        throw std::invalid_argument(
            "SecondOrderConeConstraint: State dimension must be at least 3 for "
            "Jacobian calculation.");
      }
      Eigen::Vector3d p_s = state.head(3);
      Eigen::Vector3d v = p_s - p_o_; // Vector from origin to state point
      double v_squared = v.squaredNorm();
      double reg_norm =
          std::sqrt(v_squared + epsilon_); // Regularized norm ||p_s - p_o||

      // Jacobian dg/dx = dg/dp_s * dp_s/dx
      // Assuming p_s = state.head(3), then dp_s/dx = [I_3x3, 0]
      Eigen::MatrixXd dp_s_dx = Eigen::MatrixXd::Zero(3, state.size());
      dp_s_dx.leftCols(3) = Eigen::Matrix3d::Identity();

      // Calculate dg/dp_s:
      // g = cos(fov) * sqrt(||p_s - p_o||^2 + epsilon) - (p_s - p_o) . axis
      // dg/dp_s = cos(fov) * d/dp_s sqrt(||p_s - p_o||^2 + epsilon) - d/dp_s (p_s
      // . axis) dg/dp_s = cos(fov) * ( (p_s - p_o)^T / sqrt(||p_s - p_o||^2 +
      // epsilon) ) - axis^T dg/dp_s = cos(fov) * (v^T / reg_norm) - axis^T
      Eigen::RowVector3d dg_dps;
      if (reg_norm > 1e-9)
      { // Avoid division by zero if p_s is very close to p_o
        dg_dps = cos_fov_ * (v.transpose() / reg_norm) - axis_.transpose();
      }
      else
      {
        dg_dps = -axis_.transpose(); // Or handle as appropriate, maybe zero?
      }

      // Chain rule: dg/dx = dg/dp_s * dp_s/dx
      Eigen::MatrixXd jacobian = dg_dps * dp_s_dx;
      return jacobian;
    }

    // The constraint does not depend on the control input.
    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd & /* state */,
                       const Eigen::VectorXd &control) const override
    {
      return Eigen::MatrixXd::Zero(1, control.size());
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      // Violation occurs when g > upper_bound (which is 0)
      // Ensure g has at least one element
      if (g.size() < 1)
      {
        throw std::runtime_error("SecondOrderConeConstraint: Input vector g is "
                                 "empty in computeViolationFromValue.");
      }
      return std::max(0.0, g(0));
    }

    // Hessians for SecondOrderConeConstraint
    // TODO: Implement actual Hessians for SecondOrderConeConstraint
    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      throw std::logic_error(
          "getStateHessian for SecondOrderConeConstraint not yet implemented.");
      Eigen::MatrixXd Hxx = Eigen::MatrixXd::Zero(state.size(), state.size());
      return {Hxx};
    }
    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control) const override
    {
      throw std::logic_error(
          "getControlHessian for SecondOrderConeConstraint not yet implemented.");
      return {Eigen::MatrixXd::Zero(control.size(), control.size())};
    }
    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      throw std::logic_error(
          "getCrossHessian for SecondOrderConeConstraint not yet implemented.");
      return {Eigen::MatrixXd::Zero(control.size(), state.size())};
    }

  private:
    Eigen::Vector3d p_o_; // Cone origin position
    Eigen::Vector3d
        axis_;       // Cone axis (unit vector, represents OPENING direction)
    double cos_fov_; // Cosine of the cone field-of-view half-angle
    double epsilon_; // Regularization parameter for differentiability
  };

  class ThrustMagnitudeConstraint : public Constraint
  {
  public:
    ThrustMagnitudeConstraint(double min_thrust_norm, double max_thrust_norm,
                              double epsilon = 1e-6)
        : Constraint("ThrustMagnitudeConstraint"),
          min_thrust_norm_(min_thrust_norm), max_thrust_norm_(max_thrust_norm),
          epsilon_(epsilon)
    {
      if (min_thrust_norm_ < 0.0)
      {
        throw std::invalid_argument(
            "ThrustMagnitudeConstraint: min_thrust_norm must be non-negative.");
      }
      if (max_thrust_norm_ < min_thrust_norm_)
      {
        throw std::invalid_argument(
            "ThrustMagnitudeConstraint: max_thrust_norm must be greater than or "
            "equal to min_thrust_norm.");
      }
      if (epsilon_ <= 0.0)
      {
        throw std::invalid_argument(
            "ThrustMagnitudeConstraint: epsilon must be positive.");
      }
    }

    int getDualDim() const override { return 2; }

    Eigen::VectorXd evaluate(const Eigen::VectorXd & /*state*/,
                             const Eigen::VectorXd &control) const override
    {
      double u_norm = control.norm();
      Eigen::VectorXd g(2);
      g(0) = min_thrust_norm_ - u_norm;
      g(1) = u_norm - max_thrust_norm_;
      return g;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return Eigen::VectorXd::Constant(2,
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override
    {
      return Eigen::VectorXd::Zero(2);
    }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd & /*control*/) const override
    {
      return Eigen::MatrixXd::Zero(2, state.size());
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd & /*state*/,
                       const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(2, control.size());
      double u_sq_norm = control.squaredNorm();
      double u_reg_norm = std::sqrt(u_sq_norm + epsilon_);

      if (u_reg_norm < epsilon_)
      {                // safeguard,
        jac.setZero(); // Already zero, but explicit.
      }
      else
      {
        Eigen::RowVectorXd du_norm_reg_du = control.transpose() / u_reg_norm;
        jac.row(0) = -du_norm_reg_du;
        jac.row(1) = du_norm_reg_du;
      }
      return jac;
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      if (g.size() != 2)
      {
        throw std::runtime_error("ThrustMagnitudeConstraint: Input vector g must "
                                 "have size 2 in computeViolationFromValue.");
      }
      return std::max(0.0, g(0)) + std::max(0.0, g(1));
    }

    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd & /*control*/) const override
    {
      std::vector<Eigen::MatrixXd> Hxx_list;
      Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
      Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
      return Hxx_list;
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd & /*state*/,
                      const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd H_norm_reg =
          Eigen::MatrixXd::Zero(control.size(), control.size());
      double u_sq_norm = control.squaredNorm();
      double term_in_sqrt = u_sq_norm + epsilon_;

      double denominator = std::pow(term_in_sqrt, 1.5);

      if (denominator >
          std::numeric_limits<double>::
              min()) // Avoid division by zero if denominator underflows, though
                     // epsilon should prevent this for reasonable values
      {
        H_norm_reg = (term_in_sqrt * Eigen::MatrixXd::Identity(control.size(),
                                                               control.size()) -
                      control * control.transpose()) /
                     denominator;
      }

      return {-H_norm_reg, H_norm_reg};
    }

    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      std::vector<Eigen::MatrixXd> Hux_list;
      Hux_list.push_back(Eigen::MatrixXd::Zero(control.size(), state.size()));
      Hux_list.push_back(Eigen::MatrixXd::Zero(control.size(), state.size()));
      return Hux_list;
    }

  private:
    double min_thrust_norm_;
    double max_thrust_norm_;
    double epsilon_;
  };

  class MaxThrustMagnitudeConstraint : public Constraint
  {
  public:
    MaxThrustMagnitudeConstraint(double max_thrust_norm, double epsilon = 1e-6)
        : Constraint("MaxThrustMagnitudeConstraint"),
          max_thrust_norm_(max_thrust_norm), epsilon_(epsilon)
    {
      if (max_thrust_norm_ < 0.0)
      {
        throw std::invalid_argument("MaxThrustMagnitudeConstraint: "
                                    "max_thrust_norm must be non-negative.");
      }
      if (epsilon_ <= 0.0)
      {
        throw std::invalid_argument(
            "MaxThrustMagnitudeConstraint: epsilon must be positive.");
      }
    }

    int getDualDim() const override { return 1; }

    // Constraint is ||u|| - max_thrust_norm <= 0
    Eigen::VectorXd evaluate(const Eigen::VectorXd & /*state*/,
                             const Eigen::VectorXd &control) const override
    {
      double u_norm = control.norm();
      Eigen::VectorXd g(1);
      g(0) = u_norm - max_thrust_norm_;
      return g;
    }

    Eigen::VectorXd getLowerBound() const override
    {
      return Eigen::VectorXd::Constant(1,
                                       -std::numeric_limits<double>::infinity());
    }

    Eigen::VectorXd getUpperBound() const override
    {
      return Eigen::VectorXd::Zero(1);
    }

    Eigen::MatrixXd
    getStateJacobian(const Eigen::VectorXd &state,
                     const Eigen::VectorXd & /*control*/) const override
    {
      return Eigen::MatrixXd::Zero(1, state.size());
    }

    Eigen::MatrixXd
    getControlJacobian(const Eigen::VectorXd & /*state*/,
                       const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(1, control.size());
      double u_sq_norm = control.squaredNorm();
      double u_reg_norm = std::sqrt(u_sq_norm + epsilon_);

      if (u_reg_norm > std::numeric_limits<double>::min())
      {
        jac.row(0) = control.transpose() / u_reg_norm;
      }
      // else jac remains zero, if u_reg_norm is zero or too small (should be
      // prevented by epsilon > 0)
      return jac;
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control) const override
    {
      Eigen::VectorXd g = evaluate(state, control);
      return computeViolationFromValue(g);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override
    {
      if (g.size() != 1)
      {
        throw std::runtime_error(
            "MaxThrustMagnitudeConstraint: Input vector g must have size 1 in "
            "computeViolationFromValue.");
      }
      return std::max(0.0, g(0)); // Violation if g(0) > 0
    }

    std::vector<Eigen::MatrixXd>
    getStateHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd & /*control*/) const override
    {
      return {Eigen::MatrixXd::Zero(state.size(), state.size())};
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd & /*state*/,
                      const Eigen::VectorXd &control) const override
    {
      Eigen::MatrixXd H_norm_reg =
          Eigen::MatrixXd::Zero(control.size(), control.size());
      double u_sq_norm = control.squaredNorm();
      double term_in_sqrt = u_sq_norm + epsilon_;
      double denominator = std::pow(term_in_sqrt, 1.5);

      if (denominator > std::numeric_limits<double>::min())
      {
        H_norm_reg = (term_in_sqrt * Eigen::MatrixXd::Identity(control.size(),
                                                               control.size()) -
                      control * control.transpose()) /
                     denominator;
      }
      return {H_norm_reg};
    }

    std::vector<Eigen::MatrixXd>
    getCrossHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control) const override
    {
      return {Eigen::MatrixXd::Zero(control.size(), state.size())};
    }

  private:
    double max_thrust_norm_;
    double epsilon_;
  };

} // namespace cddp

#endif // CDDP_CONSTRAINT_HPP
