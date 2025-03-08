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
#include <iostream>

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
        virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control) const = 0;

        // Get the Jacobian of the constraint w.r.t the control: dg/du
        virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
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

    private:
        std::string name_; // Name of the constraint
    };

    //------------------------------------------------------------------------------

    class ControlBoxConstraint : public Constraint
    {
    public:
        ControlBoxConstraint(const Eigen::VectorXd &lower_bound,
                             const Eigen::VectorXd &upper_bound)
            : Constraint("ControlBoxConstraint"),
              lower_bound_(lower_bound),
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

        Eigen::VectorXd getLowerBound() const override
        {
            return lower_bound_;
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return upper_bound_;
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &control) const override
        {
            return Eigen::MatrixXd::Zero(control.size(), control.size());
        }

        Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
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

    private:
        Eigen::VectorXd lower_bound_;
        Eigen::VectorXd upper_bound_;
    };

    class StateBoxConstraint : public Constraint
    {
    public:
        StateBoxConstraint(const Eigen::VectorXd &lower_bound,
                           const Eigen::VectorXd &upper_bound)
            : Constraint("StateBoxConstraint"),
              lower_bound_(lower_bound),
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

        Eigen::VectorXd getLowerBound() const override
        {
            return lower_bound_;
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return upper_bound_;
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &control) const override
        {
            return Eigen::MatrixXd::Identity(state.size(), state.size());
        }

        Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
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

    private:
        Eigen::VectorXd lower_bound_;
        Eigen::VectorXd upper_bound_;
    };

    class LinearConstraint : public Constraint
    {
    public:
        LinearConstraint(const Eigen::MatrixXd &A,
                         const Eigen::VectorXd &b,
                         double scale_factor = 1.0)
            : Constraint("LinearConstraint"),
              A_(A),
              b_(b),
              scale_factor_(scale_factor) {}

        int getDualDim() const override
        {
            return b_.size();
        }

        Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd &control) const override
        {
            return A_ * state;
        }

        Eigen::VectorXd getLowerBound() const override
        {
            return Eigen::VectorXd::Constant(A_.rows(), -std::numeric_limits<double>::infinity());
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return b_;
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &control) const override
        {
            return A_;
        }

        Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
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
            : Constraint("ControlConstraint"),
              scale_factor_(scale_factor)
        {
            // Rescale the upper bound for this constraint class
            upper_bound_.resize(2 * upper_bound.size());
            if (lower_bound.size() == 0)
            {
                upper_bound_.head(upper_bound.size()) = upper_bound * scale_factor_;
                upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
            } else
            {
                upper_bound_.head(upper_bound.size()) = -lower_bound * scale_factor_;
                upper_bound_.tail(upper_bound.size()) = upper_bound * scale_factor_;
            }
            dim_ = 2 * upper_bound.size();
        }

        int getDualDim() const override
        {
            return dim_;
        }

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
            return Eigen::VectorXd::Constant(upper_bound_.size(), -std::numeric_limits<double>::infinity());
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return upper_bound_;
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                         const Eigen::VectorXd &control) const override
        {
            return Eigen::MatrixXd::Zero(dim_, state.size());
        }

        Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                           const Eigen::VectorXd &control) const override
        {
            Eigen::MatrixXd jac(2 * control.size(), control.size());
            jac.topLeftCorner(control.size(), control.size()) = -Eigen::MatrixXd::Identity(control.size(), control.size());
            jac.bottomRightCorner(control.size(), control.size()) = Eigen::MatrixXd::Identity(control.size(), control.size());
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

    private:
        int dim_;
        Eigen::VectorXd upper_bound_;
        double scale_factor_;
    };

    class BallConstraint : public Constraint
    {
    public:
        BallConstraint(double radius,
                       const Eigen::VectorXd &center,
                       double scale_factor = 1.0)
            : Constraint("BallConstraint"),
              radius_(radius),
              center_(center),
              scale_factor_(scale_factor)
        {
            dim_ = center.size();
        }

        int getDualDim() const override
        {
            return 1;
        }

        Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd &control) const override
        {
            const Eigen::VectorXd &diff = state.head(dim_) - center_;
            return -Eigen::VectorXd::Constant(1, scale_factor_ * diff.squaredNorm());
        }

        Eigen::VectorXd getLowerBound() const override
        {
            return -Eigen::VectorXd::Constant(1, std::numeric_limits<double>::infinity());
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

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
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

        Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                           const Eigen::VectorXd &control) const override
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

    class PoleConstraint : public Constraint
    {
    public:
        /**
         * Constructor.
         *
         * @param center A 3D vector representing the center of the cylinder.
         * @param direction The axis direction as a character: 'x', 'y', or 'z'.
         * @param radius The radius of the cylinder (pole).
         * @param length The total length of the cylinder (pole). The axial bound will be ±(length/2).
         * @param scale_factor A scaling factor applied to the constraint value.
         */
        PoleConstraint(const Eigen::VectorXd &center, char direction,
                       double radius, double length, double scale_factor = 1.0)
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

        Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd & /* control */) const override
        {
            if (state.size() < 3)
            {
                throw std::invalid_argument("State must have at least 3 dimensions.");
            }
            Eigen::Vector3d p = state.head(3);
            Eigen::Vector3d diff = p - center_;
            // Axial distance along the cylinder’s axis.
            double d_axis = diff.dot(axis_);
            // Radial component (projection of diff onto the plane perpendicular to the axis).
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
            return Eigen::VectorXd::Constant(1, -std::numeric_limits<double>::infinity());
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return Eigen::VectorXd::Zero(1);
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
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
                    d_signed_distance_dp = (dx_pos * d_drad_dp + dy_pos * d_dypos_dp) / signed_distance;
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
            // Construct the Jacobian: place derivative for the first three state elements.
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, state.size());
            J.block(0, 0, 1, 3) = -scale_factor_ * d_signed_distance_dp.transpose();
            return J;
        }

        // The constraint does not depend on the control input.
        Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd & /* state */,
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

    private:
        Eigen::Vector3d center_; // Center of the cylinder.
        Eigen::Vector3d axis_;   // Unit vector for the cylinder's axis.
        double radius_;          // Cylinder radius.
        double length_;          // Total length of the cylinder.
        double half_length_;     // Precomputed half-length.
        double scale_factor_;    // Scaling factor for the constraint value.
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
