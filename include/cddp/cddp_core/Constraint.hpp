#ifndef CDDP_CONSTRAINT_HPP
#define CDDP_CONSTRAINT_HPP

#include "Eigen/Dense"
#include <vector>

namespace cddp {
// Base class for constraints
class Constraint {
public:
    virtual Eigen::VectorXd calculateConstraint(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0; // Evaluate the constraint 
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> calculateConstraintJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0; // Jacobian of the constraint
};


class ControlBoxConstraint : public Constraint {
private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;

public:
    ControlBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) : lower_bound_(lower_bound), upper_bound_(upper_bound) {}

    Eigen::VectorXd getLowerBound() const { return lower_bound_; }
    Eigen::VectorXd getUpperBound() const { return upper_bound_; }

    Eigen::VectorXd calculateConstraint(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        Eigen::VectorXd constraint = Eigen::VectorXd::Zero(2* u.size());
        for (int i = 0; i < u.size(); i++) {
            constraint[i] = lower_bound_[i] - u[i];
            constraint[i + u.size()] = u[i] - upper_bound_[i];
        }
        return constraint;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> calculateConstraintJacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        Eigen::MatrixXd g_x = Eigen::MatrixXd::Zero(2 * u.size(), x.size());
        Eigen::MatrixXd g_u = Eigen::MatrixXd::Zero(2 * u.size(), u.size());
        for (int i = 0; i < u.size(); i++) {
            g_u(i, i) = -1.0;
            g_u(i + u.size(), i) = 1.0;
        }
        return std::make_tuple(g_x, g_u);
    }
};

// class CircleConstraint : public Constraint {
// private:
//     Eigen::VectorXd center_;
//     double radius_;

// public:
//     CircleConstraint(const Eigen::VectorXd& center, double radius) : center_(center), radius_(radius) {}

//     double calculateConstraint(const Eigen::VectorXd& x) const override {
//         return (x - center_).squaredNorm() - radius_ * radius_;
//     }

//     Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
//         return 2 * (x - center_).transpose();
//     }
// };




// class StateBoxConstraint : public Constraint {
// private:
//     Eigen::VectorXd lower_bound_;
//     Eigen::VectorXd upper_bound_;

// public:
//     StateBoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) : lower_bound_(lower_bound), upper_bound_(upper_bound) {}

//     double calculateConstraint(const Eigen::VectorXd& x) const override {
//         double constraint = 0.0;
//         for (int i = 0; i < x.size(); i++) {
//             if (x[i] < lower_bound_[i]) {
//                 constraint += lower_bound_[i] - x[i];
//             } else if (x[i] > upper_bound_[i]) {
//                 constraint += x[i] - upper_bound_[i];
//             }
//         }
//         return constraint;
//     }

//     Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
//         Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(x.size());
//         for (int i = 0; i < x.size(); i++) {
//             if (x[i] < lower_bound_[i]) {
//                 jacobian[i] = -1.0;
//             } else if (x[i] > upper_bound_[i]) {
//                 jacobian[i] = 1.0;
//             }
//         }
//         return jacobian;
//     }
// };


// class ObstacleConstraint : public Constraint {
// private:
//     std::vector<Eigen::VectorXd> obstacles_;
//     double radius_;

// public:
//     ObstacleConstraint(const std::vector<Eigen::VectorXd>& obstacles, double radius) : obstacles_(obstacles), radius_(radius) {}

//     double calculateConstraint(const Eigen::VectorXd& x) const override {
//         double constraint = 0.0;
//         for (const auto& obstacle : obstacles_) {
//             constraint += (x - obstacle).squaredNorm() - radius_ * radius_;
//         }
//         return constraint;
//     }

//     Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
//         Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(x.size());
//         for (const auto& obstacle : obstacles_) {
//             jacobian += 2 * (x - obstacle).transpose();
//         }
//         return jacobian;
//     }
// };

// class LinearConstraint : public Constraint {
// private:
//     Eigen::MatrixXd A_;
//     Eigen::VectorXd b_;

// public:
//     LinearConstraint(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) : A_(A), b_(b) {}

//     double calculateConstraint(const Eigen::VectorXd& x) const override {
//         return (A_ * x - b_).norm();
//     }

//     Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
//         return 2 * A_.transpose() * (A_ * x - b_);
//     }
// };
} // namespace cddp

#endif  // CDDP_CONSTRAINT_HPP
