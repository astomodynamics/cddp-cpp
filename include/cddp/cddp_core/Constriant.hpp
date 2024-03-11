#ifndef CDDP_CONSTRAINT_HPP
#define CDDP_CONSTRAINT_HPP

#include <Eigen/Dense>
#include <vector>

// Base class for constraints
class Constraint {
public:
    virtual double calculateConstraint(const Eigen::VectorXd& state) const = 0; // Evaluate the constraint 
    virtual Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& state) const = 0; // Jacobian of the constraint
};


class CircleConstraint : public Constraint {
private:
    Eigen::VectorXd center_;
    double radius_;

public:
    CircleConstraint(const Eigen::VectorXd& center, double radius) : center_(center), radius_(radius) {}

    double calculateConstraint(const Eigen::VectorXd& x) const override {
        return (x - center_).squaredNorm() - radius_ * radius_;
    }

    Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
        return 2 * (x - center_).transpose();
    }
};

class BoxConstraint : public Constraint {
private:
    Eigen::VectorXd lower_bound_;
    Eigen::VectorXd upper_bound_;

public:
    BoxConstraint(const Eigen::VectorXd& lower_bound, const Eigen::VectorXd& upper_bound) : lower_bound_(lower_bound), upper_bound_(upper_bound) {}

    double calculateConstraint(const Eigen::VectorXd& x) const override {
        double constraint = 0.0;
        for (int i = 0; i < x.size(); i++) {
            if (x[i] < lower_bound_[i]) {
                constraint += lower_bound_[i] - x[i];
            } else if (x[i] > upper_bound_[i]) {
                constraint += x[i] - upper_bound_[i];
            }
        }
        return constraint;
    }

    Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(x.size());
        for (int i = 0; i < x.size(); i++) {
            if (x[i] < lower_bound_[i]) {
                jacobian[i] = -1.0;
            } else if (x[i] > upper_bound_[i]) {
                jacobian[i] = 1.0;
            }
        }
        return jacobian;
    }
};


class ObstacleConstraint : public Constraint {
private:
    std::vector<Eigen::VectorXd> obstacles_;
    double radius_;

public:
    ObstacleConstraint(const std::vector<Eigen::VectorXd>& obstacles, double radius) : obstacles_(obstacles), radius_(radius) {}

    double calculateConstraint(const Eigen::VectorXd& x) const override {
        double constraint = 0.0;
        for (const auto& obstacle : obstacles_) {
            constraint += (x - obstacle).squaredNorm() - radius_ * radius_;
        }
        return constraint;
    }

    Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd jacobian = Eigen::VectorXd::Zero(x.size());
        for (const auto& obstacle : obstacles_) {
            jacobian += 2 * (x - obstacle).transpose();
        }
        return jacobian;
    }
};

class LinearConstraint : public Constraint {
private:
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;

public:
    LinearConstraint(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) : A_(A), b_(b) {}

    double calculateConstraint(const Eigen::VectorXd& x) const override {
        return (A_ * x - b_).norm();
    }

    Eigen::VectorXd calculateConstraintJacobian(const Eigen::VectorXd& x) const override {
        return 2 * A_.transpose() * (A_ * x - b_);
    }
};


#endif  // CDDP_CONSTRAINT_HPP
