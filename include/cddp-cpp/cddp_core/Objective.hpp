#ifndef CDDP_OBJECTIVE_HPP
#define CDDP_OBJECTIVE_HPP

#include "Eigen/Dense"
#include <vector>

namespace cddp {
// Base class for cost functions
class Objective {
public:
    
    virtual double calculateRunningCost(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> calculateRunningCostGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   calculateRunningCostHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0; 
    virtual double calculateFinalCost(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::VectorXd calculateFinalCostGradient(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::MatrixXd calculateFinalCostHessian(const Eigen::VectorXd& state) const = 0;
    virtual double calculateCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) const = 0;
};

// Example: Quadratic Cost Function
class QuadraticCost : public Objective{
private:
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Qf_;
    Eigen::VectorXd goal_state_;
    double timestep_;

public:
    QuadraticCost(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd &Qf, Eigen::VectorXd &goal_state, double timestep) : Q_(Q), R_(R), Qf_(Qf), goal_state_(goal_state), timestep_(timestep) {}

    double calculateCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) const override {
        double cost = 0.0;
        for (int i = 0; i < U.size(); i++) {
            cost += calculateRunningCost(X[i], U[i], i);
        }
        cost += calculateFinalCost(X[X.size() - 1]);
        return cost;
    }
    
    double calculateRunningCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
        double cost = 0.0; // Initialize cost to 0
        cost += 0.5 * ((x - goal_state_).transpose() * Q_ * (x - goal_state_)).value() * timestep_;
        cost += 0.5 * (u.transpose() * R_ * u).value() * timestep_;
        return cost;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>  calculateRunningCostGradient(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
        Eigen::VectorXd l_x = Q_ * (x - goal_state_) * timestep_;
        Eigen::VectorXd l_u = R_ * u * timestep_;
        return std::make_tuple(l_x, l_u);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   calculateRunningCostHessian(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
        Eigen::MatrixXd l_xx = Q_ * timestep_;
        Eigen::MatrixXd l_ux = Eigen::MatrixXd::Zero(R_.rows(), Q_.rows());
        Eigen::MatrixXd l_uu = R_ * timestep_;
        return std::make_tuple(l_xx, l_ux, l_uu);
    }

    double calculateFinalCost(const Eigen::VectorXd& x) const override {
        return 0.5 * ((x - goal_state_).transpose() * Qf_ * (x - goal_state_)).value();
    }

    Eigen::VectorXd calculateFinalCostGradient(const Eigen::VectorXd& x) const override {
        return Qf_ * (x - goal_state_);
    }

    Eigen::MatrixXd calculateFinalCostHessian(const Eigen::VectorXd& x) const override {
        return Qf_;
    }
};

// Example: Quadratic Tracking Cost Function
class QuadraticTrackingCost : public Objective {
private:
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Qf_;
    std::vector<Eigen::VectorXd> X_ref_;
    Eigen::VectorXd goal_state_;
    double timestep_;

public:
    QuadraticTrackingCost(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd &Qf, std::vector<Eigen::VectorXd> &X_ref, Eigen::VectorXd &goal_state, double timestep) : Q_(Q), R_(R), Qf_(Qf), X_ref_(X_ref), goal_state_(goal_state), timestep_(timestep) {}

    double calculateCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) const override {
        double cost = 0.0;
        for (int i = 0; i < U.size(); i++) {
            cost += calculateRunningCost(X[i], U[i], i);
        }
        cost += calculateFinalCost(X[X.size() - 1]);
        return cost;
    }

    double calculateRunningCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
        double cost = 0.0; // Initialize cost to 0
        cost += 0.5 * ((x - X_ref_[k]).transpose() * Q_ * (x - X_ref_[k])).value() * timestep_;
        cost += 0.5 * (u.transpose() * R_ * u).value() * timestep_;
        return cost;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>  calculateRunningCostGradient(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
        Eigen::VectorXd l_x = Q_ * (x - X_ref_[k]) * timestep_;
        Eigen::VectorXd l_u = R_ * u * timestep_;
        return std::make_tuple(l_x, l_u);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   calculateRunningCostHessian(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
        Eigen::MatrixXd l_xx = Q_ * timestep_;
        Eigen::MatrixXd l_ux = Eigen::MatrixXd::Zero(R_.rows(), Q_.rows());
        Eigen::MatrixXd l_uu = R_ * timestep_;
        return std::make_tuple(l_xx, l_ux, l_uu);
    }

    double calculateFinalCost(const Eigen::VectorXd& x) const override {
        return 0.5 * ((x - goal_state_).transpose() * Qf_ * (x - goal_state_)).value();
    }

    Eigen::VectorXd calculateFinalCostGradient(const Eigen::VectorXd& x) const override {
        return Qf_ * (x - goal_state_);
    }

    Eigen::MatrixXd calculateFinalCostHessian(const Eigen::VectorXd& x) const override {
        return Qf_;
    }

};


// // Example: Quadratic Tracking Cost Function with Acceleration Cost
// class QuadraticTrackingCostWithAccel : public Objective {
// private:
//     Eigen::MatrixXd Q_;
//     Eigen::MatrixXd R_;
//     Eigen::MatrixXd Ra_;
//     Eigen::MatrixXd Qf_;
//     std::vector<Eigen::VectorXd> X_ref_;
//     Eigen::VectorXd goal_state_;
//     double timestep_;

// public:
//     QuadraticTrackingCostWithAccel(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd &Qf, std::vector<Eigen::VectorXd> &X_ref, Eigen::VectorXd &goal_state, double timestep) : Q_(Q), R_(R), Ra_(Ra), Qf_(Qf), X_ref_(X_ref), goal_state_(goal_state), timestep_(timestep) {}

//     double calculateCost(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) const override {
//         double cost = 0.0;
//         for (int i = 0; i < U.size(); i++) {
//             cost += calculateRunningCost(X[i], U[i], i);
//         }
//         cost += calculateFinalCost(X[X.size() - 1]);

//         for (int i = 0; i < U.size() - 1; i++) {
            
//         }
//         return cost;
//     }

//     double calculateRunningCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
//         double cost = 0.0; // Initialize cost to 0
//         cost += 0.5 * ((x - X_ref_[k]).transpose() * Q_ * (x - X_ref_[k])).value() * timestep_;
//         cost += 0.5 * (u.transpose() * R_ * u).value() * timestep_;
//         return cost;
//     }

//     std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>  calculateRunningCostGradient(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
//         Eigen::VectorXd l_x = Q_ * (x - X_ref_[k]) * timestep_;
//         Eigen::VectorXd l_u = R_ * u * timestep_;
//         return std::make_tuple(l_x, l_u);
//     }

//     std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   calculateRunningCostHessian(const Eigen::VectorXd& x, const Eigen::VectorXd& u, int k) const override {
//         Eigen::MatrixXd l_xx = Q_ * timestep_;
//         Eigen::MatrixXd l_ux = Eigen::MatrixXd::Zero(R_.rows(), Q_.rows());
//         Eigen::MatrixXd l_uu = R_ * timestep_;
//         return std::make_tuple(l_xx, l_ux, l_uu);
//     }

//     double calculateFinalCost(const Eigen::VectorXd& x) const override {
//         return 0.5 * ((x - goal_state_).transpose() * Qf_ * (x - goal_state_)).value();
//     }

//     Eigen::VectorXd calculateFinalCostGradient(const Eigen::VectorXd& x) const override {
//         return Qf_ * (x - goal_state_);
//     }

//     Eigen::MatrixXd calculateFinalCostHessian(const Eigen::VectorXd& x) const override {
//         return Qf_;
//     }

// };

}  // namespace cddp

#endif  // CDDP_OBJECTIVE_HPP