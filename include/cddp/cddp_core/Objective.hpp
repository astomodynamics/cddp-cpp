#ifndef CDDP_OBJECTIVE_HPP
#define CDDP_OBJECTIVE_HPP

#include <Eigen/Dense>
#include <vector>

// Base class for cost functions
class Objective {
public:
    
    virtual double calculateRunningCost(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;
    virtual std::vector<Eigen::VectorXd> calculateRunningCostGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;
    virtual std::vector<Eigen::MatrixXd> calculateRunningCostHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0; 
    virtual double calculateFinalCost(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::VectorXd calculateFinalCostGradient(const Eigen::VectorXd& state) const = 0;
    virtual Eigen::MatrixXd calculateFinalCostHessian(const Eigen::VectorXd& state) const = 0;
    virtual double calculateCost(const Eigen::MatrixXd& state, const Eigen::MatrixXd& control) const = 0;
};

// Example: Cost Function
class QuadraticCost : public Objective{
private:
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Qf_;
    Eigen::VectorXd goal_state_;

public:
    QuadraticCost(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd &Qf, Eigen::VectorXd &goal_state) : Q_(Q), R_(R), Qf_(Qf), goal_state_(goal_state) {}

    double calculateCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) const override {
        double cost = 0.0;
        for (int i = 0; i < U.cols(); i++) {
            cost += calculateRunningCost(X.col(i), U.col(i));
        }
        cost += calculateFinalCost(X.col(X.cols() - 1));
        return cost;
    }

    double calculateRunningCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        return ((x - goal_state_).transpose() * Q_ * (x - goal_state_))[0]+ (u.transpose() * R_ * u)[0];
    }

    std::vector<Eigen::VectorXd> calculateRunningCostGradient(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        std::vector<Eigen::VectorXd> gradient;
        gradient.push_back(2 * Q_ * x);
        gradient.push_back(2 * R_ * u);
        return gradient;
    }

    std::vector<Eigen::MatrixXd> calculateRunningCostHessian(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        std::vector<Eigen::MatrixXd> hessian;
        hessian.push_back(2 * Q_);
        hessian.push_back(2 * R_);
        return hessian;
    }

    double calculateFinalCost(const Eigen::VectorXd& x) const override {
        return ((x - goal_state_).transpose() * Qf_ * (x - goal_state_))[0];
    }

    Eigen::VectorXd calculateFinalCostGradient(const Eigen::VectorXd& x) const override {
        return 2 * Qf_ * (x - goal_state_);
    }

    Eigen::MatrixXd calculateFinalCostHessian(const Eigen::VectorXd& x) const override {
        return 2 * Qf_;
    }
};

#endif  // CDDP_OBJECTIVE_HPP