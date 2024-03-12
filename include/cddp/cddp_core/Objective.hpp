#ifndef CDDP_OBJECTIVE_HPP
#define CDDP_OBJECTIVE_HPP

#include <Eigen/Dense>
#include <vector>

namespace cddp {
// Base class for cost functions
class Objective {
public:
    
    virtual double calculateRunningCost(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> calculateRunningCostGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   calculateRunningCostHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0; 
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
    double timestep_;

public:
    QuadraticCost(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd &Qf, Eigen::VectorXd &goal_state, double timestep) : Q_(Q), R_(R), Qf_(Qf), goal_state_(goal_state), timestep_(timestep) {}

    double calculateCost(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) const override {
        double cost = 0.0;
        for (int i = 0; i < U.cols(); i++) {
            cost += calculateRunningCost(X.col(i), U.col(i));
        }
        cost += calculateFinalCost(X.col(X.cols() - 1));
        return cost;
    }

    double calculateRunningCost(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        double cost = 0.0; // Initialize cost to 0
        cost += 0.5 * ((x - goal_state_).transpose() * Q_ * (x - goal_state_)).value() * timestep_;
        cost += 0.5 * (u.transpose() * R_ * u).value() * timestep_;
        return cost;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>  calculateRunningCostGradient(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>  gradients;
        Eigen::VectorXd l_x = Q_ * (x - goal_state_) * timestep_;
        Eigen::VectorXd l_u = R_ * u * timestep_;
        std::get<0>(gradients) = l_x;
        std::get<1>(gradients) = l_u;
        return gradients;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   calculateRunningCostHessian(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const override {
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>   hessians;
        Eigen::MatrixXd l_xx = Q_ * timestep_;
        Eigen::MatrixXd l_ux = Eigen::MatrixXd::Zero(R_.rows(), Q_.rows());
        Eigen::MatrixXd l_uu = R_ * timestep_;
        std::get<0>(hessians) = l_xx;
        std::get<1>(hessians) = l_ux;
        std::get<2>(hessians) = l_uu;
        return hessians;
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
}  // namespace cddp

#endif  // CDDP_OBJECTIVE_HPP