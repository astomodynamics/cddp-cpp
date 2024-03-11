#ifndef CDDP_PENDULUM_HPP
#define CDDP_PENDULUM_HPP

#include <Eigen/Dense>
#include <vector>
#include "DynamicalSystem.hpp"

namespace cddp {

class Pendulum : public cddp::DynamicalSystem {
public:
    int state_size_;
    int control_size_;
    double timestep_;  // Time step
    int integration_type_;

    Pendulum(int state_dim, int control_dim, double timestep, int integration_type) :
        DynamicalSystem(state_dim, control_dim, timestep, integration_type) {
            state_size_ = state_dim;
            control_size_ = control_dim;
            timestep_ = timestep;
            integration_type_ = integration_type;
        }

    Eigen::VectorXd dynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(state_size_);
        state_dot(0) = state(1);
        state_dot(1) = -9.81 * sin(state(0)) + control(0);
        return state_dot;
    }

    std::vector<Eigen::MatrixXd> getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(state_size_, state_size_);
        A(0, 1) = 1;
        A(1, 0) = -9.81 * cos(state(0));

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);
        B(1, 0) = 1;

        std::vector<Eigen::MatrixXd> jacobians;
        jacobians.push_back(A);
        jacobians.push_back(B);
        return jacobians;
    }

    std::vector<Eigen::MatrixXd> getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        std::vector<Eigen::MatrixXd> hessians;
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);
        hessians.push_back(hessian);
        hessians.push_back(hessian);
        return hessians;
    }
};

}  // namespace cddp