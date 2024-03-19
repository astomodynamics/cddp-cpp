#ifndef CDDP_DUBINSCAR_HPP
#define CDDP_DUBINSCAR_HPP

#include "Eigen/Dense"
#include <vector>
#include "cddp/cddp_core/DynamicalSystem.hpp" 

namespace cddp {

class DubinsCar : public cddp::DynamicalSystem {
public: 
    int state_size_;
    int control_size_;
    double timestep_;  // Time step
    int integration_type_;

    DubinsCar(int state_dim, int control_dim, double timestep, int integration_type) :
        DynamicalSystem(state_dim, control_dim, timestep, integration_type) {
            state_size_ = state_dim;
            control_size_ = control_dim;
            timestep_ = timestep;
            integration_type_ = integration_type;
        }

    Eigen::VectorXd dynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(state_size_);
        state_dot(0) = control(0) * cos(state(2));
        state_dot(1) = control(0) * sin(state(2));
        state_dot(2) = control(1);
        return state_dot;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(state_size_, state_size_);
        A(0, 2) = -control(0) * sin(state(2));
        A(1, 2) = control(0) * cos(state(2));

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);
        B(0, 0) = cos(state(2));
        B(1, 0) = sin(state(2));
        B(2, 1) = 1;

        return std::make_tuple(A, B);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>  getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);
    
        return std::make_tuple(hessian, hessian, hessian);
    }
};
    
}  // namespace cddp

#endif // CDDP_DUBINSCAR_HPP
