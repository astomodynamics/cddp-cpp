#ifndef CDDP_DOUBLEINTEGRATOR_HPP
#define CDDP_DOUBLEINTEGRATOR_HPP

#include "Eigen/Dense"
#include <vector>
#include "../cddp_core/DynamicalSystem.hpp" // Include the missing header file

namespace cddp {

class DoubleIntegrator : public cddp::DynamicalSystem {
public:
    int state_size_;
    int control_size_;
    double timestep_;  // Time step
    int integration_type_;

    DoubleIntegrator(int state_dim, int control_dim, double timestep, int integration_type) :
        DynamicalSystem(state_dim, control_dim, timestep, integration_type) {
            state_size_ = state_dim;
            control_size_ = control_dim;
            timestep_ = timestep;
            integration_type_ = integration_type;
        }

    Eigen::VectorXd dynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(state_size_);
        state_dot.head(2) =  state.tail(2);
        state_dot.tail(2) = control;
        return state_dot; 
    }

    
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(state_size_, state_size_);
        A.block(0, 2, 2, 2) = Eigen::MatrixXd::Identity(2, 2);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);
        B.block(2, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2);

        return std::make_tuple(A, B);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);

        return std::make_tuple(hessian, hessian, hessian);
    }   
};
    
}  // namespace cddp
    
#endif // CDDP_DOUBLEINTEGRATOR_HPP
