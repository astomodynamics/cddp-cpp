#ifndef CDDP_DOUBLEINTEGRATOR_HPP
#define CDDP_DOUBLEINTEGRATOR_HPP

#include <Eigen/Dense>
#include <vector>
#include "cddp_core/DynamicalSystem.hpp" // Include the missing header file

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

    
    std::vector<Eigen::MatrixXd> getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        std::vector<Eigen::MatrixXd> jacobians;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(state_size_, state_size_);
        A.block(0, 2, 2, 2) = Eigen::MatrixXd::Identity(2, 2);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);
        B.block(2, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2);
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
    
#endif // CDDP_DOUBLEINTEGRATOR_HPP