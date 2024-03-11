#ifndef CDDP_DOUBLEINTEGRATOR_HPP
#define CDDP_DOUBLEINTEGRATOR_HPP

#include <Eigen/Dense>
#include <vector>
#include "DynamicalSystem.hpp" // Include the missing header file

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

    
    Eigen::MatrixXd getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_size_, state_size_);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);

        Eigen::MatrixXd combined_jacobian(state_size_, state_size_ + control_size_);

        // Insert A and B into the combined matrix
        combined_jacobian.block(0, 0, state_size_, state_size_) = A;
        combined_jacobian.block(0, state_size_, state_size_, control_size_) = B;
        return combined_jacobian;
    }

    Eigen::MatrixXd getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        // Assuming the Hessian is mostly zeros for a double integrator
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);
        return hessian;
    }

    double calculateCost(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
       // Your implementation here
        return 0.0;
   }

   double calculateFinalCost(const Eigen::VectorXd &state) override {
       // Your implementation here
       return 0.0;
   }
};
    
}  // namespace cddp
    
#endif // CDDP_DOUBLEINTEGRATOR_HPP