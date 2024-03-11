#ifndef CDDP_DOUBLEINTEGRATOR_HPP
#define CDDP_DOUBLEINTEGRATOR_HPP

#include <Eigen/Dense>
#include <vector>

namespace cddp {

class DoubleIntegrator : public cddp::DynamicalSystem {
public:
    int state_size;
    int control_size;
    double timestep;  // Time step

    DoubleIntegrator(int state_dim, int control_dim, double timestep) :
        DynamicalSystem(state_dim, control_dim, timestep) {}

    Eigen::VectorXd getDynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::VectorXd next_state = state;
        next_state.head(2) +=  state.tail(2) * dt;
        next_state.tail(2) += control * dt;
        return next_state;
    }

    Eigen::MatrixXd getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_size, state_size);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size, control_size);

        Eigen::MatrixXd combined_jacobian(state_size, state_size + control_size);

        // Insert A and B into the combined matrix
        combined_jacobian.block(0, 0, state_size, state_size) = A; 
        combined_jacobian.block(0, state_size, state_size, control_size) = B;
        return combined_jacobian;
    }

    Eigen::MatrixXd getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) override {
        // Assuming the Hessian is mostly zeros for a double integrator
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(state_size, state_size + control_size);
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