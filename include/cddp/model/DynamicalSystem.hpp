#ifndef CDDP_DYNAMICALSYSTEM_HPP
#define CDDP_DYNAMICALSYSTEM_HPP

#include <Eigen/Dense>

namespace cddp {
class DynamicalSystem {
public:
    // State and Control Dimensions
    int state_size_;
    int control_size_;
    double dt_;  // Time step
    int integration_type_;

    DynamicalSystem(int state_dim, int control_dim, double timestep, int integration_type) : 
        state_size_(state_dim), control_size_(control_dim), dt_(timestep), integration_type_(integration_type) {}
    
    // Pure virtual methods  
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) {
        Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(state_size_);
        return state_dot;
    }

    
    Eigen::VectorXd getDynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) {
        if (integration_type_ == 0) {
            return euler_step(state, control, dt_);
        } else if (integration_type_ == 1) {
            return heun_step(state, control, dt_);
        } else if (integration_type_ == 2) {
            return rk3_step(state, control, dt_);
        } else if (integration_type_ == 3) {
            return rk4_step(state, control, dt_);
        } 
    }
        
    virtual Eigen::MatrixXd getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_size_, state_size_);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);

        Eigen::MatrixXd combined_jacobian(state_size_, state_size_ + control_size_);

        // Insert A and B into the combined matrix
        combined_jacobian.block(0, 0, state_size_-1, state_size_-1) = A;
        combined_jacobian.block(0, state_size_, state_size_-1, control_size_-1) = B;
        return combined_jacobian;
    };

    virtual Eigen::MatrixXd getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) = 0;


    virtual Eigen::MatrixXd getTrajectory(const Eigen::VectorXd &initialState, const std::vector<Eigen::VectorXd> &controlSeq, int num_steps) {}; 

    // Optional virtual methods
    virtual void setCostMatrices(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R) {} 
    virtual double calculateCost(const Eigen::VectorXd &state, const Eigen::VectorXd &control) = 0;
    virtual void calculateCostGradietn(const Eigen::VectorXd &state, const Eigen::VectorXd &control, Eigen::VectorXd &cost_gradient) {}

    virtual void calculateCostHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, Eigen::MatrixXd &cost_hessian) {}
    
    virtual void setFinalCostMatrix(const Eigen::MatrixXd &Qf) {}  
    virtual double calculateFinalCost(const Eigen::VectorXd &state) = 0;
    virtual void calculateFinalCostGradient(const Eigen::VectorXd &state, Eigen::VectorXd &final_cost_gradient) {}

    virtual void calculateFinalCostHessian(const Eigen::VectorXd &state, Eigen::MatrixXd &final_cost_hessian) {}
    
    virtual void setGoal(const Eigen::VectorXd &goal) {}

private:
    Eigen::VectorXd euler_step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, double dt) {
        return state + dynamics(state, control) * dt;
    }

    Eigen::VectorXd heun_step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, double dt) {
        Eigen::VectorXd k1 = dynamics(state, control);
        Eigen::VectorXd k2 = dynamics(state + dt * k1, control);
        return state + 0.5 * dt * (k1 + k2);
    }

    Eigen::VectorXd rk3_step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, double dt) {
        Eigen::VectorXd k1 = dynamics(state, control);
        Eigen::VectorXd k2 = dynamics(state + 0.5 * dt * k1, control);
        Eigen::VectorXd k3 = dynamics(state - dt * k1 + 2 * dt * k2, control);
        return state + (dt / 6.0) * (k1 + 4 * k2 + k3);
    }
    Eigen::VectorXd rk4_step(const Eigen::VectorXd &state, const Eigen::VectorXd &control, double dt) {
        
        Eigen::VectorXd k1 = dynamics(state, control);
        Eigen::VectorXd k2 = dynamics(state + 0.5 * dt * k1, control);
        Eigen::VectorXd k3 = dynamics(state + 0.5 * dt * k2, control);
        Eigen::VectorXd k4 = dynamics(state + dt * k3, control);
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    }
};
}  // namespace cddp

#endif // CDDP_DYNAMICALSYSTEM_HPP