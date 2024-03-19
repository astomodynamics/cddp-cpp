#ifndef CDDP_DYNAMICALSYSTEM_HPP
#define CDDP_DYNAMICALSYSTEM_HPP

#include "Eigen/Dense"

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
        Eigen::VectorXd next_state;
        if (integration_type_ == 0) {
            next_state = euler_step(state, control, dt_);
        } else if (integration_type_ == 1) {
            next_state = heun_step(state, control, dt_);
        } else if (integration_type_ == 2) {
            next_state = rk3_step(state, control, dt_);
        } else if (integration_type_ == 3) {
            next_state = rk4_step(state, control, dt_);
        } 
        return next_state;
    }
        
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_size_, state_size_);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, control_size_);
        
        return std::make_tuple(A, B);
    };

    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(state_size_, state_size_ + control_size_);
        
        return std::make_tuple(A, B, C);
    };


    // Optional methods
    // virtual std::vector<Eigen::VectorXd> getTrajectory(const Eigen::VectorXd &initialState, const std::vector<Eigen::VectorXd> &controlSeq, int num_steps) {}; 
    
    
    // virtual void setGoal(const Eigen::VectorXd &goal) {}

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
