#include "dynamics_model/lti_system.hpp"
#include <unsupported/Eigen/MatrixFunctions>

namespace cddp {

LTISystem::LTISystem(int state_dim, int control_dim, double timestep, 
                     std::string integration_type)
    : DynamicalSystem(state_dim, control_dim, timestep, integration_type) {
    // initializeRandomSystem();

    A_ = Eigen::MatrixXd::Zero(state_dim, state_dim);
    A_ << 0,    0.2473,   -0.7933,    0.3470,
   -0.2473,         0,   -0.7667,    2.1307,
    0.7933,    0.7667,         0,    0.3154,
   -0.3470,   -2.1307,   -0.3154,         0;

   // Convert to discrete time using matrix exponential
    A_ = (timestep_ * A_).exp();

   B_ = Eigen::MatrixXd::Zero(state_dim, control_dim);
   B_ << -0.6387,   -0.2026,
   -0.4049,   -0.1975,
    2.3939,    1.5163,
   -0.0496,   -1.7322;

   // Generate random B matrix
    B_ = timestep_ * B_;
}

LTISystem::LTISystem(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                     double timestep, std::string integration_type)
    : DynamicalSystem(A.rows(), B.cols(), timestep, integration_type),
      A_(A), B_(B) {
    
    // Verify dimensions
    if (A_.rows() != A_.cols()) {
        throw std::invalid_argument("A matrix must be square");
    }
    if (B_.rows() != A_.rows()) {
        throw std::invalid_argument("B matrix must have same number of rows as A");
    }
}

void LTISystem::initializeRandomSystem() {
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Generate random skew-symmetric A matrix for stability
    A_ = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    for(int i = 0; i < state_dim_; ++i) {
        for(int j = i+1; j < state_dim_; ++j) {
            double val = dist(gen);
            A_(i,j) = val;
            A_(j,i) = -val;
        }
    }

    // Convert to discrete time using matrix exponential
    A_ = (timestep_ * A_).exp();

    // Generate random B matrix
    B_ = timestep_ * Eigen::MatrixXd::Random(state_dim_, control_dim_);
}

Eigen::VectorXd LTISystem::getDiscreteDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For LTI system: dx/dt = Ax + Bu
    return A_ * state + B_ * control;
}

Eigen::MatrixXd LTISystem::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For LTI system, state Jacobian is just A
    Eigen::MatrixXd A = A_;
    A.diagonal().array() -= 1.0;
    return A/timestep_;
}

Eigen::MatrixXd LTISystem::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For LTI system, control Jacobian is just B
    return B_/timestep_;
}

std::vector<Eigen::MatrixXd> LTISystem::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // State Hessian is zero for linear system
    std::vector<Eigen::MatrixXd> hessians(state_dim_);
    for (int i = 0; i < state_dim_; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(state_dim_, state_dim_);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> LTISystem::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // Control Hessian is zero for linear system
    std::vector<Eigen::MatrixXd> hessians(state_dim_);
    for (int i = 0; i < state_dim_; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(control_dim_, control_dim_);
    }
    return hessians;
}

} // namespace cddp