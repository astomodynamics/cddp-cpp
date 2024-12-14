#include "dynamics_model/lti_system.hpp"
#include <unsupported/Eigen/MatrixFunctions>

namespace cddp {

LTISystem::LTISystem(int state_dim, int control_dim, double timestep, 
                     std::string integration_type)
    : DynamicalSystem(state_dim, control_dim, timestep, integration_type) {
    initializeRandomSystem();
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

Eigen::VectorXd LTISystem::getContinuousDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For LTI system: dx/dt = Ax + Bu
    return A_ * state + B_ * control;
}

Eigen::MatrixXd LTISystem::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For LTI system, state Jacobian is just A
    return A_;
}

Eigen::MatrixXd LTISystem::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // For LTI system, control Jacobian is just B
    return B_;
}

Eigen::MatrixXd LTISystem::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // State Hessian is zero for linear system
    return Eigen::MatrixXd::Zero(state_dim_ * state_dim_, state_dim_);
}

Eigen::MatrixXd LTISystem::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
    
    // Control Hessian is zero for linear system
    return Eigen::MatrixXd::Zero(state_dim_ * control_dim_, control_dim_);
}

} // namespace cddp