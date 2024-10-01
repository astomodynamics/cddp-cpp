#ifndef CDDP_PENDULUM_H
#define CDDP_PENDULUM_H

#include "cddp-cpp/cddp_core/dynamical_system.h" 
#include <Eigen/Dense>

namespace cddp
{

class Pendulum : public DynamicalSystem {
public:
    // Constructor
    Pendulum(double mass, double length, double gravity, double timestep);

    // Dynamics: Computes the next state given the current state and control input
    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Discrete dynamics (declare it even if using base class implementation)
    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override {
        return DynamicalSystem::getDiscreteDynamics(state, control); // Use base class implementation
    }
    
    // Jacobians: Computes the Jacobian matrices of the dynamics with respect to state and control
    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Hessians: Computes the Hessian matrices of the dynamics (optional for now)
    // You might implement these later if needed for your DDP algorithm
    Eigen::MatrixXd getStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
    Eigen::MatrixXd getControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

private:
    double mass_;     // Mass of the pendulum bob
    double length_;   // Length of the pendulum rod
    double gravity_;  // Acceleration due to gravity
};

} // namespace cddp
#endif // CDDP_PENDULUM_H