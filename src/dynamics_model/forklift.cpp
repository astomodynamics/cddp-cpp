#include "dynamics_model/forklift.hpp"
#include <cmath>
#include <autodiff/forward/dual.hpp> // Include dual types and math functions
#include <autodiff/forward/dual/eigen.hpp> // Include Eigen support for dual types
#include "cddp_core/helper.hpp" // For finite_difference_jacobian (used in old methods)

namespace cddp {

Forklift::Forklift(double timestep, double wheelbase, std::string integration_type, 
                   bool rear_steer, double max_steering_angle)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      wheelbase_(wheelbase),
      rear_steer_(rear_steer),
      max_steering_angle_(max_steering_angle) {
}

Eigen::VectorXd Forklift::getDiscreteDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    const double x = state(STATE_X);         // x position
    const double y = state(STATE_Y);         // y position
    const double theta = state(STATE_THETA);  // forklift angle
    const double v = state(STATE_V);         // velocity
    const double delta = state(STATE_DELTA); // steering angle

    const double a = control(CONTROL_A);          // acceleration
    const double ddelta = control(CONTROL_DDELTA); // steering rate

    const double L = wheelbase_;  // wheelbase
    const double h = timestep_;   // timestep

    // Apply rear-steer sign convention
    const double steer_sign = rear_steer_ ? -1.0 : 1.0;
    const double effective_delta = steer_sign * delta;

    // Kinematic bicycle model
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    const double tan_delta = std::tan(effective_delta);
    
    Eigen::VectorXd dy = Eigen::VectorXd::Zero(STATE_DIM);
    dy(STATE_X) = h * v * cos_theta;
    dy(STATE_Y) = h * v * sin_theta;
    dy(STATE_THETA) = h * v * tan_delta / L; 
    dy(STATE_V) = h * a;
    dy(STATE_DELTA) = h * ddelta; 

    return state + dy;
}

Eigen::MatrixXd Forklift::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    VectorXdual2nd state_dual = state.cast<autodiff::dual2nd>();
    VectorXdual2nd control_dual = control.cast<autodiff::dual2nd>();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

    for (int i = 0; i < STATE_DIM; ++i) {
        auto dynamics_i = [this, i, &control_dual, time](const VectorXdual2nd& x) -> autodiff::dual2nd {
            VectorXdual2nd dynamics = this->getDiscreteDynamicsAutodiff(x, control_dual, time);
            return dynamics(i);
        };
        J.row(i) = autodiff::gradient(dynamics_i, autodiff::wrt(state_dual), at(state_dual));
    }

    J.diagonal().array() -= 1.0;
    J /= timestep_;
    
    return J;
}

Eigen::MatrixXd Forklift::getControlJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    VectorXdual2nd state_dual = state.cast<autodiff::dual2nd>();
    VectorXdual2nd control_dual = control.cast<autodiff::dual2nd>();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);

    for (int i = 0; i < STATE_DIM; ++i) {
        auto dynamics_i = [this, i, &state_dual, time](const VectorXdual2nd& u) -> autodiff::dual2nd {
            VectorXdual2nd dynamics = this->getDiscreteDynamicsAutodiff(state_dual, u, time);
            return dynamics(i);
        };

        J.row(i) = autodiff::gradient(dynamics_i, autodiff::wrt(control_dual), at(control_dual));
    }
    J /= timestep_;
    
    return J;
}

std::vector<Eigen::MatrixXd> Forklift::getStateHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    VectorXdual2nd state_dual = state.cast<autodiff::dual2nd>();
    VectorXdual2nd control_dual = control.cast<autodiff::dual2nd>();
    auto hessians = makeZeroTensor(STATE_DIM, STATE_DIM, STATE_DIM);

    for (int i = 0; i < STATE_DIM; ++i) {
        auto dynamics_i = [this, i, &control_dual, time](const VectorXdual2nd& x) -> autodiff::dual2nd {
            VectorXdual2nd dynamics = this->getDiscreteDynamicsAutodiff(x, control_dual, time);
            return dynamics(i);
        };

        hessians[i] = autodiff::hessian(dynamics_i, autodiff::wrt(state_dual), at(state_dual));
        hessians[i] /= timestep_;
    }
    
    return hessians;
}

std::vector<Eigen::MatrixXd> Forklift::getControlHessian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    VectorXdual2nd state_dual = state.cast<autodiff::dual2nd>();
    VectorXdual2nd control_dual = control.cast<autodiff::dual2nd>();
    auto hessians = makeZeroTensor(STATE_DIM, CONTROL_DIM, CONTROL_DIM);

    for (int i = 0; i < STATE_DIM; ++i) {
        auto dynamics_i = [this, i, &state_dual, time](const VectorXdual2nd& u) -> autodiff::dual2nd {
            VectorXdual2nd dynamics = this->getDiscreteDynamicsAutodiff(state_dual, u, time);
            return dynamics(i);
        };

        hessians[i] = autodiff::hessian(dynamics_i, autodiff::wrt(control_dual), at(control_dual));
        hessians[i] /= timestep_;
    }
    
    return hessians;
}

cddp::VectorXdual2nd Forklift::getDiscreteDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {

    const autodiff::dual2nd x = state(STATE_X);
    const autodiff::dual2nd y = state(STATE_Y);
    const autodiff::dual2nd theta = state(STATE_THETA);
    const autodiff::dual2nd v = state(STATE_V);
    const autodiff::dual2nd delta_raw = state(STATE_DELTA);

    const autodiff::dual2nd a = control(CONTROL_A);
    const autodiff::dual2nd ddelta = control(CONTROL_DDELTA);

    const double L = wheelbase_;
    const double h = timestep_;

    // Apply rear-steer sign convention
    const double steer_sign = rear_steer_ ? -1.0 : 1.0;
    const autodiff::dual2nd effective_delta = steer_sign * delta_raw;

    const autodiff::dual2nd cos_theta = cos(theta);
    const autodiff::dual2nd sin_theta = sin(theta);
    const autodiff::dual2nd tan_delta = tan(effective_delta);

    VectorXdual2nd dy = VectorXdual2nd::Zero(STATE_DIM);
    dy(STATE_X) = h * v * cos_theta;
    dy(STATE_Y) = h * v * sin_theta;
    dy(STATE_THETA) = h * v * tan_delta / L;
    dy(STATE_V) = h * a;
    dy(STATE_DELTA) = h * ddelta;

    return state + dy;
}

cddp::VectorXdual2nd Forklift::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {
    VectorXdual2nd next_state = this->getDiscreteDynamicsAutodiff(state, control, time);
    return (next_state - state) / timestep_;
}

} // namespace cddp
