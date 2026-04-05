#include "dynamics_model/car.hpp"
#include <cmath>
#include <autodiff/forward/dual.hpp> // Include dual types and math functions
#include <autodiff/forward/dual/eigen.hpp> // Include Eigen support for dual types
#include "cddp_core/helper.hpp" // For finite_difference_jacobian (used in old methods)

namespace cddp {

Car::Car(double timestep, double wheelbase, std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      wheelbase_(wheelbase) {
}

Eigen::VectorXd Car::getDiscreteDynamics(
    const Eigen::VectorXd& state, const Eigen::VectorXd& control, double time) const {
    const double x = state(STATE_X);         // x position
    const double y = state(STATE_Y);         // y position
    const double theta = state(STATE_THETA);  // car angle
    const double v = state(STATE_V);         // velocity

    const double delta = control(CONTROL_DELTA);  // steering angle
    const double a = control(CONTROL_A);          // acceleration

    const double d = wheelbase_;  // distance between back and front axles
    const double h = timestep_;   // timestep

    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    Eigen::Vector2d z(cos_theta, sin_theta);

    // Front wheel rolling distance
    const double f = h * v;

    // Back wheel rolling distance
    // b = d + f*cos(w) - sqrt(d^2 - (f*sin(w))^2)
    const double b = d + f * std::cos(delta) - 
                    std::sqrt(d*d - std::pow(f * std::sin(delta), 2));

    // Change in car angle
    // dtheta = asin(sin(w)*f/d)
    const double dtheta = std::asin(std::sin(delta) * f / d);

    Eigen::VectorXd dy = Eigen::VectorXd::Zero(STATE_DIM);
    dy(STATE_X) = b * cos_theta;
    dy(STATE_Y) = b * sin_theta;
    dy(STATE_THETA) = dtheta;
    dy(STATE_V) = h * a;

    return state + dy;
}

Eigen::MatrixXd Car::getStateJacobian(
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

Eigen::MatrixXd Car::getControlJacobian(
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

std::vector<Eigen::MatrixXd> Car::getStateHessian(
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

std::vector<Eigen::MatrixXd> Car::getControlHessian(
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

cddp::VectorXdual2nd Car::getDiscreteDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {

    const autodiff::dual2nd theta = state(STATE_THETA);
    const autodiff::dual2nd v = state(STATE_V);

    const autodiff::dual2nd delta = control(CONTROL_DELTA);
    const autodiff::dual2nd a = control(CONTROL_A);

    const double d_double = wheelbase_;
    const double h = timestep_;
    const autodiff::dual2nd d = wheelbase_;

    const autodiff::dual2nd cos_theta = cos(theta);
    const autodiff::dual2nd sin_theta = sin(theta);

    const autodiff::dual2nd f = h * v;

    autodiff::dual2nd b;
    autodiff::dual2nd f_sin_delta = f * sin(delta);
    autodiff::dual2nd d_squared = d * d;
    autodiff::dual2nd inside_sqrt = d_squared - f_sin_delta * f_sin_delta;

    if (val(inside_sqrt) < 0.0) {
        inside_sqrt = 0.0;
    }
    b = d + f * cos(delta) - sqrt(inside_sqrt);


    autodiff::dual2nd dtheta;
    autodiff::dual2nd asin_arg = sin(delta) * f / d;

     if (std::abs(val(asin_arg)) > 1.0) {
        asin_arg = (val(asin_arg) > 0.0) ? 1.0 : -1.0;
     }
     dtheta = asin(asin_arg);


    VectorXdual2nd dy = VectorXdual2nd::Zero(STATE_DIM);
    dy(STATE_X) = b * cos_theta;
    dy(STATE_Y) = b * sin_theta;
    dy(STATE_THETA) = dtheta;
    dy(STATE_V) = h * a;

    return state + dy;
}

// Required continuous dynamics using autodiff discrete dynamics
cddp::VectorXdual2nd Car::getContinuousDynamicsAutodiff(
    const VectorXdual2nd& state, const VectorXdual2nd& control, double time) const {
    VectorXdual2nd next_state = this->getDiscreteDynamicsAutodiff(state, control, time);
    return (next_state - state) / timestep_;
}

} // namespace cddp
