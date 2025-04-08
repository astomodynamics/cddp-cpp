#include "dynamics_model/spacecraft_twobody.hpp"

#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include "cddp_core/helper.hpp"

namespace cddp {

SpacecraftTwobody::SpacecraftTwobody(double timestep, double mu, double mass)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, "euler"), mu_(mu), mass_(mass) {}

Eigen::VectorXd SpacecraftTwobody::getContinuousDynamics(
    const Eigen::VectorXd &state, const Eigen::VectorXd &control) const {
  Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

  const double x = state(STATE_X);
  const double y = state(STATE_Y);
  const double z = state(STATE_Z);
  const double vx = state(STATE_VX);
  const double vy = state(STATE_VY);
  const double vz = state(STATE_VZ);

  const double ux = control(CONTROL_UX);
  const double uy = control(CONTROL_UY);
  const double uz = control(CONTROL_UZ);

  const double r = std::sqrt(x * x + y * y + z * z);
  const double r3 = r * r * r;

  // Position dynamics
  state_dot(STATE_X) = vx;
  state_dot(STATE_Y) = vy;
  state_dot(STATE_Z) = vz;

  // Velocity dynamics
  state_dot(STATE_VX) = -mu_ * x / r3 + ux / mass_;
  state_dot(STATE_VY) = -mu_ * y / r3 + uy / mass_;
  state_dot(STATE_VZ) = -mu_ * z / r3 + uz / mass_;

  return state_dot;
}

Eigen::MatrixXd SpacecraftTwobody::getStateJacobian(
    const Eigen::VectorXd &state, const Eigen::VectorXd &control) const {
  auto f = [&](const Eigen::VectorXd &x) {
    return getContinuousDynamics(x, control);
  };

  return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd SpacecraftTwobody::getControlJacobian(
    const Eigen::VectorXd &state, const Eigen::VectorXd &control) const {
  auto f = [&](const Eigen::VectorXd &u) {
    return getContinuousDynamics(state, u);
  };
  return finite_difference_jacobian(f, control);
}

std::vector<Eigen::MatrixXd> SpacecraftTwobody::getStateHessian(
    const Eigen::VectorXd &state, const Eigen::VectorXd &control) const {
  std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
  for (int i = 0; i < STATE_DIM; ++i) {
    hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
  }
  return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftTwobody::getControlHessian(
    const Eigen::VectorXd &state, const Eigen::VectorXd &control) const {
  std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
  for (int i = 0; i < STATE_DIM; ++i) {
    hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
  }
  return hessians;
}

} // namespace cddp