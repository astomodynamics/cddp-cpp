/*
 Copyright 2024 Tomo Sasaki

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "dynamics_model/spacecraft_nonlinear.hpp"
#include "cddp_core/helper.hpp"

namespace cddp {

SpacecraftNonlinear::SpacecraftNonlinear(
    double timestep,
    std::string integration_type,
    double mass,
    double r_scale,
    double v_scale,
    double mu)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      mass_(mass),
      r_scale_(r_scale),
      v_scale_(v_scale),
      mu_(mu) {
}

Eigen::VectorXd SpacecraftNonlinear::getContinuousDynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control, double time) const {

    Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

    // Unpack state vector
    const double px = state(STATE_PX);
    const double py = state(STATE_PY);
    const double pz = state(STATE_PZ);
    const double vx = state(STATE_VX);
    const double vy = state(STATE_VY);
    const double vz = state(STATE_VZ);
    const double r0 = state(STATE_R0);
    const double theta = state(STATE_THETA);
    const double dr0 = state(STATE_DR0);
    const double dtheta = state(STATE_DTHETA);

    // Unpack control vector
    const double ux = control(CONTROL_UX);
    const double uy = control(CONTROL_UY);
    const double uz = control(CONTROL_UZ);

    // Precompute common terms
    const double den = std::pow((r0 + px) * (r0 + px) + py * py + pz * pz, 1.5);
    const double r0_sq = r0 * r0;

    // Compute accelerations
    const double ddr0 = -mu_ / (r0_sq) + r0 * dtheta * dtheta;
    const double ddtheta = -2.0 * dr0 * dtheta / r0;

    const double ddx = 2.0 * dtheta * vy + ddtheta * py + dtheta * dtheta * px 
                      - mu_ * (px + r0) / den + mu_ / r0_sq + ux / mass_;

    const double ddy = -2.0 * dtheta * vx - ddtheta * px + dtheta * dtheta * py 
                      - mu_ * py / den + uy / mass_;

    const double ddz = -mu_ * pz / den + uz / mass_;

    // Pack state derivative vector
    state_dot << vx, vy, vz,              // Position derivatives
                 ddx, ddy, ddz,            // Velocity derivatives
                 dr0, dtheta,             // Orbital parameter derivatives
                 ddr0, ddtheta;           // Orbital parameter acceleration derivatives

    return state_dot;
}

Eigen::MatrixXd SpacecraftNonlinear::getStateJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control, double time) const {

    auto f = [&](const Eigen::VectorXd& x) {
        return getContinuousDynamics(x, control, time);
    };

    return finite_difference_jacobian(f, state);
}

Eigen::MatrixXd SpacecraftNonlinear::getControlJacobian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control, double time) const {

    auto f = [&](const Eigen::VectorXd& u) {
        return getContinuousDynamics(state, u, time);
    };
    return finite_difference_jacobian(f, control);
}

std::vector<Eigen::MatrixXd> SpacecraftNonlinear::getStateHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control, double time) const {
    
    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
    }
    return hessians;
}

std::vector<Eigen::MatrixXd> SpacecraftNonlinear::getControlHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control, double time) const {

    std::vector<Eigen::MatrixXd> hessians(STATE_DIM);
    for (int i = 0; i < STATE_DIM; ++i) {
        hessians[i] = Eigen::MatrixXd::Zero(CONTROL_DIM, CONTROL_DIM);
    }
    return hessians;
}


} // namespace cddp