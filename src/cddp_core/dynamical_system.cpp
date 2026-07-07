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

#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <iostream>

#include "cddp_core/dynamical_system.hpp"

using namespace cddp;
using namespace autodiff;

Eigen::VectorXd DynamicalSystem::euler_step(const Eigen::VectorXd &state,
                                            const Eigen::VectorXd &control,
                                            double dt, double time) const {
  return state + dt * getContinuousDynamics(state, control, time);
}

Eigen::VectorXd DynamicalSystem::heun_step(const Eigen::VectorXd &state,
                                           const Eigen::VectorXd &control,
                                           double dt, double time) const {
  Eigen::VectorXd k1 = getContinuousDynamics(state, control, time);
  Eigen::VectorXd k2 =
      getContinuousDynamics(state + dt * k1, control, time + dt);
  return state + 0.5 * dt * (k1 + k2);
}

Eigen::VectorXd DynamicalSystem::rk3_step(const Eigen::VectorXd &state,
                                          const Eigen::VectorXd &control,
                                          double dt, double time) const {
  Eigen::VectorXd k1 = getContinuousDynamics(state, control, time);
  Eigen::VectorXd k2 =
      getContinuousDynamics(state + 0.5 * dt * k1, control, time + 0.5 * dt);
  Eigen::VectorXd k3 =
      getContinuousDynamics(state - dt * k1 + 2 * dt * k2, control, time + dt);
  return state + (dt / 6) * (k1 + 4 * k2 + k3);
}

Eigen::VectorXd DynamicalSystem::rk4_step(const Eigen::VectorXd &state,
                                          const Eigen::VectorXd &control,
                                          double dt, double time) const {
  Eigen::VectorXd k1 = getContinuousDynamics(state, control, time);
  Eigen::VectorXd k2 =
      getContinuousDynamics(state + 0.5 * dt * k1, control, time + 0.5 * dt);
  Eigen::VectorXd k3 =
      getContinuousDynamics(state + 0.5 * dt * k2, control, time + 0.5 * dt);
  Eigen::VectorXd k4 =
      getContinuousDynamics(state + dt * k3, control, time + dt);
  return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
}

Eigen::VectorXd
DynamicalSystem::getDiscreteDynamics(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control,
                                     double time) const {
  if (integration_type_ == "euler") {
    return euler_step(state, control, timestep_, time);
  } else if (integration_type_ == "heun") {
    return heun_step(state, control, timestep_, time);
  } else if (integration_type_ == "rk3") {
    return rk3_step(state, control, timestep_, time);
  } else if (integration_type_ == "rk4") {
    return rk4_step(state, control, timestep_, time);
  } else {
    std::cerr << "Integration type not supported!" << std::endl;
    return Eigen::VectorXd::Zero(state.size());
  }
}

Eigen::VectorXd
DynamicalSystem::getContinuousDynamics(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control,
                                       double time) const {
  Eigen::VectorXd next_state = getDiscreteDynamics(state, control, time);
  Eigen::VectorXd continuous_dynamics = (next_state - state) / timestep_;

  return continuous_dynamics;
}

Eigen::MatrixXd
DynamicalSystem::getStateJacobian(const Eigen::VectorXd &state,
                                  const Eigen::VectorXd &control,
                                  double time) const {
  VectorXdual2nd x = state;
  VectorXdual2nd u = control;

  auto dynamics_wrt_x = [&](const VectorXdual2nd &x_ad) -> VectorXdual2nd {
    return this->getContinuousDynamicsAutodiff(x_ad, u, time);
  };

  Eigen::MatrixXd Jx = jacobian(dynamics_wrt_x, wrt(x), at(x));
  return Jx;
}

Eigen::MatrixXd
DynamicalSystem::getControlJacobian(const Eigen::VectorXd &state,
                                    const Eigen::VectorXd &control,
                                    double time) const {
  VectorXdual2nd x = state;
  VectorXdual2nd u = control;

  auto dynamics_wrt_u = [&](const VectorXdual2nd &u_ad) -> VectorXdual2nd {
    return this->getContinuousDynamicsAutodiff(x, u_ad, time);
  };

  Eigen::MatrixXd Ju = jacobian(dynamics_wrt_u, wrt(u), at(u));
  return Ju;
}

VectorXdual2nd
DynamicalSystem::getDiscreteDynamicsAutodiff(const VectorXdual2nd &state,
                                             const VectorXdual2nd &control,
                                             double time) const {
  const double dt = timestep_;
  if (integration_type_ == "euler") {
    return state + dt * getContinuousDynamicsAutodiff(state, control, time);
  } else if (integration_type_ == "heun") {
    VectorXdual2nd k1 = getContinuousDynamicsAutodiff(state, control, time);
    VectorXdual2nd k2 =
        getContinuousDynamicsAutodiff(state + dt * k1, control, time + dt);
    return state + 0.5 * dt * (k1 + k2);
  } else if (integration_type_ == "rk3") {
    VectorXdual2nd k1 = getContinuousDynamicsAutodiff(state, control, time);
    VectorXdual2nd k2 = getContinuousDynamicsAutodiff(state + 0.5 * dt * k1,
                                                      control, time + 0.5 * dt);
    VectorXdual2nd k3 = getContinuousDynamicsAutodiff(
        state - dt * k1 + 2 * dt * k2, control, time + dt);
    return state + (dt / 6) * (k1 + 4 * k2 + k3);
  } else if (integration_type_ == "rk4") {
    VectorXdual2nd k1 = getContinuousDynamicsAutodiff(state, control, time);
    VectorXdual2nd k2 = getContinuousDynamicsAutodiff(state + 0.5 * dt * k1,
                                                      control, time + 0.5 * dt);
    VectorXdual2nd k3 = getContinuousDynamicsAutodiff(state + 0.5 * dt * k2,
                                                      control, time + 0.5 * dt);
    VectorXdual2nd k4 =
        getContinuousDynamicsAutodiff(state + dt * k3, control, time + dt);
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
  }
  throw std::runtime_error("Integration type not supported for autodiff "
                           "discrete dynamics: " +
                           integration_type_);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
DynamicalSystem::getJacobians(const Eigen::VectorXd &state,
                              const Eigen::VectorXd &control,
                              double time) const {
  if (discrete_autodiff_status_ == 0) {
    try {
      VectorXdual2nd x = state;
      VectorXdual2nd u = control;
      VectorXdual2nd next_ad = getDiscreteDynamicsAutodiff(x, u, time);
      Eigen::VectorXd next_ad_val(next_ad.size());
      for (int i = 0; i < next_ad.size(); ++i) {
        next_ad_val(i) = val(next_ad(i));
      }
      const Eigen::VectorXd next = getDiscreteDynamics(state, control, time);
      const double value_scale = 1.0 + next.cwiseAbs().maxCoeff();
      bool consistent =
          (next_ad_val - next).cwiseAbs().maxCoeff() <= 1e-9 * value_scale;

      if (consistent) {
        auto fd_wrt_x = [&](const Eigen::VectorXd &x_in) {
          return getDiscreteDynamics(x_in, control, time);
        };
        auto fd_wrt_u = [&](const Eigen::VectorXd &u_in) {
          return getDiscreteDynamics(state, u_in, time);
        };
        auto ad_wrt_x = [&](const VectorXdual2nd &x_in) -> VectorXdual2nd {
          return getDiscreteDynamicsAutodiff(x_in, u, time);
        };
        auto ad_wrt_u = [&](const VectorXdual2nd &u_in) -> VectorXdual2nd {
          return getDiscreteDynamicsAutodiff(x, u_in, time);
        };
        const Eigen::MatrixXd A_ad = jacobian(ad_wrt_x, wrt(x), at(x));
        const Eigen::MatrixXd B_ad = jacobian(ad_wrt_u, wrt(u), at(u));
        const Eigen::MatrixXd A_fd = finite_difference_jacobian(fd_wrt_x, state);
        const Eigen::MatrixXd B_fd = finite_difference_jacobian(fd_wrt_u, control);
        const double jac_scale = 1.0 + std::max(A_fd.cwiseAbs().maxCoeff(),
                                                B_fd.cwiseAbs().maxCoeff());
        consistent = (A_ad - A_fd).cwiseAbs().maxCoeff() <= 1e-5 * jac_scale &&
                     (B_ad - B_fd).cwiseAbs().maxCoeff() <= 1e-5 * jac_scale;
      }
      discrete_autodiff_status_ = consistent ? 1 : -1;
      if (!consistent) {
        std::cerr << "Warning: autodiff discrete dynamics disagree with "
                     "getDiscreteDynamics; falling back to finite-difference "
                     "Jacobians. Check the model's autodiff implementation."
                  << std::endl;
      }
    } catch (const std::exception &) {
      discrete_autodiff_status_ = -1;
    }
  }

  if (discrete_autodiff_status_ == 1) {
    VectorXdual2nd x = state;
    VectorXdual2nd u = control;
    auto discrete_wrt_x = [&](const VectorXdual2nd &x_ad) -> VectorXdual2nd {
      return getDiscreteDynamicsAutodiff(x_ad, u, time);
    };
    auto discrete_wrt_u = [&](const VectorXdual2nd &u_ad) -> VectorXdual2nd {
      return getDiscreteDynamicsAutodiff(x, u_ad, time);
    };
    return {jacobian(discrete_wrt_x, wrt(x), at(x)),
            jacobian(discrete_wrt_u, wrt(u), at(u))};
  }

  auto dynamics_wrt_x = [&](const Eigen::VectorXd &x) {
    return getDiscreteDynamics(x, control, time);
  };
  auto dynamics_wrt_u = [&](const Eigen::VectorXd &u) {
    return getDiscreteDynamics(state, u, time);
  };

  return {finite_difference_jacobian(dynamics_wrt_x, state),
          finite_difference_jacobian(dynamics_wrt_u, control)};
}

std::vector<Eigen::MatrixXd>
DynamicalSystem::getStateHessian(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd &control,
                                 double time) const {
  int n = state_dim_;
  int m = control_dim_;
  std::vector<Eigen::MatrixXd> state_hessian_tensor(state_dim_);

  VectorXdual2nd z(n + m);
  z.head(n) = state;
  z.tail(m) = control;

  for (int i = 0; i < state_dim_; ++i) {
    auto f_i = [&](const VectorXdual2nd &z_ad) -> autodiff::dual2nd {
      VectorXdual2nd x_ad = z_ad.head(n);
      VectorXdual2nd u_ad = z_ad.tail(m);
      return this->getContinuousDynamicsAutodiff(x_ad, u_ad, time)(i);
    };

    Eigen::MatrixXd H_i = hessian(f_i, wrt(z), at(z));
    state_hessian_tensor[i] = H_i.topLeftCorner(n, n);
  }
  return state_hessian_tensor;
}

std::vector<Eigen::MatrixXd>
DynamicalSystem::getControlHessian(const Eigen::VectorXd &state,
                                   const Eigen::VectorXd &control,
                                   double time) const {
  int n = state_dim_;
  int m = control_dim_;
  std::vector<Eigen::MatrixXd> control_hessian_tensor(state_dim_);

  VectorXdual2nd z(n + m);
  z.head(n) = state;
  z.tail(m) = control;

  for (int i = 0; i < state_dim_; ++i) {
    auto f_i = [&](const VectorXdual2nd &z_ad) -> autodiff::dual2nd {
      VectorXdual2nd x_ad = z_ad.head(n);
      VectorXdual2nd u_ad = z_ad.tail(m);
      return this->getContinuousDynamicsAutodiff(x_ad, u_ad, time)(i);
    };
    Eigen::MatrixXd H_i = hessian(f_i, wrt(z), at(z));
    control_hessian_tensor[i] = H_i.bottomRightCorner(m, m);
  }
  return control_hessian_tensor;
}

std::vector<Eigen::MatrixXd>
DynamicalSystem::getCrossHessian(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd &control,
                                 double time) const {
  int n = state_dim_;
  int m = control_dim_;
  std::vector<Eigen::MatrixXd> cross_hessian_tensor(state_dim_);

  VectorXdual2nd z(n + m);
  z.head(n) = state;
  z.tail(m) = control;

  for (int i = 0; i < state_dim_; ++i) {
    auto f_i = [&](const VectorXdual2nd &z_ad) -> autodiff::dual2nd {
      VectorXdual2nd x_ad = z_ad.head(n);
      VectorXdual2nd u_ad = z_ad.tail(m);
      return this->getContinuousDynamicsAutodiff(x_ad, u_ad, time)(i);
    };
    Eigen::MatrixXd H_i = hessian(f_i, wrt(z), at(z));
    cross_hessian_tensor[i] = H_i.bottomLeftCorner(m, n);
  }
  return cross_hessian_tensor;
}
