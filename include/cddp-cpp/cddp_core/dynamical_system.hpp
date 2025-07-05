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
#ifndef CDDP_DYNAMICAL_SYSTEM_HPP
#define CDDP_DYNAMICAL_SYSTEM_HPP

#include "cddp_core/helper.hpp"
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp> // Include autodiff (defines dual, dual2nd)
#include <autodiff/forward/dual/eigen.hpp> // Include autodiff Eigen support
#include <vector>

namespace cddp {

// Define type alias for second-order dual vectors
using VectorXdual2nd = Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, 1>;
// Keep first-order alias for convenience if needed elsewhere, although not
// strictly necessary now
using VectorXdual = Eigen::Matrix<autodiff::dual, Eigen::Dynamic, 1>;

class DynamicalSystem {
public:
  // Constructor
  DynamicalSystem(int state_dim, int control_dim, double timestep,
                  std::string integration_type)
      : state_dim_(state_dim), control_dim_(control_dim), timestep_(timestep),
        integration_type_(integration_type) {}

  virtual ~DynamicalSystem() {} // Virtual destructor

  // Core continuous dynamics function: xdot = f(x_t, u_t)
  // This remains virtual, derived classes can implement it.
  // The base implementation uses discrete dynamics + finite difference.
  virtual Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd &state,
                                                const Eigen::VectorXd &control,
                                                double time) const;

  // Autodiff version of continuous dynamics using second-order duals
  // Derived classes MUST implement this to use the default autodiff-based
  // derivative functions (getJacobians, getHessians). If not overridden,
  // calling functions that depend on it will result in a runtime error.
  virtual VectorXdual2nd
  getContinuousDynamicsAutodiff(const VectorXdual2nd &state,
                                const VectorXdual2nd &control,
                                double time) const {
    throw std::logic_error(
        "getContinuousDynamicsAutodiff must be overridden in the derived class "
        "to use Autodiff-based derivatives.");
  }

  // Discrete dynamics function: x_{t+1} = f(x_t, u_t)
  // Uses integration based on getContinuousDynamics
  virtual Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                              const Eigen::VectorXd &control,
                                              double time) const;

  // Jacobian of dynamics w.r.t state: df/dx

  virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                           const Eigen::VectorXd &control,
                                           double time) const;

  // Jacobian of dynamics w.r.t control: df/du
  virtual Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                             const Eigen::VectorXd &control,
                                             double time) const;

  // Jacobians of dynamics w.r.t state and control: df/dx, df/du
  virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
  getJacobians(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
               double time) const {
    // This can now call the default implementations or overridden ones
    return {getStateJacobian(state, control, time),
            getControlJacobian(state, control, time)};
  }

  // Hessian of dynamics w.r.t state: d^2f/dx^2
  // Tensor (state_dim x state_dim x state_dim), vector<MatrixXd> (size
  // state_dim)
  virtual std::vector<Eigen::MatrixXd>
  getStateHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
                  double time) const;

  // Hessian of dynamics w.r.t control: d^2f/du^2
  // Tensor (state_dim x control_dim x control_dim), vector<MatrixXd> (size
  // state_dim)

  virtual std::vector<Eigen::MatrixXd>
  getControlHessian(const Eigen::VectorXd &state,
                    const Eigen::VectorXd &control, double time) const;

  // Hessian of dynamics w.r.t state and control: d^2f/dudx
  // Tensor (state_dim x control_dim x state_dim), vector<MatrixXd> (size
  // state_dim)
  virtual std::vector<Eigen::MatrixXd>
  getCrossHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
                  double time) const;

  // Hessian of dynamics w.r.t state and control: d^2f/dx^2, d^2f/du^2,
  // d^2f/dudx
  virtual std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>,
                     std::vector<Eigen::MatrixXd>>
  getHessians(const Eigen::VectorXd &state, const Eigen::VectorXd &control,
              double time) const {
    return {getStateHessian(state, control, time),
            getControlHessian(state, control, time),
            getCrossHessian(state, control, time)};
  }

  // Accessor methods
  int getStateDim() const { return state_dim_; }
  int getControlDim() const { return control_dim_; }
  double getTimestep() const { return timestep_; }
  std::string getIntegrationType() const { return integration_type_; }

protected:
  int state_dim_;
  int control_dim_;
  double timestep_;
  std::string integration_type_; // Integration type: Euler, Heun, RK3, RK4

  // Integration step functions
  Eigen::VectorXd euler_step(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control, double dt,
                             double time) const;
  Eigen::VectorXd heun_step(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control, double dt,
                            double time) const;
  Eigen::VectorXd rk3_step(const Eigen::VectorXd &state,
                           const Eigen::VectorXd &control, double dt,
                           double time) const;
  Eigen::VectorXd rk4_step(const Eigen::VectorXd &state,
                           const Eigen::VectorXd &control, double dt,
                           double time) const;
};
} // namespace cddp
#endif // CDDP_DYNAMICAL_SYSTEM_HPP