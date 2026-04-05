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

#include "dynamics_model/bicycle.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <cmath>

namespace cddp {

Bicycle::Bicycle(double timestep, double wheelbase,
                 std::string integration_type)
    : DynamicalSystem(STATE_DIM, CONTROL_DIM, timestep, integration_type),
      wheelbase_(wheelbase) {}

Eigen::VectorXd Bicycle::getContinuousDynamics(const Eigen::VectorXd &state,
                                               const Eigen::VectorXd &control,
                                               double time) const {

  Eigen::VectorXd state_dot = Eigen::VectorXd::Zero(STATE_DIM);

  const double theta = state(STATE_THETA); // heading angle
  const double v = state(STATE_V);         // velocity

  const double a = control(CONTROL_ACC);       // acceleration
  const double delta = control(CONTROL_DELTA); // steering angle

  state_dot(STATE_X) = v * std::cos(theta);                    // dx/dt
  state_dot(STATE_Y) = v * std::sin(theta);                    // dy/dt
  state_dot(STATE_THETA) = (v / wheelbase_) * std::tan(delta); // dtheta/dt
  state_dot(STATE_V) = a;                                      // dv/dt

  return state_dot;
}

VectorXdual2nd
Bicycle::getContinuousDynamicsAutodiff(const VectorXdual2nd &state,
                                       const VectorXdual2nd &control,
                                       double time) const {


  VectorXdual2nd state_dot = VectorXdual2nd::Zero(STATE_DIM);
  const autodiff::dual2nd theta = state(STATE_THETA); // heading angle
  const autodiff::dual2nd v = state(STATE_V);         // velocity

  const autodiff::dual2nd a = control(CONTROL_ACC);       // acceleration
  const autodiff::dual2nd delta = control(CONTROL_DELTA); // steering angle

  state_dot(STATE_X) = v * cos(theta);
  state_dot(STATE_Y) = v * sin(theta);
  state_dot(STATE_THETA) = (v / wheelbase_) * tan(delta);
  state_dot(STATE_V) = a;

  return state_dot;
}

Eigen::MatrixXd Bicycle::getStateJacobian(const Eigen::VectorXd &state,
                                          const Eigen::VectorXd &control,
                                          double time) const {

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);

  const double theta = state(STATE_THETA); // heading angle
  const double v = state(STATE_V);         // velocity

  const double delta = control(CONTROL_DELTA); // steering angle

  A(STATE_X, STATE_THETA) = -v * std::sin(theta);
  A(STATE_X, STATE_V) = std::cos(theta);

  A(STATE_Y, STATE_THETA) = v * std::cos(theta);
  A(STATE_Y, STATE_V) = std::sin(theta);

  A(STATE_THETA, STATE_V) = std::tan(delta) / wheelbase_;

  return A;
}

Eigen::MatrixXd Bicycle::getControlJacobian(const Eigen::VectorXd &state,
                                            const Eigen::VectorXd &control,
                                            double time) const {

  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(STATE_DIM, CONTROL_DIM);

  const double v = state(STATE_V); // velocity

  const double delta = control(CONTROL_DELTA); // steering angle

  B(STATE_V, CONTROL_ACC) = 1.0;

  B(STATE_THETA, CONTROL_DELTA) =
      v / (wheelbase_ * std::pow(std::cos(delta), 2));

  return B;
}

std::vector<Eigen::MatrixXd>
Bicycle::getStateHessian(const Eigen::VectorXd &state,
                         const Eigen::VectorXd &control, double time) const {

  auto hessians = makeZeroTensor(STATE_DIM, STATE_DIM, STATE_DIM);

  const double theta = state(STATE_THETA); // heading angle
  const double v = state(STATE_V);         // velocity

  hessians[STATE_X](STATE_THETA, STATE_THETA) = -v * std::cos(theta);

  hessians[STATE_X](STATE_THETA, STATE_V) = -std::sin(theta);
  hessians[STATE_X](STATE_V, STATE_THETA) = -std::sin(theta);

  hessians[STATE_Y](STATE_THETA, STATE_THETA) = -v * std::sin(theta);

  hessians[STATE_Y](STATE_THETA, STATE_V) = std::cos(theta);
  hessians[STATE_Y](STATE_V, STATE_THETA) = std::cos(theta);

  return hessians;
}

std::vector<Eigen::MatrixXd>
Bicycle::getControlHessian(const Eigen::VectorXd &state,
                           const Eigen::VectorXd &control, double time) const {

  auto hessians = makeZeroTensor(STATE_DIM, CONTROL_DIM, CONTROL_DIM);

  const double v = state(STATE_V); // velocity

  const double delta = control(CONTROL_DELTA); // steering angle

  hessians[STATE_THETA](CONTROL_DELTA, CONTROL_DELTA) =
      2.0 * v * std::sin(delta) / (wheelbase_ * std::pow(std::cos(delta), 3));

  return hessians;
}

} // namespace cddp
