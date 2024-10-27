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
#include "dynamics_model/dubins_car.hpp"
#include <Eigen/Dense>
#include <cmath>

using namespace cddp;

DubinsCar::DubinsCar(double timestep, std::string integration_type) : DynamicalSystem(3, 2, timestep, integration_type) {}

Eigen::VectorXd DubinsCar::getContinuousDynamics(const Eigen::VectorXd& state, 
                                                 const Eigen::VectorXd& control) const {
    Eigen::VectorXd xdot(3);
    double v = control(0);
    double omega = control(1);
    double theta = state(2);

    xdot(0) = v * std::cos(theta);
    xdot(1) = v * std::sin(theta);
    xdot(2) = omega;

    return xdot;
}

Eigen::MatrixXd DubinsCar::getStateJacobian(const Eigen::VectorXd& state, 
                                            const Eigen::VectorXd& control) const {
    Eigen::MatrixXd jacobian(3, 3);
    double v = control(0);
    double theta = state(2);

    jacobian << 0.0, 0.0, -v * std::sin(theta),
                0.0, 0.0,  v * std::cos(theta),
                0.0, 0.0,  0.0;

    return jacobian;
}

Eigen::MatrixXd DubinsCar::getControlJacobian(const Eigen::VectorXd& state, 
                                             const Eigen::VectorXd& control) const {
    Eigen::MatrixXd jacobian(3, 2);
    double theta = state(2);

    jacobian << std::cos(theta), 0.0,
                std::sin(theta), 0.0,
                0.0, 1.0;

    return jacobian;
}

Eigen::MatrixXd DubinsCar::getStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) 
const {
    return Eigen::MatrixXd::Zero(3*3, 2);
}

Eigen::MatrixXd DubinsCar::getControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control)
const {
    // TODO: Compute and return the Hessian tensor d^2f/du^2 (represented as a matrix)
    return Eigen::MatrixXd::Zero(3*2, 2);
}