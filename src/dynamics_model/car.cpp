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

/* This model is based on Yuval Tassa's iLQG/DDP trajectory optimization demo_car: https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
*/

#include "dynamics_model/car.hpp"
#include <Eigen/Dense>
#include <cmath>


namespace cddp {

Car::Car(double timestep) : DynamicalSystem(4, 2, timestep, "euler") {}

Eigen::VectorXd Car::getContinuousDynamics(const Eigen::VectorXd& state, 
                                           const Eigen::VectorXd& control) const {
    Eigen::VectorXd xdot(4);
    double w = control(0); // Front wheel angle
    double a = control(1); // Front wheel acceleration
    double d = 2.0;        // Distance between back and front axles

    double o = state(2);   // Car angle
    double v = state(3);   // Front wheel velocity
    double f = timestep_ * v; // Front wheel rolling distance

    // Back wheel rolling distance
    double b = d + f * std::cos(w) - std::sqrt(d * d - (f * std::sin(w)) * (f * std::sin(w))); 

    // Change in car angle
    double do_ = std::asin(std::sin(w) * f / d); 

    // Change in state
    xdot(0) = b * std::cos(o);
    xdot(1) = b * std::sin(o);
    xdot(2) = do_;
    xdot(3) = timestep_ * a;

    return xdot;
}


} // namespace cddp