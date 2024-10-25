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
#ifndef CDDP_HPP
#define CDDP_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>



#include "cddp-cpp/cddp_core/dynamical_system.hpp"
#include "cddp-cpp/cddp_core/objective.hpp"
#include "cddp-cpp/cddp_core/constraint.hpp"
#include "cddp-cpp/cddp_core/cddp_core.hpp"

// Models
#include "cddp-cpp/dynamics_model/pendulum.hpp"
#include "cddp-cpp/dynamics_model/dubins_car.hpp"
// #include "cddp-cpp/dynamics_model/cartpole.hpp"

#include "cddp-cpp/matplotlibcpp.hpp"


#endif // CDDP_HPP