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

// #include "cddp-cpp/sdqp.hpp"
#include "cddp_core/dynamical_system.hpp"
#include "cddp_core/objective.hpp"
#include "cddp_core/constraint.hpp"
#include "cddp_core/barrier.hpp"
#include "cddp_core/cddp_core.hpp"
#include "cddp_core/helper.hpp"
#include "cddp_core/boxqp.hpp"
#include "cddp_core/qp_solver.hpp"

#ifdef CDDP_CPP_TORCH_ENABLED
#include "cddp_core/neural_dynamical_system.hpp"
#endif

// Models
#include "dynamics_model/pendulum.hpp"
#include "dynamics_model/unicycle.hpp"
#include "dynamics_model/bicycle.hpp"
#include "dynamics_model/car.hpp"
#include "dynamics_model/cartpole.hpp"
#include "dynamics_model/dubins_car.hpp"
#include "dynamics_model/quadrotor.hpp"
#include "dynamics_model/manipulator.hpp"
#include "dynamics_model/spacecraft_linear.hpp"
#include "dynamics_model/spacecraft_linear_fuel.hpp"
#include "dynamics_model/spacecraft_nonlinear.hpp"
#include "dynamics_model/spacecraft_landing2d.hpp"
#include "dynamics_model/lti_system.hpp"
#include "dynamics_model/dreyfus_rocket.hpp"
#include "dynamics_model/spacecraft_roe.hpp"
#include "dynamics_model/usv_3dof.hpp"
#include "dynamics_model/euler_attitude.hpp"
#include "dynamics_model/quaternion_attitude.hpp"
#include "dynamics_model/mrp_attitude.hpp"

#include "matplot/matplot.h"

#endif // CDDP_HPP