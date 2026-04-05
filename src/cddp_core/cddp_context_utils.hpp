/*
 Copyright 2026 Tomo Sasaki

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

#ifndef CDDP_CONTEXT_UTILS_HPP
#define CDDP_CONTEXT_UTILS_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "cddp_core/constraint.hpp"
#include "cddp_core/options.hpp"

namespace cddp::detail {

std::vector<double> buildLineSearchAlphas(const LineSearchOptions &options);

bool hasCompatibleWarmStartTrajectories(
    const CDDPOptions &options, const std::vector<Eigen::VectorXd> &states,
    const std::vector<Eigen::VectorXd> &controls, int horizon, int state_dim,
    int control_dim);

void ensureTrajectoryShape(std::vector<Eigen::VectorXd> &trajectory,
                           int expected_size, int vector_dim);

void addOrReplaceConstraint(
    std::map<std::string, std::unique_ptr<Constraint>> &constraint_set,
    std::string constraint_name, std::unique_ptr<Constraint> constraint,
    int &total_dual_dim);

bool removeConstraint(
    std::map<std::string, std::unique_ptr<Constraint>> &constraint_set,
    const std::string &constraint_name, int &total_dual_dim);

} // namespace cddp::detail

#endif // CDDP_CONTEXT_UTILS_HPP
