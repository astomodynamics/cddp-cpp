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

#include "cddp_context_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace cddp::detail {

namespace {

bool hasCompatibleVectorLayout(const std::vector<Eigen::VectorXd> &trajectory,
                               int expected_size, int vector_dim) {
  return static_cast<int>(trajectory.size()) == expected_size &&
         std::all_of(trajectory.begin(), trajectory.end(),
                     [vector_dim](const Eigen::VectorXd &value) {
                       return value.size() == vector_dim;
                     });
}

} // namespace

std::vector<double> buildLineSearchAlphas(const LineSearchOptions &options) {
  std::vector<double> alphas;
  alphas.reserve(static_cast<size_t>(std::max(1, options.max_iterations)));

  double current_alpha = options.initial_step_size;
  for (int i = 0; i < options.max_iterations; ++i) {
    alphas.push_back(current_alpha);
    current_alpha *= options.step_reduction_factor;
    if (current_alpha < options.min_step_size &&
        i < options.max_iterations - 1) {
      alphas.push_back(options.min_step_size);
      break;
    }
  }

  if (alphas.empty()) {
    alphas.push_back(options.initial_step_size);
  }

  return alphas;
}

bool hasCompatibleWarmStartTrajectories(
    const CDDPOptions &options, const std::vector<Eigen::VectorXd> &states,
    const std::vector<Eigen::VectorXd> &controls, int horizon, int state_dim,
    int control_dim) {
  return options.warm_start &&
         hasCompatibleVectorLayout(states, horizon + 1, state_dim) &&
         hasCompatibleVectorLayout(controls, horizon, control_dim);
}

void ensureTrajectoryShape(std::vector<Eigen::VectorXd> &trajectory,
                           int expected_size, int vector_dim) {
  if (hasCompatibleVectorLayout(trajectory, expected_size, vector_dim)) {
    return;
  }

  trajectory.assign(static_cast<size_t>(expected_size),
                    Eigen::VectorXd::Zero(vector_dim));
}

void addOrReplaceConstraint(
    std::map<std::string, std::unique_ptr<Constraint>> &constraint_set,
    std::string constraint_name, std::unique_ptr<Constraint> constraint,
    int &total_dual_dim) {
  if (!constraint) {
    throw std::runtime_error("Cannot add null constraint.");
  }

  const int dual_dim = constraint->getDualDim();
  auto existing_constraint = constraint_set.find(constraint_name);
  if (existing_constraint != constraint_set.end()) {
    total_dual_dim -= existing_constraint->second->getDualDim();
  }

  constraint_set[constraint_name] = std::move(constraint);
  total_dual_dim += dual_dim;
}

bool removeConstraint(
    std::map<std::string, std::unique_ptr<Constraint>> &constraint_set,
    const std::string &constraint_name, int &total_dual_dim) {
  auto it = constraint_set.find(constraint_name);
  if (it == constraint_set.end()) {
    return false;
  }

  total_dual_dim -= it->second->getDualDim();
  constraint_set.erase(it);
  return true;
}

} // namespace cddp::detail
