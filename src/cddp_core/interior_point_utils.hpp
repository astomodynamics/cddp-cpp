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

#ifndef CDDP_INTERIOR_POINT_UTILS_HPP
#define CDDP_INTERIOR_POINT_UTILS_HPP

#include <map>
#include <vector>

#include <Eigen/Dense>

#include "cddp_core/cddp_core.hpp"

namespace cddp::detail {

void printInteriorPointIteration(int iter, double objective, double inf_pr,
                                 double inf_du, double inf_comp, double mu,
                                 double step_norm, double regularization,
                                 double alpha_du, double alpha_pr);

bool acceptFilterEntry(std::vector<FilterPoint> &filter, double merit_function,
                       double constraint_violation);

bool isFilterCandidateDominated(const std::vector<FilterPoint> &filter,
                                double merit_function,
                                double constraint_violation);

bool filterContainsInvalidValues(const std::vector<FilterPoint> &filter);

void pruneFilterToBestPoints(std::vector<FilterPoint> &filter);

double computeMaxConstraintViolation(
    const std::map<std::string, std::vector<Eigen::VectorXd>> &constraints);

} // namespace cddp::detail

#endif // CDDP_INTERIOR_POINT_UTILS_HPP
