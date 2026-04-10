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

#include "interior_point_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace cddp::detail {

void printInteriorPointIteration(int iter, double objective, double inf_pr,
                                 double inf_du, double inf_comp, double mu,
                                 double step_norm, double regularization,
                                 double alpha_du, double alpha_pr) {
  if (iter == 0) {
    std::cout << std::setw(4) << "iter"
              << " " << std::setw(12) << "objective"
              << " " << std::setw(9) << "inf_pr"
              << " " << std::setw(9) << "inf_du"
              << " " << std::setw(9) << "inf_comp"
              << " " << std::setw(7) << "lg(mu)"
              << " " << std::setw(9) << "||d||"
              << " " << std::setw(7) << "lg(rg)"
              << " " << std::setw(9) << "alpha_du"
              << " " << std::setw(9) << "alpha_pr" << std::endl;
  }

  std::cout << std::setw(4) << iter << " ";
  std::cout << std::setw(12) << std::scientific << std::setprecision(6)
            << objective << " ";
  std::cout << std::setw(9) << std::scientific << std::setprecision(2) << inf_pr
            << " ";
  std::cout << std::setw(9) << std::scientific << std::setprecision(2) << inf_du
            << " ";
  std::cout << std::setw(9) << std::scientific << std::setprecision(2)
            << inf_comp << " ";

  if (mu > 0.0) {
    std::cout << std::setw(7) << std::fixed << std::setprecision(1)
              << std::log10(mu) << " ";
  } else {
    std::cout << std::setw(7) << "-inf"
              << " ";
  }

  std::cout << std::setw(9) << std::scientific << std::setprecision(2)
            << step_norm << " ";

  if (regularization > 0.0) {
    std::cout << std::setw(7) << std::fixed << std::setprecision(1)
              << std::log10(regularization) << " ";
  } else {
    std::cout << std::setw(7) << "-"
              << " ";
  }

  std::cout << std::setw(9) << std::fixed << std::setprecision(6) << alpha_du
            << " ";
  std::cout << std::setw(9) << std::fixed << std::setprecision(6) << alpha_pr;
  std::cout << std::endl;
}

bool acceptFilterEntry(std::vector<FilterPoint> &filter, double merit_function,
                       double constraint_violation) {
  FilterPoint candidate(merit_function, constraint_violation);
  for (const auto &filter_point : filter) {
    if (filter_point.dominates(candidate)) {
      return false;
    }
  }

  filter.erase(std::remove_if(filter.begin(), filter.end(),
                              [&candidate](const FilterPoint &point) {
                                return candidate.dominates(point);
                              }),
               filter.end());
  filter.push_back(candidate);
  return true;
}

bool isFilterCandidateDominated(const std::vector<FilterPoint> &filter,
                                double merit_function,
                                double constraint_violation) {
  FilterPoint candidate(merit_function, constraint_violation);
  return std::any_of(filter.begin(), filter.end(),
                     [&candidate](const FilterPoint &point) {
                       return point.dominates(candidate);
                     });
}

bool filterContainsInvalidValues(const std::vector<FilterPoint> &filter) {
  return std::any_of(filter.begin(), filter.end(), [](const FilterPoint &point) {
    return !std::isfinite(point.merit_function) ||
           !std::isfinite(point.constraint_violation);
  });
}

void pruneFilterToBestPoints(std::vector<FilterPoint> &filter) {
  if (filter.empty()) {
    return;
  }

  const auto best_violation =
      *std::min_element(filter.begin(), filter.end(),
                        [](const FilterPoint &a, const FilterPoint &b) {
                          return a.constraint_violation <
                                 b.constraint_violation;
                        });
  const auto best_merit =
      *std::min_element(filter.begin(), filter.end(),
                        [](const FilterPoint &a, const FilterPoint &b) {
                          return a.merit_function < b.merit_function;
                        });

  filter.clear();
  filter.push_back(best_violation);
  if (std::abs(best_merit.constraint_violation -
                   best_violation.constraint_violation) > 1e-12 ||
      std::abs(best_merit.merit_function - best_violation.merit_function) >
          1e-12) {
    filter.push_back(best_merit);
  }
}

double computeMaxConstraintViolation(
    const std::map<std::string, std::vector<Eigen::VectorXd>> &constraints) {
  double max_violation = 0.0;

  for (const auto &constraint_pair : constraints) {
    for (const auto &constraint_value : constraint_pair.second) {
      if (constraint_value.size() == 0) {
        continue;
      }
      max_violation = std::max(max_violation, constraint_value.maxCoeff());
    }
  }

  return max_violation;
}

} // namespace cddp::detail
