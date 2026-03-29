/*
 Copyright 2025 Tomo Sasaki

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

#ifndef CDDP_EXAMPLE_UTILS_HPP
#define CDDP_EXAMPLE_UTILS_HPP

#include <Eigen/Dense>
#include <filesystem>
#include <vector>

namespace cddp::example {

/// Extract a single scalar component from a trajectory of vectors.
inline std::vector<double>
extractComponent(const std::vector<Eigen::VectorXd> &traj, int index) {
  std::vector<double> result;
  result.reserve(traj.size());
  for (const auto &v : traj)
    result.push_back(v(index));
  return result;
}

/// Build a uniform time vector: [0, dt, 2*dt, ..., (count-1)*dt].
inline std::vector<double> makeTimeVector(int count, double dt) {
  std::vector<double> t(count);
  for (int i = 0; i < count; ++i)
    t[i] = i * dt;
  return t;
}

/// Ensure a directory exists (creates parents if needed).
inline void ensurePlotDir(const std::string &dir) {
  if (!std::filesystem::exists(dir))
    std::filesystem::create_directories(dir);
}

/// Create a zero-control initial trajectory with the initial state replicated.
inline std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
makeInitialTrajectory(const Eigen::VectorXd &initial_state, int horizon,
                      int control_dim) {
  return {std::vector<Eigen::VectorXd>(horizon + 1, initial_state),
          std::vector<Eigen::VectorXd>(horizon,
                                       Eigen::VectorXd::Zero(control_dim))};
}

} // namespace cddp::example

#endif // CDDP_EXAMPLE_UTILS_HPP
