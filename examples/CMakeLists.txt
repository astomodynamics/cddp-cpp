#  Copyright 2024 Tomo Sasaki

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#       https://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Curated native reference examples that are built with
# `CDDP_CPP_BUILD_EXAMPLES=ON`. These are compile-checked examples of the
# public C++ API, while the Python portfolio remains the visualization path.

function(add_cddp_reference_example target_name source_name)
  add_executable(${target_name} ${source_name})
  target_link_libraries(${target_name} PRIVATE cddp)
endfunction()

add_cddp_reference_example(cddp_pendulum cddp_pendulum.cpp)
add_cddp_reference_example(cddp_cartpole cddp_cartpole.cpp)
add_cddp_reference_example(cddp_unicycle cddp_unicycle.cpp)
add_cddp_reference_example(cddp_quadrotor_point cddp_quadrotor_point.cpp)
add_cddp_reference_example(cddp_manipulator cddp_manipulator.cpp)
