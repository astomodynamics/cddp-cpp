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

# CMakeLists.txt for the CDDP examples

add_executable(cddp_bicycle cddp_bicycle.cpp)
target_link_libraries(cddp_bicycle cddp Eigen3::Eigen)

add_executable(cddp_car cddp_car.cpp)
target_link_libraries(cddp_car cddp Eigen3::Eigen)

add_executable(cddp_cartpole cddp_cartpole.cpp)
target_link_libraries(cddp_cartpole cddp Eigen3::Eigen)

add_executable(cddp_dubins_car cddp_dubins_car.cpp)
target_link_libraries(cddp_dubins_car cddp Eigen3::Eigen)

add_executable(cddp_manipulator cddp_manipulator.cpp)
target_link_libraries(cddp_manipulator cddp Eigen3::Eigen)

add_executable(cddp_lti_system cddp_lti_system.cpp)
target_link_libraries(cddp_lti_system cddp Eigen3::Eigen)

add_executable(cddp_pendulum cddp_pendulum.cpp)
target_link_libraries(cddp_pendulum cddp Eigen3::Eigen)

add_executable(cddp_quadrotor cddp_quadrotor.cpp)
target_link_libraries(cddp_quadrotor cddp Eigen3::Eigen)

