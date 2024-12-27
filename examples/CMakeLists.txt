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
target_link_libraries(cddp_bicycle cddp)

add_executable(cddp_car cddp_car.cpp)
target_link_libraries(cddp_car cddp)

add_executable(cddp_cartpole cddp_cartpole.cpp)
target_link_libraries(cddp_cartpole cddp)

add_executable(cddp_dubins_car cddp_dubins_car.cpp)
target_link_libraries(cddp_dubins_car cddp)

add_executable(cddp_manipulator cddp_manipulator.cpp)
target_link_libraries(cddp_manipulator cddp)

add_executable(cddp_lti_system cddp_lti_system.cpp)
target_link_libraries(cddp_lti_system cddp)

add_executable(cddp_pendulum cddp_pendulum.cpp)
target_link_libraries(cddp_pendulum cddp)

add_executable(cddp_quadrotor cddp_quadrotor.cpp)
target_link_libraries(cddp_quadrotor cddp)

# Ipopt examples
if (CDDP_CPP_CASADI)
    add_executable(ipopt_car ipopt_car.cpp)
    target_link_libraries(ipopt_car cddp)
endif()

# Neural dynamics examples
if (CDDP_CPP_TORCH)
    add_executable(prepare_pendulum neural_dynamics/prepare_pendulum.cpp)
    target_link_libraries(prepare_pendulum cddp)

    # add_executable(prepare_cartpole neural_dynamics/prepare_cartpole.cpp)
    # target_link_libraries(prepare_cartpole cddp)

    add_executable(train_pendulum neural_dynamics/train_pendulum.cpp)
    target_link_libraries(train_pendulum cddp)

    # add_executable(train_cartpole neural_dynamics/train_cartpole.cpp)
    # target_link_libraries(train_cartpole cddp)

    add_executable(run_pendulum neural_dynamics/run_pendulum.cpp)
    target_link_libraries(run_pendulum cddp)

    # add_executable(run_cartpole neural_dynamics/run_cartpole.cpp)
    # target_link_libraries(run_cartpole cddp)

    # add_executable(cddp_pendulum_neural _cddp_pendulum_neural.cpp)
    # target_link_libraries(cddp_pendulum_neural cddp)
endif()
