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


# Dynamics model tests
add_executable(test_bicycle dynamics_model/test_bicycle.cpp)
target_link_libraries(test_bicycle gtest gmock gtest_main cddp)
gtest_discover_tests(test_bicycle)

add_executable(test_car dynamics_model/test_car.cpp)
target_link_libraries(test_car gtest gmock gtest_main cddp)
gtest_discover_tests(test_car)

add_executable(test_cartpole dynamics_model/test_cartpole.cpp)
target_link_libraries(test_cartpole gtest gmock gtest_main cddp)
gtest_discover_tests(test_cartpole)

add_executable(test_dubins_car dynamics_model/test_dubins_car.cpp)
target_link_libraries(test_dubins_car gtest gmock gtest_main cddp)
gtest_discover_tests(test_dubins_car)

add_executable(test_lti_system dynamics_model/test_lti_system.cpp)
target_link_libraries(test_lti_system gtest gmock gtest_main cddp)
gtest_discover_tests(test_lti_system)

add_executable(test_manipulator dynamics_model/test_manipulator.cpp)
target_link_libraries(test_manipulator gtest gmock gtest_main cddp)
gtest_discover_tests(test_manipulator)

add_executable(test_pendulum dynamics_model/test_pendulum.cpp)
target_link_libraries(test_pendulum gtest gmock gtest_main cddp)
gtest_discover_tests(test_pendulum)

add_executable(test_quadrotor dynamics_model/test_quadrotor.cpp)
target_link_libraries(test_quadrotor gtest gmock gtest_main cddp)
gtest_discover_tests(test_quadrotor)

add_executable(test_spacecraft_landing2d dynamics_model/test_spacecraft_landing2d.cpp)
target_link_libraries(test_spacecraft_landing2d gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_spacecraft_landing2d)

add_executable(test_spacecraft_linear dynamics_model/test_spacecraft_linear.cpp)
target_link_libraries(test_spacecraft_linear gtest gmock gtest_main cddp)
gtest_discover_tests(test_spacecraft_linear)


# Core tests
add_executable(test_boxqp cddp_core/test_boxqp.cpp)
target_link_libraries(test_boxqp gtest gmock gtest_main cddp)
gtest_discover_tests(test_boxqp)

add_executable(test_finite_difference cddp_core/test_finite_difference.cpp)
target_link_libraries(test_finite_difference gtest gmock gtest_main cddp)
gtest_discover_tests(test_finite_difference)

add_executable(test_objective cddp_core/test_objective.cpp)
target_link_libraries(test_objective gtest gmock gtest_main cddp)
gtest_discover_tests(test_objective)

add_executable(test_constraint cddp_core/test_constraint.cpp)
target_link_libraries(test_constraint gtest gmock gtest_main cddp)
gtest_discover_tests(test_constraint)

add_executable(test_cddp_core cddp_core/test_cddp_core.cpp)
target_link_libraries(test_cddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_cddp_core)

add_executable(test_clddp_core cddp_core/test_clddp_core.cpp)
target_link_libraries(test_clddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_clddp_core)

add_executable(test_logcddp_core cddp_core/test_logcddp_core.cpp)
target_link_libraries(test_logcddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_logcddp_core)

add_executable(test_animation test_animation.cpp)
target_link_libraries(test_animation gtest gmock gtest_main cddp)

if (CDDP-CPP_SQP)
    add_executable(test_sqp sqp_core/test_sqp.cpp)
    target_link_libraries(test_sqp gtest gmock gtest_main cddp)
    gtest_discover_tests(test_sqp)
endif()

# Test for Eigen installation
add_executable(test_eigen test_eigen.cpp)
target_link_libraries(test_eigen gtest gmock gtest_main cddp)
gtest_discover_tests(test_eigen)

# Test for gurobi
if (CDDP-CPP_GUROBI)
    add_executable(test_gurobi test_gurobi.cpp)
    target_link_libraries(test_gurobi gtest gmock gtest_main cddp)
    gtest_discover_tests(test_gurobi)     

    # Test for qp solver comparison
    add_executable(test_qp_solvers test_qp_solvers.cpp)
    target_link_libraries(test_qp_solvers gtest gmock gtest_main cddp)
    gtest_discover_tests(test_qp_solvers)
endif()

# Test for torch
if (CDDP-CPP_TORCH)
    add_executable(test_torch test_torch.cpp)
    target_link_libraries(test_torch gtest gmock gtest_main cddp)
    gtest_discover_tests(test_torch)

    add_executable(test_torch_pendulum dynamics_model/test_torch_pendulum.cpp)
    target_link_libraries(test_torch_pendulum gtest gmock gtest_main cddp)
    gtest_discover_tests(test_torch_pendulum)
endif()