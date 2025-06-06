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

add_executable(test_dreyfus_rocket dynamics_model/test_dreyfus_rocket.cpp)
target_link_libraries(test_dreyfus_rocket gtest gmock gtest_main cddp)
gtest_discover_tests(test_dreyfus_rocket)

add_executable(test_unicycle dynamics_model/test_unicycle.cpp)
target_link_libraries(test_unicycle gtest gmock gtest_main cddp)
gtest_discover_tests(test_unicycle)

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

add_executable(test_spacecraft_linear_fuel dynamics_model/test_spacecraft_linear_fuel.cpp)
target_link_libraries(test_spacecraft_linear_fuel gtest gmock gtest_main cddp)
gtest_discover_tests(test_spacecraft_linear_fuel)

add_executable(test_spacecraft_nonlinear dynamics_model/test_spacecraft_nonlinear.cpp)
target_link_libraries(test_spacecraft_nonlinear gtest gmock gtest_main cddp)
gtest_discover_tests(test_spacecraft_nonlinear)

add_executable(test_usv_3dof dynamics_model/test_usv_3dof.cpp)
target_link_libraries(test_usv_3dof gtest gmock gtest_main cddp)
gtest_discover_tests(test_usv_3dof)

add_executable(test_mrp_attitude dynamics_model/test_mrp_attitude.cpp)
target_link_libraries(test_mrp_attitude gtest gmock gtest_main cddp matplot)
gtest_discover_tests(test_mrp_attitude)

# add_executable(test_euler_attitude dynamics_model/test_euler_attitude.cpp)
# target_link_libraries(test_euler_attitude gtest gmock gtest_main cddp matplot)
# gtest_discover_tests(test_euler_attitude)

# add_executable(test_quaternion_attitude dynamics_model/test_quaternion_attitude.cpp)
# target_link_libraries(test_quaternion_attitude gtest gmock gtest_main cddp matplot)
# gtest_discover_tests(test_quaternion_attitude)

add_executable(test_attitude_dynamics dynamics_model/test_attitude_dynamics.cpp)
target_link_libraries(test_attitude_dynamics gtest gmock gtest_main cddp matplot)
gtest_discover_tests(test_attitude_dynamics)

add_executable(test_hessian test_hessian.cpp)
target_link_libraries(test_hessian gtest gmock gtest_main cddp)
gtest_discover_tests(test_hessian)

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

add_executable(test_clddp_solver cddp_core/test_clddp_solver.cpp)
target_link_libraries(test_clddp_solver gtest gmock gtest_main cddp)
gtest_discover_tests(test_clddp_solver)

add_executable(test_asddp_solver cddp_core/test_asddp_solver.cpp)
target_link_libraries(test_asddp_solver gtest gmock gtest_main cddp)
gtest_discover_tests(test_asddp_solver)

add_executable(test_logddp_solver cddp_core/test_logddp_solver.cpp)
target_link_libraries(test_logddp_solver gtest gmock gtest_main cddp)
gtest_discover_tests(test_logddp_solver)

add_executable(test_ipddp_solver cddp_core/test_ipddp_solver.cpp)
target_link_libraries(test_ipddp_solver gtest gmock gtest_main cddp)
gtest_discover_tests(test_ipddp_solver)

# add_executable(test_asddp_core cddp_core/test_asddp_core.cpp)

# add_executable(test_logcddp_core cddp_core/test_logcddp_core.cpp)
# target_link_libraries(test_logcddp_core gtest gmock gtest_main cddp)
# gtest_discover_tests(test_logcddp_core)

# add_executable(test_ipddp_core cddp_core/test_ipddp_core.cpp)
# target_link_libraries(test_ipddp_core gtest gmock gtest_main cddp)
# gtest_discover_tests(test_ipddp_core)

# add_executable(test_msipddp_core cddp_core/test_msipddp_core.cpp)
# target_link_libraries(test_msipddp_core gtest gmock gtest_main cddp)
# gtest_discover_tests(test_msipddp_core)

# add_executable(test_ipddp_car test_ipddp_car.cpp)
# target_link_libraries(test_ipddp_car cddp gtest gmock gtest_main)
# gtest_discover_tests(test_ipddp_car)

# add_executable(test_msipddp_car test_msipddp_car.cpp)
# target_link_libraries(test_msipddp_car gtest gmock gtest_main cddp)
# gtest_discover_tests(test_msipddp_car)

# add_executable(test_ipddp_pendulum test_ipddp_pendulum.cpp)
# target_link_libraries(test_ipddp_pendulum gtest gmock gtest_main cddp)
# gtest_discover_tests(test_ipddp_pendulum)

# add_executable(test_ipddp_quadrotor test_ipddp_quadrotor.cpp)
# target_link_libraries(test_ipddp_quadrotor gtest gmock gtest_main cddp)
# gtest_discover_tests(test_ipddp_quadrotor)

# add_executable(test_spacecraft_roe dynamics_model/test_spacecraft_roe.cpp)
# target_link_libraries(test_spacecraft_roe gtest gmock gtest_main cddp)
# gtest_discover_tests(test_spacecraft_roe)

# add_executable(test_matplot test_matplot.cpp)
# target_link_libraries(test_matplot gtest gmock gtest_main cddp)
# gtest_discover_tests(test_matplot)

# Test for torch
if (CDDP_CPP_TORCH)
    add_executable(test_torch test_torch.cpp)
    target_link_libraries(test_torch gtest gmock gtest_main cddp)
    gtest_discover_tests(test_torch)

    add_executable(test_neural_pendulum dynamics_model/test_neural_pendulum.cpp)
    target_link_libraries(test_neural_pendulum gtest gmock gtest_main cddp)
    gtest_discover_tests(test_neural_pendulum)
endif()

if (CDDP_CPP_SQP AND CDDP_CPP_CASADI)
    add_executable(test_sqp_core sqp_core/test_sqp_core.cpp)
    target_link_libraries(test_sqp_core gtest gmock gtest_main cddp)
    gtest_discover_tests(test_sqp_core)
endif()

if (CDDP_CPP_CASADI)
    add_executable(test_casadi test_casadi_solver.cpp)
    target_link_libraries(test_casadi gtest gmock gtest_main cddp)
    gtest_discover_tests(test_casadi)
endif()

# Test for Eigen installation
add_executable(test_eigen test_eigen.cpp)
target_link_libraries(test_eigen gtest gmock gtest_main cddp)
gtest_discover_tests(test_eigen)

# Test for autodiff
add_executable(test_autodiff test_autodiff.cpp)
target_link_libraries(test_autodiff gtest gmock gtest_main cddp)
gtest_discover_tests(test_autodiff)

# Test for gurobi
if (CDDP_CPP_GUROBI)
    add_executable(test_gurobi test_gurobi.cpp)
    target_link_libraries(test_gurobi gtest gmock gtest_main cddp)
    gtest_discover_tests(test_gurobi)     

    # Test for qp solver comparison
    add_executable(test_qp_solvers test_qp_solvers.cpp)
    target_link_libraries(test_qp_solvers gtest gmock gtest_main cddp)
    gtest_discover_tests(test_qp_solvers)
endif()
