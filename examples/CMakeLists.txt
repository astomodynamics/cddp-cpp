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

add_executable(test_barrier_strategies test_barrier_strategies.cpp)
target_link_libraries(test_barrier_strategies cddp)

add_executable(cddp_bicycle cddp_bicycle.cpp)
target_link_libraries(cddp_bicycle cddp)

add_executable(cddp_car cddp_car.cpp)
target_link_libraries(cddp_car cddp)

add_executable(cddp_car_ipddp cddp_car_ipddp.cpp)
target_link_libraries(cddp_car_ipddp cddp)

add_executable(cddp_forklift_ipddp cddp_forklift_ipddp.cpp)
target_link_libraries(cddp_forklift_ipddp cddp)

add_executable(cddp_cartpole cddp_cartpole.cpp)
target_link_libraries(cddp_cartpole cddp)

add_executable(cddp_hcw cddp_hcw.cpp)
target_link_libraries(cddp_hcw cddp)

add_executable(cddp_lti_system cddp_lti_system.cpp)
target_link_libraries(cddp_lti_system cddp)

add_executable(cddp_manipulator cddp_manipulator.cpp)
target_link_libraries(cddp_manipulator cddp)

add_executable(cddp_pendulum cddp_pendulum.cpp)
target_link_libraries(cddp_pendulum cddp)

add_executable(cddp_acrobot cddp_acrobot.cpp)
target_link_libraries(cddp_acrobot cddp)

add_executable(cddp_quadrotor_circle cddp_quadrotor_circle.cpp)
target_link_libraries(cddp_quadrotor_circle cddp)

add_executable(cddp_quadrotor_figure_eight_horizontal cddp_quadrotor_figure_eight_horizontal.cpp)
target_link_libraries(cddp_quadrotor_figure_eight_horizontal cddp)

add_executable(cddp_quadrotor_figure_eight_vertical cddp_quadrotor_figure_eight_vertical.cpp)
target_link_libraries(cddp_quadrotor_figure_eight_vertical cddp)

add_executable(cddp_quadrotor_point cddp_quadrotor_point.cpp)
target_link_libraries(cddp_quadrotor_point cddp)

add_executable(cddp_quadrotor_figure_eight_horizontal_safe cddp_quadrotor_figure_eight_horizontal_safe.cpp)
target_link_libraries(cddp_quadrotor_figure_eight_horizontal_safe cddp)

add_executable(cddp_unicycle cddp_unicycle.cpp)
target_link_libraries(cddp_unicycle cddp)

add_executable(cddp_unicycle_safe cddp_unicycle_safe.cpp)
target_link_libraries(cddp_unicycle_safe cddp)

add_executable(cddp_unicycle_safe_ipddp cddp_unicycle_safe_ipddp.cpp)
target_link_libraries(cddp_unicycle_safe_ipddp cddp)

add_executable(cddp_unicycle_safe_ipddp_v2 cddp_unicycle_safe_ipddp_v2.cpp)
target_link_libraries(cddp_unicycle_safe_ipddp_v2 cddp)

add_executable(cddp_unicycle_safe_ipddp_v3 cddp_unicycle_safe_ipddp_v3.cpp)
target_link_libraries(cddp_unicycle_safe_ipddp_v3 cddp)

add_executable(cddp_spacecraft_linear_docking cddp_spacecraft_linear_docking.cpp)
target_link_libraries(cddp_spacecraft_linear_docking cddp)

add_executable(cddp_spacecraft_linear_rpo cddp_spacecraft_linear_rpo.cpp)
target_link_libraries(cddp_spacecraft_linear_rpo cddp)

add_executable(cddp_spacecraft_nonlinear_rpo cddp_spacecraft_nonlinear_rpo.cpp)
target_link_libraries(cddp_spacecraft_nonlinear_rpo cddp)

add_executable(cddp_spacecraft_rpo cddp_spacecraft_rpo.cpp)
target_link_libraries(cddp_spacecraft_rpo cddp)

add_executable(cddp_spacecraft_rpo_mc cddp_spacecraft_rpo_mc.cpp)
target_link_libraries(cddp_spacecraft_rpo_mc cddp)

add_executable(cddp_spacecraft_rpo_fuel cddp_spacecraft_rpo_fuel.cpp)
target_link_libraries(cddp_spacecraft_rpo_fuel cddp)

add_executable(cddp_spacecraft_roe_rpo cddp_spacecraft_roe_rpo.cpp)
target_link_libraries(cddp_spacecraft_roe_rpo cddp)

add_executable(mpc_hcw mpc_hcw.cpp)
target_link_libraries(mpc_hcw cddp)

add_executable(cddp_unicycle_mpc cddp_unicycle_mpc.cpp)
target_link_libraries(cddp_unicycle_mpc cddp)

# Ipopt examples
if (CDDP_CPP_CASADI)
    add_executable(ipopt_car ipopt_car.cpp)
    target_link_libraries(ipopt_car cddp)

    add_executable(ipopt_unicycle ipopt_unicycle.cpp)
    target_link_libraries(ipopt_unicycle cddp)

    add_executable(ipopt_cartpole ipopt_cartpole.cpp)
    target_link_libraries(ipopt_cartpole cddp)

    add_executable(ipopt_quadrotor ipopt_quadrotor.cpp)
    target_link_libraries(ipopt_quadrotor cddp)

    add_executable(ipopt_spacecrat_linear_fuel ipopt_spacecrat_linear_fuel.cpp)
    target_link_libraries(ipopt_spacecrat_linear_fuel cddp)
endif()

# SNOPT examples
if (CDDP_CPP_CASADI AND CDDP_CPP_SNOPT)
    add_executable(snopt_unicycle snopt_unicycle.cpp)
    target_link_libraries(snopt_unicycle cddp)
endif()
