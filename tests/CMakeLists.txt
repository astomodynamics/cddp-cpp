
add_executable(test_pendulum dynamics_model/test_pendulum.cpp)
target_link_libraries(test_pendulum gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_pendulum)

add_executable(test_dubins_car dynamics_model/test_dubins_car.cpp)
target_link_libraries(test_dubins_car gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_dubins_car)

add_executable(test_bicycle dynamics_model/test_bicycle.cpp)
target_link_libraries(test_bicycle gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_bicycle)

add_executable(test_cartpole dynamics_model/test_cartpole.cpp)
target_link_libraries(test_cartpole gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_cartpole)

add_executable(test_quadrotor dynamics_model/test_quadrotor.cpp)
target_link_libraries(test_quadrotor gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_quadrotor)

add_executable(test_manipulator dynamics_model/test_manipulator.cpp)
target_link_libraries(test_manipulator gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_manipulator)

add_executable(test_spacecraft_linear dynamics_model/test_spacecraft_linear.cpp)
target_link_libraries(test_spacecraft_linear gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_spacecraft_linear)

add_executable(test_objective cddp_core/test_objective.cpp)
target_link_libraries(test_objective gtest gmock gtest_main cddp)
gtest_discover_tests(test_objective)

add_executable(test_cddp_core cddp_core/test_cddp_core.cpp)
target_link_libraries(test_cddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_cddp_core)

add_executable(test_clddp_core cddp_core/test_clddp_core.cpp)
target_link_libraries(test_clddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_clddp_core)

add_executable(test_logcddp_core cddp_core/test_logcddp_core.cpp)
target_link_libraries(test_logcddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_logcddp_core)

add_executable(test_constraint cddp_core/test_constraint.cpp)
target_link_libraries(test_constraint gtest gmock gtest_main cddp)
gtest_discover_tests(test_constraint)

add_executable(test_boxqp cddp_core/test_boxqp.cpp)
target_link_libraries(test_boxqp gtest gmock gtest_main cddp)
gtest_discover_tests(test_boxqp)

add_executable(test_finite_difference cddp_core/test_finite_difference.cpp)
target_link_libraries(test_finite_difference gtest gmock gtest_main cddp)
gtest_discover_tests(test_finite_difference)

# # Test for eigen vs torch
add_executable(test_eigen test_eigen.cpp)
target_link_libraries(test_eigen gtest gmock gtest_main Eigen3::Eigen)
gtest_discover_tests(test_eigen)


# Test for gurobi
if (CDDP-CPP_GUROBI)
    add_executable(test_gurobi test_gurobi.cpp)
    target_link_libraries(test_gurobi gtest gmock gtest_main cddp)
    gtest_discover_tests(test_gurobi)     


    # Test for SDQP
    add_executable(test_sdqp test_sdqp.cpp)
    target_link_libraries(test_sdqp gtest gmock gtest_main cddp)
    gtest_discover_tests(test_sdqp)

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