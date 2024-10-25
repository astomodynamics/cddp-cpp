
add_executable(test_pendulum dynamics_model/test_pendulum.cpp)
target_link_libraries(test_pendulum gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_pendulum)

add_executable(test_dubins_car dynamics_model/test_dubins_car.cpp)
target_link_libraries(test_dubins_car gtest gmock gtest_main cddp Eigen3::Eigen)
gtest_discover_tests(test_dubins_car)

add_executable(test_objective cddp_core/test_objective.cpp)
target_link_libraries(test_objective gtest gmock gtest_main cddp)
gtest_discover_tests(test_objective)

add_executable(test_cddp_core cddp_core/test_cddp_core.cpp)
target_link_libraries(test_cddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_cddp_core)

add_executable(test_clddp_core cddp_core/test_clddp_core.cpp)
target_link_libraries(test_clddp_core gtest gmock gtest_main cddp)
gtest_discover_tests(test_clddp_core)

add_executable(test_constraint cddp_core/test_constraint.cpp)
target_link_libraries(test_constraint gtest gmock gtest_main cddp)
gtest_discover_tests(test_constraint)

# # Test for eigen vs torch
add_executable(test_eigen test_eigen.cpp)
target_link_libraries(test_eigen gtest gmock gtest_main Eigen3::Eigen)
gtest_discover_tests(test_eigen)


# Test for gurobi
if (CDDP-CPP_GUROBI)
    add_executable(test_gurobi test_gurobi.cpp)
    target_link_libraries(test_gurobi gtest gmock gtest_main cddp)
    gtest_discover_tests(test_gurobi)     
endif()

# Test for SDQP
add_executable(test_sdqp test_sdqp.cpp)
target_link_libraries(test_sdqp gtest gmock gtest_main cddp)
gtest_discover_tests(test_sdqp)