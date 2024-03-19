
# Include directories for your source files
include_directories(../include/cddp) 


# --------- CDDP Package Test ---------
# ---- DynamicalSystemTests ----
add_executable(cddp_dynamics_tests DynamicalSystemTests.cpp)
target_link_libraries(cddp_dynamics_tests PUBLIC Eigen3::Eigen)

# ---- ObjectiveFunctionTests ----
add_executable(cddp_objective_tests 
    ObjectiveFunctionTests.cpp)
target_link_libraries(cddp_objective_tests PUBLIC Eigen3::Eigen)

# ---- CDDPTests ----
add_executable(cddp_problem_tests 
    CDDPProblemTests.cpp
    ../src/cddp_core/CDDPProblem.cpp)
target_link_libraries(cddp_problem_tests PUBLIC Eigen3::Eigen PRIVATE osqp-cpp)



