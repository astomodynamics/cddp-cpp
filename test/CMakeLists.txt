
# Include directories for your source files
include_directories(../include/cddp) 

# Find the Eigen3 library
find_package(Eigen3 REQUIRED)
if (APPLE)
    include_directories(/usr/local/include/eigen3) 
else ()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Find the OSQP library
find_package(osqp REQUIRED)

include(FetchContent)
FetchContent_Declare(
  OsqpEigen
  GIT_REPOSITORY https://github.com/robotology/osqp-eigen.git
)
FetchContent_MakeAvailable(OsqpEigen)


# --------- CDDP Package Test ---------
# ---- DynamicalSystemTests ----
add_executable(cddp_dynamics_tests DynamicalSystemTests.cpp)
target_link_libraries(cddp_dynamics_tests PRIVATE Eigen3::Eigen)

# ---- ObjectiveFunctionTests ----
add_executable(cddp_objective_tests 
    ObjectiveFunctionTests.cpp)

# ---- CDDPTests ----
add_executable(cddp_problem_tests 
    CDDPProblemTests.cpp
    ../src/cddp_core/CDDPProblem.cpp)
target_link_libraries(cddp_problem_tests PRIVATE Eigen3::Eigen osqp::osqp OsqpEigen::OsqpEigen)



