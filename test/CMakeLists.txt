# Find the Eigen3 library
find_package(Eigen3 REQUIRED)
if (APPLE)
    include_directories(/usr/local/include/eigen3) 
else ()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Include directories for your source files
include_directories(../include/cddp) 


# --------- CDDP Package Test ---------
# ---- DynamicalSystemTests ----
add_executable(cddp_dynamics_tests DynamicalSystemTests.cpp)
target_link_libraries(cddp_dynamics_tests PRIVATE Eigen3::Eigen)

# ---- CDDPTests ----
add_executable(cddp_problem_tests 
    CDDPProblemTests.cpp
    ../src/cddp_core/CDDPProblem.cpp)
target_link_libraries(cddp_problem_tests PRIVATE Eigen3::Eigen)



