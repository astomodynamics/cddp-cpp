cmake_minimum_required(VERSION 3.10) 
project(cddp)

set(CMAKE_CXX_STANDARD 14) # Enforce C++14 as the minimum standard
# set(CMAKE_CXX_STANDARD_REQUIRED ON) 

include_directories(include)
include_directories(src)
add_subdirectory(test)

# Find the Eigen3 library
find_package(Eigen3 REQUIRED)
if (APPLE)
    include_directories(/usr/local/include/eigen3) 
else ()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Find the OSQP library
find_package(osqp REQUIRED)

# Find the OsqpEigen library
find_package(OsqpEigen REQUIRED)

# Eigen test
add_executable(cddp_eigen_test test/test_eigen.cpp)
target_link_libraries(cddp_eigen_test Eigen3::Eigen)

# OSQP test
add_executable(cddp_osqp_test test/test_osqp.cpp)
target_link_libraries(cddp_osqp_test PRIVATE osqp::osqp OsqpEigen::OsqpEigen) 