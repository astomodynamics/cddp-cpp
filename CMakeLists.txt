cmake_minimum_required(VERSION 3.10) 
project(cddp)

set(CMAKE_CXX_STANDARD 14) # Enforce C++14 as the minimum standard
# set(CMAKE_CXX_STANDARD_REQUIRED ON) 

# Options.
option(CDDP-CPP_BUILD_TESTS
  "Whether to build tests." OFF
)

# Set the default build type to Release
include_directories(include)
include_directories(src)

# Enable FetchContent for downloading dependencies
include(FetchContent)

# Eigen
find_package(Eigen3 REQUIRED)
if (APPLE)
      include_directories(/usr/local/include/eigen3) 
else ()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# OSQP-CPP
if (NOT TARGET osqp-cpp OR NOT TARGET osqpstatic)
    message(STATUS "CDDP-cpp: `osqp` targets not found. Attempting to fetch contents...")
    FetchContent_Declare(
    osqp-cpp
    GIT_REPOSITORY https://github.com/google/osqp-cpp
    GIT_TAG master
  )
  FetchContent_MakeAvailable(osqp-cpp)
else()
  message(STATUS "CDDP-cpp: `osqp-cpp` targets found.")
  find_package(osqp-cpp REQUIRED)
endif()

# # Googletest
# if (OSQP-CPP_BUILD_TESTS)
#   enable_testing()
#   if (NOT TARGET gtest OR NOT TARGET gmock OR NOT TARGET gtest_main)
#     message(STATUS "CDDP-cpp: `googletest` targets not found. Attempting to fetch contents...")
#     set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#     FetchContent_Declare(
#       googletest
#       GIT_REPOSITORY https://github.com/google/googletest.git
#       GIT_TAG        origin/main
#     )
#     FetchContent_MakeAvailable(googletest)
#     include(GoogleTest)
#   else()
#     message(STATUS "CDDP-cpp: `googletest` targets found.")
#   endif()
# endif()

# Add the main library
message(STATUS "CDDP-cpp: Adding CDDP-cpp library...")
add_library(cddp src/cddp_core/CDDPProblem.cpp)
target_link_libraries(cddp Eigen3::Eigen osqp-cpp)
target_include_directories(cddp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
message(STATUS "CDDP-cpp: Added CDDP-cpp library.")

# Eigen test (Placeholder for now, will be removed later on)
add_executable(cddp_eigen_test test/test_eigen.cpp)
target_link_libraries(cddp_eigen_test Eigen3::Eigen)


# TEST
add_subdirectory(test)

# Build and register tests.
# if (CDDP-CPP_BUILD_TESTS)
#   message(STATUS "CDDP-cpp: Adding CDDP-cpp tests...")
#   add_executable(osqp_test test/osqp++_test.cc)
#   target_link_libraries(cddp_test gtest gmock gtest_main absl::status absl::span cddp-cpp)
#   gtest_discover_tests(cddp_test)
#   message(STATUS "CDDP-cpp: Added CDDP-cpp tests.")
# endif()
