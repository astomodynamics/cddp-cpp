cmake_minimum_required(VERSION 3.14) 
project(cddp)

set(CMAKE_CXX_STANDARD 17) # Enforce C++17 as the minimum standard
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

# Set the default build type to Release
set(CMAKE_BUILD_TYPE "Release")

# Options.
option(CDDP-CPP_BUILD_TESTS
  "Whether to build tests." OFF
)

# Enable FetchContent for downloading dependencies
include(FetchContent)

# EIGEN
# Eigen is usually available as a system package, so we use find_package.
if (NOT TARGET Eigen3::Eigen)
  message(STATUS "CDDP-cpp: `Eigen3` targets not found. Attempting to find package...")
  find_package(Eigen3 REQUIRED NO_MODULE)
else()
  message(STATUS "CDDP-cpp: `Eigen3` targets found.")
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
target_link_libraries(cddp PUBLIC Eigen3::Eigen PRIVATE osqp-cpp ${CMAKE_DL_LIBS})
target_include_directories(cddp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
message(STATUS "CDDP-cpp: Added CDDP-cpp library.")

# Eigen test (Placeholder for now, will be removed later on)
add_executable(cddp_eigen_test test/test_eigen.cpp)
target_link_libraries(cddp_eigen_test Eigen3::Eigen)


# TEST
# add_subdirectory(test)

# Build and register tests.
# if (CDDP-CPP_BUILD_TESTS)
#   message(STATUS "CDDP-cpp: Adding CDDP-cpp tests...")
#   add_executable(osqp_test test/osqp++_test.cc)
#   target_link_libraries(cddp_test gtest gmock gtest_main absl::status absl::span cddp-cpp)
#   gtest_discover_tests(cddp_test)
#   message(STATUS "CDDP-cpp: Added CDDP-cpp tests.")
# endif()



# Install
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Default install path" FORCE)

# ... Inside the install section ...
install(TARGETS cddp 
        DESTINATION lib) 

install(DIRECTORY include/cddp 
        DESTINATION include) 
