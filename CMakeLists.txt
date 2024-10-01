cmake_minimum_required(VERSION 3.14) 
project(cddp)

set(CMAKE_CXX_STANDARD 17) # Enforce C++17 as the minimum standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enforce C++17 as the minimum standard
set(ABSL_PROPAGATE_CXX_STD ON) # Enforce C++17 for absl

# Set the default build type to Release
set(CMAKE_BUILD_TYPE "Release")

# Options.
option(CDDP-CPP_BUILD_TESTS
  "Whether to build tests." ON
)

option(CDDP-CPP_GUROBI
  "Whether to use Gurobi solver." ON
)

# EIGEN
find_package(Eigen3 REQUIRED) # for matplotlibcpp
find_package(PythonLibs 3.10 REQUIRED)

# Enable FetchContent for downloading dependencies
include(FetchContent)

# ABSL
if (NOT TARGET absl::strings OR NOT TARGET absl::status OR NOT TARGET absl::span)
  message(STATUS "CDDP-cpp: `absl` targets not found. Attempting to fetch contents...")
  FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        origin/master
  )
  FetchContent_MakeAvailable(abseil-cpp)
else()
  message(STATUS "CDDP-cpp: `absl` targets found.")
endif()


# OSQP
if (NOT TARGET osqpstatic)
  message(STATUS "CDDP-cpp: `osqp` targets not found. Attempting to fetch contents...")
  FetchContent_Declare(
    osqp
    GIT_REPOSITORY https://github.com/oxfordcontrol/osqp.git
    GIT_TAG        v0.6.0
  )
  FetchContent_MakeAvailable(osqp)
else()
  message(STATUS "CDDP-cpp: `osqp` targets found.")
endif()


# OSQP-CPP
if (NOT TARGET osqp-cpp OR NOT TARGET osqpstatic)
    message(STATUS "CDDP-cpp: `osqp-cpp` targets not found. Attempting to fetch contents...")
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

# Googletest
if (CDDP-CPP_BUILD_TESTS)
  enable_testing()
  if (NOT TARGET gtest OR NOT TARGET gmock OR NOT TARGET gtest_main)
    message(STATUS "CDDP-cpp: `googletest` targets not found. Attempting to fetch contents...")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        origin/main
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
  else()
    message(STATUS "CDDP-cpp: `googletest` targets found.")
  endif()
endif()

# gurobi
if (CDDP-CPP_GUROBI)
  set(GUROBI_ROOT /home/tom/.local/lib/gurobi1103/linux64) # Set the path to the Gurobi installation
  set(GUROBI_INCLUDE_DIRS ${GUROBI_ROOT}/include) # Set the path to the Gurobi include directory
  set(GUROBI_LIBRARIES ${GUROBI_ROOT}/lib/libgurobi_c++.a) # Set the path to the Gurobi library
  include_directories(${GUROBI_INCLUDE_DIRS}) # Include the Gurobi include directory
  link_directories(${GUROBI_ROOT}/lib) 
endif()

# Add the include directory
include_directories(include)

# Include python directories for matplotlibcpp
include_directories(/usr/include/python3.10)

# Add your source files
add_library(cddp
    src/cddp_core/dynamical_system.cpp
    # src/cddp_core/objective.cc
    # src/cddp_core/cddp_core.cc
    src/dynamics_model/pendulum.cpp
)

# Link against the gurobi library
if (CDDP-CPP_GUROBI)
  target_link_libraries(cddp Eigen3::Eigen osqp-cpp ${PYTHON_LIBRARIES} ${GUROBI_LIBRARIES} gurobi_c++ gurobi110)
else()
  target_link_libraries(cddp Eigen3::Eigen osqp-cpp ${PYTHON_LIBRARIES})
endif()


# Build and register tests.
if (CDDP-CPP_BUILD_TESTS)
  message(STATUS "cddp-cpp: Adding cddp-cpp tests...")  

  add_subdirectory(tests) # include cmake in tests

  message(STATUS "cddp-cpp: Added cddp-cpp tests.")
endif()


# Cmake compile commmand:
# $ mkdir build
# $ cd build
# $ cmake  -DCMAKE_BUILD_TYPE=Release -DCDDP-CPP_GUROBI=ON -DCDDP-CPP_BUILD_TESTS=ON ..
# $ make -j4
# $ make test


# # Add the main library
# message(STATUS "CDDP-cpp: Adding CDDP-cpp library...")
# add_library(cddp src/cddp_core/CDDPProblem.cpp)
# target_link_libraries(cddp PUBLIC 
#     Eigen3::Eigen 
#     PRIVATE osqp-cpp ${CMAKE_DL_LIBS})
# target_include_directories(cddp PUBLIC 
#     "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" 
#     include/cddp)
# message(STATUS "CDDP-cpp: Added CDDP-cpp library.")

# # Eigen test (Placeholder for now, will be removed later on)
# add_executable(cddp_eigen_test test/test_eigen.cpp)
# target_link_libraries(cddp_eigen_test Eigen3::Eigen)

# # LOCAL TEST
# 



