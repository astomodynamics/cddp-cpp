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

# EIGEN
find_package(Eigen3 REQUIRED)
set(Python_EXECUTABLE /usr/bin/python3.10) # Or the path to your desired Python interpreter
execute_process( # Find the NumPy include directory
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE 
)
find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development NumPy REQUIRED)

# Enable FetchContent for downloading dependencies
include(FetchContent)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Python3_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIR}
) 

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

# Matplotplusplus
if (NOT TARGET matplot)
    message(STATUS "CDDP-cpp: `matplot` targets not found. Attempting to fetch contents...")
    FetchContent_Declare(
      matplotplusplus
      GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
      GIT_TAG origin/master
    ) 
    FetchContent_GetProperties(matplotplusplus)
    if(NOT matplotplusplus_POPULATED)
        FetchContent_Populate(matplotplusplus)
        add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else()
    message(STATUS "CDDP-cpp: `matplot` targets found.")
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

# # Add the include directory
# include_directories(include/cddp)

# Add the main library
message(STATUS "CDDP-cpp: Adding CDDP-cpp library...")
add_library(cddp src/cddp_core/CDDPProblem.cpp)
target_link_libraries(cddp PUBLIC 
    Eigen3::Eigen 
    PRIVATE osqp-cpp ${CMAKE_DL_LIBS})
target_include_directories(cddp PUBLIC 
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" 
    include/cddp)
message(STATUS "CDDP-cpp: Added CDDP-cpp library.")

# Eigen test (Placeholder for now, will be removed later on)
add_executable(cddp_eigen_test test/test_eigen.cpp)
target_link_libraries(cddp_eigen_test Eigen3::Eigen)

# LOCAL TEST
add_subdirectory(test)

