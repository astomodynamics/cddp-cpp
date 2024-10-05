cmake_minimum_required(VERSION 3.14) 
project(cddp)

set(CMAKE_CXX_STANDARD 17) # Enforce C++17 as the minimum standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enforce C++17 as the minimum standard
set(ABSL_PROPAGATE_CXX_STD ON) # Enforce C++17 for absl

# Options.
option(CDDP-CPP_BUILD_TESTS "Whether to build tests." ON)
option(CDDP-CPP_GUROBI "Whether to use Gurobi solver." OFF)
option(GUROBI_ROOT "Path to Gurobi installation" "") 

# Find packages
find_package(Eigen3 REQUIRED) 
find_package(PythonLibs 3.10 REQUIRED) 

# Enable FetchContent for downloading dependencies
include(FetchContent)

# ABSL
FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG        origin/master
)
FetchContent_MakeAvailable(abseil-cpp) 

# OSQP
FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/oxfordcontrol/osqp.git
  GIT_TAG        v0.6.0  # Consider using a more recent version if applicable
)
FetchContent_MakeAvailable(osqp)

# OSQP-CPP
FetchContent_Declare(
  osqp-cpp
  GIT_REPOSITORY https://github.com/google/osqp-cpp
  GIT_TAG master
)
FetchContent_MakeAvailable(osqp-cpp)

# Googletest
if (CDDP-CPP_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        origin/main
  )
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

# Include python directories for matplotlibcpp
include_directories(/usr/include/python3.10)

# Include directories
include_directories(include)

# Add your library
add_library(${PROJECT_NAME} 
    src/cddp_core/dynamical_system.cpp
    src/cddp_core/objective.cpp
    # src/cddp_core/cddp_core.cc
    src/dynamics_model/pendulum.cpp
)

target_link_libraries(${PROJECT_NAME} 
    Eigen3::Eigen 
    osqp-cpp 
    ${PYTHON_LIBRARIES}
) 

# Gurobi 
if (CDDP-CPP_GUROBI)
  if (NOT GUROBI_ROOT)
    message(FATAL_ERROR "Please set GUROBI_ROOT.")
  endif()
  set(GUROBI_INCLUDE_DIRS ${GUROBI_ROOT}/include) # Set the path to the Gurobi include directory
  set(GUROBI_LIBRARIES ${GUROBI_ROOT}/lib/libgurobi_c++.a) # Set the path to the Gurobi library
  find_library(GUROBI_LIBRARY gurobi_c++ PATHS ${GUROBI_ROOT}/lib)
  find_path(GUROBI_INCLUDE_DIR gurobi_c++.h PATHS ${GUROBI_ROOT}/include)
  link_directories(${GUROBI_ROOT}/lib) 

  if (GUROBI_LIBRARIES AND GUROBI_INCLUDE_DIR)
    message(STATUS "Found Gurobi: ${GUROBI_LIBRARIES} ${GUROBI_LIBRARY}")
    message(STATUS "Gurobi include directory: ${GUROBI_INCLUDE_DIR}")
    
    include_directories(${GUROBI_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${GUROBI_LIBRARIES}  gurobi_c++ 
    gurobi110) 
  else()
    message(FATAL_ERROR "Could not find Gurobi. Please set GUROBI_ROOT.")
  endif()
endif()

# Build and register tests.
if (CDDP-CPP_BUILD_TESTS)
  add_subdirectory(tests) 
endif()


# Cmake compile commmand:
# $ mkdir build
# $ cd build
# $ cmake  -DCDDP-CPP_GUROBI=ON -DCDDP-CPP_BUILD_TESTS=ON -DGUROBI_ROOT=/home/tom/.local/lib/gurobi1103/linux64 ..
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


