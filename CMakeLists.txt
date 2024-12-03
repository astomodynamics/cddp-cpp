#  Copyright 2024 Tomo Sasaki

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#       https://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.14) 
project(
  cddp
  VERSION 0.1.0
  DESCRIPTION "CDDP: A C++ library for MPC"
  HOMEPAGE_URL  "https://github.com/astomodynamics/cddp-cpp"
)

set(CMAKE_CXX_STANDARD 17) # Enforce C++17 as the minimum standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enforce C++17 as the minimum standard
set(ABSL_PROPAGATE_CXX_STD ON) # Enforce C++17 for absl
set(CMAKE_BUILD_TYPE "Release") # Set the build type to Release by default

# Options.
option(CDDP-CPP_BUILD_TESTS "Whether to build tests." ON)
option(CDDP-CPP_GUROBI "Whether to use Gurobi solver." OFF)
option(GUROBI_ROOT "Path to Gurobi installation" "") 
set(GUROBI_ROOT /home/tom/.local/lib/gurobi1103/linux64)
set(CDDP-CPP_TORCH "Whether to use LibTorch." ON) # cannot be turned off
option(CDDP-CPP_TORCH_GPU "Whether to use GPU." ON)

# Find packages
find_package(Eigen3 REQUIRED) 
set(Python_EXECUTABLE /usr/bin/python3.10) # Or the path to your desired Python interpreter
execute_process( # Find the NumPy include directory
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE 
)
find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development NumPy REQUIRED)

# Enable FetchContent for downloading dependencies
include(FetchContent)

# OSQP-CPP
FetchContent_Declare(
  osqp-cpp
  GIT_REPOSITORY https://github.com/google/osqp-cpp
  GIT_TAG master
)
FetchContent_MakeAvailable(osqp-cpp)
FetchContent_GetProperties(osqp-cpp)

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

# LibTorch
if (CDDP-CPP_TORCH)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/libtorch/share/cmake/Torch/TorchConfig.cmake)
    message(STATUS "Found LibTorch at ${CMAKE_CURRENT_SOURCE_DIR}/build/libtorch")
  else()
    message(STATUS "Downloading LibTorch...")
    # Download and extract LibTorch
    if (CDDP-CPP_TORCH_GPU)
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip")
    else()
      set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip")
    endif()

    # Set the download directory
    set(DOWLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)

    # Create the download directory
    file(MAKE_DIRECTORY ${DOWLOAD_DIR})

    # Download the file 
    file(DOWNLOAD ${LIBTORCH_URL} ${DOWLOAD_DIR}/libtorch-shared-with-deps-latest.zip
      STATUS DOWNLOAD_STATUS
      SHOW_PROGRESS
    )

    # Check if the download was successful
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_STATUS_CODE)
    if(NOT DOWNLOAD_STATUS_CODE EQUAL 0)
      message(FATAL_ERROR "Failed to download LibTorch.")
    endif()
    
    # Extract the file    
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xvf ${DOWLOAD_DIR}/libtorch-shared-with-deps-latest.zip
      WORKING_DIRECTORY ${DOWLOAD_DIR}
    )

    # Remove the zip file
    file(REMOVE ${DOWLOAD_DIR}/libtorch-shared-with-deps-latest.zip)
  endif()

  # Set the path to the LibTorch installation
  set(LIBTORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/libtorch)

  find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR} NO_DEFAULT_PATH)
  message(STATUS "Found LibTorch: ${TORCH_LIBRARIES}")
endif()


# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Python3_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIR}
) 

# Add your library
set(cddp_core_srcs
  src/cddp_core/dynamical_system.cpp
  src/cddp_core/objective.cpp
  src/cddp_core/constraint.cpp
  src/cddp_core/cddp_core.cpp
  src/cddp_core/torch_dynamical_system.cpp
)

set(dynamics_model_srcs
  src/dynamics_model/pendulum.cpp
  src/dynamics_model/dubins_car.cpp
  src/dynamics_model/bicycle.cpp
  src/dynamics_model/cartpole.cpp
  src/dynamics_model/quadrotor.cpp
  src/dynamics_model/manipulator.cpp
  src/dynamics_model/spacecraft_linear.cpp
)

add_library(${PROJECT_NAME} 
  ${cddp_core_srcs}
  ${dynamics_model_srcs}
)

target_link_libraries(${PROJECT_NAME} 
  Eigen3::Eigen 
  osqp-cpp 
  Python3::Python
  Python3::Module
  Python3::NumPy
  ${TORCH_LIBRARIES}
) 

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cddp-cpp>
  $<INSTALL_INTERFACE:include> 
  ${TORCH_INCLUDE_DIRS}
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

# Build examples
add_subdirectory(examples)

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


