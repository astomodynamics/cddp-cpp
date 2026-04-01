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
# Set policy to support older CMake code
cmake_policy(SET CMP0048 NEW)
# Allow policy version minimum flag to be passed to sub-projects
set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING "Minimum CMake version for policy compatibility")

project(
  cddp
  VERSION 0.1.0
  DESCRIPTION "CDDP: A C++ library for Trajectory Optimization and MPC"
  HOMEPAGE_URL  "https://github.com/astomodynamics/cddp-cpp"
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17) # Enforce C++17 as the minimum standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # Enforce C++17 as the minimum standard
set(ABSL_PROPAGATE_CXX_STD ON) # Enforce C++17 for absl
set(CMAKE_BUILD_TYPE "Release") # Set the build type to Release by default

# Platform-specific settings
if(APPLE)
  # Add macOS-specific compiler and linker flags
  set(CMAKE_MACOSX_RPATH ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  # Use the new linker on macOS 11+
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-ld_classic")
  endif()
endif()

# Options
option(CDDP_CPP_BUILD_TESTS "Whether to build tests." ON)
option(CDDP_CPP_BUILD_EXAMPLES "Whether to build examples." ON)

# CasADi Configuration
option(CDDP_CPP_CASADI "Whether to use CasADi" OFF)

# ACADOS Configuration
# For ACADOS installation, see: https://docs.acados.org/
option(CDDP_CPP_ACADOS "Whether to use ACADOS solver." OFF)
option(ACADOS_ROOT "Path to ACADOS installation" "")
if(APPLE)
  set(ACADOS_ROOT /usr/local/lib/acados)  # macOS default path
else()
  set(ACADOS_ROOT /home/astomodynamics/acados)  # Linux default path
endif()

# Find packages
find_package(Eigen3 QUIET) 

# If Eigen3 is not found, download it
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found. Downloading...")
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0  # Use a stable version
  )
  FetchContent_MakeAvailable(eigen)
  set(EIGEN3_INCLUDE_DIRS ${eigen_SOURCE_DIR})
endif()

if (CDDP_CPP_CASADI)
  # Assuming that CasADi is installed in /usr/local/include/casadi by:
  # https://github.com/casadi/casadi/wiki/InstallationLinux 
  # $ echo "export LD_LIBRARY_PATH=/home/<path_to_casadi>/casadi/build/lib:$LD_LIBRARY_PATH" >> ~/.bashrc && source ~/.bashrc
  # If you use Ipopt, you may need to set flags for CasADi installation.

  find_package(casadi REQUIRED)
  set(CASADI_INCLUDE_DIR /usr/local/include/casadi)
  find_library(CASADI_LIBRARY NAMES casadi HINTS ${CASADI_INCLUDE_DIR}/../lib $ENV{CASADI_PREFIX}/lib)
  set(CASADI_LIBRARIES ${CASADI_LIBRARIES} ${CASADI_LIBRARY})
  if (NOT CASADI_LIBRARIES)
    message(FATAL_ERROR "CasADi library not found. Please set CASADI_PREFIX.")
  else()
    message(STATUS "Found CasADi: ${CASADI_LIBRARIES}")
  endif()
endif()

# Enable FetchContent for downloading dependencies
include(FetchContent)

# autodiff
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "Don't build autodiff tests")
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "Don't build autodiff examples")
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "Don't build autodiff Python bindings")
FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/autodiff/autodiff.git
  GIT_TAG v1.1.2  # Use a stable version tag instead of main
)
FetchContent_MakeAvailable(autodiff)

# Googletest
if (CDDP_CPP_BUILD_TESTS)
  enable_testing()
  set(INSTALL_GTEST OFF CACHE BOOL "Don't install GTest alongside cddp" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        origin/main
  )
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
) 

# Add your library
set(cddp_core_srcs
  src/cddp_core/dynamical_system.cpp
  src/cddp_core/objective.cpp
  src/cddp_core/constraint.cpp
  src/cddp_core/helper.cpp
  src/cddp_core/boxqp.cpp
  src/cddp_core/qp_solver.cpp
  src/cddp_core/cddp_core.cpp
  src/cddp_core/cddp_solver_base.cpp
  src/cddp_core/clddp_solver.cpp
  src/cddp_core/logddp_solver.cpp
  src/cddp_core/ipddp_solver.cpp
  src/cddp_core/msipddp_solver.cpp
)

set(dynamics_model_srcs
  src/dynamics_model/pendulum.cpp
  src/dynamics_model/unicycle.cpp
  src/dynamics_model/bicycle.cpp
  src/dynamics_model/cartpole.cpp
  src/dynamics_model/acrobot.cpp
  src/dynamics_model/car.cpp
  src/dynamics_model/forklift.cpp
  src/dynamics_model/dubins_car.cpp
  src/dynamics_model/quadrotor.cpp
  src/dynamics_model/quadrotor_rate.cpp
  src/dynamics_model/manipulator.cpp
  src/dynamics_model/spacecraft_linear.cpp
  src/dynamics_model/spacecraft_linear_fuel.cpp
  src/dynamics_model/spacecraft_nonlinear.cpp
  src/dynamics_model/dreyfus_rocket.cpp
  src/dynamics_model/spacecraft_landing2d.cpp
  src/dynamics_model/spacecraft_roe.cpp
  src/dynamics_model/lti_system.cpp
  src/dynamics_model/spacecraft_twobody.cpp
  src/dynamics_model/usv_3dof.cpp
  src/dynamics_model/quaternion_attitude.cpp
  src/dynamics_model/mrp_attitude.cpp
  src/dynamics_model/euler_attitude.cpp
)

add_library(${PROJECT_NAME}
  ${cddp_core_srcs}
  ${dynamics_model_srcs}
)

# Enable position-independent code (required for Python shared library linking)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    $<IF:$<BOOL:${Eigen3_FOUND}>,Eigen3::Eigen,>
    autodiff
)

if(NOT Eigen3_FOUND)
  target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cddp-cpp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cddp-cpp>
)

if (CDDP_CPP_CASADI)
  target_include_directories(${PROJECT_NAME} PUBLIC ${CASADI_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CASADI_LIBRARIES})
endif()

# ACADOS
if (CDDP_CPP_ACADOS)
  if (NOT ACADOS_ROOT)
    message(FATAL_ERROR "Please set ACADOS_ROOT to your ACADOS installation directory.")
  endif()
  
  # Set ACADOS paths
  set(ACADOS_INCLUDE_DIRS ${ACADOS_ROOT}/include)
  set(ACADOS_LIB_DIR ${ACADOS_ROOT}/lib)
  set(BLASFEO_INCLUDE_DIR ${ACADOS_ROOT}/include/blasfeo/include)
  set(HPIPM_INCLUDE_DIR ${ACADOS_ROOT}/include/hpipm/include)
  
  # Find ACADOS libraries 
  find_library(ACADOS_LIBRARY NAMES acados PATHS ${ACADOS_LIB_DIR})
  find_library(HPIPM_LIBRARY NAMES hpipm PATHS ${ACADOS_LIB_DIR})
  find_library(BLASFEO_LIBRARY NAMES blasfeo PATHS ${ACADOS_LIB_DIR})
  find_library(QPOASES_E_LIBRARY NAMES qpOASES_e PATHS ${ACADOS_LIB_DIR})
  
  # Find header files
  find_path(ACADOS_C_INCLUDE_DIR acados_c/ocp_nlp_interface.h PATHS ${ACADOS_INCLUDE_DIRS})
  
  # Set up library list
  set(ACADOS_LIBRARIES)
  if(ACADOS_LIBRARY)
    list(APPEND ACADOS_LIBRARIES ${ACADOS_LIBRARY})
  endif()
  if(HPIPM_LIBRARY)
    list(APPEND ACADOS_LIBRARIES ${HPIPM_LIBRARY})
  endif()
  if(BLASFEO_LIBRARY)
    list(APPEND ACADOS_LIBRARIES ${BLASFEO_LIBRARY})
  endif()
  if(QPOASES_E_LIBRARY)
    list(APPEND ACADOS_LIBRARIES ${QPOASES_E_LIBRARY})
  endif()
  
  if (ACADOS_LIBRARIES AND ACADOS_C_INCLUDE_DIR)
    message(STATUS "Found ACADOS libraries: ${ACADOS_LIBRARIES}")
    message(STATUS "ACADOS include directory: ${ACADOS_C_INCLUDE_DIR}")
    
    include_directories(${ACADOS_C_INCLUDE_DIR})
    include_directories(${BLASFEO_INCLUDE_DIR})
    include_directories(${HPIPM_INCLUDE_DIR})
    link_directories(${ACADOS_LIB_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ACADOS_LIBRARIES})
    
    # Add preprocessor definition to enable ACADOS in code
    target_compile_definitions(${PROJECT_NAME} PRIVATE CDDP_CPP_ACADOS_ENABLED=1)
    
    message(STATUS "Successfully linked ACADOS.")
  else()
    message(FATAL_ERROR "Could not find ACADOS libraries. Please check ACADOS_ROOT: ${ACADOS_ROOT}")
  endif()
endif()

# Build and register tests.
if (CDDP_CPP_BUILD_TESTS)
  add_subdirectory(tests) 
endif()

# Build examples
if (CDDP_CPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Python bindings (optional)
option(CDDP_CPP_BUILD_PYTHON "Build Python bindings" OFF)
if(CDDP_CPP_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# Install targets
include(CMakePackageConfigHelpers)

install(TARGETS ${PROJECT_NAME}
  EXPORT cddpTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/cddp-cpp/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cddp-cpp
  FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT cddpTargets
  FILE cddpTargets.cmake
  NAMESPACE cddp::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cddp
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cddpConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cddpConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cddp
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cddpConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cddpConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cddpConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cddp
)
