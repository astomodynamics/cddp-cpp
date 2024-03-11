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
# include_directories(Eigen3::Eigen)

# Find the OSQP library
# find_package(OSQP REQUIRED)

# find_package(OSQP-eigen REUIRED)

# Create the executable
add_executable(cddp test/test_eigen.cpp)

# Link the Eigen3 library to the executable
target_link_libraries(cddp Eigen3::Eigen)