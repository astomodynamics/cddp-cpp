cmake_minimum_required(VERSION 3.0) 
project(ddp_cpp)

# Find the Eigen3 library
find_package(Eigen3 3.3 REQUIRED)

# Create the executable
add_executable(ddp_cpp test/test_eigen.cpp)


# Link the Eigen3 library to the executable
target_link_libraries(ddp_cpp Eigen3::Eigen)