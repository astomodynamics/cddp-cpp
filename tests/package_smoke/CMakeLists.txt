cmake_minimum_required(VERSION 3.14)

project(cddp_package_smoke LANGUAGES CXX)

find_package(cddp REQUIRED)

add_executable(cddp_package_smoke main.cpp)
target_link_libraries(cddp_package_smoke PRIVATE cddp::cddp)
