find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found, fetching...")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_pycddp_core
  src/main.cpp
  src/bind_options.cpp
  src/bind_dynamics.cpp
  src/bind_objective.cpp
  src/bind_constraints.cpp
  src/bind_solver.cpp
)

target_link_libraries(_pycddp_core PRIVATE cddp)
target_include_directories(_pycddp_core PRIVATE
  ${CMAKE_SOURCE_DIR}/include/cddp-cpp
)

install(TARGETS _pycddp_core DESTINATION pycddp)
