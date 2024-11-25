# add_executable(cddp_pendulum cddp_pendulum.cpp)
# target_link_libraries(cddp_pendulum cddp Eigen3::Eigen)

add_executable(cddp_dubins_car cddp_dubins_car.cpp)
target_link_libraries(cddp_dubins_car cddp Eigen3::Eigen)

add_executable(cddp_bicycle cddp_bicycle.cpp)
target_link_libraries(cddp_bicycle cddp Eigen3::Eigen)