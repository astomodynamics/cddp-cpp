# add_executable(cddp_pendulum cddp_pendulum.cpp)
# target_link_libraries(cddp_pendulum cddp Eigen3::Eigen)

add_executable(cddp_dubins_car cddp_dubins_car.cpp)
target_link_libraries(cddp_dubins_car cddp Eigen3::Eigen)

add_executable(cddp_bicycle cddp_bicycle.cpp)
target_link_libraries(cddp_bicycle cddp Eigen3::Eigen)

add_executable(cddp_pendulum cddp_pendulum.cpp)
target_link_libraries(cddp_pendulum cddp Eigen3::Eigen)

add_executable(cddp_cartpole cddp_cartpole.cpp)
target_link_libraries(cddp_cartpole cddp Eigen3::Eigen)

add_executable(cddp_quadrotor cddp_quadrotor.cpp)
target_link_libraries(cddp_quadrotor cddp Eigen3::Eigen)

add_executable(cddp_manipulator cddp_manipulator.cpp)
target_link_libraries(cddp_manipulator cddp Eigen3::Eigen)