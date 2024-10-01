// GOOGLE TEST FRAMEWORK

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "CDDP.hpp"

namespace cddp {

class DubinsCarTest : public ::testing::Test {
protected:
    DubinsCarTest() {
        state_dim = 3;
        control_dim = 2;
        timestep = 0.1;
        integration_type = 0;
        dubins_car = std::make_shared<DubinsCar>(state_dim, control_dim, timestep, integration_type);
    }

    int state_dim;
    int control_dim;
    double timestep;
    int integration_type;
    std::shared_ptr<DubinsCar> dubins_car;
};

TEST_F(DubinsCarTest, Dynamics) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(state_dim);
    Eigen::VectorXd control = Eigen::VectorXd::Zero(control_dim);
    state << 0, 0, 0;
    control << 1, 1;
    Eigen::VectorXd state_dot = dubins_car->dynamics(state, control);
    Eigen::VectorXd expected_state_dot = Eigen::VectorXd::Zero(state_dim);
    expected_state_dot << 1, 0, 1;
    ASSERT_EQ(state_dot, expected_state_dot);
}


} // namespace cddp