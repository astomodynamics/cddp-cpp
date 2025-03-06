/*
 Copyright 2024 Tomo Sasaki

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include "gtest/gtest.h"
#include "cddp.hpp"

namespace cddp {

class CarParkingObjective : public NonlinearObjective {
public:
    CarParkingObjective(const Eigen::VectorXd& goal_state, double timestep)
        : NonlinearObjective(timestep), reference_state_(goal_state) {
        cu_ = Eigen::Vector2d(1e-2, 1e-4);
        cf_ = Eigen::Vector4d(0.1, 0.1, 1.0, 0.3);
        pf_ = Eigen::Vector4d(0.01, 0.01, 0.01, 1.0);
        cx_ = Eigen::Vector2d(1e-3, 1e-3);
        px_ = Eigen::Vector2d(0.1, 0.1);
    }

    double running_cost(const Eigen::VectorXd& state, 
                        const Eigen::VectorXd& control, 
                        int index) const override {
        double lu = cu_.dot(control.array().square().matrix());
        Eigen::VectorXd xy_state = state.head(2);
        double lx = cx_.dot(sabs(xy_state, px_));
        return lu + lx;
    }

    double terminal_cost(const Eigen::VectorXd& final_state) const override {
        return cf_.dot(sabs(final_state, pf_)) + running_cost(final_state, Eigen::VectorXd::Zero(2), 0);
    }

private:
    // Smooth absolute value function 
    Eigen::VectorXd sabs(const Eigen::VectorXd& x, const Eigen::VectorXd& p) const {
        return ((x.array().square() / p.array().square() + 1.0).sqrt() * p.array() - p.array()).matrix();
    }

    Eigen::VectorXd reference_state_;
    Eigen::Vector2d cu_;
    Eigen::Vector4d cf_;
    Eigen::Vector4d pf_;
    Eigen::Vector2d cx_;
    Eigen::Vector2d px_;
};

} // namespace cddp


TEST(IPDDPCarTest, PlotAndSaveGif) {
    // Problem parameters
    const int state_dim = 4;     // [x, y, theta, v]
    const int control_dim = 2;   // [wheel_angle, acceleration]
    const int horizon = 500;
    const double timestep = 0.03;
    const std::string integration_type = "euler";

    // Create a Car instance with given parameters
    double wheelbase = 2.0;
    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<cddp::Car>(timestep, wheelbase, integration_type);

    // Define initial and goal states
    Eigen::VectorXd initial_state(state_dim);
    initial_state << 1.0, 1.0, 1.5 * M_PI, 0.0;
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0, 0.0, 0.0;  // Desired parking state

    // Create the nonlinear objective for car parking
    auto objective = std::make_unique<cddp::CarParkingObjective>(goal_state, timestep);

    // Create CDDP solver for the car model
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep);
    cddp_solver.setDynamicalSystem(std::move(system));
    cddp_solver.setObjective(std::move(objective));

    // Define control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -0.5, -2.0;  // [steering_angle, acceleration]
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 0.5, 2.0;
    cddp_solver.addConstraint("ControlConstraint", std::make_unique<cddp::ControlConstraint>(control_upper_bound));

    // Set solver options
    cddp::CDDPOptions options;
    options.max_iterations = 600;
    options.verbose = false;       // Disable verbose output for the test
    options.cost_tolerance = 1e-7;
    options.grad_tolerance = 1e-4;
    options.regularization_type = "none";
    options.debug = false;
    options.use_parallel = false;
    options.num_threads = 1;
    options.barrier_coeff = 1e-1;
    cddp_solver.setOptions(options);

    // Initialize the trajectory with zero controls
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    for (int i = 0; i < horizon; ++i) {
        U[i](0) = 0.0;
        U[i](1) = 0.0;
        X[i + 1] = cddp_solver.getSystem().getDiscreteDynamics(X[i], U[i]);
    }
    cddp_solver.setInitialTrajectory(X, U);

    // Solve the problem using IPDDP
    cddp::CDDPSolution solution = cddp_solver.solve("IPDDP");

    // Verify that the solution converged.
    EXPECT_TRUE(solution.converged);
}
