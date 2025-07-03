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
#include <string>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cddp.hpp"

// This test is useful for creating a new solver along with the corresponding API.

namespace cddp {

/**
 * @brief Mock external solver for testing external solver registration
 */
class MockExternalSolver : public ISolverAlgorithm {
public:
    MockExternalSolver() = default;
    virtual ~MockExternalSolver() = default;

    void initialize(CDDP &context) override {
        initialized_ = true;
        context_ = &context;
    }

    CDDPSolution solve(CDDP &context) override {
        solve_called_ = true;
        
        CDDPSolution solution;
        solution["solver_name"] = getSolverName();
        solution["status_message"] = std::string("OptimalSolutionFound");
        solution["iterations_completed"] = 5;
        solution["solve_time_ms"] = 100.0;
        solution["final_objective"] = 1.23;
        solution["final_step_length"] = 1.0;
        
        // Create simple trajectories that match the actual format
        std::vector<double> time_points;
        time_points.reserve(static_cast<size_t>(context.getHorizon() + 1));
        for (int t = 0; t <= context.getHorizon(); ++t) {
            time_points.push_back(t * context.getTimestep());
        }
        
        std::vector<Eigen::VectorXd> state_trajectory;
        std::vector<Eigen::VectorXd> control_trajectory;
        state_trajectory.reserve(static_cast<size_t>(context.getHorizon() + 1));
        control_trajectory.reserve(static_cast<size_t>(context.getHorizon()));
        
        for (int k = 0; k <= context.getHorizon(); ++k) {
            state_trajectory.push_back(Eigen::VectorXd::Zero(context.getStateDim()));
        }
        for (int k = 0; k < context.getHorizon(); ++k) {
            control_trajectory.push_back(Eigen::VectorXd::Zero(context.getControlDim()));
        }
        
        solution["time_points"] = time_points;
        solution["state_trajectory"] = state_trajectory;
        solution["control_trajectory"] = control_trajectory;
        
        return solution;
    }

    std::string getSolverName() const override { 
        return "MockExternalSolver"; 
    }

    // Test accessors
    bool wasInitialized() const { return initialized_; }
    bool wasSolveCalled() const { return solve_called_; }
    CDDP* getContext() const { return context_; }

private:
    bool initialized_ = false;
    bool solve_called_ = false;
    CDDP* context_ = nullptr;
};

/**
 * @brief Another mock solver to test multiple solver registration
 */
class AnotherMockSolver : public ISolverAlgorithm {
public:
    void initialize(CDDP &context) override {}
    
    CDDPSolution solve(CDDP &context) override {
        CDDPSolution solution;
        solution["solver_name"] = getSolverName();
        solution["status_message"] = std::string("MaxIterationsReached");
        solution["iterations_completed"] = 10;
        solution["solve_time_ms"] = 200.0;
        solution["final_objective"] = 4.56;
        solution["final_step_length"] = 0.5;
        
        // Empty trajectories for simplicity
        solution["time_points"] = std::vector<double>();
        solution["state_trajectory"] = std::vector<Eigen::VectorXd>();
        solution["control_trajectory"] = std::vector<Eigen::VectorXd>();
        
        return solution;
    }
    
    std::string getSolverName() const override { 
        return "AnotherMockSolver"; 
    }
};

// Factory functions for the mock solvers
std::unique_ptr<ISolverAlgorithm> createMockExternalSolver() {
    return std::make_unique<MockExternalSolver>();
}

std::unique_ptr<ISolverAlgorithm> createAnotherMockSolver() {
    return std::make_unique<AnotherMockSolver>();
}

} // namespace cddp

// Test fixture for CDDP core functionality
class CDDPCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Problem parameters
        state_dim = 3;
        control_dim = 2;
        horizon = 10;
        timestep = 0.1;
        
        // Create initial and goal states
        initial_state = Eigen::VectorXd::Zero(state_dim);
        initial_state << 0.0, 0.0, 0.0;
        
        goal_state = Eigen::VectorXd::Zero(state_dim);
        goal_state << 1.0, 1.0, M_PI/2.0;
        
        // Create basic system and objective
        system = std::make_unique<cddp::Unicycle>(timestep, "euler");
        
        Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim);
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(control_dim, control_dim);
        Eigen::MatrixXd Qf = 10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim);
        
        std::vector<Eigen::VectorXd> empty_reference_states;
        objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);
        
        // Create options with reduced verbosity for tests
        options = cddp::CDDPOptions();
        options.max_iterations = 5;
        options.verbose = false;
        options.debug = false;
    }
    
    void TearDown() override {
        // Clean up any registered solvers to avoid cross-test interference
        // Note: In a real implementation, you might want a clearRegisteredSolvers() method
    }
    
    // Test data
    int state_dim, control_dim, horizon;
    double timestep;
    Eigen::VectorXd initial_state, goal_state;
    std::unique_ptr<cddp::DynamicalSystem> system;
    std::unique_ptr<cddp::Objective> objective;
    cddp::CDDPOptions options;
};

// Test external solver registration functionality
TEST_F(CDDPCoreTest, ExternalSolverRegistration) {
    // Test registering a new solver
    cddp::CDDP::registerSolver("MockExternalSolver", cddp::createMockExternalSolver);
    
    // Test that the solver is now registered
    EXPECT_TRUE(cddp::CDDP::isSolverRegistered("MockExternalSolver"));
    EXPECT_FALSE(cddp::CDDP::isSolverRegistered("NonExistentSolver"));
    
    // Test getting registered solvers
    auto registered_solvers = cddp::CDDP::getRegisteredSolvers();
    EXPECT_THAT(registered_solvers, ::testing::Contains("MockExternalSolver"));
}

// Test multiple solver registration
TEST_F(CDDPCoreTest, MultipleSolverRegistration) {
    // Register multiple solvers
    cddp::CDDP::registerSolver("MockSolver1", cddp::createMockExternalSolver);
    cddp::CDDP::registerSolver("MockSolver2", cddp::createAnotherMockSolver);
    
    // Test that both are registered
    EXPECT_TRUE(cddp::CDDP::isSolverRegistered("MockSolver1"));
    EXPECT_TRUE(cddp::CDDP::isSolverRegistered("MockSolver2"));
    
    // Test getting all registered solvers
    auto registered_solvers = cddp::CDDP::getRegisteredSolvers();
    EXPECT_THAT(registered_solvers, ::testing::Contains("MockSolver1"));
    EXPECT_THAT(registered_solvers, ::testing::Contains("MockSolver2"));
    EXPECT_GE(registered_solvers.size(), 2);
}

// Test using registered external solver
TEST_F(CDDPCoreTest, UseRegisteredExternalSolver) {
    // Register the mock solver
    cddp::CDDP::registerSolver("MockExternalSolver", cddp::createMockExternalSolver);
    
    // Create CDDP instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::make_unique<cddp::Unicycle>(timestep, "euler"),
                          std::make_unique<cddp::QuadraticObjective>(
                              Eigen::MatrixXd::Identity(state_dim, state_dim),
                              Eigen::MatrixXd::Identity(control_dim, control_dim),
                              10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim),
                              goal_state, std::vector<Eigen::VectorXd>(), timestep),
                          options);
    
    // Solve using the registered external solver
    auto solution = cddp_solver.solve("MockExternalSolver");
    
    // Verify the solution came from our mock solver
    ASSERT_TRUE(solution.count("solver_name"));
    EXPECT_EQ(std::any_cast<std::string>(solution["solver_name"]), "MockExternalSolver");
    
    ASSERT_TRUE(solution.count("status_message"));
    EXPECT_EQ(std::any_cast<std::string>(solution["status_message"]), "OptimalSolutionFound");
    
    ASSERT_TRUE(solution.count("iterations_completed"));
    EXPECT_EQ(std::any_cast<int>(solution["iterations_completed"]), 5);
    
    ASSERT_TRUE(solution.count("final_objective"));
    EXPECT_DOUBLE_EQ(std::any_cast<double>(solution["final_objective"]), 1.23);
}

// Test built-in solver still works
TEST_F(CDDPCoreTest, BuiltInSolverStillWorks) {
    // Create CDDP instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::make_unique<cddp::Unicycle>(timestep, "euler"),
                          std::make_unique<cddp::QuadraticObjective>(
                              Eigen::MatrixXd::Identity(state_dim, state_dim),
                              Eigen::MatrixXd::Identity(control_dim, control_dim),
                              10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim),
                              goal_state, std::vector<Eigen::VectorXd>(), timestep),
                          options);
    
    // Solve using built-in CLDDP solver
    auto solution = cddp_solver.solve("CLDDP");
    
    // Verify we get a valid solution (might not converge in 5 iterations, but should run)
    ASSERT_TRUE(solution.count("solver_name"));
    EXPECT_EQ(std::any_cast<std::string>(solution["solver_name"]), "CLDDP");
    
    ASSERT_TRUE(solution.count("status_message"));
    // Should have a valid status message
    std::string status = std::any_cast<std::string>(solution["status_message"]);
    EXPECT_FALSE(status.empty());
}

// Test error handling for unknown solver
TEST_F(CDDPCoreTest, UnknownSolverErrorHandling) {
    // Create CDDP instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::make_unique<cddp::Unicycle>(timestep, "euler"),
                          std::make_unique<cddp::QuadraticObjective>(
                              Eigen::MatrixXd::Identity(state_dim, state_dim),
                              Eigen::MatrixXd::Identity(control_dim, control_dim),
                              10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim),
                              goal_state, std::vector<Eigen::VectorXd>(), timestep),
                          options);
    
    // Try to solve with unknown solver
    auto solution = cddp_solver.solve("NonExistentSolver");
    
    // Verify we get an appropriate error response
    ASSERT_TRUE(solution.count("solver_name"));
    EXPECT_EQ(std::any_cast<std::string>(solution["solver_name"]), "NonExistentSolver");
    
    ASSERT_TRUE(solution.count("status_message"));
    std::string status = std::any_cast<std::string>(solution["status_message"]);
    EXPECT_THAT(status, ::testing::HasSubstr("UnknownSolver"));
    EXPECT_THAT(status, ::testing::HasSubstr("NonExistentSolver"));
    
    ASSERT_TRUE(solution.count("iterations_completed"));
    EXPECT_EQ(std::any_cast<int>(solution["iterations_completed"]), 0);
}

// Test solver precedence (external over built-in)
TEST_F(CDDPCoreTest, SolverPrecedence) {
    // Register a solver with the same name as a built-in solver
    cddp::CDDP::registerSolver("CLDDP", cddp::createMockExternalSolver);
    
    // Create CDDP instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::make_unique<cddp::Unicycle>(timestep, "euler"),
                          std::make_unique<cddp::QuadraticObjective>(
                              Eigen::MatrixXd::Identity(state_dim, state_dim),
                              Eigen::MatrixXd::Identity(control_dim, control_dim),
                              10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim),
                              goal_state, std::vector<Eigen::VectorXd>(), timestep),
                          options);
    
    // Solve - should use external solver (registered first)
    auto solution = cddp_solver.solve("CLDDP");
    
    // Verify we got the external solver (MockExternalSolver), not built-in CLDDP
    ASSERT_TRUE(solution.count("solver_name"));
    EXPECT_EQ(std::any_cast<std::string>(solution["solver_name"]), "MockExternalSolver");
    
    ASSERT_TRUE(solution.count("final_objective"));
    EXPECT_DOUBLE_EQ(std::any_cast<double>(solution["final_objective"]), 1.23); // Mock solver value
}

// Test enum-based solve still works
TEST_F(CDDPCoreTest, EnumBasedSolveStillWorks) {
    // Create CDDP instance
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::make_unique<cddp::Unicycle>(timestep, "euler"),
                          std::make_unique<cddp::QuadraticObjective>(
                              Eigen::MatrixXd::Identity(state_dim, state_dim),
                              Eigen::MatrixXd::Identity(control_dim, control_dim),
                              10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim),
                              goal_state, std::vector<Eigen::VectorXd>(), timestep),
                          options);
    
    // Solve using enum-based interface
    auto solution = cddp_solver.solve(cddp::SolverType::CLDDP);
    
    // Verify we get a valid solution
    ASSERT_TRUE(solution.count("solver_name"));
    // Note: If we registered "CLDDP" above, this might be "MockExternalSolver"
    // But the enum interface should still work
    std::string solver_name = std::any_cast<std::string>(solution["solver_name"]);
    EXPECT_FALSE(solver_name.empty());
}

// Test integration with trajectory and options
TEST_F(CDDPCoreTest, IntegrationWithTrajectoryAndOptions) {
    // Register mock solver
    cddp::CDDP::registerSolver("IntegrationTestSolver", cddp::createMockExternalSolver);
    
    // Create CDDP instance with custom options
    cddp::CDDPOptions custom_options;
    custom_options.max_iterations = 20;
    custom_options.tolerance = 1e-6;
    custom_options.verbose = false;
    
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::make_unique<cddp::Unicycle>(timestep, "euler"),
                          std::make_unique<cddp::QuadraticObjective>(
                              Eigen::MatrixXd::Identity(state_dim, state_dim),
                              Eigen::MatrixXd::Identity(control_dim, control_dim),
                              10.0 * Eigen::MatrixXd::Identity(state_dim, state_dim),
                              goal_state, std::vector<Eigen::VectorXd>(), timestep),
                          custom_options);
    
    // Set initial trajectory
    std::vector<Eigen::VectorXd> X(horizon + 1, Eigen::VectorXd::Zero(state_dim));
    std::vector<Eigen::VectorXd> U(horizon, Eigen::VectorXd::Zero(control_dim));
    X[0] = initial_state;
    cddp_solver.setInitialTrajectory(X, U);
    
    // Add a constraint
    Eigen::VectorXd control_upper_bound = Eigen::VectorXd::Ones(control_dim) * 2.0;
    cddp_solver.addPathConstraint("TestConstraint", 
                              std::make_unique<cddp::ControlConstraint>(control_upper_bound));
    
    // Solve
    auto solution = cddp_solver.solve("IntegrationTestSolver");
    
    // Verify solution structure is correct
    ASSERT_TRUE(solution.count("solver_name"));
    EXPECT_EQ(std::any_cast<std::string>(solution["solver_name"]), "MockExternalSolver");
    
    ASSERT_TRUE(solution.count("time_points"));
    auto time_points = std::any_cast<std::vector<double>>(solution["time_points"]);
    EXPECT_EQ(time_points.size(), horizon + 1);
    
    ASSERT_TRUE(solution.count("state_trajectory"));
    auto state_traj = std::any_cast<std::vector<Eigen::VectorXd>>(solution["state_trajectory"]);
    EXPECT_EQ(state_traj.size(), horizon + 1);
    
    ASSERT_TRUE(solution.count("control_trajectory"));
    auto control_traj = std::any_cast<std::vector<Eigen::VectorXd>>(solution["control_trajectory"]);
    EXPECT_EQ(control_traj.size(), horizon);
} 