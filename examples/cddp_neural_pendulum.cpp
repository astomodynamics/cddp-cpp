/*
 Copyright 2024 Tomo

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

#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <vector>

#include "cddp.hpp"

namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

struct ODEFuncImpl : public torch::nn::Module {
    ODEFuncImpl(int64_t hidden_dim=32) {
        net = register_module("net", torch::nn::Sequential(
            torch::nn::Linear(/*in_features=*/2, hidden_dim),
            torch::nn::Tanh(),
            torch::nn::Linear(hidden_dim, hidden_dim),
            torch::nn::Tanh(),
            torch::nn::Linear(hidden_dim, 2)
        ));
    }
    // forward(t, y) -> dy/dt
    torch::Tensor forward(const torch::Tensor &t, const torch::Tensor &y) {
        return net->forward(y);
    }
    torch::nn::Sequential net;
};
TORCH_MODULE(ODEFunc);

static torch::Tensor rk4_step(
    ODEFunc &func,
    const torch::Tensor &t,
    const torch::Tensor &y,
    double dt
) {
    auto half_dt = dt * 0.5;
    auto k1 = func->forward(t, y);
    auto k2 = func->forward(t + half_dt, y + half_dt * k1);
    auto k3 = func->forward(t + half_dt, y + half_dt * k2);
    auto k4 = func->forward(t + dt,      y + dt * k3);
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

struct NeuralODEImpl : public torch::nn::Module {
    NeuralODEImpl(int64_t hidden_dim=32) {
        func_ = register_module("func", ODEFunc(hidden_dim));
    }
    torch::Tensor forward(const torch::Tensor &y0,
                          const torch::Tensor &t,
                          double dt)
    {
        int64_t batch_size = y0.size(0);
        int64_t steps      = t.size(0);

        // shape: [B, steps, 2]
        auto trajectory = torch::zeros({batch_size, steps, 2},
            torch::TensorOptions().device(y0.device()).dtype(y0.dtype()));

        // first step
        trajectory.select(1, 0) = y0.clone();
        auto state = y0.clone();

        for (int64_t i = 0; i < steps - 1; ++i) {
            auto t_i = t[i];
            state = rk4_step(func_, t_i, state, dt);
            trajectory.select(1, i+1) = state;
        }
        return trajectory;
    }

    torch::Tensor step_once(const torch::Tensor &y, double dt) {
        // We'll treat 't' as 0.0 for the step
        auto t_0 = torch::tensor(0.0, y.options());
        return rk4_step(func_, t_0, y, dt);
    }

    ODEFunc func_;
};
TORCH_MODULE(NeuralODE);

class NeuralPendulum : public cddp::DynamicalSystem {
public:
    NeuralPendulum(const std::string& model_file, double dt, int64_t hidden_dim=32)
        : dt_(dt)
    {
        // create and load
        neural_ode_ = std::make_shared<NeuralODE>(hidden_dim);
        torch::load(neural_ode_, model_file);
        neural_ode_->eval();
        device_ = torch::kCPU; // keep everything CPU for simplicity
    }

    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &control) override
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        auto y_in = torch::from_blob(
            const_cast<double*>(state.data()),
            {1, 2},
            torch::TensorOptions().dtype(torch::kFloat64)
        ).clone().to(options);


        auto y_next = neural_ode_->step_once(y_in, dt_);
        auto y_next_cpu = y_next.to(torch::kCPU);
        auto data_ptr = y_next_cpu.data_ptr<float>();

        Eigen::VectorXd x_next(2);
        x_next << static_cast<double>(data_ptr[0]), static_cast<double>(data_ptr[1]);

        return x_next;
    }

    std::unique_ptr<DynamicalSystem> clone() const override {
        throw std::runtime_error("NeuralPendulum clone not implemented.");
    }

private:
    std::shared_ptr<NeuralODE> neural_ode_;
    torch::Device device_;
    double dt_;
};


int main()
{
    int state_dim = 2;
    int control_dim = 1;
    int horizon = 100;
    double timestep = 0.02;

    std::string model_file = "../examples/neural_dynamics/neural_models/neural_pendulum.pth";

    std::unique_ptr<cddp::DynamicalSystem> system =
        std::make_unique<NeuralPendulum>(model_file, timestep /*dt*/);

    // Cost matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(control_dim, control_dim);
    Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Qf << 100.0, 0.0,
          0.0, 100.0;

    // Goal state = (0, 0) upright
    Eigen::VectorXd goal_state(state_dim);
    goal_state << 0.0, 0.0;

    // We have no "reference" states, so pass an empty vector
    std::vector<Eigen::VectorXd> empty_reference_states;
    auto objective = std::make_unique<cddp::QuadraticObjective>(Q, R, Qf, goal_state, empty_reference_states, timestep);
    
    // Initial state (pendulum pointing down, or any)
    Eigen::VectorXd initial_state(state_dim);
    initial_state << M_PI, 0.0;  // (theta=pi, dot=0)

    // Construct zero control sequence
    std::vector<Eigen::VectorXd> zero_control_sequence(horizon, Eigen::VectorXd::Zero(control_dim));
    
    // Construct initial trajectory
    std::vector<Eigen::VectorXd> X_init(horizon + 1, initial_state);

    // Solver options
    cddp::CDDPOptions options;
    options.max_iterations = 20;
    options.regularization.type = "none";
    options.regularization.control = 1e-7;

    // Create CDDP solver with new API
    cddp::CDDP cddp_solver(initial_state, goal_state, horizon, timestep,
                          std::move(system), std::move(objective), options);

    // Control constraints
    Eigen::VectorXd control_lower_bound(control_dim);
    control_lower_bound << -10.0;    // clamp torque
    Eigen::VectorXd control_upper_bound(control_dim);
    control_upper_bound << 10.0; 
    cddp_solver.addPathConstraint("ControlBoxConstraint", 
        std::make_unique<cddp::ControlBoxConstraint>(control_lower_bound, control_upper_bound));

    // Set initial guess
    cddp_solver.setInitialTrajectory(X_init, zero_control_sequence);

    // Solve
    cddp::CDDPSolution solution = cddp_solver.solve(cddp::SolverType::ASDDP);

    auto X_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("state_trajectory"));
    auto U_sol = std::any_cast<std::vector<Eigen::VectorXd>>(solution.at("control_trajectory"));
    auto t_sol = std::any_cast<std::vector<double>>(solution.at("time_points"));

    // Create a directory for plots
    const std::string plotDirectory = "../results/tests_neural";
    if (!fs::exists(plotDirectory)) {
        fs::create_directories(plotDirectory);
    }

    // Extract solution data for plotting
    std::vector<double> theta_arr, theta_dot_arr, torque_arr;
    for (auto &x : X_sol) {
        theta_arr.push_back(x(0));
        theta_dot_arr.push_back(x(1));
    }
    for (auto &u : U_sol) {
        torque_arr.push_back(u(0));
    }

    // Plot
    plt::figure();
    plt::subplot(2, 1, 1);
    plt::named_plot("Theta", theta_arr);
    plt::named_plot("ThetaDot", theta_dot_arr);
    plt::title("Neural Pendulum State Trajectory");
    plt::legend();

    plt::subplot(2, 1, 2);
    plt::named_plot("Torque", torque_arr);
    plt::title("Control Input");
    plt::legend();

    std::string plot_file = plotDirectory + "/neural_pendulum_cddp.png";
    plt::save(plot_file);
    std::cout << "Saved plot: " << plot_file << std::endl;

    return 0;
}
