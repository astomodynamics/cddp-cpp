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
#include <Eigen/Dense>

#include "cddp-cpp/cddp_core/objective.hpp"

namespace cddp
{

//----------------------------------------------------------------------------------------------------
// QuadraticObjective
//----------------------------------------------------------------------------------------------------

// Constructor
QuadraticObjective::QuadraticObjective(
    const Eigen::MatrixXd &Q,
    const Eigen::MatrixXd &R,
    const Eigen::MatrixXd &Qf,
    const Eigen::VectorXd &reference_state,
    const Eigen::MatrixXd &reference_states,
    double timestep)
    : Q_(Q), R_(R), Qf_(Qf), reference_state_(reference_state), reference_states_(reference_states), timestep_(timestep) {
    // Check dimensions
    if (Q.rows() != Q.cols())
    {
        std::cerr << "QuadraticObjective: Q matrix must be square" << std::endl;
        throw std::invalid_argument("Q matrix must be square");
    }
    if (R.rows() != R.cols())
    {
        std::cerr << "QuadraticObjective: R matrix must be square" << std::endl;
        throw std::invalid_argument("R matrix must be square");
    }
    if (Qf.rows() != Qf.cols())
    {
        std::cerr << "QuadraticObjective: Qf matrix must be square" << std::endl;
        throw std::invalid_argument("Qf matrix must be square");
    }

    // Check the last reference state is same as the reference state
    if (reference_states_.size() > 0)
    {
        if ((reference_states_.col(reference_states_.cols() - 1) - reference_state_).norm() > 1e-6)
        {
            std::cerr << "QuadraticObjective: Last reference state must be same as the reference state" << std::endl;
            throw std::invalid_argument("Last reference state must be same as the reference state");
        }
    }
    
    // print to tell single reference state or multiple reference states
    if (reference_states_.size() > 0)
    {
        std::cout << "QuadraticObjective: Using multiple reference states" << std::endl;
    }
    else
    {
        std::cout << "QuadraticObjective: Using single reference state" << std::endl;
    }
}

// Evaluate the total cost: terminal cost + running cost
double QuadraticObjective::evaluate(const Eigen::MatrixXd &states, const Eigen::MatrixXd &controls) const
{
    double total_cost = 0.0;
    // Compute running cost for all time steps
    for (int t = 0; t < states.cols() - 1; ++t)
    {
        total_cost += running_cost(states.col(t), controls.col(t), t);
        
    }
    total_cost += terminal_cost(states.col(states.cols() - 1));
    return total_cost;
}

// Evaluate the running cost: (x - x_ref)^T Q (x - x_ref) +  u^T R u
double QuadraticObjective::running_cost(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    Eigen::VectorXd state_error;
    if (reference_states_.size() > 0)
    {
        state_error = state - reference_states_.col(index);
    }
    else
    {
        state_error = state - reference_state_; // Otherwise, use reference_state_
    }
    return ((state_error.transpose() * Q_ * state_error).value() + (control.transpose() * R_ * control).value()) * timestep_;
}

// Evaluate the final/terminal cost: (x_T - x_ref)^T Qf (x_T - x_ref)
double QuadraticObjective::terminal_cost(const Eigen::VectorXd &final_state) const
{
    Eigen::VectorXd state_error = final_state - reference_state_;
    return (state_error.transpose() * Qf_ * state_error).value();
}

// Gradient of the running cost w.r.t state
Eigen::VectorXd QuadraticObjective::getRunningCostStateGradient(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    Eigen::VectorXd state_error;
    if (reference_states_.size() > 0)
    {
        state_error = state - reference_states_.row(index);
    }
    else
    {
        state_error = state - reference_state_; // Otherwise, use reference_state_
    }
    return 2.0 * Q_ * state_error * timestep_;
}

// Gradient of the running cost w.r.t control
Eigen::VectorXd QuadraticObjective::getRunningCostControlGradient(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return 2.0 * R_ * control * timestep_;
}

// Gradient of the final cost w.r.t state
Eigen::VectorXd QuadraticObjective::getFinalCostGradient(const Eigen::VectorXd &final_state) const
{
    Eigen::VectorXd state_error = final_state - reference_state_;
    return 2.0 * Qf_ * state_error;
}

// Hessians of the running cost (constant for quadratic objectives)
Eigen::MatrixXd QuadraticObjective::getRunningCostStateHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return 2.0 * Q_ * timestep_;
}

Eigen::MatrixXd QuadraticObjective::getRunningCostControlHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return 2.0 * R_ * timestep_;
}

Eigen::MatrixXd QuadraticObjective::getRunningCostCrossHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return Eigen::MatrixXd::Zero(state.size(), control.size());
}

// Hessian of the final cost
Eigen::MatrixXd QuadraticObjective::getFinalCostHessian(const Eigen::VectorXd &final_state) const
{
    return 2.0 * Qf_;
}

} // namespace cddp