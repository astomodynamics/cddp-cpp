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

#include "cddp_core/objective.hpp"
#include "cddp_core/helper.hpp"

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
    const std::vector<Eigen::VectorXd> &reference_states,
    double timestep)
    : Q_(Q), R_(R), Qf_(Qf), reference_state_(reference_state), reference_states_(reference_states), timestep_(timestep) {

    // NOTE: Scale Q and R by timestep (Continuous-time fomulation)
    Q_ = Q * timestep_;
    R_ = R * timestep_;
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
        if ((reference_states_.back() - reference_state_).norm() > 1e-6)
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
double QuadraticObjective::evaluate(const std::vector<Eigen::VectorXd> &states, const std::vector<Eigen::VectorXd> &controls) const
{
    double total_cost = 0.0;
    // Compute running cost for all time steps
    for (int t = 0; t < states.size() - 1; ++t)
    {
        total_cost += running_cost(states[t], controls[t], t);
        
    }
    total_cost += terminal_cost(states.back());
    return total_cost;
}

// Evaluate the running cost: (x - x_ref)^T Q (x - x_ref) +  u^T R u
double QuadraticObjective::running_cost(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    Eigen::VectorXd state_error;
    if (reference_states_.size() > 0)
    {
        state_error = state - reference_states_[index];
    }
    else
    {
        state_error = state - reference_state_; // Otherwise, use reference_state_
    }
    return ((state_error.transpose() * Q_ * state_error).value() + (control.transpose() * R_ * control).value());
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
        state_error = state - reference_states_[index];
    }
    else
    {
        state_error = state - reference_state_; // Otherwise, use reference_state_
    }
    return 2.0 * Q_ * state_error;
}

// Gradient of the running cost w.r.t control
Eigen::VectorXd QuadraticObjective::getRunningCostControlGradient(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return 2.0 * R_ * control;
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
    return 2.0 * Q_;
}

Eigen::MatrixXd QuadraticObjective::getRunningCostControlHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return 2.0 * R_;
}

Eigen::MatrixXd QuadraticObjective::getRunningCostCrossHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, int index) const
{
    return Eigen::MatrixXd::Zero(control.size(), state.size());
}

// Hessian of the final cost
Eigen::MatrixXd QuadraticObjective::getFinalCostHessian(const Eigen::VectorXd &final_state) const
{
    return 2.0 * Qf_;
}

NonlinearObjective::NonlinearObjective(double timestep) 
    : timestep_(timestep) {
}

double NonlinearObjective::evaluate(
    const std::vector<Eigen::VectorXd>& states,
    const std::vector<Eigen::VectorXd>& controls) const {
    
    double total_cost = 0.0;
    
    // Sum running costs
    for (size_t t = 0; t < controls.size(); ++t) {
        total_cost += running_cost(states[t], controls[t], t);
    }
    
    // Add terminal cost
    total_cost += terminal_cost(states.back());
    
    return total_cost;
}

double NonlinearObjective::running_cost(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    int index) const {
    // Derived classes should override this
    return 0.0;
}

double NonlinearObjective::terminal_cost(
    const Eigen::VectorXd& final_state) const {
    // Derived classes should override this
    return 0.0;
}

Eigen::VectorXd NonlinearObjective::getRunningCostStateGradient(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    int index) const {
    
    auto cost_func = [this, &control, index](const Eigen::VectorXd& s) {
        return running_cost(s, control, index);
    };
    
    return finite_difference_gradient(cost_func, state);
}

Eigen::VectorXd NonlinearObjective::getRunningCostControlGradient(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    int index) const {
    
    auto cost_func = [this, &state, index](const Eigen::VectorXd& c) {
        return running_cost(state, c, index);
    };
    
    return finite_difference_gradient(cost_func, control);
}

Eigen::VectorXd NonlinearObjective::getFinalCostGradient(
    const Eigen::VectorXd& final_state) const {
    
    auto cost_func = [this](const Eigen::VectorXd& s) {
        return terminal_cost(s);
    };
    
    return finite_difference_gradient(cost_func, final_state);
}

Eigen::MatrixXd NonlinearObjective::getRunningCostStateHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    int index) const {
    
    auto cost_func = [this, &control, index](const Eigen::VectorXd& s) {
        return running_cost(s, control, index);
    };
    
    return finite_difference_hessian(cost_func, state);
}

Eigen::MatrixXd NonlinearObjective::getRunningCostControlHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    int index) const {
    
    auto cost_func = [this, &state, index](const Eigen::VectorXd& c) {
        return running_cost(state, c, index);
    };
    
    return finite_difference_hessian(cost_func, control);
}

Eigen::MatrixXd NonlinearObjective::getRunningCostCrossHessian(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control,
    int index) const {
    
    const int ns = state.size();
    const int nc = control.size();
    Eigen::MatrixXd cross(nc, ns);
    const double h = 2e-8;
    
    Eigen::VectorXd s_plus, s_minus, c_plus, c_minus;
    
    for (int i = 0; i < nc; ++i) {
        for (int j = 0; j < ns; ++j) {
            s_plus = state;  s_plus(j) += h;
            s_minus = state; s_minus(j) -= h;
            c_plus = control; c_plus(i) += h;
            c_minus = control; c_minus(i) -= h;
            
            double f_pp = running_cost(s_plus, c_plus, index);
            double f_pm = running_cost(s_plus, c_minus, index);
            double f_mp = running_cost(s_minus, c_plus, index);
            double f_mm = running_cost(s_minus, c_minus, index);
            
            cross(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
        }
    }
    
    return cross;
}

Eigen::MatrixXd NonlinearObjective::getFinalCostHessian(
    const Eigen::VectorXd& final_state) const {
    
    auto cost_func = [this](const Eigen::VectorXd& s) {
        return terminal_cost(s);
    };
    
    return finite_difference_hessian(cost_func, final_state, 2e-5);
}

} // namespace cddp