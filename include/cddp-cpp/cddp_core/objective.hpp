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

#ifndef CDDP_OBJECTIVE_HPP
#define CDDP_OBJECTIVE_HPP

#include <Eigen/Dense>

namespace cddp {
class Objective {
public:
    // Constructor 
    Objective() {} 

    // Core objective function (total cost)
    virtual double evaluate(const std::vector<Eigen::VectorXd> &states, const std::vector<Eigen::VectorXd> &controls) const = 0;

    // Running cost: l(x, u)
    virtual double running_cost(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;

    // Final/terminal cost: lf(x_T)
    virtual double terminal_cost(const Eigen::VectorXd& final_state) const = 0;

    // Gradient of the running cost w.r.t state: dl/dx
    virtual Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;

    // Gradient of the running cost w.r.t control: dl/du
    virtual Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;

    // Gradients of the running cost w.r.t state and control
    virtual std::tuple<Eigen::VectorXd, Eigen::VectorXd> getRunningCostGradients(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const {
        return { getRunningCostStateGradient(state, control, index), getRunningCostControlGradient(state, control, index) };
    }

    // Gradient of the final cost w.r.t state: dlf/dx
    virtual Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const = 0;

    // Hessian of the running cost w.r.t state: d^2l/dx^2
    virtual Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;

    // Hessian of the running cost w.r.t control: d^2l/du^2
    virtual Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;

    // Hessian of the running cost w.r.t state and control: d^2l/dxdu
    virtual Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const = 0;

    // Hessians of the running cost
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getRunningCostHessians(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const {
        return { getRunningCostStateHessian(state, control, index), getRunningCostControlHessian(state, control, index), getRunningCostCrossHessian(state, control, index) };
    }

    // Hessian of the final cost w.r.t state: d^2lf/dx^2
    virtual Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const = 0;

    // Accessors
    virtual const Eigen::VectorXd& getReferenceState() const { return Eigen::VectorXd::Zero(0); }
    virtual const std::vector<Eigen::VectorXd>& getReferenceStates() const { return std::vector<Eigen::VectorXd>(); }

    // Setters
    virtual void setReferenceState(const Eigen::VectorXd& reference_state) {}
    virtual void setReferenceStates(const std::vector<Eigen::VectorXd>& reference_states) {}
};

class QuadraticObjective : public Objective {
public:
    // Constructor
    QuadraticObjective(const Eigen::MatrixXd& Q, 
                       const Eigen::MatrixXd& R, 
                       const Eigen::MatrixXd& Qf, 
                       const Eigen::VectorXd& reference_state = Eigen::VectorXd::Zero(0), // Default to empty vector
                       const std::vector<Eigen::VectorXd>& reference_states = std::vector<Eigen::VectorXd>(), // Default to empty vector
                       double timestep = 0.1);

    // Evaluate the total cost: terminal cost + running cost
    double evaluate(const std::vector<Eigen::VectorXd> &states, const std::vector<Eigen::VectorXd> &controls) const override;

    // Evaluate the running cost: (x - x_ref)^T Q (x - x_ref) +  u^T R u
    double running_cost(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const override;

    // Evaluate the final/terminal cost: (x_T - x_ref)^T Qf (x_T - x_ref)
    double terminal_cost(const Eigen::VectorXd& final_state) const override;

    // Gradient of the running cost w.r.t state
    Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const override;

    // Gradient of the running cost w.r.t control
    Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const override;

    // Gradient of the final cost w.r.t state
    Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const override;

    // Hessians of the running cost (constant for quadratic objectives)
    Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const override;
    Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const override;
    Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control, int index) const override;

    // Hessian of the final cost
    Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const override;

    // Accessors
    const Eigen::MatrixXd& getQ() const { return Q_; }
    const Eigen::MatrixXd& getR() const { return R_; }
    const Eigen::MatrixXd& getQf() const { return Qf_; }
    const Eigen::VectorXd& getReferenceState() const override{ return reference_state_; }
    const std::vector<Eigen::VectorXd>& getReferenceStates() const override { return reference_states_; }

    // Setters
    void setQ(const Eigen::MatrixXd& Q) { Q_ = Q * timestep_; }
    void setR(const Eigen::MatrixXd& R) { R_ = R * timestep_; }
    void setQf(const Eigen::MatrixXd& Qf) { Qf_ = Qf; }
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }
    void setReferenceStates(const std::vector<Eigen::VectorXd>& reference_states) { reference_states_ = reference_states; }

private:
    Eigen::MatrixXd Q_, R_, Qf_;      // Weight matrices for state, control, and final state
    Eigen::VectorXd reference_state_; // Reference/target state
    std::vector<Eigen::VectorXd> reference_states_; // Reference/target states 
    double timestep_;                 // Timestep
};

/**
 * @brief NonlinearObjective class for general nonlinear cost functions
 * 
 * This class implements a general nonlinear objective function with finite difference
 * methods for computing gradients and Hessians. It extends the base Objective class
 * and provides utilities for numerical differentiation.
 */
class NonlinearObjective : public Objective {
public:
    /**
     * @brief Constructor for NonlinearObjective
     * @param timestep Time step for cost scaling
     */
    NonlinearObjective(double timestep = 0.1);

    /**
     * @brief Evaluate total cost over trajectory
     * @param states Vector of states
     * @param controls Vector of controls
     * @return Total cost over trajectory
     */
    double evaluate(const std::vector<Eigen::VectorXd>& states,
                   const std::vector<Eigen::VectorXd>& controls) const override;

    /**
     * @brief Evaluate running cost at a single timestep
     * @param state Current state
     * @param control Current control
     * @param index Time index
     * @return Running cost value
     */
    double running_cost(const Eigen::VectorXd& state,
                       const Eigen::VectorXd& control,
                       int index) const override;

    /**
     * @brief Evaluate terminal cost
     * @param final_state Final state
     * @return Terminal cost value
     */
    double terminal_cost(const Eigen::VectorXd& final_state) const override;

    /**
     * @brief Get gradient of running cost w.r.t state using finite differences
     * @param state Current state
     * @param control Current control
     * @param index Time index
     * @return Gradient vector w.r.t state
     */
    Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control,
                                               int index) const override;

    /**
     * @brief Get gradient of running cost w.r.t control using finite differences
     * @param state Current state
     * @param control Current control
     * @param index Time index
     * @return Gradient vector w.r.t control
     */
    Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state,
                                                 const Eigen::VectorXd& control,
                                                 int index) const override;

    /**
     * @brief Get gradient of terminal cost using finite differences
     * @param final_state Final state
     * @return Gradient vector of terminal cost
     */
    Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const override;

    /**
     * @brief Get Hessian of running cost w.r.t state using finite differences
     * @param state Current state
     * @param control Current control
     * @param index Time index
     * @return Hessian matrix w.r.t state
     */
    Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state,
                                              const Eigen::VectorXd& control,
                                              int index) const override;

    /**
     * @brief Get Hessian of running cost w.r.t control using finite differences
     * @param state Current state
     * @param control Current control
     * @param index Time index
     * @return Hessian matrix w.r.t control
     */
    Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& control,
                                                int index) const override;

    /**
     * @brief Get cross-term Hessian of running cost using finite differences
     * @param state Current state
     * @param control Current control
     * @param index Time index
     * @return Cross-term Hessian matrix
     */
    Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state,
                                              const Eigen::VectorXd& control,
                                              int index) const override;

    /**
     * @brief Get Hessian of terminal cost using finite differences
     * @param final_state Final state
     * @return Hessian matrix of terminal cost
     */
    Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const override;

private:
    double timestep_;  ///< Time step for cost scaling
};

} // namespace cddp

#endif // CDDP_OBJECTIVE_HPP