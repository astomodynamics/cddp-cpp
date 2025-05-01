/*
 Copyright 2025 Tomo Sasaki

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
#ifndef CDDP_PARAMETERIZED_OBJECTIVE_HPP
#define CDDP_PARAMETERIZED_OBJECTIVE_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept> // For std::logic_error
#include "cddp_core/objective.hpp"

namespace cddp {

class ParameterizedObjective : public Objective {
public:
    // Constructor
    ParameterizedObjective(int parameter_dim)
        : Objective(), parameter_dim_(parameter_dim) {}

    virtual ~ParameterizedObjective() = default; // Virtual destructor

    // --- Parameterized Cost Functions ---
    // These now accept an additional parameter vector.
    // Derived classes MUST override these to implement parameter-dependent costs.

    // Parameterized running cost: l(x, u, p, t)
    virtual double running_cost(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               const Eigen::VectorXd& parameter,
                               int index) const {
        throw std::logic_error("running_cost with parameter must be overridden in the derived parameterized class.");
    }

    // Parameterized final/terminal cost: lf(x_T, p)
    virtual double terminal_cost(const Eigen::VectorXd& final_state,
                                const Eigen::VectorXd& parameter) const {
        throw std::logic_error("terminal_cost with parameter must be overridden in the derived parameterized class.");
    }

    // --- Parameterized Derivative Functions ---
    // Derivatives now also depend on the parameter.

    // Gradient of running cost w.r.t state: dl/dx (p)
    virtual Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state,
                                                        const Eigen::VectorXd& control,
                                                        const Eigen::VectorXd& parameter,
                                                        int index) const {
        throw std::logic_error("getRunningCostStateGradient with parameter must be overridden.");
    }

    // Gradient of running cost w.r.t control: dl/du (p)
    virtual Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state,
                                                          const Eigen::VectorXd& control,
                                                          const Eigen::VectorXd& parameter,
                                                          int index) const {
        throw std::logic_error("getRunningCostControlGradient with parameter must be overridden.");
    }

    // Gradient of running cost w.r.t parameter: dl/dp
    virtual Eigen::VectorXd getRunningCostParameterGradient(const Eigen::VectorXd& state,
                                                             const Eigen::VectorXd& control,
                                                             const Eigen::VectorXd& parameter,
                                                             int index) const {
        throw std::logic_error("getRunningCostParameterGradient must be overridden.");
    }


    // Gradient of final cost w.r.t state: dlf/dx (p)
    virtual Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state,
                                                 const Eigen::VectorXd& parameter) const {
         throw std::logic_error("getFinalCostGradient with parameter must be overridden.");
    }

    // Gradient of final cost w.r.t parameter: dlf/dp
     virtual Eigen::VectorXd getFinalCostParameterGradient(const Eigen::VectorXd& final_state,
                                                          const Eigen::VectorXd& parameter) const {
         throw std::logic_error("getFinalCostParameterGradient must be overridden.");
    }


    // --- Hessians (Add later if needed) ---
    // Hessians involving parameters (d^2l/dxdp, d^2l/dudp, d^2l/dpdp, etc.)
    // d^2lf/dxdp, d^2lf/dpdp

    // Accessor methods
    int getParameterDim() const { return parameter_dim_; }


    // --- Hide or Override Non-Parameterized Base Class Functions ---
    // Similar to the dynamics class, we might want to hide or override base functions.
    // Example:
    /*
    virtual double running_cost(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control,
                               int index) const override {
        throw std::logic_error("Use parameterized version of running_cost.");
    }
    // ... similar overrides ...
    */


protected:
    int parameter_dim_;
};

class ParameterizedQuadraticObjective : public ParameterizedObjective {
public:
    // Constructor
    ParameterizedQuadraticObjective(
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& Qf,
        int parameter_dim, // Should match state dimension as parameter defines final target
        const Eigen::VectorXd& reference_state, // Default reference for running cost
        const std::vector<Eigen::VectorXd>& reference_states = std::vector<Eigen::VectorXd>(), // Optional running reference trajectory
        double timestep = 0.1);


    // --- Parameterized Cost Functions ---

    // Parameterized running cost: l(x, u, p, t)
    // Uses reference_states_[index] if available, otherwise uses reference_state_.
    double running_cost(const Eigen::VectorXd& state,
                        const Eigen::VectorXd& control,
                        const Eigen::VectorXd& parameter,
                        // Note: parameter argument is ignored in this function for this class
                        int index) const override;

    // Parameterized final/terminal cost: lf(x_T, p)
    // Uses parameter as the final reference state.
    double terminal_cost(const Eigen::VectorXd& final_state,
                         const Eigen::VectorXd& parameter) const override;


    // --- Parameterized Derivative Functions ---

    // Gradient of running cost w.r.t state: dl/dx (p)
    Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state,
                                                const Eigen::VectorXd& control,
                                                const Eigen::VectorXd& parameter,
                                                // Note: parameter argument is ignored
                                                int index) const override;


    // Gradient of running cost w.r.t control: dl/du (p)
    Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state,
                                                  const Eigen::VectorXd& control,
                                                  const Eigen::VectorXd& parameter,
                                                   // Note: parameter argument is ignored
                                                  int index) const override;


    // Gradient of running cost w.r.t parameter: dl/dp
    Eigen::VectorXd getRunningCostParameterGradient(const Eigen::VectorXd& state,
                                                     const Eigen::VectorXd& control,
                                                     const Eigen::VectorXd& parameter,
                                                     // Returns zero for this class
                                                     int index) const override;

    // Gradient of final cost w.r.t state: dlf/dx (p)
    Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state,
                                         const Eigen::VectorXd& parameter) const override;

    // Gradient of final cost w.r.t parameter: dlf/dp
    Eigen::VectorXd getFinalCostParameterGradient(const Eigen::VectorXd& final_state,
                                                   const Eigen::VectorXd& parameter) const override;

    // --- Hessians --- 
    // These override the pure virtual functions from the base Objective class.
    // Note: They ignore the 'parameter' argument as Hessians are constant for quadratic cost.
    Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control,
                                               int index) const override;

    Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state,
                                                 const Eigen::VectorXd& control,
                                                 int index) const override;

    Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control,
                                               int index) const override;

    Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const override;


    // Accessors
    const Eigen::MatrixXd& getQ() const { return Q_; }
    const Eigen::MatrixXd& getR() const { return R_; }
    const Eigen::MatrixXd& getQf() const { return Qf_; }    
    Eigen::VectorXd getReferenceState() const override { return reference_state_; }
    std::vector<Eigen::VectorXd> getReferenceStates() const override { return reference_states_; }

    // Setters
    void setQ(const Eigen::MatrixXd& Q) { Q_ = Q * timestep_; }
    void setR(const Eigen::MatrixXd& R) { R_ = R * timestep_; }
    void setQf(const Eigen::MatrixXd& Qf) { Qf_ = Qf; }
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }
    void setReferenceStates(const std::vector<Eigen::VectorXd>& reference_states) { reference_states_ = reference_states; }


    // --- Overrides for base Objective pure virtual functions (Non-Parameterized) ---
    // These are implemented to throw, encouraging use of parameterized versions.
    double evaluate(const std::vector<Eigen::VectorXd>& states, 
                    const std::vector<Eigen::VectorXd>& controls) const override;
    double running_cost(const Eigen::VectorXd& state, 
                        const Eigen::VectorXd& control, 
                        int index) const override;
    double terminal_cost(const Eigen::VectorXd& final_state) const override;
    Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state, 
                                              const Eigen::VectorXd& control, 
                                              int index) const override;
    Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state, 
                                                const Eigen::VectorXd& control, 
                                                int index) const override;
    Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const override;


private:
    Eigen::MatrixXd Q_, R_, Qf_;                // Weight matrices (Q, R scaled by timestep)
    Eigen::VectorXd reference_state_;           // Default reference state for running cost
    std::vector<Eigen::VectorXd> reference_states_; // Optional running reference trajectory
    double timestep_;                           // Timestep for scaling
};

} // namespace cddp

#endif // CDDP_PARAMETERIZED_OBJECTIVE_HPP 