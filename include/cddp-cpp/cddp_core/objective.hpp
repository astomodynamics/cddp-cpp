#ifndef CDDP_OBJECTIVE_HPP
#define CDDP_OBJECTIVE_HPP

#include <Eigen/Dense>

namespace cddp {
class Objective {
public:
    // Constructor 
    Objective() {} 

    // Core objective function (running cost): l(x_t, u_t) 
    virtual double evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Final/terminal cost: lf(x_T)
    virtual double terminal_cost(const Eigen::VectorXd& final_state) const = 0;

    // Gradient of the running cost w.r.t state: dl/dx
    virtual Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Gradient of the running cost w.r.t control: dl/du
    virtual Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Gradients of the running cost w.r.t state and control
    virtual std::tuple<Eigen::VectorXd, Eigen::VectorXd> getRunningCostGradients(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
        return { getRunningCostStateGradient(state, control), getRunningCostControlGradient(state, control) };
    }

    // Gradient of the final cost w.r.t state: dlf/dx
    virtual Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const = 0;

    // Hessian of the running cost w.r.t state: d^2l/dx^2
    virtual Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Hessian of the running cost w.r.t control: d^2l/du^2
    virtual Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Hessian of the running cost w.r.t state and control: d^2l/dxdu
    virtual Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const = 0;

    // Hessians of the running cost
    virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> getRunningCostHessians(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const {
        return { getRunningCostStateHessian(state, control), getRunningCostControlHessian(state, control), getRunningCostCrossHessian(state, control) };
    }

    // Hessian of the final cost w.r.t state: d^2lf/dx^2
    virtual Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const = 0;
};

class QuadraticObjective : public Objective {
public:
    // Constructor
    QuadraticObjective(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf, 
                       const Eigen::VectorXd& reference_state);

    // Evaluate the running cost: 0.5 * (x - x_ref)^T Q (x - x_ref) + 0.5 * u^T R u 
    double evaluate(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Evaluate the final/terminal cost: 0.5 * (x_T - x_ref)^T Qf (x_T - x_ref)
    double terminal_cost(const Eigen::VectorXd& final_state) const override;

    // Gradient of the running cost w.r.t state
    Eigen::VectorXd getRunningCostStateGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Gradient of the running cost w.r.t control
    Eigen::VectorXd getRunningCostControlGradient(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Gradient of the final cost w.r.t state
    Eigen::VectorXd getFinalCostGradient(const Eigen::VectorXd& final_state) const override;

    // Hessians of the running cost (constant for quadratic objectives)
    Eigen::MatrixXd getRunningCostStateHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
    Eigen::MatrixXd getRunningCostControlHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;
    Eigen::MatrixXd getRunningCostCrossHessian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) const override;

    // Hessian of the final cost
    Eigen::MatrixXd getFinalCostHessian(const Eigen::VectorXd& final_state) const override;

    // Accessors
    const Eigen::MatrixXd& getQ() const { return Q_; }
    const Eigen::MatrixXd& getR() const { return R_; }
    const Eigen::MatrixXd& getQf() const { return Qf_; }
    const Eigen::VectorXd& getReferenceState() const { return reference_state_; }

    // Setters
    void setQ(const Eigen::MatrixXd& Q) { Q_ = Q; }
    void setR(const Eigen::MatrixXd& R) { R_ = R; }
    void setQf(const Eigen::MatrixXd& Qf) { Qf_ = Qf; }
    void setReferenceState(const Eigen::VectorXd& reference_state) { reference_state_ = reference_state; }

private:
    Eigen::MatrixXd Q_, R_, Qf_;      // Weight matrices for state, control, and final state
    Eigen::VectorXd reference_state_; // Reference/target state
};

// TODO: Implement the QuadraticTrackingObjective class

} // namespace cddp

#endif // CDDP_OBJECTIVE_HPP