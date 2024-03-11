#ifndef CDDP_DYNAMICALSYSTEM_HPP
#define CDDP_DYNAMICALSYSTEM_HPP

#include <Eigen/Dense>

namespace cddp {
class DynamicalSystem {
public:
    // State and Control Dimensions
    int state_size;
    int control_size;
    double dt;  // Time step

    DynamicalSystem(int state_dim, int control_dim, double timestep) : 
        state_size(state_dim), control_size(control_dim), dt(timestep) {}

    // Pure virtual methods  
    virtual Eigen::VectorXd getDynamics(const Eigen::VectorXd &state, const Eigen::VectorXd &control) = 0;

    virtual Eigen::MatrixXd getDynamicsJacobian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) = 0;

    virtual Eigen::MatrixXd getDynamicsHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control) = 0;


    virtual Eigen::MatrixXd getTrajectory(const Eigen::VectorXd &initialState, const std::vector<Eigen::VectorXd> &controlSeq, int num_steps) {}; 

    // Optional virtual methods
    virtual void setCostMatrices(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R) {} 
    virtual double calculateCost(const Eigen::VectorXd &state, const Eigen::VectorXd &control) = 0;
    virtual void calculateCostGradietn(const Eigen::VectorXd &state, const Eigen::VectorXd &control, Eigen::VectorXd &cost_gradient) {}

    virtual void calculateCostHessian(const Eigen::VectorXd &state, const Eigen::VectorXd &control, Eigen::MatrixXd &cost_hessian) {}
    
    virtual void setFinalCostMatrix(const Eigen::MatrixXd &Qf) {}  
    virtual double calculateFinalCost(const Eigen::VectorXd &state) = 0;
    virtual void calculateFinalCostGradient(const Eigen::VectorXd &state, Eigen::VectorXd &final_cost_gradient) {}

    virtual void calculateFinalCostHessian(const Eigen::VectorXd &state, Eigen::MatrixXd &final_cost_hessian) {}
    
    virtual void setGoal(const Eigen::VectorXd &goal) {}


};

}  // namespace cddp

#endif // CDDP_DYNAMICALSYSTEM_HPP