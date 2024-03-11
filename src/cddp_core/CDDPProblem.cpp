#include "cddp_core/CDDPProblem.hpp" 

namespace cddp {

// Constructor 
CDDPProblem::CDDPProblem(cddp::DynamicalSystem* system, const Eigen::VectorXd& initialState, int horizon, double timestep) :
    dynamics(system), 
    initial_state(initialState), 
    goal_state(initialState), // Pre-allocate goal to initialState 
    horizon(horizon),
    dt(timestep),
    X(system->state_size, horizon + 1), // Pre-allocate trajectories
    U(system->control_size, horizon)
{
    // Initialize Cost Matrices (assuming dimensions)
    Q.resize(system->state_size, system->state_size);
    R.resize(system->control_size, system->control_size);
    Qf.resize(system->state_size, system->state_size);
    
    // Initialize Value Function Matrices (assuming dimensions)
    Q_UU.resize(horizon, Eigen::MatrixXd::Zero(system->control_size, system->control_size));
    Q_UX.resize(horizon, Eigen::MatrixXd::Zero(system->control_size, system->state_size));
    Q_U.resize(horizon, Eigen::VectorXd::Zero(system->control_size));
}

// Setup Methods
void CDDPProblem::setInitialState(const Eigen::VectorXd& x0) {
    initial_state = x0;
    X.col(0) = x0; // Update initial trajectory state
}

void CDDPProblem::setGoalState(const Eigen::VectorXd& goal) {
    goal_state = goal;
}

void CDDPProblem::setHorizon(int T) {
    horizon = T;
    
    // Resize X, U, Q_UU, Q_UX, Q_U accordingly 
    X.resize(dynamics->state_size, horizon + 1); 
    U.resize(dynamics->control_size, horizon);

    Q_UU.resize(horizon);
    Q_UX.resize(horizon);
    Q_U.resize(horizon);
}

void CDDPProblem::setTimeStep(double timestep) {
    dt = timestep;
}

void CDDPProblem::setCostMatrices(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf) {
    this->Q = Q;
    this->R = R;
    this->Qf = Qf;
}

void CDDPProblem::setOptions(const CDDPOptions& opts) {
    options = opts;
}

// Solve (Outline)
std::vector<Eigen::VectorXd> CDDPProblem::solve() {
    // ... CDDP Iteration Loop ...
    // return U.col(0); 



}

// // Backward Pass (Outline)
// void CDDPProblem::solveBackwardPass() {
//     Eigen::MatrixXd A = dynamics->getDynamicsHessian(x_trajectories.col(horizon), Eigen::VectorXd::Zero(dynamics->control_size)); // Assuming zero control for final time step Hessian
//     Eigen::VectorXd b = dynamics->getDynamicsJacobian(x_trajectories.col(horizon), Eigen::VectorXd::Zero(dynamics->control_size)).transpose() * (Q_f * (x_trajectories.col(horizon) - goal_state)); // Assuming zero control for final time step gradient 

//     for (int i = horizon - 1; i >= 0; --i) {
//         const Eigen::VectorXd&  x = x_trajectories.col(i);
//         const Eigen::VectorXd&  u = u_trajectories.col(i);

//         // Q function calculations (Replace with your system-specific expressions)
//         Eigen::MatrixXd Q_xx = dynamics->calculateCostHessian(x, u) + dynamics->getDynamicsJacobian(x, u).transpose() * A * dynamics->getDynamicsJacobian(x, u);
//         Eigen::MatrixXd Q_ux = dynamics->getDynamicsJacobian(x, u).transpose() * A; 
//         Eigen::VectorXd Q_x = dynamics->calculateCostGradient(x, u) + dynamics->getDynamicsJacobian(x, u).transpose() * b;
//         // ... (Similar for Q_uu, Q_u) ...

//         // Feedback Gain Calculation (Simplified for unconstrained)
//         Eigen::MatrixXd K = -Q_uu.inverse() * Q_ux;
//         Eigen::VectorXd k = -Q_uu.inverse() * Q_u;

//         // Store Value Function Matrices
//         Q_UU[i] = Q_uu;
//         Q_UX[i] = Q_ux;
//         Q_U[i] = Q_u;

//         // Update A and b (Backward propagation)
//         A = Q_xx + K.transpose() * Q_uu * K + K.transpose() * Q_ux + Q_ux.transpose() * K;
//         b = Q_x + K.transpose() * Q_uu * k + K.transpose() * Q_u + Q_ux.transpose() * k;
//     }
// }

// // Forward Pass (Unconstrained)
// void CDDPProblem::solveForwardPass() {
//     Eigen::VectorXd x = initial_state;

//     for (int i = 0; i < horizon; ++i) {
//         Eigen::VectorXd delta_x = x - x_trajectories.col(i);
//         Eigen::VectorXd delta_u =  K * delta_x + k; // Simplified calculation for unconstrained

//         u_trajectories.col(i) += delta_u;       // Update control (might add a line search here later)
//         x = dynamics->getDynamics(x, u_trajectories.col(i)); // Simulate forward 
//         x_trajectories.col(i + 1) = x;          // Update trajectory
//     }
// }
} // namespace cddp 