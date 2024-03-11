#include <iostream>
#include "cddp_core/CDDPProblem.hpp" 


namespace cddp {

// Constructor 
CDDPProblem::CDDPProblem(cddp::DynamicalSystem* system, const Eigen::VectorXd& initialState, int horizon, double timestep) :
    dynamics(system), 
    initial_state(initialState), 
    goal_state(initialState), // Pre-allocate goal to initialState 
    horizon(horizon),
    dt(timestep),
    X(system->state_size_, horizon + 1), // Pre-allocate trajectories
    U(system->control_size_, horizon)
{
    // Initialize Cost Matrices (assuming dimensions)
    Q.resize(system->state_size_, system->state_size_);
    R.resize(system->control_size_, system->control_size_);
    Qf.resize(system->state_size_, system->state_size_);
    
    // Initialize Intermediate Cost
    J = 0.0;

    // Initialize Intermediate value function
    V.resize(horizon + 1, 0.0);
    V_X.resize(horizon + 1, Eigen::VectorXd::Zero(system->state_size_));
    V_XX.resize(horizon + 1, Eigen::MatrixXd::Zero(system->state_size_, system->state_size_));

    // Initialize Q-function Matrices
    Q_UU.resize(horizon, Eigen::MatrixXd::Zero(system->control_size_, system->control_size_));
    Q_UX.resize(horizon, Eigen::MatrixXd::Zero(system->control_size_, system->state_size_));
    Q_U.resize(horizon, Eigen::VectorXd::Zero(system->control_size_));
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
    X.resize(dynamics->state_size_, horizon + 1); 
    U.resize(dynamics->control_size_, horizon);

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


void CDDPProblem::setInitialTrajectory(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) {
    this->X = X;
    this->U = U;

    // Raise error if dimensions do not match
    if (X.cols() != horizon + 1 || X.rows() != dynamics->state_size_) {
        throw std::invalid_argument("X dimensions do not match the horizon and state size");
    }
}


// Solve (Outline)
Eigen::MatrixXd CDDPProblem::solve() {
    // 1. Initialization
    // initializeTrajectory();
    // initializeCost();

    // 2. Main CDDP Iterative Loop
    for (int iter = 0; iter < options.max_iterations; ++iter) {
        // 3. Backward 
        // solveBackwardPass();
        
        // 4. Forward Pass
        solveForwardPass();

        std::cout << "Iteration: " << iter << std::endl;

        // 5 Convergence Check

    }

    // 6. Return Optimal Control Sequence
    return U;
}

// Forward Pass (Unconstrained)
bool CDDPProblem::solveForwardPass() {
    double alpha = options.backtracking_coeff;

    // Line-search loop 
    for (int i = 0; i < options.max_line_search_iterations; ++i) {
        Eigen::VectorXd x = initial_state;
        Eigen::MatrixXd X_new = X;
        Eigen::MatrixXd U_new = U;
        double J_new = 0.0;
        double dJ = 0.0;
        double expected_dV = 0.0;

        // 1. Simulate forward
        for (int j = 0; j < horizon; ++j) {
            Eigen::VectorXd delta_x = x - X_new.col(j);

            // Cholesky decomposition of Q_uu
            // Eigen::MatrixXd Q_L( Q_UU[j].llt().matrixL() );

            // Check if Q_UU is positive definite
            // if (Q_L.isZero()) {
            //     std::cout << "Q_UU is not positive definite" << std::endl;
            //     break;
            // }

            // Placeholder
            Eigen::MatrixXd Q_uu = Eigen::MatrixXd::Identity(dynamics->control_size_, dynamics->control_size_);

            // Eigen::MatrixXd Q_L( Q_uu.llt().matrixL() );

            // Feedfoward and Feedback Gains Calculation
            Eigen::MatrixXd k = Q_UU[j].inverse() * Q_U[j];
            Eigen::MatrixXd K = Q_UU[j].inverse() * Q_UX[j];
        
            Eigen::VectorXd delta_u =  alpha * k + K * delta_x; // Feed-forward and Feedback control law   

            U_new.col(j) += delta_u;       // Update control 
            x = dynamics->getDynamics(x, U_new.col(j)); // Simulate forward 
            X_new.col(j + 1) = x;          // Update trajectory
            J_new += dynamics->calculateCost(x, U_new.col(j)); // Update cost
        }
        J_new += dynamics->calculateFinalCost(x); // Final cost

        // 2. Calculate Cost Improvement
        dJ = J - J_new;

        // 3. Calculate Expected Improvement
        // for (int j = 0; j < horizon; ++j) {
        //     Eigen::VectorXd delta_x = X_new.col(j) - X.col(j);
        //     expected_dV += V_X[j].dot(delta_x) + 0.5 * delta_x.transpose() * V_XX[j] * delta_x;
        // }

        // 4. Check Improvement
        if (dJ > 0) {
            X = X_new;
            U = U_new;
            J = J_new;
            // V = V_new;
            // V_X = V_X_new;
            // V_XX = V_XX_new;
            return true;
        }

        // 5. Backtracking
        alpha = std::min(alpha * options.backtracking_factor, options.backtracking_min);

        return true;
    }
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


} // namespace cddp 