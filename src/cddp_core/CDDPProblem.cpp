#include <iostream>
#include "cddp_core/CDDPProblem.hpp" 


namespace cddp {

// Constructor 
CDDPProblem::CDDPProblem(cddp::DynamicalSystem* system, const Eigen::VectorXd& initialState, int horizon, double timestep) :
    dynamics_(system), 
    initial_state_(initialState), 
    goal_state_(initialState), // Pre-allocate goal to initialState 
    horizon_(horizon),
    dt_(timestep),
    X_(system->state_size_, horizon + 1), // Pre-allocate trajectories
    U_(system->control_size_, horizon)
{
    // Initialize Intermediate Cost
    J_ = 0.0;

    k_.resize(horizon, Eigen::VectorXd::Zero(system->control_size_));
    K_.resize(horizon, Eigen::MatrixXd::Zero(system->control_size_, system->state_size_));

    // Initialize Intermediate value function
    V_.resize(horizon + 1, 0.0);
    V_X_.resize(horizon + 1, Eigen::VectorXd::Zero(system->state_size_));
    V_XX_.resize(horizon + 1, Eigen::MatrixXd::Zero(system->state_size_, system->state_size_));

    // Initialize Q-function Matrices
    Q_UU_.resize(horizon, Eigen::MatrixXd::Zero(system->control_size_, system->control_size_));
    Q_UX_.resize(horizon, Eigen::MatrixXd::Zero(system->control_size_, system->state_size_));
    Q_U_.resize(horizon, Eigen::VectorXd::Zero(system->control_size_));
}

// Setup Methods
void CDDPProblem::setInitialState(const Eigen::VectorXd& x0) {
    initial_state_ = x0;
    X_.col(0) = x0; // Update initial trajectory state
}

void CDDPProblem::setGoalState(const Eigen::VectorXd& goal) {
    goal_state_ = goal;
}

void CDDPProblem::setHorizon(int T) {
    horizon_ = T;
    
    // Resize X, U, Q_UU, Q_UX, Q_U accordingly 
    X_.resize(dynamics_->state_size_, horizon_ + 1); 
    U_.resize(dynamics_->control_size_, horizon_);

    Q_UU_.resize(horizon_);
    Q_UX_.resize(horizon_);
    Q_U_.resize(horizon_);
}

void CDDPProblem::setTimeStep(double timestep) {
    dt_ = timestep;
}

// void CDDPProblem::setCostMatrices(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, const Eigen::MatrixXd& Qf) {
//     this->Q = Q;
//     this->R = R;
//     this->Qf = Qf;
// }

void CDDPProblem::setOptions(const CDDPOptions& opts) {
    options_ = opts;
}


void CDDPProblem::setInitialTrajectory(const Eigen::MatrixXd& X, const Eigen::MatrixXd& U) {
    this->X_ = X;
    this->U_ = U;

    // Raise error if dimensions do not match
    if (X.cols() != horizon_ + 1 || X.rows() != dynamics_->state_size_) {
        throw std::invalid_argument("X dimensions do not match the horizon and state size");
    }
}

void CDDPProblem::setObjective(std::unique_ptr<Objective> objective) {
    objective_ = std::move(objective);
}

// void CDDPProblem::addConstraint(std::unique_ptr<Constraint> constraint) {
//     constraints_.push_back(std::move(constraint));
// }


// Solve (Outline)
Eigen::MatrixXd CDDPProblem::solve() {
    // 1. Initialization
    // initializeTrajectory();
    // initializeCost();

    // 2. Main CDDP Iterative Loop
    for (int iter = 0; iter < options_.max_iterations; ++iter) {
        // 3. Backward 
        // solveBackwardPass();
        
        // 4. Forward Pass
        solveForwardPass();

        std::cout << "Iteration: " << iter << std::endl;

        // 5 Convergence Check

    }

    // 6. Return Optimal Control Sequence
    Eigen::MatrixXd U(dynamics_->control_size_, horizon_);
    return U;

}

// Forward Pass (Unconstrained)
bool CDDPProblem::solveForwardPass() {
    double alpha = options_.backtracking_coeff;

    // Line-search loop 
    for (int i = 0; i < options_.max_line_search_iterations; ++i) {
        Eigen::VectorXd x = initial_state_;
        Eigen::MatrixXd X_new = X_;
        Eigen::MatrixXd U_new = U_;
        double J_new = 0.0;
        double dJ = 0.0;
        double expected_dV = 0.0;

        // 1. Simulate forward
        for (int j = 0; j < horizon_; ++j) {
            Eigen::VectorXd delta_x = x - X_new.col(j);

            // Cholesky decomposition of Q_uu
            // Eigen::MatrixXd Q_L( Q_UU[j].llt().matrixL() );

            // Check if Q_UU is positive definite
            // if (Q_L.isZero()) {
            //     std::cout << "Q_UU is not positive definite" << std::endl;
            //     break;
            // }

            // Placeholder
            Eigen::MatrixXd Q_uu = Eigen::MatrixXd::Identity(dynamics_->control_size_, dynamics_->control_size_);

            // Eigen::MatrixXd Q_L( Q_uu.llt().matrixL() );

            // Feedfoward and Feedback Gains Calculation
            Eigen::MatrixXd k = Q_UU_[j].inverse() * Q_U_[j];
            Eigen::MatrixXd K = Q_UU_[j].inverse() * Q_UX_[j];
        
            Eigen::VectorXd delta_u =  alpha * k + K * delta_x; // Feed-forward and Feedback control law   

            U_new.col(j) += delta_u;       // Update control 
            x = dynamics_->getDynamics(x, U_new.col(j)); // Simulate forward 
            X_new.col(j + 1) = x;          // Update trajectory
            J_new += dynamics_->calculateCost(x, U_new.col(j)); // Update cost
        }
        J_new += dynamics_->calculateFinalCost(x); // Final cost

        // 2. Calculate Cost Improvement
        dJ = J_ - J_new;

        // 3. Calculate Expected Improvement
        // for (int j = 0; j < horizon; ++j) {
        //     Eigen::VectorXd delta_x = X_new.col(j) - X.col(j);
        //     expected_dV += V_X[j].dot(delta_x) + 0.5 * delta_x.transpose() * V_XX[j] * delta_x;
        // }

        // 4. Check Improvement
        if (dJ > 0) {
            X_ = X_new;
            U_ = U_new;
            J_ = J_new;
            // V = V_new;
            // V_X = V_X_new;
            // V_XX = V_XX_new;
            return true;
        }

        // 5. Backtracking
        alpha = std::min(alpha * options_.backtracking_factor, options_.backtracking_min);

        return true;
    }
}

// bool CDDPProblem::solveBackwardPass() {
//     // Eigen::MatrixXd A = dynamics->getDynamicsHessian(X.col(horizon), Eigen::VectorXd::Zero(dynamics->control_size)); // Assuming zero control for final time step Hessian
//     // Eigen::VectorXd b = dynamics->getDynamicsJacobian(X.col(horizon), Eigen::VectorXd::Zero(dynamics->control_size)).transpose() * (Qf * (X.col(horizon) - goal_state)); // Assuming zero control for final time step gradient 

//     V[horizon] = dynamics->calculateFinalCost(X.col(horizon)); // Final cost
//     V_X[horizon] = dynamics->calculateFinalCostGradient(X.col(horizon)); // Final cost gradient
//     V_XX[horizon] = dynamics->calculateFinalCostHessian(X.col(horizon)); // Final cost Hessian

//     for (int i = horizon - 1; i >= 0; --i) {
//         // const Eigen::VectorXd&  x = X.col(i);
//         // const Eigen::VectorXd&  u = U.col(i);

//         // // Q function calculations (Replace with your system-specific expressions)
//         // Eigen::MatrixXd Q_xx = dynamics->calculateCostHessian(x, u) + dynamics->getDynamicsJacobian(x, u).transpose() * A * dynamics->getDynamicsJacobian(x, u);
//         // Eigen::MatrixXd Q_ux = dynamics->getDynamicsJacobian(x, u).transpose() * A; 
//         // Eigen::VectorXd Q_x = dynamics->calculateCostGradient(x, u) + dynamics->getDynamicsJacobian(x, u).transpose() * b;
//         // // ... (Similar for Q_uu, Q_u) ...

//         // // Feedback Gain Calculation (Simplified for unconstrained)
//         // Eigen::MatrixXd K = -Q_uu.inverse() * Q_ux;
//         // Eigen::VectorXd k = -Q_uu.inverse() * Q_u;

//         // // Store Value Function Matrices
//         // Q_UU[i] = Q_uu;
//         // Q_UX[i] = Q_ux;
//         // Q_U[i] = Q_u;

//         // // Update A and b (Backward propagation)
//         // A = Q_xx + K.transpose() * Q_uu * K + K.transpose() * Q_ux + Q_ux.transpose() * K;
//         // b = Q_x + K.transpose() * Q_uu * k + K.transpose() * Q_u + Q_ux.transpose() * k;
    // }
// }

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