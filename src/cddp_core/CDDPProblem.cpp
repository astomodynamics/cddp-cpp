#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cddp/cddp_core/CDDPProblem.hpp" 
#include "osqp++.h"

namespace cddp {

// Constructor 
CDDPProblem::CDDPProblem(cddp::DynamicalSystem* system, const Eigen::VectorXd& initialState, int horizon, double timestep) :
    dynamics_(system), 
    initial_state_(initialState), 
    goal_state_(initialState), // Pre-allocate goal to initialState 
    horizon_(horizon),
    dt_(timestep)
{
    // Initialize Trajectory
    X_.resize(horizon + 1, initial_state_);
    U_.resize(horizon, Eigen::VectorXd::Zero(system->control_size_));

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
    X_[0] = x0; // Update initial trajectory state
}

void CDDPProblem::setGoalState(const Eigen::VectorXd& goal) {
    goal_state_ = goal;
}

void CDDPProblem::setHorizon(int T) {
    horizon_ = T;
    
    // Resize X, U, Q_UU, Q_UX, Q_U accordingly 
    X_.resize(horizon_ + 1, Eigen::VectorXd::Zero(dynamics_->state_size_)); 
    U_.resize(horizon_, Eigen::VectorXd::Zero(dynamics_->state_size_));

    Q_UU_.resize(horizon_, Eigen::MatrixXd::Zero(dynamics_->control_size_, dynamics_->control_size_));
    Q_UX_.resize(horizon_, Eigen::MatrixXd::Zero(dynamics_->control_size_, dynamics_->state_size_));
    Q_U_.resize(horizon_, Eigen::VectorXd::Zero(dynamics_->control_size_));
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

void CDDPProblem::setObjective(std::unique_ptr<Objective> objective) {
    objective_ = std::move(objective);
}

void CDDPProblem::setInitialTrajectory(const std::vector<Eigen::VectorXd>& X, const std::vector<Eigen::VectorXd>& U) {
    this->X_ = X;
    this->U_ = U;

    // Raise error if dimensions do not match
    if (X.size() != horizon_ + 1 || X[0].size() != dynamics_->state_size_) {
        throw std::invalid_argument("X dimensions do not match the horizon and state size");
    }
}

void CDDPProblem::initializeCost() {
    J_ = 0.0;
    for (int i = 0; i < horizon_; ++i) {
        J_ += objective_->calculateRunningCost(X_.at(i), U_.at(i));
    }
    J_ += objective_->calculateFinalCost(X_.back());

}

// void CDDPProblem::addConstraint(std::unique_ptr<Constraint> constraint) {
//     constraints_.push_back(std::move(constraint));
// }

// Print Methods
// Print iteration information

// void CDDPProblem::printTrajectory() {
//     std::cout << "Trajectory: " << std::endl;
//     for (int i = 0; i < horizon_ + 1; ++i) {
//         std::cout << X_.at(i).transpose() << std::endl;
//     }
// }

// void CDDPProblem::printControlSequence() {
//     std::cout << "Control Sequence: " << std::endl;
//     for (int i = 0; i < horizon_; ++i) {
//         std::cout << U_.at(i).transpose() << std::endl;
//     }
// }

// void CDDPProblem::printCost() {
//     std::cout << "Cost: " << J_ << std::endl;
// }

// void CDDPProblem::printValueFunction() {
//     std::cout << "Value Function: " << std::endl;
//     for (int i = 0; i < horizon_ + 1; ++i) {
//         std::cout << V_.at(i) << std::endl;
//     }
// }

// void CDDPProblem::printValueFunctionDerivative() {
//     std::cout << "Value Function Derivative: " << std::endl;
//     for (int i = 0; i < horizon_ + 1; ++i) {
//         std::cout << V_X_.at(i).transpose() << std::endl;
//     }
// }




// CDDP Solver
std::vector<Eigen::VectorXd> CDDPProblem::solve() {
    // 1. Initialization
    // initializeTrajectory();
    initializeCost();
    double J = std::numeric_limits<double>::infinity();
    double gradientNorm = std::numeric_limits<double>::infinity();

    // 2. Main CDDP Iterative Loop
    for (int iter = 0; iter < options_.max_iterations; ++iter) {
std::cout << "Iteration: " << iter << std::endl;
std::cout << "Cost " << J_ << std::endl;
        double J_old = J;
        // 3. Backward 
        bool backward_pass_done = solveBackwardPass();
        
        if (!backward_pass_done) {
            std::cout << "Backward Pass Failed" << std::endl;
            break;
        }

        // 4. Forward Pass
        bool forward_pass_done = solveForwardPass();

        if (!forward_pass_done) {
            std::cout << "Forward Pass Failed" << std::endl;
            break;
        }

        J = J_;

        // 5 Convergence Check
        if (std::abs(J_old - J) < options_.cost_tolerance) {
            std::cout << "Converged" << std::endl;
            break;
        }
        
        // if (gradientNorm < options_.grad_tolerance) {
        //     std::cout << "Converged" << std::endl;
        //     break;
        // }

        // 6. Print Iteration Information
        // printIteration(iter, cost, gradientNorm, lambda);


    }

    // 6. Return Optimal Control Sequence
    // place holder 
    return U_;

}

// Forward Pass
bool CDDPProblem::solveForwardPass() {
    double trust_region_radius = options_.trust_region_radius;
    bool is_feasible = false;

    // Temporary Control Bounds
    Eigen::VectorXd temp_control_max(dynamics_->control_size_);
    temp_control_max << 5.0, M_PI;
    Eigen::VectorXd temp_control_min(dynamics_->control_size_);
    temp_control_min << -5.0, -M_PI;
    int iter = 0;
    
    // Trust Region Loop
    while (is_feasible == false) {
        double J_new = 0.0;
        double dJ = 0.0;            
        double expected_dV = 0.0;

        std::vector<Eigen::VectorXd> X_new = X_;
        std::vector<Eigen::VectorXd> U_new = U_;
        Eigen::VectorXd x = initial_state_;
        X_new.front() = x;

        // 1. Simulate forward
        for (int i= 0; i < horizon_; ++i) {
            // Deviation from nominal trajectory
            Eigen::VectorXd delta_x = x - X_.at(i);
            
            // Extract Q-Function Matrices
            Eigen::MatrixXd Q_ux = Q_UX_.at(i);
            Eigen::MatrixXd Q_uu = Q_UU_.at(i);
            Eigen::VectorXd Q_u = Q_U_.at(i);

            // Create QP problem
            Eigen::SparseMatrix<double> P(Q_uu.rows(), Q_uu.cols()); // Hessian of QP objective
            int numNonZeros = Q_uu.nonZeros(); 
            P.reserve(numNonZeros);
            for (int i = 0; i < Q_uu.rows(); ++i) {
                for (int j = 0; j < Q_uu.cols(); ++j) {
                    if (Q_uu(i, j) != 0) {
                        P.insert(i, j) = Q_uu(i, j);
                    }
                }
            }
            P.makeCompressed(); // Important for efficient storage and operations

            Eigen::VectorXd q = Q_ux * delta_x + Q_u; // Gradient of QP objective

            // Create constraints
            Eigen::SparseMatrix<double> A(dynamics_->control_size_, dynamics_->control_size_);
            A.setIdentity();
            A.makeCompressed();

            // Lower and upper bounds
            Eigen::VectorXd lb = trust_region_radius * (temp_control_min - U_.at(i));
            Eigen::VectorXd ub = trust_region_radius * (temp_control_max - U_.at(i));

            // Initialize QP solver
            osqp::OsqpInstance instance;

            // Set the objective
            instance.objective_matrix = P;
            instance.objective_vector = q;
            instance.constraint_matrix = A;
            instance.lower_bounds = lb;
            instance.upper_bounds = ub;

            // Solve the QP problem
            osqp::OsqpSolver osqp_solver;
            osqp::OsqpSettings settings;
            settings.warm_start = true;
            settings.verbose = false;

            osqp_solver.Init(instance, settings);
            osqp::OsqpExitCode exit_code = osqp_solver.Solve();
            double optimal_objective = osqp_solver.objective_value();
            Eigen::VectorXd delta_u = osqp_solver.primal_solution();

            if (exit_code == osqp::OsqpExitCode::kOptimal) {
                is_feasible = true;
            } else {
                trust_region_radius *= options_.trust_region_factor;
                break;
            }

            U_new.at(i) += delta_u;       // Update control 

            J_new += objective_->calculateRunningCost(x, U_new.at(i)); // Running cost
            x = dynamics_->getDynamics(x, U_new.at(i)); // Simulate forward 
            X_new.at(i + 1) = x;          // Update trajectory
        }
        J_new += objective_->calculateFinalCost(X_new.back()); // Final cost

        // 2. Calculate Cost Improvement
        dJ = J_ - J_new;

    
        // 3. Calculate Expected Improvement
        // for (int j = 0; j < horizon; ++j) {
        //     Eigen::VectorXd delta_x = X_new.col(j) - X.col(j);
        //     expected_dV += V_X[j].dot(delta_x) + 0.5 * delta_x.transpose() * V_XX[j] * delta_x;
        // }

        // Check Improvement
        if (dJ > 0) {
            X_ = X_new;
            U_ = U_new;
            J_ = J_new;
            return true;
        }
        iter += 1;
    }
    return false;
}

bool CDDPProblem::solveBackwardPass() {
    // Initialize final value function
    V_.back() = objective_->calculateFinalCost(X_.back());
    V_X_.back() = objective_->calculateFinalCostGradient(X_.back());
    V_XX_.back() = objective_->calculateFinalCostHessian(X_.back());

    // Solve Backward Ricatti Equation
    for (int i = horizon_ - 1; i >= 0; --i) {
        const Eigen::VectorXd& x = X_.at(i);
        const Eigen::VectorXd& u = U_.at(i);

        // Calculate Dynamics Jacobians
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> jacobians = dynamics_->getDynamicsJacobian(x, u);
        Eigen::MatrixXd f_x = std::get<0>(jacobians);
        Eigen::MatrixXd f_u = std::get<1>(jacobians);

        // Calculate discrete time dynamics
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(dynamics_->state_size_, dynamics_->state_size_) + dt_ * f_x;
        Eigen::MatrixXd B = dt_ * f_u;

        // Calculate cost gradients and Hessians
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> gradients = objective_->calculateRunningCostGradient(x, u);
        Eigen::VectorXd l_x = std::get<0>(gradients);
        Eigen::VectorXd l_u = std::get<1>(gradients);

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> hessians = objective_->calculateRunningCostHessian(x, u);
        Eigen::MatrixXd l_xx = std::get<0>(hessians);
        Eigen::MatrixXd l_ux = std::get<1>(hessians);
        Eigen::MatrixXd l_uu = std::get<2>(hessians);
        
        // Q function calculations (iLQR based)
        Eigen::MatrixXd Q_x = l_x + A.transpose() * V_X_.at(i+1);
        Eigen::MatrixXd Q_u = l_u + B.transpose() * V_X_.at(i+1);
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_XX_.at(i+1) * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_XX_.at(i+1) * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_XX_.at(i+1) * B;
// std::cout << "Q_xx: " << Q_xx << std::endl;
// std::cout << "Q_uu: " << Q_uu << std::endl;

        // Symmetrize Hessian
// Q_uu = 0.5 * (Q_uu + Q_uu.transpose());

        // Check eigenvalues of Q_uu
        Eigen::EigenSolver<Eigen::MatrixXd> es(Q_uu);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real();
        if (eigenvalues.minCoeff() <= 0) {
            std::cout << "Q_uu is not positive definite" << std::endl;
            break;
        }

// TODO: Regularization
// if (options_.regularization_type == 0) {
//     Q_uu += options_.regularization_factor * Eigen::MatrixXd::Identity(dynamics_->control_size_, dynamics_->control_size_);
// } else if (options_.regularization_type == 1) {
//     Q_uu += options_.regularization_factor * Q_uu.maxCoeff() * Eigen::MatrixXd::Identity(dynamics_->control_size_, dynamics_->control_size_);
// }

        // Feedback Gain Calculation 
        Eigen::MatrixXd K = -Q_uu.inverse() * Q_ux;
        Eigen::VectorXd k = -Q_uu.inverse() * Q_u;

        // Store Q-Function Matrices
        Q_UU_.at(i) = Q_uu;
        Q_UX_.at(i) = Q_ux;
        Q_U_.at(i) = Q_u;

        k_.at(i) = k;
        K_.at(i) = K;

        // Update Value Function
        double cost = objective_->calculateRunningCost(x, u);
        V_X_.at(i) = Q_x + K.transpose() * Q_uu * k + Q_ux.transpose() * k + K.transpose() * Q_u;
        V_XX_.at(i) = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
        // Symmetrize Hessian
// V_XX_.at(i) = 0.5 * (V_XX_.at(i) + V_XX_.at(i).transpose());
        V_.at(i) = cost + V_X_.at(i).transpose() * (x - goal_state_) + 0.5 * (x - goal_state_).transpose() * V_XX_.at(i) * (x - goal_state_);
    }
    return true;
}
}  // namespace cddp