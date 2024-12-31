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

#include <iostream> // For std::cout, std::cerr
#include <iomanip>  // For std::setw
#include <memory>   // For std::unique_ptr
#include <map>      // For std::map
#include <cmath>    // For std::log
#include <Eigen/Dense>
#include <chrono>    // For timing
#include <execution> // For parallel execution policies
#include <future> // For multi-threading
#include <thread> // For multi-threading

#include "cddp_core/cddp_core.hpp"
#include "cddp_core/helper.hpp"
#include "cddp_core/boxqp.hpp"

namespace cddp
{
bool CDDP::solveIPDDPBackwardPass() {
    // Setup
    // Initialize variables
    const int state_dim = getStateDim();
    const int control_dim = getControlDim();
    const int total_dual_dim = getTotalDualDim(); // Number of dual variables across constraints

    // Terminal cost and derivatives
    Eigen::VectorXd V_x  = objective_->getFinalCostGradient(X_.back());
    Eigen::MatrixXd V_xx = objective_->getFinalCostHessian(X_.back());
    V_xx = 0.5 * (V_xx + V_xx.transpose());  // Symmetrize 

    dV_ = Eigen::Vector2d::Zero();
    double Qu_max_norm = 0.0;
    double residual_max = 0.0; // complementary residual measure: r = s ◦ y - mu_
    double dual_norm = 0.0; //

  
    // Backward Recursion
    for (int t = horizon_ - 1; t >= 0; --t) {
        // Expand cost around (x[t], u[t])
        const Eigen::VectorXd& x = X_[t];
        const Eigen::VectorXd& u = U_[t];

        // Continuous dynamics 
        const auto [Fx, Fu] = system_->getJacobians(x, u);

        // Discretize
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim) + timestep_ * Fx;
        Eigen::MatrixXd B = timestep_ * Fu;

        // Extract dual variable and constraint values
        // Gather dual and slack variables, and constraint values
        Eigen::VectorXd y = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(total_dual_dim);

        Eigen::MatrixXd Q_yu = Eigen::MatrixXd::Zero(total_dual_dim, control_dim);
        Eigen::MatrixXd Q_yx = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);

        int offset = 0; // offset in [0..total_dual_dim)
        for (auto &cKV : constraint_set_) {
            const std::string &cname = cKV.first;
            auto &constraint        = cKV.second;
            int dual_dim = constraint->getDualDim();

            // Slack & dual at time t and constraint cname
            Eigen::VectorXd y_vec = Y_[cname][t]; // dual variable

            // Evaluate constraint
            // Eigen::VectorXd g_vec = constraint->evaluate(x, u); // dimension = dual_dim
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::VectorXd g_vec = Eigen::VectorXd::Zero(dual_dim);
            if (cname == "ControlBoxConstraint") {
                Eigen::VectorXd lb = constraint->getLowerBound();
                Eigen::VectorXd ub = constraint->getUpperBound();
                g_vec.head(control_dim) = u - ub;
                g_vec.tail(control_dim) = lb - u;
            }

            // partial wrt. x => g_x
            // Eigen::MatrixXd g_x = constraint->getStateJacobian(x, u);
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::MatrixXd g_x = Eigen::MatrixXd::Zero(dual_dim, state_dim);

            // partial wrt. u => g_u
            // Eigen::MatrixXd g_u = constraint->getControlJacobian(x, u);
            // Hardcode for now (ControlBoxConstraint) TODO: Generalize
            Eigen::MatrixXd g_u = Eigen::MatrixXd::Zero(dual_dim, control_dim);
            if (cname == "ControlBoxConstraint") {
                // top half
                g_u.block(0, 0, dual_dim/2, control_dim) 
                    =  Eigen::MatrixXd::Identity(dual_dim/2, control_dim);
                // bottom half
                g_u.block(control_dim, 0,  dual_dim/2, control_dim) 
                    = -Eigen::MatrixXd::Identity( dual_dim/2, control_dim);
            }

            // Insert into big arrays
            y.segment(offset, dual_dim)   = y_vec;
            g.segment(offset, dual_dim)   = g_vec;
            Q_yx.block(offset, 0, dual_dim, state_dim)   = g_x;
            Q_yu.block(offset, 0, dual_dim, control_dim) = g_u;

            offset += dual_dim;
        }

        // Cost & derivatives
        double l = objective_->running_cost(x, u, t);
        auto [l_x, l_u] = objective_->getRunningCostGradients(x, u, t);
        auto [l_xx, l_uu, l_ux] = objective_->getRunningCostHessians(x, u, t);

        // Q expansions from cost
        Eigen::VectorXd Q_x  = l_x + Q_yx.transpose() * y + A.transpose() * V_x;
        Eigen::VectorXd Q_u  = l_u + Q_yx.transpose() * y + B.transpose() * V_x;
        Eigen::MatrixXd Q_xx = l_xx + A.transpose() * V_xx * A;
        Eigen::MatrixXd Q_ux = l_ux + B.transpose() * V_xx * A;
        Eigen::MatrixXd Q_uu = l_uu + B.transpose() * V_xx * B;


        Eigen::MatrixXd Y = y.asDiagonal(); // Diagonal matrix with y as diagonal
        Eigen::MatrixXd G = g.asDiagonal(); // Diagonal matrix with g as diagonal

        // Regularization
        Eigen::MatrixXd Q_ux_reg = Q_ux;
        Eigen::MatrixXd Q_uu_reg = Q_uu;

        // TODO: Add State regularization here
        if (options_.regularization_type == "control" || 
            options_.regularization_type == "both") {
            Q_uu_reg.diagonal().array() += regularization_control_;
        }
        Q_uu_reg = 0.5 * (Q_uu_reg + Q_uu_reg.transpose()); // symmetrize

        Eigen::VectorXd r = Y * g + mu_ * Eigen::VectorXd::Ones(total_dual_dim);
        Eigen::MatrixXd G_inv = G.inverse();
        Eigen::MatrixXd YGinv = Y * G_inv;

        Eigen::LLT<Eigen::MatrixXd> llt(Q_uu_reg - Q_yu.transpose() * YGinv * Q_yu);
        if (llt.info() != Eigen::Success) {
            if (options_.debug) {
                std::cerr << "CDDP: Backward pass failed at time " << t << std::endl;
            }
            return false;
        }
        
        Eigen::MatrixXd bigRHS(control_dim, 1 + state_dim);
        bigRHS.col(0) = Q_u - Q_yu.transpose() * G_inv * r;
        Eigen::MatrixXd M = //(control_dim, state_dim)
            Q_ux- Q_yu.transpose() * YGinv * Q_yx;
        for (int col = 0; col < state_dim; col++) {
            bigRHS.col(col+1) = M.col(col);
        }

        Eigen::MatrixXd R = llt.matrixU();
        // forward/back solve
        Eigen::MatrixXd kK = -R.transpose().triangularView<Eigen::Upper>().solve(
                                R.triangularView<Eigen::Upper>().solve(bigRHS)
                             );

        // parse out feedforward (ku) and feedback (Ku)
        Eigen::VectorXd k_u = kK.col(0); // dimension [control_dim]
        Eigen::MatrixXd K_u(control_dim, state_dim);
        for (int col = 0; col < state_dim; col++) {
            K_u.col(col) = kK.col(col+1);
        }

        // Save gains
        k_u_[t] = k_u;
        K_u_[t] = K_u;

        Eigen::VectorXd k_y = - G_inv * (r + Y * Q_yx * k_u);
        Eigen::MatrixXd K_y = - YGinv * (Q_yx + Q_yu * K_u);
        Eigen::VectorXd k_s = Eigen::VectorXd::Zero(total_dual_dim);
        Eigen::MatrixXd K_s = Eigen::MatrixXd::Zero(total_dual_dim, state_dim);

        offset = 0;
        for (auto &ckv : constraint_set_) {
            const std::string &cname = ckv.first;
            auto &constraint = ckv.second;
            int dual_dim = constraint->getDualDim();

            // Now store gains
            k_y_[cname][t] = k_y.segment(offset, dual_dim);
            K_y_[cname][t] = K_y.block(offset, 0, dual_dim, state_dim);
            k_s_[cname][t] = k_s.segment(offset, dual_dim);
            K_s_[cname][t] = K_s.block(offset, 0, dual_dim, state_dim);

            offset += dual_dim;
        }

        // Update Q expansions
        Q_x  -= Q_yx.transpose() * (G_inv * r);
        Q_u  -= Q_yu.transpose() * (G_inv * r);
        Q_xx -= Q_yx.transpose() * YGinv * Q_yx;
        Q_ux -= Q_yu.transpose() * YGinv * Q_yx;
        Q_uu -= Q_yu.transpose() * YGinv * Q_yu;

        // Update cost improvement
        dV_[0] += k_u.dot(Q_u);
        dV_[1] += 0.5 * k_u.dot(Q_uu * k_u);

        // Update value function
        V_x = Q_x + K_u.transpose() * Q_u + Q_ux.transpose() * k_u + K_u.transpose() * Q_uu * k_u;
        V_xx = Q_xx + K_u.transpose() * Q_ux + Q_ux.transpose() * K_u + K_u.transpose() * Q_uu * K_u;

        // Debug norms
        double Qu_norm = Q_u.lpNorm<Eigen::Infinity>();
        if (Qu_norm > Qu_max_norm) Qu_max_norm = Qu_norm;
        double r_norm = r.lpNorm<Eigen::Infinity>();
        if (r_norm > residual_max) residual_max = r_norm; 
    } // end for t

    // Compute optimality gap and print
    optimality_gap_ = std::max(Qu_max_norm, residual_max);

    if (options_.debug) {
        std::cout << "[IPDDP Backward Pass]\n"
                  << "    Qu_max_norm:  " << Qu_max_norm << "\n"
                  << "    residual_max:  " << residual_max << "\n"
                  << "    dV:           " << dV_.transpose() << std::endl;
    }
    return true;
} // end solveIPDDPBackwardPass

ForwardPassResult CDDP::solveIPDDPForwardPass(double alpha) {
    // Prepare result struct
    ForwardPassResult result;
    result.success = false;
    result.cost = std::numeric_limits<double>::infinity();
    result.lagrangian = std::numeric_limits<double>::infinity();
    result.alpha = alpha;

    const int state_dim = getStateDim();
    const int control_dim = getControlDim();

    double tau = std::max(0.99, 1.0 - mu_);  

    // Copy old trajectories (from the “previous” solution)
    std::vector<Eigen::VectorXd> X_new = X_;  // old states
    std::vector<Eigen::VectorXd> U_new = U_;  // old controls
    std::map<std::string, std::vector<Eigen::VectorXd>> Y_new = Y_;  // old dual

    X_new[0] = initial_state_;
    double cost_new = 0.0;
    double log_cost_new = 0.0;
    double primal_residual = 0.0;
    double sum_log_c = 0.0; 

    auto control_box_constraint = getConstraint<cddp::ControlBoxConstraint>("ControlBoxConstraint");

    // Current filter point
    FilterPoint current{J_, L_};

    for (int t = 0; t < horizon_; ++t) {
        // 1) Update dual & slack
        for (auto &ckv : constraint_set_) {
            const std::string &cname = ckv.first;
            auto &constraint = ckv.second;
            int dual_dim = ckv.second->getDualDim();

            // old y
            const Eigen::VectorXd &y_old = Y_[cname][t];

            // new y
            Eigen::VectorXd y_new = y_old + 
                                    alpha * k_y_[cname][t] + 
                                    K_y_[cname][t] * (X_new[t] - X_[t]); 

            Eigen::VectorXd g_new = Eigen::VectorXd::Zero(dual_dim);
            Eigen::VectorXd g_old = Eigen::VectorXd::Zero(dual_dim);
            if (cname == "ControlBoxConstraint") {
                Eigen::VectorXd lb = constraint->getLowerBound();
                Eigen::VectorXd ub = constraint->getUpperBound();
                g_new.head(control_dim) = U_new[t] - ub;
                g_new.tail(control_dim) = lb - U_new[t];
                g_old.head(control_dim) = U_[t] - ub;
                g_old.tail(control_dim) = lb - U_[t];

            }

            // Enforce minimal feasibility w.r.t. old solution
            for (int i = 0; i < dual_dim; i++) {
                double y_min = (1.0 - tau) * y_old[i]; 
                double g_min = (1.0 - tau) * g_old[i];
                if (y_new[i] < y_min || g_new[i] > g_min) {
                    // fail early
                    // std::cout << "y_new: " << y_new.transpose() << std::endl;
                    // if (options_.debug) {
                    //     std::cerr << "[IPDDP ForwardPass] Feasibility fail at time=" 
                    //               << t << ", constraint=" << cname 
                    //               << " y_new or s_new < (1-tau)*y_old or s_old." 
                    //               << std::endl;
                    // }
                    // return result; // success=false, cost=inf => exit
                }
                if (y_new[i] < 0.0) {
                    y_new[i] = 1e-8;
                }
            }

            // primal residual: L1 norm of g + y
            double local_res = (g_new + y_new).lpNorm<1>();
            if (local_res > primal_residual) {
                primal_residual = local_res;
            }

            // sum logs
            sum_log_c += g_new.array().log().sum();

            if (y_new.minCoeff() <= 0.0) {
                // log of non-positive => fail or large cost
                if (options_.debug) {
                    std::cerr << "[IPDDP FwdPass] y_new <= 0 => log is invalid at time=" 
                                << t << ", constraint=" << cname << std::endl;
                }
                return result;
            }

            // Save them
            Y_new[cname][t] = y_new;
        }

        // 2) Update control
        const Eigen::VectorXd &u_old = U_[t];
        U_new[t] = u_old + alpha * k_u_[t] + K_u_[t] * (X_new[t] - X_[t]);

        // 3) Accumulate cost / measure constraint violation
        double stage_cost = objective_->running_cost(X_new[t], U_new[t], t);
        cost_new += stage_cost;

        // 5) Step the dynamics
        X_new[t+1] = system_->getDiscreteDynamics(X_new[t], U_new[t]);
    }

    cost_new += objective_->terminal_cost(X_new.back());
    log_cost_new = cost_new - mu_ * sum_log_c;

    std::cout << "cost_new: " << cost_new << std::endl;
    std::cout << "log_cost_new: " << log_cost_new << std::endl;

    FilterPoint candidate{cost_new, log_cost_new};

    // Filter acceptance criteria  
    // bool sufficient_progress = 
    //             (cost_new < current.cost - gamma_ * candidate.violation) || 
    //             (candidate.violation < (1 - gamma_) * current.violation);
    bool sufficient_progress = (cost_new < current.cost);

    bool acceptable = sufficient_progress && !current.dominates(candidate);

   if (acceptable) {
       double expected = -alpha * (dV_(0) + 0.5 * alpha * dV_(1));
       double reduction_ratio = expected > 0.0 ? (J_ - cost_new) / expected : 
                                               std::copysign(1.0, J_ - cost_new);

       result.success = acceptable;
       result.state_sequence = X_new;
       result.control_sequence = U_new;
       result.dual_sequence = Y_new;
    //    result.slack_sequence = S_new;
       result.cost = cost_new;
       result.lagrangian = log_cost_new;
       result.constraint_violation = primal_residual;
   }

   return result;
} // end solveIPDDPForwardPass

} // namespace cddp