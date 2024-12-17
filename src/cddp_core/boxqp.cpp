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
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "cddp_core/boxqp.hpp"

namespace cddp {

BoxQPSolver::BoxQPSolver(const BoxQPOptions& options) : options_(options) {}

BoxQPResult BoxQPSolver::solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
                              const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                              const Eigen::VectorXd& x0) {
    const int n = H.rows();
    BoxQPResult result;
    result.status = BoxQPStatus::MAX_ITER_EXCEEDED;
    result.iterations = 0;
    result.factorizations = 0;

    // Initialize state vector x
    result.x = initializeX(x0, lower, upper, n);
    
    // Initialize free/clamped sets
    Eigen::VectorXi clamped = Eigen::VectorXi::Zero(n);
    result.free = Eigen::VectorXi::Ones(n);
    
    // Initial objective value
    double value = evaluateObjective(result.x, H, g);
    double old_value = std::numeric_limits<double>::infinity();
    
    // Main iteration loop
    for (int iter = 0; iter < options_.maxIter; ++iter) {
        result.iterations = iter + 1;
        
        // Check relative improvement
        if (iter > 0 && std::abs(old_value - value) < options_.minRelImprove * std::abs(old_value)) {
            result.status = BoxQPStatus::SMALL_IMPROVEMENT;
            break;
        }
        old_value = value;

        // Calculate gradient
        Eigen::VectorXd grad = g + H * result.x;
        
        // Update clamped set
        Eigen::VectorXi old_clamped = clamped;
        clamped = Eigen::VectorXi::Zero(n);
        
        for (int i = 0; i < n; ++i) {
            if ((result.x[i] == lower[i] && grad[i] > 0) ||
                (result.x[i] == upper[i] && grad[i] < 0)) {
                clamped[i] = 1;
            }
        }
        result.free = Eigen::VectorXi::Ones(n) - clamped;
        
        // Check if all dimensions are clamped
        if (clamped.sum() == n) {
            result.status = BoxQPStatus::ALL_CLAMPED;
            break;
        }

        // Factorize if clamped set changed
        bool any_different = false;
        for(int i = 0; i < n; i++) {
            if(old_clamped[i] != clamped[i]) {
                any_different = true;
                break;
            }
        }
        bool factorize = (iter == 0) || any_different;

        if (factorize) {
            std::vector<int> free_idx;
            for (int i = 0; i < n; ++i) {
                if (!clamped[i]) free_idx.push_back(i);
            }
            
            Eigen::MatrixXd H_free(free_idx.size(), free_idx.size());
            for (size_t i = 0; i < free_idx.size(); ++i) {
                for (size_t j = 0; j < free_idx.size(); ++j) {
                    H_free(i,j) = H(free_idx[i], free_idx[j]);
                }
            }

            result.Hfree.compute(H_free);
            if (result.Hfree.info() != Eigen::Success) {
                result.status = BoxQPStatus::HESSIAN_NOT_PD;
                break;
            }
            result.factorizations++;
        }

        // Check gradient norm
        double grad_norm = 0;
        for (int i = 0; i < n; ++i) {
            if (!clamped[i]) grad_norm += grad[i] * grad[i];
        }
        grad_norm = std::sqrt(grad_norm);
        result.final_grad_norm = grad_norm;

        if (grad_norm < options_.minGrad) {
            result.status = BoxQPStatus::SMALL_GRADIENT;
            break;
        }

        // Compute search direction
        Eigen::VectorXd search = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd grad_clamped = g;
        for (int i = 0; i < n; ++i) {
            if (clamped[i]) grad_clamped += H.col(i) * result.x[i];
        }

        // Extract free components
        std::vector<int> free_idx;
        Eigen::VectorXd grad_free;
        for (int i = 0; i < n; ++i) {
            if (!clamped[i]) {
                free_idx.push_back(i);
                grad_free.conservativeResize(free_idx.size());
                grad_free[free_idx.size()-1] = grad_clamped[i];
            }
        }

        // Solve system for free variables
        Eigen::VectorXd search_free = -result.Hfree.solve(grad_free);
        
        // Put solution back into full vector
        for (size_t i = 0; i < free_idx.size(); ++i) {
            search[free_idx[i]] = search_free[i] - result.x[free_idx[i]];
        }

        // Check descent direction
        double sdotg = search.dot(grad);
        if (sdotg >= 0) {
            result.status = BoxQPStatus::NO_DESCENT;
            break;
        }

        // Do line search
        auto [linesearch_success, step_info] = lineSearch(result.x, search, grad, value, H, g, lower, upper);
        if (!linesearch_success) {
            result.status = BoxQPStatus::MAX_LS_EXCEEDED;
            break;
        }

        // Accept step
        double step = step_info.first;
        result.x = step_info.second;
        value = evaluateObjective(result.x, H, g);
        
        if (options_.verbose) {
            printf("Iter %d: obj=%.6f |g|=%.6g step=%.2g\n", 
                   iter, value, grad_norm, step);
        }
    }

    result.final_value = value;
    return result;
}

Eigen::VectorXd BoxQPSolver::initializeX(const Eigen::VectorXd& x0,
                                        const Eigen::VectorXd& lower,
                                        const Eigen::VectorXd& upper,
                                        int n) {
    if (x0.size() == n) {
        return projectOntoBox(x0, lower, upper);
    }
    
    // Initialize at midpoint of bounds
    Eigen::VectorXd x(n);
    for (int i = 0; i < n; ++i) {
        if (std::isfinite(lower[i]) && std::isfinite(upper[i])) {
            x[i] = 0.5 * (lower[i] + upper[i]);
        } else if (std::isfinite(lower[i])) {
            x[i] = lower[i];
        } else if (std::isfinite(upper[i])) {
            x[i] = upper[i];
        } else {
            x[i] = 0.0;
        }
    }
    return x;
}

std::pair<bool, std::pair<double, Eigen::VectorXd>> BoxQPSolver::lineSearch(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& search,
    const Eigen::VectorXd& gradient,
    double value,
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& g,
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper) {
    
    double step = 1.0;
    const double sdotg = search.dot(gradient);
    
    while (step > options_.minStep) {
        // Compute candidate
        Eigen::VectorXd x_new = projectOntoBox(x + step * search, lower, upper);
        
        // Evaluate objective
        double value_new = evaluateObjective(x_new, H, g);
        
        // Check Armijo condition
        if ((value_new - value) <= options_.armijo * step * sdotg) {
            return {true, {step, x_new}};
        }
        
        step *= options_.stepDec;
    }
    
    return {false, {0.0, x}};
}

double BoxQPSolver::evaluateObjective(const Eigen::VectorXd& x,
                                    const Eigen::MatrixXd& H,
                                    const Eigen::VectorXd& g) const {
    return 0.5 * x.dot(H * x) + g.dot(x);
}

Eigen::VectorXd BoxQPSolver::projectOntoBox(const Eigen::VectorXd& x,
                                          const Eigen::VectorXd& lower,
                                          const Eigen::VectorXd& upper) const {
    Eigen::VectorXd x_proj = x;
    for (int i = 0; i < x.size(); ++i) {
        x_proj[i] = std::min(std::max(x[i], lower[i]), upper[i]);
    }
    return x_proj;
}


} // namespace cddp