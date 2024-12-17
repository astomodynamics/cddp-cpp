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

#include "cddp_core/qp_solver.hpp"
#include <chrono>

namespace cddp {

QPSolver::QPSolver(const QPSolverOptions& options) 
    : options_(options), 
      num_vars_(0), 
      num_constraints_(0),
      rng_(options.random_seed) {}

void QPSolver::setDimensions(int num_vars, int num_constraints) {
    num_vars_ = num_vars;
    num_constraints_ = num_constraints;
    
    // Pre-allocate matrices
    Q_.resize(num_vars_, num_vars_);
    c_.resize(num_vars_);
    A_.resize(num_constraints_, num_vars_);
    b_.resize(num_constraints_);
    
    // Pre-allocate workspace
    halves_.resize(num_vars_ + 1, num_constraints_);
    work_vectors_.resize((num_constraints_ + 2) * (num_vars_ + 2) * (num_vars_ - 1) / 2 + 1 - num_vars_);
    work_indices_.resize(2 * num_constraints_ + 1);
}

void QPSolver::setHessian(const Eigen::MatrixXd& Q) {
    Q_ = Q;
}

void QPSolver::setGradient(const Eigen::VectorXd& c) {
    c_ = c;
}

void QPSolver::setConstraints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    A_ = A;
    b_ = b;
}

QPResult QPSolver::solve() {
    QPResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Check problem dimensions
    if (num_vars_ <= 0 || num_constraints_ <= 0) {
        result.status = QPStatus::NUMERICAL_ERROR;
        return result;
    }

    // Initialize solution vector
    result.x.resize(num_vars_);
    result.x.setZero();

    // Compute Cholesky factorization of Q
    Eigen::LLT<Eigen::MatrixXd> llt(Q_);
    if (llt.info() != Eigen::Success) {
        result.status = QPStatus::NUMERICAL_ERROR;
        return result;
    }

    // Transform problem using Cholesky factorization
    const Eigen::MatrixXd As = llt.matrixU().solve<Eigen::OnTheRight>(A_);
    const Eigen::VectorXd v = llt.solve(c_);
    const Eigen::VectorXd bs = A_ * v + b_;

    // Scale rows of A
    const Eigen::VectorXd scale = As.rowwise().norm();
    halves_.topRows(As.cols()) = (As.array().colwise() / scale.array()).transpose();
    halves_.bottomRows(1) = (-bs.array() / scale.array()).transpose();

    // Solve minimum norm problem
    QPStatus status = solveMinNorm(result.x);

    if (status == QPStatus::OPTIMAL) {
        // Transform solution back
        llt.matrixU().solveInPlace(result.x);
        result.x -= v;
        result.objective_value = 0.5 * result.x.dot(Q_ * result.x) + c_.dot(result.x);
    } else {
        result.objective_value = std::numeric_limits<double>::infinity();
    }

    result.status = status;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.solve_time = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

QPStatus QPSolver::solveMinNorm(Eigen::VectorXd& x) {
    x.setZero();
    
    if (num_constraints_ < 1) {
        return QPStatus::OPTIMAL;
    }

    // Initialize linked list for active set management
    Eigen::VectorXi next = Eigen::VectorXi::Zero(num_constraints_);
    Eigen::VectorXi prev = Eigen::VectorXi::Zero(num_constraints_ + 1);
    
    if (num_constraints_ > 1) {
        Eigen::VectorXi perm(num_constraints_ - 1);
        generateRandomPermutation(num_constraints_ - 1, perm);

        prev(0) = 0;
        next(0) = perm(0) + 1;
        prev(perm(0) + 1) = 0;
        
        for (int i = 0; i < num_constraints_ - 2; ++i) {
            next(perm(i) + 1) = perm(i + 1) + 1;
            prev(perm(i + 1) + 1) = perm(i) + 1;
        }
        next(perm(num_constraints_ - 2) + 1) = num_constraints_;
    } else {
        prev(0) = 0;
        next(0) = 1;
        next(1) = 1;
    }

    // Main minimum norm algorithm
    for (int i = 0; i != num_constraints_; i = next(i)) {
        const Eigen::VectorXd plane_i = halves_.col(i).head(num_vars_);
        const double bi = halves_(num_vars_, i);

        if (x.dot(plane_i) + bi > (num_vars_ + 1) * options_.eps) {
            const double s = plane_i.squaredNorm();

            if (s < (num_vars_ + 1) * options_.eps * options_.eps) {
                return QPStatus::INFEASIBLE;
            }

            x = -bi * plane_i / s;

            if (i == 0) continue;

            // Householder reflection with pivoting
            const int id = x.array().abs().maxCoeff();  // Fixed line
            const double xnorm = x.norm();
            
            Eigen::VectorXd reflx = x;
            reflx(id) += x(id) < 0.0 ? -xnorm : xnorm;
            const double h = -2.0 / reflx.squaredNorm();

            // Update remaining constraints
            for (int j = 0; j != i; j = next(j)) {
                // Create single block expression that can be reused
                auto new_plane = halves_.col(j).head(num_vars_);
                const double coeff = h * new_plane.dot(reflx);
                new_plane += reflx * coeff;
            }

            i = moveToFront(i, next, prev);
        }
    }

    return QPStatus::OPTIMAL;
}

void QPSolver::generateRandomPermutation(int n, Eigen::VectorXi& perm) {
    perm.resize(n);
    for (int i = 0; i < n; ++i) {
        perm(i) = i;
    }
    
    for (int i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng_);
        std::swap(perm(i), perm(j));
    }
}

int QPSolver::moveToFront(int i, Eigen::VectorXi& next, Eigen::VectorXi& prev) {
    if (i == 0 || i == next(0)) {
        return i;
    }
    
    const int previ = prev(i);
    next(prev(i)) = next(i);
    prev(next(i)) = prev(i);
    next(i) = next(0);
    prev(i) = 0;
    prev(next(i)) = i;
    next(0) = i;
    
    return previ;
}

} // namespace cddp