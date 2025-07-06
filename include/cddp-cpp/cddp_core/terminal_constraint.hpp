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

#ifndef CDDP_TERMINAL_CONSTRAINT_HPP
#define CDDP_TERMINAL_CONSTRAINT_HPP

#include "constraint.hpp"
#include <Eigen/Dense>
#include <string>
#include <limits>
#include <vector> // Added for std::vector
#include <stdexcept> // Added for std::logic_error

namespace cddp 
{
    class TerminalConstraint : public Constraint
    {
    public:
        TerminalConstraint(const std::string &name) : Constraint(name) {}

        Eigen::MatrixXd getControlJacobian(
            const Eigen::VectorXd &/*state*/,
            const Eigen::VectorXd &control // control.size() gives control_dim
        ) const override 
        {
            return Eigen::MatrixXd::Zero(getDualDim(), control.size());
        }

        std::vector<Eigen::MatrixXd> getControlHessian(
            const Eigen::VectorXd &/*state*/,
            const Eigen::VectorXd &control
        ) const override
        {
            return {};
        }

        std::vector<Eigen::MatrixXd> getCrossHessian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &control
        ) const override
        {
            return {};
        }
    };

    class TerminalEqualityConstraint : public TerminalConstraint
    {
    public:
        TerminalEqualityConstraint(const Eigen::VectorXd &target_state, 
                                 const std::string& name = "TerminalEqualityConstraint") // Added name parameter with default
            : TerminalConstraint(name), // Pass name to TerminalConstraint constructor
              target_state_(target_state) {}

        int getDualDim() const override
        {
            return target_state_.size();
        }

        Eigen::VectorXd evaluate(const Eigen::VectorXd &final_state,
                                 const Eigen::VectorXd &/*control_is_ignored*/) const override
        {
            if (final_state.size() != target_state_.size()) {
                throw std::invalid_argument("TerminalEqualityConstraint: final_state dimension mismatch.");
            }
            return final_state - target_state_;
        }

        Eigen::VectorXd evaluate(const Eigen::VectorXd &final_state) const
        {
            // Call the 2-argument version with a dummy control vector
            return evaluate(final_state, Eigen::VectorXd()); 
        }

        Eigen::VectorXd getLowerBound() const override
        {
            return Eigen::VectorXd::Zero(target_state_.size());
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return Eigen::VectorXd::Zero(target_state_.size());
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &final_state,
                                         const Eigen::VectorXd &/*control_is_ignored*/) const override
        {
            if (final_state.size() != target_state_.size()) {
                throw std::invalid_argument("TerminalEqualityConstraint: final_state dimension mismatch for Jacobian.");
            }
            return Eigen::MatrixXd::Identity(target_state_.size(), final_state.size());
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &final_state) const
        {
            return getStateJacobian(final_state, Eigen::VectorXd());
        }
        
        double computeViolation(const Eigen::VectorXd &final_state,
                                const Eigen::VectorXd &control) const override
        {
            Eigen::VectorXd g = evaluate(final_state, control);
            return computeViolationFromValue(g);
        }

        double computeViolation(const Eigen::VectorXd &final_state) const
        {
            Eigen::VectorXd g = evaluate(final_state); // Calls 1-arg evaluate
            return computeViolationFromValue(g);
        }

        double computeViolationFromValue(const Eigen::VectorXd &g) const override
        {
            // For equality g(x_N) = 0, violation is ||g(x_N)||
            return g.norm(); 
        }

        // Hessians for g(x_N) = x_N - target are zero
        std::vector<Eigen::MatrixXd> getStateHessian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &/*control_is_ignored*/
        ) const override
        {
            std::vector<Eigen::MatrixXd> Hxx_list;
            for (int i = 0; i < getDualDim(); ++i) {
                Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
            }
            return Hxx_list;
        }

        // New 1-argument getStateHessian
        std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state) const
        {
            return getStateHessian(state, Eigen::VectorXd());
        }

    protected:
        Eigen::VectorXd target_state_;
    };

    class TerminalInequalityConstraint : public TerminalConstraint
    {
    public:
        // Constraint of the form A_N * x_N - b_N <= 0
        TerminalInequalityConstraint(const Eigen::MatrixXd &A_N,
                                          const Eigen::VectorXd &b_N)
            : TerminalConstraint("TerminalInequalityConstraint"),
              A_N_(A_N),
              b_N_(b_N)
        {
            if (A_N.rows() != b_N.size()) {
                throw std::invalid_argument("TerminalInequalityConstraint: A_N rows and b_N size mismatch.");
            }
        }

        int getDualDim() const override
        {
            return A_N_.rows();
        }

        Eigen::VectorXd evaluate(const Eigen::VectorXd &final_state,
                                 const Eigen::VectorXd &/*control_is_ignored*/) const override
        {
            if (final_state.size() != A_N_.cols()) {
                throw std::invalid_argument("TerminalInequalityConstraint: final_state dimension and A_N columns mismatch.");
            }
            return A_N_ * final_state - b_N_;
        }

        Eigen::VectorXd evaluate(const Eigen::VectorXd &final_state) const
        {
            // Call the 2-argument version with a dummy control vector
            return evaluate(final_state, Eigen::VectorXd());
        }

        Eigen::VectorXd getLowerBound() const override
        {
            return Eigen::VectorXd::Constant(A_N_.rows(), -std::numeric_limits<double>::infinity());
        }

        Eigen::VectorXd getUpperBound() const override
        {
            return Eigen::VectorXd::Zero(A_N_.rows());
        }

        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &final_state,
                                         const Eigen::VectorXd &/*control_is_ignored*/) const override
        {
             if (final_state.size() != A_N_.cols()) {
                throw std::invalid_argument("TerminalStateInequalityConstraint: final_state dimension for Jacobian and A_N columns mismatch.");
            }
            return A_N_;
        }
        
        Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &final_state) const
        {
            return getStateJacobian(final_state, Eigen::VectorXd());
        }

        double computeViolation(const Eigen::VectorXd &final_state,
                                const Eigen::VectorXd &control) const override
        {
            Eigen::VectorXd g = evaluate(final_state, control);
            return computeViolationFromValue(g);
        }

        double computeViolation(const Eigen::VectorXd &final_state) const
        {
            Eigen::VectorXd g = evaluate(final_state); // Calls 1-arg evaluate
            return computeViolationFromValue(g);
        }

        double computeViolationFromValue(const Eigen::VectorXd &g) const override
        {
            // For inequality g(x_N) <= 0, violation is sum of positive parts of g(x_N)
            return g.cwiseMax(0.0).sum();
        }

        // Hessians for g(x_N) = A_N * x_N - b_N are zero
        std::vector<Eigen::MatrixXd> getStateHessian(
            const Eigen::VectorXd &state,
            const Eigen::VectorXd &/*control_is_ignored*/
        ) const override
        {
            std::vector<Eigen::MatrixXd> Hxx_list;
            for (int i = 0; i < getDualDim(); ++i) {
                Hxx_list.push_back(Eigen::MatrixXd::Zero(state.size(), state.size()));
            }
            return Hxx_list;
        }

        std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state) const
        {
            return getStateHessian(state, Eigen::VectorXd());
        }

    private:
        Eigen::MatrixXd A_N_;
        Eigen::VectorXd b_N_;
    };
} // namespace cddp
#endif // CDDP_TERMINAL_CONSTRAINT_HPP