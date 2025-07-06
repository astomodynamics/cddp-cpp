# MSIPDDP Terminal Constraints Implementation Guide

## Overview

The current MSIPDDP solver in `cddp-cpp` has incomplete support for terminal constraints. While the infrastructure exists (storage for terminal constraint variables, derivatives, and gains), the actual computation and update logic is missing in both the backward and forward passes.

## Current Implementation Gaps

### 1. Backward Pass Issues
- **Missing Terminal KKT System**: The `modifyTerminalValueFunction()` only adds constraint contributions to the value function but doesn't solve for terminal constraint gains
- **No Feedforward Gains Computation**: Terminal constraint feedforward gains (`k_y_ineq_terminal_`, `k_s_ineq_terminal_`, `k_y_eq_terminal_`) are initialized to zero but never computed
- **No Feedback Gains**: Terminal constraints don't have feedback gains storage or computation (which would be zero anyway since there's no control at terminal state)

### 2. Forward Pass Issues  
- **Missing Terminal Updates**: Terminal constraint dual/slack variables are not updated in the forward pass
- **Missing Terminal Evaluation**: Terminal constraints are not evaluated after the main loop
- **Missing Violation Computation**: Terminal constraint violations are not included in `constraint_violation_new` for filter line search
- **Missing Barrier Terms**: Terminal barrier terms are not added to the merit function

## Required Implementation

### 1. Backward Pass Modifications

#### A. Compute Terminal Constraint Gains

After modifying the terminal value function in `backwardPass()`, add:

```cpp
// After modifyTerminalValueFunction() call (around line 1489)
if (!terminal_constraint_set.empty()) {
    // Compute terminal constraint gains
    computeTerminalConstraintGains(context, V_x, V_xx);
}
```

#### B. New Method: `computeTerminalConstraintGains()`

```cpp
void MSIPDDPSolver::computeTerminalConstraintGains(CDDP &context,
                                                   const Eigen::VectorXd& V_x,
                                                   const Eigen::MatrixXd& V_xx) {
    const auto &terminal_constraint_set = context.getTerminalConstraintSet();
    const Eigen::VectorXd &x_terminal = context.X_.back();
    
    // For each terminal constraint
    for (const auto &constraint_pair : terminal_constraint_set) {
        const std::string &constraint_name = constraint_pair.first;
        const auto &constraint = constraint_pair.second;
        
        // Check if equality or inequality
        Eigen::VectorXd lower_bound = constraint->getLowerBound();
        Eigen::VectorXd upper_bound = constraint->getUpperBound();
        bool is_equality = (lower_bound - upper_bound).isZero(1e-10);
        
        if (is_equality) {
            // Equality constraint: h(x) = 0
            const Eigen::VectorXd &g = G_eq_terminal_[constraint_name];
            const Eigen::VectorXd &y = Y_eq_terminal_[constraint_name];
            const Eigen::MatrixXd &G_x = G_x_eq_terminal_[constraint_name];
            
            // Terminal KKT residual for equality constraint
            // ∇L = V_x + G_x^T * y (already incorporated in V_x)
            // Constraint residual = g
            
            // Feedforward gain for dual variable
            // k_y = -g (to drive constraint to zero)
            k_y_eq_terminal_[constraint_name] = -g;
            
        } else {
            // Inequality constraint: g(x) ≤ 0 with slack s
            const Eigen::VectorXd &g = G_ineq_terminal_[constraint_name];
            const Eigen::VectorXd &y = Y_ineq_terminal_[constraint_name];
            const Eigen::VectorXd &s = S_ineq_terminal_[constraint_name];
            const Eigen::MatrixXd &G_x = G_x_ineq_terminal_[constraint_name];
            
            int dual_dim = constraint->getDualDim();
            
            // Build terminal KKT system for inequality constraints
            // Similar to regular time steps but without control variables
            
            // Diagonal matrices
            Eigen::MatrixXd Y = y.asDiagonal();
            Eigen::MatrixXd S = s.asDiagonal();
            Eigen::MatrixXd S_inv = S.inverse();
            Eigen::MatrixXd YSinv = Y * S_inv;
            
            // Residuals
            Eigen::VectorXd primal_residual = g + s;
            Eigen::VectorXd complementary_residual = y.cwiseProduct(s).array() - mu_;
            Eigen::VectorXd rhat = y.cwiseProduct(primal_residual) - complementary_residual;
            
            // Compute gains
            k_y_ineq_terminal_[constraint_name] = S_inv * rhat;
            k_s_ineq_terminal_[constraint_name] = -primal_residual;
        }
    }
}
```

### 2. Forward Pass Modifications

#### A. Update Terminal Constraints

Add after line 2185 (after the main time loop):

```cpp
// Handle terminal constraints
if (!terminal_constraint_set.empty()) {
    const Eigen::VectorXd &x_terminal = result.state_trajectory.back();
    
    // Update terminal dual variables with fraction-to-boundary rule
    for (const auto &constraint_pair : terminal_constraint_set) {
        const std::string &constraint_name = constraint_pair.first;
        const auto &constraint = constraint_pair.second;
        
        // Check if equality or inequality
        Eigen::VectorXd lower_bound = constraint->getLowerBound();
        Eigen::VectorXd upper_bound = constraint->getUpperBound();
        bool is_equality = (lower_bound - upper_bound).isZero(1e-10);
        
        if (is_equality) {
            // Update equality dual variables (no fraction-to-boundary needed)
            Y_eq_terminal_[constraint_name] = Y_eq_terminal_[constraint_name] + 
                                              result.alpha_du * k_y_eq_terminal_[constraint_name];
            
            // Evaluate equality constraint
            G_eq_terminal_[constraint_name] = constraint->evaluate(x_terminal, Eigen::VectorXd()) - upper_bound;
            
            // Add to constraint violation
            constraint_violation_new += G_eq_terminal_[constraint_name].lpNorm<1>();
            
        } else {
            // Update inequality dual/slack with fraction-to-boundary
            const Eigen::VectorXd &y_old = Y_ineq_terminal_[constraint_name];
            const Eigen::VectorXd &s_old = S_ineq_terminal_[constraint_name];
            
            // Compute maximum feasible step for dual variables
            Eigen::VectorXd y_new = y_old + result.alpha_du * k_y_ineq_terminal_[constraint_name];
            Eigen::VectorXd y_min = (1.0 - tau) * y_old;
            
            // Check feasibility
            for (int i = 0; i < constraint->getDualDim(); ++i) {
                if (y_new[i] < y_min[i]) {
                    // Reduce step size if needed
                    y_new[i] = y_min[i];
                }
            }
            
            Y_ineq_terminal_[constraint_name] = y_new;
            
            // Update slack variables with fraction-to-boundary
            Eigen::VectorXd s_new = s_old + alpha * k_s_ineq_terminal_[constraint_name];
            Eigen::VectorXd s_min = (1.0 - tau) * s_old;
            
            for (int i = 0; i < constraint->getDualDim(); ++i) {
                if (s_new[i] < s_min[i]) {
                    s_new[i] = s_min[i];
                }
            }
            
            S_ineq_terminal_[constraint_name] = s_new;
            
            // Evaluate inequality constraint
            G_ineq_terminal_[constraint_name] = constraint->evaluate(x_terminal, Eigen::VectorXd()) - upper_bound;
            
            // Add barrier term to merit function
            merit_function_new -= mu_ * s_new.array().log().sum();
            
            // Add to constraint violation (primal infeasibility)
            Eigen::VectorXd primal_residual = G_ineq_terminal_[constraint_name] + s_new;
            constraint_violation_new += primal_residual.lpNorm<1>();
        }
    }
}
```

### 3. Additional Required Modifications

#### A. Update `computeMaxConstraintViolation()`

Add terminal constraint violations:

```cpp
// Add after regular constraint loop
for (const auto &constraint_pair : terminal_constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    
    if (G_eq_terminal_.count(constraint_name) > 0) {
        max_violation = std::max(max_violation, 
                                G_eq_terminal_[constraint_name].lpNorm<Eigen::Infinity>());
    }
    
    if (G_ineq_terminal_.count(constraint_name) > 0) {
        const Eigen::VectorXd primal_residual = 
            G_ineq_terminal_[constraint_name] + S_ineq_terminal_[constraint_name];
        max_violation = std::max(max_violation, 
                                primal_residual.lpNorm<Eigen::Infinity>());
    }
}
```

#### B. Update `computeScaledDualInfeasibility()`

Include terminal constraints in dual infeasibility computation:

```cpp
// Add after regular constraint loop
for (const auto &constraint_pair : terminal_constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    const auto &constraint = constraint_pair.second;
    
    // Get constraint gradient at terminal state
    const Eigen::VectorXd &x_terminal = context.X_.back();
    Eigen::MatrixXd G_x = constraint->getStateJacobian(x_terminal, Eigen::VectorXd());
    
    if (Y_eq_terminal_.count(constraint_name) > 0) {
        // Equality constraint gradient of Lagrangian
        Eigen::VectorXd grad_L = G_x.transpose() * Y_eq_terminal_[constraint_name];
        
        // Scale by dual variable magnitude
        double y_max = Y_eq_terminal_[constraint_name].lpNorm<Eigen::Infinity>();
        double scale = std::max(s_d, y_max / s_d);
        
        dual_infeasibility = std::max(dual_infeasibility, 
                                     grad_L.lpNorm<Eigen::Infinity>() / scale);
    }
    
    if (Y_ineq_terminal_.count(constraint_name) > 0) {
        // Similar for inequality constraints
        // Include barrier gradient contribution
    }
}
```

### 4. Storage Initialization

Ensure terminal constraint gains are properly initialized in `initializeConstraintStorage()`:

```cpp
// Add to existing initialization
for (const auto &constraint_pair : terminal_constraint_set) {
    const std::string &constraint_name = constraint_pair.first;
    int dual_dim = constraint_pair.second->getDualDim();
    
    // Initialize gains to zero
    k_y_ineq_terminal_[constraint_name] = Eigen::VectorXd::Zero(dual_dim);
    k_s_ineq_terminal_[constraint_name] = Eigen::VectorXd::Zero(dual_dim);
    k_y_eq_terminal_[constraint_name] = Eigen::VectorXd::Zero(dual_dim);
}
```

## Testing Recommendations

1. **Unit Tests**: Create tests with simple terminal constraints to verify:
   - Terminal constraint evaluation
   - Dual/slack variable updates
   - Constraint violation computation
   - Merit function calculation

2. **Integration Tests**: Test with problems that have:
   - Only terminal constraints
   - Mixed path and terminal constraints
   - Terminal equality and inequality constraints

3. **Numerical Tests**: Verify:
   - Convergence to feasible solutions
   - Proper barrier parameter reduction
   - Filter acceptance with terminal constraints

## Implementation Priority

1. **High Priority**: Forward pass terminal constraint updates and violation computation
2. **Medium Priority**: Backward pass terminal gain computation
3. **Low Priority**: Optimization of terminal constraint handling for special cases

## Notes

- Terminal constraints don't have control variables, simplifying the KKT system
- Terminal feedback gains would be zero (no control at terminal time)
- The implementation should maintain consistency with the regular time-step constraint handling
- Special care needed for numerical stability when slack variables approach zero