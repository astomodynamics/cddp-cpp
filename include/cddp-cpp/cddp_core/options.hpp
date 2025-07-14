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
#ifndef CDDP_OPTIONS_HPP
#define CDDP_OPTIONS_HPP

#include "cddp_core/boxqp.hpp"
#include <iostream> // For std::cout, std::cerr
#include <string>   // For std::string

namespace cddp
{
    /**
     * @brief Options for the line search procedure.
     *
     * These parameters control how the step size (alpha) is determined during the
     * forward pass to ensure sufficient decrease in the cost or merit function.
     */
    struct LineSearchOptions
    {
        int max_iterations =
            11; ///< Maximum number of backtracking steps in line search.
        double initial_step_size =
            1.0;                     ///< Initial step size (alpha) for line search.
        double min_step_size = 1e-8; ///< Minimum allowed step size in line search.
        double step_reduction_factor =
            0.5; ///< Factor by which to reduce alpha at each backtracking iteration.
    };

    /**
     * @brief Options for the regularization scheme.
     *
     * Regularization is used to ensure that the Hessian matrices in the backward
     * pass are positive definite, improving numerical stability.
     */
    struct RegularizationOptions
    {
        double initial_value = 1e-6; ///< Initial regularization value.
        double update_factor = 10.0; ///< Factor to increase/decrease regularization.
        double max_value = 1e7;      ///< Maximum allowed regularization value.
        double min_value = 1e-10;    ///< Minimum allowed regularization value.
        double step_initial_value =
            1.0; ///< Initial step/rate for changing regularization.
    };

    /**
     * @brief Options for the barrier mechanism within an Interior Point Method.
     * This struct will be instantiated within each solver's specific options if it
     * uses a barrier method.
     */
    struct SolverSpecificBarrierOptions
    {
        double mu_initial = 1e-0;    ///< Initial barrier coefficient (mu).
        double mu_min_value = 1e-10; ///< Minimum allowed value for mu.
        double mu_update_factor =
            0.5; ///< Factor to reduce mu when KKT error improves sufficiently.
        double mu_update_power =
            1.2; ///< Power for mu update rule (mu_new = mu^power or factor*mu).
        double min_fraction_to_boundary =
            0.99; ///< Minimum fraction to boundary for primal/dual step calculation
                  ///< (tau).
    };

    /**
     * @brief Options for the filter line search mechanism within an Interior Point
     * Method. This struct will be instantiated within each solver's specific
     * options if it uses a filter line search.
     */
    struct SolverSpecificFilterOptions
    {
        double merit_acceptance_threshold =
            1e-6; ///< Small value for merit filter acceptance.
        double violation_acceptance_threshold =
            1e-6; ///< Small value for violation filter acceptance.
        double max_violation_threshold =
            1e+4; ///< Maximum violation for filter acceptance.
        double min_violation_for_armijo_check =
            1e-7; ///< Min violation below which Armijo is checked on cost.
        double armijo_constant =
            1e-4; ///< Armijo constant (c1) for filter acceptance.
    };

    /**
     * @brief Options for the relaxed log-barrier method.
     * This struct will be instantiated within each solver's specific options if it
     * uses a relaxed log-barrier method.
     */
    struct LogBarrierOptions
    {
        bool use_relaxed_log_barrier_penalty =
            false; ///< Use relaxed log-barrier method (if applicable to the solver).
        double relaxed_log_barrier_delta =
            1e-10; ///< Relaxation delta for relaxed log-barrier (if applicable).

        // Multi-shooting specific
        int segment_length =
            5; ///< Number of shooting intervals before a gap-closing constraint.
        std::string rollout_type =
            "nonlinear"; ///< Rollout type: "nonlinear", "hybrid".
        bool use_controlled_rollout =
            true; ///< Use controlled rollout (propagates x_{k+1} = f(x_k, u_k) during
                  ///< initial rollout).

        SolverSpecificBarrierOptions
            barrier; ///< Barrier method parameters for relaxed log-barrier.
    };

    /**
     * @brief Comprehensive options specifically for the IPDDP (Interior-Point
     * Differential Dynamic Programming) algorithm. Includes its own instances of
     * barrier and filter parameters.
     */
    struct IPDDPAlgorithmOptions
    {
        // Initialization scales for duals/slacks
        double dual_var_init_scale = 1e-1;  ///< Initial scale for dual variables.
        double slack_var_init_scale = 1e-2; ///< Initial scale for slack variables.


        SolverSpecificBarrierOptions
            barrier; ///< Barrier method parameters for IPDDP.
    };

    /**
     * @brief Comprehensive options specifically for the MSIPDDP (Multi-Shooting
     * Interior-Point Differential Dynamic Programming) algorithm. Includes its own
     * instances of barrier and filter parameters, plus multi-shooting specifics.
     */
    struct MSIPDDPAlgorithmOptions
    {
        // Initialization scales
        double dual_var_init_scale = 1e-1;  ///< Initial scale for dual variables.
        double slack_var_init_scale = 1e-2; ///< Initial scale for slack variables.
        double costate_var_init_scale =
            1e-6; ///< Initial scale for costate variables.

        // Multi-shooting specific
        int segment_length =
            5; ///< Number of shooting intervals before a gap-closing constraint.
        std::string rollout_type =
            "nonlinear"; ///< Rollout type: "nonlinear", "hybrid".
        bool use_controlled_rollout =
            false; ///< Use controlled rollout (propagates x_{k+1} = f(x_k, u_k)
                   ///< during initial rollout).

        // Predictor-corrector options
        bool use_predictor_corrector = true; ///< Enable Mehrotra predictor-corrector step (experimental - debugging numerical stability).
        double predictor_fraction_to_boundary = 1e-3; ///< Safety margin for predictor step length.

        SolverSpecificBarrierOptions
            barrier; ///< Barrier method parameters for MSIPDDP..
    };

    /**
     * @brief Options specifically for TC-MSIPDDP algorithm. (Placeholder)
     */
    struct TCMSIPDDPAlgorithmOptions
    {
        // Initialization scales
        double dual_var_init_scale = 1e-1;    ///< Initial scale for dual variables
        double slack_var_init_scale = 1e-2;   ///< Initial scale for slack variables
        double costate_var_init_scale = 1e-6; ///< Initial scale for costate variables

        // Terminal constraint specific
        double terminal_dual_init_scale = 1e-1;      ///< Initial scale for terminal dual variables
        double terminal_slack_init_scale = 1e-2;     ///< Initial scale for terminal slack variables
        double terminal_constraint_tolerance = 1e-6; ///< Tolerance for terminal constraint satisfaction

        // Multi-shooting specific
        int segment_length = 5;                 ///< Number of shooting intervals
        std::string rollout_type = "nonlinear"; ///< Rollout type: "nonlinear", "hybrid"
        bool use_controlled_rollout = false;    ///< Use controlled rollout

        // Barrier and filter options
        SolverSpecificBarrierOptions barrier; ///< Barrier method parameters
    };

    /**
     * @brief Comprehensive options specifically for the ALTRO (Augmented Lagrangian
     * Trajectory Optimizer) algorithm. ALTRO uses augmented Lagrangian methods to
     * handle constraints through penalty terms and dual variable updates.
     */
    struct AltroAlgorithmOptions
    {
        // Penalty parameters
        double penalty_scaling = 10.0; ///< Initial penalty scaling parameter (rho)
                                       ///< for augmented Lagrangian.
        double penalty_scaling_increase_factor =
            10.0; ///< Factor to increase penalty when constraints are violated.
        double penalty_scaling_max =
            1e8; ///< Maximum allowed penalty scaling parameter.
        double penalty_scaling_min =
            1e-6; ///< Minimum allowed penalty scaling parameter.

        // Dual variable parameters
        double dual_var_init_scale =
            0.1;                         ///< Initial scale for dual variables (Lagrange multipliers).
        double dual_var_max = 1e6;       ///< Maximum allowed magnitude for dual variables.
        double dual_var_min = -1e6;      ///< Minimum allowed magnitude for dual variables.
        double dual_update_factor = 1.0; ///< Factor for dual variable updates.

        // Defect constraint parameters
        double defect_dual_init_scale =
            0.01; ///< Initial scale for defect constraint dual variables.
        double defect_penalty_scaling =
            1.0; ///< Penalty scaling for dynamics defect constraints.

        // Convergence parameters
        double constraint_tolerance = 1e-6;       ///< Tolerance for constraint violation.
        double dual_feasibility_tolerance = 1e-6; ///< Tolerance for dual feasibility.
        double complementarity_tolerance =
            1e-6; ///< Tolerance for complementarity conditions.

        // Penalty update strategy
        bool adaptive_penalty_update =
            true; ///< Use adaptive penalty parameter updates.
        double penalty_update_threshold =
            0.25; ///< Threshold for penalty parameter updates (relative to constraint
                  ///< violation reduction).
        int max_penalty_increases =
            5; ///< Maximum number of consecutive penalty increases per iteration.

        // AL-specific convergence criteria
        double al_convergence_tolerance =
            1e-4; ///< Convergence tolerance for augmented Lagrangian subproblems.
        int max_al_iterations =
            1; ///< Maximum iterations for augmented Lagrangian outer loop.
        bool use_constraint_norm_termination =
            true; ///< Use constraint norm for termination criteria.

        // Multiple-shooting parameters
        bool use_multiple_shooting =
            false; ///< Enable multiple-shooting approach for forward pass.
        int segment_length =
            5; ///< Number of shooting intervals before a gap-closing constraint.
        std::string rollout_type =
            "nonlinear"; ///< Rollout type: "nonlinear", "hybrid".
    };

    /**
     * @brief Comprehensive options specifically for the DBAS-DDP (Discrete Barrier
     * State DDP) algorithm. DBAS-DDP augments the state space with explicit barrier
     * states to handle inequality constraints.
     *
     * Key Design Principles:
     * - Barrier states evolve as: s_{k+1} = decay_factor * s_k + weight * violation
     * * dt
     * - Cost includes: original_cost + barrier_log_terms + barrier_state_penalty
     * - Requires feasible initial guess since state space is augmented
     */
    struct DbasDdpAlgorithmOptions
    {
        // Barrier state initialization parameters
        double barrier_state_init_value =
            0.1; ///< Initial value for barrier state variables (smaller values
                 ///< encourage faster constraint satisfaction).
        double barrier_state_weight =
            10.0; ///< Weight for barrier state dynamics penalty in the cost function
                  ///< (higher values respond faster to violations).
        double barrier_state_decay_rate =
            0.05; ///< Decay rate for barrier state evolution dynamics (smaller values
                  ///< provide longer memory).

        // Adaptive parameters
        bool use_adaptive_barrier_weights =
            true; ///< Enable adaptive adjustment of barrier state weights based on
                  ///< constraint satisfaction.
        double barrier_weight_increase_factor =
            1.5; ///< Factor to increase barrier weights when constraints are
                 ///< consistently violated.
        double barrier_weight_decrease_factor =
            0.9; ///< Factor to decrease barrier weights when making consistent
                 ///< progress.
        double barrier_weight_max =
            1e4;                          ///< Maximum allowed barrier weight (prevents numerical issues).
        double barrier_weight_min = 1e-2; ///< Minimum allowed barrier weight
                                          ///< (maintains constraint awareness).

        // Log-barrier integration parameters
        double mu_initial = 1e-2; ///< Initial barrier coefficient for log-barrier
                                  ///< terms (smaller values are more aggressive).
        double mu_min_value =
            1e-6; ///< Minimum allowed value for barrier coefficient.
        double mu_update_factor =
            0.2; ///< Factor to reduce barrier coefficient (smaller values reduce mu
                 ///< more aggressively).
        double relaxed_log_barrier_delta =
            1e-4; ///< Relaxation delta for relaxed log-barrier method (balance
                  ///< between accuracy and robustness).

        // Constraint violation handling
        double constraint_violation_tolerance =
            1e-6; ///< Tolerance for constraint violations in barrier state updates.
        double barrier_state_convergence_tol =
            1e-4;                              ///< Convergence tolerance for barrier state changes (when to
                                               ///< consider barrier states converged).
        double max_barrier_state_norm = 100.0; ///< Maximum allowed norm for barrier
                                               ///< states (prevents runaway growth).

        // Cost function parameters
        bool penalize_barrier_state_deviation =
            true; ///< Penalize deviation of barrier states from zero in the cost
                  ///< function.
        double barrier_state_reference_weight =
            1.0; ///< Weight for barrier state reference tracking cost (drives barrier
                 ///< states toward zero).

        // Numerical stability parameters
        double min_barrier_state_value =
            1e-8; ///< Minimum allowed value for barrier state variables (prevents
                  ///< numerical issues).
        double max_barrier_state_value =
            1e6; ///< Maximum allowed value for barrier state variables (prevents
                 ///< overflow).
        bool enable_barrier_state_regularization =
            true; ///< Add small regularization to barrier state dynamics for
                  ///< numerical stability.
        double barrier_state_regularization =
            1e-6; ///< Regularization value for barrier state dynamics.
    };

    /**
     * @brief Main options structure for the CDDP solver.
     *
     * This structure holds all configurable parameters for the CDDP algorithm and
     * its variants.
     */
    struct CDDPOptions
    {
        // General Solver Configuration
        double tolerance =
            1e-5; ///< Tolerance for KKT/optimality error magnitude for convergence.
        double acceptable_tolerance =
            1e-6;               ///< Tolerance for changes in cost function for convergence.
        int max_iterations = 1; ///< Maximum number of CDDP iterations.
        double max_cpu_time =
            0.0;             ///< Maximum CPU time for the solver in seconds (0 for unlimited).
        bool verbose = true; ///< Enable verbose printing during solving.
        bool debug = false;  ///< Enable additional debug prints.
        bool print_solver_header_footer =
            true; ///< Print header and footer summary for the solve.
        bool use_ilqr =
            true; ///< Use iLQR (ignore second-order dynamics derivatives).
        bool enable_parallel =
            false;           ///< Enable parallel computation for line search.
        int num_threads = 1; ///< Number of threads for parallel computation.
        bool return_iteration_info =
            false; ///< Return detailed iteration info in the solution.
        bool warm_start =
            false; ///< Use warm start (preserve existing solver state and gains).
        double termination_scaling_max_factor =
            100.0; ///< Max scaling factor for KKT error in termination.

        // General sub-configurations (used by potentially multiple solvers)
        LineSearchOptions line_search;        ///< General line search parameters.
        RegularizationOptions regularization; ///< General regularization parameters.
        BoxQPOptions box_qp;                  ///< General BoxQP solver parameters.
        SolverSpecificFilterOptions
            filter; ///< General filter line search parameters.

        // Solver-specific comprehensive options
        LogBarrierOptions
            log_barrier;             ///< Comprehensive options for the log-barrier method.
        IPDDPAlgorithmOptions ipddp; ///< Comprehensive options for the IPDDP solver.
        MSIPDDPAlgorithmOptions
            msipddp;                 ///< Comprehensive options for the MSIPDDP solver.
        AltroAlgorithmOptions altro; ///< Comprehensive options for the ALTRO solver.
        DbasDdpAlgorithmOptions
            dbas_ddp; ///< Comprehensive options for the DBAS-DDP solver.

        // Constructor with defaults (relies on member initializers)
        CDDPOptions() = default;
    };

} // namespace cddp
#endif // CDDP_OPTIONS_HPP