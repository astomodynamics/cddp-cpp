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
     * @brief Barrier parameter update strategy for interior point methods.
     */
    enum class BarrierStrategy
    {
        ADAPTIVE,    ///< Adaptive strategy based on KKT progress (default)
        MONOTONIC,   ///< Monotonic decrease with fixed reduction factor
        IPOPT        ///< IPOPT-style adaptive barrier update
    };

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
        BarrierStrategy strategy = 
            BarrierStrategy::ADAPTIVE; ///< Barrier parameter update strategy.
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
     * @brief Common options for interior-point solvers (dual/slack initialization + barrier).
     */
    struct InteriorPointOptions
    {
        double dual_var_init_scale = 1e-1;  ///< Initial scale for dual variables.
        double slack_var_init_scale = 1e-2; ///< Initial scale for slack variables.
        SolverSpecificBarrierOptions barrier; ///< Barrier method parameters.
    };

    /**
     * @brief Common options for multi-shooting solvers.
     */
    struct MultiShootingOptions
    {
        int segment_length =
            5; ///< Number of shooting intervals before a gap-closing constraint.
        std::string rollout_type =
            "nonlinear"; ///< Rollout type: "nonlinear", "hybrid".
        bool use_controlled_rollout =
            false; ///< Use controlled rollout during initial rollout.
        double costate_var_init_scale =
            1e-6; ///< Initial scale for costate variables.
    };

    /**
     * @brief Options for the relaxed log-barrier method (single-shooting).
     */
    struct LogBarrierOptions
    {
        bool use_relaxed_log_barrier_penalty =
            false; ///< Use relaxed log-barrier method.
        double relaxed_log_barrier_delta =
            1e-10; ///< Relaxation delta for relaxed log-barrier.
        SolverSpecificBarrierOptions
            barrier; ///< Barrier method parameters for relaxed log-barrier.
    };

    /**
     * @brief Options for the IPDDP solver.
     */
    struct IPDDPAlgorithmOptions : InteriorPointOptions {};

    /**
     * @brief Options for the MSIPDDP solver (interior-point + multi-shooting).
     */
    struct MSIPDDPAlgorithmOptions : InteriorPointOptions, MultiShootingOptions {};

    /**
     * @brief Options for the TC-MSIPDDP solver (terminal-constrained multi-shooting).
     */
    struct TCMSIPDDPAlgorithmOptions : InteriorPointOptions, MultiShootingOptions
    {
        double terminal_dual_init_scale = 1e-1;      ///< Initial scale for terminal dual variables.
        double terminal_slack_init_scale = 1e-2;     ///< Initial scale for terminal slack variables.
        double terminal_constraint_tolerance = 1e-6; ///< Tolerance for terminal constraint satisfaction.
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
        bool print_solver_header =
            true; ///< Print solver header banner.
        bool print_solver_options = false; ///< Print solver options. 
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

        // Constructor with defaults (relies on member initializers)
        CDDPOptions() = default;
    };

} // namespace cddp
#endif // CDDP_OPTIONS_HPP
