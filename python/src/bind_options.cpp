#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cddp_core/options.hpp"
#include "cddp_core/cddp_core.hpp"

namespace py = pybind11;

void bind_options(py::module_& m) {
    // BarrierStrategy enum
    py::enum_<cddp::BarrierStrategy>(m, "BarrierStrategy")
        .value("ADAPTIVE", cddp::BarrierStrategy::ADAPTIVE)
        .value("MONOTONIC", cddp::BarrierStrategy::MONOTONIC)
        .value("IPOPT", cddp::BarrierStrategy::IPOPT);

    // SolverType enum
    py::enum_<cddp::SolverType>(m, "SolverType")
        .value("CLDDP", cddp::SolverType::CLDDP)
        .value("LogDDP", cddp::SolverType::LogDDP)
        .value("IPDDP", cddp::SolverType::IPDDP)
        .value("MSIPDDP", cddp::SolverType::MSIPDDP);

    // BoxQPOptions
    py::class_<cddp::BoxQPOptions>(m, "BoxQPOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &cddp::BoxQPOptions::max_iterations)
        .def_readwrite("min_gradient_norm", &cddp::BoxQPOptions::min_gradient_norm)
        .def_readwrite("min_relative_improvement", &cddp::BoxQPOptions::min_relative_improvement)
        .def_readwrite("step_decrease_factor", &cddp::BoxQPOptions::step_decrease_factor)
        .def_readwrite("min_step_size", &cddp::BoxQPOptions::min_step_size)
        .def_readwrite("armijo_constant", &cddp::BoxQPOptions::armijo_constant)
        .def_readwrite("verbose", &cddp::BoxQPOptions::verbose);

    // LineSearchOptions
    py::class_<cddp::LineSearchOptions>(m, "LineSearchOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &cddp::LineSearchOptions::max_iterations)
        .def_readwrite("initial_step_size", &cddp::LineSearchOptions::initial_step_size)
        .def_readwrite("min_step_size", &cddp::LineSearchOptions::min_step_size)
        .def_readwrite("step_reduction_factor", &cddp::LineSearchOptions::step_reduction_factor);

    // RegularizationOptions
    py::class_<cddp::RegularizationOptions>(m, "RegularizationOptions")
        .def(py::init<>())
        .def_readwrite("initial_value", &cddp::RegularizationOptions::initial_value)
        .def_readwrite("update_factor", &cddp::RegularizationOptions::update_factor)
        .def_readwrite("max_value", &cddp::RegularizationOptions::max_value)
        .def_readwrite("min_value", &cddp::RegularizationOptions::min_value)
        .def_readwrite("step_initial_value", &cddp::RegularizationOptions::step_initial_value);

    // SolverSpecificBarrierOptions
    py::class_<cddp::SolverSpecificBarrierOptions>(m, "BarrierOptions")
        .def(py::init<>())
        .def_readwrite("mu_initial", &cddp::SolverSpecificBarrierOptions::mu_initial)
        .def_readwrite("mu_min_value", &cddp::SolverSpecificBarrierOptions::mu_min_value)
        .def_readwrite("mu_update_factor", &cddp::SolverSpecificBarrierOptions::mu_update_factor)
        .def_readwrite("mu_update_power", &cddp::SolverSpecificBarrierOptions::mu_update_power)
        .def_readwrite("min_fraction_to_boundary", &cddp::SolverSpecificBarrierOptions::min_fraction_to_boundary)
        .def_readwrite("strategy", &cddp::SolverSpecificBarrierOptions::strategy);

    // SolverSpecificFilterOptions
    py::class_<cddp::SolverSpecificFilterOptions>(m, "FilterOptions")
        .def(py::init<>())
        .def_readwrite("merit_acceptance_threshold", &cddp::SolverSpecificFilterOptions::merit_acceptance_threshold)
        .def_readwrite("violation_acceptance_threshold", &cddp::SolverSpecificFilterOptions::violation_acceptance_threshold)
        .def_readwrite("max_violation_threshold", &cddp::SolverSpecificFilterOptions::max_violation_threshold)
        .def_readwrite("min_violation_for_armijo_check", &cddp::SolverSpecificFilterOptions::min_violation_for_armijo_check)
        .def_readwrite("armijo_constant", &cddp::SolverSpecificFilterOptions::armijo_constant);

    // InteriorPointOptions
    py::class_<cddp::InteriorPointOptions>(m, "InteriorPointOptions")
        .def(py::init<>())
        .def_readwrite("dual_var_init_scale", &cddp::InteriorPointOptions::dual_var_init_scale)
        .def_readwrite("slack_var_init_scale", &cddp::InteriorPointOptions::slack_var_init_scale)
        .def_readwrite("barrier", &cddp::InteriorPointOptions::barrier);

    // LogBarrierOptions (inherits MultiShootingOptions)
    py::class_<cddp::LogBarrierOptions>(m, "LogBarrierOptions")
        .def(py::init<>())
        .def_readwrite("segment_length", &cddp::LogBarrierOptions::segment_length)
        .def_readwrite("rollout_type", &cddp::LogBarrierOptions::rollout_type)
        .def_readwrite("use_controlled_rollout", &cddp::LogBarrierOptions::use_controlled_rollout)
        .def_readwrite("costate_var_init_scale", &cddp::LogBarrierOptions::costate_var_init_scale)
        .def_readwrite("use_relaxed_log_barrier_penalty", &cddp::LogBarrierOptions::use_relaxed_log_barrier_penalty)
        .def_readwrite("relaxed_log_barrier_delta", &cddp::LogBarrierOptions::relaxed_log_barrier_delta)
        .def_readwrite("barrier", &cddp::LogBarrierOptions::barrier);

    // IPDDPAlgorithmOptions (inherits InteriorPointOptions)
    py::class_<cddp::IPDDPAlgorithmOptions>(m, "IPDDPOptions")
        .def(py::init<>())
        .def_readwrite("dual_var_init_scale", &cddp::IPDDPAlgorithmOptions::dual_var_init_scale)
        .def_readwrite("slack_var_init_scale", &cddp::IPDDPAlgorithmOptions::slack_var_init_scale)
        .def_readwrite("barrier", &cddp::IPDDPAlgorithmOptions::barrier);

    // MSIPDDPAlgorithmOptions (inherits InteriorPointOptions + MultiShootingOptions)
    py::class_<cddp::MSIPDDPAlgorithmOptions>(m, "MSIPDDPOptions")
        .def(py::init<>())
        .def_readwrite("dual_var_init_scale", &cddp::MSIPDDPAlgorithmOptions::dual_var_init_scale)
        .def_readwrite("slack_var_init_scale", &cddp::MSIPDDPAlgorithmOptions::slack_var_init_scale)
        .def_readwrite("barrier", &cddp::MSIPDDPAlgorithmOptions::barrier)
        .def_readwrite("segment_length", &cddp::MSIPDDPAlgorithmOptions::segment_length)
        .def_readwrite("rollout_type", &cddp::MSIPDDPAlgorithmOptions::rollout_type)
        .def_readwrite("use_controlled_rollout", &cddp::MSIPDDPAlgorithmOptions::use_controlled_rollout)
        .def_readwrite("costate_var_init_scale", &cddp::MSIPDDPAlgorithmOptions::costate_var_init_scale);

    // CDDPOptions
    py::class_<cddp::CDDPOptions>(m, "CDDPOptions")
        .def(py::init<>())
        .def_readwrite("tolerance", &cddp::CDDPOptions::tolerance)
        .def_readwrite("acceptable_tolerance", &cddp::CDDPOptions::acceptable_tolerance)
        .def_readwrite("max_iterations", &cddp::CDDPOptions::max_iterations)
        .def_readwrite("max_cpu_time", &cddp::CDDPOptions::max_cpu_time)
        .def_readwrite("verbose", &cddp::CDDPOptions::verbose)
        .def_readwrite("debug", &cddp::CDDPOptions::debug)
        .def_readwrite("print_solver_header", &cddp::CDDPOptions::print_solver_header)
        .def_readwrite("print_solver_options", &cddp::CDDPOptions::print_solver_options)
        .def_readwrite("use_ilqr", &cddp::CDDPOptions::use_ilqr)
        .def_readwrite("enable_parallel", &cddp::CDDPOptions::enable_parallel)
        .def_readwrite("num_threads", &cddp::CDDPOptions::num_threads)
        .def_readwrite("return_iteration_info", &cddp::CDDPOptions::return_iteration_info)
        .def_readwrite("warm_start", &cddp::CDDPOptions::warm_start)
        .def_readwrite("termination_scaling_max_factor", &cddp::CDDPOptions::termination_scaling_max_factor)
        .def_readwrite("line_search", &cddp::CDDPOptions::line_search)
        .def_readwrite("regularization", &cddp::CDDPOptions::regularization)
        .def_readwrite("box_qp", &cddp::CDDPOptions::box_qp)
        .def_readwrite("filter", &cddp::CDDPOptions::filter)
        .def_readwrite("log_barrier", &cddp::CDDPOptions::log_barrier)
        .def_readwrite("ipddp", &cddp::CDDPOptions::ipddp)
        .def_readwrite("msipddp", &cddp::CDDPOptions::msipddp);
}
