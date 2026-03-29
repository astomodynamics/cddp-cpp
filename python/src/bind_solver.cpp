#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cddp_core/cddp_core.hpp"

namespace py = pybind11;

void bind_solver(py::module_& m) {
    // CDDPSolution::History
    py::class_<cddp::CDDPSolution::History>(m, "SolutionHistory")
        .def_readonly("objective", &cddp::CDDPSolution::History::objective)
        .def_readonly("merit_function", &cddp::CDDPSolution::History::merit_function)
        .def_readonly("step_length_primal", &cddp::CDDPSolution::History::step_length_primal)
        .def_readonly("step_length_dual", &cddp::CDDPSolution::History::step_length_dual)
        .def_readonly("dual_infeasibility", &cddp::CDDPSolution::History::dual_infeasibility)
        .def_readonly("primal_infeasibility", &cddp::CDDPSolution::History::primal_infeasibility)
        .def_readonly("complementary_infeasibility", &cddp::CDDPSolution::History::complementary_infeasibility)
        .def_readonly("barrier_mu", &cddp::CDDPSolution::History::barrier_mu)
        .def_readonly("regularization", &cddp::CDDPSolution::History::regularization);

    // CDDPSolution
    py::class_<cddp::CDDPSolution>(m, "CDDPSolution")
        .def_readonly("solver_name", &cddp::CDDPSolution::solver_name)
        .def_readonly("status_message", &cddp::CDDPSolution::status_message)
        .def_readonly("iterations_completed", &cddp::CDDPSolution::iterations_completed)
        .def_readonly("solve_time_ms", &cddp::CDDPSolution::solve_time_ms)
        .def_readonly("final_objective", &cddp::CDDPSolution::final_objective)
        .def_readonly("final_step_length", &cddp::CDDPSolution::final_step_length)
        .def_readonly("final_regularization", &cddp::CDDPSolution::final_regularization)
        .def_readonly("time_points", &cddp::CDDPSolution::time_points)
        .def_readonly("state_trajectory", &cddp::CDDPSolution::state_trajectory)
        .def_readonly("control_trajectory", &cddp::CDDPSolution::control_trajectory)
        .def_readonly("feedback_gains", &cddp::CDDPSolution::feedback_gains)
        .def_readonly("final_primal_infeasibility", &cddp::CDDPSolution::final_primal_infeasibility)
        .def_readonly("final_dual_infeasibility", &cddp::CDDPSolution::final_dual_infeasibility)
        .def_readonly("final_complementary_infeasibility", &cddp::CDDPSolution::final_complementary_infeasibility)
        .def_readonly("final_barrier_mu", &cddp::CDDPSolution::final_barrier_mu)
        .def_readonly("history", &cddp::CDDPSolution::history);

    // CDDP solver
    // Objects are created with py::nodelete holders, so CDDP can safely take
    // ownership via unique_ptr without double-free. Python references become
    // dangling after passing to the solver - this is documented behavior.
    py::class_<cddp::CDDP>(m, "CDDP")
        .def(py::init([](const Eigen::VectorXd& initial_state,
                         const Eigen::VectorXd& reference_state,
                         int horizon, double timestep,
                         const cddp::CDDPOptions& options) {
            return std::make_unique<cddp::CDDP>(
                initial_state, reference_state, horizon, timestep,
                nullptr, nullptr, options);
        }),
        py::arg("initial_state"), py::arg("reference_state"),
        py::arg("horizon"), py::arg("timestep"),
        py::arg("options") = cddp::CDDPOptions())

        .def("set_initial_state", &cddp::CDDP::setInitialState, py::arg("initial_state"))
        .def("set_reference_state", &cddp::CDDP::setReferenceState, py::arg("reference_state"))
        .def("set_reference_states", &cddp::CDDP::setReferenceStates, py::arg("reference_states"))
        .def("set_horizon", &cddp::CDDP::setHorizon, py::arg("horizon"))
        .def("set_timestep", &cddp::CDDP::setTimestep, py::arg("timestep"))
        .def("set_options", &cddp::CDDP::setOptions, py::arg("options"))

        // Ownership transfer: raw pointer -> unique_ptr (safe because py::nodelete)
        .def("set_dynamical_system", [](cddp::CDDP& self, cddp::DynamicalSystem* sys) {
            self.setDynamicalSystem(std::unique_ptr<cddp::DynamicalSystem>(sys));
        }, py::arg("system"))
        .def("set_objective", [](cddp::CDDP& self, cddp::Objective* obj) {
            self.setObjective(std::unique_ptr<cddp::Objective>(obj));
        }, py::arg("objective"))
        .def("add_constraint", [](cddp::CDDP& self, const std::string& name, cddp::Constraint* c) {
            self.addPathConstraint(name, std::unique_ptr<cddp::Constraint>(c));
        }, py::arg("name"), py::arg("constraint"))
        .def("add_terminal_constraint", [](cddp::CDDP& self, const std::string& name, cddp::Constraint* c) {
            self.addTerminalConstraint(name, std::unique_ptr<cddp::Constraint>(c));
        }, py::arg("name"), py::arg("constraint"))
        .def("remove_constraint", &cddp::CDDP::removePathConstraint, py::arg("name"))
        .def("remove_terminal_constraint", &cddp::CDDP::removeTerminalConstraint, py::arg("name"))

        .def("set_initial_trajectory", &cddp::CDDP::setInitialTrajectory,
             py::arg("X"), py::arg("U"))

        .def("solve", py::overload_cast<cddp::SolverType>(&cddp::CDDP::solve),
             py::arg("solver_type") = cddp::SolverType::CLDDP)
        .def("solve_by_name", py::overload_cast<const std::string&>(&cddp::CDDP::solve),
             py::arg("solver_type"))

        .def_property_readonly("initial_state", &cddp::CDDP::getInitialState)
        .def_property_readonly("reference_state", &cddp::CDDP::getReferenceState)
        .def_property_readonly("horizon", &cddp::CDDP::getHorizon)
        .def_property_readonly("timestep", &cddp::CDDP::getTimestep)
        .def_property_readonly("state_dim", &cddp::CDDP::getStateDim)
        .def_property_readonly("control_dim", &cddp::CDDP::getControlDim)
        .def_property_readonly("options", &cddp::CDDP::getOptions);
}
