#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cddp_core/objective.hpp"

namespace py = pybind11;

template <typename T>
using nodeleter = std::unique_ptr<T, py::nodelete>;

// Trampoline for NonlinearObjective
class PyNonlinearObjective : public cddp::NonlinearObjective {
public:
    using cddp::NonlinearObjective::NonlinearObjective;

    double running_cost(const Eigen::VectorXd& state,
                        const Eigen::VectorXd& control,
                        int index) const override {
        PYBIND11_OVERRIDE(double, cddp::NonlinearObjective,
                          running_cost, state, control, index);
    }

    double terminal_cost(const Eigen::VectorXd& final_state) const override {
        PYBIND11_OVERRIDE(double, cddp::NonlinearObjective,
                          terminal_cost, final_state);
    }

    double evaluate(const std::vector<Eigen::VectorXd>& states,
                    const std::vector<Eigen::VectorXd>& controls) const override {
        PYBIND11_OVERRIDE(double, cddp::NonlinearObjective,
                          evaluate, states, controls);
    }
};

void bind_objective(py::module_& m) {
    // Base class - py::nodelete so CDDP can take ownership
    py::class_<cddp::Objective, nodeleter<cddp::Objective>>(m, "Objective")
        .def("evaluate", &cddp::Objective::evaluate,
             py::arg("states"), py::arg("controls"))
        .def("running_cost", &cddp::Objective::running_cost,
             py::arg("state"), py::arg("control"), py::arg("index"))
        .def("terminal_cost", &cddp::Objective::terminal_cost,
             py::arg("final_state"))
        .def("get_reference_state", &cddp::Objective::getReferenceState)
        .def("set_reference_state", &cddp::Objective::setReferenceState,
             py::arg("reference_state"))
        .def("set_reference_states", &cddp::Objective::setReferenceStates,
             py::arg("reference_states"));

    py::class_<cddp::QuadraticObjective, cddp::Objective, nodeleter<cddp::QuadraticObjective>>(m, "QuadraticObjective")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                       const Eigen::MatrixXd&, const Eigen::VectorXd&,
                       const std::vector<Eigen::VectorXd>&, double>(),
             py::arg("Q"), py::arg("R"), py::arg("Qf"),
             py::arg("reference_state") = Eigen::VectorXd::Zero(0),
             py::arg("reference_states") = std::vector<Eigen::VectorXd>(),
             py::arg("timestep") = 0.1)
        .def_property_readonly("Q", &cddp::QuadraticObjective::getQ)
        .def_property_readonly("R", &cddp::QuadraticObjective::getR)
        .def_property_readonly("Qf", &cddp::QuadraticObjective::getQf)
        .def("set_Q", &cddp::QuadraticObjective::setQ, py::arg("Q"))
        .def("set_R", &cddp::QuadraticObjective::setR, py::arg("R"))
        .def("set_Qf", &cddp::QuadraticObjective::setQf, py::arg("Qf"));

    py::class_<cddp::NonlinearObjective, cddp::Objective, PyNonlinearObjective, nodeleter<cddp::NonlinearObjective>>(m, "NonlinearObjective")
        .def(py::init<double>(), py::arg("timestep") = 0.1);
}
