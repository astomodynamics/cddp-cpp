#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cddp_core/constraint.hpp"

namespace py = pybind11;

void bind_constraints(py::module_& m) {
    py::class_<cddp::Constraint>(m, "Constraint")
        .def("evaluate", &cddp::Constraint::evaluate,
             py::arg("state"), py::arg("control"), py::arg("index") = 0)
        .def("get_lower_bound", &cddp::Constraint::getLowerBound)
        .def("get_upper_bound", &cddp::Constraint::getUpperBound)
        .def("get_state_jacobian", &cddp::Constraint::getStateJacobian,
             py::arg("state"), py::arg("control"), py::arg("index") = 0)
        .def("get_control_jacobian", &cddp::Constraint::getControlJacobian,
             py::arg("state"), py::arg("control"), py::arg("index") = 0)
        .def("compute_violation", &cddp::Constraint::computeViolation,
             py::arg("state"), py::arg("control"), py::arg("index") = 0)
        .def("get_dual_dim", &cddp::Constraint::getDualDim)
        .def_property_readonly("name", &cddp::Constraint::getName);

    py::class_<cddp::ControlConstraint, cddp::Constraint>(m, "ControlConstraint")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double>(),
             py::arg("lower_bound"), py::arg("upper_bound"),
             py::arg("scale_factor") = 1.0);

    py::class_<cddp::StateConstraint, cddp::Constraint>(m, "StateConstraint")
        .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&, double>(),
             py::arg("lower_bound"), py::arg("upper_bound"),
             py::arg("scale_factor") = 1.0);

    py::class_<cddp::LinearConstraint, cddp::Constraint>(m, "LinearConstraint")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&, double>(),
             py::arg("A"), py::arg("b"), py::arg("scale_factor") = 1.0);

    py::class_<cddp::BallConstraint, cddp::Constraint>(m, "BallConstraint")
        .def(py::init<double, const Eigen::VectorXd&, double>(),
             py::arg("radius"), py::arg("center"),
             py::arg("scale_factor") = 1.0)
        .def("get_center", &cddp::BallConstraint::getCenter);

    py::class_<cddp::PoleConstraint, cddp::Constraint>(m, "PoleConstraint")
        .def(py::init<const Eigen::VectorXd&, char, double, double, double>(),
             py::arg("center"), py::arg("direction"),
             py::arg("radius"), py::arg("length"),
             py::arg("scale_factor") = 1.0);

    py::class_<cddp::SecondOrderConeConstraint, cddp::Constraint>(m, "SecondOrderConeConstraint")
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&, double, double, const std::string&>(),
             py::arg("cone_origin"), py::arg("opening_direction"),
             py::arg("cone_angle_fov"), py::arg("epsilon") = 1e-6,
             py::arg("name") = "SecondOrderConeConstraint");

    py::class_<cddp::ThrustMagnitudeConstraint, cddp::Constraint>(m, "ThrustMagnitudeConstraint")
        .def(py::init<double, double, double>(),
             py::arg("min_thrust"), py::arg("max_thrust"),
             py::arg("epsilon") = 1e-6);

    py::class_<cddp::MaxThrustMagnitudeConstraint, cddp::Constraint>(m, "MaxThrustMagnitudeConstraint")
        .def(py::init<double, double>(),
             py::arg("max_thrust"), py::arg("epsilon") = 1e-6);
}
