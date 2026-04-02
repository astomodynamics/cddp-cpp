#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_options(py::module_& m);
void bind_dynamics(py::module_& m);
void bind_objective(py::module_& m);
void bind_constraints(py::module_& m);
void bind_solver(py::module_& m);

PYBIND11_MODULE(_pycddp_core, m) {
    m.doc() = "CDDP: Constrained Differential Dynamic Programming";
    bind_options(m);
    bind_dynamics(m);
    bind_objective(m);
    bind_constraints(m);
    bind_solver(m);
}
