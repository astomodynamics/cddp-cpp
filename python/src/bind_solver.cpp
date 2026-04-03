#include <array>
#include <sstream>
#include <string_view>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cddp_core/cddp_core.hpp"

namespace py = pybind11;

namespace {

template <typename Func>
auto callWrapped(bool requires_gil, Func &&func) -> decltype(func()) {
    if (requires_gil) {
        py::gil_scoped_acquire gil;
        return func();
    }
    return func();
}

template <std::size_t N>
bool isExactCoreType(const py::handle &object,
                     const std::array<std::string_view, N> &type_names) {
    const py::type object_type = py::type::of(object);
    const std::string module = py::str(object_type.attr("__module__"));
    if (module != "pycddp._pycddp_core") {
        return false;
    }

    const std::string name = py::str(object_type.attr("__name__"));
    for (std::string_view type_name : type_names) {
        if (name == type_name) {
            return true;
        }
    }
    return false;
}

bool nativeDynamicsCanSkipGil(const py::handle &object) {
    static constexpr std::array<std::string_view, 23> kNativeDynamicsTypes = {
        "Pendulum",         "Unicycle",        "Bicycle",
        "Car",              "CartPole",        "DubinsCar",
        "Forklift",         "Acrobot",         "Quadrotor",
        "QuadrotorRate",    "Manipulator",     "HCW",
        "SpacecraftLinearFuel",                "SpacecraftNonlinear",
        "DreyfusRocket",    "SpacecraftLanding2D",
        "SpacecraftTwobody",
        "LTISystem",        "Usv3Dof",         "EulerAttitude",
        "QuaternionAttitude",                  "MrpAttitude",
    };
    return isExactCoreType(object, kNativeDynamicsTypes);
}

bool nativeObjectiveCanSkipGil(const py::handle &object) {
    static constexpr std::array<std::string_view, 1> kNativeObjectiveTypes = {
        "QuadraticObjective",
    };
    return isExactCoreType(object, kNativeObjectiveTypes);
}

bool nativeConstraintCanSkipGil(const py::handle &object) {
    static constexpr std::array<std::string_view, 8> kNativeConstraintTypes = {
        "ControlConstraint",        "StateConstraint",
        "LinearConstraint",         "BallConstraint",
        "PoleConstraint",           "SecondOrderConeConstraint",
        "ThrustMagnitudeConstraint", "MaxThrustMagnitudeConstraint",
    };
    return isExactCoreType(object, kNativeConstraintTypes);
}

bool isExactConstraintBase(const py::handle &object) {
    static constexpr std::array<std::string_view, 1> kConstraintBaseTypes = {
        "Constraint",
    };
    return isExactCoreType(object, kConstraintBaseTypes);
}

bool isExactDynamicsBase(const py::handle &object) {
    static constexpr std::array<std::string_view, 1> kDynamicsBaseTypes = {
        "DynamicalSystem",
    };
    return isExactCoreType(object, kDynamicsBaseTypes);
}

bool isExactObjectiveBase(const py::handle &object) {
    static constexpr std::array<std::string_view, 1> kObjectiveBaseTypes = {
        "Objective",
    };
    return isExactCoreType(object, kObjectiveBaseTypes);
}

bool isKnownSolverName(const std::string &solver_name) {
    return solver_name == "CLDDP" || solver_name == "CLCDDP" ||
           solver_name == "LogDDP" || solver_name == "LOGDDP" ||
           solver_name == "IPDDP" || solver_name == "MSIPDDP" ||
           solver_name == "ALDDP" ||
           cddp::CDDP::isSolverRegistered(solver_name);
}

bool isUnknownSolverStatus(const std::string &status_message) {
    return status_message.rfind("UnknownSolver", 0) == 0;
}

void validateInitialTrajectory(cddp::CDDP &solver,
                               const std::vector<Eigen::VectorXd> &X,
                               const std::vector<Eigen::VectorXd> &U) {
    int state_dim = 0;
    int control_dim = 0;
    try {
        state_dim = solver.getStateDim();
        control_dim = solver.getControlDim();
    } catch (const std::exception &e) {
        throw py::value_error(
            std::string("set_initial_trajectory failed while querying dimensions "
                        "(is a dynamical system set?): ") + e.what());
    }

    const std::size_t expected_state_count =
        static_cast<std::size_t>(solver.getHorizon() + 1);
    const std::size_t expected_control_count =
        static_cast<std::size_t>(solver.getHorizon());

    if (X.size() != expected_state_count || U.size() != expected_control_count) {
        std::ostringstream stream;
        stream << "set_initial_trajectory expected X length "
               << expected_state_count << " and U length "
               << expected_control_count << ", got X length " << X.size()
               << " and U length " << U.size() << ".";
        throw py::value_error(stream.str());
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != state_dim) {
            std::ostringstream stream;
            stream << "set_initial_trajectory expected state vector " << i
                   << " to have dimension " << state_dim << ", got "
                   << X[i].size() << ".";
            throw py::value_error(stream.str());
        }
    }

    for (std::size_t i = 0; i < U.size(); ++i) {
        if (U[i].size() != control_dim) {
            std::ostringstream stream;
            stream << "set_initial_trajectory expected control vector " << i
                   << " to have dimension " << control_dim << ", got "
                   << U[i].size() << ".";
            throw py::value_error(stream.str());
        }
    }
}

// These wrappers let the C++ solver hold Python-defined solver objects without
// adopting the Python object's raw allocation. The py::object keeps the
// original Python instance alive, the raw pointer only forwards virtual calls,
// and each forwarded method only reacquires the GIL when the object is truly
// Python-backed. Native extension types can stay on the fast path.
class PythonBackedDynamicalSystem : public cddp::DynamicalSystem {
public:
    PythonBackedDynamicalSystem(py::object owner, cddp::DynamicalSystem *wrapped,
                                bool requires_gil)
        : cddp::DynamicalSystem(wrapped->getStateDim(), wrapped->getControlDim(),
                                wrapped->getTimestep(),
                                wrapped->getIntegrationType()),
          owner_(std::move(owner)), wrapped_(wrapped),
          requires_gil_(requires_gil) {}

    ~PythonBackedDynamicalSystem() override {
        try {
            if (Py_IsInitialized()) {
                py::gil_scoped_acquire gil;
                owner_.release().dec_ref();
            }
        } catch (...) {}
    }

    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd &state,
                                          const Eigen::VectorXd &control,
                                          double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getContinuousDynamics(state, control, time);
        });
    }

    cddp::VectorXdual2nd
    getContinuousDynamicsAutodiff(const cddp::VectorXdual2nd &state,
                                  const cddp::VectorXdual2nd &control,
                                  double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getContinuousDynamicsAutodiff(state, control, time);
        });
    }

    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &control,
                                        double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getDiscreteDynamics(state, control, time);
        });
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control,
                                     double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getStateJacobian(state, control, time);
        });
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control,
                                       double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getControlJacobian(state, control, time);
        });
    }

    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getStateHessian(state, control, time);
        });
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control,
                      double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getControlHessian(state, control, time);
        });
    }

    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 double time) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getCrossHessian(state, control, time);
        });
    }

private:
    py::object owner_;
    cddp::DynamicalSystem *wrapped_;
    bool requires_gil_;
};

class PythonBackedObjective : public cddp::Objective {
public:
    PythonBackedObjective(py::object owner, cddp::Objective *wrapped,
                          bool requires_gil)
        : owner_(std::move(owner)), wrapped_(wrapped),
          requires_gil_(requires_gil) {}

    ~PythonBackedObjective() override {
        try {
            if (Py_IsInitialized()) {
                py::gil_scoped_acquire gil;
                owner_.release().dec_ref();
            }
        } catch (...) {}
    }

    double evaluate(const std::vector<Eigen::VectorXd> &states,
                    const std::vector<Eigen::VectorXd> &controls) const override {
        return callWrapped(requires_gil_,
                           [&] { return wrapped_->evaluate(states, controls); });
    }

    double running_cost(const Eigen::VectorXd &state,
                        const Eigen::VectorXd &control,
                        int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->running_cost(state, control, index);
        });
    }

    double terminal_cost(const Eigen::VectorXd &final_state) const override {
        return callWrapped(requires_gil_,
                           [&] { return wrapped_->terminal_cost(final_state); });
    }

    Eigen::VectorXd
    getRunningCostStateGradient(const Eigen::VectorXd &state,
                                const Eigen::VectorXd &control,
                                int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getRunningCostStateGradient(state, control, index);
        });
    }

    Eigen::VectorXd
    getRunningCostControlGradient(const Eigen::VectorXd &state,
                                  const Eigen::VectorXd &control,
                                  int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getRunningCostControlGradient(state, control,
                                                           index);
        });
    }

    Eigen::VectorXd
    getFinalCostGradient(const Eigen::VectorXd &final_state) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getFinalCostGradient(final_state);
        });
    }

    Eigen::MatrixXd
    getRunningCostStateHessian(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &control,
                               int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getRunningCostStateHessian(state, control, index);
        });
    }

    Eigen::MatrixXd
    getRunningCostControlHessian(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd &control,
                                 int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getRunningCostControlHessian(state, control,
                                                          index);
        });
    }

    Eigen::MatrixXd
    getRunningCostCrossHessian(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &control,
                               int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getRunningCostCrossHessian(state, control, index);
        });
    }

    Eigen::MatrixXd
    getFinalCostHessian(const Eigen::VectorXd &final_state) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getFinalCostHessian(final_state);
        });
    }

    Eigen::VectorXd getReferenceState() const override {
        return callWrapped(requires_gil_,
                           [&] { return wrapped_->getReferenceState(); });
    }

    std::vector<Eigen::VectorXd> getReferenceStates() const override {
        return callWrapped(requires_gil_,
                           [&] { return wrapped_->getReferenceStates(); });
    }

    void setReferenceState(const Eigen::VectorXd &reference_state) override {
        callWrapped(requires_gil_,
                    [&] { wrapped_->setReferenceState(reference_state); });
    }

    void setReferenceStates(
        const std::vector<Eigen::VectorXd> &reference_states) override {
        callWrapped(requires_gil_,
                    [&] { wrapped_->setReferenceStates(reference_states); });
    }

private:
    py::object owner_;
    cddp::Objective *wrapped_;
    bool requires_gil_;
};

class PythonBackedConstraint : public cddp::Constraint {
public:
    PythonBackedConstraint(py::object owner, cddp::Constraint *wrapped,
                           bool requires_gil)
        : cddp::Constraint(wrapped->getName()), owner_(std::move(owner)),
          wrapped_(wrapped), requires_gil_(requires_gil) {}

    ~PythonBackedConstraint() override {
        try {
            if (Py_IsInitialized()) {
                py::gil_scoped_acquire gil;
                owner_.release().dec_ref();
            }
        } catch (...) {}
    }

    int getDualDim() const override {
        return callWrapped(requires_gil_, [&] { return wrapped_->getDualDim(); });
    }

    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control,
                             int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->evaluate(state, control, index);
        });
    }

    Eigen::VectorXd getLowerBound() const override {
        return callWrapped(requires_gil_,
                           [&] { return wrapped_->getLowerBound(); });
    }

    Eigen::VectorXd getUpperBound() const override {
        return callWrapped(requires_gil_,
                           [&] { return wrapped_->getUpperBound(); });
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control,
                                     int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getStateJacobian(state, control, index);
        });
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control,
                                       int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getControlJacobian(state, control, index);
        });
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control,
                            int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->computeViolation(state, control, index);
        });
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->computeViolationFromValue(g);
        });
    }

    Eigen::VectorXd getCenter() const override {
        return callWrapped(requires_gil_, [&] { return wrapped_->getCenter(); });
    }

    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getStateHessian(state, control, index);
        });
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control,
                      int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getControlHessian(state, control, index);
        });
    }

    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 int index) const override {
        return callWrapped(requires_gil_, [&] {
            return wrapped_->getCrossHessian(state, control, index);
        });
    }

private:
    py::object owner_;
    cddp::Constraint *wrapped_;
    bool requires_gil_;
};

std::unique_ptr<cddp::DynamicalSystem> makeOwnedDynamicalSystem(
    py::object system) {
    if (isExactDynamicsBase(system)) {
        throw py::type_error(
            "pycddp.DynamicalSystem is an abstract base class. "
            "Pass a concrete built-in model or a Python subclass that "
            "implements the required methods.");
    }
    auto *wrapped = system.cast<cddp::DynamicalSystem *>();
    const bool requires_gil = !nativeDynamicsCanSkipGil(system);
    return std::make_unique<PythonBackedDynamicalSystem>(
        std::move(system), wrapped, requires_gil);
}

std::unique_ptr<cddp::Objective> makeOwnedObjective(py::object objective) {
    if (isExactObjectiveBase(objective)) {
        throw py::type_error(
            "pycddp.Objective is an abstract base class. "
            "Pass a concrete built-in objective or a Python subclass that "
            "implements the required methods.");
    }
    auto *wrapped = objective.cast<cddp::Objective *>();
    const bool requires_gil = !nativeObjectiveCanSkipGil(objective);
    return std::make_unique<PythonBackedObjective>(
        std::move(objective), wrapped, requires_gil);
}

std::unique_ptr<cddp::Constraint> makeOwnedConstraint(py::object constraint) {
    if (isExactConstraintBase(constraint)) {
        throw py::type_error(
            "pycddp.Constraint is an abstract base class. "
            "Pass a concrete built-in constraint or a Python subclass that "
            "implements the required methods.");
    }
    auto *wrapped = constraint.cast<cddp::Constraint *>();
    const bool requires_gil = !nativeConstraintCanSkipGil(constraint);
    return std::make_unique<PythonBackedConstraint>(
        std::move(constraint), wrapped, requires_gil);
}

} // namespace

void bind_solver(py::module_ &m) {
    py::class_<cddp::CDDPSolution::History>(m, "SolutionHistory")
        .def_readonly("objective", &cddp::CDDPSolution::History::objective)
        .def_readonly("merit_function",
                      &cddp::CDDPSolution::History::merit_function)
        .def_readonly("step_length_primal",
                      &cddp::CDDPSolution::History::step_length_primal)
        .def_readonly("step_length_dual",
                      &cddp::CDDPSolution::History::step_length_dual)
        .def_readonly("dual_infeasibility",
                      &cddp::CDDPSolution::History::dual_infeasibility)
        .def_readonly("primal_infeasibility",
                      &cddp::CDDPSolution::History::primal_infeasibility)
        .def_readonly("complementary_infeasibility",
                      &cddp::CDDPSolution::History::complementary_infeasibility)
        .def_readonly("barrier_mu", &cddp::CDDPSolution::History::barrier_mu)
        .def_readonly("regularization",
                      &cddp::CDDPSolution::History::regularization);

    py::class_<cddp::CDDPSolution>(m, "CDDPSolution")
        .def_readonly("solver_name", &cddp::CDDPSolution::solver_name)
        .def_readonly("status_message", &cddp::CDDPSolution::status_message)
        .def_readonly("iterations_completed",
                      &cddp::CDDPSolution::iterations_completed)
        .def_readonly("solve_time_ms", &cddp::CDDPSolution::solve_time_ms)
        .def_readonly("final_objective", &cddp::CDDPSolution::final_objective)
        .def_readonly("final_step_length",
                      &cddp::CDDPSolution::final_step_length)
        .def_readonly("final_regularization",
                      &cddp::CDDPSolution::final_regularization)
        .def_readonly("time_points", &cddp::CDDPSolution::time_points)
        .def_readonly("state_trajectory",
                      &cddp::CDDPSolution::state_trajectory)
        .def_readonly("control_trajectory",
                      &cddp::CDDPSolution::control_trajectory)
        .def_readonly("feedback_gains", &cddp::CDDPSolution::feedback_gains)
        .def_readonly("final_primal_infeasibility",
                      &cddp::CDDPSolution::final_primal_infeasibility)
        .def_readonly("final_dual_infeasibility",
                      &cddp::CDDPSolution::final_dual_infeasibility)
        .def_readonly("final_complementary_infeasibility",
                      &cddp::CDDPSolution::final_complementary_infeasibility)
        .def_readonly("final_barrier_mu",
                      &cddp::CDDPSolution::final_barrier_mu)
        .def_readonly("history", &cddp::CDDPSolution::history);

    py::class_<cddp::CDDP>(m, "CDDP")
        .def(py::init([](const Eigen::VectorXd &initial_state,
                         const Eigen::VectorXd &reference_state, int horizon,
                         double timestep, const cddp::CDDPOptions &options) {
                 return std::make_unique<cddp::CDDP>(initial_state,
                                                     reference_state, horizon,
                                                     timestep, nullptr, nullptr,
                                                     options);
             }),
             py::arg("initial_state"), py::arg("reference_state"),
             py::arg("horizon"), py::arg("timestep"),
             py::arg("options") = cddp::CDDPOptions())

        .def("set_initial_state", &cddp::CDDP::setInitialState,
             py::arg("initial_state"))
        .def("set_reference_state", &cddp::CDDP::setReferenceState,
             py::arg("reference_state"))
        .def("set_reference_states", &cddp::CDDP::setReferenceStates,
             py::arg("reference_states"))
        .def("set_horizon", &cddp::CDDP::setHorizon, py::arg("horizon"))
        .def("set_timestep", &cddp::CDDP::setTimestep, py::arg("timestep"))
        .def("set_options", &cddp::CDDP::setOptions, py::arg("options"))
        .def("set_dynamical_system",
             [](cddp::CDDP &self, py::object system) {
                 self.setDynamicalSystem(makeOwnedDynamicalSystem(
                     std::move(system)));
             },
             py::arg("system"),
             "The solver keeps a Python reference that keeps this dynamics "
             "object alive. Mutating or sharing the object after this call "
             "may produce unexpected behavior.")
        .def("set_objective",
             [](cddp::CDDP &self, py::object objective) {
                 self.setObjective(makeOwnedObjective(std::move(objective)));
             },
             py::arg("objective"),
             "The solver keeps a Python reference that keeps this objective "
             "alive. Mutating or sharing the object after this call may "
             "produce unexpected behavior.")
        .def("add_constraint",
             [](cddp::CDDP &self, const std::string &name, py::object constraint) {
                 self.addPathConstraint(name,
                                        makeOwnedConstraint(
                                            std::move(constraint)));
             },
             py::arg("name"), py::arg("constraint"),
             "The solver keeps a Python reference that keeps this path "
             "constraint alive. Mutating or sharing the object after this "
             "call may produce unexpected behavior.")
        .def("add_terminal_constraint",
             [](cddp::CDDP &self, const std::string &name, py::object constraint) {
                 self.addTerminalConstraint(
                     name, makeOwnedConstraint(std::move(constraint)));
             },
             py::arg("name"), py::arg("constraint"),
             "The solver keeps a Python reference that keeps this terminal "
             "constraint alive. Mutating or sharing the object after this "
             "call may produce unexpected behavior.")
        .def("remove_constraint", &cddp::CDDP::removePathConstraint,
             py::arg("name"))
        .def("remove_terminal_constraint",
             &cddp::CDDP::removeTerminalConstraint, py::arg("name"))
        .def("set_initial_trajectory",
             [](cddp::CDDP &self, const std::vector<Eigen::VectorXd> &X,
                const std::vector<Eigen::VectorXd> &U) {
                 validateInitialTrajectory(self, X, U);
                 self.setInitialTrajectory(X, U);
             },
             py::arg("X"), py::arg("U"))
        .def("solve", py::overload_cast<cddp::SolverType>(&cddp::CDDP::solve),
             py::arg("solver_type") = cddp::SolverType::CLDDP,
             py::call_guard<py::gil_scoped_release>())
        .def("solve_by_name",
             [](cddp::CDDP &self, const std::string &solver_type) {
                 if (!isKnownSolverName(solver_type)) {
                     throw py::value_error("Unknown solver '" + solver_type +
                                           "'.");
                 }
                 cddp::CDDPSolution solution;
                 {
                     py::gil_scoped_release release;
                     solution = self.solve(solver_type);
                 }
                 if (isUnknownSolverStatus(solution.status_message)) {
                     throw py::value_error("Unknown solver '" + solver_type +
                                           "'.");
                 }
                 return solution;
             },
             py::arg("solver_type"))
        .def_property_readonly("initial_state", &cddp::CDDP::getInitialState)
        .def_property_readonly("reference_state",
                               &cddp::CDDP::getReferenceState)
        .def_property_readonly("horizon", &cddp::CDDP::getHorizon)
        .def_property_readonly("timestep", &cddp::CDDP::getTimestep)
        .def_property_readonly("state_dim", &cddp::CDDP::getStateDim)
        .def_property_readonly("control_dim", &cddp::CDDP::getControlDim)
        .def_property_readonly("options", &cddp::CDDP::getOptions);
}
