#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cddp_core/cddp_core.hpp"

namespace py = pybind11;

namespace {

class PythonBackedDynamicalSystem : public cddp::DynamicalSystem {
public:
    PythonBackedDynamicalSystem(py::object owner, cddp::DynamicalSystem *wrapped)
        : cddp::DynamicalSystem(wrapped->getStateDim(), wrapped->getControlDim(),
                                wrapped->getTimestep(),
                                wrapped->getIntegrationType()),
          owner_(std::move(owner)), wrapped_(wrapped) {}

    Eigen::VectorXd getContinuousDynamics(const Eigen::VectorXd &state,
                                          const Eigen::VectorXd &control,
                                          double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getContinuousDynamics(state, control, time);
    }

    cddp::VectorXdual2nd
    getContinuousDynamicsAutodiff(const cddp::VectorXdual2nd &state,
                                  const cddp::VectorXdual2nd &control,
                                  double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getContinuousDynamicsAutodiff(state, control, time);
    }

    Eigen::VectorXd getDiscreteDynamics(const Eigen::VectorXd &state,
                                        const Eigen::VectorXd &control,
                                        double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getDiscreteDynamics(state, control, time);
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control,
                                     double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getStateJacobian(state, control, time);
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control,
                                       double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getControlJacobian(state, control, time);
    }

    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getStateHessian(state, control, time);
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control,
                      double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getControlHessian(state, control, time);
    }

    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 double time) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getCrossHessian(state, control, time);
    }

private:
    py::object owner_;
    cddp::DynamicalSystem *wrapped_;
};

class PythonBackedObjective : public cddp::Objective {
public:
    PythonBackedObjective(py::object owner, cddp::Objective *wrapped)
        : owner_(std::move(owner)), wrapped_(wrapped) {}

    double evaluate(const std::vector<Eigen::VectorXd> &states,
                    const std::vector<Eigen::VectorXd> &controls) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->evaluate(states, controls);
    }

    double running_cost(const Eigen::VectorXd &state,
                        const Eigen::VectorXd &control,
                        int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->running_cost(state, control, index);
    }

    double terminal_cost(const Eigen::VectorXd &final_state) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->terminal_cost(final_state);
    }

    Eigen::VectorXd
    getRunningCostStateGradient(const Eigen::VectorXd &state,
                                const Eigen::VectorXd &control,
                                int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getRunningCostStateGradient(state, control, index);
    }

    Eigen::VectorXd
    getRunningCostControlGradient(const Eigen::VectorXd &state,
                                  const Eigen::VectorXd &control,
                                  int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getRunningCostControlGradient(state, control, index);
    }

    Eigen::VectorXd
    getFinalCostGradient(const Eigen::VectorXd &final_state) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getFinalCostGradient(final_state);
    }

    Eigen::MatrixXd
    getRunningCostStateHessian(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &control,
                               int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getRunningCostStateHessian(state, control, index);
    }

    Eigen::MatrixXd
    getRunningCostControlHessian(const Eigen::VectorXd &state,
                                 const Eigen::VectorXd &control,
                                 int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getRunningCostControlHessian(state, control, index);
    }

    Eigen::MatrixXd
    getRunningCostCrossHessian(const Eigen::VectorXd &state,
                               const Eigen::VectorXd &control,
                               int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getRunningCostCrossHessian(state, control, index);
    }

    Eigen::MatrixXd
    getFinalCostHessian(const Eigen::VectorXd &final_state) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getFinalCostHessian(final_state);
    }

    Eigen::VectorXd getReferenceState() const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getReferenceState();
    }

    std::vector<Eigen::VectorXd> getReferenceStates() const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getReferenceStates();
    }

    void setReferenceState(const Eigen::VectorXd &reference_state) override {
        py::gil_scoped_acquire gil;
        wrapped_->setReferenceState(reference_state);
    }

    void setReferenceStates(
        const std::vector<Eigen::VectorXd> &reference_states) override {
        py::gil_scoped_acquire gil;
        wrapped_->setReferenceStates(reference_states);
    }

private:
    py::object owner_;
    cddp::Objective *wrapped_;
};

class PythonBackedConstraint : public cddp::Constraint {
public:
    PythonBackedConstraint(py::object owner, cddp::Constraint *wrapped)
        : cddp::Constraint(wrapped->getName()), owner_(std::move(owner)),
          wrapped_(wrapped) {}

    int getDualDim() const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getDualDim();
    }


    Eigen::VectorXd evaluate(const Eigen::VectorXd &state,
                             const Eigen::VectorXd &control,
                             int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->evaluate(state, control, index);
    }

    Eigen::VectorXd getLowerBound() const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getLowerBound();
    }

    Eigen::VectorXd getUpperBound() const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getUpperBound();
    }

    Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd &state,
                                     const Eigen::VectorXd &control,
                                     int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getStateJacobian(state, control, index);
    }

    Eigen::MatrixXd getControlJacobian(const Eigen::VectorXd &state,
                                       const Eigen::VectorXd &control,
                                       int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getControlJacobian(state, control, index);
    }

    double computeViolation(const Eigen::VectorXd &state,
                            const Eigen::VectorXd &control,
                            int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->computeViolation(state, control, index);
    }

    double computeViolationFromValue(const Eigen::VectorXd &g) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->computeViolationFromValue(g);
    }

    Eigen::VectorXd getCenter() const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getCenter();
    }

    std::vector<Eigen::MatrixXd> getStateHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getStateHessian(state, control, index);
    }

    std::vector<Eigen::MatrixXd>
    getControlHessian(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &control,
                      int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getControlHessian(state, control, index);
    }

    std::vector<Eigen::MatrixXd> getCrossHessian(const Eigen::VectorXd &state,
                                                 const Eigen::VectorXd &control,
                                                 int index) const override {
        py::gil_scoped_acquire gil;
        return wrapped_->getCrossHessian(state, control, index);
    }

private:
    py::object owner_;
    cddp::Constraint *wrapped_;
};

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
                 auto *wrapped = system.cast<cddp::DynamicalSystem *>();
                 self.setDynamicalSystem(
                     std::make_unique<PythonBackedDynamicalSystem>(
                         std::move(system), wrapped));
             },
             py::arg("system"),
             "Keep a Python-owned dynamics model alive inside the solver.")
        .def("set_objective",
             [](cddp::CDDP &self, py::object objective) {
                 auto *wrapped = objective.cast<cddp::Objective *>();
                 self.setObjective(std::make_unique<PythonBackedObjective>(
                     std::move(objective), wrapped));
             },
             py::arg("objective"),
             "Keep a Python-owned objective alive inside the solver.")
        .def("add_constraint",
             [](cddp::CDDP &self, const std::string &name, py::object constraint) {
                 auto *wrapped = constraint.cast<cddp::Constraint *>();
                 self.addPathConstraint(name,
                                        std::make_unique<PythonBackedConstraint>(
                                            std::move(constraint), wrapped));
             },
             py::arg("name"), py::arg("constraint"),
             "Keep a Python-owned path constraint alive inside the solver.")
        .def("add_terminal_constraint",
             [](cddp::CDDP &self, const std::string &name, py::object constraint) {
                 auto *wrapped = constraint.cast<cddp::Constraint *>();
                 self.addTerminalConstraint(
                     name, std::make_unique<PythonBackedConstraint>(
                               std::move(constraint), wrapped));
             },
             py::arg("name"), py::arg("constraint"),
             "Keep a Python-owned terminal constraint alive inside the solver.")
        .def("remove_constraint", &cddp::CDDP::removePathConstraint,
             py::arg("name"))
        .def("remove_terminal_constraint",
             &cddp::CDDP::removeTerminalConstraint, py::arg("name"))
        .def("set_initial_trajectory", &cddp::CDDP::setInitialTrajectory,
             py::arg("X"), py::arg("U"))
        .def("solve", py::overload_cast<cddp::SolverType>(&cddp::CDDP::solve),
             py::arg("solver_type") = cddp::SolverType::CLDDP)
        .def("solve_by_name",
             py::overload_cast<const std::string &>(&cddp::CDDP::solve),
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
