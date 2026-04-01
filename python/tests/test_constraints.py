"""Test constraint construction and evaluation."""
import numpy as np
import pycddp
import pytest


class CountingAffineConstraint(pycddp.Constraint):
    def __init__(self, counters):
        super().__init__("CountingAffineConstraint")
        self._counters = counters

    def get_dual_dim(self):
        return 1

    def evaluate(self, state, control, index=0):
        self._counters["evaluate"] += 1
        return np.array([state[0] - 10.0])

    def get_lower_bound(self):
        return np.array([-np.inf])

    def get_upper_bound(self):
        return np.array([0.0])

    def get_state_jacobian(self, state, control, index=0):
        self._counters["state_jacobian"] += 1
        return np.array([[1.0, 0.0]])

    def get_control_jacobian(self, state, control, index=0):
        self._counters["control_jacobian"] += 1
        return np.array([[0.0]])

    def compute_violation(self, state, control, index=0):
        self._counters["compute_violation"] += 1
        return max(0.0, float(self.evaluate(state, control, index)[0]))

    def compute_violation_from_value(self, g):
        self._counters["compute_violation_from_value"] += 1
        return max(0.0, float(g[0]))


def test_control_constraint():
    c = pycddp.ControlConstraint(np.array([-1.0, -2.0]), np.array([1.0, 2.0]))
    assert c.get_dual_dim() == 4  # 2 lower + 2 upper bounds
    assert c.name == "ControlConstraint"


def test_state_constraint():
    c = pycddp.StateConstraint(np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    assert c.get_dual_dim() == 4


def test_ball_constraint():
    center = np.array([1.0, 1.0])
    c = pycddp.BallConstraint(radius=0.5, center=center)
    assert c.get_dual_dim() == 1
    np.testing.assert_array_equal(c.get_center(), center)

    # Point far from obstacle should satisfy constraint
    state_far = np.array([5.0, 5.0])
    control = np.array([0.0])
    val = c.evaluate(state_far, control)
    assert val.shape[0] == 1


def test_linear_constraint():
    A = np.array([[1.0, 1.0], [-1.0, 1.0]])
    b = np.array([1.0, 1.0])
    c = pycddp.LinearConstraint(A, b)
    assert c.get_dual_dim() == 2

    state = np.array([0.0, 0.0])
    control = np.array([0.0])
    val = c.evaluate(state, control)
    assert val.shape[0] == 2


def test_custom_python_constraint_with_solver():
    counters = {
        "evaluate": 0,
        "state_jacobian": 0,
        "control_jacobian": 0,
        "compute_violation": 0,
        "compute_violation_from_value": 0,
    }

    dt = 0.05
    horizon = 20
    x0 = np.array([np.pi, 0.0])
    xref = np.array([0.0, 0.0])

    opts = pycddp.CDDPOptions()
    opts.max_iterations = 10
    opts.verbose = False
    opts.print_solver_header = False

    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(
        pycddp.Pendulum(dt, length=0.5, mass=1.0, damping=0.01)
    )
    solver.set_objective(
        pycddp.QuadraticObjective(
            np.zeros((2, 2)), 0.1 * np.eye(1), 100.0 * np.eye(2), xref, [], dt
        )
    )
    solver.add_constraint("custom", CountingAffineConstraint(counters))

    solution = solver.solve(pycddp.SolverType.LogDDP)

    assert solution.solver_name == "LogDDP"
    assert solution.status_message
    assert counters["evaluate"] > 0
    assert counters["state_jacobian"] > 0
    assert counters["control_jacobian"] > 0
def test_constraint_base_is_rejected_cleanly():
    dt = 0.1
    opts = pycddp.CDDPOptions()
    opts.max_iterations = 2
    opts.verbose = False
    opts.print_solver_header = False
    opts.enable_parallel = True
    opts.num_threads = 2

    solver = pycddp.CDDP(np.array([1.0, 0.0]), np.zeros(2), 8, dt, opts)
    solver.set_dynamical_system(
        pycddp.LTISystem(
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0], [1.0]]),
            dt,
        )
    )
    solver.set_objective(
        pycddp.QuadraticObjective(
            np.eye(2), 0.1 * np.eye(1), 10.0 * np.eye(2), np.zeros(2), [], dt
        )
    )

    with pytest.raises(TypeError, match="Constraint is an abstract base class"):
        solver.add_constraint("bad", pycddp.Constraint("bad"))
