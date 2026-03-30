"""Regression tests for Python-defined nonlinear objectives."""
import numpy as np

import pycddp


class CountingNonlinearObjective(pycddp.NonlinearObjective):
    def __init__(self, timestep, counters):
        super().__init__(timestep)
        self._counters = counters

    def evaluate(self, states, controls):
        self._counters["evaluate"] += 1
        total = 0.0
        for state, control in zip(states[:-1], controls):
            total += self.running_cost(state, control, 0)
        total += self.terminal_cost(states[-1])
        return total

    def running_cost(self, state, control, index):
        self._counters["running_cost"] += 1
        return float(state @ state + 0.1 * control @ control)

    def terminal_cost(self, final_state):
        self._counters["terminal_cost"] += 1
        return float(10.0 * final_state @ final_state)


def test_python_nonlinear_objective_dispatches_through_solver():
    dt = 0.1
    horizon = 15
    counters = {"evaluate": 0, "running_cost": 0, "terminal_cost": 0}

    opts = pycddp.CDDPOptions()
    opts.max_iterations = 20
    opts.verbose = False
    opts.print_solver_header = False

    solver = pycddp.CDDP(np.array([1.0, 0.0]), np.zeros(2), horizon, dt, opts)
    solver.set_dynamical_system(
        pycddp.LTISystem(
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0], [1.0]]),
            dt,
        )
    )
    solver.set_objective(CountingNonlinearObjective(dt, counters))
    solver.add_constraint(
        "ctrl", pycddp.ControlConstraint(np.array([-2.0]), np.array([2.0]))
    )

    solution = solver.solve(pycddp.SolverType.LogDDP)

    assert solution.solver_name == "LogDDP"
    assert solution.status_message
    assert counters["evaluate"] > 0
    assert counters["running_cost"] > 0
    assert counters["terminal_cost"] > 0
