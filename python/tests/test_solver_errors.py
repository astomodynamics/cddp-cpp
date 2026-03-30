"""Regression tests for Python-facing solver validation."""
import numpy as np
import pytest

import pycddp


def _make_solver(horizon=6, dt=0.1):
    x0 = np.array([0.0, 0.0])
    xref = np.array([0.0, 0.0])
    opts = pycddp.CDDPOptions()
    opts.verbose = False
    opts.print_solver_header = False
    return pycddp.CDDP(x0, xref, horizon, dt, opts)


def test_solve_by_name_raises_for_unknown_solver():
    solver = _make_solver()

    with pytest.raises(ValueError, match="Unknown solver 'NONEXISTENT'"):
        solver.solve_by_name("NONEXISTENT")


def test_set_initial_trajectory_requires_dynamical_system():
    solver = _make_solver()
    X = [np.zeros(2) for _ in range(solver.horizon + 1)]
    U = [np.zeros(1) for _ in range(solver.horizon)]

    with pytest.raises(ValueError, match="requires a dynamical system"):
        solver.set_initial_trajectory(X, U)


def test_set_initial_trajectory_rejects_bad_lengths():
    solver = _make_solver()
    solver.set_dynamical_system(pycddp.Pendulum(0.1))

    X = [np.zeros(2) for _ in range(solver.horizon)]
    U = [np.zeros(1) for _ in range(solver.horizon)]

    with pytest.raises(ValueError, match="expected X length"):
        solver.set_initial_trajectory(X, U)


def test_set_initial_trajectory_rejects_bad_state_dimension():
    solver = _make_solver()
    solver.set_dynamical_system(pycddp.Pendulum(0.1))

    X = [np.zeros(2) for _ in range(solver.horizon + 1)]
    X[2] = np.zeros(3)
    U = [np.zeros(1) for _ in range(solver.horizon)]

    with pytest.raises(ValueError, match="state vector 2"):
        solver.set_initial_trajectory(X, U)


def test_set_initial_trajectory_rejects_bad_control_dimension():
    solver = _make_solver()
    solver.set_dynamical_system(pycddp.Pendulum(0.1))

    X = [np.zeros(2) for _ in range(solver.horizon + 1)]
    U = [np.zeros(1) for _ in range(solver.horizon)]
    U[1] = np.zeros(2)

    with pytest.raises(ValueError, match="control vector 1"):
        solver.set_initial_trajectory(X, U)
