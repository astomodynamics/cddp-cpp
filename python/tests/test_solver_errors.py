"""Regression tests for Python-facing solver validation."""
import pathlib
import numpy as np
import pytest
import shutil
import subprocess
import sys

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


def test_solve_by_name_accepts_builtin_solver():
    dt = 0.05
    horizon = 20
    x0 = np.array([np.pi, 0.0])
    xref = np.array([0.0, 0.0])

    opts = pycddp.CDDPOptions()
    opts.max_iterations = 20
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
    solver.add_constraint(
        "ctrl", pycddp.ControlConstraint(np.array([-50.0]), np.array([50.0]))
    )

    solution = solver.solve_by_name("CLDDP")

    assert solution.solver_name == "CLDDP"
    assert solution.status_message
    assert len(solution.state_trajectory) == horizon + 1


def test_set_initial_trajectory_requires_dynamical_system():
    solver = _make_solver()
    X = [np.zeros(2) for _ in range(solver.horizon + 1)]
    U = [np.zeros(1) for _ in range(solver.horizon)]

    with pytest.raises(ValueError, match="is a dynamical system set"):
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


def test_import_error_message_is_actionable(tmp_path):
    package_dir = tmp_path / "pycddp"
    package_dir.mkdir()

    source_dir = pathlib.Path(__file__).resolve().parents[1] / "pycddp"
    shutil.copy(source_dir / "__init__.py", package_dir / "__init__.py")
    shutil.copy(source_dir / "_version.py", package_dir / "_version.py")

    proc = subprocess.run(
        [sys.executable, "-c", "import pycddp"],
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "Failed to import the native pycddp extension" in proc.stderr
