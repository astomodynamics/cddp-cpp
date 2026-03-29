"""Integration test: solve a pendulum swing-up problem."""
import numpy as np
import pycddp


def test_pendulum_swing_up():
    dt = 0.05
    horizon = 50
    x0 = np.array([np.pi, 0.0])
    xref = np.array([0.0, 0.0])

    Q = np.zeros((2, 2))
    R = 0.1 * np.eye(1)
    Qf = 100.0 * np.eye(2)

    opts = pycddp.CDDPOptions()
    opts.max_iterations = 100
    opts.verbose = False
    opts.print_solver_header = False

    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(pycddp.Pendulum(dt, length=0.5, mass=1.0, damping=0.01))
    solver.set_objective(pycddp.QuadraticObjective(Q, R, Qf, xref, [], dt))
    solver.add_constraint("ctrl", pycddp.ControlConstraint(np.array([-50.0]), np.array([50.0])))

    solution = solver.solve(pycddp.SolverType.CLDDP)

    assert solution.iterations_completed > 0
    assert len(solution.state_trajectory) == horizon + 1
    assert len(solution.control_trajectory) == horizon
    assert solution.solve_time_ms >= 0


def test_pendulum_logddp():
    dt = 0.05
    horizon = 50
    x0 = np.array([np.pi, 0.0])
    xref = np.array([0.0, 0.0])

    Q = np.zeros((2, 2))
    R = 0.1 * np.eye(1)
    Qf = 100.0 * np.eye(2)

    opts = pycddp.CDDPOptions()
    opts.max_iterations = 100
    opts.verbose = False
    opts.print_solver_header = False

    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(pycddp.Pendulum(dt, length=0.5, mass=1.0))
    solver.set_objective(pycddp.QuadraticObjective(Q, R, Qf, xref, [], dt))
    solver.add_constraint("ctrl", pycddp.ControlConstraint(np.array([-50.0]), np.array([50.0])))

    solution = solver.solve(pycddp.SolverType.LogDDP)
    assert solution.iterations_completed > 0
    assert len(solution.state_trajectory) == horizon + 1
