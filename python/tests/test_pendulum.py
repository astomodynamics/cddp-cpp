"""Integration test: solve a pendulum swing-up problem."""
import numpy as np
import pycddp


def _assert_common_solution_fields(solution, horizon, solver_name):
    assert solution.solver_name == solver_name
    assert solution.status_message
    assert solution.iterations_completed > 0
    assert solution.solve_time_ms >= 0
    assert np.isfinite(solution.final_objective)
    assert np.isfinite(solution.final_step_length)
    assert np.isfinite(solution.final_regularization)
    assert len(solution.time_points) == horizon + 1
    assert len(solution.state_trajectory) == horizon + 1
    assert len(solution.control_trajectory) == horizon
    assert len(solution.feedback_gains) == horizon


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

    _assert_common_solution_fields(solution, horizon, "CLDDP")
    assert np.linalg.norm(solution.state_trajectory[-1] - xref) < np.linalg.norm(x0 - xref)


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
    opts.return_iteration_info = True

    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(pycddp.Pendulum(dt, length=0.5, mass=1.0))
    solver.set_objective(pycddp.QuadraticObjective(Q, R, Qf, xref, [], dt))
    solver.add_constraint("ctrl", pycddp.ControlConstraint(np.array([-50.0]), np.array([50.0])))

    solution = solver.solve(pycddp.SolverType.LogDDP)
    _assert_common_solution_fields(solution, horizon, "LogDDP")
    history = solution.history
    assert len(history.objective) >= 1
    assert len(history.objective) == len(history.merit_function)
    assert len(history.objective) == len(history.step_length_primal)
    assert len(history.objective) == len(history.step_length_dual)
    assert len(history.objective) == len(history.dual_infeasibility)
    assert len(history.objective) == len(history.primal_infeasibility)
    assert len(history.objective) == len(history.complementary_infeasibility)
    assert len(history.objective) == len(history.regularization)
    assert len(history.objective) == len(history.barrier_mu)
