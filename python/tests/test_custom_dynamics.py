"""Test custom dynamics defined in Python via the trampoline."""
import numpy as np
import pycddp


class DoubleIntegrator(pycddp.DynamicalSystem):
    """x = [position, velocity], u = [acceleration]."""
    def __init__(self, dt):
        super().__init__(2, 1, dt, "euler")

    def get_continuous_dynamics(self, state, control, time=0.0):
        return np.array([state[1], control[0]])

    def get_state_jacobian(self, state, control, time=0.0):
        return np.array([[0.0, 1.0], [0.0, 0.0]])

    def get_control_jacobian(self, state, control, time=0.0):
        return np.array([[0.0], [1.0]])

    def get_state_hessian(self, state, control, time=0.0):
        return [np.zeros((2, 2)), np.zeros((2, 2))]

    def get_control_hessian(self, state, control, time=0.0):
        return [np.zeros((1, 1)), np.zeros((1, 1))]

    def get_cross_hessian(self, state, control, time=0.0):
        return [np.zeros((1, 2)), np.zeros((1, 2))]


def test_custom_dynamics_continuous():
    sys = DoubleIntegrator(0.1)
    assert sys.state_dim == 2
    assert sys.control_dim == 1

    x = np.array([0.0, 1.0])
    u = np.array([0.5])
    xdot = sys.get_continuous_dynamics(x, u)
    np.testing.assert_allclose(xdot, [1.0, 0.5])


def test_custom_dynamics_discrete():
    """Test discrete dynamics via Euler integration of custom continuous dynamics."""
    dt = 0.1
    sys = DoubleIntegrator(dt)
    x = np.array([0.0, 1.0])
    u = np.array([0.5])

    expected_next = np.array([0.1, 1.05])
    x_next = sys.get_discrete_dynamics(x, u)
    np.testing.assert_allclose(x_next, expected_next, atol=1e-10)


def test_custom_dynamics_with_solver():
    dt = 0.1
    horizon = 20
    x0 = np.array([1.0, 0.0])
    xref = np.array([0.0, 0.0])

    Q = np.zeros((2, 2))
    R = 0.1 * np.eye(1)
    Qf = 10.0 * np.eye(2)

    opts = pycddp.CDDPOptions()
    opts.max_iterations = 30
    opts.verbose = False
    opts.print_solver_header = False
    opts.enable_parallel = True
    opts.num_threads = 2

    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(DoubleIntegrator(dt))
    solver.set_objective(pycddp.QuadraticObjective(Q, R, Qf, xref, [], dt))
    solver.add_constraint("ctrl", pycddp.ControlConstraint(np.array([-5.0]), np.array([5.0])))

    solution = solver.solve(pycddp.SolverType.CLDDP)

    assert solution.iterations_completed >= 0
    assert len(solution.state_trajectory) == horizon + 1
    assert len(solution.control_trajectory) == horizon
    assert solution.solve_time_ms >= 0
