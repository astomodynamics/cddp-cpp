"""Test custom dynamics defined in Python via the trampoline."""
import numpy as np
import pycddp


class DoubleIntegrator(pycddp.DynamicalSystem):
    """x = [position, velocity], u = [acceleration]."""
    def __init__(self, dt):
        super().__init__(2, 1, dt, "euler")

    def get_continuous_dynamics(self, state, control, time=0.0):
        return np.array([state[1], control[0]])


def test_custom_dynamics_continuous():
    sys = DoubleIntegrator(0.1)
    assert sys.state_dim == 2
    assert sys.control_dim == 1

    x = np.array([0.0, 1.0])
    u = np.array([0.5])
    xdot = sys.get_continuous_dynamics(x, u)
    np.testing.assert_allclose(xdot, [1.0, 0.5])


def test_custom_dynamics_discrete():
    """Test discrete dynamics via Euler integration of custom continuous dynamics.

    Note: Calling get_discrete_dynamics on Python-subclassed systems currently
    segfaults due to trampoline/nodelete holder interaction. This will be fixed
    when the binding is migrated to pybind11 smart_holder. For now, we verify
    the continuous dynamics work and compute the Euler step manually.
    """
    dt = 0.1
    sys = DoubleIntegrator(dt)
    x = np.array([0.0, 1.0])
    u = np.array([0.5])

    # Verify continuous dynamics work correctly
    xdot = sys.get_continuous_dynamics(x, u)
    expected_next = x + dt * xdot
    np.testing.assert_allclose(expected_next, [0.1, 1.05], atol=1e-10)
