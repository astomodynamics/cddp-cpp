"""Smoke test: construct all dynamics models and call get_discrete_dynamics."""
import numpy as np
import pycddp


def _test_model(model, x, u):
    """Verify a dynamics model can compute discrete dynamics and Jacobians."""
    assert model.state_dim == x.shape[0]
    assert model.control_dim == u.shape[0]
    assert model.timestep > 0

    x_next = model.get_discrete_dynamics(x, u)
    assert x_next.shape == (model.state_dim,)

    A = model.get_state_jacobian(x, u)
    assert A.shape == (model.state_dim, model.state_dim)

    B = model.get_control_jacobian(x, u)
    assert B.shape == (model.state_dim, model.control_dim)


def test_pendulum():
    m = pycddp.Pendulum(0.01, length=1.0, mass=1.0, damping=0.0)
    _test_model(m, np.array([0.1, 0.0]), np.array([0.5]))


def test_unicycle():
    m = pycddp.Unicycle(0.1)
    _test_model(m, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.1]))


def test_bicycle():
    m = pycddp.Bicycle(0.1, wheelbase=2.0)
    _test_model(m, np.zeros(4), np.array([1.0, 0.1]))


def test_car():
    m = pycddp.Car(0.03, wheelbase=2.0)
    _test_model(m, np.zeros(4), np.array([1.0, 0.1]))


def test_cartpole():
    m = pycddp.CartPole(0.01)
    _test_model(m, np.array([0.0, 0.0, 0.1, 0.0]), np.array([1.0]))


def test_dubins_car():
    m = pycddp.DubinsCar(1.0, 0.1)
    _test_model(m, np.zeros(3), np.array([0.5]))


def test_acrobot():
    m = pycddp.Acrobot(0.01)
    _test_model(m, np.zeros(4), np.array([1.0]))


def test_manipulator():
    m = pycddp.Manipulator(0.01)
    _test_model(m, np.zeros(6), np.array([0.1, 0.1, 0.1]))


def test_hcw():
    m = pycddp.HCW(1.0, mean_motion=0.001, mass=1.0)
    _test_model(m, np.zeros(6), np.array([0.01, 0.01, 0.01]))


def test_spacecraft_linear_fuel():
    m = pycddp.SpacecraftLinearFuel(1.0, mean_motion=0.001, isp=300.0)
    _test_model(m, np.zeros(8), np.array([0.01, 0.01, 0.01]))


def test_dreyfus_rocket():
    m = pycddp.DreyfusRocket(0.01)
    _test_model(m, np.array([0.0, 100.0]), np.array([0.5]))


def test_usv_3dof():
    m = pycddp.Usv3Dof(0.1)
    _test_model(m, np.zeros(6), np.array([1.0, 0.0, 0.0]))


def test_lti_system():
    A = np.array([[0, 1], [-1, 0]])
    B = np.array([[0], [1]])
    m = pycddp.LTISystem(A, B, 0.01)
    _test_model(m, np.array([1.0, 0.0]), np.array([0.5]))
