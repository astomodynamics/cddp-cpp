"""Test constraint construction and evaluation."""
import numpy as np
import pycddp


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
