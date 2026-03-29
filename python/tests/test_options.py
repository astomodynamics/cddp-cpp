"""Test that CDDPOptions and sub-option structs work correctly."""
import pycddp


def test_cdpp_options_defaults():
    opts = pycddp.CDDPOptions()
    assert opts.tolerance == 1e-5
    assert opts.max_iterations == 1
    assert opts.verbose is True
    assert opts.use_ilqr is True


def test_options_modification():
    opts = pycddp.CDDPOptions()
    opts.max_iterations = 200
    opts.tolerance = 1e-8
    opts.verbose = False
    assert opts.max_iterations == 200
    assert opts.tolerance == 1e-8
    assert opts.verbose is False


def test_line_search_options():
    opts = pycddp.CDDPOptions()
    opts.line_search.max_iterations = 20
    opts.line_search.step_reduction_factor = 0.3
    assert opts.line_search.max_iterations == 20
    assert abs(opts.line_search.step_reduction_factor - 0.3) < 1e-10


def test_regularization_options():
    opts = pycddp.CDDPOptions()
    opts.regularization.initial_value = 1e-4
    opts.regularization.update_factor = 5.0
    assert abs(opts.regularization.initial_value - 1e-4) < 1e-15


def test_solver_type_enum():
    assert pycddp.SolverType.CLDDP is not None
    assert pycddp.SolverType.LogDDP is not None
    assert pycddp.SolverType.IPDDP is not None
    assert pycddp.SolverType.MSIPDDP is not None


def test_barrier_strategy_enum():
    assert pycddp.BarrierStrategy.ADAPTIVE is not None
    assert pycddp.BarrierStrategy.MONOTONIC is not None
    assert pycddp.BarrierStrategy.IPOPT is not None
