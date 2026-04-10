"""Regression tests for the Python portfolio demos."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

import python_portfolio_lib as portfolio


def test_portfolio_demos_reach_targets():
    pendulum = portfolio.solve_pendulum_demo()
    cartpole = portfolio.solve_cartpole_demo()
    unicycle = portfolio.solve_unicycle_demo()
    mpcc = portfolio.solve_mpcc_demo(simulation_steps=10, horizon=12)

    assert pendulum.solver_name == "CLDDP"
    assert pendulum.final_error < 1e-3
    assert pendulum.states[:, 0].max() > 3.0
    assert abs(pendulum.controls[:, 0]).max() > 1.0

    assert cartpole.solver_name == "CLDDP"
    assert cartpole.final_error < 0.05

    assert unicycle.solver_name == "IPDDP"
    assert unicycle.final_error < 0.02
    assert unicycle.solution.final_primal_infeasibility < 1e-3

    assert mpcc.solver_name.endswith("MPC")
    assert mpcc.slug == "mpcc_racing_line"
    assert mpcc.final_error < 0.5
    assert np.all(np.diff(mpcc.states[:, 3]) >= -1e-9)
    assert np.max(np.abs(mpcc.metadata["contour_errors"])) < 0.85
    assert np.mean(mpcc.metadata["solve_times_ms"]) > 0.0


def test_portfolio_animation_writes_gif(tmp_path):
    result = portfolio.solve_pendulum_demo()

    output_path = tmp_path / "pendulum.gif"
    written = portfolio.save_animation(
        result,
        output_path,
        fps=10,
        dpi=72,
        frame_step=6,
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_portfolio_animation_rejects_invalid_settings(tmp_path):
    result = portfolio.solve_pendulum_demo()

    with pytest.raises(ValueError, match="fps must be >= 1"):
        portfolio.save_animation(result, tmp_path / "bad.gif", fps=0)

    with pytest.raises(ValueError, match="frame_step must be >= 1"):
        portfolio.save_animation(result, tmp_path / "bad.gif", frame_step=0)


def test_mpcc_demo_supports_package_style_import():
    package_portfolio = importlib.import_module("examples.python_portfolio_lib")

    result = package_portfolio.solve_mpcc_demo(simulation_steps=1, horizon=12)

    assert result.slug == "mpcc_racing_line"
    assert result.solver_name == "IPDDP MPC"


def test_mpcc_demo_rejects_nonpositive_steps():
    with pytest.raises(ValueError, match="simulation_steps must be >= 1"):
        portfolio.solve_mpcc_demo(simulation_steps=0)

    package_mpcc = importlib.import_module("examples.ipddp_mpcc_rc")
    track = portfolio._load_track_csv(portfolio._portfolio_track_path())

    with pytest.raises(ValueError, match="simulation_steps must be >= 1"):
        package_mpcc.run_ipddp_mpc(track=track, simulation_steps=0)
