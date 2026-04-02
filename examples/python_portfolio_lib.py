"""Portfolio-quality Python demos and animations for pycddp."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.transforms import Affine2D

import pycddp


_BACKGROUND = "#f5efe3"
_PANEL = "#fffaf2"
_TEXT = "#1d2533"
_GRID = "#d8cdb8"
_ACCENT = "#1f6f8b"
_SECONDARY = "#d97706"
_TERTIARY = "#9c2f2f"
_SUCCESS = "#1f7a5c"
_MUTED = "#8d99a6"


@dataclass(slots=True)
class DemoResult:
    slug: str
    title: str
    solver_name: str
    solution: pycddp.CDDPSolution
    states: np.ndarray
    controls: np.ndarray
    time_points: np.ndarray
    target_state: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_error(self) -> float:
        return float(np.linalg.norm(self.states[-1] - self.target_state))


def _solution_arrays(solution: pycddp.CDDPSolution) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.vstack([np.asarray(x, dtype=float) for x in solution.state_trajectory])
    if solution.control_trajectory:
        controls = np.vstack([np.asarray(u, dtype=float) for u in solution.control_trajectory])
    else:
        controls = np.zeros((0, 0), dtype=float)
    time_points = np.asarray(solution.time_points, dtype=float)
    return states, controls, time_points


def _rollout_system(
    model: pycddp.DynamicalSystem,
    initial_state: np.ndarray,
    controls: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    state = np.asarray(initial_state, dtype=float).copy()
    states = [state.copy()]
    rolled_controls: list[np.ndarray] = []
    for control in controls:
        control_array = np.asarray(control, dtype=float).copy()
        rolled_controls.append(control_array)
        state = np.asarray(model.get_discrete_dynamics(state, control_array), dtype=float)
        states.append(state.copy())
    return states, rolled_controls


def _default_options(max_iterations: int = 120) -> pycddp.CDDPOptions:
    opts = pycddp.CDDPOptions()
    opts.max_iterations = max_iterations
    opts.verbose = False
    opts.print_solver_header = False
    opts.return_iteration_info = True
    return opts


def _make_result(
    slug: str,
    title: str,
    target_state: np.ndarray,
    solution: pycddp.CDDPSolution,
    **metadata: Any,
) -> DemoResult:
    states, controls, time_points = _solution_arrays(solution)
    return DemoResult(
        slug=slug,
        title=title,
        solver_name=solution.solver_name,
        solution=solution,
        states=states,
        controls=controls,
        time_points=time_points,
        target_state=np.asarray(target_state, dtype=float),
        metadata=metadata,
    )


def solve_pendulum_demo() -> DemoResult:
    dt = 0.05
    horizon = 120
    x0 = np.array([0.0, 0.0], dtype=float)
    xref = np.array([np.pi, 0.0], dtype=float)

    opts = _default_options(max_iterations=150)
    opts.tolerance = 1e-5
    opts.acceptable_tolerance = 1e-4
    opts.regularization.initial_value = 1e-6

    model = pycddp.Pendulum(dt, length=0.5, mass=1.0, damping=0.01)
    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(model)
    solver.set_objective(
        pycddp.QuadraticObjective(
            0.1 * np.eye(2),
            0.02 * np.eye(1),
            200.0 * np.eye(2),
            xref,
            [],
            dt,
        )
    )
    solver.add_constraint(
        "control_limits",
        pycddp.ControlConstraint(np.array([-30.0]), np.array([30.0])),
    )
    seed_controls = [
        np.array([8.0], dtype=float) if index < 25 else np.zeros(1, dtype=float)
        for index in range(horizon)
    ]
    seed_states, seed_controls = _rollout_system(model, x0, seed_controls)
    solver.set_initial_trajectory(seed_states, seed_controls)

    solution = solver.solve(pycddp.SolverType.CLDDP)
    return _make_result(
        slug="pendulum_swing_up",
        title="Pendulum Swing-Up",
        target_state=xref,
        solution=solution,
        timestep=dt,
        length=0.5,
        subtitle="Seeded CLDDP swing-up from the hanging equilibrium",
    )


def solve_cartpole_demo() -> DemoResult:
    dt = 0.05
    horizon = 100
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    xref = np.array([0.0, np.pi, 0.0, 0.0], dtype=float)

    opts = _default_options(max_iterations=120)
    opts.tolerance = 1e-6
    opts.acceptable_tolerance = 1e-5
    opts.regularization.initial_value = 1e-5

    solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
    solver.set_dynamical_system(pycddp.CartPole(dt))
    solver.set_objective(
        pycddp.QuadraticObjective(
            np.zeros((4, 4)),
            0.1 * np.eye(1),
            80.0 * np.eye(4),
            xref,
            [],
            dt,
        )
    )
    solver.add_constraint(
        "force_limits",
        pycddp.ControlConstraint(np.array([-5.0]), np.array([5.0])),
    )
    solver.set_initial_trajectory(
        [x0.copy() for _ in range(horizon + 1)],
        [np.zeros(1, dtype=float) for _ in range(horizon)],
    )

    solution = solver.solve(pycddp.SolverType.CLDDP)
    return _make_result(
        slug="cartpole_swing_up",
        title="Cart-Pole Swing-Up",
        target_state=xref,
        solution=solution,
        timestep=dt,
        pole_length=0.5,
        subtitle="Control-limited CLDDP solve",
    )


def solve_unicycle_demo() -> DemoResult:
    dt = 0.03
    horizon = 100
    x0 = np.array([0.0, 0.0, np.pi / 4.0], dtype=float)
    xref = np.array([2.0, 2.0, np.pi / 2.0], dtype=float)
    obstacle_center = np.array([1.0, 1.0], dtype=float)
    obstacle_radius = 0.4

    opts = _default_options(max_iterations=100)
    opts.tolerance = 1e-4

    solver_objective = pycddp.QuadraticObjective(
        np.zeros((3, 3)),
        0.05 * np.eye(2),
        np.diag([100.0, 100.0, 50.0]),
        xref,
        [],
        dt,
    )
    control_limits = pycddp.ControlConstraint(
        np.array([-1.1, -np.pi]),
        np.array([1.1, np.pi]),
    )
    initial_states = [x0.copy() for _ in range(horizon + 1)]
    initial_controls = [np.zeros(2, dtype=float) for _ in range(horizon)]

    # Seed the constrained solve with the smoother unconstrained route so the
    # obstacle-avoidance case is stable across repeated runs.
    baseline = pycddp.CDDP(x0, xref, horizon, dt, opts)
    baseline.set_dynamical_system(pycddp.Unicycle(dt))
    baseline.set_objective(solver_objective)
    baseline.add_constraint("control_limits", control_limits)
    baseline.set_initial_trajectory(initial_states, initial_controls)
    baseline_solution = baseline.solve(pycddp.SolverType.CLDDP)

    best_solution: pycddp.CDDPSolution | None = None
    best_score: tuple[float, float, float] | None = None

    for _ in range(4):
        solver = pycddp.CDDP(x0, xref, horizon, dt, opts)
        solver.set_dynamical_system(pycddp.Unicycle(dt))
        solver.set_objective(
            pycddp.QuadraticObjective(
                np.zeros((3, 3)),
                0.05 * np.eye(2),
                np.diag([100.0, 100.0, 50.0]),
                xref,
                [],
                dt,
            )
        )
        solver.add_constraint(
            "control_limits",
            pycddp.ControlConstraint(
                np.array([-1.1, -np.pi]),
                np.array([1.1, np.pi]),
            ),
        )
        solver.add_constraint(
            "obstacle",
            pycddp.BallConstraint(obstacle_radius, obstacle_center),
        )
        solver.set_initial_trajectory(
            list(baseline_solution.state_trajectory),
            list(baseline_solution.control_trajectory),
        )

        candidate = solver.solve(pycddp.SolverType.IPDDP)
        candidate_error = float(
            np.linalg.norm(np.asarray(candidate.state_trajectory[-1]) - xref)
        )
        candidate_score = (
            float(candidate.final_primal_infeasibility),
            candidate_error,
            float(candidate.final_objective),
        )
        if best_score is None or candidate_score < best_score:
            best_score = candidate_score
            best_solution = candidate
        if candidate.final_primal_infeasibility < 1e-3 and candidate_error < 0.02:
            break

    assert best_solution is not None
    return _make_result(
        slug="unicycle_obstacle_avoidance",
        title="Unicycle Obstacle Avoidance",
        target_state=xref,
        solution=best_solution,
        timestep=dt,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        subtitle="IPDDP with a ball constraint",
    )


DEMO_BUILDERS: dict[str, Callable[[], DemoResult]] = {
    "pendulum": solve_pendulum_demo,
    "cartpole": solve_cartpole_demo,
    "unicycle": solve_unicycle_demo,
}


def build_demo(name: str) -> DemoResult:
    try:
        return DEMO_BUILDERS[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(DEMO_BUILDERS))
        raise ValueError(f"Unknown demo '{name}'. Expected one of: {choices}.") from exc


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.grid(True, color=_GRID, alpha=0.55, linewidth=0.7)


def _style_scene_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _metric_bounds(values: np.ndarray, minimum: float = 1.0) -> tuple[float, float]:
    amplitude = max(float(np.max(np.abs(values))) * 1.15, minimum)
    return -amplitude, amplitude


def _apply_title(fig: plt.Figure, result: DemoResult) -> None:
    fig.suptitle(result.title, fontsize=18, fontweight="bold", color=_TEXT, y=0.965)
    fig.subplots_adjust(top=0.81)


def save_animation(
    result: DemoResult,
    output_path: str | Path,
    fps: int = 16,
    dpi: int = 110,
    frame_step: int = 2,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if result.slug == "pendulum_swing_up":
        fig, anim = _animate_pendulum(result, fps=fps, frame_step=frame_step)
    elif result.slug == "cartpole_swing_up":
        fig, anim = _animate_cartpole(result, fps=fps, frame_step=frame_step)
    elif result.slug == "unicycle_obstacle_avoidance":
        fig, anim = _animate_unicycle(result, fps=fps, frame_step=frame_step)
    else:
        raise ValueError(f"No animation renderer for demo '{result.slug}'.")

    writer = animation.PillowWriter(fps=fps)
    anim.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    return output


def _animate_pendulum(
    result: DemoResult,
    fps: int,
    frame_step: int,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    states = result.states
    controls = result.controls[:, 0]
    time_points = result.time_points
    pendulum_length = float(result.metadata["length"])
    frames = list(range(0, len(states), frame_step))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)

    fig = plt.figure(figsize=(9.2, 5.2), facecolor=_BACKGROUND)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.16,
        hspace=0.20,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[1, 1])
    _apply_title(fig, result)
    _style_scene_axes(ax_main)
    _style_axes(ax_phase)
    _style_axes(ax_ctrl)

    limit = pendulum_length * 1.45
    ax_main.set_xlim(-limit, limit)
    ax_main.set_ylim(-limit, limit)
    ax_main.set_aspect("equal")
    ax_main.set_title("Swing-Up Motion", color=_TEXT, fontsize=11, pad=10)
    orbit = Circle((0.0, 0.0), pendulum_length, facecolor="none", edgecolor=_GRID, linewidth=1.0, linestyle="--")
    ax_main.add_patch(orbit)
    ax_main.scatter([0.0], [0.0], s=35, color=_TEXT, zorder=4)
    target_x = pendulum_length * np.sin(result.target_state[0])
    target_y = -pendulum_length * np.cos(result.target_state[0])
    ghost_rod, = ax_main.plot([0.0, target_x], [0.0, target_y], color=_MUTED, linewidth=2.0, linestyle="--")
    ghost_bob = Circle((target_x, target_y), 0.06, facecolor="none", edgecolor=_MUTED, linewidth=1.6, linestyle="--")
    ax_main.add_patch(ghost_bob)

    ax_phase.set_title("Phase Portrait", color=_TEXT, fontsize=11, pad=10)
    ax_phase.set_xlabel("Angle [rad]", color=_TEXT)
    ax_phase.set_ylabel("Angular rate [rad/s]", color=_TEXT)
    theta_min = float(np.min(states[:, 0]))
    theta_max = float(np.max(states[:, 0]))
    omega_min = float(np.min(states[:, 1]))
    omega_max = float(np.max(states[:, 1]))
    theta_pad = max(0.2, 0.08 * max(abs(theta_min), abs(theta_max), 1.0))
    omega_pad = max(0.2, 0.12 * max(abs(omega_min), abs(omega_max), 1.0))
    ax_phase.set_xlim(theta_min - theta_pad, theta_max + theta_pad)
    ax_phase.set_ylim(omega_min - omega_pad, omega_max + omega_pad)
    ax_phase.axvline(result.target_state[0], color=_MUTED, linewidth=1.0, linestyle="--")
    ax_phase.axhline(0.0, color=_GRID, linewidth=1.0)

    ax_ctrl.set_xlim(time_points[0], time_points[-2] if len(time_points) > 1 else 1.0)
    ctrl_min, ctrl_max = _metric_bounds(controls)
    ax_ctrl.set_ylim(ctrl_min, ctrl_max)
    ax_ctrl.set_title("Control Torque", color=_TEXT, fontsize=11, pad=10)
    ax_ctrl.set_xlabel("Time [s]", color=_TEXT)
    ax_ctrl.set_ylabel("Torque", color=_TEXT)

    bob_trail, = ax_main.plot([], [], color=_SECONDARY, linewidth=2.2, alpha=0.7)
    rod, = ax_main.plot([], [], color=_ACCENT, linewidth=3.2)
    bob = Circle((0.0, 0.0), 0.07, color=_TERTIARY)
    ax_main.add_patch(bob)
    torque_text = ax_main.text(
        0.03,
        0.95,
        "",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=_TEXT,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ea", "edgecolor": _GRID},
    )

    phase_path, = ax_phase.plot(states[:, 0], states[:, 1], color=_GRID, linewidth=1.8)
    phase_trace, = ax_phase.plot([], [], color=_ACCENT, linewidth=2.2)
    phase_marker, = ax_phase.plot([], [], "o", color=_TERTIARY, markersize=7)

    control_history, = ax_ctrl.plot(time_points[:-1], controls, color=_ACCENT, linewidth=2.0)
    control_marker, = ax_ctrl.plot([], [], "o", color=_SECONDARY, markersize=7)

    def update(frame_index: int) -> tuple[Any, ...]:
        theta = states[frame_index, 0]
        x = pendulum_length * np.sin(theta)
        y = -pendulum_length * np.cos(theta)
        trail_start = max(0, frame_index - 30)
        trail_x = pendulum_length * np.sin(states[trail_start : frame_index + 1, 0])
        trail_y = -pendulum_length * np.cos(states[trail_start : frame_index + 1, 0])
        bob_trail.set_data(trail_x, trail_y)
        rod.set_data([0.0, x], [0.0, y])
        bob.center = (x, y)
        phase_trace.set_data(states[: frame_index + 1, 0], states[: frame_index + 1, 1])
        phase_marker.set_data([states[frame_index, 0]], [states[frame_index, 1]])
        control_idx = min(frame_index, len(controls) - 1)
        control_marker.set_data([time_points[control_idx]], [controls[control_idx]])
        torque_text.set_text(
            f"angle = {theta:+.2f} rad\nrate = {states[frame_index, 1]:+.2f} rad/s\ntorque = {controls[control_idx]:+.2f}"
        )
        return (
            orbit,
            ghost_rod,
            ghost_bob,
            bob_trail,
            rod,
            bob,
            phase_path,
            phase_trace,
            phase_marker,
            control_history,
            control_marker,
            torque_text,
        )

    return fig, animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )


def _animate_cartpole(
    result: DemoResult,
    fps: int,
    frame_step: int,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    states = result.states
    controls = result.controls[:, 0]
    time_points = result.time_points
    pole_length = float(result.metadata["pole_length"])
    frames = list(range(0, len(states), frame_step))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)

    fig = plt.figure(figsize=(8.8, 5.0), facecolor=_BACKGROUND)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.18)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_force = fig.add_subplot(gs[0, 1])
    _apply_title(fig, result)
    _style_scene_axes(ax_main)
    _style_axes(ax_force)

    ax_main.set_xlim(-2.0, 2.0)
    ax_main.set_ylim(-0.9, 1.2)
    ax_main.set_title("Swing-Up Motion", color=_TEXT, fontsize=11, pad=10)
    ax_main.axhline(0.0, color=_GRID, linewidth=2.0)
    ghost_cart = Rectangle((-0.18, -0.1), 0.36, 0.2, facecolor="none", edgecolor=_MUTED, linewidth=1.4, linestyle="--")
    ax_main.add_patch(ghost_cart)
    ghost_pole, = ax_main.plot([0.0, 0.0], [0.1, -pole_length + 0.1], color=_MUTED, linewidth=2.0, linestyle="--")

    ax_force.set_xlim(time_points[0], time_points[-2] if len(time_points) > 1 else 1.0)
    force_min, force_max = _metric_bounds(controls)
    ax_force.set_ylim(force_min, force_max)
    ax_force.set_title("Control Force", color=_TEXT, fontsize=11, pad=10)
    ax_force.set_xlabel("Time [s]", color=_TEXT)
    ax_force.set_ylabel("Force", color=_TEXT)

    cart = Rectangle((-0.18, -0.1), 0.36, 0.2, facecolor=_ACCENT, edgecolor=_TEXT, linewidth=1.2)
    ax_main.add_patch(cart)
    pole, = ax_main.plot([], [], color=_TERTIARY, linewidth=3.0)
    bob = Circle((0.0, 0.0), 0.05, color=_SECONDARY)
    ax_main.add_patch(bob)
    path_line, = ax_main.plot([], [], color=_SUCCESS, linewidth=1.6, alpha=0.8)
    state_text = ax_main.text(
        0.03,
        0.95,
        "",
        transform=ax_main.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=_TEXT,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ea", "edgecolor": _GRID},
    )

    force_line, = ax_force.plot(time_points[:-1], controls, color=_ACCENT, linewidth=2.0)
    force_marker, = ax_force.plot([], [], "o", color=_SECONDARY, markersize=7)

    def update(frame_index: int) -> tuple[Any, ...]:
        cart_x = states[frame_index, 0]
        theta = states[frame_index, 1]
        cart.set_x(cart_x - cart.get_width() / 2.0)
        pivot = np.array([cart_x, cart.get_y() + cart.get_height() / 2.0])
        pole_tip = pivot + np.array([pole_length * np.sin(theta), -pole_length * np.cos(theta)])
        pole.set_data([pivot[0], pole_tip[0]], [pivot[1], pole_tip[1]])
        bob.center = tuple(pole_tip)
        path_line.set_data(states[: frame_index + 1, 0], np.zeros(frame_index + 1))
        force_idx = min(frame_index, len(controls) - 1)
        force_marker.set_data([time_points[force_idx]], [controls[force_idx]])
        state_text.set_text(
            f"x = {cart_x:+.2f} m\nangle = {theta:+.2f} rad"
        )
        return (
            ghost_cart,
            ghost_pole,
            cart,
            pole,
            bob,
            path_line,
            force_line,
            force_marker,
            state_text,
        )

    return fig, animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )


def _animate_unicycle(
    result: DemoResult,
    fps: int,
    frame_step: int,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    states = result.states
    controls = result.controls
    time_points = result.time_points
    obstacle_center = np.asarray(result.metadata["obstacle_center"], dtype=float)
    obstacle_radius = float(result.metadata["obstacle_radius"])
    frames = list(range(0, len(states), frame_step))
    if frames[-1] != len(states) - 1:
        frames.append(len(states) - 1)

    fig = plt.figure(figsize=(9.2, 5.2), facecolor=_BACKGROUND)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.16,
        hspace=0.20,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_turn = fig.add_subplot(gs[1, 1])
    _apply_title(fig, result)
    _style_axes(ax_map)
    _style_axes(ax_speed)
    _style_axes(ax_turn)

    ax_map.set_xlim(-0.35, 2.35)
    ax_map.set_ylim(-0.35, 2.35)
    ax_map.set_aspect("equal")
    ax_map.set_title("Trajectory", color=_TEXT, fontsize=11, pad=10)
    ax_map.set_xlabel("x [m]", color=_TEXT)
    ax_map.set_ylabel("y [m]", color=_TEXT)
    ax_map.set_xticks([0.0, 1.0, 2.0])
    ax_map.set_yticks([0.0, 1.0, 2.0])

    obstacle = Circle(obstacle_center, obstacle_radius, facecolor="#fbd38d", edgecolor=_SECONDARY, linewidth=2.0, alpha=0.55)
    ax_map.add_patch(obstacle)
    ax_map.scatter(states[0, 0], states[0, 1], color=_SUCCESS, s=55)
    ax_map.scatter(result.target_state[0], result.target_state[1], color=_TERTIARY, s=55)
    ax_map.annotate("start", (states[0, 0], states[0, 1]), xytext=(6, -14), textcoords="offset points", color=_SUCCESS, fontsize=9)
    ax_map.annotate("goal", (result.target_state[0], result.target_state[1]), xytext=(6, 6), textcoords="offset points", color=_TERTIARY, fontsize=9)
    full_path, = ax_map.plot(states[:, 0], states[:, 1], color=_GRID, linewidth=1.8, linestyle="--")
    trail, = ax_map.plot([], [], color=_ACCENT, linewidth=2.4)
    vehicle = Rectangle((-0.12, -0.05), 0.24, 0.10, facecolor=_ACCENT, edgecolor=_TEXT, linewidth=1.2)
    ax_map.add_patch(vehicle)
    heading = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), arrowstyle="-|>", mutation_scale=12, color=_SECONDARY, linewidth=2.0)
    ax_map.add_patch(heading)

    control_speed = controls[:, 0]
    control_turn = controls[:, 1]
    ax_speed.set_xlim(time_points[0], time_points[-2] if len(time_points) > 1 else 1.0)
    speed_min, speed_max = _metric_bounds(control_speed)
    ax_speed.set_ylim(speed_min, speed_max)
    ax_speed.set_title("Linear Speed", color=_TEXT, fontsize=10, pad=8)
    ax_speed.set_ylabel("m/s", color=_TEXT)
    speed_line, = ax_speed.plot(time_points[:-1], control_speed, color=_ACCENT, linewidth=2.0)
    speed_marker, = ax_speed.plot([], [], "o", color=_ACCENT, markersize=7)

    ax_turn.set_xlim(time_points[0], time_points[-2] if len(time_points) > 1 else 1.0)
    turn_min, turn_max = _metric_bounds(control_turn)
    ax_turn.set_ylim(turn_min, turn_max)
    ax_turn.set_title("Turn Rate", color=_TEXT, fontsize=10, pad=8)
    ax_turn.set_xlabel("Time [s]", color=_TEXT)
    ax_turn.set_ylabel("rad/s", color=_TEXT)
    turn_line, = ax_turn.plot(time_points[:-1], control_turn, color=_SECONDARY, linewidth=2.0)
    turn_marker, = ax_turn.plot([], [], "o", color=_SECONDARY, markersize=7)

    status_text = ax_map.text(
        0.03,
        0.95,
        "",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=_TEXT,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ea", "edgecolor": _GRID},
    )

    def update(frame_index: int) -> tuple[Any, ...]:
        x, y, theta = states[frame_index]
        trail.set_data(states[: frame_index + 1, 0], states[: frame_index + 1, 1])
        vehicle.set_xy((x - vehicle.get_width() / 2.0, y - vehicle.get_height() / 2.0))
        vehicle.set_transform(
            Affine2D().rotate_around(x, y, theta) + ax_map.transData
        )
        arrow_length = 0.25
        heading.set_positions((x, y), (x + arrow_length * np.cos(theta), y + arrow_length * np.sin(theta)))
        ctrl_idx = min(frame_index, len(control_speed) - 1)
        speed_marker.set_data([time_points[ctrl_idx]], [control_speed[ctrl_idx]])
        turn_marker.set_data([time_points[ctrl_idx]], [control_turn[ctrl_idx]])
        obstacle_margin = np.min(
            np.linalg.norm(states[: frame_index + 1, :2] - obstacle_center, axis=1)
        ) - obstacle_radius
        status_text.set_text(
            f"heading = {theta:+.2f} rad\nclearance = {obstacle_margin:+.2f} m"
        )
        return (
            obstacle,
            full_path,
            trail,
            vehicle,
            heading,
            speed_line,
            speed_marker,
            turn_line,
            turn_marker,
            status_text,
        )

    return fig, animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )
