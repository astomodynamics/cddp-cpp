"""Closed-loop IPDDP MPCC on the bundled RC racing track.

This module is a pycddp IPDDP port of AIRCoM's acados kinematic MPCC
(``control_pkg/control_pkg/mpcc/solver.py``). It drives a 6 cm wheelbase
RC car around the same ``examples/data/mpcc_racing_track.csv`` that the
acados demo consumes, using a receding-horizon MPC loop with cold-seeded
solves.

Model — :class:`IpddpKinematicBicycle7`
    7-state augmented kinematic bicycle:
    ``[x, y, psi, theta, v_prev, delta_prev, v_theta_prev]`` with control
    ``[v_w, delta, v_theta]``. The last three states are control-history
    latches implemented via the continuous expression
    ``d(v_prev)/dt = (v_w - v_prev)/dt``; under Euler integration with
    step size ``dt`` this latches ``v_prev_next = v_w`` exactly, giving
    AIRCoM's rate-of-change residuals without having to override
    ``get_discrete_dynamics`` from Python (which the pycddp binding does
    not allow).

Cost — :class:`IpddpRcMpccObjective`
    11-residual cost mirroring AIRCoM's ``NONLINEAR_LS`` shape (contour,
    lag, speed, control regs, pose regs, three rate penalties) plus a
    smooth one-sided quadratic boundary penalty that stands in for
    AIRCoM's L1+quadratic slack on the soft track-boundary constraint
    (pycddp's IPDDP has no slack variables). Weight tuning — notably
    ``w_contour = 200`` and ``w_dv = 300`` — was chosen by sweep; see
    the git log for the experiment that drove the numbers.

What diverges from AIRCoM, and why
    * **No per-stage parameter vector.** AIRCoM sets ``p[0..8]`` per
      stage to hand the solver a time-varying reference; pycddp has no
      stage-parameter slot, so the reference is derived from ``theta``
      (state[3]) inside the cost via ``track.interpolate``.
    * **No soft-constraint slack.** See the ``w_boundary`` note above.
    * **Terminal regulariser zeroed.** AIRCoM's
      ``base_terminal_velocity_weight = 40`` is a *parking-mode* term
      that pulls the terminal ``v_prev`` toward zero; over 20 stages of
      the ``w_dv`` rate-penalty chain, that quietly bleeds the cruise
      speed every MPC tick. We anchor ``v_w`` directly via an
      ``w_speed_w * (v_w - v_target)^2`` term on the *running* cost
      instead, which has no backward-propagation artefact.
    * **Cold-seeded every tick.** A primal-warm-start from the previous
      IPDDP solution sounds like a good idea but in practice drags the
      new solve into a stale basin near the old terminal — iter counts
      climb and the first-stage control drifts off the line. A fresh
      forward-roll along the reference converges in ~15-40 iterations
      and produces clean tracking.

Entry points
    * :func:`solve_ipddp_mpcc_rc` — single-shot open-loop trajectory
      optimisation against the track.
    * :func:`run_ipddp_mpc` — closed-loop receding-horizon MPC loop that
      feeds a ``DemoResult``-compatible history object to the existing
      ``_animate_mpcc`` portfolio renderer.
    * CLI: ``python ipddp_mpcc_rc.py --mpc --lap`` runs a full lap and
      writes ``examples/out_ipddp_mpcc_rc/ipddp_mpcc_rc.gif``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make ``python_portfolio_lib`` importable regardless of how this file is
# loaded (direct CLI invocation or lazy import from ``python_portfolio_lib``
# itself).
_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import pycddp  # noqa: E402

if __package__:
    from .python_portfolio_lib import (  # noqa: E402
        TrackData,
        _load_track_csv,
        _portfolio_track_path,
        _wrap_angle,
    )
else:
    from python_portfolio_lib import (  # noqa: E402
        TrackData,
        _load_track_csv,
        _portfolio_track_path,
        _wrap_angle,
    )


# ---------------------------------------------------------------------------
# 7-state augmented kinematic bicycle (AIRCoM-style)
# ---------------------------------------------------------------------------


class IpddpKinematicBicycle7(pycddp.DynamicalSystem):
    """7-state augmented kinematic bicycle that mirrors AIRCoM's MPCC model.

    State (7-D):  ``[x, y, psi, theta, v_prev, delta_prev, v_theta_prev]``
    Control (3-D): ``[v_w, delta, v_theta]``

    The first four states are the standard MPCC kinematic states (planar
    pose plus the progress variable). The last three are *latches* that
    hold the previously-applied control. They exist purely so the cost
    can penalise rate-of-change residuals
    ``(v_w - v_prev), (delta - delta_prev), (v_theta - v_theta_prev)`` —
    exactly the smoothing AIRCoM uses to keep the controls calm.

    The latch is implemented at the **continuous-dynamics** level via
    ``d(v_prev)/dt = (v_w - v_prev) / dt``. Under Euler integration with
    step size equal to ``dt``, this expression yields ``v_prev_next = v_w``
    *exactly*, so we get the AIRCoM latching behaviour without having to
    override ``get_discrete_dynamics`` from Python (which the pycddp
    binding does not allow). Use ``"euler"`` as the integrator type or
    the latch will only be approximate.
    """

    def __init__(self, timestep: float, wheelbase: float = 0.20) -> None:
        super().__init__(7, 3, timestep, "euler")
        self.dt = float(timestep)
        self.wheelbase = float(wheelbase)
        self._inv_dt = 1.0 / float(timestep)

    def get_continuous_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        psi = float(state[2])
        v_prev = float(state[4])
        delta_prev = float(state[5])
        v_theta_prev = float(state[6])
        v_w = float(control[0])
        delta = float(control[1])
        v_theta = float(control[2])
        return np.array(
            [
                v_w * np.cos(psi),
                v_w * np.sin(psi),
                v_w * np.tan(delta) / self.wheelbase,
                v_theta,
                (v_w - v_prev) * self._inv_dt,
                (delta - delta_prev) * self._inv_dt,
                (v_theta - v_theta_prev) * self._inv_dt,
            ],
            dtype=float,
        )

    def get_state_jacobian(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        psi = float(state[2])
        v_w = float(control[0])
        A = np.zeros((7, 7), dtype=float)
        A[0, 2] = -v_w * np.sin(psi)
        A[1, 2] = v_w * np.cos(psi)
        A[4, 4] = -self._inv_dt
        A[5, 5] = -self._inv_dt
        A[6, 6] = -self._inv_dt
        return A

    def get_control_jacobian(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float = 0.0,
    ) -> np.ndarray:
        psi = float(state[2])
        v_w = float(control[0])
        delta = float(control[1])
        sec_sq = 1.0 / (np.cos(delta) ** 2)
        B = np.zeros((7, 3), dtype=float)
        B[0, 0] = np.cos(psi)
        B[1, 0] = np.sin(psi)
        B[2, 0] = np.tan(delta) / self.wheelbase
        B[2, 1] = v_w * sec_sq / self.wheelbase
        B[3, 2] = 1.0
        B[4, 0] = self._inv_dt
        B[5, 1] = self._inv_dt
        B[6, 2] = self._inv_dt
        return B

    def get_state_hessian(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float = 0.0,
    ) -> list[np.ndarray]:
        return [np.zeros((7, 7), dtype=float) for _ in range(7)]

    def get_control_hessian(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float = 0.0,
    ) -> list[np.ndarray]:
        return [np.zeros((3, 3), dtype=float) for _ in range(7)]

    def get_cross_hessian(
        self,
        state: np.ndarray,
        control: np.ndarray,
        time: float = 0.0,
    ) -> list[np.ndarray]:
        return [np.zeros((3, 7), dtype=float) for _ in range(7)]


# State indices for the 7-state augmented model.
IDX_X = 0
IDX_Y = 1
IDX_PSI = 2
IDX_THETA = 3
IDX_V_PREV = 4
IDX_DELTA_PREV = 5
IDX_V_THETA_PREV = 6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IpddpRcMpccConfig:
    """Knobs for the IPDDP MPCC problem (AIRCoM kinematic shape).

    Cost weights mirror the AIRCoM ``MpccConfig`` defaults for the
    *kinematic* MPCC (the simpler model in
    ``control_pkg/control_pkg/mpcc/solver.py``), with one departure: the
    soft track-boundary constraint is replaced by a smooth quadratic
    penalty in the cost because pycddp's IPDDP has no slack variables.
    """

    dt: float = 0.05
    # Horizon 20 × dt = 1 s lookahead, ~1.5 m at the 1.5 m/s cruise we
    # actually execute. A grid probe (H∈{20,25} × band∈{0.85,0.90}) showed
    # that longer horizon costs ~20% more wall-clock per step and doesn't
    # improve the trajectory at all — the rate-penalty smoothing already
    # sees as far forward as it needs to.
    horizon: int = 20
    # AIRCoM RC car: wheelbase = lf + lr = 0.029 + 0.033 = 0.062 m. With
    # delta_max = 0.6 rad the bicycle's min turning radius is
    # L/tan(0.6) ≈ 0.091 m, ~2x tighter than the 0.185 m min radius of the
    # bundled CSV track.
    wheelbase: float = 0.062
    # Floor used when the track's curvature-scaled ``v_ref`` drops below
    # this. AIRCoM keeps this at 1.0 m/s. The *upper* end of the speed
    # profile comes from the CSV (which caps at 2.2 m/s on straights) via
    # ``v_target = max(ref.v_ref, reference_speed)`` in the cost.
    reference_speed: float = 1.0

    # Control bounds (hard, via pycddp.ControlConstraint). AIRCoM's acados
    # version caps v_max at 1.5 m/s because the Pacejka tire model gets
    # numerically stiff above that for a 6 cm wheelbase. The kinematic
    # bicycle has no such issue, so we let v_w follow the full 2.2 m/s
    # track-speed profile on the straights for a proper racing line.
    speed_min: float = 0.1
    speed_max: float = 2.2
    delta_max: float = 0.60
    v_theta_min: float = 0.0
    v_theta_max: float = 2.2

    # 11-residual cost weights (NONLINEAR_LS diagonal W). Names match
    # AIRCoM's ``control_pkg/.../mpcc/solver.py`` but the magnitudes are
    # tuned for IPDDP on this track — see the weight-sweep in the commit
    # message for the experimental justification.
    #
    # Notable deviations from AIRCoM's defaults:
    # * ``w_contour = 200`` (up from 100): tightens tracking AND makes the
    #   cost Hessian more strictly convex in position, so IPDDP converges
    #   ~12% faster per step.
    # * ``w_dv = 300`` (down from 1000): over-strong rate penalty was the
    #   main per-step cost driver without meaningfully improving
    #   smoothness. Lowering it speeds solves up ~14%.
    w_contour: float = 200.0   # (n . d)^2
    w_lag: float = 100.0       # (t . d)^2
    w_speed: float = 5.0       # (v_theta - v_ref)^2
    w_control: float = 0.1     # v_w^2 and delta^2
    w_x: float = 0.0           # dx_err^2 (zero by default like AIRCoM)
    w_y: float = 0.0           # dy_err^2
    w_yaw: float = 0.0         # e_yaw^2
    # Extra: explicit v_w tracking on every stage. AIRCoM doesn't have this
    # (their v_w comes for free through lag-error coupling + terminal v_prev
    # regulariser), but here it anchors v_w directly so the rate-penalty
    # chain has nothing to bleed against. Cheaper and more stable for IPDDP
    # than a pure terminal anchor.
    w_speed_w: float = 10.0    # (v_w - v_target)^2 on every stage
    # Rate-of-change weights — the key smoothing terms.
    w_dv: float = 300.0
    w_ddelta: float = 1000.0
    w_dv_theta: float = 100.0

    # Smooth track-boundary penalty (replaces AIRCoM's L1+quadratic slack).
    # The grid probe showed the band doesn't get reached on this track —
    # w_contour (100) already pulls the car within 0.12 m of the
    # centerline, well inside 0.85 * 0.18 = 0.153 m. This is a safety
    # margin, not an active shaping term.
    w_boundary: float = 200.0
    boundary_band: float = 0.85

    # Terminal weights — AIRCoM W_e diagonal, but with the velocity/steering
    # anchors *zeroed out*. AIRCoM uses those for parking mode (pull the car
    # to a stop at the goal); for racing-line MPC they propagate backward
    # through 20 stages of the rate-penalty chain and bleed the forward
    # speed every tick. We replace their role with ``w_speed_w`` on the
    # running cost, which anchors v_w directly at every stage.
    w_terminal: float = 50.0
    base_terminal_velocity_weight: float = 0.0
    base_terminal_steering_weight: float = 0.0
    # Linear reward for advancing the progress variable. Small — just a
    # tiebreaker so the terminal cost prefers reaching farther ``theta``.
    w_terminal_progress: float = 2.0

    # IPDDP solver options.
    max_iterations: int = 100
    tolerance: float = 1e-4
    acceptable_tolerance: float = 5e-4
    initial_regularization: float = 1e-4
    line_search_iters: int = 12


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


class IpddpRcMpccObjective(pycddp.NonlinearObjective):
    """11-residual MPCC cost matching AIRCoM's NONLINEAR_LS shape.

    Operates on the 7-state augmented kinematic bicycle:
    ``state = [x, y, psi, theta, v_prev, delta_prev, v_theta_prev]`` and
    ``control = [v_w, delta, v_theta]``.

    The running cost is ``dt * sum_i w_i * r_i^2`` over the same eleven
    residuals AIRCoM uses (`solver.py:240`):

    1.  contour error ``e_c = n . (p - p_ref)``
    2.  lag error ``e_l = t . (p - p_ref)``
    3.  speed-tracking ``v_theta - v_ref``
    4.  control regulariser on ``v_w``
    5.  control regulariser on ``delta``
    6.  position error ``dx``  (zero weight by default — AIRCoM)
    7.  position error ``dy``  (zero weight by default — AIRCoM)
    8.  yaw error ``e_yaw``    (zero weight by default — AIRCoM)
    9.  rate ``v_w - v_prev``
    10. rate ``delta - delta_prev``
    11. rate ``v_theta - v_theta_prev``

    Plus a smooth quadratic boundary penalty
    ``w_boundary * max(0, |e_c| - boundary_band * half_width)^2``
    that stands in for AIRCoM's L1+quadratic slack on the soft
    ``-half_width <= e_c <= half_width`` constraint (pycddp IPDDP has no
    slack variables).

    The reference parameters AIRCoM passes via ``p[stage]`` are looked
    up from ``theta`` (state[3]) on the track instead — that turns the
    time-varying reference into a function of state, which is what IPDDP
    needs.
    """

    def __init__(self, cfg: IpddpRcMpccConfig, track: TrackData) -> None:
        super().__init__(cfg.dt)
        self.cfg = cfg
        self.track = track
        self.dt = float(cfg.dt)
        self._half_width = float(track.width)

    def _tracking(self, state: np.ndarray) -> tuple[float, float, float, float, float, float]:
        theta = float(state[IDX_THETA])
        ref = self.track.interpolate(theta)
        dx = float(state[IDX_X]) - ref.x
        dy = float(state[IDX_Y]) - ref.y
        e_c = ref.normal[0] * dx + ref.normal[1] * dy
        e_l = ref.tangent[0] * dx + ref.tangent[1] * dy
        e_yaw = _wrap_angle(float(state[IDX_PSI]) - ref.heading)
        return e_c, e_l, e_yaw, ref.v_ref, dx, dy

    def running_cost(
        self,
        state: np.ndarray,
        control: np.ndarray,
        index: int,
    ) -> float:
        cfg = self.cfg
        e_c, e_l, e_yaw, v_ref_track, dx, dy = self._tracking(state)
        v_prev = float(state[IDX_V_PREV])
        delta_prev = float(state[IDX_DELTA_PREV])
        v_theta_prev = float(state[IDX_V_THETA_PREV])
        v_w = float(control[0])
        delta = float(control[1])
        v_theta = float(control[2])

        v_target = max(v_ref_track, cfg.reference_speed)
        boundary = max(0.0, abs(e_c) - cfg.boundary_band * self._half_width)

        return float(
            self.dt
            * (
                cfg.w_contour * e_c * e_c
                + cfg.w_lag * e_l * e_l
                + cfg.w_speed * (v_theta - v_target) ** 2
                + cfg.w_speed_w * (v_w - v_target) ** 2
                + cfg.w_control * v_w * v_w
                + cfg.w_control * delta * delta
                + cfg.w_x * dx * dx
                + cfg.w_y * dy * dy
                + cfg.w_yaw * e_yaw * e_yaw
                + cfg.w_dv * (v_w - v_prev) ** 2
                + cfg.w_ddelta * (delta - delta_prev) ** 2
                + cfg.w_dv_theta * (v_theta - v_theta_prev) ** 2
                + cfg.w_boundary * boundary * boundary
            )
        )

    def terminal_cost(self, final_state: np.ndarray) -> float:
        cfg = self.cfg
        e_c, e_l, _e_yaw, _v_ref, _dx, _dy = self._tracking(final_state)
        theta = float(final_state[IDX_THETA])
        return float(
            cfg.w_terminal * e_c * e_c
            + cfg.w_terminal * e_l * e_l
            - cfg.w_terminal_progress * theta
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class IpddpRcMpccResult:
    """One-shot IPDDP MPCC solution."""

    solution: pycddp.CDDPSolution
    states: np.ndarray
    controls: np.ndarray
    time_points: np.ndarray
    track: TrackData
    cfg: IpddpRcMpccConfig

    @property
    def lap_progress(self) -> float:
        if self.states.size == 0:
            return 0.0
        return float(self.states[-1, IDX_THETA] - self.states[0, IDX_THETA])

    @property
    def contour_errors(self) -> np.ndarray:
        out = np.empty(self.states.shape[0], dtype=float)
        for i, state in enumerate(self.states):
            ref = self.track.interpolate(float(state[IDX_THETA]))
            d = state[:2] - np.array([ref.x, ref.y], dtype=float)
            out[i] = float(ref.normal @ d)
        return out

    @property
    def lag_errors(self) -> np.ndarray:
        out = np.empty(self.states.shape[0], dtype=float)
        for i, state in enumerate(self.states):
            ref = self.track.interpolate(float(state[IDX_THETA]))
            d = state[:2] - np.array([ref.x, ref.y], dtype=float)
            out[i] = float(ref.tangent @ d)
        return out


# ---------------------------------------------------------------------------
# Solver setup + run
# ---------------------------------------------------------------------------


def _initial_state(track: TrackData, cfg: IpddpRcMpccConfig) -> np.ndarray:
    """Place the car on the centerline at theta = s[0], aligned heading.

    The control-latch components are seeded at the cruise speed so the very
    first stage's rate residuals (``v_w - v_prev`` etc.) start near zero —
    important for the strong AIRCoM rate-penalty weights.
    """
    ref = track.interpolate(float(track.s[0]))
    cruise = float(np.clip(cfg.reference_speed, cfg.speed_min, cfg.speed_max))
    v_theta = float(np.clip(cruise, cfg.v_theta_min, cfg.v_theta_max))
    return np.array(
        [
            ref.x,
            ref.y,
            ref.heading,
            float(track.s[0]),
            cruise,    # v_prev
            0.0,       # delta_prev
            v_theta,   # v_theta_prev
        ],
        dtype=float,
    )


def _seed_controls(
    track: TrackData,
    cfg: IpddpRcMpccConfig,
    initial_progress: float,
) -> list[np.ndarray]:
    """Cold-start control seed: roll forward along the reference at v_ref."""
    seeds: list[np.ndarray] = []
    progress = float(initial_progress)
    wheelbase = cfg.wheelbase
    for _ in range(cfg.horizon):
        ref = track.interpolate(progress)
        v_target = float(np.clip(max(ref.v_ref, cfg.reference_speed), cfg.speed_min, cfg.speed_max))
        steer_guess = float(
            np.clip(np.arctan(wheelbase * ref.curvature), -cfg.delta_max, cfg.delta_max)
        )
        v_theta = float(np.clip(v_target, cfg.v_theta_min, cfg.v_theta_max))
        seeds.append(np.array([v_target, steer_guess, v_theta], dtype=float))
        progress += cfg.dt * v_theta
    return seeds


def _rollout(
    model: IpddpKinematicBicycle7, x0: np.ndarray, controls: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    state = x0.copy()
    states = [state.copy()]
    rolled: list[np.ndarray] = []
    for u in controls:
        u_arr = np.asarray(u, dtype=float).copy()
        rolled.append(u_arr)
        state = np.asarray(model.get_discrete_dynamics(state, u_arr), dtype=float)
        states.append(state.copy())
    return states, rolled


def build_solver(
    cfg: IpddpRcMpccConfig,
    track: TrackData,
) -> tuple[pycddp.CDDP, IpddpKinematicBicycle7, IpddpRcMpccObjective, np.ndarray]:
    """Build a fully configured pycddp CDDP solver for the simplified MPCC."""
    model = IpddpKinematicBicycle7(cfg.dt, wheelbase=cfg.wheelbase)
    objective = IpddpRcMpccObjective(cfg, track)

    opts = pycddp.CDDPOptions()
    opts.max_iterations = cfg.max_iterations
    opts.tolerance = cfg.tolerance
    opts.acceptable_tolerance = cfg.acceptable_tolerance
    opts.regularization.initial_value = cfg.initial_regularization
    opts.line_search.max_iterations = cfg.line_search_iters
    opts.use_ilqr = True  # avoid relying on dynamics Hessians
    opts.verbose = False
    opts.print_solver_header = False
    opts.return_iteration_info = True

    x0 = _initial_state(track, cfg)
    # Reference state argument is consumed by the QuadraticObjective path; for
    # NonlinearObjective the solver only uses it as bookkeeping, so we point it
    # at the same starting reference.
    x_ref = x0.copy()

    solver = pycddp.CDDP(x0, x_ref, cfg.horizon, cfg.dt, opts)
    solver.set_dynamical_system(model)
    solver.set_objective(objective)
    solver.add_constraint(
        "control_limits",
        pycddp.ControlConstraint(
            np.array([cfg.speed_min, -cfg.delta_max, cfg.v_theta_min], dtype=float),
            np.array([cfg.speed_max, cfg.delta_max, cfg.v_theta_max], dtype=float),
        ),
    )

    return solver, model, objective, x0


def solve_ipddp_mpcc_rc(
    cfg: IpddpRcMpccConfig | None = None,
    track: TrackData | None = None,
) -> IpddpRcMpccResult:
    """Run a single open-loop IPDDP solve of the simplified MPCC."""
    if cfg is None:
        cfg = IpddpRcMpccConfig()
    if track is None:
        track = _load_track_csv(_portfolio_track_path())

    solver, model, _objective, x0 = build_solver(cfg, track)

    seed_u = _seed_controls(track, cfg, float(x0[3]))
    seed_x, seed_u = _rollout(model, x0, seed_u)
    solver.set_initial_trajectory(seed_x, seed_u)

    solution = solver.solve(pycddp.SolverType.IPDDP)
    if not solution.state_trajectory or not solution.control_trajectory:
        raise RuntimeError("IPDDP MPCC RC solve produced an empty trajectory.")

    states = np.vstack([np.asarray(x, dtype=float) for x in solution.state_trajectory])
    controls = np.vstack([np.asarray(u, dtype=float) for u in solution.control_trajectory])
    time_points = np.asarray(solution.time_points, dtype=float)

    return IpddpRcMpccResult(
        solution=solution,
        states=states,
        controls=controls,
        time_points=time_points,
        track=track,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Closed-loop MPC runner
# ---------------------------------------------------------------------------


@dataclass
class IpddpRcMpccHistory:
    """Closed-loop history captured per MPC step."""

    executed_states: list[np.ndarray] = field(default_factory=list)
    executed_controls: list[np.ndarray] = field(default_factory=list)
    contour_errors: list[float] = field(default_factory=list)
    lag_errors: list[float] = field(default_factory=list)
    solve_times_ms: list[float] = field(default_factory=list)
    predicted_paths: list[np.ndarray] = field(default_factory=list)
    reference_windows: list[np.ndarray] = field(default_factory=list)
    failures: int = 0


def _reference_window_from_predicted(
    track: TrackData, predicted_states: list[np.ndarray]
) -> np.ndarray:
    """Sample the track centerline at the progress of each predicted stage.

    The visualization overlays this alongside the IPDDP prediction, so the
    two lines need to span the **same** stages — sampling at a fixed
    ``reference_speed`` stride makes the orange line shorter than the red
    one whenever the solver is running above that cruise speed (the case
    on straights). Keying off ``predicted_states[k][IDX_THETA]`` keeps the
    orange line aligned with what the solver is actually aiming at.
    """
    pts = np.empty((len(predicted_states), 2), dtype=float)
    for stage, x in enumerate(predicted_states):
        ref = track.interpolate(float(x[IDX_THETA]))
        pts[stage, 0] = ref.x
        pts[stage, 1] = ref.y
    return pts


def run_ipddp_mpc(
    cfg: IpddpRcMpccConfig | None = None,
    track: TrackData | None = None,
    *,
    simulation_steps: int = 60,
    stop_at_progress: float | None = None,
    show_progress: bool = False,
) -> IpddpRcMpccHistory:
    """Run closed-loop receding-horizon IPDDP MPC.

    At each step we re-solve the IPDDP problem from the current plant state,
    apply the first control to the kinematic bicycle, and cold-seed the
    next solve from a forward-roll along the reference. The whole history
    is captured in a form compatible with ``_animate_mpcc``.

    If ``stop_at_progress`` is given, the simulation breaks as soon as the
    executed ``theta`` crosses that threshold — use
    ``stop_at_progress=track.length`` to stop after one full lap.
    ``simulation_steps`` still acts as an upper bound so a wandering run
    cannot loop forever.
    """
    if cfg is None:
        cfg = IpddpRcMpccConfig()
    if track is None:
        track = _load_track_csv(_portfolio_track_path())
    if simulation_steps < 1:
        raise ValueError("simulation_steps must be >= 1")

    solver, model, _objective, x0 = build_solver(cfg, track)

    state = x0.copy()
    history = IpddpRcMpccHistory(executed_states=[state.copy()])

    last_control = np.zeros(3, dtype=float)

    for step_index in range(simulation_steps):
        # Track-relative metrics at the current state.
        ref_now = track.interpolate(float(state[IDX_THETA]))
        d_pos = state[:2] - np.array([ref_now.x, ref_now.y], dtype=float)
        history.contour_errors.append(float(ref_now.normal @ d_pos))
        history.lag_errors.append(float(ref_now.tangent @ d_pos))

        # Cold-seed every step. The shifted warm-start from the previous
        # IPDDP solution sounds like it should be a good idea but in
        # practice it tends to drag the new solve into a stale basin near
        # the old terminal: iter counts climb and the first-stage control
        # drifts off the line. A fresh forward-roll along the reference
        # happens to be both faster and far more stable for this problem.
        seed_u = _seed_controls(track, cfg, float(state[IDX_THETA]))
        seed_x, seed_u = _rollout(model, state, seed_u)
        solver.set_initial_state(state)
        solver.set_initial_trajectory(seed_x, seed_u)

        t0 = time.perf_counter()
        sol = solver.solve(pycddp.SolverType.IPDDP)
        history.solve_times_ms.append((time.perf_counter() - t0) * 1000.0)

        if not sol.state_trajectory or not sol.control_trajectory:
            history.failures += 1
            control = last_control.copy()
            predicted_states = [state.copy() for _ in range(cfg.horizon + 1)]
            predicted_controls = [last_control.copy() for _ in range(cfg.horizon)]
        else:
            predicted_states = [
                np.asarray(x, dtype=float).copy() for x in sol.state_trajectory
            ]
            predicted_controls = [
                np.asarray(u, dtype=float).copy() for u in sol.control_trajectory
            ]
            control = predicted_controls[0].copy()
            if not np.all(np.isfinite(control)):
                history.failures += 1
                control = last_control.copy()
            else:
                last_control = control.copy()

        history.predicted_paths.append(
            np.vstack([s[:2] for s in predicted_states])
        )
        history.reference_windows.append(
            _reference_window_from_predicted(track, predicted_states)
        )

        # Hard-clip the control (defensive — IPDDP should already respect bounds).
        control[0] = float(np.clip(control[0], cfg.speed_min, cfg.speed_max))
        control[1] = float(np.clip(control[1], -cfg.delta_max, cfg.delta_max))
        control[2] = float(np.clip(control[2], cfg.v_theta_min, cfg.v_theta_max))

        # Advance the plant.
        next_state = np.asarray(model.get_discrete_dynamics(state, control), dtype=float)
        next_state[IDX_PSI] = _wrap_angle(float(next_state[IDX_PSI]))
        history.executed_controls.append(control.copy())
        state = next_state
        history.executed_states.append(state.copy())

        if show_progress and (
            step_index == 0
            or step_index + 1 == simulation_steps
            or (step_index + 1) % 5 == 0
        ):
            print(
                f"[ipddp_mpc] step {step_index + 1:3d}/{simulation_steps} "
                f"solve={history.solve_times_ms[-1]:6.1f} ms iter={sol.iterations_completed:3d} "
                f"contour={history.contour_errors[-1]:+.3f} progress={float(state[IDX_THETA]):.2f} m",
                flush=True,
            )

        if stop_at_progress is not None and float(state[IDX_THETA]) >= stop_at_progress:
            if show_progress:
                print(
                    f"[ipddp_mpc] lap finished at step {step_index + 1}, "
                    f"progress={float(state[IDX_THETA]):.2f} m",
                    flush=True,
                )
            break

    # Final tracking pair so the time series matches len(executed_states).
    final_ref = track.interpolate(float(state[3]))
    final_d = state[:2] - np.array([final_ref.x, final_ref.y], dtype=float)
    history.contour_errors.append(float(final_ref.normal @ final_d))
    history.lag_errors.append(float(final_ref.tangent @ final_d))

    return history


def history_to_demo_result(
    history: IpddpRcMpccHistory,
    track: TrackData,
    cfg: IpddpRcMpccConfig,
) -> "object":
    """Wrap the MPC history as a ``DemoResult`` so ``save_animation`` works."""
    if not history.executed_controls:
        raise ValueError("history must contain at least one executed control")
    if __package__:
        from .python_portfolio_lib import DemoResult  # local import
    else:
        from python_portfolio_lib import DemoResult  # local import

    full_states = np.vstack(history.executed_states)
    # The animator destructures (x, y, heading, progress) from each state row,
    # so project the 7-state augmented vector down to its first four
    # components before handing it over.
    states = full_states[:, :4]
    controls = np.vstack(history.executed_controls)
    time_points_arr = np.arange(len(history.executed_states), dtype=float) * cfg.dt
    final_ref = track.interpolate(float(full_states[-1, IDX_THETA]))
    target_state = np.array(
        [final_ref.x, final_ref.y, final_ref.heading, float(full_states[-1, IDX_THETA])],
        dtype=float,
    )

    # Synthesize a minimal CDDPSolution-shaped object — _animate_mpcc only
    # touches result.metadata + result.states/controls/time_points/target_state,
    # not the solution itself, so a sentinel with the right attributes works.
    stub = type(
        "_StubSolution",
        (),
        {
            "solver_name": "IPDDP MPC",
            "final_objective": 0.0,
            "final_primal_infeasibility": 0.0,
            "iterations_completed": 0,
            "solve_time_ms": float(np.sum(history.solve_times_ms)),
            "state_trajectory": list(history.executed_states),
            "control_trajectory": list(history.executed_controls),
            "time_points": time_points_arr,
        },
    )()

    return DemoResult(
        slug="mpcc_racing_line",
        title="IPDDP MPCC RC",
        solver_name="IPDDP MPC",
        solution=stub,
        states=states,
        controls=controls,
        time_points=time_points_arr,
        target_state=target_state,
        metadata={
            "timestep": cfg.dt,
            "track": track,
            "track_width": track.width,
            "contour_errors": np.asarray(history.contour_errors, dtype=float),
            "lag_errors": np.asarray(history.lag_errors, dtype=float),
            "solve_times_ms": np.asarray(history.solve_times_ms, dtype=float),
            "predicted_paths": history.predicted_paths,
            "reference_windows": history.reference_windows,
            "subtitle": "Closed-loop IPDDP receding-horizon MPC on the bundled racing track",
            "final_error_override": float(
                np.hypot(history.contour_errors[-1], history.lag_errors[-1])
            ),
        },
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="IPDDP MPCC RC demo.")
    parser.add_argument("--mpc", action="store_true", help="Run closed-loop MPC and save a GIF.")
    parser.add_argument("--steps", type=int, default=60, help="Closed-loop simulation steps (MPC mode). Ignored if --lap.")
    parser.add_argument("--lap", action="store_true",
                        help="Run until a full lap is completed (theta >= track.length).")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Safety cap on simulation steps when running a full lap.")
    parser.add_argument("--horizon", type=int, default=12, help="MPC horizon length.")
    parser.add_argument("--cap", type=int, default=60, help="Per-solve IPDDP iteration cap (MPC mode).")
    parser.add_argument("--gif", type=str, default="examples/out_ipddp_mpcc_rc/ipddp_mpcc_rc.gif",
                        help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    if args.cap < 1:
        parser.error("--cap must be >= 1")
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    if args.dpi < 1:
        parser.error("--dpi must be >= 1")

    if args.mpc:
        cfg = IpddpRcMpccConfig(
            horizon=args.horizon,
            max_iterations=args.cap,
            dt=0.05,
        )
        track = _load_track_csv(_portfolio_track_path())
        if args.lap:
            steps = args.max_steps
            stop_at = float(track.length)
        else:
            steps = args.steps
            stop_at = None
        history = run_ipddp_mpc(
            cfg,
            track,
            simulation_steps=steps,
            stop_at_progress=stop_at,
            show_progress=True,
        )
        print(
            f"[ipddp_mpc] total solve = {sum(history.solve_times_ms):.0f} ms, "
            f"failures = {history.failures}"
        )

        if __package__:
            from .python_portfolio_lib import save_animation  # local import
        else:
            from python_portfolio_lib import save_animation  # local import
        result = history_to_demo_result(history, track, cfg)
        out = save_animation(result, args.gif, fps=args.fps, dpi=args.dpi, frame_step=1)
        print(f"[ipddp_mpc] saved gif -> {out}")
        return

    cfg = IpddpRcMpccConfig()
    result = solve_ipddp_mpcc_rc(cfg)
    sol = result.solution
    print(f"[ipddp_mpcc_rc] solver        = {sol.solver_name}")
    print(f"[ipddp_mpcc_rc] iterations    = {sol.iterations_completed}")
    print(f"[ipddp_mpcc_rc] solve_time_ms = {sol.solve_time_ms:.1f}")
    print(f"[ipddp_mpcc_rc] objective     = {sol.final_objective:+.4f}")
    print(
        f"[ipddp_mpcc_rc] primal_inf    = {sol.final_primal_infeasibility:+.3e}"
    )
    print(f"[ipddp_mpcc_rc] lap_progress  = {result.lap_progress:.3f} m")
    contour = result.contour_errors
    print(
        f"[ipddp_mpcc_rc] |contour|: max={np.max(np.abs(contour)):.3f}, "
        f"rms={float(np.sqrt(np.mean(contour ** 2))):.3f}"
    )


if __name__ == "__main__":
    main()
