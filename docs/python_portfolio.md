# Python Portfolio

This gallery is the Python-facing showcase for the `pycddp` bindings. Each
demo is solved directly from Python, then rendered as an animation with
Matplotlib.

<div class="portfolio-hero">
  <div>
    <p class="portfolio-kicker">Interactive Demo Gallery</p>
    <h2>Solver results presented as a web-first portfolio</h2>
    <p>
      These examples are the quickest way to understand what the Python bindings
      actually produce: constrained solves, animated trajectories, and compact
      reproducible workflows.
    </p>
    <p>
      The gallery is intentionally path-focused. Terminal-equality and
      terminal-inequality support in <code>IPDDP</code> is covered in the solver
      regression suite rather than in a separate animation card.
    </p>
    <p>
      <a class="md-button md-button--primary" href="../python/">Python package guide</a>
      <a class="md-button" href="https://github.com/astomodynamics/cddp-cpp/tree/master/examples">Browse examples</a>
    </p>
  </div>
  <img src="../assets/cddp_in_cpp.png" alt="CDDP illustration">
</div>

## Regenerate the gallery

```bash
source .venv/bin/activate
python examples/python_portfolio.py --demo all --output-dir docs/assets/python_portfolio
```

## Demos

<div class="portfolio-grid">
  <section class="portfolio-card">
    <h3>Pendulum Swing-Up</h3>
    <p>Torque-limited <code>CLDDP</code> solve for a damped pendulum.</p>
    <img src="../assets/python_portfolio/pendulum_swing_up.gif" alt="Pendulum swing-up animation">
  </section>

  <section class="portfolio-card">
    <h3>Cart-Pole Swing-Up</h3>
    <p>
      Bounded-force <code>CLDDP</code> solve that swings the pole upright and
      settles near the origin.
    </p>
    <img src="../assets/python_portfolio/cartpole_swing_up.gif" alt="Cart-pole swing-up animation">
  </section>
  <section class="portfolio-card">
    <h3>Unicycle Obstacle Avoidance</h3>
    <p>
      <code>IPDDP</code> solve with a circular obstacle constraint, showing how
      the Python bindings can drive constrained trajectory visualization
      directly from Python.
    </p>
    <img src="../assets/python_portfolio/unicycle_obstacle_avoidance.gif" alt="Unicycle obstacle avoidance animation">
  </section>

  <section class="portfolio-card">
    <h3>MPCC Racing Line</h3>
    <p>
      Full-lap contouring control on a compact circuit, solved from Python with
      a custom nonlinear objective and a lightweight bicycle model.
    </p>
    <p>
      This is a compact kinematic MPCC-style showcase rather than a full
      reproduction of the controller in Liniger et al. The bundled track data
      is derived from the
      <a href="https://github.com/alexliniger/MPCC"><code>alexliniger/MPCC</code></a>
      reference implementation.
    </p>
    <img src="../assets/python_portfolio/mpcc_racing_line.gif" alt="MPCC racing line animation">
  </section>
</div>
