# Python Portfolio

This gallery is the Python-facing showcase for the `pycddp` bindings. Each
demo is solved directly from Python, then rendered as an animation with
Matplotlib.

## Regenerate the gallery

```bash
source .venv/bin/activate
python examples/python_portfolio.py --demo all --output-dir docs/assets/python_portfolio
```

## Demos

### Pendulum Swing-Up

Torque-limited `CLDDP` solve for a damped pendulum.

<img src="assets/python_portfolio/pendulum_swing_up.gif" width="820" alt="Pendulum swing-up animation">

### Cart-Pole Swing-Up

Bounded-force `CLDDP` solve that swings the pole upright and settles near the
origin.

<img src="assets/python_portfolio/cartpole_swing_up.gif" width="820" alt="Cart-pole swing-up animation">

### Unicycle Obstacle Avoidance

`IPDDP` solve with a circular obstacle constraint, showing how the Python
bindings can be used for constrained trajectory visualization without building
the legacy C++ plotting stack.

<img src="assets/python_portfolio/unicycle_obstacle_avoidance.gif" width="820" alt="Unicycle obstacle avoidance animation">
