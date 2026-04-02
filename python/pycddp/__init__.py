"""Python bindings for Constrained Differential Dynamic Programming.

Main entry points:
- `CDDP` for solver setup and execution
- `DynamicalSystem` for system models
- `Objective` for cost functions
- `Constraint` for path and terminal constraints
"""

try:
    from pycddp._pycddp_core import (
    # Enums
    SolverType,
    BarrierStrategy,

    # Options
    CDDPOptions,
    BoxQPOptions,
    LineSearchOptions,
    RegularizationOptions,
    BarrierOptions,
    FilterOptions,
    InteriorPointOptions,
    LogBarrierOptions,
    IPDDPOptions,
    MSIPDDPOptions,

    # Core solver
    CDDP,
    CDDPSolution,
    SolutionHistory,

    # Dynamics base
    DynamicalSystem,

    # Concrete dynamics models
    Pendulum,
    Unicycle,
    Bicycle,
    Car,
    CartPole,
    DubinsCar,
    Forklift,
    Acrobot,
    Quadrotor,
    QuadrotorRate,
    Manipulator,
    HCW,
    SpacecraftLinearFuel,
    SpacecraftNonlinear,
    DreyfusRocket,
    SpacecraftLanding2D,
    SpacecraftTwobody,
    LTISystem,
    Usv3Dof,
    EulerAttitude,
    QuaternionAttitude,
    MrpAttitude,

    # Objectives
    Objective,
    QuadraticObjective,
    NonlinearObjective,

    # Constraints
    Constraint,
    ControlConstraint,
    StateConstraint,
    LinearConstraint,
    BallConstraint,
    PoleConstraint,
    SecondOrderConeConstraint,
    ThrustMagnitudeConstraint,
    MaxThrustMagnitudeConstraint,

    )
except ImportError as exc:
    raise ImportError(
        "Failed to import the native pycddp extension '_pycddp_core'. "
        "This usually means the extension was built for a different Python "
        "version or a required native runtime library is missing. Reinstall "
        "pycddp with the active interpreter and verify your C++ runtime "
        "dependencies."
    ) from exc

from pycddp._version import __version__

__all__ = [name for name in dir() if not name.startswith("_")]
