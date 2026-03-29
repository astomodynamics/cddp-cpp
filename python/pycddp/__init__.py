"""pycddp - Python bindings for CDDP trajectory optimization."""

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
    SpacecraftROE,
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

from pycddp._version import __version__

__all__ = [name for name in dir() if not name.startswith("_")]
