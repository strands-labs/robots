"""Strands Robots Simulation — MuJoCo-based programmatic simulation.

Re-exports all public API from the old monolithic simulation.py so that
existing imports continue to work unchanged:

    from strands_robots.simulation import Simulation
    from strands_robots.simulation import SimWorld, SimRobot, MJCFBuilder
    from strands_robots.simulation import resolve_model, register_urdf
    from strands_robots.simulation import _configure_gl_backend
"""

from strands_robots.simulation._backend import _configure_gl_backend, _ensure_mujoco, _is_headless
from strands_robots.simulation._mjcf_builder import MJCFBuilder
from strands_robots.simulation._model_registry import (
    list_available_models,
    list_registered_urdfs,
    register_urdf,
    resolve_model,
    resolve_urdf,
)
from strands_robots.simulation._models import (
    SimCamera,
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)
from strands_robots.simulation.simulation import Simulation

# Convenience alias — allows ``from strands_robots.simulation import simulation``
simulation = Simulation

__all__ = [
    "Simulation",
    "SimStatus",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "SimWorld",
    "TrajectoryStep",
    "MJCFBuilder",
    "register_urdf",
    "resolve_model",
    "resolve_urdf",
    "list_registered_urdfs",
    "list_available_models",
    "simulation",
    "_configure_gl_backend",
    "_ensure_mujoco",
    "_is_headless",
]
