"""Strands Robots Simulation - multi-backend simulation framework.

Architecture::

    simulation/
    ├ __init__.py          ← this file (re-exports, lazy loading)
    ├ base.py              ← SimEngine ABC
    ├ factory.py           ← create_simulation() + backend registration
    ├ models.py            ← shared dataclasses (SimWorld, SimRobot, ...)
    ├ model_registry.py    ← URDF/MJCF resolution (shared across backends)
    └ mujoco/              ← MuJoCo CPU backend
        ├ __init__.py
        ├ backend.py       ← lazy mujoco import + GL config
        ├ spec_builder.py  ← MjSpec-based scene builder/mutator
        ├ physics.py       ← advanced physics (raycasting, jacobians, forces)
        ├ scene_ops.py     ← live scene mutation via spec.recompile()
        ├ rendering.py     ← render RGB/depth, observations
        ├ policy_runner.py ← run_policy, eval_policy, replay
        ├ randomization.py ← domain randomization
        ├ recording.py     ← LeRobotDataset recording
        ├ tool_spec.json   ← AgentTool input schema
        └ simulation.py    ← Simulation (AgentTool orchestrator)

Usage::

    # Default (MuJoCo) via factory
    from strands_robots.simulation import create_simulation
    sim = create_simulation()

    # Direct class access
    from strands_robots.simulation import Simulation
    sim = Simulation()

    # Explicit backend
    from strands_robots.simulation.mujoco import MuJoCoSimulation

    # Shared types (no heavy deps)
    from strands_robots.simulation import SimWorld, SimRobot, SimObject

    # ABC for custom backends
    from strands_robots.simulation.base import SimEngine

Future backends::

    from strands_robots.simulation.isaac import IsaacSimulation
    from strands_robots.simulation.newton import NewtonSimulation
"""

import importlib as _importlib
from typing import TYPE_CHECKING, Any

# TYPE_CHECKING-only eager imports so mypy can resolve the heavy classes
# below to concrete types instead of the dynamic __getattr__ -> Any. PEP 562.
if TYPE_CHECKING:
    from strands_robots.simulation.isaac.config import IsaacConfig
    from strands_robots.simulation.isaac.simulation import IsaacSimulation
    from strands_robots.simulation.mujoco.simulation import Simulation
    from strands_robots.simulation.mujoco.simulation import Simulation as MuJoCoSimulation
    from strands_robots.simulation.mujoco.spec_builder import SpecBuilder

# Light imports (no heavy deps - stdlib + dataclasses only)
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    BenchmarkCompatibilityError,
    BenchmarkProtocol,
    StepInfo,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
    unregister_benchmark,
)
from strands_robots.simulation.benchmark_spec import (
    DeclarativeBenchmark,
    register_benchmark_from_file,
)
from strands_robots.simulation.factory import (
    create_simulation,
    list_backends,
    register_backend,
)
from strands_robots.simulation.model_registry import (
    list_available_models,
    list_registered_urdfs,
    register_urdf,
    resolve_model,
    resolve_urdf,
)
from strands_robots.simulation.models import (
    SimCamera,
    SimObject,
    SimRobot,
    SimStatus,
    SimWorld,
    TrajectoryStep,
)
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    make_predicate,
    register_predicate,
)

# Heavy imports (lazy - need strands SDK + mujoco)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MuJoCoSimEngine": ("strands_robots.simulation.mujoco.simulation", "MuJoCoSimEngine"),
    "Simulation": ("strands_robots.simulation.mujoco.simulation", "MuJoCoSimEngine"),
    "MuJoCoSimulation": ("strands_robots.simulation.mujoco.simulation", "MuJoCoSimEngine"),
    "SpecBuilder": ("strands_robots.simulation.mujoco.spec_builder", "SpecBuilder"),
    # Isaac Sim backend (heavy - needs NVIDIA Isaac Sim / Omniverse, installed
    # out-of-band). Kept lazy so ``import strands_robots.simulation`` never
    # triggers an omni/isaacsim import.
    "IsaacSimulation": ("strands_robots.simulation.isaac.simulation", "IsaacSimulation"),
    "IsaacConfig": ("strands_robots.simulation.isaac.config", "IsaacConfig"),
    "_configure_gl_backend": ("strands_robots.simulation.mujoco.backend", "_configure_gl_backend"),
    "_ensure_mujoco": ("strands_robots.simulation.mujoco.backend", "_ensure_mujoco"),
    "_is_headless": ("strands_robots.simulation.mujoco.backend", "_is_headless"),
}


__all__ = [
    # ABC
    "SimEngine",
    # Factory
    "create_simulation",
    "list_backends",
    "register_backend",
    # Default backend alias
    "MuJoCoSimEngine",
    "Simulation",
    "MuJoCoSimulation",
    # Isaac Sim backend (lazy - heavy NVIDIA Isaac Sim / Omniverse deps)
    "IsaacSimulation",
    "IsaacConfig",
    # Shared dataclasses
    "SimStatus",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "SimWorld",
    "TrajectoryStep",
    # MuJoCo scene builder (MjSpec-based, replaces MJCFBuilder)
    "SpecBuilder",
    # Model registry
    "register_urdf",
    "resolve_model",
    "resolve_urdf",
    "list_registered_urdfs",
    "list_available_models",
    # Benchmark protocol + registry
    "BenchmarkProtocol",
    "BenchmarkCompatibilityError",
    "StepInfo",
    "register_benchmark",
    "unregister_benchmark",
    "get_benchmark",
    "list_benchmarks",
    # Declarative DSL + predicates
    "DeclarativeBenchmark",
    "register_benchmark_from_file",
    "PREDICATE_REGISTRY",
    "make_predicate",
    "register_predicate",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'strands_robots.simulation' has no attribute {name!r}")


# NOTE: MuJoCo GL backend configuration lives in the top-level
# strands_robots/__init__.py to ensure it runs before any `import mujoco`.
# Do NOT duplicate it here - see PR #86 for the canonical location.
