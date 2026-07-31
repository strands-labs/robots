"""strands_robots.simulation.isaac -- GPU-native Isaac Sim simulation backend.

This subpackage provides :class:`IsaacSimulation`, a ``SimEngine`` backend
built on **NVIDIA Isaac Sim / Omniverse** for photorealistic rendering,
synthetic data generation, and GPU-batched sensor simulation.

Usage::

    from strands_robots.simulation.isaac import IsaacSimulation, IsaacConfig
    config = IsaacConfig(num_envs=1, headless=True)
    sim = IsaacSimulation(config)
    ok, msg = IsaacSimulation.is_available()

Requires NVIDIA Isaac Sim 6.0+ (Python 3.12). Install via the pip wheels
(``isaacsim[all,extscache]`` from pypi.nvidia.com; see the caveats in
``docs/simulation/isaac.md``), Omniverse Launcher, Isaac Lab, or the NGC
docker image. The exact supported image tag and install commands live in
:mod:`strands_robots.simulation.isaac._install` -- update there, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import under TYPE_CHECKING only: keeps the runtime export lazy (no
    # omni/Isaac import cost) while statically defining the names promised by
    # ``__all__`` for type checkers and static analysis.
    from strands_robots.simulation.isaac.config import IsaacConfig
    from strands_robots.simulation.isaac.delta_eef import IsaacDeltaEEFController
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

__all__ = ["IsaacSimulation", "IsaacConfig", "IsaacDeltaEEFController"]


def _lazy_isaac_simulation() -> type[IsaacSimulation]:
    """Lazy import to avoid pulling omni/Isaac at module-import time."""
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    return IsaacSimulation


def _lazy_isaac_config() -> type[IsaacConfig]:
    """Lazy import to avoid pulling dataclass internals at import time."""
    from strands_robots.simulation.isaac.config import IsaacConfig

    return IsaacConfig


def _lazy_delta_eef_controller() -> type[IsaacDeltaEEFController]:
    """Lazy import for the delta-EEF controller (numpy-only, no Isaac dep)."""
    from strands_robots.simulation.isaac.delta_eef import IsaacDeltaEEFController

    return IsaacDeltaEEFController


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access."""
    if name == "IsaacSimulation":
        return _lazy_isaac_simulation()
    if name == "IsaacConfig":
        return _lazy_isaac_config()
    if name == "IsaacDeltaEEFController":
        return _lazy_delta_eef_controller()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
