"""Isaac Sim / Isaac Lab Integration for strands-robots.

Bridges NVIDIA Isaac Sim's GPU-accelerated simulation and Isaac Lab's
RL framework into strands-robots' Policy ABC and unified Robot() factory.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  strands-robots                                              │
    │  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
    │  │ Robot()     │  │ Policy ABC   │  │ Trainer ABC       │   │
    │  │  mode=sim   │  │  .get_actions │  │  .train()         │   │
    │  └─────┬──────┘  └──────┬───────┘  └────────┬──────────┘   │
    │        │                │                    │               │
    │  ┌─────▼──────┐  ┌─────▼────────┐  ┌───────▼──────────┐   │
    │  │ MuJoCo         │  │ 17 providers  │  │ groot, lerobot,  │   │
    │  │ Simulator      │  │ (VLA models)  │  │ dreamgen trainers│   │
    │  └────────────┘  └──────────────┘  └──────────────────┘   │
    │         +                                    +              │
    │  ┌─────────────────┐             ┌──────────────────────┐  │
    │  │ IsaacSimBackend  │             │ IsaacLabTrainer       │  │
    │  │ (GPU-accelerated │             │ (RSL-RL, SB3, SKRL)  │  │
    │  │  physics + render│             │                       │  │
    │  └────────┬────────┘             └──────────────────────┘  │
    │           │ SimulationApp=None?                             │
    │  ┌────────▼────────┐                                       │
    │  │ IsaacSimBridge   │  ← ZMQ subprocess bridge             │
    │  │  Client ↔ Server │  (python.sh ↔ pip Python)            │
    │  └─────────────────┘                                       │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │ IsaacLabEnv → Gymnasium wrapper → Policy evaluation  │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Modules:
    - isaac_sim_backend: GPU-accelerated sim backend (alternative to MuJoCo)
    - isaac_sim_bridge: ZMQ subprocess bridge for cross-runtime communication
    - isaac_lab_env: Wraps Isaac Lab envs for strands-robots Policy eval
    - isaac_lab_trainer: RL training via Isaac Lab (RSL-RL, SB3, SKRL, RL-Games)
    - asset_converter: MJCF ↔ USD conversion for cross-backend compatibility

Requirements:
    - NVIDIA GPU with CUDA 12+
    - Isaac Sim 5.1+ (installed via pip or Omniverse Launcher)
    - Isaac Lab (pip install isaaclab)
"""

import logging as _logging
import os
from pathlib import Path as _Path
from typing import Optional as _Optional

_logger = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Isaac Sim discovery utilities
# ---------------------------------------------------------------------------

_ISAAC_SIM_SEARCH_PATHS = [
    os.environ.get("ISAAC_SIM_PATH", ""),
    os.path.expanduser("~/IsaacSim"),
    "/home/ubuntu/IsaacSim",
    "/isaac-sim",
    os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-5.1.0"),
    os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.5.0"),
]


def get_isaac_sim_path() -> _Optional[str]:
    """Return the path to the Isaac Sim installation, or None if not found.

    Checks (in order):
        1. ``ISAAC_SIM_PATH`` environment variable
        2. ``~/IsaacSim``
        3. ``/home/ubuntu/IsaacSim``
        4. ``/isaac-sim``
        5. Common Omniverse Launcher install locations

    Returns:
        Absolute path string, or ``None``.
    """
    for p in _ISAAC_SIM_SEARCH_PATHS:
        if p and os.path.isdir(p):
            version_file = os.path.join(p, "VERSION")
            shell_script = os.path.join(p, "isaac-sim.sh")
            if os.path.isfile(version_file) or os.path.isfile(shell_script):
                return os.path.abspath(p)
    return None


def is_isaac_sim_available() -> bool:
    """Return ``True`` if an Isaac Sim installation is detected on this host."""
    return get_isaac_sim_path() is not None


__all__ = [
    "IsaacSimBackend",
    "IsaacSimBridgeClient",
    "IsaacSimBridgeServer",
    "IsaacGymEnv",
    "IsaacLabEnv",
    "IsaacLabTrainer",
    "IsaacLabTrainerConfig",
    "AssetConverter",
    "create_isaac_env",
    "list_isaac_tasks",
    "convert_mjcf_to_usd",
    "convert_usd_to_mjcf",
    "convert_all_robots_to_usd",
    "get_isaac_sim_path",
    "is_isaac_sim_available",
]


def __getattr__(name):
    """Lazy imports to avoid hard dependency on Isaac Sim."""
    if name == "IsaacSimBackend":
        from .isaac_sim_backend import IsaacSimBackend

        return IsaacSimBackend
    elif name in ("IsaacSimBridgeClient", "IsaacSimBridgeServer"):
        from .isaac_sim_bridge import IsaacSimBridgeClient, IsaacSimBridgeServer

        return (
            IsaacSimBridgeClient
            if name == "IsaacSimBridgeClient"
            else IsaacSimBridgeServer
        )
    elif name == "IsaacGymEnv":
        from .isaac_gym_env import IsaacGymEnv

        return IsaacGymEnv
    elif name == "IsaacLabEnv":
        from .isaac_lab_env import IsaacLabEnv

        return IsaacLabEnv
    elif name in ("IsaacLabTrainer", "IsaacLabTrainerConfig"):
        from .isaac_lab_trainer import IsaacLabTrainer, IsaacLabTrainerConfig

        return IsaacLabTrainer if name == "IsaacLabTrainer" else IsaacLabTrainerConfig
    elif name == "AssetConverter":
        from .asset_converter import AssetConverter

        return AssetConverter
    elif name == "create_isaac_env":
        from .isaac_lab_env import create_isaac_env

        return create_isaac_env
    elif name == "list_isaac_tasks":
        from .isaac_lab_env import list_isaac_tasks

        return list_isaac_tasks
    elif name in (
        "convert_mjcf_to_usd",
        "convert_usd_to_mjcf",
        "convert_all_robots_to_usd",
    ):
        from .asset_converter import (
            convert_all_robots_to_usd,
            convert_mjcf_to_usd,
            convert_usd_to_mjcf,
        )

        if name == "convert_mjcf_to_usd":
            return convert_mjcf_to_usd
        elif name == "convert_all_robots_to_usd":
            return convert_all_robots_to_usd
        return convert_usd_to_mjcf
    raise AttributeError(f"module 'strands_robots.isaac' has no attribute {name!r}")
