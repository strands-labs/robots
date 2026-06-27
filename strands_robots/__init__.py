#!/usr/bin/env python3
"""
Strands Robotics - Universal Robot Control with Policy Abstraction

A unified Python interface for controlling diverse robot hardware through
any VLA provider with clean policy abstraction architecture.

Key features:
- Policy abstraction for any VLA provider (GR00T, ACT, SmolVLA, etc.)
- Universal robot support through LeRobot integration
- Clean separation between robot control and policy inference
- Direct policy injection for maximum flexibility
- Multi-camera support with rich configuration options
- MuJoCo simulation backend (no GPU required)

Lazy Loading:
    Heavy imports (Robot, tools, Gr00tPolicy, Simulation) are deferred until
    first access. Heavy imports are deferred so ``import strands_robots`` stays
    fast when lerobot/torch/mujoco are installed but not yet needed.

    Light-weight symbols (Policy, MockPolicy, create_policy) are available
    immediately since they don't pull in torch/lerobot.
"""

import importlib as _importlib
import warnings as _warnings
from typing import TYPE_CHECKING, Any

# TYPE_CHECKING-only eager imports so type-checkers see concrete types for
# the lazy attributes below (the runtime __getattr__ resolves them to Any
# from the static analyzer's perspective). PEP 562.
if TYPE_CHECKING:
    from strands_robots.device_connect import (
        ReachyMiniDriver,
        RobotDeviceDriver,
        SimulationDeviceDriver,
        init_device_connect,
        init_device_connect_sync,
    )
    from strands_robots.policies.groot import Gr00tPolicy
    from strands_robots.registry import (
        get_robot,
        is_discoverable,
        list_discoverable,
        list_robots,
    )
    from strands_robots.robot import Robot
    from strands_robots.simulation import (
        SimCamera,
        SimObject,
        SimRobot,
        Simulation,
        SimWorld,
        create_simulation,
        list_backends,
        register_backend,
    )
    from strands_robots.streaming_dataset import StreamingDatasetReader
    from strands_robots.teleoperator import Teleoperator
    from strands_robots.tools.gr00t_inference import gr00t_inference
    from strands_robots.tools.lerobot_calibrate import lerobot_calibrate
    from strands_robots.tools.lerobot_camera import lerobot_camera
    from strands_robots.tools.lerobot_teleoperate import lerobot_teleoperate
    from strands_robots.tools.lerobot_train import lerobot_train
    from strands_robots.tools.pose_tool import pose_tool
    from strands_robots.tools.robot_mesh import robot_mesh
    from strands_robots.tools.serial_tool import serial_tool

# ------------------------------------------------------------------
# Light-weight imports - no torch / lerobot / mujoco dependency
# ------------------------------------------------------------------
from strands_robots.policies import MockPolicy, Policy, create_policy  # noqa: F401

# ------------------------------------------------------------------
# Lazy-loaded heavy symbols
# ------------------------------------------------------------------
# Maps public name -> (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Hardware robot
    "Robot": ("strands_robots.robot", "Robot"),
    "Teleoperator": ("strands_robots.teleoperator", "Teleoperator"),
    "list_robots": ("strands_robots.registry", "list_robots"),
    "get_robot": ("strands_robots.registry", "get_robot"),
    "list_discoverable": ("strands_robots.registry", "list_discoverable"),
    "is_discoverable": ("strands_robots.registry", "is_discoverable"),
    # Policies
    "Gr00tPolicy": ("strands_robots.policies.groot", "Gr00tPolicy"),
    # Simulation (MuJoCo)
    "Simulation": ("strands_robots.simulation", "Simulation"),
    "create_simulation": ("strands_robots.simulation.factory", "create_simulation"),
    "list_backends": ("strands_robots.simulation.factory", "list_backends"),
    "register_backend": ("strands_robots.simulation.factory", "register_backend"),
    "SimWorld": ("strands_robots.simulation", "SimWorld"),
    "SimRobot": ("strands_robots.simulation", "SimRobot"),
    "SimObject": ("strands_robots.simulation", "SimObject"),
    "SimCamera": ("strands_robots.simulation", "SimCamera"),
    # Tools
    "gr00t_inference": ("strands_robots.tools.gr00t_inference", "gr00t_inference"),
    "lerobot_calibrate": ("strands_robots.tools.lerobot_calibrate", "lerobot_calibrate"),
    "lerobot_camera": ("strands_robots.tools.lerobot_camera", "lerobot_camera"),
    "lerobot_teleoperate": ("strands_robots.tools.lerobot_teleoperate", "lerobot_teleoperate"),
    "lerobot_train": ("strands_robots.tools.lerobot_train", "lerobot_train"),
    "pose_tool": ("strands_robots.tools.pose_tool", "pose_tool"),
    "serial_tool": ("strands_robots.tools.serial_tool", "serial_tool"),
    # Robot mesh coordination tool (Device Connect dispatch + mesh fallback)
    "robot_mesh": ("strands_robots.tools.robot_mesh", "robot_mesh"),
    # Device Connect integration - wraps robots as Device Connect devices
    "init_device_connect": ("strands_robots.device_connect", "init_device_connect"),
    "init_device_connect_sync": ("strands_robots.device_connect", "init_device_connect_sync"),
    "RobotDeviceDriver": ("strands_robots.device_connect", "RobotDeviceDriver"),
    "SimulationDeviceDriver": ("strands_robots.device_connect", "SimulationDeviceDriver"),
    "ReachyMiniDriver": ("strands_robots.device_connect", "ReachyMiniDriver"),
    "StreamingDatasetReader": ("strands_robots.streaming_dataset", "StreamingDatasetReader"),
}

__all__ = [
    # Always available
    "Policy",
    "MockPolicy",
    "create_policy",
    # Lazy-loaded
    "Robot",
    "Teleoperator",
    "Gr00tPolicy",
    "Simulation",
    "SimWorld",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "list_robots",
    "get_robot",
    "list_discoverable",
    "is_discoverable",
    "create_simulation",
    "list_backends",
    "register_backend",
    "gr00t_inference",
    "lerobot_camera",
    "lerobot_teleoperate",
    "lerobot_train",
    "lerobot_calibrate",
    "serial_tool",
    "pose_tool",
    "robot_mesh",
    "init_device_connect",
    "init_device_connect_sync",
    "RobotDeviceDriver",
    "SimulationDeviceDriver",
    "ReachyMiniDriver",
    "StreamingDatasetReader",
]


# Auto-configure MuJoCo GL backend for headless environments BEFORE any
# module imports mujoco at the top level.  MuJoCo locks the OpenGL backend
# at import time, so MUJOCO_GL must be set first.
#
# WHY EAGER: This MUST run at module import time, not lazily, because:
# 1. MuJoCo reads MUJOCO_GL only on first `import mujoco`
# 2. Any downstream code doing `from strands_robots.simulation import ...`
#    triggers mujoco import via the lazy-load chain
# 3. If we defer to first use, the env var would be set too late
#
# GUARD: Skip when mujoco is not installed so users without the [sim-mujoco]
# extra do not pay import-attempt cost on every `import strands_robots`.
# This is the canonical location - strands_robots/simulation/__init__.py
# intentionally does NOT duplicate this call.
import importlib.util as _importlib_util  # noqa: E402

if _importlib_util.find_spec("mujoco") is not None:
    try:
        from strands_robots.simulation.mujoco.backend import _configure_gl_backend

        _configure_gl_backend()
    except (ImportError, AttributeError, OSError):
        pass


# Auto-configure the macOS dyld search path so torchcodec can find Homebrew's
# ffmpeg with zero user setup - making ``sim.stream_dataset(...)`` video decode
# work out of the box. No-op off macOS, without torchcodec, or when already set.
# May re-exec the interpreter ONCE on a plain script run (guarded; never in
# Jupyter/REPL/pytest). Opt out with STRANDS_ROBOTS_NO_DYLD_SHIM=1. See _dyld.py.
try:
    from strands_robots._dyld import ensure_ffmpeg_on_dyld_path

    ensure_ffmpeg_on_dyld_path()
except Exception:  # noqa: BLE001 - never let the shim block import
    pass


def __getattr__(name: str) -> Any:  # noqa: N807
    """Lazy-load heavy modules on first attribute access.

    This avoids importing torch, lerobot, numpy, mujoco, pyserial, etc. at
    ``import strands_robots`` time.  The first access to e.g.
    ``strands_robots.Robot`` or ``strands_robots.Simulation`` triggers the
    real import.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            module = _importlib.import_module(module_path)
            value = getattr(module, attr_name)
            # Cache in module dict so __getattr__ is not called again
            globals()[name] = value
            return value
        except ImportError as exc:
            _warnings.warn(
                f"{name} not available (missing dependencies): {exc}",
                stacklevel=2,
            )
            raise AttributeError(name) from exc
    raise AttributeError(f"module 'strands_robots' has no attribute {name!r}")
