"""Programmatic Simulation Environment for Strands Robots

An AgentTool that gives AI agents full control over MuJoCo simulation environments.
Agents can create worlds, add robots (via URDF), add objects, run policies, observe,
and adjust — all through natural language mapped to tool actions.

Mirrors robot.py's architecture but targets simulated robots instead of real hardware.
Shares the same Policy ABC so policies trained in sim transfer to real and vice versa.

Features:
- Programmatic world generation (agent builds the scene on-the-fly)
- N robots in one scene, each with its own policy provider
- URDF/MJCF loading for any robot embodiment
- URDF registry: data_config name → URDF path auto-resolution
- Multi-robot MJCF composition via xml_include merging
- Object spawning (primitives + mesh files)
- Per-robot policy execution using the same Policy ABC as robot.py
- Camera rendering (RGB + depth) for observation
- State introspection (joint positions, velocities, contacts)
- Real-time adjustment (move objects, change properties, reset)
- Domain randomization (textures, lighting, physics, object placement)
- Trajectory recording in LeRobot-compatible format
- Headless or visualized execution (open_viewer / close_viewer)

Design Philosophy:
- Same Policy ABC as robot.py — zero code changes to switch sim ↔ real
- Data configs from data_config.py map to URDF joint names
- Agent is the scene designer — no hardcoded environments
- Everything is an action the agent can call

Example (via Strands Agent):
    "Create a simulation with a SO-100 arm on a table. Add a red cube at (0.3, 0, 0.05).
     Run the groot policy to pick up the cube."

    → sim(action="create_world")
    → sim(action="add_robot", urdf_path="so100.urdf", name="arm1", position=[0,0,0.5])
    → sim(action="add_object", shape="box", name="red_cube", size=[0.05,0.05,0.05],
           position=[0.3,0,0.55], color=[1,0,0,1])
    → sim(action="run_policy", robot_name="arm1", policy_provider="groot",
           instruction="pick up the red cube", policy_port=5555)

Multi-robot example:
    → sim(action="create_world")
    → sim(action="add_robot", data_config="so100", name="left_arm", position=[-0.3, 0, 0])
    → sim(action="add_robot", data_config="so100", name="right_arm", position=[0.3, 0, 0])
    → sim(action="add_object", name="block", shape="box", position=[0, 0.2, 0.05])
    → sim(action="run_policy", robot_name="left_arm", policy_provider="groot",
           instruction="hand the block to the right arm", policy_port=5555)

Domain randomization:
    → sim(action="randomize", randomize_colors=true, randomize_lighting=true,
           randomize_physics=true, randomize_positions=true, position_noise=0.02)
"""

import io
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import numpy as np
from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from strands_robots._async_utils import _resolve_coroutine

# LeRobotDataset recording (optional — falls back to JSON if not installed)
try:
    from .dataset_recorder import DatasetRecorder, has_lerobot_dataset
except ImportError:
    def has_lerobot_dataset():
        return False

    DatasetRecorder = None

logger = logging.getLogger(__name__)

__all__ = [
    "Simulation",
    "SimStatus",
    "SimRobot",
    "SimObject",
    "SimCamera",
    "SimWorld",
    "MJCFBuilder",
    "register_urdf",
    "resolve_model",
    "resolve_urdf",
    "list_registered_urdfs",
    "list_available_models",
    "simulation",
]

# Lazy import mujoco — only needed when simulation is actually used
_mujoco = None
_mujoco_viewer = None


def _is_headless() -> bool:
    """Detect if running in a headless environment (no display server).

    Returns True on Linux when no DISPLAY or WAYLAND_DISPLAY is set,
    which means GLFW-based rendering will fail.
    """
    if sys.platform != "linux":
        return False  # macOS has CGL, Windows has WGL — always available
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


def _configure_gl_backend() -> None:
    """Auto-configure MuJoCo's OpenGL backend for headless environments.

    MuJoCo reads MUJOCO_GL at import time to select the OpenGL backend:
    - "egl"    → EGL (GPU-accelerated offscreen, requires libEGL + NVIDIA driver)
    - "osmesa" → OSMesa (CPU software rendering, slower but always works)
    - "glfw"   → GLFW (default, requires X11/Wayland display server)

    This function MUST be called before `import mujoco`. Setting MUJOCO_GL
    after import has no effect — the backend is locked at import time.

    Never overrides a user-set MUJOCO_GL value.
    """
    if os.environ.get("MUJOCO_GL"):
        logger.debug(
            f"MUJOCO_GL already set to '{os.environ['MUJOCO_GL']}', respecting user config"
        )
        return

    if not _is_headless():
        return  # Display available, GLFW will work fine

    # Headless Linux — probe for EGL first (GPU-accelerated), then fall back to OSMesa (CPU)
    import ctypes

    # Try EGL (fast, GPU-accelerated)
    try:
        ctypes.cdll.LoadLibrary("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
        logger.info(
            "Headless environment detected — using MUJOCO_GL=egl (GPU-accelerated offscreen)"
        )
        return
    except OSError:
        pass

    # Try OSMesa (CPU software rendering)
    try:
        ctypes.cdll.LoadLibrary("libOSMesa.so")
        os.environ["MUJOCO_GL"] = "osmesa"
        logger.info(
            "Headless environment detected — using MUJOCO_GL=osmesa (CPU software rendering)"
        )
        return
    except OSError:
        pass

    logger.warning(
        "Headless environment detected but neither EGL nor OSMesa found. "
        "MuJoCo rendering will likely fail. Install one of:\n"
        "  GPU: apt-get install libegl1-mesa-dev  (or NVIDIA driver provides libEGL)\n"
        "  CPU: apt-get install libosmesa6-dev\n"
        "Then set: export MUJOCO_GL=egl  (or osmesa)"
    )


def _ensure_mujoco():
    """Lazy import MuJoCo to avoid hard dependency.

    Auto-configures the OpenGL backend for headless environments before
    importing mujoco, since MUJOCO_GL must be set at import time.
    """
    global _mujoco, _mujoco_viewer
    if _mujoco is None:
        # CRITICAL: Configure GL backend BEFORE importing mujoco.
        # MuJoCo reads MUJOCO_GL at import time and locks the backend.
        _configure_gl_backend()
        try:
            import mujoco

            _mujoco = mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo is required for simulation. Install with:\n"
                "  pip install strands-robots[sim]\n"
                "Or: pip install mujoco"
            )
    if _mujoco_viewer is None:
        try:
            import mujoco.viewer as viewer

            _mujoco_viewer = viewer
        except ImportError:
            pass  # Viewer is optional — headless mode works fine
    return _mujoco


# ===================================================================
# URDF Registry — Maps data_config names to URDF paths
# ===================================================================

# Default URDF search paths (checked in order)
_URDF_SEARCH_PATHS = [
    Path.cwd() / "urdfs",
    Path.cwd() / "assets" / "urdfs",
    Path.cwd() / "robots",
    Path.home() / ".strands_robots" / "urdfs",
    Path("/opt/strands_robots/urdfs"),
]

# ─────────────────────────────────────────────────────────────────────
# Robot Model Resolution (MJCF + URDF) — delegates to unified registry
# ─────────────────────────────────────────────────────────────────────

try:
    from strands_robots.assets import (  # noqa: I001
        format_robot_table as _format_robot_table,
        list_available_robots as _list_menagerie_robots,
        resolve_model_path as _resolve_menagerie_model,
    )
    _HAS_ASSET_MANAGER = True
except ImportError:
    _HAS_ASSET_MANAGER = False

# Legacy URDF registry — now just a runtime cache for user-registered
# URDFs via register_urdf().  Builtin legacy_urdf paths are in robots.json.
_URDF_REGISTRY: Dict[str, str] = {}

# Allow overrides from environment
_URDF_DIR_OVERRIDE = os.getenv("STRANDS_URDF_DIR")
if _URDF_DIR_OVERRIDE:
    _URDF_SEARCH_PATHS.insert(0, Path(_URDF_DIR_OVERRIDE))


def register_urdf(data_config: str, urdf_path: str):
    """Register a URDF/MJCF file for a data_config name.

    Args:
        data_config: Data config name (e.g., "so100", "my_custom_robot")
        urdf_path: Absolute path or relative filename to the URDF/MJCF file
    """
    _URDF_REGISTRY[data_config] = urdf_path
    logger.info("📋 Registered model for '%s': %s", data_config, urdf_path)


def resolve_model(name: str, prefer_scene: bool = True) -> Optional[str]:
    """Resolve a robot name or data_config to an MJCF/URDF model path.

    Resolution order:
    1. Asset manager (32 bundled robots + 40 aliases)
    2. Legacy URDF registry (custom registrations)
    3. URDF search paths (STRANDS_URDF_DIR, ./urdfs, etc.)

    Args:
        name: Robot name, data_config, or alias (e.g. "so100", "panda", "unitree_g1")
        prefer_scene: If True, prefer scene.xml (includes ground plane, lighting, cameras)
                      over bare model.xml. Default True for simulation use.

    Returns:
        Absolute path to model file, or None if not found
    """
    # 1. Try asset manager (Menagerie MJCF models — preferred)
    if _HAS_ASSET_MANAGER:
        # Prefer scene.xml which includes ground plane, lighting, cameras
        path = _resolve_menagerie_model(name, prefer_scene=prefer_scene)
        if path and path.exists():
            return str(path)
        # Fallback: try without scene preference
        if prefer_scene:
            path = _resolve_menagerie_model(name, prefer_scene=False)
            if path and path.exists():
                return str(path)

    # 2. Try legacy URDF registry
    return resolve_urdf(name)


def resolve_urdf(data_config: str) -> Optional[str]:
    """Resolve a data_config name to a URDF file path (legacy).

    Checks runtime-registered URDFs first, then falls back to the
    ``legacy_urdf`` field in ``registry/robots.json``.

    Args:
        data_config: Data config name

    Returns:
        Absolute path to URDF file, or None if not found
    """
    # 1. Runtime-registered URDFs (via register_urdf())
    if data_config in _URDF_REGISTRY:
        urdf_rel = _URDF_REGISTRY[data_config]
        if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
            return urdf_rel
        for search_dir in _URDF_SEARCH_PATHS:
            candidate = search_dir / urdf_rel
            if candidate.exists():
                return str(candidate)

    # 2. Check registry robots.json for legacy_urdf
    try:
        from strands_robots.registry import get_robot, resolve_name
        canonical = resolve_name(data_config)
        info = get_robot(canonical)
        if info and "legacy_urdf" in info:
            urdf_rel = info["legacy_urdf"]
            if os.path.isabs(urdf_rel) and os.path.exists(urdf_rel):
                return urdf_rel
            for search_dir in _URDF_SEARCH_PATHS:
                candidate = search_dir / urdf_rel
                if candidate.exists():
                    return str(candidate)
    except ImportError:
        pass

    logger.debug("URDF not found for '%s' in search paths", data_config)
    return None


def list_registered_urdfs() -> Dict[str, Optional[str]]:
    """List all registered URDF mappings and their resolved paths."""
    result = {}
    for config_name in _URDF_REGISTRY:
        result[config_name] = resolve_urdf(config_name)
    return result


def list_available_models() -> str:
    """List all available robot models (Menagerie + custom).

    Returns:
        Formatted table string of all robots
    """
    if _HAS_ASSET_MANAGER:
        return _format_robot_table()

    # Fallback: legacy registry only
    lines = ["Registered URDFs:"]
    for name, path in _URDF_REGISTRY.items():
        resolved = resolve_urdf(name)
        status = "✅" if resolved else "❌"
        lines.append(f"  {status} {name}: {path}")
    return "\n".join(lines)


# ===================================================================
# Simulation World State
# ===================================================================


class SimStatus(Enum):
    """Simulation execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimRobot:
    """A robot instance within the simulation."""

    name: str
    urdf_path: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0]
    )  # wxyz quat
    data_config: Optional[str] = None
    body_id: int = -1
    joint_ids: List[int] = field(default_factory=list)
    joint_names: List[str] = field(default_factory=list)
    actuator_ids: List[int] = field(default_factory=list)
    # Namespace prefix for multi-robot scenes (avoids name collisions)
    namespace: str = ""
    # Policy execution state
    policy_running: bool = False
    policy_steps: int = 0
    policy_instruction: str = ""


@dataclass
class SimObject:
    """An object in the simulation scene."""

    name: str
    shape: str  # "box", "sphere", "cylinder", "capsule", "mesh"
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    size: List[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])  # RGBA
    mass: float = 0.1
    mesh_path: Optional[str] = None
    body_id: int = -1
    is_static: bool = False
    # Original values for domain randomization reset
    _original_position: List[float] = field(default_factory=list)
    _original_color: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._original_position = list(self.position)
        self._original_color = list(self.color)


@dataclass
class SimCamera:
    """A camera in the simulation."""

    name: str
    position: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    target: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 60.0
    width: int = 640
    height: int = 480
    camera_id: int = -1


@dataclass
class TrajectoryStep:
    """A single step in a recorded trajectory."""

    timestamp: float
    sim_time: float
    robot_name: str
    observation: Dict[str, Any]
    action: Dict[str, Any]
    instruction: str = ""


@dataclass
class SimWorld:
    """Complete simulation world state."""

    robots: Dict[str, SimRobot] = field(default_factory=dict)
    objects: Dict[str, SimObject] = field(default_factory=dict)
    cameras: Dict[str, SimCamera] = field(default_factory=dict)
    timestep: float = 0.002  # 500Hz physics
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    ground_plane: bool = True
    status: SimStatus = SimStatus.IDLE
    sim_time: float = 0.0
    step_count: int = 0
    # MuJoCo internals (set after world is built)
    _xml: str = ""
    _model: Any = None
    _data: Any = None
    _robot_base_xml: str = ""  # Path to robot scene (for object injection)
    # Trajectory recording
    _recording: bool = False
    _trajectory: List[TrajectoryStep] = field(default_factory=list)
    # LeRobotDataset recorder (training-ready format)
    _dataset_recorder: Any = None
    # Temp directory for scene composition (cleaned up on __del__)
    _tmpdir: Any = None


# ===================================================================
# MJCF XML Builder — Programmatic scene construction
# ===================================================================


class MJCFBuilder:
    """Builds MuJoCo MJCF XML from SimWorld state.

    The agent builds up the world through actions, then we compile
    the full XML and load it into MuJoCo. When the scene changes
    (add/remove objects), we recompile.

    For multi-robot scenes, we use MuJoCo's XML composition:
    each robot's URDF is loaded separately, converted to MJCF XML,
    then merged into a single scene with namespaced bodies/joints.
    """

    @staticmethod
    def build_objects_only(world: SimWorld) -> str:
        """Build MJCF XML for a world with only objects (robots loaded separately).

        Used for object-only recompilation when robots are already loaded.
        """
        _ensure_mujoco()

        parts = []
        parts.append('<mujoco model="strands_sim">')
        parts.append('  <compiler angle="radian" autolimits="true"/>')

        gx, gy, gz = world.gravity
        parts.append(
            f'  <option timestep="{world.timestep}" gravity="{gx} {gy} {gz}"/>'
        )

        parts.append("  <visual>")
        parts.append('    <global offwidth="1280" offheight="960"/>')
        parts.append('    <quality shadowsize="4096"/>')
        parts.append("  </visual>")

        # Assets
        parts.append("  <asset>")
        parts.append(
            '    <texture type="2d" name="grid_tex" builtin="checker" '
            'width="512" height="512" rgb1=".9 .9 .9" rgb2=".7 .7 .7"/>'
        )
        parts.append(
            '    <material name="grid_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.1"/>'
        )
        for obj in world.objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                parts.append(
                    f'    <mesh name="mesh_{obj.name}" file="{obj.mesh_path}"/>'
                )
        parts.append("  </asset>")

        # Worldbody
        parts.append("  <worldbody>")
        parts.append(
            '    <light name="main_light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>'
        )
        parts.append(
            '    <light name="fill_light" pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>'
        )

        if world.ground_plane:
            parts.append(
                '    <geom name="ground" type="plane" size="5 5 0.01" '
                'material="grid_mat" conaffinity="1" condim="3"/>'
            )

        # Cameras
        for cam in world.cameras.values():
            px, py, pz = cam.position
            parts.append(
                f'    <camera name="{cam.name}" pos="{px} {py} {pz}" '
                f'fovy="{cam.fov}" mode="fixed"/>'
            )

        # Objects
        for obj in world.objects.values():
            parts.append(MJCFBuilder._object_xml(obj, indent=4))

        parts.append("  </worldbody>")
        parts.append("</mujoco>")

        return "\n".join(parts)

    @staticmethod
    def _object_xml(obj: SimObject, indent: int = 4) -> str:
        """Generate MJCF XML for a single object."""
        pad = " " * indent
        px, py, pz = obj.position
        qw, qx, qy, qz = obj.orientation
        r, g, b, a = obj.color
        lines = []

        lines.append(
            f'{pad}<body name="{obj.name}" pos="{px} {py} {pz}" '
            f'quat="{qw} {qx} {qy} {qz}">'
        )

        if not obj.is_static:
            lines.append(f'{pad}  <freejoint name="{obj.name}_joint"/>')
            lines.append(
                f'{pad}  <inertial pos="0 0 0" mass="{obj.mass}" '
                f'diaginertia="0.001 0.001 0.001"/>'
            )

        if obj.shape == "box":
            sx, sy, sz = [s / 2 for s in obj.size]
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="box" size="{sx} {sy} {sz}" '
                f'rgba="{r} {g} {b} {a}" condim="3" friction="1 0.5 0.001"/>'
            )
        elif obj.shape == "sphere":
            radius = obj.size[0] / 2 if obj.size else 0.025
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="sphere" size="{radius}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "cylinder":
            radius = obj.size[0] / 2 if obj.size else 0.025
            half_h = obj.size[2] / 2 if len(obj.size) > 2 else 0.05
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="cylinder" size="{radius} {half_h}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "capsule":
            radius = obj.size[0] / 2 if obj.size else 0.025
            half_h = obj.size[2] / 2 if len(obj.size) > 2 else 0.05
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="capsule" size="{radius} {half_h}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "mesh" and obj.mesh_path:
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="mesh" mesh="mesh_{obj.name}" '
                f'rgba="{r} {g} {b} {a}" condim="3"/>'
            )
        elif obj.shape == "plane":
            sx = obj.size[0] if obj.size else 1.0
            sy = obj.size[1] if len(obj.size) > 1 else sx
            lines.append(
                f'{pad}  <geom name="{obj.name}_geom" type="plane" size="{sx} {sy} 0.01" '
                f'rgba="{r} {g} {b} {a}"/>'
            )

        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    @staticmethod
    def compose_multi_robot_scene(
        robots: Dict[str, SimRobot],
        objects: Dict[str, SimObject],
        cameras: Dict[str, SimCamera],
        world: SimWorld,
    ) -> str:
        """Compose a multi-robot scene by merging URDF-derived MJCF fragments.

        Strategy:
        1. Load each robot URDF → get its MJCF via mujoco.MjModel
        2. Save each as temporary MJCF XML
        3. Build a master MJCF that <include>s each robot with position offset
        4. Add objects, cameras, ground, lighting

        Each robot body is wrapped in a site at its configured position.
        Joint/actuator names are namespaced to avoid collisions.
        """
        mj = _ensure_mujoco()
        world._tmpdir = tempfile.TemporaryDirectory(prefix="strands_sim_")
        tmpdir = world._tmpdir.name

        # Convert each robot URDF to MJCF and save
        robot_xmls = {}
        for robot_name, robot in robots.items():
            try:
                # Load the URDF
                model = mj.MjModel.from_xml_path(str(robot.urdf_path))
                # Save as MJCF
                robot_xml_path = os.path.join(tmpdir, f"{robot_name}.xml")
                mj.mj_saveLastXML(robot_xml_path, model)
                robot_xmls[robot_name] = robot_xml_path
                logger.debug("Converted %s → %s", robot.urdf_path, robot_xml_path)
            except Exception as e:
                logger.error("Failed to convert URDF for '%s': %s", robot_name, e)
                raise

        # Build master scene XML
        parts = []
        parts.append('<mujoco model="strands_sim_multi">')
        parts.append('  <compiler angle="radian" autolimits="true" meshdir="."/>')

        gx, gy, gz = world.gravity
        parts.append(
            f'  <option timestep="{world.timestep}" gravity="{gx} {gy} {gz}"/>'
        )

        parts.append("  <visual>")
        parts.append('    <global offwidth="1280" offheight="960"/>')
        parts.append('    <quality shadowsize="4096"/>')
        parts.append("  </visual>")

        # Assets
        parts.append("  <asset>")
        parts.append(
            '    <texture type="2d" name="grid_tex" builtin="checker" '
            'width="512" height="512" rgb1=".9 .9 .9" rgb2=".7 .7 .7"/>'
        )
        parts.append(
            '    <material name="grid_mat" texture="grid_tex" texrepeat="8 8" reflectance="0.1"/>'
        )
        for obj in objects.values():
            if obj.shape == "mesh" and obj.mesh_path:
                parts.append(
                    f'    <mesh name="mesh_{obj.name}" file="{obj.mesh_path}"/>'
                )
        parts.append("  </asset>")

        parts.append("  <worldbody>")
        parts.append(
            '    <light name="main_light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>'
        )
        parts.append(
            '    <light name="fill_light" pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>'
        )

        if world.ground_plane:
            parts.append(
                '    <geom name="ground" type="plane" size="5 5 0.01" '
                'material="grid_mat" conaffinity="1" condim="3"/>'
            )

        # Cameras
        for cam in cameras.values():
            px, py, pz = cam.position
            parts.append(
                f'    <camera name="{cam.name}" pos="{px} {py} {pz}" fovy="{cam.fov}" mode="fixed"/>'
            )

        # Robot includes — each wrapped in a body at configured position
        for robot_name, robot in robots.items():
            px, py, pz = robot.position
            qw, qx, qy, qz = robot.orientation
            xml_path = robot_xmls[robot_name]
            parts.append(f"    <!-- Robot: {robot_name} -->")
            parts.append(f'    <include file="{xml_path}"/>')

        # Objects
        for obj in objects.values():
            parts.append(MJCFBuilder._object_xml(obj, indent=4))

        parts.append("  </worldbody>")
        parts.append("</mujoco>")

        master_xml = "\n".join(parts)

        # Save master and load
        master_path = os.path.join(tmpdir, "master_scene.xml")
        with open(master_path, "w") as f:
            f.write(master_xml)

        return master_path


# ===================================================================
# Simulation Tool — The AgentTool interface
# ===================================================================


class Simulation(AgentTool):
    """Programmatic simulation environment as a Strands AgentTool.

    Gives AI agents the ability to create, modify, and control MuJoCo
    simulation environments through natural language → tool actions.

    Actions:
        World Management:
        - create_world: Create a new empty simulation world
        - load_scene: Load a pre-built MJCF/URDF scene file
        - reset: Reset simulation to initial state
        - get_state: Get full simulation state
        - destroy: Destroy the simulation world

        Robot Management:
        - add_robot: Add a robot from URDF/MJCF file or data_config name
        - remove_robot: Remove a robot from the scene
        - list_robots: List all robots and their states
        - get_robot_state: Get joint positions/velocities for a robot

        Object Management:
        - add_object: Add a primitive or mesh object
        - remove_object: Remove an object
        - move_object: Move/reposition an object
        - list_objects: List all objects in the scene

        Camera Management:
        - add_camera: Add a camera to the scene
        - remove_camera: Remove a camera

        Policy Execution:
        - run_policy: Run a policy on a robot (blocking)
        - start_policy: Start async policy execution on a robot
        - stop_policy: Stop a running policy on a robot

        Observation:
        - render: Render camera view (returns base64 image)
        - render_depth: Render depth map
        - get_contacts: Get contact/collision information

        Simulation Control:
        - step: Advance simulation by N steps
        - set_gravity: Change gravity
        - set_timestep: Change physics timestep

        Domain Randomization:
        - randomize: Apply domain randomization to the scene

        Trajectory Recording:
        - start_recording: Start recording trajectory data
        - stop_recording: Stop and export trajectory
        - get_recording_status: Check recording state

        Visualization:
        - open_viewer: Open interactive 3D viewer window
        - close_viewer: Close the viewer

        URDF Registry:
        - list_urdfs: List registered URDF mappings
        - register_urdf: Register a new URDF for a data_config
    """

    def __init__(
        self,
        tool_name: str = "sim",
        default_timestep: float = 0.002,
        default_width: int = 640,
        default_height: int = 480,
        mesh: bool = True,
        peer_id: str = None,
        **kwargs,
    ):
        super().__init__()
        self.tool_name_str = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        # World state
        self._world: Optional[SimWorld] = None
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix=f"{tool_name}_sim"
        )
        self._policy_threads: Dict[str, Future] = {}
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Viewer handle
        self._viewer_handle = None
        self._viewer_thread = None

        # Cache renderers per (width, height) to avoid GPU context churn
        self._renderers: Dict[tuple, Any] = {}
        self._renderer_model = None  # Track which model the renderers are for

        logger.info("🎮 Simulation tool '%s' initialized", tool_name)

        # Zenoh mesh — every Simulation is a peer by default
        try:
            from strands_robots.zenoh_mesh import init_mesh

            self.mesh = init_mesh(self, peer_id=peer_id, peer_type="sim", mesh=mesh)
        except Exception as e:
            logger.debug("Mesh init skipped: %s", e)
            self.mesh = None

    # -------------------------------------------------------------------
    # Public Properties — Direct MuJoCo model/data access
    # -------------------------------------------------------------------

    @property
    def mj_model(self):
        """Direct access to the MuJoCo model (mujoco.MjModel).

        Enables advanced use cases like Kinematics, custom rendering,
        and direct MuJoCo API calls that require model/data objects.

        Returns:
            mujoco.MjModel or None if no world created

        Example:
            sim = Simulation()
            sim.create_world()
            sim.add_robot(name="arm", data_config="so100")
            kin = MuJoCoKinematics(sim.mj_model, sim.mj_data, body_name="gripper")
        """
        return self._world._model if self._world else None

    @property
    def mj_data(self):
        """Direct access to the MuJoCo data (mujoco.MjData).

        Returns:
            mujoco.MjData or None if no world created
        """
        return self._world._data if self._world else None

    # -------------------------------------------------------------------
    # Robot-compatible interface — get_observation / send_action
    # -------------------------------------------------------------------

    def get_observation(
        self, robot_name: str = None, camera_name: str = None
    ) -> Dict[str, Any]:
        """Get observation from simulation (Robot ABC compatible).

        This mirrors the real robot's get_observation() interface, making
        Simulation a drop-in replacement for Robot in control loops.

        Args:
            robot_name: Robot to observe. If None, uses the first robot.
            camera_name: Specific camera. If None, renders all scene cameras.

        Returns:
            Dict with joint positions (float values) and camera images (numpy arrays).
            Keys match robot joint names + camera names.

        Example:
            sim = Simulation()
            sim.create_world()
            sim.add_robot(name="arm", data_config="so100")
            obs = sim.get_observation("arm")
            # obs = {"shoulder_pan": 0.0, "shoulder_lift": -1.57, ..., "cam_top": np.array(...)}
        """
        if self._world is None or self._world._model is None:
            return {}

        # Resolve robot name
        if robot_name is None:
            if not self._world.robots:
                return {}
            robot_name = next(iter(self._world.robots))

        if robot_name not in self._world.robots:
            return {}

        return self._get_sim_observation(robot_name, cam_name=camera_name)

    def send_action(
        self, action: Dict[str, Any], robot_name: str = None, n_substeps: int = 1
    ) -> None:
        """Apply action to simulation (Robot ABC compatible).

        This mirrors the real robot's send_action() interface, making
        Simulation a drop-in replacement for Robot in control loops.

        Args:
            action: Dict mapping joint/actuator names to target values.
                    Example: {"shoulder_pan": 0.5, "elbow": -1.2}
            robot_name: Robot to act on. If None, uses the first robot.
            n_substeps: Number of physics substeps per action (for stability).

        Example:
            sim = Simulation()
            sim.create_world()
            sim.add_robot(name="arm", data_config="so100")

            # Control loop (same as real robot!)
            obs = sim.get_observation("arm")
            actions = policy.get_actions(obs, "pick up cube")
            for action in actions:
                sim.send_action(action, "arm")
        """
        if self._world is None or self._world._model is None:
            return

        # Resolve robot name
        if robot_name is None:
            if not self._world.robots:
                return
            robot_name = next(iter(self._world.robots))

        if robot_name not in self._world.robots:
            return

        self._apply_sim_action(robot_name, action, n_substeps=n_substeps)

    # -------------------------------------------------------------------
    # World Management
    # -------------------------------------------------------------------

    def create_world(
        self,
        timestep: float = None,
        gravity: List[float] = None,
        ground_plane: bool = True,
    ) -> Dict[str, Any]:
        """Create a new simulation world."""
        _ensure_mujoco()

        if self._world is not None and self._world._model is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "❌ World already exists. Use action='destroy' first, or action='reset'."
                    }
                ],
            }

        # Normalize gravity: accept scalar (e.g. -9.81) or 3-element list
        if gravity is None:
            _gravity = [0.0, 0.0, -9.81]
        elif isinstance(gravity, (int, float)):
            _gravity = [0.0, 0.0, float(gravity)]
        else:
            _gravity = list(gravity)

        self._world = SimWorld(
            timestep=timestep or self.default_timestep,
            gravity=_gravity,
            ground_plane=ground_plane,
        )

        # Default camera
        self._world.cameras["default"] = SimCamera(
            name="default",
            position=[1.5, 1.5, 1.2],
            target=[0.0, 0.0, 0.3],
            width=self.default_width,
            height=self.default_height,
        )

        # Build initial empty scene
        self._compile_world()

        logger.info("🌍 Simulation world created")
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        "🌍 Simulation world created\n"
                        f"⚙️ Timestep: {self._world.timestep}s ({1/self._world.timestep:.0f}Hz physics)\n"
                        f"🌐 Gravity: {self._world.gravity}\n"
                        f"📷 Default camera ready\n"
                        f"🤖 Robot models: {len(_list_menagerie_robots()) if _HAS_ASSET_MANAGER else len(_URDF_REGISTRY)} available\n"
                        "💡 Add robots: action='add_robot' (urdf_path or data_config)\n"
                        "💡 Add objects: action='add_object'\n"
                        "💡 List URDFs: action='list_urdfs'"
                    )
                }
            ],
        }

    def load_scene(self, scene_path: str) -> Dict[str, Any]:
        """Load a complete scene from MJCF XML or URDF file."""
        mj = _ensure_mujoco()

        if not os.path.exists(scene_path):
            return {
                "status": "error",
                "content": [{"text": f"❌ Scene file not found: {scene_path}"}],
            }

        try:
            self._world = SimWorld()
            self._world._model = mj.MjModel.from_xml_path(str(scene_path))
            self._world._data = mj.MjData(self._world._model)
            self._world.status = SimStatus.IDLE

            n_bodies = self._world._model.nbody
            n_joints = self._world._model.njnt
            n_actuators = self._world._model.nu

            logger.info("🌍 Scene loaded: %s", scene_path)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🌍 Scene loaded from {os.path.basename(scene_path)}\n"
                            f"🦴 Bodies: {n_bodies}, 🔩 Joints: {n_joints}, ⚡ Actuators: {n_actuators}\n"
                            "💡 Use action='get_state' to inspect, action='step' to simulate"
                        )
                    }
                ],
            }
        except Exception as e:
            logger.error("Failed to load scene: %s", e)
            return {
                "status": "error",
                "content": [{"text": f"❌ Failed to load scene: {e}"}],
            }

    def _compile_world(self):
        """Compile current world state into a MuJoCo model."""
        mj = _ensure_mujoco()
        try:
            xml = MJCFBuilder.build_objects_only(self._world)
            self._world._xml = xml
            self._world._model = mj.MjModel.from_xml_string(xml)
            self._world._data = mj.MjData(self._world._model)
            self._world.status = SimStatus.IDLE
        except Exception as e:
            logger.error("World compilation failed: %s", e)
            raise

    def _recompile_world(self) -> Dict[str, Any]:
        """Recompile world after structural changes."""
        try:
            self._compile_world()
            return {"status": "success"}
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Recompile failed: {e}"}],
            }

    # -------------------------------------------------------------------
    # Robot Management (with URDF registry + multi-robot support)
    # -------------------------------------------------------------------

    @staticmethod
    def _ensure_meshes(model_path: str, robot_name: str):
        """Check if mesh files referenced by a model XML exist; auto-download if missing.

        Inspects the MJCF XML (and any <include>d files) for mesh file
        references and, if any are missing, triggers an automatic download
        from MuJoCo Menagerie via the asset download system.
        This is a no-op when all meshes are already present.
        """
        import re as _re

        model_dir = os.path.dirname(os.path.abspath(model_path))

        # Collect XML files to inspect (model itself + any <include>d files)
        files_to_check = [model_path]
        try:
            with open(model_path) as _f:
                top_content = _f.read()
            for inc in _re.findall(r'<include\s+file="([^"]+)"', top_content):
                inc_path = os.path.join(model_dir, inc)
                if os.path.exists(inc_path):
                    files_to_check.append(inc_path)
        except Exception:
            pass

        # Scan all XML files for mesh references
        missing = False
        for xml_path in files_to_check:
            try:
                with open(xml_path) as _f:
                    content = _f.read()
            except Exception:
                continue

            mesh_files = _re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', content)
            if not mesh_files:
                continue

            meshdir_match = _re.search(r'meshdir="([^"]*)"', content)
            meshdir = meshdir_match.group(1) if meshdir_match else ""
            xml_dir = os.path.dirname(os.path.abspath(xml_path))

            for mf in mesh_files:
                if not os.path.exists(os.path.join(xml_dir, meshdir, mf)):
                    missing = True
                    break
            if missing:
                break

        if not missing:
            return

        # Auto-download from Menagerie
        logger.info(
            "Downloading mesh files for '%s' from MuJoCo Menagerie (first time only)...",
            robot_name,
        )
        try:
            from strands_robots.assets import resolve_robot_name
            from strands_robots.assets.download import download_robots

            canonical = resolve_robot_name(robot_name)
            download_robots(names=[canonical], force=True)
        except Exception as e:
            logger.warning(
                "Auto-download failed for '%s': %s. "
                "Try manually: python -m strands_robots.assets.download %s",
                robot_name, e, robot_name,
            )

    def add_robot(
        self,
        name: str,
        urdf_path: str = None,
        data_config: str = None,
        position: List[float] = None,
        orientation: List[float] = None,
    ) -> Dict[str, Any]:
        """Add a robot to the simulation.

        Loads the robot's MJCF/URDF scene file directly and replaces the
        simulation model. Objects added via add_object will be injected
        into the loaded model as MuJoCo XML bodies.

        Args:
            name: Unique robot name
            urdf_path: Direct path to URDF/MJCF file
            data_config: Data config name — auto-resolves from asset registry
            position: [x, y, z] position
            orientation: [w, x, y, z] quaternion
        """
        if self._world is None:
            return {
                "status": "error",
                "content": [{"text": "❌ No world. Use action='create_world' first."}],
            }

        if name in self._world.robots:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{name}' already exists."}],
            }

        # Resolve model path (prefers Menagerie scene files with objects/cameras)
        resolved_path = urdf_path
        if not resolved_path and data_config:
            resolved_path = resolve_model(data_config)
            if not resolved_path:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"❌ No model found for '{data_config}'.\n"
                                f"💡 Use action='list_urdfs' to see available robots\n"
                                f"💡 Register with: action='register_urdf', data_config='...', urdf_path='...'"
                            )
                        }
                    ],
                }
        elif not resolved_path and name:
            # Try resolving the name itself as a robot identifier
            resolved_path = resolve_model(name)

        if not resolved_path:
            return {
                "status": "error",
                "content": [
                    {"text": "❌ Either urdf_path or data_config is required."}
                ],
            }

        if not os.path.exists(resolved_path):
            return {
                "status": "error",
                "content": [{"text": f"❌ File not found: {resolved_path}"}],
            }

        mj = _ensure_mujoco()

        robot = SimRobot(
            name=name,
            urdf_path=resolved_path,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
            data_config=data_config,
            namespace=f"{name}/",
        )

        try:
            # Auto-download missing mesh assets before loading
            self._ensure_meshes(resolved_path, data_config or name)

            # Load the robot scene (this becomes the simulation model)
            model = mj.MjModel.from_xml_path(str(resolved_path))
            data = mj.MjData(model)

            # Discover joints and actuators from the loaded model
            joint_names = []
            for i in range(model.njnt):
                jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
                if jnt_name:
                    joint_names.append(jnt_name)
                    robot.joint_ids.append(i)
            robot.joint_names = joint_names

            for i in range(model.nu):
                act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
                # Only assign actuators that relate to this robot's joints
                if act_name:
                    # Check if actuator's joint is one of this robot's joints
                    jnt_id = model.actuator_trnid[i, 0]
                    if jnt_id in robot.joint_ids:
                        robot.actuator_ids.append(i)
                    elif not robot.actuator_ids:
                        # Fallback: if no match found yet, assign all (single-robot scene)
                        pass
                else:
                    robot.actuator_ids.append(i)
            # If no actuators were matched by joint, assign all (backward compat for single-robot)
            if not robot.actuator_ids:
                for i in range(model.nu):
                    robot.actuator_ids.append(i)

            # Discover cameras already in the scene
            for i in range(model.ncam):
                cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
                if cam_name and cam_name not in self._world.cameras:
                    self._world.cameras[cam_name] = SimCamera(
                        name=cam_name,
                        camera_id=i,
                        width=self.default_width,
                        height=self.default_height,
                    )

            # Replace the world model with this robot's scene
            self._world._model = model
            self._world._data = data
            self._world._robot_base_xml = resolved_path  # Remember for recompilation
            self._world.robots[name] = robot

            # Settle physics
            for _ in range(100):
                mj.mj_step(model, data)

            logger.info("Robot '%s' added from %s", name, os.path.basename(resolved_path))
            source = (
                f"data_config='{data_config}'"
                if data_config
                else os.path.basename(resolved_path)
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🤖 Robot '{name}' added to simulation\n"
                            f"📁 Source: {source} → {os.path.basename(resolved_path)}\n"
                            f"📍 Position: {robot.position}\n"
                            f"🔩 Joints: {len(robot.joint_names)} ({', '.join(robot.joint_names[:8])}{'...' if len(robot.joint_names) > 8 else ''})\n"
                            f"⚡ Actuators: {len(robot.actuator_ids)}\n"
                            f"📷 Cameras: {list(self._world.cameras.keys())}\n"
                            f"💡 Run policy: action='run_policy', robot_name='{name}'"
                        )
                    }
                ],
            }

        except Exception as e:
            logger.error("Failed to add robot '%s': %s", name, e)
            return {"status": "error", "content": [{"text": f"❌ Failed to load: {e}"}]}

    def remove_robot(self, name: str) -> Dict[str, Any]:
        if self._world is None or name not in self._world.robots:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{name}' not found."}],
            }
        if name in self._policy_threads:
            self._world.robots[name].policy_running = False
            # Wait for the policy thread to finish before removing the robot
            future = self._policy_threads[name]
            try:
                future.result(timeout=5.0)
            except Exception:
                logger.debug("Policy thread for '%s' did not finish cleanly", name)
            del self._policy_threads[name]
        del self._world.robots[name]
        return {
            "status": "success",
            "content": [{"text": f"🗑️ Robot '{name}' removed."}],
        }

    def list_robots(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if not self._world.robots:
            return {
                "status": "success",
                "content": [{"text": "No robots. Use action='add_robot'."}],
            }

        lines = ["🤖 Robots in simulation:\n"]
        for name, robot in self._world.robots.items():
            status = "🟢 running" if robot.policy_running else "⚪ idle"
            lines.append(
                f"  • {name} ({os.path.basename(robot.urdf_path)})\n"
                f"    Position: {robot.position}, Joints: {len(robot.joint_names)}, "
                f"Config: {robot.data_config or 'direct'}, Status: {status}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def get_robot_state(self, robot_name: str) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {
                "status": "error",
                "content": [{"text": "❌ No simulation running."}],
            }
        if robot_name not in self._world.robots:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
            }

        mj = _ensure_mujoco()
        robot = self._world.robots[robot_name]
        model, data = self._world._model, self._world._data

        state = {}
        for jnt_name in robot.joint_names:
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                qpos_addr = model.jnt_qposadr[jnt_id]
                qvel_addr = model.jnt_dofadr[jnt_id]
                state[jnt_name] = {
                    "position": float(data.qpos[qpos_addr]),
                    "velocity": float(data.qvel[qvel_addr]),
                }

        text = f"🤖 '{robot_name}' state (t={self._world.sim_time:.3f}s):\n"
        for jnt, vals in state.items():
            text += f"  {jnt}: pos={vals['position']:.4f}, vel={vals['velocity']:.4f}\n"

        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"state": state}}],
        }

    # -------------------------------------------------------------------
    # Object Management
    # -------------------------------------------------------------------

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: List[float] = None,
        orientation: List[float] = None,
        size: List[float] = None,
        color: List[float] = None,
        mass: float = 0.1,
        is_static: bool = False,
        mesh_path: str = None,
    ) -> Dict[str, Any]:
        """Add an object to the simulation.

        If a robot has been loaded, objects cannot be added via XML recompilation
        (it would destroy robot actuators). In that case, objects are tracked
        but require loading a scene that includes them, or using load_scene
        with a pre-built scene.

        For robot-free worlds, objects are compiled into the MJCF XML directly.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if name in self._world.objects:
            return {
                "status": "error",
                "content": [{"text": f"❌ Object '{name}' exists."}],
            }

        obj = SimObject(
            name=name,
            shape=shape,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
            size=size or [0.05, 0.05, 0.05],
            color=color or [0.5, 0.5, 0.5, 1.0],
            mass=mass,
            mesh_path=mesh_path,
            is_static=is_static,
        )
        self._world.objects[name] = obj

        # If a robot is loaded, inject the object via XML round-trip
        # (preserves actuators since they're serialized in the XML)
        if self._world.robots:
            try:
                result = self._inject_object_into_scene(obj)
                if result:
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"📦 '{name}' spawned: {shape} at {obj.position}"}
                        ],
                    }
                else:
                    # Fallback: metadata only
                    logger.info(
                        f"Object '{name}' registered (injection returned False — metadata only)"
                    )
                    return {
                        "status": "success",
                        "content": [
                            {
                                "text": (
                                    f"📦 '{name}' registered: {shape} at {obj.position}\n"
                                    f"⚠️ Robot scene loaded — object is tracked but not physically spawned.\n"
                                    f"💡 For objects + robots: use load_scene with a pre-built scene,\n"
                                    f"   or use MuJoCo's interactive viewer to position objects."
                                )
                            }
                        ],
                    }
            except Exception as e:
                logger.warning("Object injection failed, tracking metadata only: %s", e)
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"📦 '{name}' registered: {shape} at {obj.position}\n"
                                f"⚠️ Injection failed ({e}) — object tracked as metadata only."
                            )
                        }
                    ],
                }

        # No robots — safe to recompile XML
        result = self._recompile_world()
        if result["status"] == "error":
            del self._world.objects[name]
            return result

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📦 '{name}' added: {shape} at {obj.position}, "
                        f"size={obj.size}, {'static' if is_static else f'{mass}kg'}"
                    )
                }
            ],
        }

    def _inject_object_into_scene(self, obj: SimObject) -> bool:
        """Inject object into a running simulation via XML round-trip.

        Saves current model to XML, injects the object body XML before
        </worldbody>, then reloads. Preserves actuators because they are
        serialized in the saved XML.

        Args:
            obj: SimObject to inject

        Returns:
            True if injection succeeded, False otherwise
        """
        mj = _ensure_mujoco()
        if self._world._model is None:
            return False

        # Determine the original model directory for mesh resolution
        robot_base_dir = None
        if self._world._robot_base_xml:
            robot_base_dir = os.path.dirname(
                os.path.abspath(self._world._robot_base_xml)
            )

        # Always use tempdir to avoid polluting shared asset directories.
        # Set meshdir in XML to point back to original asset dir for mesh resolution.
        tmpdir = tempfile.mkdtemp(prefix="strands_sim_")
        scene_path = os.path.join(tmpdir, "scene_with_objects.xml")

        mj.mj_saveLastXML(scene_path, self._world._model)

        # Patch meshdir/texturedir to point to original asset dir for mesh resolution
        if robot_base_dir and os.path.isdir(robot_base_dir):
            with open(scene_path) as f:
                xml_content = f.read()
            # Add or update meshdir/texturedir in <compiler> to point to asset dir
            if "<compiler" in xml_content and "meshdir=" not in xml_content:
                xml_content = xml_content.replace(
                    "<compiler",
                    f'<compiler meshdir="{robot_base_dir}" texturedir="{robot_base_dir}"',
                    1,
                )
            with open(scene_path, "w") as f:
                f.write(xml_content)

        # Read XML, inject object before </worldbody>
        with open(scene_path) as f:
            xml_content = f.read()

        obj_xml = MJCFBuilder._object_xml(obj, indent=4)
        xml_content = xml_content.replace("</worldbody>", f"{obj_xml}\n</worldbody>")

        with open(scene_path, "w") as f:
            f.write(xml_content)

        # Reload model (preserves actuators since they're in the XML)
        try:
            new_model = mj.MjModel.from_xml_path(str(scene_path))
            new_data = mj.MjData(new_model)

            # Copy state from old model
            old_nq = min(self._world._data.qpos.shape[0], new_data.qpos.shape[0])
            old_nv = min(self._world._data.qvel.shape[0], new_data.qvel.shape[0])
            new_data.qpos[:old_nq] = self._world._data.qpos[:old_nq]
            new_data.qvel[:old_nv] = self._world._data.qvel[:old_nv]

            # Copy ctrl
            old_nu = min(self._world._data.ctrl.shape[0], new_data.ctrl.shape[0])
            new_data.ctrl[:old_nu] = self._world._data.ctrl[:old_nu]

            mj.mj_forward(new_model, new_data)

            # Update world
            self._world._model = new_model
            self._world._data = new_data

            # Re-discover robot joints/actuators (IDs may shift)
            for robot_name, robot in self._world.robots.items():
                robot.joint_ids = []
                robot.actuator_ids = []
                for jnt_name in robot.joint_names:
                    jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                    if jid >= 0:
                        robot.joint_ids.append(jid)
                for i in range(new_model.nu):
                    jnt_id = new_model.actuator_trnid[i, 0]
                    if jnt_id in robot.joint_ids:
                        robot.actuator_ids.append(i)
                # Fallback: if no actuators matched by joint, assign all (single-robot scene)
                if not robot.actuator_ids:
                    for i in range(new_model.nu):
                        robot.actuator_ids.append(i)

            # Clean up temp directory
            try:
                import shutil

                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error("Object injection reload failed: %s", e)
            # Clean up temp directory on failure too
            try:
                import shutil

                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
            return False

    def remove_object(self, name: str) -> Dict[str, Any]:
        if self._world is None or name not in self._world.objects:
            return {
                "status": "error",
                "content": [{"text": f"❌ Object '{name}' not found."}],
            }
        del self._world.objects[name]

        # Use XML round-trip when robots are present to preserve their state.
        if self._world.robots:
            self._eject_body_from_scene(name)
        else:
            self._recompile_world()
        return {"status": "success", "content": [{"text": f"🗑️ '{name}' removed."}]}

    def _eject_body_from_scene(self, body_name: str) -> bool:
        """Remove a named body from the scene via XML round-trip, preserving robot state.

        Uses xml.etree.ElementTree for correct handling of nested <body> elements
        (regex .*? would match only the first closing tag, potentially orphaning
        children in multi-body hierarchies).
        """
        mj = _ensure_mujoco()
        import xml.etree.ElementTree as ET

        tmpdir = tempfile.mkdtemp(prefix="strands_eject_")
        scene_path = os.path.join(tmpdir, "scene_ejected.xml")

        try:
            mj.mj_saveLastXML(scene_path, self._world._model)

            tree = ET.parse(scene_path)
            root = tree.getroot()

            # Patch meshdir / texturedir if needed
            robot_base_dir = None
            if self._world._robot_base_xml:
                robot_base_dir = os.path.dirname(
                    os.path.abspath(self._world._robot_base_xml)
                )
            if robot_base_dir:
                compiler = root.find("compiler")
                if compiler is not None:
                    if compiler.get("meshdir") is None:
                        compiler.set("meshdir", robot_base_dir)
                    if compiler.get("texturedir") is None:
                        compiler.set("texturedir", robot_base_dir)

            # Walk the tree and remove the target body element.
            # We iterate over all elements and check children, because
            # ET requires the parent reference to call remove().
            removed = False
            for parent in root.iter():
                for child in list(parent):
                    if child.tag == "body" and child.get("name") == body_name:
                        parent.remove(child)
                        removed = True

            if not removed:
                logger.warning(
                    f"Body '{body_name}' not found in MJCF XML — skipping ejection."
                )

            tree.write(scene_path, xml_declaration=True)

            new_model = mj.MjModel.from_xml_path(scene_path)
            new_data = mj.MjData(new_model)

            # Copy state
            old_nq = min(self._world._data.qpos.shape[0], new_data.qpos.shape[0])
            new_data.qpos[:old_nq] = self._world._data.qpos[:old_nq]
            old_nv = min(self._world._data.qvel.shape[0], new_data.qvel.shape[0])
            new_data.qvel[:old_nv] = self._world._data.qvel[:old_nv]
            old_nu = min(self._world._data.ctrl.shape[0], new_data.ctrl.shape[0])
            new_data.ctrl[:old_nu] = self._world._data.ctrl[:old_nu]

            mj.mj_forward(new_model, new_data)
            self._world._model = new_model
            self._world._data = new_data
            return True
        except Exception as e:
            logger.error("Body ejection failed for \'%s\': %s", body_name, e)
            # Fallback to recompile (lossy but at least works)
            self._recompile_world()
            return False
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def move_object(
        self, name: str, position: List[float] = None, orientation: List[float] = None
    ) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"❌ '{name}' not found."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        jnt_name = f"{name}_joint"
        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            qpos_addr = model.jnt_qposadr[jnt_id]
            if position:
                data.qpos[qpos_addr : qpos_addr + 3] = position
                self._world.objects[name].position = position
            if orientation:
                data.qpos[qpos_addr + 3 : qpos_addr + 7] = orientation
                self._world.objects[name].orientation = orientation
            mj.mj_forward(model, data)

        return {
            "status": "success",
            "content": [{"text": f"📍 '{name}' moved to {position or 'same'}"}],
        }

    def list_objects(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if not self._world.objects:
            return {"status": "success", "content": [{"text": "No objects."}]}

        lines = ["📦 Objects:\n"]
        for name, obj in self._world.objects.items():
            lines.append(
                f"  • {name}: {obj.shape} at {obj.position}, "
                f"{'static' if obj.is_static else f'{obj.mass}kg'}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # -------------------------------------------------------------------
    # Camera Management
    # -------------------------------------------------------------------

    def add_camera(
        self,
        name: str,
        position: List[float] = None,
        target: List[float] = None,
        fov: float = 60.0,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}

        cam = SimCamera(
            name=name,
            position=position or [1.0, 1.0, 1.0],
            target=target or [0.0, 0.0, 0.0],
            fov=fov,
            width=width,
            height=height,
        )
        self._world.cameras[name] = cam

        # If robots are loaded, inject camera via XML round-trip to preserve
        # actuators/joints (build_objects_only would destroy them).
        if self._world.robots and self._world._model is not None:
            try:
                self._inject_camera_into_scene(cam)
            except Exception as e:
                logger.warning(
                    f"Camera injection failed: {e}. Camera tracked as metadata only."
                )
        else:
            self._recompile_world()

        return {
            "status": "success",
            "content": [{"text": f"📷 Camera '{name}' added at {cam.position}"}],
        }

    def _inject_camera_into_scene(self, cam: SimCamera) -> bool:
        """Inject a camera into a running simulation via XML round-trip.

        Same approach as _inject_object_into_scene: saves the current model
        to XML, injects a <camera> element, and reloads. Preserves all
        actuators/joints because they are serialized in the saved XML.

        Args:
            cam: SimCamera to inject

        Returns:
            True if injection succeeded, False otherwise
        """
        mj = _ensure_mujoco()
        if self._world._model is None:
            return False

        robot_base_dir = None
        if self._world._robot_base_xml:
            robot_base_dir = os.path.dirname(
                os.path.abspath(self._world._robot_base_xml)
            )

        if robot_base_dir and os.path.isdir(robot_base_dir):
            scene_path = os.path.join(robot_base_dir, "_strands_scene_with_cameras.xml")
        else:
            tmpdir = tempfile.mkdtemp(prefix="strands_cam_")
            scene_path = os.path.join(tmpdir, "scene_with_cameras.xml")

        mj.mj_saveLastXML(scene_path, self._world._model)

        with open(scene_path) as f:
            xml_content = f.read()

        px, py, pz = cam.position
        cam_xml = f'    <camera name="{cam.name}" pos="{px} {py} {pz}" fovy="{cam.fov}" mode="fixed"/>'
        xml_content = xml_content.replace("</worldbody>", f"{cam_xml}\n</worldbody>")

        with open(scene_path, "w") as f:
            f.write(xml_content)

        try:
            new_model = mj.MjModel.from_xml_path(str(scene_path))
            new_data = mj.MjData(new_model)

            old_nq = min(self._world._data.qpos.shape[0], new_data.qpos.shape[0])
            old_nv = min(self._world._data.qvel.shape[0], new_data.qvel.shape[0])
            new_data.qpos[:old_nq] = self._world._data.qpos[:old_nq]
            new_data.qvel[:old_nv] = self._world._data.qvel[:old_nv]

            old_nu = min(self._world._data.ctrl.shape[0], new_data.ctrl.shape[0])
            new_data.ctrl[:old_nu] = self._world._data.ctrl[:old_nu]

            mj.mj_forward(new_model, new_data)

            self._world._model = new_model
            self._world._data = new_data

            for robot_name, robot in self._world.robots.items():
                robot.joint_ids = []
                robot.actuator_ids = []
                for jnt_name in robot.joint_names:
                    jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                    if jid >= 0:
                        robot.joint_ids.append(jid)
                for i in range(new_model.nu):
                    robot.actuator_ids.append(i)

            try:
                if scene_path.endswith("_strands_scene_with_cameras.xml"):
                    os.remove(scene_path)
            except OSError:
                pass

            return True
        except Exception as e:
            logger.error("Camera injection reload failed: %s", e)
            try:
                if scene_path.endswith("_strands_scene_with_cameras.xml"):
                    os.remove(scene_path)
            except OSError:
                pass
            return False

    def remove_camera(self, name: str) -> Dict[str, Any]:
        if self._world is None or name not in self._world.cameras:
            return {
                "status": "error",
                "content": [{"text": f"❌ Camera '{name}' not found."}],
            }
        del self._world.cameras[name]
        return {
            "status": "success",
            "content": [{"text": f"🗑️ Camera '{name}' removed."}],
        }

    # -------------------------------------------------------------------
    # Policy Execution
    # -------------------------------------------------------------------
    def _get_renderer(self, width: int, height: int):
        """Get a cached MuJoCo renderer, creating one only if needed."""
        mj = _ensure_mujoco()
        key = (width, height)
        # Invalidate cache if model changed (e.g. after add/remove object)
        if self._renderer_model is not self._world._model:
            self._renderers.clear()
            self._renderer_model = self._world._model
        if key not in self._renderers:
            self._renderers[key] = mj.Renderer(
                self._world._model, height=height, width=width
            )
        return self._renderers[key]

    def _get_sim_observation(
        self, robot_name: str, cam_name: str = None
    ) -> Dict[str, Any]:
        """Get observation from sim (same format as real robot).

        Args:
            robot_name: Name of the robot to observe
            cam_name: Optional specific camera name. If None, uses first available.
        """
        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        robot = self._world.robots[robot_name]

        obs = {}
        for jnt_name in robot.joint_names:
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                obs[jnt_name] = float(data.qpos[model.jnt_qposadr[jnt_id]])

        # Render camera(s) for observation
        cameras_to_render = []
        if cam_name:
            cameras_to_render = [cam_name]
        else:
            # Render all scene cameras
            cameras_to_render = [
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
                for i in range(model.ncam)
            ]

        for cname in cameras_to_render:
            if not cname:
                continue
            cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cname)
            if cam_id >= 0:
                cam_info = self._world.cameras.get(cname)
                h = cam_info.height if cam_info else self.default_height
                w = cam_info.width if cam_info else self.default_width
                try:
                    renderer = self._get_renderer(w, h)
                    renderer.update_scene(data, camera=cam_id)
                    obs[cname] = renderer.render().copy()
                except Exception as e:
                    logger.debug("Camera render failed for %s: %s", cname, e)

        return obs

    def _apply_sim_action(
        self, robot_name: str, action_dict: Dict[str, Any], n_substeps: int = 1
    ):
        """Apply action dict to sim (same interface as robot.send_action).

        Args:
            robot_name: Robot to apply action to
            action_dict: Joint name -> value mapping
            n_substeps: Number of physics substeps per action (for stability)
        """
        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        for key, value in action_dict.items():
            # Try actuator first (preferred)
            act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, key)
            if act_id >= 0:
                data.ctrl[act_id] = float(value)
            else:
                # Fallback: try matching joint index to actuator index
                jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, key)
                if jnt_id >= 0 and jnt_id < model.nu:
                    data.ctrl[jnt_id] = float(value)

        for _ in range(max(1, n_substeps)):
            mj.mj_step(model, data)

        self._world.sim_time = data.time
        self._world.step_count += n_substeps

    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        action_horizon: int = 8,
        control_frequency: float = 50.0,
        fast_mode: bool = False,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Run a policy on a simulated robot (blocking). Same Policy ABC as robot.py."""
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if robot_name not in self._world.robots:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
            }

        robot = self._world.robots[robot_name]

        try:
            # Runtime import to avoid stale module-level references from test mocks
            from strands_robots.policies import create_policy as _create_policy

            policy = _create_policy(policy_provider, **policy_kwargs)
            policy.set_robot_state_keys(robot.joint_names)

            robot.policy_running = True
            robot.policy_instruction = instruction
            robot.policy_steps = 0

            start_time = time.time()
            action_sleep = 1.0 / control_frequency

            while time.time() - start_time < duration and robot.policy_running:
                observation = self._get_sim_observation(robot_name)

                # Call policy.get_actions — handle both sync and async
                coro_or_result = policy.get_actions(observation, instruction)
                actions = _resolve_coroutine(coro_or_result)

                for action_dict in actions[:action_horizon]:
                    if not robot.policy_running:
                        break

                    # Record trajectory if recording
                    if self._world._recording:
                        self._world._trajectory.append(
                            TrajectoryStep(
                                timestamp=time.time(),
                                sim_time=self._world.sim_time,
                                robot_name=robot_name,
                                observation={
                                    k: v
                                    for k, v in observation.items()
                                    if not isinstance(v, np.ndarray)
                                },
                                action=action_dict,
                                instruction=instruction,
                            )
                        )
                        # Also write to LeRobotDataset if active
                        if self._world._dataset_recorder is not None:
                            self._world._dataset_recorder.add_frame(
                                observation=observation,
                                action=action_dict,
                                task=instruction,
                            )

                    self._apply_sim_action(robot_name, action_dict)
                    robot.policy_steps += 1
                    if not fast_mode:
                        time.sleep(action_sleep)

            elapsed = time.time() - start_time
            robot.policy_running = False

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"✅ Policy complete on '{robot_name}'\n"
                            f"🧠 {policy_provider} | 🎯 {instruction}\n"
                            f"⏱️ {elapsed:.1f}s | 📊 {robot.policy_steps} steps | "
                            f"🕐 sim_t={self._world.sim_time:.3f}s"
                        )
                    }
                ],
            }

        except Exception as e:
            robot.policy_running = False
            return {"status": "error", "content": [{"text": f"❌ Policy failed: {e}"}]}

    def start_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        fast_mode: bool = False,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Start policy execution in background (non-blocking).

        Submits run_policy to a thread pool so the agent can continue
        issuing other actions while the policy runs.

        Args:
            robot_name: Robot to run policy on
            policy_provider: Policy provider name (e.g. "groot", "lerobot_local", "mock")
            instruction: Natural language instruction
            duration: Maximum duration in seconds
            fast_mode: If True, skip sleep between actions for faster data collection
            **policy_kwargs: Extra kwargs forwarded to create_policy
        """
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if robot_name not in self._world.robots:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
            }

        future = self._executor.submit(
            self.run_policy,
            robot_name,
            policy_provider,
            instruction,
            duration,
            fast_mode=fast_mode,
            **policy_kwargs,
        )
        self._policy_threads[robot_name] = future

        return {
            "status": "success",
            "content": [{"text": f"🚀 Policy started on '{robot_name}' (async)"}],
        }

    # -------------------------------------------------------------------
    # Rendering & Observation
    # -------------------------------------------------------------------

    def render(
        self, camera_name: str = "default", width: int = None, height: int = None
    ) -> Dict[str, Any]:
        """Render a camera view as base64 PNG image.

        Args:
            camera_name: Name of camera to render from. Uses scene camera if found,
                otherwise falls back to default free camera.
            width: Override render width
            height: Override render height
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        w = width or self.default_width
        h = height or self.default_height

        try:
            renderer = self._get_renderer(w, h)

            # Try to use named camera
            cam_id = mj.mj_name2id(
                self._world._model, mj.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if cam_id >= 0:
                renderer.update_scene(self._world._data, camera=cam_id)
            else:
                # Fall back to default free camera
                renderer.update_scene(self._world._data)

            img = renderer.render().copy()

            from PIL import Image

            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"📸 {w}x{h} from '{camera_name}' at t={self._world.sim_time:.3f}s"
                    },
                    {"image": {"format": "png", "source": {"bytes": png_bytes}}},
                ],
            }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Render failed: {e}"}]}

    def render_depth(
        self, camera_name: str = "default", width: int = None, height: int = None
    ) -> Dict[str, Any]:
        """Render depth map from a camera."""
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        w = width or self.default_width
        h = height or self.default_height

        try:
            cam_id = -1
            if camera_name and camera_name != "default":
                cam_id = mj.mj_name2id(
                    self._world._model, mj.mjtObj.mjOBJ_CAMERA, camera_name
                )

            renderer = self._get_renderer(w, h)
            if cam_id >= 0:
                renderer.update_scene(self._world._data, camera=cam_id)
            else:
                renderer.update_scene(self._world._data)
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📸 Depth {w}x{h} from '{camera_name}'\n"
                            f"Min: {float(depth.min()):.3f}m, Max: {float(depth.max()):.3f}m"
                        )
                    },
                    {
                        "json": {
                            "depth_min": float(depth.min()),
                            "depth_max": float(depth.max()),
                        }
                    },
                ],
            }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Depth render failed: {e}"}],
            }

    def get_contacts(self) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = (
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
            )
            g2 = (
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
            )
            contacts.append(
                {"geom1": g1, "geom2": g2, "dist": float(c.dist), "pos": c.pos.tolist()}
            )

        text = f"💥 {len(contacts)} contacts" if contacts else "No contacts."
        if contacts:
            for c in contacts[:10]:
                text += f"\n  • {c['geom1']} ↔ {c['geom2']} (d={c['dist']:.4f})"

        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"contacts": contacts}}],
        }

    # -------------------------------------------------------------------
    # Simulation Control
    # -------------------------------------------------------------------

    def step(self, n_steps: int = 1) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        for _ in range(n_steps):
            mj.mj_step(self._world._model, self._world._data)

        self._world.sim_time = self._world._data.time
        self._world.step_count += n_steps

        return {
            "status": "success",
            "content": [
                {
                    "text": f"⏩ +{n_steps} steps | t={self._world.sim_time:.4f}s | total={self._world.step_count}"
                }
            ],
        }

    def reset(self) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}

        mj = _ensure_mujoco()
        mj.mj_resetData(self._world._model, self._world._data)
        self._world.sim_time = 0.0
        self._world.step_count = 0
        for r in self._world.robots.values():
            r.policy_running = False
            r.policy_steps = 0

        return {
            "status": "success",
            "content": [{"text": "🔄 Reset to initial state."}],
        }

    def get_state(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}

        lines = [
            "🌍 Simulation State",
            f"🕐 t={self._world.sim_time:.4f}s (step {self._world.step_count})",
            f"⚙️ dt={self._world.timestep}s | 🌐 g={self._world.gravity}",
            f"🤖 Robots: {len(self._world.robots)} | 📦 Objects: {len(self._world.objects)} | 📷 Cameras: {len(self._world.cameras)}",
        ]
        if self._world._model:
            lines.append(
                f"🦴 Bodies: {self._world._model.nbody} | 🔩 Joints: {self._world._model.njnt} | ⚡ Actuators: {self._world._model.nu}"
            )
        if self._world._recording:
            lines.append(f"🔴 Recording: {len(self._world._trajectory)} steps")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def destroy(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "success", "content": [{"text": "No world to destroy."}]}
        for r in self._world.robots.values():
            r.policy_running = False
        self._close_viewer()
        self._world = None
        return {"status": "success", "content": [{"text": "🗑️ World destroyed."}]}

    def set_gravity(self, gravity) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if isinstance(gravity, (int, float)):
            gravity = [0.0, 0.0, float(gravity)]
        self._world._model.opt.gravity[:] = gravity
        self._world.gravity = gravity
        return {"status": "success", "content": [{"text": f"🌐 Gravity: {gravity}"}]}

    def set_timestep(self, timestep: float) -> Dict[str, Any]:
        """Change the physics simulation timestep.

        Args:
            timestep: New timestep in seconds (e.g. 0.002 for 500Hz)
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        self._world._model.opt.timestep = timestep
        self._world.timestep = timestep
        return {
            "status": "success",
            "content": [{"text": f"⏱️ Timestep: {timestep}s ({1/timestep:.0f}Hz)"}],
        }

    # -------------------------------------------------------------------
    # Domain Randomization
    # -------------------------------------------------------------------

    def randomize(
        self,
        randomize_colors: bool = True,
        randomize_lighting: bool = True,
        randomize_physics: bool = False,
        randomize_positions: bool = False,
        position_noise: float = 0.02,
        color_range: Tuple[float, float] = (0.1, 1.0),
        friction_range: Tuple[float, float] = (0.5, 1.5),
        mass_range: Tuple[float, float] = (0.5, 2.0),
        seed: int = None,
    ) -> Dict[str, Any]:
        """Apply domain randomization to the scene.

        Useful for sim-to-real transfer — trains policies to be robust
        to visual and physical variations.

        Args:
            randomize_colors: Randomize object/geom colors
            randomize_lighting: Randomize light positions and intensities
            randomize_physics: Randomize friction and mass
            randomize_positions: Add noise to object positions
            position_noise: Position noise magnitude (meters)
            color_range: (min, max) for random color channels
            friction_range: (min, max) multiplier for friction
            mass_range: (min, max) multiplier for mass
            seed: Random seed for reproducibility
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        mj = _ensure_mujoco()
        model = self._world._model
        data = self._world._data
        changes = []

        # Randomize geom colors
        if randomize_colors:
            for i in range(model.ngeom):
                geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
                if geom_name and geom_name != "ground":
                    model.geom_rgba[i, :3] = rng.uniform(
                        color_range[0], color_range[1], size=3
                    )
            changes.append(f"🎨 Colors: {model.ngeom} geoms randomized")

        # Randomize lighting
        if randomize_lighting:
            for i in range(model.nlight):
                # Randomize position within bounds
                model.light_pos[i] += rng.uniform(-0.5, 0.5, size=3)
                # Randomize diffuse intensity
                model.light_diffuse[i] = rng.uniform(0.3, 1.0, size=3)
            changes.append(f"💡 Lighting: {model.nlight} lights randomized")

        # Randomize physics (friction, mass)
        if randomize_physics:
            for i in range(model.ngeom):
                # Randomize friction
                model.geom_friction[i, 0] *= rng.uniform(*friction_range)
            for i in range(model.nbody):
                if model.body_mass[i] > 0:
                    model.body_mass[i] *= rng.uniform(*mass_range)
            changes.append(
                f"⚙️ Physics: friction×[{friction_range}], mass×[{mass_range}]"
            )

        # Randomize object positions
        if randomize_positions:
            for obj_name, obj in self._world.objects.items():
                if not obj.is_static:
                    jnt_name = f"{obj_name}_joint"
                    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                    if jnt_id >= 0:
                        qpos_addr = model.jnt_qposadr[jnt_id]
                        noise = rng.uniform(-position_noise, position_noise, size=3)
                        data.qpos[qpos_addr : qpos_addr + 3] += noise
            mj.mj_forward(model, data)
            changes.append(f"📍 Positions: ±{position_noise}m noise on dynamic objects")

        return {
            "status": "success",
            "content": [
                {"text": "🎲 Domain Randomization applied:\n" + "\n".join(changes)}
            ],
        }

    # -------------------------------------------------------------------
    # Trajectory Recording
    # -------------------------------------------------------------------

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str = None,
        push_to_hub: bool = False,
        vcodec: str = "libsvtav1",
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """Start recording to LeRobotDataset format (parquet + video).

        Every frame captured during run_policy is written via
        DatasetRecorder.add_frame() — producing training-ready data.

        Args:
            repo_id: Dataset ID (e.g. "user/my_sim_data")
            task: Task description for the episode
            fps: Recording FPS
            root: Local directory for dataset storage
            push_to_hub: Auto-push to HuggingFace when stopping
            vcodec: Video codec (h264, hevc, libsvtav1)
            overwrite: If True, remove existing dataset dir before recording
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world."}]}
        # Re-check at call time to avoid stale module-level state from test mocks
        try:
            from .dataset_recorder import DatasetRecorder as _DatasetRecorder
            from .dataset_recorder import has_lerobot_dataset as _has_lerobot
        except ImportError:
            def _has_lerobot():
                return False

            _DatasetRecorder = None
        if not _has_lerobot() or _DatasetRecorder is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "lerobot not installed. Install with: pip install lerobot\n"
                            "Required for dataset recording."
                        )
                    }
                ],
            }

        self._world._recording = True
        self._world._trajectory = []
        self._world._push_to_hub = push_to_hub

        try:
            # Clean up existing dataset directory if overwrite is True
            if overwrite:
                import shutil
                from pathlib import Path as _Path

                # repo_id could be a path or a HF-style "user/dataset"
                if root:
                    dataset_dir = _Path(root)
                elif (
                    "/" not in repo_id
                    or repo_id.startswith("/")
                    or repo_id.startswith("./")
                ):
                    dataset_dir = _Path(repo_id)
                else:
                    dataset_dir = (
                        _Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
                    )
                if dataset_dir.exists() and dataset_dir.is_dir():
                    shutil.rmtree(dataset_dir)
                    logger.info("Removed existing dataset dir: %s", dataset_dir)

            joint_names = []
            camera_keys = []
            robot_type = "unknown"
            for rname, robot in self._world.robots.items():
                joint_names.extend(robot.joint_names)
                robot_type = robot.data_config or rname
            mj = _ensure_mujoco()
            for i in range(self._world._model.ncam):
                cam_name = mj.mj_id2name(self._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
                if cam_name:
                    camera_keys.append(cam_name)

            self._world._dataset_recorder = _DatasetRecorder.create(
                repo_id=repo_id,
                fps=fps,
                robot_type=robot_type,
                joint_names=joint_names,
                camera_keys=camera_keys,
                task=task,
                root=root,
                vcodec=vcodec,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Recording to LeRobotDataset: {repo_id}\n"
                            f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                            f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                            f"Run policies to capture frames, then stop_recording to save episode"
                        )
                    }
                ],
            }
        except Exception as e:
            self._world._recording = False
            logger.error("Dataset recorder init failed: %s", e)
            return {
                "status": "error",
                "content": [{"text": f"Dataset init failed: {e}"}],
            }

    def stop_recording(self, output_path: str = None) -> Dict[str, Any]:
        """Stop recording and save episode to LeRobotDataset.

        Writes parquet, encodes video, computes stats.
        Optionally pushes to HuggingFace Hub.
        """
        if self._world is None or not self._world._recording:
            return {"status": "error", "content": [{"text": "Not recording."}]}

        self._world._recording = False
        recorder = self._world._dataset_recorder

        if recorder is None:
            return {
                "status": "error",
                "content": [{"text": "No dataset recorder active."}],
            }

        recorder.save_episode()
        push_result = None
        if getattr(self._world, "_push_to_hub", False):
            push_result = recorder.push_to_hub(tags=["strands-robots", "sim"])

        repo_id = recorder.repo_id
        frame_count = recorder.frame_count
        episode_count = recorder.episode_count
        root = recorder.root

        recorder.finalize()
        self._world._dataset_recorder = None
        self._world._trajectory = []

        text = (
            f"Episode saved to LeRobotDataset\n"
            f"{repo_id} -- {frame_count} frames, {episode_count} episode(s)\n"
            f"Local: {root}"
        )
        if push_result and push_result.get("status") == "success":
            text += "\nPushed to HuggingFace Hub"

        return {"status": "success", "content": [{"text": text}]}

    def get_recording_status(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}

        recording = self._world._recording
        steps = len(self._world._trajectory)

        return {
            "status": "success",
            "content": [
                {
                    "text": f"{'🔴 Recording' if recording else '⚪ Not recording'}: {steps} steps captured"
                }
            ],
        }

    # -------------------------------------------------------------------
    # Viewer
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Video Recording
    # -------------------------------------------------------------------

    def record_video(
        self,
        robot_name: str,
        policy_provider: str = "lerobot_local",
        instruction: str = "",
        duration: float = 10.0,
        fps: int = 30,
        camera_name: str = None,
        width: int = 640,
        height: int = 480,
        output_path: str = None,
        cosmos_transfer: bool = False,
        cosmos_prompt: str = None,
        cosmos_control: str = "depth",
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Run a policy and record video simultaneously.

        This is the main demo recording method. It:
        1. Creates a policy (lerobot_local by default)
        2. Runs the policy loop
        3. Renders each frame from a camera
        4. Writes frames to MP4
        5. (Optional) Runs Cosmos Transfer 2.5 to make photorealistic

        Args:
            robot_name: Robot to run policy on
            policy_provider: Policy provider (lerobot_local, groot, mock, etc.)
            instruction: Natural language instruction
            duration: Duration in seconds
            fps: Video frames per second
            camera_name: Camera to render from (None = first available or free camera)
            width: Video width
            height: Video height
            output_path: Output MP4 path (auto-generated if None)
            cosmos_transfer: If True, run Cosmos Transfer 2.5 on the sim video
            cosmos_prompt: Text prompt for Cosmos Transfer (default: auto from instruction)
            cosmos_control: Control type for Cosmos Transfer: "depth", "edge", "seg", "multi"
            **policy_kwargs: Passed to create_policy (e.g. pretrained_name_or_path)
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if robot_name not in self._world.robots:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
            }

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        robot = self._world.robots[robot_name]

        # Auto-generate output path
        if output_path is None:
            output_dir = os.path.join(tempfile.gettempdir(), "strands_sim")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"video_{robot_name}_{int(time.time())}.mp4"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Resolve camera
        cam_id = -1
        if camera_name:
            cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        else:
            # Use first available camera
            if model.ncam > 0:
                cam_id = 0
                camera_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, 0)

        try:
            import imageio

            # Create policy — runtime import to avoid stale module-level refs
            from strands_robots.policies import create_policy as _create_policy

            policy = _create_policy(policy_provider, **policy_kwargs)
            policy.set_robot_state_keys(robot.joint_names)

            total_frames = int(duration * fps)
            dt = 1.0 / fps
            phys_per_frame = max(1, int(dt / model.opt.timestep))

            logger.info(
                f"Recording {total_frames} frames at {fps}fps, {phys_per_frame} physics steps/frame"
            )

            writer = imageio.get_writer(
                output_path, fps=fps, quality=8, macro_block_size=1
            )
            robot.policy_running = True
            robot.policy_instruction = instruction
            robot.policy_steps = 0

            start_time = time.time()

            for frame_idx in range(total_frames):
                if not robot.policy_running:
                    break

                # Get observation (with camera)
                observation = self._get_sim_observation(
                    robot_name, cam_name=camera_name
                )

                # Policy inference — use _resolve_coroutine helper
                try:
                    coro_or_result = policy.get_actions(observation, instruction)
                    actions = _resolve_coroutine(coro_or_result)
                except Exception as e:
                    logger.warning("Policy inference failed at frame %s: %s", frame_idx, e)
                    actions = [{key: 0.0 for key in robot.joint_names}]

                # Apply first action
                if actions:
                    self._apply_sim_action(
                        robot_name, actions[0], n_substeps=phys_per_frame
                    )
                    robot.policy_steps += 1
                else:
                    # Just step physics
                    for _ in range(phys_per_frame):
                        mj.mj_step(model, data)
                    self._world.sim_time = data.time

                # Render frame
                renderer = self._get_renderer(width, height)
                if cam_id >= 0:
                    renderer.update_scene(data, camera=cam_id)
                else:
                    renderer.update_scene(data)
                frame = renderer.render().copy()

                writer.append_data(frame)

                if (frame_idx + 1) % 60 == 0:
                    elapsed = time.time() - start_time
                    real_fps = (frame_idx + 1) / elapsed
                    logger.info(
                        f"  frame {frame_idx+1}/{total_frames} | {elapsed:.1f}s | {real_fps:.1f} fps"
                    )

            writer.close()
            robot.policy_running = False

            elapsed = time.time() - start_time
            file_kb = os.path.getsize(output_path) / 1024

            # Cosmos Transfer 2.5: Convert sim video to photorealistic
            cosmos_output = None
            if cosmos_transfer:
                try:
                    from strands_robots.cosmos_transfer import (
                        CosmosTransferConfig,
                        CosmosTransferPipeline,
                    )

                    prompt = (
                        cosmos_prompt
                        or instruction
                        or f"Robot {robot_name} performing task in modern workspace"
                    )
                    cosmos_out_path = output_path.replace(".mp4", "_photorealistic.mp4")

                    logger.info(
                        f"🎬 Running Cosmos Transfer 2.5 ({cosmos_control}) → {cosmos_out_path}"
                    )

                    config = CosmosTransferConfig(control_type=cosmos_control)
                    pipeline = CosmosTransferPipeline(config=config)
                    pipeline.transfer_video(
                        input_video=output_path,
                        prompt=prompt,
                        output_path=cosmos_out_path,
                    )

                    cosmos_output = cosmos_out_path
                    cosmos_kb = (
                        os.path.getsize(cosmos_out_path) / 1024
                        if os.path.exists(cosmos_out_path)
                        else 0
                    )
                    logger.info("🎬 Cosmos Transfer complete: %.0f KB", cosmos_kb)
                except ImportError:
                    logger.warning(
                        "Cosmos Transfer not available (missing deps). Skipping."
                    )
                except Exception as e:
                    logger.warning(
                        f"Cosmos Transfer failed: {e}. Sim video still available."
                    )

            result_text = (
                f"🎬 Video recorded: {output_path}\n"
                f"📹 {total_frames} frames, {fps}fps, {width}x{height}\n"
                f"🤖 Robot: {robot_name} | 🧠 Policy: {policy_provider}\n"
                f"⏱️ {elapsed:.1f}s real time | 💾 {file_kb:.0f} KB\n"
                f"📊 {robot.policy_steps} policy steps"
            )
            if cosmos_output:
                result_text += f"\n🌌 Photorealistic: {cosmos_output}"

            return {
                "status": "success",
                "content": [{"text": result_text}],
            }

        except Exception as e:
            robot.policy_running = False
            logger.error("Video recording failed: %s", e)
            return {
                "status": "error",
                "content": [{"text": f"❌ Video recording failed: {e}"}],
            }

    def replay_episode(
        self,
        repo_id: str,
        robot_name: str = None,
        episode: int = 0,
        root: str = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Replay actions from a LeRobotDataset episode in simulation.

        Loads a recorded episode, maps action keys to sim actuators,
        and steps physics at the original recorded frequency.

        Args:
            repo_id: HuggingFace dataset repo ID
            robot_name: Robot name in sim to replay on (first robot if None)
            episode: Episode index
            root: Local dataset root directory
            speed: Playback speed multiplier
        """
        if self._world is None:
            return {
                "status": "error",
                "content": [{"text": "❌ No world. Call create_world first."}],
            }

        # Resolve robot
        if robot_name is None:
            if not self._world.robots:
                return {
                    "status": "error",
                    "content": [{"text": "❌ No robots in sim. Add one first."}],
                }
            robot_name = next(iter(self._world.robots))

        robot = self._world.robots.get(robot_name)
        if robot is None:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{robot_name}' not found"}],
            }

        # Load dataset and resolve episode frame range
        try:
            from .dataset_recorder import load_lerobot_episode

            ds, episode_start, episode_length = load_lerobot_episode(
                repo_id, episode, root
            )
        except ImportError:
            return {
                "status": "error",
                "content": [{"text": "❌ lerobot not installed"}],
            }
        except (ValueError, Exception) as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ {e}"}],
            }

        # Replay loop
        dataset_fps = getattr(ds, "fps", 30)
        frame_interval = 1.0 / (dataset_fps * speed)
        model = self._world._model
        data = self._world._data
        n_actuators = model.nu
        frames_applied = 0
        start_time = time.time()

        for frame_idx in range(episode_length):
            step_start = time.time()

            frame = ds[episode_start + frame_idx]

            # Extract action and apply to sim
            if "action" in frame:
                action_vals = frame["action"]
                if hasattr(action_vals, "numpy"):
                    action_vals = action_vals.numpy()
                if hasattr(action_vals, "tolist"):
                    action_vals = action_vals.tolist()

                # Map to actuators (truncate or pad)
                for i in range(min(len(action_vals), n_actuators)):
                    data.ctrl[i] = float(action_vals[i])

            _mujoco.mj_step(model, data)
            frames_applied += 1

            # Maintain replay frequency
            elapsed = time.time() - step_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = time.time() - start_time
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"▶️ Replayed episode {episode} from {repo_id} on '{robot_name}'\n"
                        f"Frames: {frames_applied}/{episode_length} | Duration: {duration:.1f}s | Speed: {speed}x"
                    )
                },
                {
                    "json": {
                        "episode": episode,
                        "robot_name": robot_name,
                        "frames_applied": frames_applied,
                        "total_frames": episode_length,
                        "duration_s": round(duration, 2),
                        "speed": speed,
                    }
                },
            ],
        }

    def eval_policy(
        self,
        robot_name: str = None,
        policy_provider: str = "mock",
        instruction: str = "",
        n_episodes: int = 10,
        max_steps: int = 300,
        success_fn: str = None,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a policy over multiple episodes with success metrics.

        Runs the policy for n_episodes, resets between each, and computes
        success rate + avg reward.

        Args:
            robot_name: Robot name (first if None)
            policy_provider: Policy provider
            instruction: Task instruction
            n_episodes: Number of evaluation episodes
            max_steps: Max steps per episode
            success_fn: Optional success function name ('contact' = any contact = success)
            **policy_kwargs: Provider kwargs (pretrained_name_or_path, etc.)
        """
        if self._world is None:
            return {
                "status": "error",
                "content": [{"text": "❌ No world. Call create_world first."}],
            }

        if robot_name is None:
            if not self._world.robots:
                return {"status": "error", "content": [{"text": "❌ No robots"}]}
            robot_name = next(iter(self._world.robots))

        robot = self._world.robots.get(robot_name)
        if robot is None:
            return {
                "status": "error",
                "content": [{"text": f"❌ Robot '{robot_name}' not found"}],
            }

        # Create policy
        from strands_robots.policies import create_policy

        policy_instance = create_policy(policy_provider, **policy_kwargs)
        policy_instance.set_robot_state_keys(robot.joint_names)

        model = self._world._model
        data = self._world._data

        results = []
        for ep in range(n_episodes):
            # Reset
            _mujoco.mj_resetData(model, data)
            _mujoco.mj_forward(model, data)

            total_reward = 0.0
            success = False
            steps = 0

            for step in range(max_steps):
                # Get observation
                obs = self._get_sim_observation(robot_name=robot_name)

                # Get action from policy (may be sync or async)
                coro_or_result = policy_instance.get_actions(obs, instruction)
                actions = _resolve_coroutine(coro_or_result)

                if actions:
                    self._apply_sim_action(robot_name, actions[0])

                _mujoco.mj_step(model, data)
                steps += 1

                # Check success condition
                if success_fn == "contact":
                    contacts = []
                    for i in range(data.ncon):
                        c = data.contact[i]
                        if c.dist < 0:
                            contacts.append(i)
                    if contacts:
                        success = True
                        break

            results.append(
                {
                    "episode": ep,
                    "steps": steps,
                    "success": success,
                    "reward": total_reward,
                }
            )

        # Compute summary metrics
        n_success = sum(1 for r in results if r["success"])
        success_rate = n_success / max(n_episodes, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_episodes, 1)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📊 Evaluation: {policy_provider} on '{robot_name}'\n"
                        f"Episodes: {n_episodes} | Success: {n_success}/{n_episodes} ({success_rate:.1%})\n"
                        f"Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "n_episodes": n_episodes,
                        "n_success": n_success,
                        "avg_steps": round(avg_steps, 1),
                        "max_steps": max_steps,
                        "episodes": results,
                    }
                },
            ],
        }

    def open_viewer(self) -> Dict[str, Any]:
        """Open interactive 3D viewer window."""
        if self._world is None or self._world._model is None:
            return {
                "status": "error",
                "content": [{"text": "❌ No simulation to view."}],
            }

        if _mujoco_viewer is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "❌ mujoco.viewer not available. Install: pip install mujoco"
                    }
                ],
            }

        if self._viewer_handle is not None:
            return {
                "status": "success",
                "content": [{"text": "👁️ Viewer already open."}],
            }

        try:
            self._viewer_handle = _mujoco_viewer.launch_passive(
                self._world._model, self._world._data
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": "👁️ Interactive viewer opened. Close window or use action='close_viewer'."
                    }
                ],
            }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Viewer failed: {e}"}]}

    def _close_viewer(self):
        """Internal: close viewer if open."""
        if self._viewer_handle is not None:
            try:
                self._viewer_handle.close()
            except Exception:
                pass
            self._viewer_handle = None

    def close_viewer(self) -> Dict[str, Any]:
        self._close_viewer()
        return {"status": "success", "content": [{"text": "👁️ Viewer closed."}]}

    # -------------------------------------------------------------------
    # URDF Registry Management
    # -------------------------------------------------------------------

    def list_urdfs_action(self) -> Dict[str, Any]:
        """List all available robot models (Menagerie MJCF + custom URDF)."""
        text = list_available_models()
        return {"status": "success", "content": [{"text": text}]}

    def register_urdf_action(self, data_config: str, urdf_path: str) -> Dict[str, Any]:
        """Register a URDF/MJCF for a data_config name."""
        register_urdf(data_config, urdf_path)
        resolved = resolve_model(data_config)
        return {
            "status": "success",
            "content": [
                {
                    "text": f"📋 Registered '{data_config}' → {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"
                }
            ],
        }

    # -------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------

    def get_features(self) -> Dict[str, Any]:
        """Return simulation introspection: robot joints, actuators, cameras.

        Gives the agent full visibility into the loaded model so it can
        build correct action dicts and choose cameras for rendering.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model = self._world._model

        features: Dict[str, Any] = {
            "n_bodies": model.nbody,
            "n_joints": model.njnt,
            "n_actuators": model.nu,
            "n_cameras": model.ncam,
            "timestep": model.opt.timestep,
        }

        # Joint names
        joint_names = []
        for i in range(model.njnt):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_names.append(name)
        features["joint_names"] = joint_names

        # Actuator names
        actuator_names = []
        for i in range(model.nu):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                actuator_names.append(name)
        features["actuator_names"] = actuator_names

        # Camera names
        camera_names = []
        for i in range(model.ncam):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
            if name:
                camera_names.append(name)
        features["camera_names"] = camera_names

        # Per-robot breakdown
        robots_info = {}
        for rname, robot in self._world.robots.items():
            robots_info[rname] = {
                "joint_names": robot.joint_names,
                "n_joints": len(robot.joint_names),
                "n_actuators": len(robot.actuator_ids),
                "data_config": robot.data_config,
                "source": os.path.basename(robot.urdf_path),
            }
        features["robots"] = robots_info

        # Build human-readable text
        lines = [
            "🔍 Simulation Features",
            f"🦴 Joints ({model.njnt}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"⚡ Actuators ({model.nu}): {', '.join(actuator_names[:12])}{'...' if len(actuator_names) > 12 else ''}",
            f"📷 Cameras ({model.ncam}): {', '.join(camera_names) if camera_names else 'none (free camera only)'}",
            f"⏱️ Timestep: {model.opt.timestep}s ({1/model.opt.timestep:.0f}Hz)",
        ]
        for rname, rinfo in robots_info.items():
            lines.append(
                f"🤖 {rname}: {rinfo['n_joints']} joints, {rinfo['n_actuators']} actuators ({rinfo['source']})"
            )

        return {
            "status": "success",
            "content": [
                {"text": "\n".join(lines)},
                {"json": {"features": features}},
            ],
        }

    # -------------------------------------------------------------------
    # AgentTool Interface
    # -------------------------------------------------------------------

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "simulation"

    @property
    def tool_spec(self) -> ToolSpec:
        return {
            "name": self.tool_name_str,
            "description": (
                "Programmatic MuJoCo simulation environment. Create worlds, add robots from URDF "
                "(direct path or auto-resolve from data_config name), add objects, run VLA policies, "
                "render cameras, record trajectories, domain randomize. "
                "Same Policy ABC as real robot control — sim ↔ real with zero code changes. "
                "10 embodiment configs pre-registered (so100, unitree_g1, panda, etc.). "
                "Actions: create_world, load_scene, reset, get_state, destroy, "
                "add_robot, remove_robot, list_robots, get_robot_state, "
                "add_object, remove_object, move_object, list_objects, "
                "add_camera, remove_camera, "
                "run_policy, start_policy, stop_policy, "
                "render, render_depth, get_contacts, "
                "step, set_gravity, set_timestep, "
                "randomize, "
                "start_recording, stop_recording, get_recording_status, "
                "open_viewer, close_viewer, "
                "list_urdfs, register_urdf, get_features"
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform",
                            "enum": [
                                "create_world",
                                "load_scene",
                                "reset",
                                "get_state",
                                "destroy",
                                "add_robot",
                                "remove_robot",
                                "list_robots",
                                "get_robot_state",
                                "add_object",
                                "remove_object",
                                "move_object",
                                "list_objects",
                                "add_camera",
                                "remove_camera",
                                "run_policy",
                                "start_policy",
                                "stop_policy",
                                "render",
                                "render_depth",
                                "get_contacts",
                                "step",
                                "set_gravity",
                                "set_timestep",
                                "randomize",
                                "start_recording",
                                "stop_recording",
                                "get_recording_status",
                                "record_video",
                                "open_viewer",
                                "close_viewer",
                                "list_urdfs",
                                "register_urdf",
                                "get_features",
                                "replay_episode",
                                "eval_policy",
                            ],
                        },
                        "scene_path": {
                            "type": "string",
                            "description": "Path to MJCF/URDF scene file",
                        },
                        "timestep": {"type": "number"},
                        "gravity": {"type": "array", "items": {"type": "number"}},
                        "ground_plane": {"type": "boolean"},
                        "urdf_path": {
                            "type": "string",
                            "description": "Path to URDF/MJCF file",
                        },
                        "robot_name": {"type": "string"},
                        "data_config": {
                            "type": "string",
                            "description": "Data config name (auto-resolves URDF)",
                        },
                        "name": {"type": "string", "description": "Object/camera name"},
                        "shape": {
                            "type": "string",
                            "enum": [
                                "box",
                                "sphere",
                                "cylinder",
                                "capsule",
                                "mesh",
                                "plane",
                            ],
                        },
                        "position": {"type": "array", "items": {"type": "number"}},
                        "orientation": {"type": "array", "items": {"type": "number"}},
                        "size": {"type": "array", "items": {"type": "number"}},
                        "color": {"type": "array", "items": {"type": "number"}},
                        "mass": {"type": "number"},
                        "is_static": {"type": "boolean"},
                        "mesh_path": {"type": "string"},
                        "target": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Camera target point",
                        },
                        "fov": {
                            "type": "number",
                            "description": "Camera field of view",
                        },
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider name (e.g. groot, lerobot_async, lerobot_local, dreamgen, mock)",
                        },
                        "instruction": {"type": "string"},
                        "duration": {"type": "number"},
                        "policy_port": {"type": "integer"},
                        "policy_host": {"type": "string"},
                        "model_path": {"type": "string"},
                        "action_horizon": {"type": "integer"},
                        "control_frequency": {"type": "number"},
                        "camera_name": {"type": "string"},
                        "n_steps": {"type": "integer"},
                        "output_path": {
                            "type": "string",
                            "description": "Trajectory/video export path",
                        },
                        "fps": {
                            "type": "integer",
                            "description": "Video frames per second (for record_video)",
                        },
                        "pretrained_name_or_path": {
                            "type": "string",
                            "description": "HuggingFace model ID for lerobot_local",
                        },
                        # Domain randomization
                        "randomize_colors": {"type": "boolean"},
                        "randomize_lighting": {"type": "boolean"},
                        "randomize_physics": {"type": "boolean"},
                        "randomize_positions": {"type": "boolean"},
                        "position_noise": {"type": "number"},
                        "seed": {"type": "integer", "description": "Random seed"},
                        # Replay and eval
                        "repo_id": {
                            "type": "string",
                            "description": "HuggingFace dataset repo ID (for start_recording: creates LeRobotDataset; for replay_episode: loads dataset)",
                        },
                        "push_to_hub": {
                            "type": "boolean",
                            "description": "Auto-push dataset to HuggingFace Hub on stop_recording",
                        },
                        "vcodec": {
                            "type": "string",
                            "description": "Video codec for dataset recording (h264, hevc, libsvtav1)",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task description for dataset recording",
                        },
                        "episode": {
                            "type": "integer",
                            "description": "Episode index for replay_episode",
                        },
                        "root": {
                            "type": "string",
                            "description": "Local dataset root directory",
                        },
                        "speed": {
                            "type": "number",
                            "description": "Replay speed multiplier (1.0 = original)",
                        },
                        "n_episodes": {
                            "type": "integer",
                            "description": "Number of eval episodes",
                        },
                        "max_steps": {
                            "type": "integer",
                            "description": "Max steps per eval episode",
                        },
                        "success_fn": {
                            "type": "string",
                            "description": "Success function ('contact')",
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            result = self._dispatch_action(input_data.get("action", ""), input_data)
            yield ToolResultEvent({"toolUseId": tool_use_id, **result})
        except Exception as e:
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "status": "error",
                    "content": [{"text": f"❌ Sim error: {e}"}],
                }
            )

    def _dispatch_action(self, action: str, d: Dict[str, Any]) -> Dict[str, Any]:
        """Route action to method via dispatch table."""
        handler = self._ACTION_DISPATCH.get(action)
        if handler is None:
            return {
                "status": "error",
                "content": [{"text": f"❌ Unknown action: {action}"}],
            }
        return handler(self, d)

    @staticmethod
    def _extract_policy_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        """Extract optional policy kwargs from dispatch dict."""
        return {
            k: d[k]
            for k in ("policy_port", "policy_host", "model_path", "server_address", "policy_type")
            if k in d
        }

    @staticmethod
    def _extract_optional(d: Dict[str, Any], *keys: str) -> Dict[str, Any]:
        """Extract keys from d only if their values are not None."""
        return {k: d[k] for k in keys if d.get(k) is not None}

    # --- Individual dispatch handlers ---

    def _do_create_world(self, d):
        kw = {"ground_plane": d.get("ground_plane", True)}
        kw.update(self._extract_optional(d, "timestep", "gravity"))
        return self.create_world(**kw)

    def _do_load_scene(self, d):
        return self.load_scene(d.get("scene_path", ""))

    def _do_reset(self, d):
        return self.reset()

    def _do_get_state(self, d):
        return self.get_state()

    def _do_destroy(self, d):
        return self.destroy()

    def _do_add_robot(self, d):
        kw = {"name": d.get("robot_name", d.get("name", "robot_0"))}
        kw.update(self._extract_optional(d, "urdf_path", "data_config", "position", "orientation"))
        return self.add_robot(**kw)

    def _do_remove_robot(self, d):
        return self.remove_robot(d.get("robot_name", d.get("name", "")))

    def _do_list_robots(self, d):
        return self.list_robots()

    def _do_get_robot_state(self, d):
        return self.get_robot_state(d.get("robot_name", ""))

    def _do_add_object(self, d):
        return self.add_object(
            name=d.get("name", ""),
            shape=d.get("shape", "box"),
            position=d.get("position"),
            orientation=d.get("orientation"),
            size=d.get("size"),
            color=d.get("color"),
            mass=d.get("mass", 0.1),
            is_static=d.get("is_static", False),
            mesh_path=d.get("mesh_path"),
        )

    def _do_remove_object(self, d):
        return self.remove_object(d.get("name", ""))

    def _do_move_object(self, d):
        return self.move_object(d.get("name", ""), d.get("position"), d.get("orientation"))

    def _do_list_objects(self, d):
        return self.list_objects()

    def _do_add_camera(self, d):
        return self.add_camera(
            name=d.get("name", "cam"),
            position=d.get("position"),
            target=d.get("target"),
            fov=d.get("fov", 60.0),
            width=d.get("width", 640),
            height=d.get("height", 480),
        )

    def _do_remove_camera(self, d):
        return self.remove_camera(d.get("name", ""))

    def _do_run_policy(self, d):
        return self.run_policy(
            robot_name=d.get("robot_name", ""),
            policy_provider=d.get("policy_provider", "mock"),
            instruction=d.get("instruction", ""),
            duration=d.get("duration", 10.0),
            action_horizon=d.get("action_horizon", 8),
            control_frequency=d.get("control_frequency", 50.0),
            fast_mode=d.get("fast_mode", False),
            **self._extract_policy_kwargs(d),
        )

    def _do_start_policy(self, d):
        return self.start_policy(
            robot_name=d.get("robot_name", ""),
            policy_provider=d.get("policy_provider", "mock"),
            instruction=d.get("instruction", ""),
            duration=d.get("duration", 10.0),
            fast_mode=d.get("fast_mode", False),
            **self._extract_policy_kwargs(d),
        )

    def _do_stop_policy(self, d):
        rn = d.get("robot_name", "")
        if self._world and rn in self._world.robots:
            self._world.robots[rn].policy_running = False
            return {"status": "success", "content": [{"text": f"🛑 Stopped on '{rn}'"}]}
        return {"status": "error", "content": [{"text": f"❌ '{rn}' not found."}]}

    def _do_render(self, d):
        return self.render(d.get("camera_name", "default"), d.get("width"), d.get("height"))

    def _do_render_depth(self, d):
        return self.render_depth(d.get("camera_name", "default"), d.get("width"), d.get("height"))

    def _do_get_contacts(self, d):
        return self.get_contacts()

    def _do_step(self, d):
        return self.step(d.get("n_steps", 1))

    def _do_set_gravity(self, d):
        return self.set_gravity(d.get("gravity", [0, 0, -9.81]))

    def _do_set_timestep(self, d):
        return self.set_timestep(d.get("timestep", 0.002))

    def _do_randomize(self, d):
        return self.randomize(
            randomize_colors=d.get("randomize_colors", True),
            randomize_lighting=d.get("randomize_lighting", True),
            randomize_physics=d.get("randomize_physics", False),
            randomize_positions=d.get("randomize_positions", False),
            position_noise=d.get("position_noise", 0.02),
            seed=d.get("seed"),
        )

    def _do_start_recording(self, d):
        kw = {
            "task": d.get("instruction", d.get("task", "")),
            "fps": d.get("fps", 30),
            "push_to_hub": d.get("push_to_hub", False),
            "vcodec": d.get("vcodec", "libsvtav1"),
        }
        kw.update(self._extract_optional(d, "repo_id", "root"))
        return self.start_recording(**kw)

    def _do_stop_recording(self, d):
        return self.stop_recording(d.get("output_path"))

    def _do_get_recording_status(self, d):
        return self.get_recording_status()

    def _do_record_video(self, d):
        kw = {
            k: d[k]
            for k in (
                "policy_port", "policy_host", "model_path",
                "server_address", "policy_type", "pretrained_name_or_path",
            )
            if k in d
        }
        return self.record_video(
            robot_name=d.get("robot_name", ""),
            policy_provider=d.get("policy_provider", "lerobot_local"),
            instruction=d.get("instruction", ""),
            duration=d.get("duration", 10.0),
            fps=d.get("fps", 30),
            camera_name=d.get("camera_name"),
            width=d.get("width", 640),
            height=d.get("height", 480),
            output_path=d.get("output_path"),
            **kw,
        )

    def _do_open_viewer(self, d):
        return self.open_viewer()

    def _do_close_viewer(self, d):
        return self.close_viewer()

    def _do_list_urdfs(self, d):
        return self.list_urdfs_action()

    def _do_register_urdf(self, d):
        return self.register_urdf_action(d.get("data_config", ""), d.get("urdf_path", ""))

    def _do_get_features(self, d):
        return self.get_features()

    def _do_replay_episode(self, d):
        repo_id = d.get("repo_id")
        if not repo_id:
            return {"status": "error", "content": [{"text": "❌ repo_id required for replay_episode"}]}
        return self.replay_episode(
            repo_id=repo_id,
            robot_name=d.get("robot_name"),
            episode=d.get("episode", 0),
            root=d.get("root"),
            speed=d.get("speed", 1.0),
        )

    def _do_eval_policy(self, d):
        return self.eval_policy(
            robot_name=d.get("robot_name"),
            policy_provider=d.get("policy_provider", "mock"),
            instruction=d.get("instruction", ""),
            n_episodes=d.get("n_episodes", 10),
            max_steps=d.get("max_steps", 300),
            success_fn=d.get("success_fn"),
            **{
                k: v
                for k, v in d.items()
                if k.startswith("pretrained") or k in ("policy_type", "device")
            },
        )

    _ACTION_DISPATCH: Dict[str, Callable] = {
        "create_world": _do_create_world,
        "load_scene": _do_load_scene,
        "reset": _do_reset,
        "get_state": _do_get_state,
        "destroy": _do_destroy,
        "add_robot": _do_add_robot,
        "remove_robot": _do_remove_robot,
        "list_robots": _do_list_robots,
        "get_robot_state": _do_get_robot_state,
        "add_object": _do_add_object,
        "remove_object": _do_remove_object,
        "move_object": _do_move_object,
        "list_objects": _do_list_objects,
        "add_camera": _do_add_camera,
        "remove_camera": _do_remove_camera,
        "run_policy": _do_run_policy,
        "start_policy": _do_start_policy,
        "stop_policy": _do_stop_policy,
        "render": _do_render,
        "render_depth": _do_render_depth,
        "get_contacts": _do_get_contacts,
        "step": _do_step,
        "set_gravity": _do_set_gravity,
        "set_timestep": _do_set_timestep,
        "randomize": _do_randomize,
        "start_recording": _do_start_recording,
        "stop_recording": _do_stop_recording,
        "get_recording_status": _do_get_recording_status,
        "record_video": _do_record_video,
        "open_viewer": _do_open_viewer,
        "close_viewer": _do_close_viewer,
        "list_urdfs": _do_list_urdfs,
        "register_urdf": _do_register_urdf,
        "get_features": _do_get_features,
        "replay_episode": _do_replay_episode,
        "eval_policy": _do_eval_policy,
    }

    def cleanup(self):
        if hasattr(self, "mesh") and self.mesh:
            self.mesh.stop()
        if self._world:
            for r in self._world.robots.values():
                r.policy_running = False
            self._world = None
        self._close_viewer()
        self._executor.shutdown(wait=False)
        self._shutdown_event.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# Convenience alias — allows ``from strands_robots.simulation import simulation``
simulation = Simulation
