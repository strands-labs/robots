"""Dataclasses for simulation state.

These dataclasses provide a backend-independent typed state representation
consumed by simulation engine implementations (e.g. MuJoCo, Isaac Sim,
PyBullet).

They enable:
    - Type-safe state tracking across simulation steps.
    - Serialisation for checkpoints and trajectory recording.
    - A backend-independent interface for agent tools.

They are defined alongside the ``SimEngine`` ABC because its method
signatures reference them (e.g. ``create_world() → SimWorld``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SimStatus(Enum):
    """Simulation execution status."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimRobot:
    """A robot instance within the simulation.

    ``mesh`` / ``peer_id`` (post-PR #101): when the parent ``Simulation`` is
    itself attached to a Zenoh mesh, every robot added via ``add_robot``
    auto-joins as its own peer so the agent can address it directly
    (e.g. ``robot_mesh tell target=<peer_id>``) instead of having to talk to
    the sim container and then route by robot name. Both fields stay
    ``None`` / ``""`` for stand-alone sims that are not on a mesh.
    """

    name: str
    urdf_path: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz quat
    data_config: str | None = None
    body_id: int = -1
    joint_ids: list[int] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    actuator_ids: list[int] = field(default_factory=list)
    namespace: str = ""
    policy_running: bool = False
    policy_steps: int = 0
    policy_instruction: str = ""
    # Per-robot mesh peer. Populated by ``Simulation.add_robot`` when the
    # parent sim is on a mesh; ``None`` otherwise. Carried as ``Any`` to
    # avoid a hard import dependency on ``strands_robots.mesh.Mesh`` from
    # this backend-independent module.
    mesh: Any = None
    peer_id: str = ""


@dataclass
class SimObject:
    """An object in the simulation scene."""

    name: str
    shape: str  # "box", "sphere", "cylinder", "capsule", "mesh"
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    size: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05])
    color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])  # RGBA
    mass: float = 0.1
    mesh_path: str | None = None
    body_id: int = -1
    is_static: bool = False
    _original_position: list[float] = field(default_factory=list)
    _original_color: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._original_position = list(self.position)
        self._original_color = list(self.color)


@dataclass
class SimCamera:
    """A camera in the simulation.

    ``origin_robot`` (post-PR #85): when the camera was discovered inside a
    robot's URDF during ``add_robot``, this is set to the robot's name so the
    scene builder knows NOT to re-add the camera at the top level (it'll be
    re-introduced via ``spec.attach(robot_spec)``). For user-added cameras
    (via the ``add_camera`` tool action) this stays empty.
    """

    name: str
    position: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fov: float = 60.0
    width: int = 640
    height: int = 480
    camera_id: int = -1
    origin_robot: str = ""
    #: Optional body name to mount the camera ON (e.g. a robot's gripper for
    #: a wrist cam). When set, ``position``/``target`` are interpreted in that
    #: body's LOCAL frame and the camera tracks the body as it moves --
    #: matching a real wrist-mounted camera. Empty = world-fixed camera.
    parent_body: str = ""


@dataclass
class TrajectoryStep:
    """A single step in a recorded trajectory."""

    timestamp: float
    sim_time: float
    robot_name: str
    observation: dict[str, Any]
    action: dict[str, Any]
    instruction: str = ""


@dataclass
class SimWorld:
    """Complete simulation world state.

    Backend-independent state with engine-specific internals kept in three
    escape hatches, each with a distinct role so backend implementers know
    which to use:

    * ``_model``: the physics engine's **core model handle** - the single
      compiled/loaded representation of the scene (e.g. ``mujoco.MjModel``,
      Isaac's ``Scene``, PyBullet's body registry). Every backend has one.
    * ``_data``: the physics engine's **core simulation state handle** -
      the mutable per-step state companion to ``_model``
      (e.g. ``mujoco.MjData``, Isaac's ``World``). Every backend has one.
    * ``_backend_state``: a **catch-all dict** for everything else the
      backend needs to persist - generated XML, temp dirs, recording
      buffers, caches, etc. Prefer this over adding new fields here.

    All three are typed ``Any``/``dict`` so nothing leaks engine-specific
    types into this base module.
    """

    robots: dict[str, SimRobot] = field(default_factory=dict)
    objects: dict[str, SimObject] = field(default_factory=dict)
    cameras: dict[str, SimCamera] = field(default_factory=dict)
    timestep: float = 0.002  # 500Hz physics
    gravity: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    ground_plane: bool = True
    status: SimStatus = SimStatus.IDLE
    sim_time: float = 0.0
    step_count: int = 0
    # Engine core handles - set after the backend builds the world.
    # Use these for the primary model/state objects only; put everything
    # else in ``_backend_state`` below.
    _model: Any = None  # Engine-specific model handle (e.g. MjModel, Scene)
    _data: Any = None  # Engine-specific data handle (e.g. MjData, World)
    # Catch-all for backend-specific state that isn't the core model/data.
    # Examples: generated XML strings, temp dirs, recording buffers
    # (``_recording``, ``_trajectory``, ``_dataset_recorder``), caches, etc.
    # Prefer this over adding new fields to ``SimWorld``.
    _backend_state: dict[str, Any] = field(default_factory=dict)
    # Physics state checkpoints (used by save_state/restore_state in PR #85).
    # Kept as a top-level field - requested by @yinsong1986 during review to
    # avoid monkey-patching when ``reset()`` creates a fresh ``SimWorld``.
    _checkpoints: dict[str, Any] = field(default_factory=dict)
