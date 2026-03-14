"""Dataclasses for simulation state."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz quat
    data_config: Optional[str] = None
    body_id: int = -1
    joint_ids: List[int] = field(default_factory=list)
    joint_names: List[str] = field(default_factory=list)
    actuator_ids: List[int] = field(default_factory=list)
    namespace: str = ""
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
    _robot_base_xml: str = ""
    # Trajectory recording
    _recording: bool = False
    _trajectory: List[TrajectoryStep] = field(default_factory=list)
    # LeRobotDataset recorder
    _dataset_recorder: Any = None
    # Temp directory for scene composition
    _tmpdir: Any = None
