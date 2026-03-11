"""Universal Robot Control with Policy Abstraction for Any VLA Provider

This module provides a clean robot interface that works with any LeRobot-compatible
robot and any VLA provider through the Policy abstraction.

Features:
- Async robot task execution with real-time status reporting
- Non-blocking operations - robot moves while tool returns status
- Stop functionality to interrupt running tasks
- Connection state management with proper error handling
- Policy abstraction for any VLA provider (GR00T, LeRobot Async, DreamGen, etc.)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional, Union

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from strands_robots._async_utils import _resolve_coroutine, _run_async
from strands_robots.registry import build_policy_kwargs

from .policies import Policy, create_policy

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot as LeRobotRobot

logger = logging.getLogger(__name__)


def _import_lerobot():
    """Lazy-import lerobot modules. Raises ImportError with a helpful message."""
    try:
        from lerobot.cameras.opencv.configuration_opencv import (
            OpenCVCameraConfig as _OpenCVCameraConfig,
        )
        from lerobot.robots.config import RobotConfig as _RobotConfig
        from lerobot.robots.robot import Robot as _LeRobotRobot
        from lerobot.robots.utils import make_robot_from_config as _make_robot

        return _OpenCVCameraConfig, _RobotConfig, _LeRobotRobot, _make_robot
    except ImportError as e:
        raise ImportError(
            f"LeRobot is required for Robot but could not be imported: {e}. "
            "Install it with: pip install lerobot"
        ) from e


class TaskStatus(Enum):
    """Robot task execution status"""

    IDLE = "idle"
    CONNECTING = "connecting"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RobotTaskState:
    """Robot task execution state — guarded by _task_lock for thread safety."""

    status: TaskStatus = TaskStatus.IDLE
    instruction: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    step_count: int = 0
    error_message: str = ""
    task_future: Optional[Future] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        """Thread-safe batch update of multiple fields."""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        """Thread-safe snapshot of all fields for consistent reads."""
        with self._lock:
            return {
                "status": self.status,
                "instruction": self.instruction,
                "start_time": self.start_time,
                "duration": self.duration,
                "step_count": self.step_count,
                "error_message": self.error_message,
            }


class Robot(AgentTool):
    """Universal robot control with async task execution and status reporting."""

    def __init__(
        self,
        tool_name: str,
        robot: Union[LeRobotRobot, RobotConfig, str],
        cameras: Optional[Dict[str, Dict[str, Any]]] = None,
        action_horizon: int = 8,
        data_config: Union[str, Any, None] = None,
        control_frequency: float = 50.0,
        mesh: bool = True,
        peer_id: str = None,
        **kwargs,
    ):
        """Initialize Robot with async capabilities.

        Args:
            tool_name: Name for this robot tool
            robot: LeRobot Robot instance, RobotConfig, or robot type string
            cameras: Camera configuration dict:
                {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}
            action_horizon: Actions per inference step
            data_config: Data configuration (for GR00T compatibility)
            control_frequency: Control loop frequency in Hz (default: 50Hz)
            **kwargs: Robot-specific parameters (port, etc.)
        """
        super().__init__()

        self.tool_name_str = tool_name
        self.action_horizon = action_horizon
        self.data_config = data_config
        self.control_frequency = control_frequency
        self.action_sleep_time = 1.0 / control_frequency  # Time between actions

        # Task execution state
        self._task_state = RobotTaskState()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{tool_name}_executor"
        )
        self._shutdown_event = threading.Event()

        # Initialize robot using lerobot's abstraction
        self.robot = self._initialize_robot(robot, cameras, **kwargs)

        logger.info(f"🤖 {tool_name} initialized with async capabilities")
        logger.info(
            f"📱 Robot: {self.robot.name} (type: {getattr(self.robot, 'robot_type', 'unknown')})"
        )
        logger.info(
            f"⏱️ Control frequency: {control_frequency}Hz ({self.action_sleep_time*1000:.1f}ms per action)"
        )

        # Get camera info if available
        if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
            cameras_list = list(self.robot.config.cameras.keys())
            logger.info(f"📹 Cameras: {cameras_list}")

        # Get features from LeRobot robot if available
        self._observation_features = getattr(self.robot, "observation_features", {})
        self._action_features = getattr(self.robot, "action_features", {})

        if data_config:
            logger.info(f"⚙️ Data config: {data_config}")

        # Zenoh mesh — every Robot is a peer by default
        try:
            from strands_robots.zenoh_mesh import init_mesh

            self.mesh = init_mesh(self, peer_id=peer_id, peer_type="robot", mesh=mesh)
        except Exception as e:
            logger.debug(f"Mesh init skipped: {e}")
            self.mesh = None

    def _initialize_robot(
        self,
        robot: Union[LeRobotRobot, RobotConfig, str],
        cameras: Optional[Dict[str, Dict[str, Any]]],
        **kwargs,
    ) -> LeRobotRobot:
        """Initialize LeRobot robot instance using native lerobot patterns."""
        _OpenCVCameraConfig, _RobotConfig, _LeRobotRobot, _make_robot = (
            _import_lerobot()
        )

        # Direct robot instance - use as-is
        if isinstance(robot, _LeRobotRobot):
            return robot

        # Robot config - use lerobot's factory
        elif isinstance(robot, _RobotConfig):
            return _make_robot(robot)

        # Robot type string - create config and use lerobot's factory
        elif isinstance(robot, str):
            config = self._create_minimal_config(robot, cameras, **kwargs)
            return _make_robot(config)

        else:
            raise ValueError(
                f"Unsupported robot type: {type(robot)}. "
                f"Expected LeRobot Robot instance, RobotConfig, or robot type string."
            )

    def _create_minimal_config(
        self, robot_type: str, cameras: Optional[Dict[str, Dict[str, Any]]], **kwargs
    ) -> RobotConfig:
        """Create minimal robot config by auto-discovering LeRobot config classes.

        No hardcoded config_mapping — scans lerobot.robots submodules for
        *Config classes that are RobotConfig subclasses. Any robot LeRobot
        supports, we support.
        """
        _OpenCVCameraConfig, _RobotConfig, _LeRobotRobot, _make_robot = (
            _import_lerobot()
        )

        # Convert cameras to lerobot format
        camera_configs = {}
        if cameras:
            for name, config in cameras.items():
                if config.get("type", "opencv") == "opencv":
                    camera_configs[name] = _OpenCVCameraConfig(
                        index_or_path=config["index_or_path"],
                        fps=config.get("fps", 30),
                        width=config.get("width", 640),
                        height=config.get("height", 480),
                        rotation=config.get("rotation", 0),
                        color_mode=config.get("color_mode", "rgb"),
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {config.get('type')}")

        # Auto-discover: find the right Config class from LeRobot
        ConfigClass = self._resolve_robot_config_class(robot_type)

        # Build config data
        config_data = {
            "id": self.tool_name_str,
            "cameras": camera_configs,
        }

        # Forward common kwargs that the config might accept
        for key in ["port", "calibration_dir", "mock", "use_degrees"]:
            if key in kwargs:
                config_data[key] = kwargs[key]

        # Also try to forward any kwargs that match the config's dataclass fields
        if hasattr(ConfigClass, "__dataclass_fields__"):
            for field_name in ConfigClass.__dataclass_fields__:
                if field_name in kwargs and field_name not in config_data:
                    config_data[field_name] = kwargs[field_name]

        try:
            return ConfigClass(**config_data)
        except TypeError as e:
            # If required fields are missing, give a helpful error
            raise ValueError(
                f"Failed to create {ConfigClass.__name__} for robot type '{robot_type}': {e}. "
                f"Hint: check required fields for {ConfigClass.__name__}."
            )

    @staticmethod
    def _resolve_robot_config_class(robot_type: str):
        """Resolve a robot type string to its LeRobot Config class.

        Resolution order:
        1. Scan lerobot.robots submodules for *Config classes
        2. Match by class name convention (e.g. "so100_follower" -> SO100FollowerConfig)
        3. Match by config.type property (instantiate and check)

        No hardcoding — any Config class LeRobot ships, we find.
        """
        import importlib
        import pkgutil

        import lerobot.robots as lr_robots

        # Normalize the robot_type for matching
        robot_type_lower = robot_type.lower().replace("-", "_")

        # Build expected class name: "so100_follower" -> "So100FollowerConfig" or "SO100FollowerConfig"
        # We'll do case-insensitive matching
        expected_class_lower = robot_type_lower.replace("_", "") + "config"

        candidates = []

        _, _RobotConfig, _, _ = _import_lerobot()

        for importer, modname, ispkg in pkgutil.iter_modules(lr_robots.__path__):
            if modname in ("config", "robot", "utils"):
                continue
            try:
                mod = importlib.import_module(f"lerobot.robots.{modname}")
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name)
                    if (
                        isinstance(obj, type)
                        and attr_name.endswith("Config")
                        and issubclass(obj, _RobotConfig)
                        and obj is not _RobotConfig
                    ):
                        # Strategy 1: Class name match (case-insensitive)
                        if attr_name.lower().replace("_", "") == expected_class_lower:
                            return obj

                        candidates.append((attr_name, obj, modname))
            except Exception:
                continue

        # Strategy 2: Check if any candidate's .type property matches robot_type
        for class_name, cls, modname in candidates:
            try:
                # Some configs need required args — try common patterns
                for test_kwargs in [
                    {"id": "test"},
                    {"id": "test", "port": "/dev/null"},
                    {},
                ]:
                    try:
                        instance = cls(**test_kwargs)
                        if (
                            hasattr(instance, "type")
                            and instance.type == robot_type_lower
                        ):
                            return cls
                        break  # Instance created, type didn't match, move on
                    except TypeError:
                        continue  # Try next kwargs pattern
            except Exception:
                continue

        # Strategy 3: Fuzzy match — module name contains robot_type parts
        for class_name, cls, modname in candidates:
            # e.g. robot_type="so100_follower", modname="so_follower"
            type_parts = robot_type_lower.split("_")
            if all(
                part in modname.lower() or part in class_name.lower()
                for part in type_parts
                if len(part) > 2
            ):
                return cls

        available = sorted(set(name for name, _, _ in candidates))
        raise ValueError(
            f"Could not find LeRobot config for robot type '{robot_type}'. "
            f"Available config classes: {available}. "
            f"Check `lerobot.robots` submodules."
        )

    async def _get_policy(
        self,
        policy_provider: str = "groot",
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        model_path: Optional[str] = None,
        server_address: Optional[str] = None,
        policy_type: Optional[str] = None,
        **policy_kwargs,
    ) -> Policy:
        """Create policy on-the-fly from invocation parameters.

        Uses ``registry/policies.json`` to map generic parameters
        (policy_port, model_path, ...) to provider-specific kwargs.
        No hardcoded if/elif chains — adding a new provider is a
        JSON edit + a Python module.

        Args:
            policy_provider: Provider name (groot, lerobot_async, dreamgen, mock, ...)
            policy_port: Port for remote services (groot, lerobot_async).
            policy_host: Hostname (default: "localhost").
            model_path: Model path or HF ID (dreamgen, lerobot_local).
            server_address: Full gRPC address (lerobot_async).
            policy_type: Sub-type (pi0, act, smolvla, ...).
            **policy_kwargs: Additional provider-specific parameters.
        """
        policy_config = build_policy_kwargs(
            provider=policy_provider,
            policy_port=policy_port,
            policy_host=policy_host,
            model_path=model_path,
            server_address=server_address,
            policy_type=policy_type,
            data_config=self.data_config,
            **policy_kwargs,
        )

        return create_policy(policy_provider, **policy_config)

    async def _connect_robot(self) -> tuple[bool, str]:
        """Connect to robot hardware with proper error handling.

        Returns:
            tuple[bool, str]: (success, error_message) - error_message is empty on success
        """
        try:
            # Import lerobot exceptions
            from lerobot.utils.errors import DeviceAlreadyConnectedError

            # Check if already connected
            if self.robot.is_connected:
                logger.info(f"✅ {self.robot} already connected")
                return True, ""

            logger.info(f"🔌 Connecting to {self.robot}...")

            # Handle robot connection using lerobot's error handling patterns
            try:
                if not self.robot.is_connected:
                    await asyncio.to_thread(
                        self.robot.connect, False
                    )  # calibrate=False

            except DeviceAlreadyConnectedError:
                # This is expected and fine - robot is already connected
                logger.info(f"✅ {self.robot} was already connected")

            except Exception:
                raise

            # Final connection check
            if not self.robot.is_connected:
                error_msg = f"Failed to connect to {self.robot}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            # Check robot calibration
            if hasattr(self.robot, "is_calibrated") and not self.robot.is_calibrated:
                error_msg = (
                    f"Robot {self.robot} is not calibrated. Please calibrate the robot manually"
                    " first using LeRobot's calibration process (lerobot-calibrate)"
                )
                logger.error(f"❌ {error_msg}")
                return False, error_msg

            logger.info(f"✅ {self.robot} connected and ready")
            return True, ""

        except Exception as e:
            error_msg = f"Robot connection failed: {e}. Ensure robot is calibrated and accessible on the specified port"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    async def _initialize_policy(self, policy: Policy) -> bool:
        """Initialize policy with robot state keys."""
        try:
            if self._action_features:
                robot_state_keys = [
                    k for k, v in self._action_features.items() if v is float
                ]
            else:
                # Fallback: get from observation
                test_obs = await asyncio.to_thread(self.robot.get_observation)
                camera_keys = []
                if hasattr(self.robot, "config") and hasattr(
                    self.robot.config, "cameras"
                ):
                    camera_keys = list(self.robot.config.cameras.keys())
                robot_state_keys = [k for k in test_obs.keys() if k not in camera_keys]

            # Set robot state keys in policy
            policy.set_robot_state_keys(robot_state_keys)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize policy: {e}")
            return False

    async def _execute_task_async(
        self,
        instruction: str,
        policy_provider: str = "groot",
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        duration: float = 30.0,
        **policy_kwargs,
    ) -> None:
        """Execute robot task in background thread (internal method)."""
        try:
            # Update task state
            self._task_state.update(
                status=TaskStatus.CONNECTING,
                instruction=instruction,
                start_time=time.time(),
                step_count=0,
                error_message="",
            )

            # Connect to robot
            connected, connect_error = await self._connect_robot()
            if not connected:
                self._task_state.update(
                    status=TaskStatus.ERROR,
                    error_message=connect_error
                    or f"Failed to connect to {self.tool_name_str}",
                )
                return

            # Get policy instance
            policy_instance = await self._get_policy(
                policy_provider=policy_provider,
                policy_port=policy_port,
                policy_host=policy_host,
                **policy_kwargs,
            )

            # Initialize policy with robot state keys
            if not await self._initialize_policy(policy_instance):
                self._task_state.update(
                    status=TaskStatus.ERROR,
                    error_message="Failed to initialize policy",
                )
                return

            logger.info(f"🎯 Starting task: '{instruction}' on {self.tool_name_str}")
            logger.info(f"🧠 Using policy: {policy_provider}")

            self._task_state.update(status=TaskStatus.RUNNING)
            start_time = time.time()

            while (
                time.time() - start_time < duration
                and self._task_state.status == TaskStatus.RUNNING
                and not self._shutdown_event.is_set()
            ):

                # Get observation from robot
                observation = await asyncio.to_thread(self.robot.get_observation)

                # Get actions from policy
                robot_actions = await policy_instance.get_actions(
                    observation, instruction
                )

                # Execute actions from chunk with proper timing control
                # Wait between actions for smooth execution
                for action_dict in robot_actions[: self.action_horizon]:
                    if self._task_state.status != TaskStatus.RUNNING:
                        break
                    await asyncio.to_thread(self.robot.send_action, action_dict)
                    with self._task_state._lock:
                        self._task_state.step_count += 1

                    # Stream step to mesh (observation + action)
                    if self.mesh and self.mesh.alive:
                        try:
                            self.mesh.publish_step(
                                step=self._task_state.step_count,
                                observation=observation,
                                action=action_dict,
                                instruction=instruction,
                                policy=policy_provider,
                            )
                        except Exception:
                            pass  # Never let mesh errors break the control loop

                    # Wait for action to complete before sending next action
                    # Default 50Hz (0.02s)
                    await asyncio.sleep(self.action_sleep_time)

            # Update final state
            elapsed = time.time() - start_time
            self._task_state.update(duration=elapsed)

            if self._task_state.status == TaskStatus.RUNNING:
                self._task_state.update(status=TaskStatus.COMPLETED)
                logger.info(
                    f"✅ Task completed: '{instruction}' in {elapsed:.1f}s ({self._task_state.step_count} steps)"
                )

        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            self._task_state.update(status=TaskStatus.ERROR, error_message=str(e))

    def _execute_task_sync(
        self,
        instruction: str,
        policy_provider: str = "groot",
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        duration: float = 30.0,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Execute task synchronously in thread - no new event loop."""

        # Import here to avoid conflicts

        # Run task without creating new event loop - let it run in thread
        async def task_runner():
            await self._execute_task_async(
                instruction,
                policy_provider,
                policy_port,
                policy_host,
                duration,
                **policy_kwargs,
            )

        # use _run_async helper for clean event loop handling
        _run_async(lambda: task_runner())

        # Build policy display string
        policy_display = policy_provider
        if policy_port:
            policy_display += f" on {policy_host}:{policy_port}"
        if policy_kwargs.get("model_path"):
            policy_display += f" ({policy_kwargs['model_path']})"

        # Return final status
        return {
            "status": (
                "success"
                if self._task_state.status == TaskStatus.COMPLETED
                else "error"
            ),
            "content": [
                {
                    "text": f"✅ Task: '{instruction}' - {self._task_state.status.value}\n"
                    f"🤖 Robot: {self.tool_name_str} ({self.robot})\n"
                    f"🧠 Policy: {policy_display}\n"
                    f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
                    f"🎯 Steps: {self._task_state.step_count}"
                    + (
                        f"\n❌ Error: {self._task_state.error_message}"
                        if self._task_state.error_message
                        else ""
                    )
                }
            ],
        }

    def start_task(
        self,
        instruction: str,
        policy_provider: str = "groot",
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        duration: float = 30.0,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Start robot task asynchronously and return immediately."""

        # Check if task is already running
        if self._task_state.status == TaskStatus.RUNNING:
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Task already running: {self._task_state.instruction}"}
                ],
            }

        # Start task in background
        self._task_state.task_future = self._executor.submit(
            self._execute_task_sync,
            instruction,
            policy_provider,
            policy_port,
            policy_host,
            duration,
            **policy_kwargs,
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"🚀 Task started: '{instruction}'\n"
                    f"🤖 Robot: {self.tool_name_str}\n"
                    f"🧠 Policy: {policy_provider}\n"
                    f"💡 Use action='status' to check progress\n"
                    f"💡 Use action='stop' to interrupt"
                }
            ],
        }

    def get_task_status(self) -> Dict[str, Any]:
        """Get current task execution status."""

        # Update duration for running tasks
        if self._task_state.status == TaskStatus.RUNNING:
            self._task_state.duration = time.time() - self._task_state.start_time

        status_text = f"📊 Robot Status: {self._task_state.status.value.upper()}\n"

        if self._task_state.instruction:
            status_text += f"🎯 Task: {self._task_state.instruction}\n"

        if self._task_state.status == TaskStatus.RUNNING:
            status_text += f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"🔄 Steps: {self._task_state.step_count}\n"
        elif self._task_state.status in [
            TaskStatus.COMPLETED,
            TaskStatus.STOPPED,
            TaskStatus.ERROR,
        ]:
            status_text += f"⏱️ Total Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"🎯 Total Steps: {self._task_state.step_count}\n"

        if self._task_state.error_message:
            status_text += f"❌ Error: {self._task_state.error_message}\n"

        return {
            "status": "success",
            "content": [{"text": status_text}],
        }

    def stop_task(self) -> Dict[str, Any]:
        """Stop currently running task."""

        if self._task_state.status != TaskStatus.RUNNING:
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"💤 No task running to stop (current: {self._task_state.status.value})"
                    }
                ],
            }

        # Signal task to stop
        self._task_state.update(status=TaskStatus.STOPPED)

        # Cancel future if it exists
        if self._task_state.task_future:
            self._task_state.task_future.cancel()

        logger.info(f"🛑 Task stopped: {self._task_state.instruction}")

        return {
            "status": "success",
            "content": [
                {
                    "text": f"🛑 Task stopped: '{self._task_state.instruction}'\n"
                    f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
                    f"🎯 Steps completed: {self._task_state.step_count}"
                }
            ],
        }

    def record_task(
        self,
        instruction: str,
        policy_provider: str = "groot",
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        duration: float = 30.0,
        fps: int = 30,
        output_path: str = None,
        repo_id: str = None,
        push_to_hub: bool = False,
        vcodec: str = "libsvtav1",
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Execute a policy and record to LeRobotDataset (training-ready).

        Every control step writes obs + action via DatasetRecorder.add_frame().
        Produces parquet episodes + encoded video, ready for lerobot-train.

        Args:
            instruction: Natural language task instruction
            policy_provider: Policy provider name
            policy_port: Port for remote policy services
            policy_host: Host for remote policy services
            duration: Maximum execution time in seconds
            fps: Recording frame rate
            output_path: (unused, kept for compat)
            repo_id: HuggingFace dataset ID (e.g. "user/my_data").
                     Auto-generated if None.
            push_to_hub: Push to HuggingFace Hub after recording
            vcodec: Video codec (h264, hevc, libsvtav1)
            **policy_kwargs: Provider-specific parameters
        """

        # Auto-generate repo_id
        if repo_id is None:
            repo_id = f"local/{self.tool_name_str}_{int(time.time())}"

        # Try to use DatasetRecorder (LeRobotDataset)
        try:
            from .dataset_recorder import HAS_LEROBOT_DATASET, DatasetRecorder
        except ImportError:
            HAS_LEROBOT_DATASET = False

        if not HAS_LEROBOT_DATASET:
            return {
                "status": "error",
                "content": [
                    {"text": "lerobot not installed. Install with: pip install lerobot"}
                ],
            }

        import asyncio

        async def _record_async():
            # Connect robot
            connected, connect_error = await self._connect_robot()
            if not connected:
                return {"status": "error", "content": [{"text": f"{connect_error}"}]}

            # Create policy
            policy_instance = await self._get_policy(
                policy_provider=policy_provider,
                policy_port=policy_port,
                policy_host=policy_host,
                **policy_kwargs,
            )

            if not await self._initialize_policy(policy_instance):
                return {
                    "status": "error",
                    "content": [{"text": "Failed to initialize policy"}],
                }

            # Get camera keys
            camera_keys = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                camera_keys = list(self.robot.config.cameras.keys())

            # Get robot type
            robot_type = getattr(
                self.robot, "robot_type", getattr(self.robot, "name", "unknown")
            )

            # Get joint names from action features
            joint_names = (
                list(self._action_features.keys()) if self._action_features else None
            )
            if not joint_names:
                # Fallback: get from observation
                test_obs = await asyncio.to_thread(self.robot.get_observation)
                joint_names = [k for k in test_obs.keys() if k not in camera_keys]

            # Create DatasetRecorder
            recorder = DatasetRecorder.create(
                repo_id=repo_id,
                fps=fps,
                robot_type=robot_type,
                joint_names=joint_names,
                camera_keys=camera_keys,
                task=instruction,
                vcodec=vcodec,
            )

            step_count = 0
            start_time = time.time()
            self._task_state.update(status=TaskStatus.RUNNING)
            self._task_state.instruction = instruction
            self._task_state.start_time = start_time

            try:
                while (
                    time.time() - start_time < duration
                    and self._task_state.status == TaskStatus.RUNNING
                    and not self._shutdown_event.is_set()
                ):
                    # Get observation
                    observation = await asyncio.to_thread(self.robot.get_observation)

                    # Get actions from policy
                    robot_actions = await policy_instance.get_actions(
                        observation, instruction
                    )

                    # Execute actions
                    for action_dict in robot_actions[: self.action_horizon]:
                        if self._task_state.status != TaskStatus.RUNNING:
                            break

                        await asyncio.to_thread(self.robot.send_action, action_dict)
                        step_count += 1

                        # Record frame to LeRobotDataset
                        recorder.add_frame(
                            observation=observation,
                            action=action_dict,
                            task=instruction,
                            camera_keys=camera_keys,
                        )

                        await asyncio.sleep(self.action_sleep_time)

            finally:
                self._task_state.status = TaskStatus.COMPLETED

            # Save episode
            recorder.save_episode()
            push_result = None
            if push_to_hub:
                push_result = recorder.push_to_hub(tags=["strands-robots", "real"])

            ds_root = recorder.root
            ds_repo = recorder.repo_id
            ds_frames = recorder.frame_count
            ds_episodes = recorder.episode_count
            recorder.finalize()

            elapsed = time.time() - start_time

            text = (
                f"Recorded to LeRobotDataset: {ds_repo}\n"
                f"Robot: {self.tool_name_str} | Policy: {policy_provider}\n"
                f"Task: {instruction}\n"
                f"{ds_frames} frames, {ds_episodes} episode(s), {elapsed:.1f}s\n"
                f"Local: {ds_root}"
            )
            if push_result and push_result.get("status") == "success":
                text += "\nPushed to HuggingFace Hub"

            return {
                "status": "success",
                "content": [{"text": text}],
            }

        # Run the async recording
        # use _run_async helper
        result = _resolve_coroutine(_record_async())

        return result

    def replay_episode(
        self,
        repo_id: str,
        episode: int = 0,
        root: Optional[str] = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Replay actions from a LeRobotDataset episode on real hardware.

        Loads actions from a recorded episode and sends them to the robot
        at the original recorded frequency (scaled by speed).

        Args:
            repo_id: HuggingFace dataset repo ID (e.g., "user/my_data")
            episode: Episode index to replay
            root: Local root directory for dataset
            speed: Playback speed multiplier (1.0 = original, 0.5 = half speed)
        """
        import asyncio

        async def _replay_async():
            # Load dataset
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                ds = LeRobotDataset(repo_id=repo_id, root=root)
            except ImportError:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "❌ lerobot not installed. Install: pip install lerobot"
                        }
                    ],
                }
            except Exception as e:
                return {
                    "status": "error",
                    "content": [
                        {"text": f"❌ Failed to load dataset '{repo_id}': {e}"}
                    ],
                }

            # Get episode info
            num_episodes = (
                ds.meta.total_episodes
                if hasattr(ds.meta, "total_episodes")
                else len(ds.meta.episodes)
            )
            if episode >= num_episodes:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ Episode {episode} out of range (0-{num_episodes - 1})"
                        }
                    ],
                }

            # Resolve the frame range for the target episode.
            # LeRobot datasets store frames contiguously; we need the
            # global start index and length for this episode.
            #
            # Strategy:
            #   1. episode_data_index (LeRobot v2) — O(1) lookup
            #   2. meta.episodes (LeRobot v1) — sum lengths
            #   3. Fallback: linear scan of episode_index per frame
            episode_start = 0
            episode_length = 0
            try:
                if hasattr(ds, "episode_data_index"):
                    # LeRobot v2: direct from/to index table
                    from_idx = ds.episode_data_index["from"][episode].item()
                    to_idx = ds.episode_data_index["to"][episode].item()
                    episode_start = from_idx
                    episode_length = to_idx - from_idx
                else:
                    # LeRobot v1: accumulate lengths from episode metadata
                    for i in range(episode):
                        episode_info = (
                            ds.meta.episodes[i] if hasattr(ds.meta, "episodes") else {}
                        )
                        episode_start += episode_info.get("length", 0)
                    episode_info = (
                        ds.meta.episodes[episode]
                        if hasattr(ds.meta, "episodes")
                        else {}
                    )
                    episode_length = episode_info.get("length", 0)
            except Exception:
                # Last resort: scan frames to find episode boundaries
                episode_length = 0
                for idx in range(len(ds)):
                    frame = ds[idx]
                    frame_episode = (
                        frame.get("episode_index", -1) if hasattr(frame, "get") else -1
                    )
                    if hasattr(frame_episode, "item"):
                        frame_episode = frame_episode.item()
                    if frame_episode == episode:
                        if episode_length == 0:
                            episode_start = idx
                        episode_length += 1
                    elif episode_length > 0:
                        break

            if episode_length == 0:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Episode {episode} has no frames"}],
                }

            # Connect robot
            connected, connect_error = await self._connect_robot()
            if not connected:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ {connect_error}"}],
                }

            # Get dataset FPS
            dataset_fps = getattr(ds, "fps", 30)
            frame_interval = 1.0 / (dataset_fps * speed)

            # Extract action keys from first frame
            first_frame = ds[episode_start]
            action_keys = [k for k in first_frame.keys() if "action" in k]

            logger.info(
                f"▶️ Replaying episode {episode}: {episode_length} frames at "
                f"{dataset_fps * speed:.1f} fps (speed={speed}x)"
            )

            # Replay loop
            import numpy as np

            self._task_state.update(status=TaskStatus.RUNNING)
            self._task_state.instruction = f"replay:{repo_id}/ep{episode}"
            self._task_state.start_time = time.time()
            frames_sent = 0

            try:
                for frame_idx in range(episode_length):
                    if (
                        self._task_state.status != TaskStatus.RUNNING
                        or self._shutdown_event.is_set()
                    ):
                        break

                    step_start = time.time()

                    frame = ds[episode_start + frame_idx]

                    # Extract action and send to robot
                    action_dict = {}
                    for k in action_keys:
                        v = frame[k]
                        if hasattr(v, "numpy"):
                            v = v.numpy()
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        action_dict[k] = v

                    # Send action to robot hardware
                    if hasattr(self.robot, "send_action"):
                        import torch

                        # Convert to the format robot.send_action expects
                        if "action" in action_dict:
                            action_tensor = action_dict["action"]
                            if not isinstance(action_tensor, torch.Tensor):
                                action_tensor = torch.tensor(
                                    (
                                        action_tensor
                                        if isinstance(action_tensor, list)
                                        else np.asarray(action_tensor)
                                    ),
                                    dtype=torch.float32,
                                )
                            await asyncio.to_thread(
                                self.robot.send_action, action_tensor
                            )

                    frames_sent += 1

                    # Maintain replay frequency
                    elapsed = time.time() - step_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Replay error at frame {frames_sent}: {e}")
            finally:
                self._task_state.status = TaskStatus.IDLE

            duration = time.time() - self._task_state.start_time
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"▶️ Replayed episode {episode} from {repo_id}\n"
                            f"Frames: {frames_sent}/{episode_length} | "
                            f"Duration: {duration:.1f}s | "
                            f"Speed: {speed}x | "
                            f"Effective FPS: {frames_sent / max(duration, 0.001):.1f}"
                        )
                    },
                    {
                        "json": {
                            "episode": episode,
                            "frames_sent": frames_sent,
                            "total_frames": episode_length,
                            "duration_s": round(duration, 2),
                            "speed": speed,
                            "action_keys": action_keys,
                        }
                    },
                ],
            }

        # use _resolve_coroutine helper
        result = _resolve_coroutine(_replay_async())

        return result

    def get_features(self) -> Dict[str, Any]:
        """Return the robot's observation_features and action_features."""
        obs_features = {}
        for k, v in self._observation_features.items():
            obs_features[k] = str(v) if not isinstance(v, str) else v

        act_features = {}
        for k, v in self._action_features.items():
            act_features[k] = str(v) if not isinstance(v, str) else v

        text = f"🤖 Robot Features for {self.tool_name_str}\n"
        text += f"\n📥 Observation Features ({len(obs_features)}):\n"
        for k, v in obs_features.items():
            text += f"  • {k}: {v}\n"
        text += f"\n📤 Action Features ({len(act_features)}):\n"
        for k, v in act_features.items():
            text += f"  • {k}: {v}\n"

        if not obs_features and not act_features:
            text += "\n⚠️ No features available. Robot may not expose observation_features/action_features."

        return {
            "status": "success",
            "content": [
                {"text": text},
                {
                    "json": {
                        "observation_features": obs_features,
                        "action_features": act_features,
                    }
                },
            ],
        }

    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """Get tool specification with async actions and multi-provider support."""
        return {
            "name": self.tool_name_str,
            "description": (
                f"Universal robot control with async task execution ({self.robot}). "
                f"Actions: execute (blocking), start (async), status, stop, record (video), replay (dataset episode on hardware), features (robot capabilities). "
                f"Supports multiple policy providers: groot (NVIDIA GR00T via ZMQ), "
                f"lerobot_local (direct HuggingFace inference - ACT/Pi0/SmolVLA/Diffusion, no server), "
                f"lerobot_async (LeRobot gRPC - Pi0/ACT/SmolVLA/Diffusion), "
                f"dreamgen (GR00T-Dreams IDM/VLA), mock (testing). "
                f"For lerobot_local: instruction + pretrained_name_or_path required. "
                f"For groot/lerobot_async: instruction + policy_port required. "
                f"For dreamgen: instruction + model_path required. "
                f"For record: saves to LeRobotDataset (parquet + video), training-ready. "
                f"For status/stop: no additional parameters needed."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform: execute (blocking), start (async), status, stop, record (video), replay (dataset episode), features (robot capabilities)",
                            "enum": [
                                "execute",
                                "start",
                                "status",
                                "stop",
                                "record",
                                "replay",
                                "features",
                            ],
                            "default": "execute",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start)",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider name. Common options: groot, lerobot_local, lerobot_async, dreamgen, mock. Any provider supported by create_policy() works.",
                            "default": "groot",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for groot/lerobot_async)",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                            "default": "localhost",
                        },
                        "model_path": {
                            "type": "string",
                            "description": "Model path or HuggingFace ID (required for dreamgen)",
                        },
                        "server_address": {
                            "type": "string",
                            "description": "Full gRPC server address for lerobot_async (e.g. localhost:8080)",
                        },
                        "policy_type": {
                            "type": "string",
                            "description": "LeRobot policy type for lerobot_async (pi0, act, smolvla, diffusion, etc.)",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds",
                            "default": 30.0,
                        },
                        "pretrained_name_or_path": {
                            "type": "string",
                            "description": "HuggingFace model ID for lerobot_local (e.g. lerobot/act_aloha_sim_transfer_cube_human)",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "(deprecated) kept for compat",
                        },
                        "repo_id": {
                            "type": "string",
                            "description": "HuggingFace dataset repo ID (e.g. user/my_data). Used for record (auto-generated if omitted) and replay actions.",
                        },
                        "push_to_hub": {
                            "type": "boolean",
                            "description": "Push dataset to HuggingFace Hub after recording",
                            "default": False,
                        },
                        "vcodec": {
                            "type": "string",
                            "description": "Video codec for dataset (h264, hevc, libsvtav1)",
                            "default": "libsvtav1",
                        },
                        "fps": {
                            "type": "integer",
                            "description": "Video frames per second for record action",
                            "default": 30,
                        },
                        "episode": {
                            "type": "integer",
                            "description": "Episode index for replay action",
                            "default": 0,
                        },
                        "speed": {
                            "type": "number",
                            "description": "Playback speed for replay (1.0 = original, 0.5 = half speed)",
                            "default": 1.0,
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream robot task execution with async actions."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})

            action = input_data.get("action", "execute")

            # Handle different actions
            if action in ("execute", "start"):
                instruction = input_data.get("instruction", "")
                policy_provider = input_data.get("policy_provider", "groot")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                duration = input_data.get("duration", 30.0)

                # Build provider-specific kwargs from input
                policy_kwargs = {}
                for key in [
                    "model_path",
                    "server_address",
                    "policy_type",
                    "pretrained_name_or_path",
                    "actions_per_chunk",
                    "mode",
                    "embodiment_tag",
                    "device",
                ]:
                    if key in input_data:
                        policy_kwargs[key] = input_data[key]

                # Validate required fields
                if not instruction:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [
                                {
                                    "text": "❌ instruction is required for execute/start action"
                                }
                            ],
                        }
                    )
                    return

                # Validate provider-specific requirements
                if (
                    policy_provider in ("groot", "lerobot_async")
                    and not policy_port
                    and not policy_kwargs.get("server_address")
                ):
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [
                                {
                                    "text": f"❌ policy_port or server_address required for {policy_provider}"
                                }
                            ],
                        }
                    )
                    return

                if policy_provider == "dreamgen" and not policy_kwargs.get(
                    "model_path"
                ):
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [
                                {
                                    "text": "❌ model_path is required for dreamgen provider"
                                }
                            ],
                        }
                    )
                    return

                if action == "execute":
                    task_result = self._execute_task_sync(
                        instruction,
                        policy_provider,
                        policy_port,
                        policy_host,
                        duration,
                        **policy_kwargs,
                    )
                    yield ToolResultEvent({"toolUseId": tool_use_id, **task_result})
                else:  # start
                    start_result = self.start_task(
                        instruction,
                        policy_provider,
                        policy_port,
                        policy_host,
                        duration,
                        **policy_kwargs,
                    )
                    yield ToolResultEvent({"toolUseId": tool_use_id, **start_result})

            elif action == "status":
                status_result = self.get_task_status()
                yield ToolResultEvent({"toolUseId": tool_use_id, **status_result})

            elif action == "stop":
                stop_result = self.stop_task()
                yield ToolResultEvent({"toolUseId": tool_use_id, **stop_result})

            elif action == "features":
                features_result = self.get_features()
                yield ToolResultEvent({"toolUseId": tool_use_id, **features_result})

            elif action == "record":
                instruction = input_data.get("instruction", "")
                policy_provider = input_data.get("policy_provider", "groot")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                duration = input_data.get("duration", 30.0)
                record_fps = input_data.get("fps", 30)
                record_output = input_data.get("output_path")

                policy_kwargs = {}
                for key in [
                    "model_path",
                    "server_address",
                    "policy_type",
                    "pretrained_name_or_path",
                    "actions_per_chunk",
                    "mode",
                    "embodiment_tag",
                    "device",
                ]:
                    if key in input_data:
                        policy_kwargs[key] = input_data[key]

                if not instruction:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [
                                {"text": "❌ instruction is required for record action"}
                            ],
                        }
                    )
                    return

                record_result = self.record_task(
                    instruction,
                    policy_provider,
                    policy_port,
                    policy_host,
                    duration,
                    record_fps,
                    record_output,
                    repo_id=input_data.get("repo_id"),
                    push_to_hub=input_data.get("push_to_hub", False),
                    vcodec=input_data.get("vcodec", "libsvtav1"),
                    **policy_kwargs,
                )
                yield ToolResultEvent({"toolUseId": tool_use_id, **record_result})

            elif action == "replay":
                replay_repo_id = input_data.get("repo_id")
                replay_episode = input_data.get("episode", 0)
                replay_root = input_data.get("root")
                replay_speed = input_data.get("speed", 1.0)

                if not replay_repo_id:
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [
                                {"text": "❌ repo_id required for replay action"}
                            ],
                        }
                    )
                    return

                replay_result = self.replay_episode(
                    repo_id=replay_repo_id,
                    episode=replay_episode,
                    root=replay_root,
                    speed=replay_speed,
                )
                yield ToolResultEvent({"toolUseId": tool_use_id, **replay_result})

            else:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [
                            {
                                "text": f"❌ Unknown action: {action}. Valid: execute, start, status, stop, record, replay, features"
                            }
                        ],
                    }
                )

        except Exception as e:
            logger.error(f"❌ {self.tool_name_str} error: {e}")
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "status": "error",
                    "content": [{"text": f"❌ {self.tool_name_str} error: {str(e)}"}],
                }
            )

    def cleanup(self):
        """Cleanup resources and stop any running tasks."""
        try:
            # Stop mesh
            if hasattr(self, "mesh") and self.mesh:
                self.mesh.stop()

            # Signal shutdown
            self._shutdown_event.set()

            # Stop any running task
            if self._task_state.status == TaskStatus.RUNNING:
                self.stop_task()

            # Shutdown executor
            self._executor.shutdown(wait=True)

            logger.info(f"🧹 {self.tool_name_str} cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error for {self.tool_name_str}: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor

    async def get_status(self) -> Dict[str, Any]:
        """Get robot status including connection and task state."""
        try:
            is_connected = (
                self.robot.is_connected
                if hasattr(self.robot, "is_connected")
                else False
            )
            is_calibrated = (
                self.robot.is_calibrated
                if hasattr(self.robot, "is_calibrated")
                else True
            )

            camera_status = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                for name in self.robot.config.cameras.keys():
                    camera_status.append(name)

            status_data = {
                "robot_name": self.tool_name_str,
                "robot_type": getattr(self.robot, "robot_type", self.robot.name),
                "robot_info": str(self.robot),
                "data_config": self.data_config,
                "is_connected": is_connected,
                "is_calibrated": is_calibrated,
                "cameras": camera_status,
                "task_status": self._task_state.status.value,
                "current_instruction": self._task_state.instruction,
                "task_duration": self._task_state.duration,
                "task_steps": self._task_state.step_count,
            }

            if self._task_state.error_message:
                status_data["task_error"] = self._task_state.error_message

            return status_data

        except Exception as e:
            logger.error(f"❌ Error getting status for {self.tool_name_str}: {e}")
            return {
                "robot_name": self.tool_name_str,
                "error": str(e),
                "is_connected": False,
                "task_status": "error",
            }

    async def stop(self):
        """Stop robot and disconnect."""
        try:
            if self._task_state.status == TaskStatus.RUNNING:
                self.stop_task()

            if hasattr(self.robot, "disconnect"):
                await asyncio.to_thread(self.robot.disconnect)

            self.cleanup()
            logger.info(f"🛑 {self.tool_name_str} stopped and disconnected")

        except Exception as e:
            logger.error(f"❌ Error stopping robot: {e}")
