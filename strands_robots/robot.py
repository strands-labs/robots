#!/usr/bin/env python3
"""
Universal Robot Control with Policy Abstraction for Any VLA Provider

This module provides a clean robot interface that works with any LeRobot-compatible
robot and any VLA provider through the Policy abstraction.

Features:
- Async robot task execution with real-time status reporting
- Non-blocking operations - robot moves while tool returns status
- Stop functionality to interrupt running tasks
- Connection state management with proper error handling
- Policy abstraction for any VLA provider
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolResult, ToolSpec, ToolUse

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot as LeRobotRobot

    from .policies import Policy

logger = logging.getLogger(__name__)


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
    """Robot task execution state"""

    status: TaskStatus = TaskStatus.IDLE
    instruction: str = ""
    start_time: float = 0.0
    duration: float = 0.0
    step_count: int = 0
    error_message: str = ""
    task_future: Future | None = None


class Robot(AgentTool):
    """Universal robot control with async task execution and status reporting."""

    def __init__(
        self,
        tool_name: str,
        robot: LeRobotRobot | RobotConfig | str,
        cameras: dict[str, dict[str, Any]] | None = None,
        action_horizon: int = 8,
        data_config: str | Any | None = None,
        control_frequency: float = 50.0,
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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{tool_name}_executor")
        self._shutdown_event = threading.Event()

        # Initialize robot using lerobot's abstraction
        self.robot = self._initialize_robot(robot, cameras, **kwargs)

        logger.info(f"🤖 {tool_name} initialized with async capabilities")
        logger.info(f"📱 Robot: {self.robot.name} (type: {getattr(self.robot, 'robot_type', 'unknown')})")
        logger.info(f"⏱️ Control frequency: {control_frequency}Hz ({self.action_sleep_time * 1000:.1f}ms per action)")

        # Get camera info if available
        if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
            cameras_list = list(self.robot.config.cameras.keys())
            logger.info(f"📹 Cameras: {cameras_list}")

        if data_config:
            logger.info(f"⚙️ Data config: {data_config}")

    def _initialize_robot(
        self, robot: LeRobotRobot | RobotConfig | str, cameras: dict[str, dict[str, Any]] | None, **kwargs
    ) -> LeRobotRobot:
        """Initialize LeRobot robot instance using native lerobot patterns."""
        from lerobot.robots.config import RobotConfig
        from lerobot.robots.robot import Robot as LeRobotRobot
        from lerobot.robots.utils import make_robot_from_config

        # Direct robot instance - use as-is
        if isinstance(robot, LeRobotRobot):
            return robot

        # Robot config - use lerobot's factory
        elif isinstance(robot, RobotConfig):
            return make_robot_from_config(robot)

        # Robot type string - create config and use lerobot's factory
        elif isinstance(robot, str):
            config = self._create_minimal_config(robot, cameras, **kwargs)
            return make_robot_from_config(config)

        else:
            raise ValueError(
                f"Unsupported robot type: {type(robot)}. "
                f"Expected LeRobot Robot instance, RobotConfig, or robot type string."
            )

    def _create_minimal_config(
        self, robot_type: str, cameras: dict[str, dict[str, Any]] | None, **kwargs
    ) -> RobotConfig:
        """Create minimal robot config using specific robot config classes."""
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        # Convert cameras to lerobot format
        camera_configs = {}
        if cameras:
            for name, config in cameras.items():
                if config.get("type", "opencv") == "opencv":
                    camera_configs[name] = OpenCVCameraConfig(
                        index_or_path=config["index_or_path"],
                        fps=config.get("fps", 30),
                        width=config.get("width", 640),
                        height=config.get("height", 480),
                        rotation=config.get("rotation", 0),
                        color_mode=config.get("color_mode", "rgb"),
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {config.get('type')}")

        # Map robot type to specific config class
        config_mapping = {
            "so101_follower": ("lerobot.robots.so101_follower", "SO101FollowerConfig"),
            "so100_follower": ("lerobot.robots.so100_follower", "SO100FollowerConfig"),
            "bi_so100_follower": ("lerobot.robots.bi_so100_follower", "BiSO100FollowerConfig"),
            "viperx": ("lerobot.robots.viperx", "ViperXConfig"),
            "koch_follower": ("lerobot.robots.koch_follower", "KochFollowerConfig"),
            # Add more as needed
        }

        if robot_type not in config_mapping:
            raise ValueError(f"Unsupported robot type: {robot_type}. Supported types: {list(config_mapping.keys())}")

        # Import specific config class dynamically
        module_name, class_name = config_mapping[robot_type]
        try:
            import importlib

            module = importlib.import_module(module_name)
            ConfigClass = getattr(module, class_name)
        except Exception as e:
            raise ValueError(f"Failed to import {class_name} from {module_name}: {e}")

        # Create config with proper parameters
        config_data = {
            "id": self.tool_name_str,
            "cameras": camera_configs,
        }

        # Filter kwargs to only include supported fields for this robot type
        # Port is common for most serial robots
        if "port" in kwargs:
            config_data["port"] = kwargs["port"]

        # Add other common fields as needed
        for key in ["calibration_dir", "mock", "use_degrees"]:
            if key in kwargs:
                config_data[key] = kwargs[key]

        try:
            return ConfigClass(**config_data)
        except Exception as e:
            raise ValueError(f"Failed to create {class_name} for robot type '{robot_type}': {e}. Config: {config_data}")

    async def _get_policy(
        self, policy_port: int | None = None, policy_host: str = "localhost", policy_provider: str = "groot"
    ) -> Policy:
        """Create policy on-the-fly from invocation parameters."""
        from .policies import create_policy

        if not policy_port:
            raise ValueError("policy_port is required for robot operation")

        policy_config = {"port": policy_port, "host": policy_host}

        if self.data_config:
            policy_config["data_config"] = self.data_config

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
                    await asyncio.to_thread(self.robot.connect, False)  # calibrate=False

            except DeviceAlreadyConnectedError:
                # This is expected and fine - robot is already connected
                logger.info(f"✅ {self.robot} was already connected")

            except Exception as e:
                # Check if it's the string version of "already connected" error
                error_str = str(e).lower()
                if "already connected" in error_str or "is already connected" in error_str:
                    logger.info(f"✅ {self.robot} connection already established")
                else:
                    # Re-raise if it's a different error
                    raise e

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
            # Get robot state keys from observation
            test_obs = await asyncio.to_thread(self.robot.get_observation)

            # Filter out camera keys to get robot state keys
            camera_keys = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
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
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
    ) -> None:
        """Execute robot task in background thread (internal method)."""
        try:
            # Update task state
            self._task_state.status = TaskStatus.CONNECTING
            self._task_state.instruction = instruction
            self._task_state.start_time = time.time()
            self._task_state.step_count = 0
            self._task_state.error_message = ""

            # Connect to robot
            connected, connect_error = await self._connect_robot()
            if not connected:
                self._task_state.status = TaskStatus.ERROR
                self._task_state.error_message = connect_error or f"Failed to connect to {self.tool_name_str}"
                return

            # Get policy instance
            policy_instance = await self._get_policy(policy_port, policy_host, policy_provider)

            # Initialize policy with robot state keys
            if not await self._initialize_policy(policy_instance):
                self._task_state.status = TaskStatus.ERROR
                self._task_state.error_message = "Failed to initialize policy"
                return

            logger.info(f"🎯 Starting task: '{instruction}' on {self.tool_name_str}")
            logger.info(f"🧠 Using policy: {policy_provider} on {policy_host}:{policy_port}")

            self._task_state.status = TaskStatus.RUNNING
            start_time = time.time()

            while (
                time.time() - start_time < duration
                and self._task_state.status == TaskStatus.RUNNING
                and not self._shutdown_event.is_set()
            ):
                # Get observation from robot
                observation = await asyncio.to_thread(self.robot.get_observation)

                # Get actions from policy
                robot_actions = await policy_instance.get_actions(observation, instruction)

                # Execute actions from chunk with proper timing control
                # Wait between actions for smooth execution
                for action_dict in robot_actions[: self.action_horizon]:
                    if self._task_state.status != TaskStatus.RUNNING:
                        break
                    await asyncio.to_thread(self.robot.send_action, action_dict)
                    self._task_state.step_count += 1
                    # Wait for action to complete before sending next action
                    # Default 50Hz (0.02s)
                    await asyncio.sleep(self.action_sleep_time)

            # Update final state
            elapsed = time.time() - start_time
            self._task_state.duration = elapsed

            if self._task_state.status == TaskStatus.RUNNING:
                self._task_state.status = TaskStatus.COMPLETED
                logger.info(
                    f"✅ Task completed: '{instruction}' in {elapsed:.1f}s ({self._task_state.step_count} steps)"
                )

        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            self._task_state.status = TaskStatus.ERROR
            self._task_state.error_message = str(e)

    def _execute_task_sync(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
    ) -> dict[str, Any]:
        """Execute task synchronously in thread - no new event loop."""

        # Import here to avoid conflicts
        import asyncio

        # Run task without creating new event loop - let it run in thread
        async def task_runner():
            await self._execute_task_async(instruction, policy_port, policy_host, policy_provider, duration)

        # Use asyncio.run only if no loop is running, otherwise run in existing loop
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're already in an event loop, we need to run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as exec:
                future = exec.submit(lambda: asyncio.run(task_runner()))
                future.result()  # Wait for completion
        except RuntimeError:
            # No event loop running - safe to create one
            asyncio.run(task_runner())

        # Return final status
        return {
            "status": "success" if self._task_state.status == TaskStatus.COMPLETED else "error",
            "content": [
                {
                    "text": f"✅ Task: '{instruction}' - {self._task_state.status.value}\n"
                    f"🤖 Robot: {self.tool_name_str} ({self.robot})\n"
                    f"🧠 Policy: {policy_provider} on {policy_host}:{policy_port}\n"
                    f"⏱️ Duration: {self._task_state.duration:.1f}s\n"
                    f"🎯 Steps: {self._task_state.step_count}"
                    + (f"\n❌ Error: {self._task_state.error_message}" if self._task_state.error_message else "")
                }
            ],
        }

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
    ) -> dict[str, Any]:
        """Start robot task asynchronously and return immediately."""

        # Check if task is already running
        if self._task_state.status == TaskStatus.RUNNING:
            return {
                "status": "error",
                "content": [{"text": f"❌ Task already running: {self._task_state.instruction}"}],
            }

        # Start task in background
        self._task_state.task_future = self._executor.submit(
            self._execute_task_sync, instruction, policy_port, policy_host, policy_provider, duration
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"🚀 Task started: '{instruction}'\n"
                    f"🤖 Robot: {self.tool_name_str}\n"
                    f"💡 Use action='status' to check progress\n"
                    f"💡 Use action='stop' to interrupt"
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
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
        elif self._task_state.status in [TaskStatus.COMPLETED, TaskStatus.STOPPED, TaskStatus.ERROR]:
            status_text += f"⏱️ Total Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"🎯 Total Steps: {self._task_state.step_count}\n"

        if self._task_state.error_message:
            status_text += f"❌ Error: {self._task_state.error_message}\n"

        return {
            "status": "success",
            "content": [{"text": status_text}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop currently running task."""

        if self._task_state.status != TaskStatus.RUNNING:
            return {
                "status": "success",
                "content": [{"text": f"💤 No task running to stop (current: {self._task_state.status.value})"}],
            }

        # Signal task to stop
        self._task_state.status = TaskStatus.STOPPED

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

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """Get tool specification with async actions."""
        return {
            "name": self.tool_name_str,
            "description": f"Universal robot control with async task execution ({self.robot}). "
            f"Actions: execute (blocking), start (async), status, stop. "
            f"For execute/start actions: instruction and policy_port are required. "
            f"For status/stop actions: no additional parameters needed.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform: execute (blocking), start (async), status, stop",
                            "enum": ["execute", "start", "status", "stop"],
                            "default": "execute",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start actions)",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for execute/start actions)",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                            "default": "localhost",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider (groot, openai, etc.)",
                            "default": "groot",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds",
                            "default": 30.0,
                        },
                    },
                    "required": ["action"],
                }
            },
        }

    @staticmethod
    def _make_tool_result(tool_use_id: str, result: dict[str, Any]) -> ToolResult:
        """Create a ToolResult dict with the given tool_use_id merged into result."""
        return cast(ToolResult, {"toolUseId": tool_use_id, **result})

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream robot task execution with async actions."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})

            action = input_data.get("action", "execute")

            # Handle different actions
            if action == "execute":
                # Blocking execution (legacy behavior)
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                duration = input_data.get("duration", 30.0)

                if not instruction or not policy_port:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {
                                "status": "error",
                                "content": [{"text": "❌ instruction and policy_port are required for execute action"}],
                            },
                        )
                    )
                    return

                # Execute task synchronously
                task_result = self._execute_task_sync(instruction, policy_port, policy_host, policy_provider, duration)
                yield ToolResultEvent(self._make_tool_result(tool_use_id, task_result))

            elif action == "start":
                # Asynchronous execution start
                instruction = input_data.get("instruction", "")
                policy_port = input_data.get("policy_port")
                policy_host = input_data.get("policy_host", "localhost")
                policy_provider = input_data.get("policy_provider", "groot")
                duration = input_data.get("duration", 30.0)

                if not instruction or not policy_port:
                    yield ToolResultEvent(
                        self._make_tool_result(
                            tool_use_id,
                            {
                                "status": "error",
                                "content": [{"text": "❌ instruction and policy_port are required for start action"}],
                            },
                        )
                    )
                    return

                # Start task asynchronously
                start_result = self.start_task(instruction, policy_port, policy_host, policy_provider, duration)
                yield ToolResultEvent(self._make_tool_result(tool_use_id, start_result))

            elif action == "status":
                # Get current task status
                status_result = self.get_task_status()
                yield ToolResultEvent(self._make_tool_result(tool_use_id, status_result))

            elif action == "stop":
                # Stop current task
                stop_result = self.stop_task()
                yield ToolResultEvent(self._make_tool_result(tool_use_id, stop_result))

            else:
                yield ToolResultEvent(
                    self._make_tool_result(
                        tool_use_id,
                        {
                            "status": "error",
                            "content": [
                                {"text": f"❌ Unknown action: {action}. Valid actions: execute, start, status, stop"}
                            ],
                        },
                    )
                )

        except Exception as e:
            logger.error(f"❌ {self.tool_name_str} error: {e}")
            yield ToolResultEvent(
                self._make_tool_result(
                    tool_use_id,
                    {
                        "status": "error",
                        "content": [{"text": f"❌ {self.tool_name_str} error: {str(e)}"}],
                    },
                )
            )

    def cleanup(self):
        """Cleanup resources and stop any running tasks."""
        try:
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

    async def get_status(self) -> dict[str, Any]:
        """Get robot status including connection and task state."""
        try:
            # Get robot connection status
            is_connected = self.robot.is_connected if hasattr(self.robot, "is_connected") else False
            is_calibrated = self.robot.is_calibrated if hasattr(self.robot, "is_calibrated") else True

            # Get camera status
            camera_status = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                for name in self.robot.config.cameras.keys():
                    camera_status.append(name)

            # Build status dict
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

            # Add error info if present
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
            # Stop any running task first
            if self._task_state.status == TaskStatus.RUNNING:
                self.stop_task()

            # Disconnect robot hardware
            if hasattr(self.robot, "disconnect"):
                await asyncio.to_thread(self.robot.disconnect)

            # Cleanup resources
            self.cleanup()

            logger.info(f"🛑 {self.tool_name_str} stopped and disconnected")

        except Exception as e:
            logger.error(f"❌ Error stopping robot: {e}")
