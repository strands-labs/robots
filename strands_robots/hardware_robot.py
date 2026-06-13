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
import dataclasses
import functools
import importlib
import logging
import pkgutil
import shutil
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolResult, ToolSpec, ToolUse

if TYPE_CHECKING:
    from lerobot.robots.config import RobotConfig
    from lerobot.robots.robot import Robot as LeRobotRobot

    from .policies import Policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy lerobot RobotConfig registration helper.
# ---------------------------------------------------------------------------
#
# lerobot's robot drivers register themselves with ``RobotConfig`` (a
# draccus ``ChoiceRegistry``) via ``@RobotConfig.register_subclass(...)``
# at module import time. Because ``lerobot.robots.__init__`` does not
# eagerly import every subpackage, the registry is empty until something
# triggers the import. ``_create_minimal_config`` calls this helper once
# per process to populate it. ``@functools.cache`` makes the second call
# a dict lookup, so the per-Robot() overhead amortises to ~0.


# Cross-robot kwargs forwarded to lerobot config constructors.  Exposed
# as a module-level constant so tests can import it (single source of
# truth).
#
# Post-R5, the dual-gate semantics are:
#   - kwargs in this allowlist BUT NOT on the resolved target dataclass
#     are silently dropped (cross-robot polymorphism: passing ``kp=...``
#     to so101 doesn't blow up just because ``kp`` is a unitree_g1
#     kwarg).
#   - kwargs declared on the resolved target dataclass are forwarded
#     automatically, regardless of whether they appear in this list
#     (so a future lerobot field like ``wifi_ssid`` Just Works without
#     a strands_robots release).
#   - kwargs unknown to BOTH are rejected at config-build time
#     (typos like ``prot=``, kwargs from another subsystem entirely).
#
# So this allowlist's job is narrow: it's the set of kwargs whose
# silent-drop on a non-matching robot we tolerate as a documented
# polymorphism win.  It is not a forwarding gate -- ``valid_fields``
# is.
_FORWARDABLE_KWARGS = (
    "port",  # serial robots (so100/so101, koch, openarm, ...)
    "robot_ip",  # network robots (unitree_g1, lekiwi, reachy2, ...)
    "kp",
    "kd",  # PD-controlled robots (g1, h1, ...)
    "default_positions",  # humanoids
    "control_dt",  # humanoids / locomotion
    "is_simulation",  # robots that share a sim/real driver
    "gravity_compensation",  # arms with IK comp
    "controller",  # locomotion controller selection
    "calibration_dir",
    "mock",
    "use_degrees",
    "max_relative_target",
    "disable_torque_on_disconnect",
)


@functools.cache
def _ensure_lerobot_robots_registered() -> None:
    """Import every robot driver subpackage so RobotConfig is populated.

    Walks ``lerobot.robots`` with ``pkgutil`` so we automatically pick up
    every robot lerobot ships -- past, present, and future -- including
    those whose ``robot_type`` doesn't match its subpackage name (e.g.
    ``hope_jr_arm`` in ``hope_jr/``, ``lekiwi_client`` in ``lekiwi/``,
    ``so100_follower`` and ``so101_follower`` both in ``so_follower/``).
    Then invokes lerobot's third-party plugin loader so any installed
    ``lerobot_robot_*`` distribution registers itself too.

    Idempotent via ``@functools.cache`` -- the first call walks the tree,
    subsequent calls are dict lookups.
    """
    try:
        import lerobot.robots as _lr_robots
    except ImportError as exc:
        # Distinguish two failure modes so the log level matches signal
        # value:
        #   1. lerobot wholly absent -- expected on sim-only / CI-only
        #      hosts that never reach hardware code; debug is enough.
        #      Caller will get a clean ``Unsupported robot type`` at the
        #      ChoiceRegistry lookup site.
        #   2. lerobot present but ``lerobot.robots`` unimportable --
        #      genuine partial-install signal worth a warning so the
        #      operator can triage without ``--log-level=DEBUG``.
        try:
            import lerobot  # noqa: F401  (probe-only)
        except ImportError:
            logger.debug("lerobot not installed: %s", exc)
        else:
            logger.warning(
                "lerobot is installed but lerobot.robots is not importable (partial install?): %s",
                exc,
            )
        return

    # Walk every immediate subpackage of ``lerobot.robots`` and import
    # it. Each subpackage's ``__init__`` (or its ``config_*`` module)
    # runs the ``@RobotConfig.register_subclass(...)`` decorator as a
    # side effect.
    for _, sub_name, is_pkg in pkgutil.iter_modules(_lr_robots.__path__):
        if not is_pkg:
            continue
        full_name = f"{_lr_robots.__name__}.{sub_name}"
        try:
            importlib.import_module(full_name)
        except (ImportError, OSError) as exc:
            # Driver-specific runtime dep missing (e.g. ``unitree_sdk2py``,
            # ``reachy2_sdk``) OR an OS-level probe failure inside a
            # driver's ``__init__`` (USB enumeration in ``unitree_sdk2py``
            # raising ``OSError``, ``FileNotFoundError`` on a missing SDK
            # config, etc.). Robot simply won't appear in the choice
            # registry -- that is the correct outcome: trying to construct
            # it later will raise ``Unsupported robot type`` with the
            # actual list of available types. Per AGENTS.md > Review
            # Learnings (#86) > "Exception Clauses Must Be Narrow" the
            # canonical pattern for hardware-probing imports is
            # ``(ImportError, OSError)``; widening further would mask
            # genuine bugs in driver registration code.
            logger.debug("[hardware_robot] skip %s: %s", full_name, exc)

    # Pick up third-party plugins (``lerobot_robot_*`` distributions) via
    # lerobot's own loader if available -- lets external robot vendors
    # expose drivers without any strands_robots involvement.
    try:
        from lerobot.utils.import_utils import register_third_party_plugins
    except ImportError:
        # ``register_third_party_plugins`` lives in modern lerobot only;
        # older versions skip this opt-in step (built-ins still work).
        logger.debug("[hardware_robot] register_third_party_plugins unavailable")
    else:
        try:
            register_third_party_plugins()
        except (ImportError, AttributeError, OSError) as exc:
            # #291: narrowed from bare ``except Exception`` per AGENTS.md
            # Review Learnings (#86). Third-party plugin registration can fail
            # for three benign, recoverable reasons: a plugin distribution
            # whose import chain is broken (ImportError), a lerobot version
            # whose loader entry-point shape differs (AttributeError), or an
            # OS-level probe inside a plugin's registration (OSError). Any of
            # these should degrade to "that plugin is absent from the registry"
            # -- not crash hardware init. A genuinely unexpected exception
            # (e.g. a plugin raising ValueError from buggy registration code)
            # now propagates so it is not silently masked.
            logger.warning("[hardware_robot] third-party plugin registration failed: %s", exc)


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
        **kwargs: Any,
    ) -> None:
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

        # Mesh attributes — populated by the Robot() factory after init.
        # Plain attributes (not properties) so test code can swap a fake mesh
        # in without going through the factory.
        # Set BEFORE _initialize_robot so cleanup()/__del__ never see an
        # AttributeError if construction fails partway through.
        self.mesh: Any = None
        self.peer_id: str | None = None

        # Initialize robot using lerobot's abstraction
        self.robot = self._initialize_robot(robot, cameras, **kwargs)

        # lerobot 0.5.1 unified the SO-family calibration directory from
        # per-variant subdirs (``so100_follower/``, ``so101_follower/``) to a
        # single shared ``so_follower/``. Customers who calibrated on a
        # pre-0.5 lerobot have their JSON at the OLD path; the new
        # ``calibration_fpath`` resolves to the NEW path, finds nothing, and
        # reports ``is_calibrated=False`` -- which only surfaces as a confusing
        # RuntimeError on the first ``get_observation()``, not on ``connect()``.
        # Migrate (copy, never move -- old lerobot installs may still read it)
        # so a fresh customer's existing calibration Just Works.
        self._migrate_legacy_calibration()

        logger.info("%s initialized with async capabilities", tool_name)
        logger.info("Robot: %s (type: %s)", self.robot.name, getattr(self.robot, "robot_type", "unknown"))
        logger.info("Control frequency: %sHz (%.1fms per action)", control_frequency, self.action_sleep_time * 1000)

        # Get camera info if available
        if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
            cameras_list = list(self.robot.config.cameras.keys())
            logger.info("Cameras: %s", cameras_list)

        if data_config:
            logger.info("Data config: %s", data_config)

    def _initialize_robot(
        self, robot: LeRobotRobot | RobotConfig | str, cameras: dict[str, dict[str, Any]] | None, **kwargs: Any
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

    def _migrate_legacy_calibration(self) -> None:
        """Copy a pre-0.5 SO-family calibration file to the new shared path.

        lerobot 0.5.1 unified ``so100_follower/`` + ``so101_follower/`` (and
        the leader variants) into a single ``so_follower/`` /
        ``so_leader/`` directory under ``HF_LEROBOT_CALIBRATION``. The robot's
        ``calibration_fpath`` now points at the NEW location; an existing
        customer's JSON sits at the OLD location and is never found, so
        ``is_calibrated`` is ``False`` and the first ``get_observation()``
        raises.

        This best-effort migration copies (never moves -- a still-installed
        old lerobot may read the original) the legacy file into place when:
          * the robot exposes a ``calibration_fpath`` (lerobot >=0.5), and
          * the NEW path does not already exist, and
          * exactly one matching legacy file is found.

        Any failure is logged and swallowed -- a calibration that can't be
        migrated simply leaves the robot in its pre-existing (uncalibrated)
        state, which the connect path already reports clearly.
        """
        try:
            new_path = getattr(self.robot, "calibration_fpath", None)
            if new_path is None:
                return
            new_path = Path(new_path)
            if new_path.is_file():
                return  # already calibrated at the new path; nothing to do

            # The shared dir is the parent (e.g. ``.../so_follower``); the
            # legacy dirs are siblings named after the concrete variant.
            shared_dir = new_path.parent  # so_follower / so_leader
            calib_root = shared_dir.parent  # HF_LEROBOT_CALIBRATION/robots
            shared_name = shared_dir.name  # "so_follower"
            file_name = new_path.name  # "<id>.json"

            # Only the SO-family was renamed; restrict to *_follower / *_leader
            # subdirs sharing the same role suffix so we don't pull an
            # unrelated robot's file.
            if shared_name not in ("so_follower", "so_leader"):
                return
            role = shared_name.split("_", 1)[1]  # "follower" | "leader"

            candidates = [
                p for p in calib_root.glob(f"*_{role}/{file_name}") if p.is_file() and p.parent.name != shared_name
            ]
            if len(candidates) != 1:
                # Zero -> nothing to migrate. >1 -> ambiguous, refuse to guess.
                if len(candidates) > 1:
                    logger.warning(
                        "Multiple legacy calibration files found for %s; skipping auto-migration to avoid guessing: %s",
                        file_name,
                        [str(c) for c in candidates],
                    )
                return

            old_path = candidates[0]
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            logger.info("Migrated calibration file: %s -> %s", old_path, new_path)
        except OSError as exc:
            logger.warning("Calibration auto-migration failed (%s); leaving as-is", exc)

    def _create_minimal_config(
        self, robot_type: str, cameras: dict[str, dict[str, Any]] | None, **kwargs: Any
    ) -> RobotConfig:
        """Create a minimal lerobot RobotConfig for ``robot_type``.

        Uses lerobot's draccus ``ChoiceRegistry`` to resolve the registered
        config subclass. This is the same lookup ``make_robot_from_config``
        performs internally and means we automatically support every robot
        lerobot ships (so100/so101, koch, openarm, unitree_g1, aloha, ...)
        without maintaining a hand-rolled mapping.

        Robot-specific kwargs (``port``, ``robot_ip``, ``kp``, ``kd``,
        ``default_positions``, ``calibration_dir``, ``mock``, ``use_degrees``,
        ``is_simulation``, ``control_dt``, ``gravity_compensation``,
        ``controller``, ``max_relative_target``, ``disable_torque_on_disconnect``)
        are forwarded if and only if the resolved config dataclass declares
        a matching field. This means kwargs that exist in the union-of-
        robots allowlist but not on the current robot's dataclass are
        dropped silently -- that is the deliberate cross-robot
        polymorphism (``Robot('so101', kp=[...])`` won't fail just because
        ``kp`` is a unitree_g1 thing).

        A kwarg that is NOT in the allowlist at all is rejected with
        ``ValueError`` rather than dropped, per AGENTS.md > Review
        Learnings (#86) > "Reject silently-dropped kwargs". This catches
        typos like ``prot=`` (instead of ``port=``) at config-build time
        rather than as a delayed connection failure with no kwarg in
        sight.
        """
        # ``lerobot`` is already a hard dep at this point (``_initialize_robot``
        # imports it eagerly). Importing the camera + config modules here is
        # cheap and the only reason it isn't at module top is that some
        # downstream packagers tree-shake unused submodules.
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.config import RobotConfig

        # Convert cameras to lerobot format.
        camera_configs: dict[str, Any] = {}
        if cameras:
            for name, config in cameras.items():
                cam_type = config.get("type", "opencv")
                if cam_type == "opencv":
                    camera_configs[name] = OpenCVCameraConfig(
                        index_or_path=config["index_or_path"],
                        fps=config.get("fps", 30),
                        width=config.get("width", 640),
                        height=config.get("height", 480),
                        rotation=config.get("rotation", 0),
                        color_mode=config.get("color_mode", "rgb"),
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {cam_type}")

        # Trigger lerobot's lazy registration. Each robot driver registers
        # its config via @RobotConfig.register_subclass at module-import
        # time, but ``lerobot.robots.__init__`` does NOT eagerly import
        # every driver subpackage (deliberate: keeps ``import lerobot``
        # cheap when only one robot is needed, and avoids hard deps on
        # robot-specific SDKs). The mapping ``robot_type → import path``
        # is also non-trivial:
        #
        #     so101_follower  → lerobot.robots.so_follower (shared module)
        #     hope_jr_arm     → lerobot.robots.hope_jr     (shared module)
        #     lekiwi_client   → lerobot.robots.lekiwi      (shared module)
        #
        # so we cannot just ``import_module(f"lerobot.robots.{robot_type}")``.
        # Instead we walk every subpackage of ``lerobot.robots`` once
        # (filesystem-driven, future-proof) and use lerobot's own
        # third-party plugin loader for ``lerobot_robot_*`` distributions.
        # Both calls are cached after the first invocation so subsequent
        # ``Robot()`` calls are essentially free.
        _ensure_lerobot_robots_registered()

        # Resolve the config class via lerobot's draccus ChoiceRegistry —
        # this is the source-of-truth lookup that ``make_robot_from_config``
        # uses; staying on it means we track upstream renames automatically.
        try:
            ConfigClass = RobotConfig.get_choice_class(robot_type)
        except KeyError:
            available = sorted(RobotConfig.get_known_choices().keys())
            # ``from None`` -- the KeyError is an internal detail of
            # lerobot's draccus registry; suppress the chained traceback
            # for a cleaner user-facing error.
            raise ValueError(
                f"Unsupported robot type: {robot_type!r}. Known lerobot robot types: {available}"
            ) from None

        # Build candidate field set so we only pass kwargs the dataclass
        # actually accepts. ``RobotConfig.get_choice_class`` always returns
        # a dataclass today (every ``@RobotConfig.register_subclass`` site
        # is a ``@dataclass``-decorated class). If that contract ever
        # breaks we want a loud error here, not a silent default that
        # blindly forwards every kwarg downstream (per AGENTS.md > Key
        # Conventions #6 -- "no silent defaults on error").
        try:
            valid_fields = {f.name for f in dataclasses.fields(ConfigClass)}
        except TypeError as exc:
            raise TypeError(
                f"lerobot returned a non-dataclass config class "
                f"{ConfigClass!r} for robot_type={robot_type!r}; strands_robots "
                f"cannot filter kwargs safely. Please file an issue against "
                f"lerobot or strands_robots."
            ) from exc

        config_data: dict[str, Any] = {}

        # ``id`` namespaces lerobot's calibration files. Users can override
        # by passing ``id=...`` (e.g. when one calibration file is shared by
        # multiple peer instances of the same robot type -- left_arm.json,
        # right_arm.json). Default to the strands tool name otherwise.
        if "id" in valid_fields:
            config_data["id"] = kwargs.get("id", self.tool_name_str)
        elif "id" in kwargs:
            # #292: every lerobot RobotConfig declares ``id`` today, so an
            # operator-supplied ``id=`` is normally consumed above. If a future
            # RobotConfig subclass drops the field, silently discarding an
            # explicit ``id=`` would namespace calibration files wrong with no
            # signal. Surface it: the generic unknown-kwarg gate below would
            # also catch it, but this names the specific regression so the
            # diagnostic is actionable.
            logger.warning(
                "[hardware_robot] robot_type=%r config %s does not declare an 'id' "
                "field; the explicit id=%r will not namespace calibration files. "
                "This is unexpected for a lerobot RobotConfig -- please file an issue.",
                robot_type,
                ConfigClass.__name__,
                kwargs["id"],
            )

        # Cameras are common to every lerobot Robot.
        if "cameras" in valid_fields:
            config_data["cameras"] = camera_configs

        # Forward known robot-specific kwargs only if the target dataclass
        # declares them. The full set is union-of-all known lerobot robot
        # configs — adding new ones here is safe because we filter against
        # ``valid_fields`` before constructing.
        forwardable = _FORWARDABLE_KWARGS
        for key in forwardable:
            if key in kwargs and key in valid_fields:
                config_data[key] = kwargs[key]
            elif key in kwargs:
                # #294/#297: the kwarg is in the cross-robot allowlist but the
                # resolved dataclass does not declare it -- the documented
                # polymorphism carve-out (e.g. ``Robot('so101', kp=[...])``
                # against a heterogeneous fleet). This is intentional, but a
                # silent drop leaves operators with no way to audit why a kwarg
                # they passed had no effect. Emit a debug signal naming the
                # dropped kwarg and the robot type so the drop is observable
                # without changing the tolerant behaviour.
                logger.debug(
                    "[hardware_robot] dropping cross-robot kwarg %r for robot_type=%r: "
                    "not declared on %s (forwardable-allowlist polymorphism carve-out)",
                    key,
                    robot_type,
                    ConfigClass.__name__,
                )

        # Forward kwargs that are declared on the target dataclass but not
        # in the cross-robot allowlist. This future-proofs new lerobot fields
        # without requiring a strands_robots release to add them to forwardable.
        for key in kwargs:
            if key not in config_data and key not in {"id", "cameras"} and key in valid_fields:
                config_data[key] = kwargs[key]

        # Reject kwargs unknown to BOTH the cross-robot allowlist AND the
        # resolved target dataclass. Per AGENTS.md > Review Learnings (#86)
        # > "Reject silently-dropped kwargs", a typo like ``prot=`` must
        # surface immediately -- but a genuinely new lerobot field that the
        # target dataclass declares should Just Work without a strands_robots
        # release. This keeps typo-rejection while preserving the "zero
        # strands_robots changes for new robots" promise for new *kwargs* too.
        always_allowed = {"id", "cameras"}
        recognised = set(forwardable) | always_allowed | valid_fields
        unknown = set(kwargs) - recognised
        if unknown:
            raise ValueError(
                f"Unknown kwarg(s) for robot_type={robot_type!r}: "
                f"{sorted(unknown)}. This robot's dataclass accepts: "
                f"{sorted(valid_fields)}. The cross-robot allowlist is: "
                f"{sorted(set(forwardable) | always_allowed)}. "
                f"(If this is a typo, fix it.)"
            )

        try:
            return ConfigClass(**config_data)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to construct {ConfigClass.__name__} for robot type {robot_type!r}: {e}. Config: {config_data}"
            ) from e

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
                logger.info(f"{self.robot} already connected")
                return True, ""

            logger.info(f"Connecting to {self.robot}...")

            # Handle robot connection using lerobot's error handling patterns
            try:
                if not self.robot.is_connected:
                    await asyncio.to_thread(self.robot.connect, False)  # calibrate=False

            except DeviceAlreadyConnectedError:
                # This is expected and fine - robot is already connected
                logger.info(f"{self.robot} was already connected")

            except Exception as e:
                # Check if it's the string version of "already connected" error
                error_str = str(e).lower()
                if "already connected" in error_str or "is already connected" in error_str:
                    logger.info(f"{self.robot} connection already established")
                else:
                    # Re-raise if it's a different error
                    raise e

            # Final connection check
            if not self.robot.is_connected:
                error_msg = f"Failed to connect to {self.robot}"
                logger.error(f"{error_msg}")
                return False, error_msg

            # Check robot calibration
            if hasattr(self.robot, "is_calibrated") and not self.robot.is_calibrated:
                error_msg = (
                    f"Robot {self.robot} is not calibrated. Please calibrate the robot manually"
                    " first using LeRobot's calibration process (lerobot-calibrate)"
                )
                logger.error(f"{error_msg}")
                return False, error_msg

            logger.info(f"{self.robot} connected and ready")
            return True, ""

        except Exception as e:
            error_msg = f"Robot connection failed: {e}. Ensure robot is calibrated and accessible on the specified port"
            logger.error(f"{error_msg}")
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
            logger.error(f"Failed to initialize policy: {e}")
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

            logger.info(f"Starting task: '{instruction}' on {self.tool_name_str}")
            logger.info(f"Using policy: {policy_provider} on {policy_host}:{policy_port}")

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
                logger.info(f"Task completed: '{instruction}' in {elapsed:.1f}s ({self._task_state.step_count} steps)")

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
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
        async def task_runner() -> None:
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
                    "text": f"Task: '{instruction}' - {self._task_state.status.value}\n"
                    f"Robot: {self.tool_name_str} ({self.robot})\n"
                    f"Policy: {policy_provider} on {policy_host}:{policy_port}\n"
                    f"Duration: {self._task_state.duration:.1f}s\n"
                    f"Steps: {self._task_state.step_count}"
                    + (f"\nError: {self._task_state.error_message}" if self._task_state.error_message else "")
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
                "content": [{"text": f"Task already running: {self._task_state.instruction}"}],
            }

        # Start task in background
        self._task_state.task_future = self._executor.submit(
            self._execute_task_sync, instruction, policy_port, policy_host, policy_provider, duration
        )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Task started: '{instruction}'\n"
                    f"Robot: {self.tool_name_str}\n"
                    f"Use action='status' to check progress\n"
                    f"Use action='stop' to interrupt"
                }
            ],
        }

    def get_task_status(self) -> dict[str, Any]:
        """Get current task execution status."""

        # Update duration for running tasks
        if self._task_state.status == TaskStatus.RUNNING:
            self._task_state.duration = time.time() - self._task_state.start_time

        status_text = f"Robot Status: {self._task_state.status.value.upper()}\n"

        if self._task_state.instruction:
            status_text += f"Task: {self._task_state.instruction}\n"

        if self._task_state.status == TaskStatus.RUNNING:
            status_text += f"Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"Steps: {self._task_state.step_count}\n"
        elif self._task_state.status in [TaskStatus.COMPLETED, TaskStatus.STOPPED, TaskStatus.ERROR]:
            status_text += f"Total Duration: {self._task_state.duration:.1f}s\n"
            status_text += f"Total Steps: {self._task_state.step_count}\n"

        if self._task_state.error_message:
            status_text += f"Error: {self._task_state.error_message}\n"

        return {
            "status": "success",
            "content": [{"text": status_text}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop currently running task."""

        if self._task_state.status != TaskStatus.RUNNING:
            return {
                "status": "success",
                "content": [{"text": f"No task running to stop (current: {self._task_state.status.value})"}],
            }

        # Signal task to stop
        self._task_state.status = TaskStatus.STOPPED

        # Cancel future if it exists
        if self._task_state.task_future:
            self._task_state.task_future.cancel()

        logger.info(f"Task stopped: {self._task_state.instruction}")

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Task stopped: '{self._task_state.instruction}'\n"
                    f"Duration: {self._task_state.duration:.1f}s\n"
                    f"Steps completed: {self._task_state.step_count}"
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
                                "content": [{"text": "instruction and policy_port are required for execute action"}],
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
                                "content": [{"text": "instruction and policy_port are required for start action"}],
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
                                {"text": f"Unknown action: {action}. Valid actions: execute, start, status, stop"}
                            ],
                        },
                    )
                )

        except Exception as e:
            logger.error(f"{self.tool_name_str} error: {e}")
            yield ToolResultEvent(
                self._make_tool_result(
                    tool_use_id,
                    {
                        "status": "error",
                        "content": [{"text": f"{self.tool_name_str} error: {str(e)}"}],
                    },
                )
            )

    def cleanup(self) -> None:
        """Cleanup resources and stop any running tasks."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Stop any running task
            if self._task_state.status == TaskStatus.RUNNING:
                self.stop_task()

            # Shutdown executor
            self._executor.shutdown(wait=True)

            # Tear down the Zenoh mesh component if one was attached.
            # ``self.mesh`` is any object exposing ``.stop()``; falsy values
            # (None — the construction-time default and what a hardware robot
            # gets when ``mesh=False``) are skipped silently.
            if self.mesh:
                try:
                    self.mesh.stop()
                except Exception as mesh_exc:  # noqa: BLE001
                    # Mesh teardown should never block hardware cleanup.
                    logger.warning(
                        "%s: mesh.stop() raised during cleanup: %s",
                        self.tool_name_str,
                        mesh_exc,
                    )

            logger.info(f"{self.tool_name_str} cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error for {self.tool_name_str}: {e}")

    def __del__(self) -> None:
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
            logger.error(f"Error getting status for {self.tool_name_str}: {e}")
            return {
                "robot_name": self.tool_name_str,
                "error": str(e),
                "is_connected": False,
                "task_status": "error",
            }

    async def stop(self) -> None:
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

            logger.info(f"{self.tool_name_str} stopped and disconnected")

        except Exception as e:
            logger.error(f"Error stopping robot: {e}")

    # ------------------------------------------------------------------
    # Teleoperation over mesh — input publishing and receiving
    # ------------------------------------------------------------------

    def start_teleop_publish(
        self,
        teleoperator: Any,
        device_name: str = "leader",
        method: str = "arm",
        hz: float = 50.0,
    ) -> dict[str, Any]:
        """Start publishing teleoperator actions to the mesh.

        This makes the robot a *teleop source*: another peer on the mesh
        can call ``start_teleop_receive(source_peer_id=self.peer_id)`` to
        have its hardware follow along.

        Args:
            teleoperator: Any object with a ``get_action() -> dict`` method.
                Typically a lerobot Teleoperator (SOLeader, GamepadTeleop,
                KeyboardTeleop, Phone).
            device_name: Name for this input stream (e.g. "leader", "gamepad").
            method: Input method label ("arm", "gamepad", "keyboard", "phone").
            hz: Publishing frequency in Hz.

        Returns:
            Status dict with topic and peer_id for the receiver to use.
        """
        if not self.mesh or not self.mesh.alive:
            return {"status": "error", "content": [{"text": "Mesh not active. Cannot publish input."}]}

        from strands_robots.mesh import InputPublisher

        # Store publisher on the robot instance
        if not hasattr(self, "_input_publishers"):
            self._input_publishers: dict[str, InputPublisher] = {}

        if device_name in self._input_publishers:
            # Stop existing publisher for this device
            self._input_publishers[device_name].stop()

        publisher = InputPublisher(
            mesh=self.mesh,
            teleoperator=teleoperator,
            device_name=device_name,
            method=method,
            hz=hz,
        )
        publisher.start()
        self._input_publishers[device_name] = publisher

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Input publisher started: {device_name} ({method} @ {hz}Hz)\n"
                    f"Topic: {publisher.topic}\n"
                    f"Peer ID: {self.peer_id}\n"
                    f"Remote peers can receive with: start_teleop_receive(source_peer_id='{self.peer_id}')"
                }
            ],
        }

    def start_teleop_receive(
        self,
        source_peer_id: str,
        device_name: str = "leader",
        apply_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Start receiving teleoperator actions from a remote peer and applying to hardware.

        This makes the robot a *teleop follower*: it listens for input frames
        published by the source peer and applies them to its own hardware via
        ``self.robot.send_action(action)``.

        Args:
            source_peer_id: The peer ID of the publishing robot.
            device_name: Name of the input stream to subscribe to.
            apply_fn: Optional custom function ``(robot, action_dict) -> None``.
                Defaults to calling ``robot.send_action(action)``.

        Returns:
            Status dict.
        """
        if not self.mesh or not self.mesh.alive:
            return {"status": "error", "content": [{"text": "Mesh not active. Cannot receive input."}]}

        from strands_robots.mesh import InputReceiver

        if not hasattr(self, "_input_receivers"):
            self._input_receivers: dict[str, InputReceiver] = {}

        key = f"{source_peer_id}/{device_name}"
        if key in self._input_receivers:
            self._input_receivers[key].stop()

        receiver = InputReceiver(
            mesh=self.mesh,
            robot=self.robot,
            source_peer_id=source_peer_id,
            device_name=device_name,
            apply_fn=apply_fn,
        )
        receiver.start()
        self._input_receivers[key] = receiver

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Input receiver started: listening to {source_peer_id}/{device_name}\n"
                    f"Topic: {receiver.topic}\n"
                    f"Actions will be applied to: {self.tool_name_str}"
                }
            ],
        }

    def stop_teleop(self, device_name: str | None = None) -> dict[str, Any]:
        """Stop all or a specific teleop publisher/receiver.

        Args:
            device_name: If provided, stop only the named publisher/receiver.
                If None, stop all.

        Returns:
            Stats from stopped sessions.
        """
        results = []

        # Stop publishers
        if hasattr(self, "_input_publishers"):
            if device_name:
                pub = self._input_publishers.pop(device_name, None)
                if pub:
                    results.append(pub.stop())
            else:
                for name, pub in list(self._input_publishers.items()):
                    results.append(pub.stop())
                self._input_publishers.clear()

        # Stop receivers
        if hasattr(self, "_input_receivers"):
            if device_name:
                # Match by device name suffix
                to_remove = [k for k in self._input_receivers if k.endswith(f"/{device_name}")]
                for k in to_remove:
                    results.append(self._input_receivers.pop(k).stop())
            else:
                for key, rcv in list(self._input_receivers.items()):
                    results.append(rcv.stop())
                self._input_receivers.clear()

        if not results:
            return {"status": "success", "content": [{"text": "No active teleop sessions."}]}

        stats_text = "\n".join(
            f"  {r.get('device', r.get('source', '?'))}: "
            f"{r.get('frames', r.get('frames_received', 0))} frames, "
            f"{r.get('hz_actual', 0):.1f} Hz"
            for r in results
        )
        return {
            "status": "success",
            "content": [{"text": f"Teleop stopped:\n{stats_text}"}],
        }

    def get_teleop_status(self) -> dict[str, Any]:
        """Get status of all active teleop sessions."""
        publishers = {}
        receivers = {}

        if hasattr(self, "_input_publishers"):
            for name, pub in self._input_publishers.items():
                publishers[name] = pub.stats

        if hasattr(self, "_input_receivers"):
            for key, rcv in self._input_receivers.items():
                receivers[key] = rcv.stats

        return {
            "status": "success",
            "publishers": publishers,
            "receivers": receivers,
            "content": [
                {
                    "text": f"Teleop status:\n"
                    f"  Publishers: {len(publishers)} active\n"
                    f"  Receivers: {len(receivers)} active\n"
                    + "".join(
                        f"  [pub] {n}: {s.get('frames', 0)} frames @ {s.get('hz_actual', 0):.1f}Hz\n"
                        for n, s in publishers.items()
                    )
                    + "".join(
                        f"  [rcv] {k}: {s.get('frames_received', 0)} frames @ {s.get('hz_actual', 0):.1f}Hz\n"
                        for k, s in receivers.items()
                    )
                }
            ],
        }
