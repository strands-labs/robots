#!/usr/bin/env python3
"""
LeRobot teleoperation tool with recording capabilities for robot training data collection.

This tool integrates teleoperation and recording functionality from lerobot, allowing users to:
- Control robots through teleoperation devices
- Record demonstrations for training machine learning models
- Replay recorded episodes
- Manage multiple teleoperation sessions
"""

import importlib.util
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
from strands import tool

from strands_robots.utils import (
    boolean_flag_error,
    non_negative_whole_number_error,
    positive_finite_number_error,
    positive_whole_number_error,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session storage directory
SESSION_DIR = Path.cwd() / ".strands_robots/.sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# The numeric knobs each command mode actually puts on the lerobot argv. Every
# one is interpolated with ``str()`` into the command line of a DETACHED
# subprocess, so a value the lerobot CLI cannot parse is never reported by the
# call that supplied it: the session starts, ``status="success"`` is returned,
# and the failure appears minutes later in that session's log. A value the CLI
# *can* parse but should not have been given - a zero recording rate, a negative
# episode index - is worse, because nothing reports it at all.
#
# Refusing a knob the requested mode never emits would be a false rejection, so
# the scoping is driven by this table rather than by validating the whole
# signature unconditionally.
_MODE_NUMERIC_OPTIONS: dict[str, tuple[str, ...]] = {
    "replay": ("replay_episode", "dataset_fps"),
    "record": ("dataset_num_episodes", "dataset_fps", "dataset_episode_time_s", "dataset_reset_time_s"),
    "teleoperate": ("fps", "teleop_time_s"),
    "dagger": ("dagger_num_episodes", "dataset_num_episodes", "dataset_fps", "fps"),
}

# The domain each knob is checked against, in the order errors are reported.
# The whole-number knobs are declared ``int`` by lerobot's own config
# dataclasses (``DatasetRecordConfig.fps`` / ``.num_episodes``,
# ``TeleoperateConfig.fps``, ``DatasetReplayConfig.episode``) and by this
# module's own signature, so a whole number is the honest domain and an integral
# float read from a config is still honored. ``teleop_time_s`` is lerobot's
# ``float | None`` session budget, where a fractional value is perfectly usable.
#
# Two knobs take the non-negative floor instead of the positive one:
# ``dataset_reset_time_s=0`` (no operator pause between episodes) and
# ``replay_episode=0`` (the first episode) are both real requests, whereas a
# zero recording rate, a zero-length episode and a zero-episode recording are
# each a request no run can satisfy.
_OPTION_DOMAINS: tuple[tuple[str, Callable[[Any, str, str], str | None]], ...] = (
    ("dataset_fps", positive_whole_number_error),
    ("dataset_num_episodes", positive_whole_number_error),
    ("dataset_episode_time_s", positive_whole_number_error),
    ("dataset_reset_time_s", non_negative_whole_number_error),
    ("dagger_num_episodes", positive_whole_number_error),
    ("fps", positive_whole_number_error),
    ("replay_episode", non_negative_whole_number_error),
    ("teleop_time_s", positive_finite_number_error),
)

# Knobs whose ``None`` means "omit the flag and take the lerobot default", so
# there ``None`` is a supplied value rather than an unusable one. Every other
# knob in the table has a non-None default, so ``None`` is a caller mistake.
_OPTIONAL_OPTIONS: frozenset[str] = frozenset({"teleop_time_s", "dagger_num_episodes"})


def _numeric_option_error(mode: str, supplied: dict[str, Any]) -> str | None:
    """Error text for the first numeric knob ``mode`` emits but cannot honor.

    Args:
        mode: A key of :data:`_MODE_NUMERIC_OPTIONS`; decides which knobs are
            effective. A mode absent from that map emits none of them and is
            never refused here.
        supplied: Every knob in :data:`_OPTION_DOMAINS`, as supplied by the
            caller.

    Returns:
        An error message naming the knob and its domain, or ``None`` when every
        knob this mode emits is usable.
    """
    consumed = set(_MODE_NUMERIC_OPTIONS.get(mode, ()))
    for param, check in _OPTION_DOMAINS:
        if param not in consumed:
            continue
        value = supplied[param]
        if value is None and param in _OPTIONAL_OPTIONS:
            continue
        if error := check(value, param, "build_lerobot_command"):
            return error
    return None


# The boolean flags each command mode actually puts on the lerobot argv, in the
# order that argv emits them. Each selects a posture rather than scaling a
# quantity - ``dataset_video`` picks the literal ``"true"`` or ``"false"``, while
# ``record_resume`` decides whether ``--resume true`` appears at all - and both
# spellings were read by truthiness, where every non-empty string is truthy. So
# the words an operator reaches for when opting out selected the *opposite*
# posture from the one they read as, on a command line the supplying call cannot
# read a failure back from. Measured on ``3ce3da7``:
#
#   dataset_push_to_hub="false"   ->  --dataset.push_to_hub true
#   record_resume="false"         ->  --resume true
#   dagger_record_autonomous="off"->  --strategy.record_autonomous true
#
# The first uploads an unattended recording to the Hub, the second appends into
# an existing dataset instead of creating the fresh one that was asked for, and
# the third records autonomous episodes into a corrections dataset. ``None`` and
# ``[]`` took the other branch just as silently, without ever being a declared
# spelling of it.
#
# ``teleoperate`` carries only ``display_data`` because lerobot's
# ``TeleoperateConfig`` declares no other field this module offers: refusing a
# flag a mode never puts on its argv would be a false rejection, the same scoping
# rule :data:`_MODE_NUMERIC_OPTIONS` encodes.
#
# ``play_sounds`` is scoped by that same table rather than exempted from it. It
# was previously in no tuple because no mode emitted it at all - declared,
# documented and forwarded, and then read by nothing (#2072), so a domain would
# have refused values for an option that had no effect either way. It is now
# emitted for the three entry points that accept it, which is what makes the
# domain honest rather than decorative. Measured against ``lerobot==0.6.1``, the
# version this package's ``[lerobot]`` extra floors:
#
#   lerobot/scripts/lerobot_record.py:188   RecordConfig.play_sounds  = True
#   lerobot/scripts/lerobot_replay.py:99    ReplayConfig.play_sounds  = True
#   lerobot/rollout/configs.py:256          RolloutConfig.play_sounds = True
#   lerobot/scripts/lerobot_teleoperate.py  TeleoperateConfig - absent
#
# All three are fields of the top-level ``@parser.wrap()`` config, so the flag is
# spelled ``--play_sounds`` rather than nested. ``teleoperate`` is excluded
# because that entry point never speaks - it holds no ``log_say`` call and no
# such field - so emitting the flag there would not be a silent no-op but an
# unrecognized argument, which is the failure the class docstring above warns the
# removed pre-0.5 flat flags cause.
#
# ``play_sounds`` is emitted *unconditionally* as an explicit literal, like
# ``dataset_video`` and ``dataset_push_to_hub``, rather than only when set, like
# ``display_data`` and ``record_resume``. That split is not a style choice: it
# follows the upstream default. A flag lerobot defaults to ``False`` is expressed
# by its presence, so omitting it says "false" exactly; a flag lerobot defaults
# to ``True`` cannot express an opt-out by omission at all, so the only spelling
# of ``play_sounds=False`` that reaches the CLI is the explicit literal.
_MODE_FLAG_OPTIONS: dict[str, tuple[str, ...]] = {
    "record": ("record_resume", "dataset_push_to_hub", "dataset_video", "display_data", "play_sounds"),
    "teleoperate": ("display_data",),
    "replay": ("play_sounds",),
    "dagger": (
        "dagger_record_autonomous",
        "dataset_push_to_hub",
        "dataset_video",
        "display_data",
        "play_sounds",
    ),
}


def _flag_error(mode: str, supplied: dict[str, Any]) -> str | None:
    """Error text for the first boolean flag ``mode`` emits but cannot honor.

    Each flag is checked against the shared
    :func:`~strands_robots.utils.boolean_flag_error` domain - the one the mesh
    provisioning entry points already apply - so a posture flag is refused
    identically wherever it is supplied rather than merely equivalently.

    No coercion follows the check, unlike
    :func:`~strands_robots.mesh.iot.bootstrap.bootstrap_account`: every reader
    here either selects a literal or gates a ``cmd.extend``, and the numpy
    booleans ``boolean_flag_error`` also accepts drive both correctly.

    Args:
        mode: A key of :data:`_MODE_FLAG_OPTIONS`; decides which flags are
            effective. A mode absent from that map emits none of them and is
            never refused here.
        supplied: Every flag named in :data:`_MODE_FLAG_OPTIONS`, as supplied by
            the caller.

    Returns:
        An error message naming the flag and its domain, or ``None`` when every
        flag this mode emits is usable.
    """
    for param in _MODE_FLAG_OPTIONS.get(mode, ()):
        if error := boolean_flag_error(supplied[param], param, "build_lerobot_command"):
            return error
    return None


# The two flags the per-mode table above cannot scope, because neither is a
# parameter of :func:`build_lerobot_command`: no mode emits them and no argv
# records them. They choose how :func:`lerobot_teleoperate` *runs* the command it
# has already built - ``background`` gates the detach, log file and session
# persistence, and ``auto_accept_calibration`` gates a thread writing two
# newlines into the child's stdin ~2s in, which accepts whatever calibration
# prompt the robot is showing.
#
# Both were read as ``if <flag>:``, and every non-empty string is truthy, so an
# opt-out selected the affirmative posture (#2074, measured on ``3ce3da7``):
#
#   background="false"               ->  detached, not the foreground run asked for
#   auto_accept_calibration="false"  ->  a calibration accepted by no operator
#
# ``auto_accept_calibration`` is the sharper of the two. The posture
# ``background`` reaches is at least reported - the returned envelope carries
# ``"background": True``, a pid and a log file, so a caller who asked for the
# foreground can see it did not get it - while nothing anywhere reports that
# stdin was written to.
#
# A flat tuple rather than a per-mode map: both are read unconditionally for a
# ``start`` / ``dagger`` call, so the effectiveness question
# :data:`_MODE_FLAG_OPTIONS` answers does not arise for them. They are read by no
# other action, which is why the check is scoped to that branch rather than to
# the whole tool - refusing ``background`` for an ``action="list"`` that never
# reads it would be a false rejection.
_EXECUTION_FLAGS: tuple[str, ...] = ("background", "auto_accept_calibration")


def _execution_flag_error(supplied: dict[str, Any]) -> str | None:
    """Error text for the first execution-posture flag the tool cannot honor.

    The same shared :func:`~strands_robots.utils.boolean_flag_error` domain
    :func:`_flag_error` applies to the argv flags, so a posture flag is refused
    identically wherever it is supplied rather than merely equivalently. Only the
    context differs: these are refused by the tool and not by the builder, and
    the message says so.

    Args:
        supplied: Every flag named in :data:`_EXECUTION_FLAGS`, as supplied by
            the caller.

    Returns:
        An error message naming the flag and its domain, or ``None`` when both
        can be honoured.
    """
    for param in _EXECUTION_FLAGS:
        if error := boolean_flag_error(supplied[param], param, "lerobot_teleoperate"):
            return error
    return None


class SessionManager:
    """Manage teleoperation sessions with persistence."""

    def __init__(self):
        self.sessions_file = SESSION_DIR / "active_sessions.json"

    def _load_sessions(self) -> dict[str, Any]:
        """Load the session store, pruning records whose process is gone.

        ``psutil.pid_exists`` answers whether the PID exists;
        ``Process(pid).is_running()`` refines that (it also rules out PID reuse).
        The two probes can disagree, and the two ways they disagree mean opposite
        things, so they are handled separately:

        * :class:`psutil.NoSuchProcess` - the process was reaped between the two
          calls. The record names nothing, so it is pruned.
        * :class:`psutil.AccessDenied` - the process exists (``pid_exists`` just
          said so) but this user may not inspect it; a session started under
          ``sudo`` for serial-port access and then listed as the invoking user
          reads this way. That is not death, so the record is kept.

        Keeping it matters because the prune below is *written back to disk* and
        this store is the only place a detached session's PID is recorded: a
        pruned record leaves the teleoperation process running with no supported
        way to stop it. Presence here is not the running claim - ``list`` and
        ``status`` each derive that from ``pid_exists`` - so a retained record is
        reported running only while its PID really exists.

        Returns:
            The surviving session records, keyed by session name.
        """
        if not self.sessions_file.exists():
            return {}

        try:
            with open(self.sessions_file) as f:
                sessions = json.load(f)

            # Check if processes are still running and clean up dead sessions
            active_sessions = {}
            for name, info in sessions.items():
                pid = info.get("pid")
                if pid and psutil.pid_exists(pid):
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            active_sessions[name] = info
                    except psutil.NoSuchProcess:
                        # Reaped between pid_exists and this probe: the record
                        # names nothing, so pruning it loses no live session.
                        pass
                    except psutil.AccessDenied:
                        # Exists but not inspectable: keep the record (see above)
                        # and say so, because the store is written back below and
                        # silence here loses the PID for good.
                        active_sessions[name] = info
                        logger.warning(
                            "Teleop session PID %s exists but cannot be inspected; "
                            "keeping its record so the session stays stoppable",
                            pid,
                        )

            # Update sessions file with only active sessions
            if len(active_sessions) != len(sessions):
                self._save_sessions(active_sessions)

            return active_sessions

        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading sessions: {e}")
            return {}

    def _save_sessions(self, sessions: dict[str, Any]):
        """Save sessions to disk."""
        try:
            with open(self.sessions_file, "w") as f:
                json.dump(sessions, f, indent=2)
        except OSError as e:
            logger.error(f"Error saving sessions: {e}")

    def add_session(self, name: str, info: dict[str, Any]):
        """Add a new session."""
        sessions = self._load_sessions()
        sessions[name] = info
        self._save_sessions(sessions)

    def remove_session(self, name: str):
        """Remove a session."""
        sessions = self._load_sessions()
        if name in sessions:
            del sessions[name]
            self._save_sessions(sessions)

    def get_session(self, name: str) -> dict[str, Any] | None:
        """Get session info."""
        sessions = self._load_sessions()
        return sessions.get(name)

    def list_sessions(self) -> dict[str, Any]:
        """List all active sessions."""
        return self._load_sessions()


def _build_camera_arg(robot_cameras: dict[str, Any]) -> str:
    """Render a camera map as a lerobot 0.5 nested ``--robot.cameras`` value.

    lerobot 0.5's draccus CLI parses ``--robot.cameras`` as a nested dict, e.g.
    ``{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}``.
    The pre-0.5 ``--camera-config name=type:path:fps:WxH`` flat form no longer
    exists. Each entry defaults to opencv/index 0/640x480/30fps when unset.

    Args:
        robot_cameras: Map of camera name to a config dict with optional keys
            ``type``, ``index_or_path``, ``width``, ``height``, ``fps``.

    Returns:
        The nested dict string suitable for ``--robot.cameras=<value>``.
    """
    entries = []
    for cam_name, cam_config in robot_cameras.items():
        cam_type = cam_config.get("type", "opencv")
        cam_path = cam_config.get("index_or_path", 0)
        fps_val = cam_config.get("fps", 30)
        width = cam_config.get("width", 640)
        height = cam_config.get("height", 480)
        entries.append(
            f"{cam_name}: {{type: {cam_type}, index_or_path: {cam_path}, "
            f"width: {width}, height: {height}, fps: {fps_val}}}"
        )
    return "{" + ", ".join(entries) + "}"


def _robot_args(
    robot_type: str,
    robot_port: str | None,
    robot_id: str | None,
    robot_left_arm_port: str | None,
    robot_right_arm_port: str | None,
    robot_cameras: dict[str, Any] | None,
) -> list[str]:
    """Build the nested ``--robot.*`` argv for lerobot 0.5's draccus CLI.

    Single-arm robots bind ``--robot.port``; bimanual robots have no single port
    and instead pass ``--robot.left_arm_port`` / ``--robot.right_arm_port``.
    """
    args = ["--robot.type", robot_type]
    if robot_port:
        args.extend(["--robot.port", robot_port])
    if robot_id:
        args.extend(["--robot.id", robot_id])
    if robot_left_arm_port:
        args.extend(["--robot.left_arm_port", robot_left_arm_port])
    if robot_right_arm_port:
        args.extend(["--robot.right_arm_port", robot_right_arm_port])
    if robot_cameras:
        args.append(f"--robot.cameras={_build_camera_arg(robot_cameras)}")
    return args


def _teleop_args(
    teleop_type: str | None,
    teleop_port: str | None,
    teleop_id: str | None,
    teleop_left_arm_port: str | None,
    teleop_right_arm_port: str | None,
) -> list[str]:
    """Build the nested ``--teleop.*`` argv for lerobot 0.5's draccus CLI."""
    args: list[str] = []
    if teleop_type:
        args.extend(["--teleop.type", teleop_type])
    if teleop_id:
        args.extend(["--teleop.id", teleop_id])
    if teleop_port:
        args.extend(["--teleop.port", teleop_port])
    if teleop_left_arm_port:
        args.extend(["--teleop.left_arm_port", teleop_left_arm_port])
    if teleop_right_arm_port:
        args.extend(["--teleop.right_arm_port", teleop_right_arm_port])
    return args


def build_lerobot_command(
    action: str,
    robot_type: str,
    robot_port: str | None = None,
    robot_id: str | None = None,
    robot_cameras: dict[str, Any] | None = None,
    robot_left_arm_port: str | None = None,
    robot_right_arm_port: str | None = None,
    teleop_type: str | None = None,
    teleop_port: str | None = None,
    teleop_id: str | None = None,
    teleop_left_arm_port: str | None = None,
    teleop_right_arm_port: str | None = None,
    dataset_repo_id: str | None = None,
    dataset_single_task: str | None = None,
    dataset_num_episodes: int = 50,
    dataset_fps: int = 30,
    dataset_episode_time_s: int = 60,
    dataset_reset_time_s: int = 60,
    dataset_root: str | None = None,
    dataset_video: bool = True,
    dataset_push_to_hub: bool = False,
    record_resume: bool = False,
    replay_episode: int = 0,
    display_data: bool = False,
    fps: int = 60,
    teleop_time_s: float | None = None,
    play_sounds: bool = True,
    # DAgger / teleop-takeover (lerobot-rollout --strategy.type=dagger)
    policy_path: str | None = None,
    dagger_record_autonomous: bool = False,
    dagger_input_device: str = "keyboard",
    dagger_num_episodes: int | None = None,
    **kwargs,
) -> list[str]:
    """Build the lerobot CLI argv for a teleoperate/record/replay/dagger action.

    Emits the lerobot 0.5 draccus-nested schema (``--robot.* / --teleop.* /
    --dataset.*``). The pre-0.5 flat flags (``--robot-path``, ``--repo-id``,
    ``--num-episodes``, ``--single-task``, ``--episode``, ``--no-video`` ...) were
    removed upstream and are NOT emitted; passing them makes the lerobot scripts
    exit with an unrecognized-argument error.

    Args:
        action: One of ``"start"`` (teleoperate or record) or ``"replay"``.
            ``"start"`` records when ``dataset_repo_id`` is set, otherwise it
            runs plain teleoperation.
        robot_type: Follower robot type (e.g. ``"so101_follower"``).
        dataset_repo_id: HuggingFace dataset id; presence enables record mode for
            ``start`` and is required for ``replay``.
        record_resume: For record mode, emit ``--resume true`` to append to an
            existing dataset (preserving its repo_id) instead of creating a fresh
            one. A fresh record always emits an explicit ``--dataset.root``
            (resolved from ``dataset_repo_id``) so lerobot HEAD's repo_id
            timestamp-stamping never relocates the on-disk dataset.
        replay_episode: Episode index to replay (``--dataset.episode``).
        play_sounds: Emit ``--play_sounds true|false`` for the modes whose lerobot
            entry point declares the field - ``record``, ``replay`` and
            ``dagger``. Always explicit, because lerobot defaults it to ``True``
            and an opt-out therefore cannot be expressed by omitting the flag.
            Plain teleoperation ignores it: ``TeleoperateConfig`` has no such
            field, so emitting it there would be an unrecognized argument.

            Every caller reaching a mode that emits it must forward it. Making
            the builder honor a flag is only half the fix, because the tool spec
            sits on :func:`lerobot_teleoperate` rather than here: the ``replay``
            dispatch omitted this kwarg, so the builder's default won whatever
            the agent asked for, and the argv read ``--play_sounds true`` on the
            only model-reachable path to replay. Pinned at the tool level, not
            just here, since a builder-level test cannot observe a dropped
            forward.

    Returns:
        The argv list, beginning with ``["python", "-m", "lerobot.scripts...."]``.

    Raises:
        ValueError: If ``action`` is unknown, ``replay`` is requested without
            ``dataset_repo_id``, or a numeric knob (see :data:`_OPTION_DOMAINS`)
            or boolean flag (see :data:`_MODE_FLAG_OPTIONS`) the requested mode
            emits cannot be honored. The refusal precedes the argv, so no
            subprocess is launched.
    """
    # Every numeric knob below is interpolated into a detached subprocess's
    # command line, which is not a channel this call can read a failure back
    # from. Check the ones this mode actually emits before building the argv.
    numeric_options: dict[str, Any] = {
        "dataset_fps": dataset_fps,
        "dataset_num_episodes": dataset_num_episodes,
        "dataset_episode_time_s": dataset_episode_time_s,
        "dataset_reset_time_s": dataset_reset_time_s,
        "dagger_num_episodes": dagger_num_episodes,
        "fps": fps,
        "replay_episode": replay_episode,
        "teleop_time_s": teleop_time_s,
    }
    # The boolean flags reach that same command line, but as a posture rather
    # than a magnitude, so an unusable value does not fail there - it selects the
    # other posture and reports success. Same per-mode scoping.
    flag_options: dict[str, Any] = {
        "record_resume": record_resume,
        "dataset_push_to_hub": dataset_push_to_hub,
        "dataset_video": dataset_video,
        "display_data": display_data,
        "dagger_record_autonomous": dagger_record_autonomous,
        "play_sounds": play_sounds,
    }
    if action == "replay":
        if not dataset_repo_id:
            raise ValueError("dataset_repo_id is required for replay action")
        if error := _numeric_option_error("replay", numeric_options):
            raise ValueError(error)
        if error := _flag_error("replay", flag_options):
            raise ValueError(error)
        cmd = ["python", "-m", "lerobot.scripts.lerobot_replay"]
        cmd.extend(
            _robot_args(robot_type, robot_port, robot_id, robot_left_arm_port, robot_right_arm_port, robot_cameras)
        )
        cmd.extend(["--dataset.repo_id", dataset_repo_id, "--dataset.episode", str(int(replay_episode))])
        if dataset_root:
            cmd.extend(["--dataset.root", dataset_root])
        cmd.extend(["--dataset.fps", str(int(dataset_fps))])
        cmd.extend(["--play_sounds", "true" if play_sounds else "false"])
        return cmd

    if action == "start":
        if dataset_repo_id:
            # Recording mode -> lerobot-record (pure data collection via teleop).
            if error := _numeric_option_error("record", numeric_options):
                raise ValueError(error)
            if error := _flag_error("record", flag_options):
                raise ValueError(error)
            from strands_robots.dataset_recorder import resolve_dataset_dir

            cmd = ["python", "-m", "lerobot.scripts.lerobot_record"]
            cmd.extend(
                _robot_args(robot_type, robot_port, robot_id, robot_left_arm_port, robot_right_arm_port, robot_cameras)
            )
            cmd.extend(_teleop_args(teleop_type, teleop_port, teleop_id, teleop_left_arm_port, teleop_right_arm_port))
            cmd.extend(["--dataset.repo_id", dataset_repo_id])
            cmd.extend(["--dataset.num_episodes", str(int(dataset_num_episodes))])
            if dataset_single_task:
                cmd.extend(["--dataset.single_task", dataset_single_task])
            cmd.extend(["--dataset.fps", str(int(dataset_fps))])
            cmd.extend(["--dataset.episode_time_s", str(int(dataset_episode_time_s))])
            cmd.extend(["--dataset.reset_time_s", str(int(dataset_reset_time_s))])
            # Always pin --dataset.root to a resolved on-disk path. On a fresh
            # (non-resume) recording, lerobot-record calls
            # ``cfg.dataset.stamp_repo_id()`` which rewrites repo_id to
            # ``{repo_id}_YYYYMMDD_HHMMSS`` -- moving the on-disk dataset and Hub
            # push target off the requested id. Stamping only rewrites repo_id;
            # an explicit root pins where the data lands, so downstream steps
            # (train, verify, path reporting) find it at the requested location.
            cmd.extend(["--dataset.root", str(resolve_dataset_dir(dataset_repo_id, dataset_root))])
            # Append to an existing dataset (skips stamping, preserves repo_id).
            if record_resume:
                cmd.extend(["--resume", "true"])
            cmd.extend(["--dataset.push_to_hub", "true" if dataset_push_to_hub else "false"])
            cmd.extend(["--dataset.video", "true" if dataset_video else "false"])
            if display_data:
                cmd.extend(["--display_data", "true"])
            cmd.extend(["--play_sounds", "true" if play_sounds else "false"])
            return cmd

        # Simple teleoperation mode -> lerobot-teleoperate.
        if error := _numeric_option_error("teleoperate", numeric_options):
            raise ValueError(error)
        if error := _flag_error("teleoperate", flag_options):
            raise ValueError(error)
        cmd = ["python", "-m", "lerobot.scripts.lerobot_teleoperate"]
        cmd.extend(
            _robot_args(robot_type, robot_port, robot_id, robot_left_arm_port, robot_right_arm_port, robot_cameras)
        )
        cmd.extend(_teleop_args(teleop_type, teleop_port, teleop_id, teleop_left_arm_port, teleop_right_arm_port))
        cmd.extend(["--fps", str(int(fps))])
        # ``is not None``, not truthiness: ``0`` is now refused above, and reading
        # it as "unset" made the one value meaning "stop at once" emit no budget
        # at all - an unbounded session where the caller asked for none.
        if teleop_time_s is not None:
            cmd.extend(["--teleop_time_s", str(teleop_time_s)])
        if display_data:
            cmd.extend(["--display_data", "true"])
        # No ``--play_sounds`` here, and deliberately not for symmetry's sake:
        # lerobot's ``TeleoperateConfig`` declares no such field and the entry
        # point makes no ``log_say`` call, so the flag would be an unrecognized
        # argument rather than an accepted no-op. Plain teleoperation has no
        # audio to suppress; see the note on :data:`_MODE_FLAG_OPTIONS`.
        return cmd

    if action == "dagger":
        # Human-in-the-loop correction (DAgger): a policy drives the follower
        # while the leader can pre-empt to record corrections, appended to the
        # dataset as new episodes. lerobot 0.5 implements this natively in
        # lerobot-rollout with --strategy.type=dagger (RolloutConfig +
        # DAggerStrategyConfig); the pre-0.5 "record with a policy" path was
        # removed from lerobot_record, which now refuses a policy and points
        # callers here.
        if not policy_path:
            raise ValueError("policy_path is required for dagger action (the policy to roll out)")
        if not dataset_repo_id:
            raise ValueError("dataset_repo_id is required for dagger action (corrections are recorded)")
        if dagger_input_device not in ("keyboard", "pedal"):
            raise ValueError(f"dagger_input_device must be 'keyboard' or 'pedal', got '{dagger_input_device}'")
        # Before the lerobot-version preflight below, so the same caller mistake
        # reports identically whether or not the rollout entry point is present.
        if error := _numeric_option_error("dagger", numeric_options):
            raise ValueError(error)
        if error := _flag_error("dagger", flag_options):
            raise ValueError(error)

        # lerobot.scripts.lerobot_rollout (the DAgger entry point) landed in
        # lerobot 0.6.0; on an older install the subprocess would fail with an
        # opaque "No module named" error. Preflight so the caller gets an
        # actionable upgrade hint instead.
        if importlib.util.find_spec("lerobot.scripts.lerobot_rollout") is None:
            raise RuntimeError(
                "dagger requires lerobot>=0.6.0: the installed lerobot has no "
                "'lerobot.scripts.lerobot_rollout' (the DAgger rollout entry "
                "point). Reinstall with: uv pip install 'strands-robots[lerobot]'."
            )

        cmd = ["python", "-m", "lerobot.scripts.lerobot_rollout"]
        cmd.extend(
            _robot_args(robot_type, robot_port, robot_id, robot_left_arm_port, robot_right_arm_port, robot_cameras)
        )
        cmd.extend(_teleop_args(teleop_type, teleop_port, teleop_id, teleop_left_arm_port, teleop_right_arm_port))
        cmd.extend(["--policy.path", policy_path])
        cmd.extend(["--strategy.type", "dagger"])
        cmd.extend(["--strategy.input_device", dagger_input_device])
        if dagger_record_autonomous:
            cmd.extend(["--strategy.record_autonomous", "true"])
        if dagger_num_episodes is not None:
            cmd.extend(["--strategy.num_episodes", str(int(dagger_num_episodes))])
        cmd.extend(["--dataset.repo_id", dataset_repo_id])
        if dataset_single_task:
            cmd.extend(["--dataset.single_task", dataset_single_task])
        cmd.extend(["--dataset.num_episodes", str(int(dataset_num_episodes))])
        cmd.extend(["--dataset.fps", str(int(dataset_fps))])
        if dataset_root:
            cmd.extend(["--dataset.root", dataset_root])
        # push_to_hub defaults to True in lerobot's DatasetRecordConfig; make it
        # explicit so an unattended correction run never uploads by surprise.
        cmd.extend(["--dataset.push_to_hub", "true" if dataset_push_to_hub else "false"])
        cmd.extend(["--dataset.video", "true" if dataset_video else "false"])
        cmd.extend(["--fps", str(int(fps))])
        if display_data:
            cmd.extend(["--display_data", "true"])
        cmd.extend(["--play_sounds", "true" if play_sounds else "false"])
        return cmd

    raise ValueError(f"Unknown action: {action}")


@tool
def lerobot_teleoperate(
    action: str = "start",
    session_name: str | None = None,
    background: bool = True,
    # Robot configuration
    robot_type: str = "so101_follower",
    robot_port: str | None = "/dev/ttyACM0",
    robot_id: str | None = None,
    robot_cameras: dict[str, Any] | None = None,
    robot_left_arm_port: str | None = None,
    robot_right_arm_port: str | None = None,
    # Teleoperator configuration
    teleop_type: str | None = "so101_leader",
    teleop_port: str | None = "/dev/ttyACM1",
    teleop_id: str | None = None,
    teleop_left_arm_port: str | None = None,
    teleop_right_arm_port: str | None = None,
    # Dataset configuration (for recording)
    dataset_repo_id: str | None = None,
    dataset_single_task: str | None = None,
    dataset_num_episodes: int = 50,
    dataset_fps: int = 30,
    dataset_episode_time_s: int = 60,
    dataset_reset_time_s: int = 60,
    dataset_root: str | None = None,
    dataset_video: bool = True,
    dataset_push_to_hub: bool = False,
    record_resume: bool = False,
    # Replay configuration
    replay_episode: int = 0,
    # Common options
    display_data: bool = False,
    fps: int = 60,
    teleop_time_s: float | None = None,
    play_sounds: bool = True,
    auto_accept_calibration: bool = True,
    # DAgger / teleop-takeover (action="dagger")
    policy_path: str | None = None,
    dagger_record_autonomous: bool = False,
    dagger_input_device: str = "keyboard",
    dagger_num_episodes: int | None = None,
) -> dict[str, Any]:
    """
    Advanced LeRobot teleoperation tool with recording capabilities for robot training data collection.

    This tool integrates teleoperation and recording functionality from lerobot, allowing users to:
    - Control robots through teleoperation devices
    - Record demonstrations for training machine learning models
    - Replay recorded episodes
    - Manage multiple teleoperation sessions

    Features:
    - Session Management: Start, stop, list, and monitor teleoperation sessions
    - Background Execution: Run teleoperation in background with logging
    - Recording Mode: Automatically record demonstrations when dataset configuration is provided
    - Multi-Robot Support: Support for single-arm and bimanual robots
    - Camera Integration: Multi-camera support with configurable settings
    - Replay Capability: Replay recorded episodes on physical robots
    - Safety Features: Graceful shutdown and process management

    Actions:
        start: Start a new teleoperation session
            - Simple teleoperation (just teleop_type specified)
            - Recording mode (dataset_repo_id specified)
            - Background or foreground execution

        stop: Stop a running session by name

        list: List all active teleoperation sessions

        status: Get detailed status of a specific session
            - Process information, uptime, logs

        replay: Replay a recorded episode on the robot
            - Requires dataset_repo_id and replay_episode

        dagger: Human-in-the-loop correction (DAgger / teleop takeover)
            - A policy drives the follower; the leader can pre-empt to record
              corrections, appended to the dataset as new episodes.
            - Requires policy_path (policy to roll out) and dataset_repo_id.
            - Drives lerobot-rollout with --strategy.type=dagger. Toggle the
              correction window with the keyboard/pedal (dagger_input_device);
              set dagger_record_autonomous=True to also record the autonomous
              phase, dagger_num_episodes to cap collected corrections.

    Robot Types:
        - so101_follower: Single-arm SO-101 robot
        - bi_so100_follower: Dual-arm SO-100 robot
        - koch_follower: Koch robot
        - hope_jr: HOPE Jr robot

    Teleoperator Types:
        - so101_leader: SO-101 leader device
        - bi_so100_leader: Dual SO-100 leader devices
        - koch_leader: Koch leader device
        - gamepad: Gamepad controller
        - homunculus: Homunculus teleoperator

    Camera Configuration Format:
        {
            "camera_name": {
                "type": "opencv",  # or "realsense"
                "index_or_path": 0,  # camera index or device path
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }

    Examples:
        # Simple teleoperation
        lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM1"
        )

        # Recording demonstrations
        lerobot_teleoperate(
            action="start",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM0",
            teleop_type="so101_leader",
            teleop_port="/dev/ttyACM1",
            dataset_repo_id="my_user/cube_picking",
            dataset_single_task="Pick up the red cube and place it in the box",
            dataset_num_episodes=25,
            robot_cameras={
                "front": {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30}
            }
        )

        # Bimanual robot teleoperation
        lerobot_teleoperate(
            action="start",
            robot_type="bi_so100_follower",
            robot_left_arm_port="/dev/ttyACM0",
            robot_right_arm_port="/dev/ttyACM1",
            teleop_type="bi_so100_leader",
            teleop_left_arm_port="/dev/ttyACM2",
            teleop_right_arm_port="/dev/ttyACM3"
        )

        # List sessions
        lerobot_teleoperate(action="list")

        # Stop session
        lerobot_teleoperate(action="stop", session_name="teleop_1234567890")

        # Replay episode
        lerobot_teleoperate(
            action="replay",
            robot_type="so101_follower",
            robot_port="/dev/ttyACM0",
            dataset_repo_id="my_user/cube_picking",
            replay_episode=5
        )

    Calibration Management:
        For calibration management (list, view, backup, etc.), use the separate
        lerobot_calibrate tool:

        # List available calibrations
        lerobot_calibrate(action="list")

        # View specific calibration
        lerobot_calibrate(action="view", device_type="robots",
                         device_model="so101_follower", device_id="orange_arm")

    Args:
        action: Action to perform (start, stop, list, status, replay)
        session_name: Session identifier (auto-generated for start, required for stop/status)
        background: Run session in background with logging (default: True).
            Must be a boolean: it selects an execution posture rather than
            scaling a quantity, so a truthy spelling of off such as
            ``"false"`` is refused rather than detaching the session it reads
            as declining to detach.

        robot_type: Robot type identifier
        robot_port: Serial port for single-arm robots
        robot_id: Robot instance identifier
        robot_cameras: Camera configuration dictionary
        robot_left_arm_port: Left arm port for bimanual robots
        robot_right_arm_port: Right arm port for bimanual robots

        teleop_type: Teleoperator type identifier
        teleop_port: Serial port for single-arm teleoperators
        teleop_id: Teleoperator instance identifier
        teleop_left_arm_port: Left arm port for bimanual teleoperators
        teleop_right_arm_port: Right arm port for bimanual teleoperators

        dataset_repo_id: HuggingFace dataset repository ID (enables recording mode)
        dataset_single_task: Task description for recordings
        dataset_num_episodes: Number of episodes to record
        dataset_fps: Recording frame rate
        dataset_episode_time_s: Episode duration in seconds
        dataset_reset_time_s: Reset time between episodes
        dataset_root: Local dataset storage directory. When omitted for a
            recording, an explicit root is still pinned (resolved from
            ``dataset_repo_id`` under ``$HF_LEROBOT_HOME``) and returned as
            ``dataset_root`` in the result. lerobot HEAD stamps a fresh record's
            ``repo_id`` with a ``_YYYYMMDD_HHMMSS`` timestamp (affecting the Hub
            push target and dataset metadata); pinning the root keeps the on-disk
            data at the requested location regardless, so downstream train/verify
            steps find it. Use ``record_resume=True`` to append instead.
        dataset_video: Enable video encoding
        dataset_push_to_hub: Upload dataset to HuggingFace Hub
        record_resume: Append to an existing dataset at the resolved root
            (lerobot-record ``--resume true``) instead of creating a fresh one.
            Resume preserves the existing (already-stamped) repo_id rather than
            re-stamping, so repeated sessions accumulate in one dataset.

        replay_episode: Episode number to replay

        display_data: Show live camera feeds and telemetry
        fps: Teleoperation control loop frequency
        teleop_time_s: Session duration limit
        play_sounds: Enable lerobot's spoken event announcements ("Recording
            episode 3", "Stop recording"). Effective for recording, replay and
            dagger; plain teleoperation emits no audio and ignores it. Must be a
            boolean - a string such as ``"false"`` is refused rather than read by
            truthiness, since every non-empty string is truthy.
        auto_accept_calibration: Answer the calibration prompt on the session's
            behalf, by writing two newlines into the process's stdin shortly
            after it starts. Withhold it (``False``) to answer the prompt
            yourself; nothing reports that stdin was written to, so an
            unintended acceptance is not visible afterwards. A write that
            *fails* is reported at WARNING, since by then the start result
            has already told the caller the session started. Must be a
            boolean, on the same reasoning as ``background``.

        dagger_record_autonomous: Record the autonomous rollout episodes into the
            corrections dataset as well, rather than only the teleoperated
            takeovers. Must be a boolean; a truthy spelling of off would
            otherwise land autonomous episodes in a corrections dataset.
        policy_path: Checkpoint the ``dagger`` action rolls out autonomously
            between human takeovers. Required for ``dagger``; ignored by every
            other action.
        dagger_input_device: How the operator seizes control during a ``dagger``
            rollout - ``"keyboard"`` (default) or ``"pedal"``. Any other value
            is refused.
        dagger_num_episodes: Cap on the corrections collected in one ``dagger``
            session. A positive whole number, or None for no cap.

    Returns:
        Dict with operation status and results:
        {
            "status": "success|error",
            "content": [{"text": "Description of operation"}],
            "session_name": "session_id",  # for start action
            "pid": 12345,  # process ID for background sessions
            "command": "full_command_executed",
            "log_file": "/tmp/session.log",  # for background sessions
            "sessions": {...},  # for list action
            "uptime": 123.45,  # session uptime in seconds
            "is_running": true  # for status action
        }
    """

    session_manager = SessionManager()

    try:
        if action in ("start", "dagger"):
            # Before anything is named, recorded or launched: a refused call must
            # leave no session behind and start no process (see
            # :data:`_EXECUTION_FLAGS`).
            if error := _execution_flag_error(
                {"background": background, "auto_accept_calibration": auto_accept_calibration}
            ):
                return {"status": "error", "content": [{"text": error}]}

            # Generate session name if not provided
            if not session_name:
                session_name = f"teleop_{int(time.time())}"

            # Check if session already exists
            if session_manager.get_session(session_name):
                return {"status": "error", "content": [{"text": f"Session '{session_name}' already exists"}]}

            # Build command
            try:
                cmd = build_lerobot_command(
                    action=action,
                    robot_type=robot_type,
                    robot_port=robot_port,
                    robot_id=robot_id,
                    robot_cameras=robot_cameras,
                    robot_left_arm_port=robot_left_arm_port,
                    robot_right_arm_port=robot_right_arm_port,
                    teleop_type=teleop_type,
                    teleop_port=teleop_port,
                    teleop_id=teleop_id,
                    teleop_left_arm_port=teleop_left_arm_port,
                    teleop_right_arm_port=teleop_right_arm_port,
                    dataset_repo_id=dataset_repo_id,
                    dataset_single_task=dataset_single_task,
                    dataset_num_episodes=dataset_num_episodes,
                    dataset_fps=dataset_fps,
                    dataset_episode_time_s=dataset_episode_time_s,
                    dataset_reset_time_s=dataset_reset_time_s,
                    dataset_root=dataset_root,
                    dataset_video=dataset_video,
                    dataset_push_to_hub=dataset_push_to_hub,
                    record_resume=record_resume,
                    replay_episode=replay_episode,
                    display_data=display_data,
                    fps=fps,
                    teleop_time_s=teleop_time_s,
                    play_sounds=play_sounds,
                    policy_path=policy_path,
                    dagger_record_autonomous=dagger_record_autonomous,
                    dagger_input_device=dagger_input_device,
                    dagger_num_episodes=dagger_num_episodes,
                )
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Command build failed: {str(e)}"}]}

            # Resolve the on-disk dataset location so downstream consumers (train,
            # verify, path reporting) use the true path. lerobot HEAD stamps a
            # fresh record's repo_id with a timestamp; the pinned --dataset.root
            # (see build_lerobot_command) keeps the data at this resolved path
            # regardless, so this is the authoritative on-disk location.
            resolved_dataset_root: str | None = None
            if dataset_repo_id:
                from strands_robots.dataset_recorder import resolve_dataset_dir

                resolved_dataset_root = str(resolve_dataset_dir(dataset_repo_id, dataset_root))

            if background:
                # Start in background
                log_file = SESSION_DIR / f"{session_name}.log"

                if auto_accept_calibration:
                    # Start process with stdin for automatic calibration acceptance
                    with open(log_file, "w") as f:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                            text=True,
                            start_new_session=True,
                        )

                    # Send automatic "ENTER" to accept existing calibrations
                    # We'll do this in a separate thread to not block
                    import threading

                    def auto_respond():
                        try:
                            time.sleep(2)  # Allow process to initialize before writing to stdin
                            proc.stdin.write("\n")  # Send ENTER
                            proc.stdin.flush()
                            time.sleep(1)
                            proc.stdin.write("\n")  # Send another ENTER (for robot calibration)
                            proc.stdin.flush()
                            proc.stdin.close()  # Close stdin after sending responses
                        except Exception as exc:
                            # Report it. The start result above has already told the
                            # caller the session started, so this record is the only
                            # signal that the prompt went unanswered. Every other
                            # handler in this tool reports its failure - one even
                            # surfaces a log-read failure into the caller's content -
                            # and the write this guards is the whole job of
                            # ``auto_accept_calibration``. WARNING rather than DEBUG
                            # because the visible report is a success.
                            logger.warning(
                                "[teleop] session %r: auto-accept did not complete (%s); "
                                "the calibration prompt may be unanswered - check "
                                "action='status' and the session log",
                                session_name,
                                exc,
                            )

                    threading.Thread(target=auto_respond, daemon=True).start()
                else:
                    # Start normally without stdin handling
                    with open(log_file, "w") as f:
                        proc = subprocess.Popen(
                            cmd, stdout=f, stderr=subprocess.STDOUT, text=True, start_new_session=True
                        )

                # Store session info
                session_info = {
                    "action": "teleoperate" if not dataset_repo_id else "record",
                    "pid": proc.pid,
                    "command": " ".join(cmd),
                    "log_file": str(log_file),
                    "start_time": time.time(),
                    "background": True,
                    "robot_type": robot_type,
                    "teleop_type": teleop_type,
                    "dataset_repo_id": dataset_repo_id,
                    "dataset_root": resolved_dataset_root,
                    "resume": record_resume,
                }
                session_manager.add_session(session_name, session_info)

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"**Teleoperation Session Started**\n"
                            f"Session: `{session_name}`\n"
                            f"Process ID: {proc.pid}\n"
                            f"Command: `{' '.join(cmd)}`\n"
                            f"Log file: `{log_file}`\n"
                            f"Running in background"
                        },
                        {
                            "json": {
                                "session_name": session_name,
                                "pid": proc.pid,
                                "command": " ".join(cmd),
                                "log_file": str(log_file),
                                "background": True,
                                "dataset_repo_id": dataset_repo_id,
                                "dataset_root": resolved_dataset_root,
                                "resume": record_resume,
                            }
                        },
                    ],
                }
            else:
                # Start in foreground
                result = subprocess.run(cmd, capture_output=True, text=True)

                return {
                    "status": "success" if result.returncode == 0 else "error",
                    "content": [
                        {
                            "text": f"**Foreground Execution Complete**\n"
                            f"Return code: {result.returncode}\n"
                            f"Command: `{' '.join(cmd)}`\n\n"
                            f"**Output:**\n```\n{result.stdout}\n```\n\n"
                            f"**Errors:**\n```\n{result.stderr}\n```"
                        },
                        {
                            "json": {
                                "command": " ".join(cmd),
                                "return_code": result.returncode,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "dataset_repo_id": dataset_repo_id,
                                "dataset_root": resolved_dataset_root,
                                "resume": record_resume,
                            }
                        },
                    ],
                }

        elif action == "stop":
            if not session_name:
                return {"status": "error", "content": [{"text": "Session name required for stop action"}]}

            session_info = session_manager.get_session(session_name)  # type: ignore[assignment]  # narrow Optional
            if not session_info:
                return {"status": "error", "content": [{"text": f"Session '{session_name}' not found"}]}

            pid = session_info.get("pid")
            if not pid:
                return {"status": "error", "content": [{"text": f"No PID found for session '{session_name}'"}]}

            pid_int = int(pid)
            try:
                # Try graceful termination first
                os.kill(pid_int, signal.SIGTERM)
                time.sleep(2)  # Grace period for process to flush buffers and exit cleanly

                # Force kill if still running after grace period
                if psutil.pid_exists(pid_int):
                    os.kill(pid_int, signal.SIGKILL)

                session_manager.remove_session(session_name)

                return {
                    "status": "success",
                    "content": [
                        {"text": f"**Session Stopped**\nSession: `{session_name}`\nPID: {pid}"},
                        {"json": {"session_name": session_name, "session_info": session_info}},
                    ],
                }

            except ProcessLookupError:
                # Process already dead
                session_manager.remove_session(session_name)
                return {
                    "status": "success",
                    "content": [
                        {"text": f"Session '{session_name}' was already stopped"},
                        {"json": {"session_name": session_name}},
                    ],
                }
            except Exception as e:
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to stop session '{session_name}': {str(e)}"}],
                }

        elif action == "list":
            sessions = session_manager.list_sessions()

            content_lines = [f"**Active Teleoperation Sessions** ({len(sessions)})", ""]

            if sessions:
                for name, info in sessions.items():
                    uptime = time.time() - info.get("start_time", 0)
                    uptime_min = uptime / 60
                    pid = info.get("pid")
                    is_running = pid and psutil.pid_exists(pid)

                    content_lines.extend(
                        [
                            f"**{name}**",
                            f"   - Action: {info.get('action', 'Unknown')}",
                            f"   - PID: {pid}",
                            f"   - Uptime: {uptime_min:.1f} min",
                            f"   - Status: {'Running' if is_running else 'Stopped'}",
                            f"   - Robot: {info.get('robot_type', 'Unknown')}",
                            f"   - Teleop: {info.get('teleop_type', 'Unknown')}",
                            "",
                        ]
                    )
            else:
                content_lines.append("No active sessions")

            return {
                "status": "success",
                "content": [
                    {"text": "\n".join(content_lines)},
                    {"json": {"sessions": sessions, "count": len(sessions)}},
                ],
            }

        elif action == "status":
            if not session_name:
                return {"status": "error", "content": [{"text": "Session name required for status action"}]}

            session_info = session_manager.get_session(session_name)  # type: ignore[assignment]  # narrow Optional
            if not session_info:
                return {"status": "error", "content": [{"text": f"Session '{session_name}' not found"}]}

            pid = session_info.get("pid")
            start_time: float = float(session_info.get("start_time") or 0)
            uptime = time.time() - start_time
            uptime_min = uptime / 60
            is_running = pid and psutil.pid_exists(int(pid))

            content_lines = [
                f"**Session Status: `{session_name}`**",
                f"PID: {pid}",
                f"Action: {session_info.get('action', 'Unknown')}",
                f"Uptime: {uptime_min:.1f} min",
                f"Status: {'Running' if is_running else 'Stopped'}",
                f"Robot: {session_info.get('robot_type', 'Unknown')}",
                f"Teleop: {session_info.get('teleop_type', 'Unknown')}",
            ]

            # Add log tail if available
            log_file_path = session_info.get("log_file")
            if log_file_path and Path(str(log_file_path)).exists():
                content_lines.append(f"Log file: `{log_file_path}`")

                try:
                    with open(str(log_file_path)) as f:
                        lines = f.readlines()
                        if lines:
                            tail_lines = lines[-10:]  # Last 10 lines
                            content_lines.extend(
                                ["", "**Recent Log Output:**", "```", "".join(tail_lines).strip(), "```"]
                            )
                except Exception as e:
                    content_lines.append(f"Error reading log: {str(e)}")

            return {
                "status": "success",
                "content": [
                    {"text": "\n".join(content_lines)},
                    {
                        "json": {
                            **session_info,
                            "session_name": session_name,
                            "pid": pid,
                            "uptime": uptime,
                            "is_running": is_running,
                        }
                    },
                ],
            }

        elif action == "replay":
            if not dataset_repo_id:
                return {"status": "error", "content": [{"text": "dataset_repo_id required for replay action"}]}

            try:
                cmd = build_lerobot_command(
                    action="replay",
                    robot_type=robot_type,
                    robot_port=robot_port,
                    robot_id=robot_id,
                    robot_left_arm_port=robot_left_arm_port,
                    robot_right_arm_port=robot_right_arm_port,
                    dataset_repo_id=dataset_repo_id,
                    replay_episode=replay_episode,
                    display_data=display_data,
                    play_sounds=play_sounds,
                )
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Replay command build failed: {str(e)}"}]}

            # Execute replay
            result = subprocess.run(cmd, capture_output=True, text=True)

            content_lines = [
                "**Episode Replay Complete**",
                f"Return code: {result.returncode}",
                f"Command: `{' '.join(cmd)}`",
            ]

            if result.stdout:
                content_lines.extend(["", "**Output:**", "```", result.stdout, "```"])

            if result.stderr:
                content_lines.extend(["", "**Errors:**", "```", result.stderr, "```"])

            return {
                "status": "success" if result.returncode == 0 else "error",
                "content": [
                    {"text": "\n".join(content_lines)},
                    {
                        "json": {
                            "command": " ".join(cmd),
                            "return_code": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        }
                    },
                ],
            }

        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

    except Exception as e:
        logger.error(f"LeRobot teleoperate error: {e}")
        return {"status": "error", "content": [{"text": f"Tool execution failed: {str(e)}"}]}
