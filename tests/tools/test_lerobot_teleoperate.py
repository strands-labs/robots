"""Behavior tests for the ``lerobot_teleoperate`` agent tool.

The teleoperate tool wraps LeRobot's record/replay/teleoperate scripts behind a
single agent-facing dispatcher with on-disk session tracking. These tests
exercise every action branch hardware-free by substituting fakes for
``subprocess`` and ``psutil``, and pin two invariants the tool must uphold:

1. Every user-facing ``text`` field is plain ASCII (the project's no-emoji rule).
2. The command builder maps each action + option to the correct lerobot CLI
   argv, and the session lifecycle (start -> list -> status -> stop) round-trips
   through the persisted session store.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod

# Bind the public names off the single module handle rather than a second
# ``from ... import`` of the same module (CodeQL: import + import-from of one
# module). ``tele_mod`` is still needed directly so monkeypatch can rebind
# module globals (``subprocess``/``psutil``/``os``/``time``/``SESSION_DIR``).
SessionManager = tele_mod.SessionManager
build_lerobot_command = tele_mod.build_lerobot_command
lerobot_teleoperate = tele_mod.lerobot_teleoperate


def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _assert_ascii(text: str) -> None:
    """Fail if any character is outside the ASCII range."""
    offenders = {hex(ord(c)) for c in text if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the module-level session dir + manager to a temp location.

    The module computes ``SESSION_DIR`` at import time from ``cwd``; rebind it so
    tests never touch the real working tree and start from an empty store.
    """
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    return session_dir


class _FakeProc:
    """Minimal stand-in for ``subprocess.Popen`` / ``run`` results."""

    def __init__(self, pid: int = 4242, returncode: int = 0, stdout: str = "ok", stderr: str = "") -> None:
        self.pid = pid
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stdin: Any = None


# ---------------------------------------------------------------------------
# build_lerobot_command - argv mapping for each action
# ---------------------------------------------------------------------------
def test_build_replay_command_includes_episode_and_paths() -> None:
    cmd = build_lerobot_command(
        action="replay",
        robot_type="so101_follower",
        robot_port="/dev/ttyACM0",
        dataset_repo_id="user/cubes",
        replay_episode=5,
        display_data=True,
    )
    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_replay"]
    assert "--episode" in cmd and cmd[cmd.index("--episode") + 1] == "5"
    assert "--policy-path" in cmd and "user/cubes" in cmd
    assert "--robot-port" in cmd
    assert "--display-data" in cmd


def test_build_replay_command_requires_dataset_repo_id() -> None:
    with pytest.raises(ValueError, match="dataset_repo_id is required"):
        build_lerobot_command(action="replay", robot_type="so101_follower")


def test_build_replay_command_routes_bimanual_arm_ports() -> None:
    """Replay on a bimanual (ALOHA-class) robot must forward both arm ports.

    Single-arm SO-100/SO-101 robots use ``--robot-port``; bimanual robots have
    no single port and instead pass ``--robot-left-arm-port`` /
    ``--robot-right-arm-port``. The replay builder must emit the per-arm flags
    (and may omit ``--robot-port`` entirely) so the lerobot CLI binds each arm.
    """
    cmd = build_lerobot_command(
        action="replay",
        robot_type="aloha",
        dataset_repo_id="user/bimanual_pick",
        replay_episode=2,
        robot_left_arm_port="/dev/ttyACM0",
        robot_right_arm_port="/dev/ttyACM1",
    )
    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_replay"]
    assert cmd[cmd.index("--robot-left-arm-port") + 1] == "/dev/ttyACM0"
    assert cmd[cmd.index("--robot-right-arm-port") + 1] == "/dev/ttyACM1"
    # No single robot_port was given, so the single-arm flag must be absent.
    assert "--robot-port" not in cmd


def test_build_start_teleop_command_routes_bimanual_ports_ids_and_root() -> None:
    """A bimanual teleop start must forward per-arm ports, ids, root + display.

    Exercises the option branches a leader->follower ALOHA recording session
    needs: robot/teleop ``--*.id`` namespacing, the left/right arm ports for
    both the follower (robot) and leader (teleop), a dataset ``--root`` for the
    on-disk recording location, and the ``--display_data`` viewer flag. Each
    must map to its lerobot CLI argument with the supplied value.
    """
    cmd = build_lerobot_command(
        action="start",
        robot_type="aloha",
        robot_id="follower_arm",
        robot_left_arm_port="/dev/ttyACM0",
        robot_right_arm_port="/dev/ttyACM1",
        teleop_type="aloha_leader",
        teleop_id="leader_arm",
        teleop_left_arm_port="/dev/ttyACM2",
        teleop_right_arm_port="/dev/ttyACM3",
        dataset_repo_id="user/bimanual_pick",
        dataset_root="/data/lerobot/bimanual_pick",
        display_data=True,
    )
    # Recording mode (dataset given) -> lerobot_record entrypoint.
    assert "lerobot.scripts.lerobot_record" in cmd
    assert cmd[cmd.index("--root") + 1] == "/data/lerobot/bimanual_pick"
    # Robot (follower) per-arm config.
    assert cmd[cmd.index("--robot.id") + 1] == "follower_arm"
    assert cmd[cmd.index("--robot.left_arm_port") + 1] == "/dev/ttyACM0"
    assert cmd[cmd.index("--robot.right_arm_port") + 1] == "/dev/ttyACM1"
    # Teleop (leader) per-arm config.
    assert cmd[cmd.index("--teleop.id") + 1] == "leader_arm"
    assert cmd[cmd.index("--teleop.left_arm_port") + 1] == "/dev/ttyACM2"
    assert cmd[cmd.index("--teleop.right_arm_port") + 1] == "/dev/ttyACM3"
    # Viewer flag is emitted as the explicit "true" value form.
    assert cmd[cmd.index("--display_data") + 1] == "true"


def test_build_start_record_command_when_dataset_given() -> None:
    cmd = build_lerobot_command(
        action="start",
        robot_type="so101_follower",
        robot_port="/dev/ttyACM0",
        dataset_repo_id="user/cubes",
        dataset_single_task="pick the cube",
        dataset_num_episodes=10,
        dataset_push_to_hub=True,
        dataset_video=False,
    )
    assert "lerobot.scripts.lerobot_record" in cmd
    assert "--repo-id" in cmd and "user/cubes" in cmd
    assert cmd[cmd.index("--num-episodes") + 1] == "10"
    assert "--single-task" in cmd
    assert "--push-to-hub" in cmd
    assert "--no-video" in cmd


def test_build_start_teleop_command_without_dataset() -> None:
    cmd = build_lerobot_command(
        action="start",
        robot_type="so101_follower",
        robot_port="/dev/ttyACM0",
        teleop_type="so101_leader",
        teleop_port="/dev/ttyACM1",
        teleop_time_s=30.0,
    )
    assert "lerobot.scripts.lerobot_teleoperate" in cmd
    assert "--robot.type" in cmd
    assert "--teleop.type" in cmd and "so101_leader" in cmd
    assert "--teleop.port" in cmd
    assert "--teleop_time_s" in cmd


def test_build_start_command_emits_camera_config() -> None:
    cmd = build_lerobot_command(
        action="start",
        robot_type="so101_follower",
        robot_cameras={"front": {"type": "opencv", "index_or_path": 2, "width": 1280, "height": 720, "fps": 60}},
    )
    cam_args = [a for a in cmd if a.startswith("front=")]
    assert cam_args == ["front=opencv:2:60:1280x720"]


def test_build_command_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="Unknown action"):
        build_lerobot_command(action="bogus", robot_type="so101_follower")


# ---------------------------------------------------------------------------
# SessionManager - persisted store, dead-process pruning
# ---------------------------------------------------------------------------
def test_session_manager_add_get_remove_round_trip() -> None:
    mgr = SessionManager()
    info = {"pid": os.getpid(), "robot_type": "so101_follower"}
    mgr.add_session("s1", info)
    assert mgr.get_session("s1") == info
    mgr.remove_session("s1")
    assert mgr.get_session("s1") is None


def test_session_manager_prunes_dead_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SessionManager()
    # Persist a session with a pid that no longer exists.
    mgr.sessions_file.write_text(json.dumps({"ghost": {"pid": 999999}}))
    monkeypatch.setattr(tele_mod.psutil, "pid_exists", lambda pid: False)
    assert mgr.list_sessions() == {}


def test_session_manager_handles_corrupt_store(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SessionManager()
    mgr.sessions_file.write_text("{ not valid json")
    # Corrupt store degrades to empty rather than raising.
    assert mgr.list_sessions() == {}


# ---------------------------------------------------------------------------
# lerobot_teleoperate dispatcher - ASCII output + lifecycle
# ---------------------------------------------------------------------------
def test_start_background_session_is_ascii_and_persisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tele_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(pid=os.getpid()))

    result = lerobot_teleoperate(
        action="start",
        session_name="teleop_test",
        robot_type="so101_follower",
        teleop_type="so101_leader",
        auto_accept_calibration=False,
    )
    assert result["status"] == "success"
    assert result["pid"] == os.getpid()
    _assert_ascii(_texts(result))
    # Session was persisted and is discoverable.
    listed = lerobot_teleoperate(action="list")
    assert "teleop_test" in listed["sessions"]


def test_start_rejects_duplicate_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tele_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(pid=1))
    lerobot_teleoperate(action="start", session_name="dup", auto_accept_calibration=False)
    again = lerobot_teleoperate(action="start", session_name="dup", auto_accept_calibration=False)
    assert again["status"] == "error"
    assert "already exists" in _texts(again)


def test_list_empty_is_ascii() -> None:
    result = lerobot_teleoperate(action="list")
    assert result["status"] == "success"
    assert result["count"] == 0
    _assert_ascii(_texts(result))


def test_status_unknown_session_errors() -> None:
    result = lerobot_teleoperate(action="status", session_name="missing")
    assert result["status"] == "error"
    assert "not found" in _texts(result)


def test_status_running_session_is_ascii(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SessionManager()
    mgr.add_session("live", {"pid": os.getpid(), "action": "record", "start_time": 0.0, "robot_type": "so101_follower"})
    result = lerobot_teleoperate(action="status", session_name="live")
    assert result["status"] == "success"
    assert result["is_running"]
    _assert_ascii(_texts(result))


def test_stop_session_terminates_and_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SessionManager()
    live_pid = os.getpid()
    mgr.add_session("kill", {"pid": live_pid, "start_time": 0.0})
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(tele_mod.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(tele_mod.time, "sleep", lambda s: None)
    result = lerobot_teleoperate(action="stop", session_name="kill")
    assert result["status"] == "success"
    assert killed and killed[0][0] == live_pid
    assert SessionManager().get_session("kill") is None
    _assert_ascii(_texts(result))


def test_stop_without_name_errors() -> None:
    result = lerobot_teleoperate(action="stop")
    assert result["status"] == "error"


def test_replay_requires_dataset_repo_id() -> None:
    result = lerobot_teleoperate(action="replay")
    assert result["status"] == "error"
    assert "dataset_repo_id required" in _texts(result)


def test_replay_runs_command_and_is_ascii(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tele_mod.subprocess, "run", lambda *a, **k: _FakeProc(returncode=0, stdout="done", stderr=""))
    result = lerobot_teleoperate(action="replay", dataset_repo_id="user/cubes", replay_episode=2)
    assert result["status"] == "success"
    assert result["return_code"] == 0
    _assert_ascii(_texts(result))


def test_unknown_action_errors() -> None:
    result = lerobot_teleoperate(action="frobnicate")
    assert result["status"] == "error"
    assert "Unknown action" in _texts(result)


def test_start_foreground_runs_and_is_ascii(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tele_mod.subprocess, "run", lambda *a, **k: _FakeProc(returncode=0, stdout="hi", stderr=""))
    result = lerobot_teleoperate(
        action="start",
        session_name="fg",
        robot_type="so101_follower",
        teleop_type="so101_leader",
        background=False,
    )
    assert result["status"] == "success"
    assert result["return_code"] == 0
    _assert_ascii(_texts(result))


def test_list_with_active_session_is_ascii(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SessionManager()
    mgr.add_session(
        "live", {"pid": os.getpid(), "action": "teleoperate", "start_time": 0.0, "robot_type": "so101_follower"}
    )
    result = lerobot_teleoperate(action="list")
    assert result["status"] == "success"
    assert result["count"] == 1
    body = _texts(result)
    assert "live" in body
    assert "Running" in body
    _assert_ascii(body)


# ---------------------------------------------------------------------------
# auto_accept_calibration - background start auto-answers calibration prompts
# ---------------------------------------------------------------------------
class _CapturingStdin:
    """Stand-in for ``Popen.stdin`` that records what the auto-responder writes."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self.closed = False

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _SyncThread:
    """Run a thread target inline on ``start`` so the daemon body is covered."""

    def __init__(self, target=None, daemon=None, **_: Any) -> None:
        self._target = target

    def start(self) -> None:
        if self._target is not None:
            self._target()


def test_start_auto_accept_calibration_sends_enter_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """With auto_accept_calibration, the tool spawns a thread that writes two
    ENTER keystrokes to the child stdin and closes it, accepting calibration
    prompts without blocking."""
    stdin = _CapturingStdin()
    proc = _FakeProc(pid=os.getpid())
    proc.stdin = stdin
    monkeypatch.setattr(tele_mod.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(tele_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr("threading.Thread", _SyncThread)

    result = lerobot_teleoperate(
        action="start",
        session_name="autocal",
        robot_type="so101_follower",
        teleop_type="so101_leader",
        auto_accept_calibration=True,
    )

    assert result["status"] == "success"
    assert result["background"] is True
    # Two ENTER presses were sent and stdin was closed afterwards.
    assert stdin.writes == ["\n", "\n"]
    assert stdin.closed
    _assert_ascii(_texts(result))


def test_start_record_action_label_when_dataset_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """A background start with a dataset repo records the session as a 'record'
    session (not plain teleoperate)."""
    monkeypatch.setattr(tele_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(pid=os.getpid()))

    result = lerobot_teleoperate(
        action="start",
        session_name="rec",
        robot_type="so101_follower",
        teleop_type="so101_leader",
        dataset_repo_id="user/cubes",
        dataset_single_task="pick the cube",
        auto_accept_calibration=False,
    )

    assert result["status"] == "success"
    session = SessionManager().get_session("rec")
    assert session is not None
    assert session["action"] == "record"


# ---------------------------------------------------------------------------
# stop - dead-process and generic-failure branches
# ---------------------------------------------------------------------------
def test_stop_already_dead_process_is_cleaned_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the OS reports the process is already gone, the session is still
    removed and the call succeeds."""
    # Use a live PID so the session survives the load-time liveness prune,
    # then have the kill report the process is already gone.
    SessionManager().add_session("ghost", {"pid": os.getpid(), "start_time": 0.0})

    def _raise_lookup(pid: int, sig: int) -> None:
        # sig 0 is psutil's liveness probe during session load; only the real
        # termination signals should report the process is already gone.
        if sig != 0:
            raise ProcessLookupError

    monkeypatch.setattr(tele_mod.os, "kill", _raise_lookup)
    monkeypatch.setattr(tele_mod.time, "sleep", lambda s: None)

    result = lerobot_teleoperate(action="stop", session_name="ghost")
    assert result["status"] == "success"
    assert "already stopped" in _texts(result)
    assert SessionManager().get_session("ghost") is None


def test_stop_kill_failure_reports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected kill failure surfaces as an error result, not an exception."""
    SessionManager().add_session("stuck", {"pid": os.getpid(), "start_time": 0.0})

    def _raise_perm(pid: int, sig: int) -> None:
        if sig != 0:
            raise PermissionError("operation not permitted")

    monkeypatch.setattr(tele_mod.os, "kill", _raise_perm)
    monkeypatch.setattr(tele_mod.time, "sleep", lambda s: None)

    result = lerobot_teleoperate(action="stop", session_name="stuck")
    assert result["status"] == "error"
    assert "Failed to stop" in _texts(result)
    _assert_ascii(_texts(result))


# ---------------------------------------------------------------------------
# status - log tail rendering
# ---------------------------------------------------------------------------
def test_status_includes_log_tail(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """When a session has a readable log file, status echoes its last lines
    inside a fenced block and stays ASCII."""
    log_file = tmp_path / "sess.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")
    SessionManager().add_session(
        "withlog",
        {"pid": os.getpid(), "start_time": 0.0, "log_file": str(log_file), "robot_type": "so101_follower"},
    )

    result = lerobot_teleoperate(action="status", session_name="withlog")
    assert result["status"] == "success"
    body = _texts(result)
    assert "Recent Log Output" in body
    assert "line 19" in body
    # Only the tail (last 10 lines) is shown, not the head.
    assert "line 0" not in body
    _assert_ascii(body)


# ---------------------------------------------------------------------------
# replay / foreground - non-zero return codes map to error status
# ---------------------------------------------------------------------------
def test_replay_nonzero_return_code_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tele_mod.subprocess, "run", lambda *a, **k: _FakeProc(returncode=1, stdout="", stderr="boom"))
    result = lerobot_teleoperate(action="replay", dataset_repo_id="user/cubes", replay_episode=0)
    assert result["status"] == "error"
    assert result["return_code"] == 1
    _assert_ascii(_texts(result))


def test_start_foreground_nonzero_return_code_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tele_mod.subprocess, "run", lambda *a, **k: _FakeProc(returncode=2, stdout="", stderr="nope"))
    result = lerobot_teleoperate(
        action="start",
        session_name="fgfail",
        robot_type="so101_follower",
        teleop_type="so101_leader",
        background=False,
    )
    assert result["status"] == "error"
    assert result["return_code"] == 2
    _assert_ascii(_texts(result))
