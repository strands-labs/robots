"""``status`` reports on a detached run whose log is not valid UTF-8.

Both session tools launch their subprocess detached with the log file opened as
the child's ``stdout``/``stderr`` sink::

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, ...)

What lands in that file is whatever the child wrote to the fd it inherited, so
its bytes are the child's and need not be UTF-8: a native library printing
latin-1, a progress renderer, or a multi-byte character torn in half when the
run was killed all produce one. The parent then reads the file back to show the
tail, and reading it strictly turns the child's byte into this process's
exception.

That exception is not the failure either tail handler was written for. In
``lerobot_train`` the handler names ``OSError``, so a ``UnicodeDecodeError`` -
which is a ``ValueError`` - passes it and reaches the action's outer guard: the
whole ``status`` report is replaced by ``Tool execution failed: 'utf-8' codec
can't decode byte ...``, discarding the pid, the uptime and the running verdict
that were already computed before the log was opened. ``lerobot_teleoperate``
catches broadly and so keeps its report, but loses the tail and prints the codec
message in its place.

Both are read with substitution instead. The undecodable byte becomes U+FFFD,
which is asserted here rather than merely tolerated: a dropped byte would change
the text an operator is reading without saying so, while a replacement
character shows exactly where the log stopped being text.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod


class Tool(NamedTuple):
    """One session tool, reduced to what this contract drives."""

    name: str
    module: Any
    status: Callable[[str], dict[str, Any]]
    record: dict[str, Any]

    def __repr__(self) -> str:  # pragma: no cover - test ids only
        return self.name


TOOLS = [
    Tool(
        "lerobot_train",
        train_mod,
        lambda session: train_mod.lerobot_train(dataset_root="/unused", action="status", session_name=session),
        {"action": "train", "policy_type": "act", "output_dir": "/out"},
    ),
    Tool(
        "lerobot_teleoperate",
        tele_mod,
        lambda session: tele_mod.lerobot_teleoperate(action="status", session_name=session),
        {"action": "teleoperate", "robot_type": "so101_follower"},
    ),
]

#: A line the child wrote as latin-1. Valid text to the child, undecodable here.
UNDECODABLE_LOG = b"INFO step:1.2K loss:0.123\nWARN saving to caf\xe9/checkpoint\n"


def _seed(tool: Tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log: bytes) -> tuple[str, int]:
    """One live session whose log file holds ``log``. Returns (name, pid)."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tool.module, "SESSION_DIR", session_dir)

    pid = os.getpid()
    assert tool.module.psutil.pid_exists(pid), "premise: the test process must exist"
    log_file = session_dir / "run.log"
    log_file.write_bytes(log)
    record = {"pid": pid, "start_time": 0.0, "log_file": str(log_file), **tool.record}
    (session_dir / "active_sessions.json").write_text(json.dumps({"run": record}, indent=2))
    return "run", pid


def _text(result: dict[str, Any]) -> str:
    return next(block["text"] for block in result["content"] if "text" in block)


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_status_still_reports_the_session_it_was_asked_about(
    tool: Tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pid and the running verdict do not depend on the log being decodable."""
    name, pid = _seed(tool, tmp_path, monkeypatch, UNDECODABLE_LOG)

    result = tool.status(name)

    assert result["status"] == "success", f"{tool.name} status failed on a log it did not encode: {result}"
    text = _text(result)
    assert f"PID: {pid}" in text, f"the report must still name the pid it was about: {text}"
    assert "Status: Running" in text, f"the running verdict is taken before the log is read: {text}"


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_the_tail_survives_with_the_damage_shown_where_it_is(
    tool: Tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Substitution, not omission: the decodable log stays and U+FFFD marks the rest."""
    name, _ = _seed(tool, tmp_path, monkeypatch, UNDECODABLE_LOG)

    text = _text(tool.status(name))

    assert "step:1.2K loss:0.123" in text, f"the decodable lines are what the tail is for: {text}"
    assert "\ufffd" in text, (
        f"a byte dropped silently would misreport the child's output; it must read as U+FFFD: {text}"
    )


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_a_decodable_log_is_reported_unchanged(tool: Tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the ordinary case must read exactly as before, U+FFFD-free."""
    name, _ = _seed(tool, tmp_path, monkeypatch, b"INFO step:1.2K loss:0.123\n")

    text = _text(tool.status(name))

    assert "step:1.2K loss:0.123" in text, f"a clean log must still be tailed: {text}"
    assert "\ufffd" not in text, f"nothing was damaged, so nothing may be marked as damaged: {text}"
