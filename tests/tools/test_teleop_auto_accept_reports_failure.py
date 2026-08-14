"""The teleop auto-accept reports a failed write, and the ``stop`` refusals.

``lerobot_teleoperate(action="start", auto_accept_calibration=True)`` answers the
child's calibration prompt on the operator's behalf by writing two newlines into
its stdin from a background thread. That write is the whole job of the flag, and
the start result has already reported ``status="success"`` by the time it runs -
so if it fails, the record this module pins is the only signal the operator gets
that the prompt went unanswered.

Every other ``except Exception`` in the tool reports its failure (four return an
error envelope, one surfaces a log-read failure into the caller's own content,
one logs); this handler was the only silent one. It went untested because the
auto-accept runs on a daemon thread that sleeps two seconds first, so a test
process exits before it executes - hence the synchronous thread stand-in below,
which makes the outcome deterministic rather than a race.

This module also closes the ``stop`` half of a two-cell refusal matrix: ``stop``
and ``status`` refuse an unknown session with byte-identical text, and only
``status``'s was driven. The ``stop`` case is the common one in practice - a
session whose process has exited is pruned from the store on the next read - and
the same pruning is what makes the tool's "No PID found" branch unreachable, a
property pinned here rather than assumed.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod

SessionManager = tele_mod.SessionManager
lerobot_teleoperate = tele_mod.lerobot_teleoperate

_LOGGER_NAME = "strands_robots.tools.lerobot_teleoperate"

#: Session name used by the failure fixtures. Deliberately not a substring of
#: the record's own prose: ``"cal"`` was, because the record says
#: ``"calibration"``, which made the "names the session" assertion vacuous.
_SESSION = "wrist-rig-7"

# The store prunes any record whose pid is not a live process on every read, so
# a fake pid would make the session vanish before a test could read it back.
# This process's own pid is live for the duration of the test, which is what a
# real session's pid is too.
_LIVE_PID = os.getpid()


def _texts(result: dict[str, Any]) -> str:
    """Concatenate all content ``text`` fields from a tool result."""
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the module-level session dir so tests never touch the real store."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    return session_dir


class _SyncThread:
    """Run the target inline so the auto-accept is observable in-test.

    The tool starts the auto-accept on a daemon thread. Left asynchronous, the
    test process exits during its first ``time.sleep``, so neither outcome is
    observable - which is why the handler under test had no coverage. The tool
    does ``import threading`` inside the function body, so the class is resolved
    off the module at call time and rebinding the attribute is enough.
    """

    def __init__(self, target: Any = None, daemon: bool | None = None, **_kw: Any) -> None:
        self._target = target

    def start(self) -> None:
        if self._target is not None:
            self._target()


class _Stdin:
    """Recording stdin whose ``write`` optionally fails, as a closed pipe would."""

    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.writes: list[str] = []
        self.closed = False

    def write(self, data: str) -> None:
        if self.fail:
            raise OSError("Broken pipe")
        self.writes.append(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _CalProc:
    """``Popen`` stand-in carrying a real stdin, unlike the shared ``_FakeProc``."""

    def __init__(self, pid: int | None = None, fail: bool = False) -> None:
        self.pid = _LIVE_PID if pid is None else pid
        self.returncode: int | None = None
        self.stdin = _Stdin(fail)

    def poll(self) -> int | None:
        return None


def _start_with_auto_accept(monkeypatch: pytest.MonkeyPatch, *, fail: bool) -> tuple[dict[str, Any], _CalProc]:
    """Drive a background start whose auto-accept write succeeds or fails."""
    proc = _CalProc(fail=fail)
    monkeypatch.setattr(threading, "Thread", _SyncThread)
    monkeypatch.setattr(tele_mod.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(tele_mod.time, "sleep", lambda s: None)
    result = lerobot_teleoperate(
        action="start",
        session_name=_SESSION,
        robot_type="so101_follower",
        teleop_type="so101_leader",
        auto_accept_calibration=True,
        background=True,
    )
    return result, proc


# ---------------------------------------------------------------------------
# the auto-accept write
# ---------------------------------------------------------------------------
def test_a_failed_auto_accept_is_reported(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A write that fails leaves a record naming the session and the reason.

    Without it the two outcomes are indistinguishable: the start result reads
    ``success`` either way and the session store reports a live pid either way,
    so nothing tells the operator the prompt is still waiting.
    """
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _result, proc = _start_with_auto_accept(monkeypatch, fail=True)

    assert proc.stdin.writes == [], "the failing pipe must not have accepted a newline"
    records = [r for r in caplog.records if "auto-accept" in r.getMessage()]
    assert records, "a failed auto-accept left no record at all"
    message = records[0].getMessage()
    assert records[0].levelno >= logging.WARNING, f"reported below WARNING: {records[0].levelname}"
    assert _SESSION in message, f"the record does not name the session: {message}"
    assert "Broken pipe" in message, f"the record does not carry the reason: {message}"


def test_a_failed_auto_accept_points_at_the_prompt_and_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The record says what is wrong and where to look, not just that it failed."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _start_with_auto_accept(monkeypatch, fail=True)

    reported = [r.getMessage() for r in caplog.records if "auto-accept" in r.getMessage()]
    assert reported, "a failed auto-accept left no record to inspect"
    message = reported[0]
    assert "prompt" in message, f"the record does not name the unanswered prompt: {message}"
    assert "status" in message and "log" in message, f"the record offers no next step: {message}"


def test_the_session_name_needle_is_not_satisfiable_by_the_prose(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The "names the session" assertion must not be satisfiable by the record's
    own wording.

    With the name stripped out, the remaining prose must not contain it. A name
    of ``"cal"`` failed this: the record says ``"calibration"``, so the
    assertion held whether or not the session was ever named.
    """
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _start_with_auto_accept(monkeypatch, fail=True)

    reported = [r.getMessage() for r in caplog.records if "auto-accept" in r.getMessage()]
    assert reported, "a failed auto-accept left no record to inspect"
    prose = reported[0].replace(repr(_SESSION), "").replace(_SESSION, "")
    assert _SESSION not in prose, (
        f"{_SESSION!r} occurs in the record's own prose, so naming the session proves nothing: {prose}"
    )


def test_the_start_result_still_reports_success_when_the_auto_accept_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record is additive: the launch itself succeeded and still says so.

    The child process really did start; only the courtesy write into its stdin
    failed. Turning that into an error envelope would misreport a running
    session, so the envelope is deliberately unchanged.
    """
    result, _proc = _start_with_auto_accept(monkeypatch, fail=True)

    assert result["status"] == "success"
    assert "Session Started" in _texts(result)
    assert SessionManager().get_session(_SESSION) is not None, (
        "the store must still report the session running - that is what makes the "
        "silent failure indistinguishable from a healthy start"
    )


def test_a_successful_auto_accept_stays_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A write that succeeds reports nothing, as the parameter's docs promise.

    ``auto_accept_calibration``'s documented posture is that nothing reports
    stdin *was* written to, so an unintended acceptance stays invisible. Only
    the failure is newly reported; this is the over-reach control for that.
    """
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        _start_with_auto_accept(monkeypatch, fail=False)

    assert [r.getMessage() for r in caplog.records if "auto-accept" in r.getMessage()] == []


def test_the_happy_path_delivers_two_newlines_and_closes_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-vacuity for the silence above: the auto-accept really did run.

    Without this, a body that wrote nothing at all would satisfy the
    silence assertion just as well as one that answered the prompt.
    """
    _result, proc = _start_with_auto_accept(monkeypatch, fail=False)

    assert proc.stdin.writes == ["\n", "\n"], "the calibration prompt was not answered twice"
    assert proc.stdin.closed, "stdin was left open after the responses"


# ---------------------------------------------------------------------------
# the stop refusal, and why the "No PID found" branch cannot be reached
# ---------------------------------------------------------------------------
def test_stop_refuses_an_absent_session() -> None:
    """``stop`` on a session the store does not hold refuses, naming it.

    This is the common case in practice rather than a corner: a session whose
    process has exited is pruned on the next store read, so the operator's
    ``stop`` arrives after it is already gone.
    """
    result = lerobot_teleoperate(action="stop", session_name="gone-forever")

    assert result["status"] == "error"
    text = _texts(result)
    assert "gone-forever" in text and "not found" in text


def test_stop_and_status_refuse_an_absent_session_identically() -> None:
    """One session-lookup failure, one wording across both actions.

    ``status``'s refusal was pinned and ``stop``'s was not, which is how the
    two could have drifted apart without anything failing.
    """
    stop = lerobot_teleoperate(action="stop", session_name="ghost-session")
    status = lerobot_teleoperate(action="status", session_name="ghost-session")

    assert stop["status"] == status["status"] == "error"
    assert _texts(stop) == _texts(status)


def test_a_pidless_session_record_is_pruned_so_the_no_pid_refusal_is_unreachable() -> None:
    """The store drops any record without a live pid, on every read.

    ``stop`` carries a "No PID found" refusal below the lookup. It cannot be
    reached through any input, because a record that reaches a caller has
    already been filtered on ``pid and psutil.pid_exists(pid)``. Pinning the
    pruning keeps that accounting honest: the day the store stops filtering,
    this fails and the refusal becomes live code that needs its own test.
    """
    manager = SessionManager()
    manager.add_session("nopid", {"start_time": 0.0, "action": "record"})

    assert manager.get_session("nopid") is None, "a pidless record survived the store's pruning"
    assert "nopid" not in manager.list_sessions()
