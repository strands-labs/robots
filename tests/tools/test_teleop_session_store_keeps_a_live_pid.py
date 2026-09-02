"""The teleop session store must not delete the record of a live process.

``SessionManager._load_sessions`` prunes finished sessions and *writes the
pruned map back to disk*, and that store is the only place a detached
teleoperation subprocess's PID is recorded. So the prune's classification is
load-bearing: a record dropped by mistake is gone for good, and the process it
named keeps driving the arm with no supported way to stop it.

Two probes decide the classification. ``psutil.pid_exists`` answers existence;
``Process(pid).is_running()`` refines it. When the second one raises they
disagree, and the two ways it can raise mean opposite things:

* ``NoSuchProcess`` - reaped between the two calls, so the record names nothing.
* ``AccessDenied`` - the process exists and this user may not inspect it (a
  session started under ``sudo`` for serial-port access, listed as the invoking
  user). Existence was already established, so this is not death.

These tests pin that only the first prunes, that no read path erases a record it
could not inspect, and that ``stop`` can still reach such a session. The prune
of a genuinely finished session is pinned unchanged alongside, so the retention
cannot grow into "never prune anything".

The training tool keeps a session store of the same shape and is held to the
same rule -- no record dropped on the strength of a probe that could not be
taken -- in ``tests.tools.test_train_session_store_keeps_a_live_pid``. Its
retention policy differs: it keeps a finished run for its log tail where this
one prunes it.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod

SessionManager = tele_mod.SessionManager
lerobot_teleoperate = tele_mod.lerobot_teleoperate


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the session store to a temp dir so no test touches the tree."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    return session_dir


def _live_pid() -> int:
    """A PID that certainly exists and that we own: this test process."""
    pid = os.getpid()
    assert tele_mod.psutil.pid_exists(pid), "premise: the test process must exist"
    return pid


def _raise_on_probe(monkeypatch: pytest.MonkeyPatch, module: Any, exc: type[Exception]) -> None:
    """Make every probe of the process raise ``exc`` while ``pid_exists`` stays truthful.

    ``stop`` confirms the exit with ``Process.wait()``, so a stand-in for an
    uninspectable process has to refuse that the same way it refuses
    ``is_running()``; answering one and not the other would model a process no
    kernel produces.
    """

    class _Probe:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def is_running(self) -> bool:
            raise exc(self._pid)

        def wait(self, timeout: float | None = None) -> int:
            raise exc(self._pid)

    monkeypatch.setattr(module.psutil, "Process", _Probe)


def _stored(mgr: Any) -> dict[str, Any]:
    """The records the store holds on disk, independent of what a load returns."""
    if not mgr.sessions_file.exists():
        return {}
    return json.loads(mgr.sessions_file.read_text())


# ---------------------------------------------------------------------------
# A process that exists but cannot be inspected keeps its record.
# ---------------------------------------------------------------------------
def test_an_uninspectable_live_session_survives_on_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AccessDenied`` on a PID that exists must not erase the stored record."""
    mgr = SessionManager()
    pid = _live_pid()
    mgr.add_session("arm_teleop", {"pid": pid, "action": "teleoperate", "start_time": 0.0})
    _raise_on_probe(monkeypatch, tele_mod, tele_mod.psutil.AccessDenied)

    assert mgr.get_session("arm_teleop") is not None, (
        "a session whose PID exists must remain reachable when the deeper probe is denied"
    )
    assert "arm_teleop" in _stored(mgr), (
        "the prune is written back to disk, so dropping the record destroys the only copy of the PID"
    )


def test_a_read_only_query_does_not_delete_such_a_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """``list`` is a query; it must not mutate the store it reads."""
    mgr = SessionManager()
    mgr.add_session("arm_teleop", {"pid": _live_pid(), "action": "teleoperate", "start_time": 0.0})
    _raise_on_probe(monkeypatch, tele_mod, tele_mod.psutil.AccessDenied)

    mgr.list_sessions()

    assert "arm_teleop" in _stored(mgr), "listing sessions erased the record it could not inspect"


def test_adding_a_session_does_not_erase_an_uninspectable_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every write path loads first, so a denied probe must not take bystanders."""
    mgr = SessionManager()
    mgr.add_session("arm_teleop", {"pid": _live_pid(), "action": "teleoperate", "start_time": 0.0})
    _raise_on_probe(monkeypatch, tele_mod, tele_mod.psutil.AccessDenied)

    mgr.add_session("second", {"pid": _live_pid(), "action": "record", "start_time": 0.0})

    stored = _stored(mgr)
    assert "second" in stored, "premise: the new session must be persisted"
    assert "arm_teleop" in stored, "starting a session erased a sibling whose probe was denied"


def test_stop_can_still_reach_a_session_it_could_not_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator-visible point: such a session stays stoppable.

    Reaching it is the property being pinned - the record survives the load and
    the signals go to the recorded PID. The *verdict* cannot be affirmative
    here: the same ``AccessDenied`` that hid the process from the store also
    hides whether it exited, and ``stop`` reports that as unknown rather than
    claiming an exit it could not observe. The record is kept either way, which
    is what keeps the session stoppable on the next attempt.
    """
    pid = _live_pid()
    SessionManager().add_session("arm_teleop", {"pid": pid, "action": "teleoperate", "start_time": 0.0})
    _raise_on_probe(monkeypatch, tele_mod, tele_mod.psutil.AccessDenied)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(tele_mod.os, "kill", lambda p, sig: signalled.append((p, sig)))
    monkeypatch.setattr(tele_mod.time, "sleep", lambda s: None)

    result = lerobot_teleoperate(action="stop", session_name="arm_teleop")

    assert signalled and signalled[0][0] == pid, "stop must signal the recorded PID"
    verdict = next(block["json"] for block in result["content"] if "json" in block)["stopped"]
    assert verdict is None, "an exit that could not be observed is unknown, not reported either way"
    assert "arm_teleop" in _stored(SessionManager()), "the record must survive so the session stays stoppable"


def test_retaining_the_record_is_reported(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A denied probe is the operator's only clue, so it must not be silent."""
    mgr = SessionManager()
    pid = _live_pid()
    mgr.add_session("arm_teleop", {"pid": pid, "action": "teleoperate", "start_time": 0.0})
    _raise_on_probe(monkeypatch, tele_mod, tele_mod.psutil.AccessDenied)

    with caplog.at_level("WARNING"):
        mgr.list_sessions()

    assert any(str(pid) in r.getMessage() for r in caplog.records), (
        "a session that could not be inspected must be reported, naming the PID"
    )


# ---------------------------------------------------------------------------
# Controls: a session that really is finished is still pruned.
# ---------------------------------------------------------------------------
def test_a_process_reaped_mid_probe_is_still_pruned(monkeypatch: pytest.MonkeyPatch) -> None:
    """``NoSuchProcess`` names nothing, so the record goes - including on disk."""
    mgr = SessionManager()
    mgr.add_session("racy", {"pid": _live_pid(), "action": "teleoperate", "start_time": 0.0})
    _raise_on_probe(monkeypatch, tele_mod, tele_mod.psutil.NoSuchProcess)

    assert mgr.list_sessions() == {}
    assert _stored(mgr) == {}, "a reaped session must still be pruned from the store"


def test_a_pid_that_no_longer_exists_is_still_pruned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary finished-session path is unchanged."""
    mgr = SessionManager()
    mgr.add_session("done", {"pid": _live_pid(), "action": "teleoperate", "start_time": 0.0})
    monkeypatch.setattr(tele_mod.psutil, "pid_exists", lambda pid: False)

    assert mgr.list_sessions() == {}
    assert _stored(mgr) == {}, "a session whose PID is gone must still be pruned"


def test_a_pid_that_is_not_running_is_still_pruned(monkeypatch: pytest.MonkeyPatch) -> None:
    """``is_running() -> False`` (a zombie, or a reused PID) is not our session."""
    mgr = SessionManager()
    mgr.add_session("zombie", {"pid": _live_pid(), "action": "teleoperate", "start_time": 0.0})

    class _NotRunning:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def is_running(self) -> bool:
            return False

    monkeypatch.setattr(tele_mod.psutil, "Process", _NotRunning)

    assert mgr.list_sessions() == {}
    assert _stored(mgr) == {}, "a PID that is not running must still be pruned"


# The sibling store is held to the same rule in
# ``tests.tools.test_train_session_store_keeps_a_live_pid``. It used to be
# checked from here, but only through ``list_sessions`` - a read - and that
# store's prune reaches disk through ``add_session``/``remove_session``, so the
# read alone could not see it drop the record. The write paths are graded there.
