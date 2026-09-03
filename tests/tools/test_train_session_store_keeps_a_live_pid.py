"""The training session store must not delete the record of a live process.

``lerobot_train`` runs training detached, and its on-disk session store is the
only place the subprocess's pid is written down: ``list``, ``status`` and
``stop`` all look the session up there. So what the load step chooses to leave
out is load-bearing, and it is load-bearing twice over, because
``add_session`` and ``remove_session`` are load-modify-write - a record the load
omits is erased from disk by the next session started or stopped, after which
the training process holds the GPU with no supported way left to stop it.

Leaving a record out is also not needed to avoid over-reporting it: presence in
the store is not the running claim. ``list`` and ``status`` each derive that from
``psutil.pid_exists`` at the moment they are asked, so a retained record reads as
running only while its pid really exists. That is pinned below as a control.

Two probes classify a record. ``psutil.pid_exists`` answers existence;
``Process(pid).is_running()`` refines it. Neither disagreement between them is
grounds for deleting the record:

* ``pid_exists`` False, or ``is_running()`` False - the run finished, or the pid
  was reused. Kept, so ``status`` can still report the final log tail.
* ``NoSuchProcess`` - reaped between the two probes. That is the same finished
  run as the row above, and which of the two paths a finished run takes is a
  race, so they must not be classified differently.
* ``AccessDenied`` - the process exists and this user may not inspect it; a
  session started under ``sudo`` and later listed as the invoking user reads this
  way. That is not death, and it is the one case where dropping the record loses
  a pid that still names a *live* process.

The teleoperation store is held to the same rule for the same reason, in
``tests.tools.test_teleop_session_store_keeps_a_live_pid``. The two policies are
not identical - that store prunes a finished session and this one retains it for
the log tail - but neither may drop a record on the strength of a probe it could
not take.
"""

from __future__ import annotations

import json
import os
import signal
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as train_mod

SessionManager = train_mod.SessionManager
lerobot_train = train_mod.lerobot_train

# Required by the tool signature and never read by the session actions. Spelled
# at each call rather than splatted from a dict, which mypy cannot match against
# the tool's typed keywords.
UNUSED_DATASET = "/unused"


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the session store to a temp dir so no test touches the tree."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


def _live_pid() -> int:
    """A pid that certainly exists and that we own: this test process."""
    pid = os.getpid()
    assert train_mod.psutil.pid_exists(pid), "premise: the test process must exist"
    return pid


def _raise_on_probe(monkeypatch: pytest.MonkeyPatch, exc: type[Exception]) -> None:
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

    monkeypatch.setattr(train_mod.psutil, "Process", _Probe)


def _stored(mgr: Any) -> dict[str, Any]:
    """The records the store holds on disk, independent of what a load returns."""
    if not mgr.sessions_file.exists():
        return {}
    return json.loads(mgr.sessions_file.read_text())


def _seed(name: str = "training", **extra: Any) -> tuple[Any, int]:
    """A store holding one session whose pid is live, and that pid."""
    mgr = SessionManager()
    pid = _live_pid()
    mgr.add_session(name, {"pid": pid, "action": "train", "start_time": 0.0, **extra})
    assert name in _stored(mgr), "premise: the session must reach disk"
    return mgr, pid


# ---------------------------------------------------------------------------
# A process that exists but cannot be inspected keeps its record.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("read", "why"),
    [
        pytest.param(lambda mgr: "training" in mgr.list_sessions(), "list must show it", id="list"),
        pytest.param(lambda mgr: mgr.get_session("training") is not None, "stop looks it up here", id="get"),
    ],
)
def test_an_uninspectable_live_session_is_still_visible(monkeypatch: pytest.MonkeyPatch, read: Any, why: str) -> None:
    """``AccessDenied`` on a pid that exists must not hide the session."""
    mgr, _ = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    assert read(mgr), f"a session whose pid exists must stay visible when the deeper probe is denied: {why}"


@pytest.mark.parametrize(
    ("mutate", "id_"),
    [
        pytest.param(lambda mgr: mgr.add_session("second", {"pid": 4242, "action": "train"}), "starting", id="start"),
        pytest.param(lambda mgr: mgr.remove_session("second"), "stopping", id="stop"),
    ],
)
def test_a_write_path_does_not_erase_an_uninspectable_sibling(
    monkeypatch: pytest.MonkeyPatch, mutate: Any, id_: str
) -> None:
    """The consequence a read-only check cannot see: the omission reaches disk.

    ``add_session`` and ``remove_session`` both load, modify and write back, so a
    record missing from the load is deleted from the store by the next session
    started or stopped - not by anything the operator did to *this* session.
    """
    mgr, _ = _seed()
    mgr.add_session("second", {"pid": 4242, "action": "train"})
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    mutate(mgr)

    assert "training" in _stored(mgr), (
        f"{id_} another session erased the record of one whose probe was denied, "
        "and that store is the only place its pid was written down"
    )


def test_an_uninspectable_session_can_still_be_deleted_on_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inversion this rule removes, in its sharpest form.

    ``remove_session`` deletes ``name`` only if the load it does first returned
    it, so a record the load omitted could not be deleted *on request* - while
    the very same omission deleted it as a side effect of touching an unrelated
    session. Retention makes the explicit removal the thing that works, which is
    also what keeps retention from growing into "nothing can ever be deleted".
    """
    mgr, _ = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    mgr.remove_session("training")

    assert mgr.get_session("training") is None, "an explicit removal must delete the record"
    assert _stored(mgr) == {}, "and must reach disk"


def test_stop_can_still_reach_a_session_it_could_not_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator-visible point: such a session stays stoppable.

    Reaching it is the property pinned here - the record survives the load and
    the signals go to the recorded pid. The *verdict* cannot be affirmative: the
    same ``AccessDenied`` that hid the process from the store also hides whether
    it exited, and ``stop`` reports that as unknown rather than claiming an exit
    it could not observe. Before this rule reached the training store that report
    was unreachable here, because the record was gone by the time ``stop``
    looked it up.
    """
    _, pid = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(train_mod.os, "kill", lambda p, sig: signalled.append((p, sig)))

    result = lerobot_train(dataset_root=UNUSED_DATASET, action="stop", session_name="training")

    # psutil.pid_exists probes with signal 0, so only a real signal counts as a stop.
    sent = [entry for entry in signalled if entry[1] != 0]
    assert sent and sent[0] == (pid, signal.SIGTERM), f"stop must signal the recorded pid, sent {sent}"
    verdict = next(block["json"] for block in result["content"] if "json" in block)["stopped"]
    assert verdict is None, "an exit that could not be observed is unknown, not reported either way"
    assert "training" in _stored(SessionManager()), "the record must survive so the session stays stoppable"


def test_retaining_the_record_is_reported(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A denied probe is the operator's only clue, so it must not be silent."""
    mgr, pid = _seed()
    _raise_on_probe(monkeypatch, train_mod.psutil.AccessDenied)

    with caplog.at_level("WARNING"):
        mgr.list_sessions()

    assert any(str(pid) in r.getMessage() for r in caplog.records), (
        "a session that could not be inspected must be reported, naming the pid"
    )


class _NotRunning:
    """A process object that answers the probe with "not running"."""

    def __init__(self, pid: int) -> None:
        self._pid = pid

    def is_running(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# A finished run is retained too, however the probes report it.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arrange", "id_"),
    [
        pytest.param(
            lambda mp: mp.setattr(train_mod.psutil, "pid_exists", lambda pid: False),
            "the pid is gone",
            id="pid-absent",
        ),
        pytest.param(
            lambda mp: _raise_on_probe(mp, train_mod.psutil.NoSuchProcess),
            "reaped between the two probes",
            id="no-such-process",
        ),
        pytest.param(
            lambda mp: mp.setattr(train_mod.psutil, "Process", _NotRunning),
            "is_running() says no",
            id="not-running",
        ),
    ],
)
def test_a_finished_run_keeps_its_record(monkeypatch: pytest.MonkeyPatch, arrange: Any, id_: str) -> None:
    """All three spellings of "finished" are one state, so all three are kept.

    This store retains a finished session on purpose, so ``status`` can still
    report the final log tail. Which spelling a given finish produces is a race
    between two probes, so classifying them differently would make retention
    depend on timing rather than on the run.
    """
    mgr, _ = _seed()
    arrange(monkeypatch)

    assert "training" in mgr.list_sessions(), f"a finished run must keep its record ({id_})"
    assert "training" in _stored(mgr), f"and must keep it on disk ({id_})"


# ---------------------------------------------------------------------------
# Controls: retention is not a running claim.
# ---------------------------------------------------------------------------
def test_a_retained_record_is_not_reported_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeping the record cannot be what makes a session read as running.

    This is the reason retention is safe: ``status`` derives the live/finished
    line from the pid's existence when asked, not from the record's presence.
    """
    _seed()
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: False)

    result = lerobot_train(dataset_root=UNUSED_DATASET, action="status", session_name="training")

    text = next(block["text"] for block in result["content"] if "text" in block)
    assert "Status: Stopped" in text, f"a retained record whose pid is gone must read as stopped: {text}"


def test_a_corrupt_store_still_degrades_to_empty() -> None:
    """Retention is about classification, not about tolerating a broken file."""
    mgr = SessionManager()
    mgr.sessions_file.parent.mkdir(parents=True, exist_ok=True)
    mgr.sessions_file.write_text("{not json")

    assert mgr.list_sessions() == {}


def _store_one_record_with_an_undecodable_byte(mgr: Any, pid: int) -> None:
    """Write a store whose JSON is well-formed but whose bytes are not UTF-8.

    The 0xE9 sits inside a session *name*, which is the field a hand-edit or a
    latin-1 writer touches; every other field, the pid included, stays ASCII.
    """
    mgr.sessions_file.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"run-cafe": {"pid": pid, "action": "train", "start_time": 0.0}}, indent=2)
    mgr.sessions_file.write_bytes(raw.encode("utf-8").replace(b"cafe", b"caf\xe9"))


def test_a_store_byte_that_is_not_utf8_still_yields_the_pid_it_names() -> None:
    """The one damage shape the handler cannot answer, so the read must not raise.

    ``UnicodeDecodeError`` is a ``ValueError``, so it is neither of the two
    failures ``except (OSError, JSONDecodeError)`` above was written for - the
    store is gone, or it is not JSON. A strict read therefore aborts whichever
    action consulted the store, and by the module docstring that includes
    ``stop``, which is the only supported way to end a detached run. Damage
    stays in the field that carries it instead: a pid is ASCII either way.
    """
    mgr = SessionManager()
    pid = _live_pid()
    _store_one_record_with_an_undecodable_byte(mgr, pid)

    loaded = mgr.list_sessions()

    assert [info["pid"] for info in loaded.values()] == [pid], (
        f"a byte outside the pid's field must not cost the pid: {loaded}"
    )


def test_stop_reaches_a_session_whose_record_carries_an_undecodable_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator-visible point, driven through the tool rather than the store.

    The name to pass is the one ``list`` reports, since that is the only
    spelling of it the operator can see; what is pinned is that the signals
    still reach the recorded pid.
    """
    mgr = SessionManager()
    pid = _live_pid()
    _store_one_record_with_an_undecodable_byte(mgr, pid)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(train_mod.os, "kill", lambda p, sig: signalled.append((p, sig)))

    listed = lerobot_train(dataset_root=UNUSED_DATASET, action="list")
    assert listed["status"] == "success", f"list must report the store it can read: {listed}"
    name = next(iter(next(block["json"] for block in listed["content"] if "json" in block)["sessions"]))

    lerobot_train(dataset_root=UNUSED_DATASET, action="stop", session_name=name)

    # psutil.pid_exists probes with signal 0, so only a real signal counts as a stop.
    sent = [entry for entry in signalled if entry[1] != 0]
    assert sent and sent[0] == (pid, signal.SIGTERM), f"stop must signal the recorded pid, sent {sent}"
