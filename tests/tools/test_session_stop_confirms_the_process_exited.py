"""A session reported stopped must be one whose process is known to be gone.

``lerobot_teleoperate`` and ``lerobot_train`` both expose ``action="stop"``, and
both used to escalate SIGTERM -> SIGKILL and then report ``"**Session
Stopped**"`` on the strength of having *sent* the signals:

    os.kill(pid, SIGTERM); time.sleep(2)
    if psutil.pid_exists(pid): os.kill(pid, SIGKILL)
    session_manager.remove_session(name)   # <- unconditional
    return {"status": "success", ...}      # <- unconditional

Sending SIGKILL is not the same as the process exiting. The kernel delivers it
asynchronously, and a task inside an uninterruptible wait - a serial ioctl on a
teleoperation bus, a stalled CUDA or network call in a training step - stays in
the process table until that wait returns. Two things then go wrong at once:
the caller is told the arm is released and the GPU is free, and the record is
dropped. That store is the only place a detached session's PID is written down
(``SessionManager._load_sessions`` says so, and
``tests.tools.test_teleop_session_store_keeps_a_live_pid`` pins it), so the
process carries on driving the robot with no supported way left to stop it.

The sibling teardown in the same package already refuses to do this:
``gr00t_inference._stop_service`` rescans the port after the escalation and
returns an error when anything still holds it, because "reporting success there
would tell the caller the port is free when the next bind is about to fail".
``policies.vera.server_runner.stop`` likewise waits after each signal. These
tests hold the two session verbs to the same standard, on both modules, and
additionally pin the identity half: the escalation is aimed at the process that
was captured before the first signal, so a PID recycled during the grace period
is not signalled a second time.
"""

from __future__ import annotations

import signal
from typing import Any

import pytest

import strands_robots.tools.lerobot_teleoperate as tele_mod
import strands_robots.tools.lerobot_train as train_mod
from strands_robots.tools import _process_stop

# Every recorded PID below is fake and every signal is captured, so nothing here
# can reach a real process.
FAKE_PID = 424242

# Both stop verbs are the same shape over their own session store, so each
# behaviour is pinned on both. ``kwargs`` carries the argument the tool needs
# beyond the action.
MODULES = [
    pytest.param(tele_mod, "lerobot_teleoperate", {}, id="teleoperate"),
    pytest.param(train_mod, "lerobot_train", {"dataset_root": "/x"}, id="train"),
]

_SIGNAMES = {signal.SIGTERM: "SIGTERM", signal.SIGKILL: "SIGKILL"}


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirect both session stores to a temp dir so no test touches the tree."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(tele_mod, "SESSION_DIR", session_dir)
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


def _install(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    *,
    survives_waits: int = 0,
    wait_exc: type[Exception] | None = None,
    missing_after: int | None = None,
) -> list[str]:
    """Stand in for the process being stopped and record what reaches it.

    Args:
        monkeypatch: pytest patcher.
        module: The tool module whose ``psutil``/``os`` seams to patch.
        survives_waits: How many confirmation waits time out before the process
            is reported gone. ``0`` exits within the SIGTERM grace period, ``1``
            exits only once SIGKILL lands, ``2`` outlives both.
        wait_exc: Raised by every wait instead of timing out, for the cases the
            probe itself reports (``NoSuchProcess`` for a recycled PID,
            ``AccessDenied`` for a process this user may not inspect).
        missing_after: Number of successful ``Process()`` constructions before
            the next one raises ``NoSuchProcess``. ``1`` models the real race:
            alive when the session store probes it, gone by the time the stop
            verb captures it.

    Returns:
        The event log: ``"wait"`` per confirmation and ``"kill:<SIGNAME>"`` per
        signal, in the order they happened.
    """
    events: list[str] = []
    real_psutil = module.psutil
    constructed = {"n": 0}

    class _Proc:
        def __init__(self, pid: int) -> None:
            constructed["n"] += 1
            if missing_after is not None and constructed["n"] > missing_after:
                raise real_psutil.NoSuchProcess(pid)
            self.pid = pid
            self._waits = 0

        def is_running(self) -> bool:
            # The session store's own liveness probe; keeps the record visible.
            return True

        def wait(self, timeout: float | None = None) -> int:
            self._waits += 1
            events.append("wait")
            if wait_exc is not None:
                raise wait_exc(self.pid)
            if self._waits <= survives_waits:
                raise real_psutil.TimeoutExpired(timeout)
            return 0

    monkeypatch.setattr(real_psutil, "Process", _Proc)
    monkeypatch.setattr(real_psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: events.append(f"kill:{_SIGNAMES[sig]}"))
    # The pre-fix implementation paced itself with a bare sleep; keep the suite
    # fast if it is ever reintroduced.
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    return events


def _stop(module: Any, tool_name: str, kwargs: dict[str, Any], name: str) -> dict[str, Any]:
    tool = getattr(module, tool_name)
    return tool(action="stop", session_name=name, **kwargs)


def _texts(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


def _add(module: Any, name: str) -> None:
    module.SessionManager().add_session(name, {"pid": FAKE_PID, "start_time": 0.0, "action": "train"})


# ---------------------------------------------------------------------------
# The report must be true.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("module", "tool_name", "kwargs"), MODULES)
def test_a_process_that_outlives_sigkill_is_reported_as_an_error(
    module: Any, tool_name: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Survived both signals: an error, and the record is kept so it stays stoppable."""
    _add(module, "stubborn")
    events = _install(monkeypatch, module, survives_waits=2)

    result = _stop(module, tool_name, kwargs, "stubborn")

    assert result["status"] == "error", "a process still in the table has not stopped"
    assert str(FAKE_PID) in _texts(result), "the report must name the PID that is still there"
    assert _json(result)["stopped"] is False
    assert module.SessionManager().get_session("stubborn") is not None, (
        "dropping the record loses the only note of the PID, so the session becomes unstoppable"
    )
    # The escalation itself is unchanged: both signals are still sent.
    assert events.count("kill:SIGTERM") == 1
    assert events.count("kill:SIGKILL") == 1


@pytest.mark.parametrize(("module", "tool_name", "kwargs"), MODULES)
def test_a_process_that_exits_in_the_grace_period_is_not_escalated(
    module: Any, tool_name: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM was enough: success, record dropped, and no SIGKILL is sent."""
    _add(module, "polite")
    events = _install(monkeypatch, module, survives_waits=0)

    result = _stop(module, tool_name, kwargs, "polite")

    assert result["status"] == "success"
    assert _json(result)["stopped"] is True
    assert module.SessionManager().get_session("polite") is None
    assert events == ["kill:SIGTERM", "wait"], "a process that exited must not also be SIGKILLed"


@pytest.mark.parametrize(("module", "tool_name", "kwargs"), MODULES)
def test_a_process_that_exits_only_after_sigkill_is_reported_stopped(
    module: Any, tool_name: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escalation still works: SIGTERM ignored, SIGKILL lands, stop succeeds."""
    _add(module, "stubborn_but_mortal")
    events = _install(monkeypatch, module, survives_waits=1)

    result = _stop(module, tool_name, kwargs, "stubborn_but_mortal")

    assert result["status"] == "success"
    assert module.SessionManager().get_session("stubborn_but_mortal") is None
    assert events == ["kill:SIGTERM", "wait", "kill:SIGKILL", "wait"]


@pytest.mark.parametrize(("module", "tool_name", "kwargs"), MODULES)
def test_a_process_that_cannot_be_inspected_is_not_reported_stopped(
    module: Any, tool_name: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AccessDenied`` is not death - the PID existing already said that.

    A session started under ``sudo`` for serial-port access and stopped as the
    invoking user reads this way. Not being allowed to look is no evidence the
    process left, so the verb must not claim it did.
    """
    _add(module, "sudo_session")
    _install(monkeypatch, module, wait_exc=module.psutil.AccessDenied)

    result = _stop(module, tool_name, kwargs, "sudo_session")

    assert result["status"] == "error"
    assert _json(result)["stopped"] is None, "unknown must stay unknown, not become False"
    assert module.SessionManager().get_session("sudo_session") is not None


# ---------------------------------------------------------------------------
# The escalation must be aimed at the process that was captured.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("module", "tool_name", "kwargs"), MODULES)
def test_a_pid_recycled_during_the_grace_period_is_not_signalled_again(
    module: Any, tool_name: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The captured process is gone, so nothing is escalated at its old PID.

    The probe reports ``NoSuchProcess`` (the captured process exited) while the
    PID still exists, which is what PID reuse looks like from here. Deciding the
    escalation on bare existence would send SIGKILL to whatever inherited the
    number; deciding it on the captured identity does not.
    """
    _add(module, "recycled")
    events = _install(monkeypatch, module, wait_exc=module.psutil.NoSuchProcess)

    result = _stop(module, tool_name, kwargs, "recycled")

    assert result["status"] == "success"
    assert "kill:SIGKILL" not in events, "SIGKILL at a recycled PID hits an unrelated process"
    assert module.SessionManager().get_session("recycled") is None


@pytest.mark.parametrize(("module", "tool_name", "kwargs"), MODULES)
def test_a_process_that_exits_before_the_capture_is_reported_already_stopped(
    module: Any, tool_name: str, kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alive at the store's liveness probe, gone at the capture: no signal is sent."""
    _add(module, "raced")
    events = _install(monkeypatch, module, missing_after=1)

    result = _stop(module, tool_name, kwargs, "raced")

    assert result["status"] == "success", "premise: the record must still be found"
    assert "already stopped" in _texts(result)
    assert events == [], "a process known to be gone must not be signalled at all"
    assert module.SessionManager().get_session("raced") is None


# ---------------------------------------------------------------------------
# The shared rule both verbs read.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("outcome", "expected", "why"),
    [
        pytest.param(None, True, "wait() returned, so the process is gone", id="returned"),
        pytest.param("timeout", False, "still in the process table", id="still-present"),
        pytest.param("no-such-process", True, "identity-checked disappearance", id="gone"),
        pytest.param("access-denied", None, "not inspectable is neither answer", id="unknown"),
    ],
)
def test_confirm_exit_reports_gone_present_and_unknown_apart(
    outcome: str | None, expected: bool | None, why: str
) -> None:
    """One rule, stated once, so both stop verbs cannot disagree about it.

    The third answer matters: collapsing "could not look" into either bool makes
    the caller state something it has no evidence for, in one direction or the
    other.
    """
    import psutil

    class _Proc:
        def wait(self, timeout: float | None = None) -> int:
            if outcome == "timeout":
                raise psutil.TimeoutExpired(timeout)
            if outcome == "no-such-process":
                raise psutil.NoSuchProcess(1)
            if outcome == "access-denied":
                raise psutil.AccessDenied(1)
            return 0

    assert _process_stop.confirm_exit(_Proc(), 0.01) is expected, why
