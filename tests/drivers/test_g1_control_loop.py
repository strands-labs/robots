"""Tests for the G1 control loop wired by harness#361 PR-C.

These grade the loop's transport primitive: 500 Hz cadence, per-step re-gate,
zero-torque frame on every terminal path except a wire refusal, exit reasons
named on every branch.  The unitree_sdk2py imports are lazy on both the
builders and the loop, so the tests run without the SDK on the box - the
publisher is a callable double the driver records writes on.

The loop's shutdown path is verified two ways: by inspecting the exit reason
in the snapshot, and by counting the frames the publisher recorded.  A stop
always leaves the driver with ``_loop = None`` so a subsequent ``run_policy``
starts fresh.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.drivers import g1 as g1_mod
from strands_robots.drivers.g1 import (
    _CONTROL_LOOP_DT,
    _CONTROL_LOOP_HZ,
    _ControlLoop,
    _refusal_text,
)

# ---------------------------------------------------------------------------
# SDK stub.  The production ``_run`` lane imports
# ``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_`` and
# ``_build_lowcmd_from_action`` imports ``unitree_sdk2py.idl.default`` and
# ``unitree_sdk2py.utils.crc``.  On an SDK-less CI box we install a stub via
# :mod:`sys.modules` so the same lane hardware runs is exercised here: the
# alternative - a separate "SDK missing" branch - would grade a fallback
# hardware can never take - the concern review feedback raised on #2779.
#
# Follows AGENTS.md > Testing Patterns > Restore a sys.modules entry you
# remove: an autouse fixture installs the stubs before every test in this
# module and removes them after, so cross-test pollution is impossible.
# ---------------------------------------------------------------------------


class _StubMotorCmd:
    """One slot of ``LowCmd_.motor_cmd``.  Fields track the real IDL layout."""

    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd")

    def __init__(self) -> None:
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class _StubLowCmd:
    """Stand-in for ``unitree_sdk2py.idl.default.unitree_hg_msg_dds__LowCmd_()``.

    ``motor_cmd`` is a 35-slot list, matching the real IDL width the
    builder's ``assert len(cmd.motor_cmd) >= _G1_NAMED_JOINTS`` checks.
    """

    def __init__(self) -> None:
        self.mode_pr: int = 0
        self.mode_machine: int = 0
        self.motor_cmd: list[_StubMotorCmd] = [_StubMotorCmd() for _ in range(35)]
        self.crc: int = 0


class _StubCRC:
    """Stand-in for ``unitree_sdk2py.utils.crc.CRC``.

    The wire's CRC is verified by firmware; the test suite grades that the
    builder invokes ``.Crc(cmd)`` after populating every other field, not
    the CRC value itself.  ``42`` is a distinguishable non-zero.
    """

    def Crc(self, _cmd: Any) -> int:
        return 42


@pytest.fixture(autouse=True)
def _stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``unitree_sdk2py`` stub for the duration of one test.

    Every submodule the driver imports is registered on :mod:`sys.modules`
    so ``from unitree_sdk2py.idl.default import ...`` and its siblings
    resolve here.  ``monkeypatch.setitem`` restores the previous value
    (typically absent) on teardown.
    """
    root = types.ModuleType("unitree_sdk2py")
    idl = types.ModuleType("unitree_sdk2py.idl")
    default = types.ModuleType("unitree_sdk2py.idl.default")
    unitree_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    unitree_hg_msg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    dds_ = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    utils = types.ModuleType("unitree_sdk2py.utils")
    crc = types.ModuleType("unitree_sdk2py.utils.crc")

    default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    dds_.LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    crc.CRC = _StubCRC  # type: ignore[attr-defined]

    for name, mod in [
        ("unitree_sdk2py", root),
        ("unitree_sdk2py.idl", idl),
        ("unitree_sdk2py.idl.default", default),
        ("unitree_sdk2py.idl.unitree_hg", unitree_hg),
        ("unitree_sdk2py.idl.unitree_hg.msg", unitree_hg_msg),
        ("unitree_sdk2py.idl.unitree_hg.msg.dds_", dds_),
        ("unitree_sdk2py.utils", utils),
        ("unitree_sdk2py.utils.crc", crc),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


# ---------------------------------------------------------------------------
# Fakes.  The loop reaches into the driver's cached observations and its
# ``_pubs``; a MagicMock driver with those attributes exercises the loop
# without needing a real DDS bus or an SDK on the box.
# ---------------------------------------------------------------------------


class _RecordingPublisher:
    """Stand-in for ``DDSPublisher`` that records every ``publish`` call.

    Enough of the interface to grade the loop's write path.  ``publish``
    returns ``None`` on success and the string on failure - the same shape
    the real publisher returns.
    """

    def __init__(self, refuse_after: int | None = None, reason: str = "publish refused") -> None:
        self.calls: list[Any] = []
        self.closed = False
        self._refuse_after = refuse_after
        self._reason = reason

    def publish(self, topic: str, klass: Any, cmd: Any) -> str | None:
        self.calls.append((topic, klass, cmd))
        if self._refuse_after is not None and len(self.calls) > self._refuse_after:
            return self._reason
        return None

    def close(self) -> None:
        """No-op release.  Matches ``DDSPublisher.close()``."""
        self.closed = True


def _fake_driver(
    mode_machine: int | None = 9,
    fsm_id: int | None = 500,
    gate_result: dict[str, Any] | None = None,
    publisher: _RecordingPublisher | None = None,
) -> Any:
    """Return a MagicMock driver good enough for the loop.

    Every attribute the loop reads is a real value (not a Mock stand-in) so
    a typo in the loop code fails on AttributeError rather than reading a
    Mock silently.  ``_task_admission`` is a real ``threading.Lock`` so
    the exit path can acquire it as production does; the alternative
    (a MagicMock context manager) would grade a lock that never contended.
    """
    import threading as _threading

    driver = MagicMock(
        spec=[
            "_mode_machine",
            "_fsm_id",
            "_battery",
            "_imu",
            "_pubs",
            "_check_motion_gates",
            "_loop",
            "_task_admission",
            "_tool_name",
            "_refresh_fsm_id",
            "_fsm_read_at",
            "_motion_switcher_lock",
        ]
    )
    driver._mode_machine = mode_machine
    driver._fsm_id = fsm_id
    driver._tool_name = "g1"
    # The FSM refresher thread the loop owns calls this at ``_FSM_REFRESH_HZ``
    # and stamps ``_fsm_read_at`` on every authoritative reading.  A no-op
    # ``MagicMock`` plus a live timestamp is the healthy-wire case, which is
    # what every test in this file is about; the refresher's own contract is
    # graded in test_g1_fsm_refresh_is_off_the_control_loop_thread.py.
    driver._refresh_fsm_id = MagicMock(side_effect=lambda: setattr(driver, "_fsm_read_at", time.monotonic()))
    driver._fsm_read_at = time.monotonic()
    driver._motion_switcher_lock = _threading.Lock()
    driver._battery = {"pct": 80.0}
    driver._imu = {"rpy": [0.0, 0.0, 0.0]}
    driver._pubs = publisher if publisher is not None else _RecordingPublisher()
    driver._check_motion_gates = MagicMock(return_value=gate_result)
    driver._loop = None
    driver._task_admission = _threading.Lock()
    return driver


def _wait_finished(loop: _ControlLoop, timeout: float = 2.0) -> None:
    """Poll ``is_running`` until the loop has joined its thread."""
    deadline = time.monotonic() + timeout
    while loop.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not loop.is_running, "loop did not finish within timeout"


# ---------------------------------------------------------------------------
# Cadence constants.
# ---------------------------------------------------------------------------


class TestCadenceConstants:
    """The SDK example uses 500 Hz; assert the constants match."""

    def test_500hz_matches_sdk_reference(self) -> None:
        assert _CONTROL_LOOP_HZ == 500.0

    def test_dt_is_the_reciprocal(self) -> None:
        assert _CONTROL_LOOP_DT == pytest.approx(0.002, rel=1e-6)


# ---------------------------------------------------------------------------
# Exit reasons.  Every terminal path names itself.
# ---------------------------------------------------------------------------


class TestExitReasons:
    """Every branch that leaves ``_run`` sets ``exit_reason``."""

    def test_n_steps_budget_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=3)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "n_steps"
        assert snap["steps"] == 3
        assert driver._loop is None, "loop must clear its reference on exit"

    def test_duration_budget_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        # Sub-tick duration so ``now >= deadline`` fires on the first check.
        loop = _ControlLoop(driver=driver, policy=policy, duration=0.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "duration"

    def test_gate_flip_exits_named_with_reason(self) -> None:
        refusal = {"status": "error", "content": [{"text": "FSM 999 refuses arm writes"}]}
        driver = _fake_driver(gate_result=refusal)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "gate"
        assert snap["exit_detail"] == "FSM 999 refuses arm writes"

    def test_policy_returning_none_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> None:
            return None

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "policy"
        assert "None" in (snap["exit_detail"] or "")
        assert snap["refusals"] == 1

    def test_policy_raising_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> None:
            raise RuntimeError("boom")

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "policy"
        assert "RuntimeError" in (snap["exit_detail"] or "")
        assert "boom" in (snap["exit_detail"] or "")

    def test_stop_task_exits_named(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=None)
        loop.start()
        # Give it a moment to take at least one step.
        time.sleep(0.05)
        loop.stop("stop_task")
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "stop_task"

    def test_missing_publisher_exits_named(self) -> None:
        driver = _fake_driver()
        driver._pubs = None

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "publish"
        assert "not connected" in (snap["exit_detail"] or "")


# ---------------------------------------------------------------------------
# Zero-torque shutdown.  Every exit publishes it except a wire refusal.
# ---------------------------------------------------------------------------


class TestZeroTorqueShutdown:
    """The last frame the loop publishes is the zero-torque frame."""

    def _last_frame(self, pub: _RecordingPublisher) -> Any:
        assert pub.calls, "expected at least one publish call"
        return pub.calls[-1][2]

    def test_stop_publishes_zero_torque(self) -> None:
        pub = _RecordingPublisher()
        driver = _fake_driver(publisher=pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=None)
        loop.start()
        time.sleep(0.05)
        loop.stop("stop_task")
        _wait_finished(loop)

        # The last publish must be the zero-torque frame.  Distinguish it
        # from the loop's action frames by checking that the commanded slot
        # for left_knee (index looked up via the module's mapping) carries
        # q=0.0/tau=0.0.  Any post-stop frame beyond the last action counts.
        cmd = self._last_frame(pub)
        left_knee_slot = g1_mod._G1_JOINT_INDEX["left_knee"]
        assert cmd.motor_cmd[left_knee_slot].q == 0.0
        assert cmd.motor_cmd[left_knee_slot].tau == 0.0
        assert cmd.motor_cmd[left_knee_slot].kp == 0.0

    def test_gate_flip_publishes_zero_torque(self) -> None:
        # First step passes the gate, second step refuses.
        refusal = {"status": "error", "content": [{"text": "FSM 999 refuses arm writes"}]}
        gate_returns = [None, refusal]
        driver = _fake_driver()
        driver._check_motion_gates = MagicMock(side_effect=gate_returns)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        pub = driver._pubs
        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        # First action frame, then zero-torque on gate refusal.
        assert len(pub.calls) >= 2
        left_knee_slot = g1_mod._G1_JOINT_INDEX["left_knee"]
        stop_frame = pub.calls[-1][2]
        assert stop_frame.motor_cmd[left_knee_slot].kp == 0.0
        assert loop.snapshot()["exit_reason"] == "gate"

    def test_publish_refusal_does_not_double_stamp(self) -> None:
        """When the wire refuses, the loop does not stamp another frame.

        A second publish would clobber the reason with a fresh wire error
        rather than surfacing the original refusal.
        """
        # Refuse immediately - the first action publish fails.
        pub = _RecordingPublisher(refuse_after=0, reason="dds refused")
        driver = _fake_driver(publisher=pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "publish"
        assert snap["exit_detail"] == "dds refused"
        # Exactly one call - the action attempt.  No zero-torque follow-up.
        assert len(pub.calls) == 1


# ---------------------------------------------------------------------------
# Per-step re-gate.  The gate is consulted on every step, not once at start.
# ---------------------------------------------------------------------------


class TestPerStepReGate:
    """The FSM gate runs on every iteration, not just at start."""

    def test_gate_is_called_once_per_step(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=5)
        loop.start()
        _wait_finished(loop)

        # Called ``n_steps`` times (one per step; the ``n_steps`` budget
        # check fires before the gate on the 6th iteration).
        assert driver._check_motion_gates.call_count == 5

    def test_gate_scope_is_motion(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=2)
        loop.start()
        _wait_finished(loop)

        # ``_check_motion_gates("motion", refresh=False)`` on every call: scope
        # is "motion", and the per-step re-gate consults the cache rather than
        # performing a DDS round trip inside a 2 ms step.
        for call in driver._check_motion_gates.call_args_list:
            args, kwargs = call
            assert args == ("motion",)
            assert kwargs == {"refresh": False}


# ---------------------------------------------------------------------------
# Snapshot invariants.
# ---------------------------------------------------------------------------


class TestSnapshot:
    """``snapshot()`` returns a consistent shape from any thread."""

    def test_snapshot_before_start_is_stable(self) -> None:
        driver = _fake_driver()
        loop = _ControlLoop(driver=driver, policy=lambda _o: None, duration=1.0, n_steps=1)
        snap = loop.snapshot()
        assert snap["running"] is False
        assert snap["steps"] == 0
        assert snap["exit_reason"] is None
        assert snap["elapsed_s"] is None
        assert snap["hz"] == _CONTROL_LOOP_HZ

    def test_snapshot_shape_after_exit(self) -> None:
        driver = _fake_driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=1)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        # Every documented field is present.
        assert set(snap) == {
            "running",
            "steps",
            "refusals",
            "elapsed_s",
            "duration_budget_s",
            "n_steps_budget",
            "exit_reason",
            "exit_detail",
            "hz",
            "fsm_refresh_hz",
            "fsm_reads",
        }
        assert snap["running"] is False
        assert snap["elapsed_s"] is not None and snap["elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# Refusal text helper.
# ---------------------------------------------------------------------------


class TestRefusalText:
    """The helper extracts the first text entry, or a default."""

    def test_extracts_text_from_content(self) -> None:
        env = {"status": "error", "content": [{"text": "the reason"}]}
        assert _refusal_text(env) == "the reason"

    def test_returns_default_when_content_is_empty(self) -> None:
        assert _refusal_text({"status": "error", "content": []}) == "refused"

    def test_skips_non_text_entries(self) -> None:
        env = {"status": "error", "content": [{"json": {}}, {"text": "found"}]}
        assert _refusal_text(env) == "found"


# ---------------------------------------------------------------------------
# Single production lane.  Response to review feedback on #2779: the loop
# must not carry a second
# publish branch for SDK-less boxes.  Every publish here goes through
# ``_build_lowcmd_from_action`` so an unknown joint name refuses the whole
# action, and the wire frame passed to the publisher has the SDK-shaped
# ``motor_cmd`` array the mypy signature declares.
# ---------------------------------------------------------------------------


class TestSingleProductionLane:
    """The publish path is one code path, whether the SDK is on the box or not."""

    def test_publish_receives_a_low_cmd_never_a_dict(self) -> None:
        """The second argument to ``publish`` is the ``LowCmd_`` class, not ``None``.

        Line 1248 of the previous head passed ``None`` on SDK-less boxes,
        which failed mypy (``Argument 2 ... incompatible type "None"; expected "type"``)
        and skipped the builder's joint-name allowlist.
        """
        pub = _RecordingPublisher()
        driver = _fake_driver(publisher=pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.1}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=2)
        loop.start()
        _wait_finished(loop)

        # Every recorded publish carried a class (not None) and a
        # ``LowCmd_``-shaped object (not the raw action dict).
        for _topic, klass, cmd in pub.calls:
            assert klass is _StubLowCmd
            assert hasattr(cmd, "motor_cmd"), "publish received a raw dict"

    def test_policy_returning_unknown_joint_refuses(self) -> None:
        """An unknown joint name refuses the whole action.

        The removed SDK-less lane published the raw dict and counted it
        as a step, so ``{"no_such_joint": 1e9}`` would advance ``n_steps``
        and exit as success.  Now the builder rejects it up front.
        """
        pub = _RecordingPublisher()
        driver = _fake_driver(publisher=pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"no_such_joint": 1.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=1.0, n_steps=None)
        loop.start()
        _wait_finished(loop)

        snap = loop.snapshot()
        assert snap["exit_reason"] == "policy"
        assert snap["steps"] == 0
        assert "unknown joint name" in (snap["exit_detail"] or "")


# ---------------------------------------------------------------------------
# Zero-torque contract: ``cleanup()`` and ``stop()`` join the loop first.
# Response to review feedback: ``cleanup()`` used to
# close ``_pubs`` under a live 500 Hz thread, which drove the loop into its
# ``publish`` branch and skipped the zero-torque frame.
# ---------------------------------------------------------------------------


class TestCleanupJoinsLoopBeforeClosingPubs:
    """``G1Driver.cleanup()`` publishes zero-torque before releasing pubs."""

    def _build_connected_driver(self, pub: _RecordingPublisher) -> Any:
        """A real ``G1Driver`` wired to the recording publisher.

        Instantiated with the tools sentinel that bypasses DDS init - the
        driver's task-admission lock and lifecycle methods are what we grade.
        """
        from strands_robots.drivers.g1 import G1Driver

        driver = G1Driver(port="127.0.0.1", network_interface="lo")
        driver._pubs = pub  # type: ignore[assignment]
        driver._subs = None
        driver._connected = True
        driver._mode_machine = 9
        driver._fsm_id = 500
        driver._battery = {"pct": 80.0}
        driver._imu = {"rpy": [0.0, 0.0, 0.0]}
        return driver

    def test_cleanup_stops_running_loop_first(self) -> None:
        pub = _RecordingPublisher()
        driver = self._build_connected_driver(pub)

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=None)
        driver._loop = loop
        loop.start()
        time.sleep(0.05)
        # ``cleanup()`` in the pre-fix head closed ``_pubs`` while the
        # loop was still publishing; the next iteration then took the
        # publish branch and the zero-torque frame never went out.
        driver.cleanup()

        assert not loop.is_running
        # ``_pubs`` was set to ``None`` after the join, but the publisher
        # we saved on the side retains the recorded calls.
        assert driver._pubs is None
        # Last recorded frame is the zero-torque frame (kp=0 on left_knee).
        assert pub.calls, "cleanup joined the loop with no publishes"
        left_knee_slot = g1_mod._G1_JOINT_INDEX["left_knee"]
        assert pub.calls[-1][2].motor_cmd[left_knee_slot].kp == 0.0

    def test_cleanup_is_idempotent_with_no_loop(self) -> None:
        pub = _RecordingPublisher()
        driver = self._build_connected_driver(pub)
        driver.cleanup()  # No loop running.
        assert driver._pubs is None
        assert driver._connected is False


# ---------------------------------------------------------------------------
# Admission race.  Response to review feedback: the
# check-then-act on ``self._loop`` was unlocked, and a concurrent
# ``run_policy`` could pass ``is_running == False`` before either thread
# assigned the reference.
# ---------------------------------------------------------------------------


class TestRunPolicyAdmissionLock:
    """Two concurrent ``run_policy`` calls never both start."""

    def test_second_run_policy_refuses_when_first_is_running(self) -> None:
        from strands_robots.drivers.g1 import G1Driver

        driver = G1Driver(port="127.0.0.1", network_interface="lo")
        driver._pubs = _RecordingPublisher()  # type: ignore[assignment]
        driver._connected = True
        driver._mode_machine = 9
        driver._fsm_id = 500
        driver._battery = {"pct": 80.0}
        driver._imu = {"rpy": [0.0, 0.0, 0.0]}

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        first = driver.run_policy(policy, duration=60.0)
        assert first["status"] == "success"

        second = driver.run_policy(policy, duration=60.0)
        assert second["status"] == "error"
        text = second["content"][0]["text"]
        assert "already running" in text

        # Clean up.
        driver.cleanup()

    def test_concurrent_run_policy_only_one_wins(self) -> None:
        """Fifty threads calling ``run_policy`` at once produce one loop.

        Without the admission lock, two threads could pass the
        ``is_running`` check before either assigned ``self._loop``, and
        both loops would publish on ``rt/lowcmd`` at 500 Hz.
        """
        import threading as _threading

        from strands_robots.drivers.g1 import G1Driver

        driver = G1Driver(port="127.0.0.1", network_interface="lo")
        driver._pubs = _RecordingPublisher()  # type: ignore[assignment]
        driver._connected = True
        driver._mode_machine = 9
        driver._fsm_id = 500
        driver._battery = {"pct": 80.0}
        driver._imu = {"rpy": [0.0, 0.0, 0.0]}

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        started = _threading.Barrier(50)
        results: list[dict[str, Any]] = []
        results_lock = _threading.Lock()

        def caller() -> None:
            started.wait()
            r = driver.run_policy(policy, duration=60.0)
            with results_lock:
                results.append(r)

        threads = [_threading.Thread(target=caller) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one succeeded; the rest carried the refusal.
        successes = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]
        assert len(successes) == 1, f"admission lock leaked: {len(successes)} rollouts started concurrently"
        assert len(errors) == 49

        driver.cleanup()


# ---------------------------------------------------------------------------
# Duration and n_steps validation.  Response to review feedback:
# unvalidated ``duration=nan`` actuated the loop with no budget
# at all, and ``n_steps=True`` silently capped at 1.
# ---------------------------------------------------------------------------


class TestRunPolicyValidatesBudgets:
    """``duration`` and ``n_steps`` are refused on the same domains HardwareRobot enforces."""

    def _driver(self) -> Any:
        from strands_robots.drivers.g1 import G1Driver

        driver = G1Driver(port="127.0.0.1", network_interface="lo")
        driver._pubs = _RecordingPublisher()  # type: ignore[assignment]
        driver._connected = True
        driver._mode_machine = 9
        driver._fsm_id = 500
        driver._battery = {"pct": 80.0}
        driver._imu = {"rpy": [0.0, 0.0, 0.0]}
        return driver

    def test_duration_nan_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=float("nan"))
        assert r["status"] == "error"
        assert "duration" in r["content"][0]["text"]

    def test_duration_inf_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=float("inf"))
        assert r["status"] == "error"
        assert "duration" in r["content"][0]["text"]

    def test_duration_zero_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=0.0)
        assert r["status"] == "error"

    def test_duration_negative_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=-1.0)
        assert r["status"] == "error"

    def test_duration_string_refused_not_raised(self) -> None:
        """A non-numeric ``duration`` returns an envelope, not raises.

        The pre-fix head ran ``float(duration)`` in the constructor, which
        raised ``ValueError`` out of a method whose siblings return error
        dicts - escaping the envelope contract.
        """
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration="abc")  # type: ignore[arg-type]
        assert r["status"] == "error"

    def test_n_steps_bool_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=1.0, n_steps=True)  # type: ignore[arg-type]
        assert r["status"] == "error"
        assert "n_steps" in r["content"][0]["text"]

    def test_n_steps_zero_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=1.0, n_steps=0)
        assert r["status"] == "error"

    def test_n_steps_negative_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=1.0, n_steps=-5)
        assert r["status"] == "error"

    def test_n_steps_float_refused(self) -> None:
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=1.0, n_steps=2.7)  # type: ignore[arg-type]
        assert r["status"] == "error"

    def test_duration_positive_and_n_steps_none_accepted(self) -> None:
        """A valid duration with default ``n_steps=None`` accepts."""
        driver = self._driver()

        def policy(_obs: Any) -> dict[str, float]:
            return {"left_knee": 0.0}

        r = driver.run_policy(policy, duration=60.0)
        assert r["status"] == "success"
        driver.cleanup()
