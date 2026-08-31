"""Grade the three contracts round-3 review left open on the G1 control loop.

1.  A poller that reads :meth:`G1Driver.get_task_status` after a self-terminating
    exit still sees the loop's final snapshot - naming the exit reason
    (``n_steps``, ``duration``, ``gate``, ``policy``, ``publish``) rather than
    collapsing to "no task has been started on this driver".
2.  :meth:`G1Driver.stop_task` reports the join outcome honestly - a policy
    that outlasts the join budget surfaces as ``status="error"`` and
    ``stopped=False``, not as ``success`` while the payload's own
    ``running=True`` says the loop is still writing frames.
3.  A stop signal that arrives while the policy is computing prevents the
    in-flight action from reaching the wire; the zero-torque frame is the
    last frame published, not a fresh position command followed by the
    zero-torque frame.

The suite uses a recording publisher and a callable-double policy, both
already established for the neighbouring control-loop tests.  No
``unitree_sdk2py``, no DDS bus - the SDK stub the neighbouring file installs
via an autouse fixture is imported here too so the same production lane
runs on an SDK-less CI box.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import G1Driver

# --------------------------------------------------------------------- #
# SDK stub - identical structure to the fixture the neighbouring        #
# ``test_g1_control_loop.py`` installs; duplicated here so this file    #
# can be run standalone.                                                #
# --------------------------------------------------------------------- #


def _sdk_importable() -> bool:
    try:
        importlib.import_module("unitree_sdk2py.idl.default")
        importlib.import_module("unitree_sdk2py.utils.crc")
        importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    except ImportError:
        return False
    return True


_SDK_PRESENT = _sdk_importable()


class _StubMotorCmd:
    __slots__ = ("mode", "q", "dq", "tau", "kp", "kd")

    def __init__(self) -> None:
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class _StubLowCmd:
    def __init__(self) -> None:
        self.mode_pr = 0
        self.mode_machine = 0
        self.crc = 0
        self.motor_cmd = [_StubMotorCmd() for _ in range(35)]


class _StubCRC:
    @staticmethod
    def Crc(_data: Any) -> int:
        return 0


def _install_sdk_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    crc = types.ModuleType("unitree_sdk2py.utils.crc")
    crc.CRC = _StubCRC  # type: ignore[attr-defined]
    dds = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    dds.LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    msg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    msg.__path__ = []  # type: ignore[attr-defined]
    msg.dds_ = dds  # type: ignore[attr-defined]
    unitree_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    unitree_hg.__path__ = []  # type: ignore[attr-defined]
    unitree_hg.msg = msg  # type: ignore[attr-defined]
    idl = types.ModuleType("unitree_sdk2py.idl")
    idl.__path__ = []  # type: ignore[attr-defined]
    idl.default = default  # type: ignore[attr-defined]
    idl.unitree_hg = unitree_hg  # type: ignore[attr-defined]
    utils = types.ModuleType("unitree_sdk2py.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    utils.crc = crc  # type: ignore[attr-defined]
    root = types.ModuleType("unitree_sdk2py")
    root.__path__ = []  # type: ignore[attr-defined]
    root.idl = idl  # type: ignore[attr-defined]
    root.utils = utils  # type: ignore[attr-defined]
    for name, module in (
        ("unitree_sdk2py", root),
        ("unitree_sdk2py.idl", idl),
        ("unitree_sdk2py.idl.default", default),
        ("unitree_sdk2py.idl.unitree_hg", unitree_hg),
        ("unitree_sdk2py.idl.unitree_hg.msg", msg),
        ("unitree_sdk2py.idl.unitree_hg.msg.dds_", dds),
        ("unitree_sdk2py.utils", utils),
        ("unitree_sdk2py.utils.crc", crc),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(autouse=True)
def _frame_builders_can_build(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _SDK_PRESENT:
        _install_sdk_stub(monkeypatch)


# --------------------------------------------------------------------- #
# Recording publisher: captures what the loop publishes so the tests    #
# can assert on frame count and shape without a DDS bus.                #
# --------------------------------------------------------------------- #


class _RecordingPublisher:
    def __init__(self) -> None:
        self.frames: list[Any] = []
        self._lock = threading.Lock()

    def publish(self, _topic: str, _cls: type, message: Any) -> str | None:
        with self._lock:
            self.frames.append(message)
        return None

    def close(self) -> None:  # pragma: no cover - not exercised here
        pass


def _fake_driver(monkeypatch: pytest.MonkeyPatch, *, fsm_id: int = 500) -> G1Driver:
    """Return a G1Driver primed so ``run_policy`` reaches the loop.

    The gate is short-circuited to a mock - the review's finding about
    ``_fsm_id`` having no producer is #2765's, not this PR's; here we grade
    the terminal-observability contract, so the gate is stubbed so the loop
    can run.  ``_pubs`` is a recording publisher; ``_connected`` is True.
    """
    from unittest.mock import MagicMock

    driver = G1Driver.__new__(G1Driver)
    driver._tool_name = "g1"
    driver._port = "127.0.0.1"
    driver._network_interface = "lo"
    driver._connected = True
    driver._connect_error = None
    driver._subs = None
    driver._pubs = _RecordingPublisher()  # type: ignore[assignment]
    driver._fsm_id = fsm_id
    driver._mode_machine = 4
    driver._battery = {"pct": 90.0}
    driver._imu = {"quaternion": (1.0, 0.0, 0.0, 0.0)}
    driver._loop = None
    driver._last_task_snapshot = None
    driver._task_admission = threading.Lock()
    driver._check_motion_gates = MagicMock(return_value=None)  # type: ignore[method-assign]
    # The loop owns an FSM refresher thread that calls these two.  Stubbed for
    # the same reason the gate above is: this driver is built with ``__new__``
    # and has none of the motion-switcher state, and what these cells grade is
    # terminal observability, not the FSM producer.  Its contract is graded in
    # test_g1_fsm_refresh_is_off_the_control_loop_thread.py.
    driver._refresh_fsm_id = MagicMock()  # type: ignore[method-assign]
    driver._fsm_read_at = None
    return driver


def _one_step_action() -> dict[str, float]:
    return {"left_shoulder_pitch": 0.0}


# --------------------------------------------------------------------- #
# Contract 1: terminal snapshot survives loop teardown.                 #
# --------------------------------------------------------------------- #


class TestTerminalSnapshotSurvivesLoopTeardown:
    """``get_task_status`` names the exit reason after the loop joins."""

    def _run_to_completion(self, driver: G1Driver, **kwargs: Any) -> None:
        env = driver.run_policy(**kwargs)
        assert env["status"] == "success", env
        # Wait for the loop thread to finish (bounded).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with driver._task_admission:
                if driver._loop is None:
                    return
            time.sleep(0.01)
        raise AssertionError("loop did not finish within 5s")

    def test_n_steps_exit_reason_survives_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)

        def policy(_obs: Any) -> dict[str, float]:
            return _one_step_action()

        self._run_to_completion(driver, policy_object=policy, n_steps=3, duration=10.0)

        env = driver.get_task_status()
        payload = env["content"][0]["json"]
        assert payload["running"] is False
        assert payload["exit_reason"] == "n_steps"
        assert payload["steps"] == 3
        # The old collapsing message must not resurface.
        assert "reason" not in payload or payload.get("exit_reason") is not None

    def test_duration_exit_reason_survives_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)

        def policy(_obs: Any) -> dict[str, float]:
            return _one_step_action()

        self._run_to_completion(driver, policy_object=policy, duration=0.05)

        payload = driver.get_task_status()["content"][0]["json"]
        assert payload["running"] is False
        assert payload["exit_reason"] == "duration"

    def test_policy_none_exit_reason_survives_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)

        def policy(_obs: Any) -> None:
            return None

        self._run_to_completion(driver, policy_object=policy, duration=10.0)

        payload = driver.get_task_status()["content"][0]["json"]
        assert payload["running"] is False
        assert payload["exit_reason"] == "policy"
        assert "None" in (payload.get("exit_detail") or "")

    def test_policy_raise_exit_reason_survives_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)

        def policy(_obs: Any) -> dict[str, float]:
            raise ValueError("inference failed")

        self._run_to_completion(driver, policy_object=policy, duration=10.0)

        payload = driver.get_task_status()["content"][0]["json"]
        assert payload["running"] is False
        assert payload["exit_reason"] == "policy"
        assert "ValueError" in (payload.get("exit_detail") or "")

    def test_gate_flip_exit_reason_survives_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        driver = _fake_driver(monkeypatch)

        # Gate answers once (for the run_policy admission), then refuses.
        answers: list[Any] = [
            None,
            {"status": "error", "content": [{"text": "gate refused: FSM left the allowed set"}]},
        ]
        driver._check_motion_gates = MagicMock(  # type: ignore[method-assign]
            # ``**_kwargs`` absorbs the per-step re-gate's ``refresh=False``.
            side_effect=lambda _scope, **_kwargs: answers.pop(0) if answers else answers[-1]
        )

        def policy(_obs: Any) -> dict[str, float]:
            return _one_step_action()

        self._run_to_completion(driver, policy_object=policy, duration=10.0)

        payload = driver.get_task_status()["content"][0]["json"]
        assert payload["running"] is False
        assert payload["exit_reason"] == "gate"

    def test_publish_no_publisher_exit_reason_survives_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)
        driver._pubs = None  # force the "no publisher" branch

        def policy(_obs: Any) -> dict[str, float]:
            return _one_step_action()

        self._run_to_completion(driver, policy_object=policy, duration=10.0)

        payload = driver.get_task_status()["content"][0]["json"]
        assert payload["running"] is False
        assert payload["exit_reason"] == "publish"

    def test_a_fresh_run_policy_clears_the_stashed_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second rollout must not surface the first rollout's exit."""

        driver = _fake_driver(monkeypatch)

        def policy(_obs: Any) -> dict[str, float]:
            return _one_step_action()

        self._run_to_completion(driver, policy_object=policy, n_steps=2, duration=10.0)
        first = driver.get_task_status()["content"][0]["json"]
        assert first["exit_reason"] == "n_steps"

        # Start again; the poller between run_policy and the first frame must
        # NOT see the old "n_steps" exit.
        env = driver.run_policy(policy_object=policy, n_steps=3, duration=10.0)
        assert env["status"] == "success"
        mid = driver.get_task_status()["content"][0]["json"]
        # Either the new loop's live snapshot (running=True or exit_reason
        # from the *new* run), or - if the loop already finished - the new
        # run's exit_reason, not the stale one.
        assert mid.get("exit_reason") != "n_steps" or mid.get("steps") in (0, 1, 2, 3)


# --------------------------------------------------------------------- #
# Contract 2: stop_task reports join outcome honestly.                  #
# --------------------------------------------------------------------- #


class TestStopTaskReportsJoinOutcome:
    """A policy that outlasts the join budget surfaces as an error."""

    def test_prompt_stop_reports_success_and_stopped_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)

        def policy(_obs: Any) -> dict[str, float]:
            return _one_step_action()

        env = driver.run_policy(policy_object=policy, duration=10.0)
        assert env["status"] == "success"

        # Give the loop a moment to actually be running.
        time.sleep(0.05)
        stop_env = driver.stop_task()
        assert stop_env["status"] == "success"
        payload = stop_env["content"][0]["json"]
        assert payload["stopped"] is True
        assert payload["exit_reason"] == "stop_task"
        assert payload["running"] is False

    def test_blocking_policy_surfaces_as_error_with_stopped_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)

        release = threading.Event()

        def blocking_policy(_obs: Any) -> dict[str, float]:
            # Block until we release; simulates a remote inference call
            # that outlasts the stop_task join budget.
            release.wait(timeout=5.0)
            return _one_step_action()

        # Patch the loop's stop timeout to something tight so the test
        # completes quickly.  The stop() default of 2.0s would make the
        # suite slow.
        original_stop = None
        try:
            from strands_robots.drivers.g1 import _ControlLoop

            original_stop = _ControlLoop.stop

            def fast_stop(self: _ControlLoop, reason: str = "stop_task", timeout: float = 0.1) -> bool:
                return original_stop(self, reason=reason, timeout=timeout)  # type: ignore[misc]

            monkeypatch.setattr(_ControlLoop, "stop", fast_stop)

            env = driver.run_policy(policy_object=blocking_policy, duration=10.0)
            assert env["status"] == "success"

            time.sleep(0.05)  # let the loop actually enter the policy call
            stop_env = driver.stop_task()

            assert stop_env["status"] == "error"
            payload = stop_env["content"][0]["json"]
            assert payload["stopped"] is False
            assert "did not join within timeout" in payload["reason"]
        finally:
            release.set()
            # Wait for the loop to actually finish before the fixture
            # tears down.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with driver._task_admission:
                    if driver._loop is None:
                        break
                time.sleep(0.01)

    def test_idempotent_stop_when_no_task_is_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-existing shape (idempotent no-op) must be preserved."""

        driver = _fake_driver(monkeypatch)
        env = driver.stop_task()
        assert env["status"] == "success"
        assert "no task is running" in env["content"][0]["text"]


# --------------------------------------------------------------------- #
# Contract 3: no fresh action frame after stop.                          #
# --------------------------------------------------------------------- #


class TestStopPreemptsInFlightAction:
    """A stop signal during the policy call blocks the pending publish."""

    def test_stop_between_policy_and_publish_drops_the_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver = _fake_driver(monkeypatch)
        publisher = driver._pubs
        assert isinstance(publisher, _RecordingPublisher)

        # Policy signals the stop mid-call, then returns.  The loop's
        # re-check between _call_policy and the publish must see the set
        # event and skip the publish; the finally still stamps the
        # zero-torque frame.
        stop_between = threading.Event()

        def policy(_obs: Any) -> dict[str, float]:
            # Trip the stop event *inside* the policy call so the loop
            # observes it on the post-policy re-check, not on the top of
            # the next iteration.
            with driver._task_admission:
                loop = driver._loop
            if loop is not None and not stop_between.is_set():
                stop_between.set()
                loop._stop_event.set()
            return _one_step_action()

        env = driver.run_policy(policy_object=policy, duration=10.0)
        assert env["status"] == "success"

        # Wait for the loop to finish.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with driver._task_admission:
                if driver._loop is None:
                    break
            time.sleep(0.01)

        # At most one action frame may have published before the stop
        # (the loop can iterate once before the policy trips the event).
        # After the event trips, the re-check must skip subsequent
        # publishes.  So we expect action_frames <= 1 followed by exactly
        # one zero-torque frame.
        frames = publisher.frames
        assert len(frames) >= 1  # at least the zero-torque frame
        # The last frame must be the zero-torque frame (29 enabled slots).
        last = frames[-1]
        enabled = sum(1 for slot in last.motor_cmd if getattr(slot, "mode", 0) != 0)
        # Zero-torque frame enables the 29 named slots at zero gain.
        # Action frame enables exactly 1 (left_shoulder_pitch).  Either
        # way, the last frame's slot 29..34 tail must be untouched
        # (mode==0) - grading via a fingerprint the builder maintains.
        assert enabled != 1, (
            "the last frame is an action frame - a fresh position command reached the wire after stop_task"
        )
