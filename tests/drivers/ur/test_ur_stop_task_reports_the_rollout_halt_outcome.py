"""A halt this driver reports as done is a halt the rollout thread has left.

:meth:`~strands_robots.drivers.ur.URDriver.stop_task` signals the rollout, waits
for its thread and then decelerates the arm with ``servoStop``.  The wait is
bounded, and a caller-supplied policy blocking on a remote inference call
outlasts any bound - so the interesting case is not the happy one.  Two things
were true of it:

* ``_Rollout.join`` was annotated ``-> None`` and swallowed the timeout, so the
  envelope reported ``stopped=True`` for a thread still inside the loop, while
  :meth:`~strands_robots.drivers.ur.URDriver.get_task_status` reported
  ``running=True`` in the same instant.  One driver, two envelopes, opposite
  answers about whether the arm is under a task.
* the loop read its stop event at the top of a step and not again after the
  policy returned, so the setpoint that policy call was computing went out
  *after* the ``servoStop`` - measured landing 0.5 ms later.  An arm an operator
  was told had stopped resumed moving.

:class:`~strands_robots.drivers.g1.G1Driver` and
:class:`~strands_robots.drivers.go2.Go2Driver` are the fleet's other two drivers
that roll a policy out on a thread, and both already report this: their loop's
``stop`` returns whether it joined, ``stop_task`` puts that in ``stopped`` and
returns an error envelope when it is false, and the G1 loop re-reads the stop
signal once more after the policy returns and before the frame publishes.  This
driver now holds the same two contracts, so the third rollout driver is graded on
the same relation rather than on its own weaker one.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from strands_robots.drivers.ur import JOINT_NAMES, URDriver, _Rollout
from tests.mocks.ur_rtde import MEASURED_Q, FakeRTDE, json_of, text_of

HOST = "192.168.1.10"

#: Tight enough that a blocked-policy cell costs milliseconds rather than the
#: 2 s production budget, which is sized for a real controller's period.
FAST_JOIN_S = 0.05


def _reachable_setpoint() -> dict[str, float]:
    """A setpoint one step off the measured pose, so the step gate admits it."""
    return {name: q + 0.001 for name, q in zip(JOINT_NAMES, MEASURED_Q, strict=True)}


def _fast_join(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the join budget without patching over the decision it feeds."""
    original = _Rollout.join

    def fast(self: _Rollout, timeout: float = FAST_JOIN_S) -> bool:
        return original(self, timeout=timeout)

    monkeypatch.setattr(_Rollout, "join", fast)


@pytest.fixture
def blocked_rollout(fake_rtde: FakeRTDE, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[URDriver, threading.Event]]:
    """A connected arm whose rollout is stuck inside one policy call.

    The policy returns a *reachable* setpoint once released, so a write that
    escapes the halt reaches ``servoJ`` rather than being turned away by the step
    gate - the cell would otherwise pass for the wrong reason.
    """
    driver = URDriver(tool_name="ur5e", port=HOST, control_frequency=200.0)
    assert driver.connect_eagerly() is None
    _fast_join(monkeypatch)

    entered = threading.Event()
    release = threading.Event()

    def policy(observation: dict[str, Any]) -> dict[str, float]:
        entered.set()
        release.wait(10.0)
        return _reachable_setpoint()

    assert driver.run_policy(policy, instruction="hold", duration=30.0)["status"] == "success"
    assert entered.wait(5.0), "the rollout never reached the policy"
    try:
        yield driver, release
    finally:
        release.set()
        driver.cleanup()


def _await_exit(rollout: _Rollout) -> None:
    """Wait for the rollout thread to leave the loop, polling its own flag.

    Deliberately not ``join`` - the cells below that grade ``join``'s verdict
    would otherwise share a helper with the ones that grade the setpoints, and a
    tree where the verdict is missing would fail them all for the same reason.
    """
    deadline = time.monotonic() + 5.0
    while rollout.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not rollout.is_running, "the rollout thread never exited"


def _drain(driver: URDriver, release: threading.Event) -> None:
    """Release the policy and wait for the rollout thread to leave the loop."""
    release.set()
    rollout = driver._rollout
    assert rollout is not None
    _await_exit(rollout)


class TestAHaltIsNotClaimedWhileTheThreadIsStillInTheLoop:
    """The envelope agrees with the driver's own task status."""

    def test_a_rollout_that_did_not_join_is_reported_as_not_stopped(
        self, blocked_rollout: tuple[URDriver, threading.Event]
    ) -> None:
        driver, _release = blocked_rollout
        envelope = driver.stop_task()
        assert envelope["status"] == "error"
        payload = json_of(envelope)
        assert payload["stopped"] is False
        assert payload["running"] is True
        assert "did not join" in str(payload["reason"])

    def test_the_two_envelopes_do_not_contradict_each_other(
        self, blocked_rollout: tuple[URDriver, threading.Event]
    ) -> None:
        driver, _release = blocked_rollout
        stopped = json_of(driver.stop_task())["stopped"]
        running = json_of(driver.get_task_status())["running"]
        assert stopped is not running, f"stop_task said stopped={stopped} while the task ran"


class TestNoSetpointOutlivesTheHalt:
    """The stop signal wins over a policy call that was already in flight."""

    def test_stop_task_is_the_last_thing_the_controller_hears(
        self, blocked_rollout: tuple[URDriver, threading.Event]
    ) -> None:
        driver, release = blocked_rollout
        control = driver._control
        assert control is not None
        driver.stop_task()
        assert control.servo_stops == 1
        commanded = len(control.servoj_calls)
        _drain(driver, release)
        assert len(control.servoj_calls) == commanded, "a setpoint reached the arm after the halt"

    def test_the_verdictless_stop_hook_holds_the_same_line(
        self, blocked_rollout: tuple[URDriver, threading.Event]
    ) -> None:
        # ``stop`` is annotated ``-> None`` so it cannot report anything; the
        # property has to hold by construction there rather than be reported.
        driver, release = blocked_rollout
        control = driver._control
        assert control is not None
        asyncio.run(driver.stop())
        commanded = len(control.servoj_calls)
        _drain(driver, release)
        assert len(control.servoj_calls) == commanded, "a setpoint reached the arm after stop()"

    def test_the_loop_records_the_caller_as_the_reason_it_exited(
        self, blocked_rollout: tuple[URDriver, threading.Event]
    ) -> None:
        driver, release = blocked_rollout
        driver.stop_task()
        _drain(driver, release)
        assert json_of(driver.get_task_status())["exit_reason"] == "stopped"


class TestTheJoinReportsWhatItObserved:
    """``_Rollout.join`` is the single source of the halt verdict."""

    def test_a_thread_still_in_the_loop_is_not_joined(self, blocked_rollout: tuple[URDriver, threading.Event]) -> None:
        driver, _release = blocked_rollout
        rollout = driver._rollout
        assert rollout is not None
        rollout.request_stop()
        assert rollout.join(timeout=FAST_JOIN_S) is False

    def test_a_thread_that_left_the_loop_is_joined(self, blocked_rollout: tuple[URDriver, threading.Event]) -> None:
        driver, release = blocked_rollout
        driver.stop_task()
        _drain(driver, release)
        assert driver._rollout is not None
        assert driver._rollout.join(timeout=FAST_JOIN_S) is True

    def test_a_rollout_that_never_started_is_trivially_joined(self) -> None:
        # ``cleanup`` joins without checking ``is_running``, and a rollout is
        # published to the driver just before its thread starts, so this is the
        # state that race can observe.
        rollout = _Rollout(
            driver=URDriver(tool_name="ur5e", port=HOST),
            policy=lambda observation: _reachable_setpoint(),
            instruction="",
            duration=1.0,
            n_steps=1,
            period=0.01,
        )
        assert rollout.join(timeout=FAST_JOIN_S) is True


class TestWhatTheHappyPathStillReports:
    """Recorded because they pass before and after: only the unjoined case moved."""

    def test_a_rollout_that_finished_its_budget_reports_a_clean_stop(self, fake_rtde: FakeRTDE) -> None:
        driver = URDriver(tool_name="ur5e", port=HOST, control_frequency=200.0)
        assert driver.connect_eagerly() is None
        assert driver.run_policy(lambda _obs: _reachable_setpoint(), n_steps=2)["status"] == "success"
        assert driver._rollout is not None
        _await_exit(driver._rollout)
        payload = json_of(driver.stop_task())
        assert payload["stopped"] is True
        assert payload["steps"] == 2

    def test_stopping_an_idle_arm_is_a_success(self, fake_rtde: FakeRTDE) -> None:
        driver = URDriver(tool_name="ur5e", port=HOST)
        assert driver.connect_eagerly() is None
        assert json_of(driver.stop_task()) == {"stopped": True, "steps": 0, "robot": "ur5e"}

    def test_stopping_a_disconnected_arm_is_still_refused(self) -> None:
        envelope = URDriver(tool_name="ur5e", port=HOST).stop_task()
        assert envelope["status"] == "error"
        assert "not connected" in text_of(envelope)


class TestTeardownDoesNotDisconnectUnderALiveWrite:
    """``cleanup`` carries no verdict, so the property holds by construction."""

    def test_a_late_write_finds_no_interface_rather_than_a_disconnected_one(
        self, blocked_rollout: tuple[URDriver, threading.Event]
    ) -> None:
        driver, release = blocked_rollout
        control = driver._control
        assert control is not None
        driver.cleanup()
        assert control.disconnected is True
        commanded = len(control.servoj_calls)
        release.set()
        rollout = driver._rollout
        assert rollout is not None
        _await_exit(rollout)
        assert len(control.servoj_calls) == commanded
        assert json_of(driver.get_task_status())["exit_reason"] == "stopped"
