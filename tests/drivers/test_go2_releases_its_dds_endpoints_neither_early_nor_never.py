"""The Go2 driver releases a DDS endpoint once, and only once it is safe to.

Two exits decide how long a Go2 endpoint lives, and each fails in the opposite
direction.

**Bring-up releases too late.**
:meth:`~strands_robots.drivers.go2.Go2Driver.connect_eagerly` builds a
:class:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet`, subscribes the
two ``unitree_go`` topics through it, then starts a publisher, and records the
set on ``self._subs`` only once all of that has succeeded. On every exit before
that, ``self._subs`` is still ``None`` - and
:meth:`~strands_robots.drivers.go2.Go2Driver.cleanup` closes ``self._subs`` and
nothing else, so a ladder that gave up part way and kept its local would leave
the topics it had already subscribed with no reference the driver can reach. That
is not inert: :meth:`DDSSubscriberSet.subscribe` builds every subscriber with a
non-zero queue length, and above zero ``unitree_sdk2py`` starts a ``ch_reader``
daemon thread whose target is a bound method of the channel's reader, so the
reader stays matched and its decoder keeps filling ``_joints`` and ``_battery``
for a driver whose ``_connected`` is ``False``. ``connect_eagerly`` documents a
retry as the supported response to a failure, so the next attempt would subscribe
the same topics again - the duplicate subscription the idempotence guard on a
*connected* driver exists to prevent, reached through a failed one instead.

**Teardown releases too early.**
:meth:`~strands_robots.drivers.go2._ControlLoop.stop` signals the loop and joins
its thread within a budget, and returns *whether the thread joined*. A
caller-supplied policy that outlasts that budget - a remote inference call is the
ordinary case - leaves the loop running and the answer ``False``. Both teardown
paths read it, and for ``cleanup`` that is load-bearing: releasing ``_pubs``
under a live loop thread makes
:meth:`~strands_robots.drivers.go2._ControlLoop._emit_zero_torque` take its
``pubs is None`` branch, so the zero-torque shutdown frame is never published -
not by ``cleanup``, and not later when the policy finally returns. A quadruped
standing on twelve position-controlled joints is exactly what that frame exists
for, so the cells that grade it park a policy past the join budget and then check
the wire.

No cell here needs a Go2, a DDS bus or ``unitree_sdk2py``: the recording
subscriber the sibling engine suite already uses stands in for the SDK channel,
and the publisher is the recorder with :class:`DDSPublisher`'s acceptance
contract that the Go2 write-path suite already drives. Both are registered on
:mod:`sys.modules`, so the production lane under test is the one hardware runs.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import strands_robots.drivers.go2 as go2_module
import strands_robots.tools.g1._dds_engine as engine_module
from strands_robots.drivers.go2 import GO2_JOINT_INDEX, Go2Driver
from tests.drivers.test_g1_dds_engine_release import _install_channel, _RecordingEndpoint
from tests.drivers.test_go2_driver import (
    _RecordingPublisher,
    _released_driver,
    stub_unitree_sdk,  # noqa: F401 - re-exported so pytest resolves it as a fixture here
)

#: The Go2 driver's logger, whose ERROR records say which endpoint was kept.
_DRIVER_LOGGER = "strands_robots.drivers.go2"

#: Longer than :meth:`_ControlLoop.stop`'s join budget, so a policy parked on
#: this event is guaranteed to outlast it.
_LONGER_THAN_THE_JOIN_BUDGET = 30.0


class _RefusingEndpoint(_RecordingEndpoint):
    """An endpoint whose ``Init`` fails for one topic, the way the SDK's can."""

    refuse_topic: str = ""

    def Init(self, handler: Any = None, queue_len: int | None = None) -> None:  # noqa: N802 - SDK spelling
        if self.topic == type(self).refuse_topic:
            raise RuntimeError("dds reader could not be created")
        super().Init(handler, queue_len)


def _topics() -> tuple[str, ...]:
    """The topics the driver's own subscription plan names, in order."""
    return tuple(topic for topic, _cls, _decoder in Go2Driver()._subscription_plan())


def _bring_up(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refuse_topic: str | None = None,
    resolve_fails_on_call: int | None = None,
    dds_errors: tuple[str | None, ...] = (),
    attempts: int = 1,
) -> tuple[Go2Driver, list[_RecordingEndpoint], str | None]:
    """Run ``connect_eagerly`` against recording subscribers and report what it built.

    Args:
        monkeypatch: Pytest's patcher, used for every seam so a machine that
            really has the SDK gets it back.
        refuse_topic: A topic whose subscriber construction fails, driving the
            ``subscribe`` exit.
        resolve_fails_on_call: Make the Nth ``_resolve_message_class`` call
            return a reason, driving the IDL-resolution exit.
        dds_errors: Answers for successive ``ensure_dds`` calls, of which the
            first is the subscriber set's ``start`` and the second the
            publisher's. ``(None, "...")`` is therefore the exit where the
            readers are up and the writer failed.
        attempts: Call ``connect_eagerly`` this many times; two models both the
            retry the docstring invites and the idempotence guard.

    Returns:
        The driver, every endpoint ``subscribe`` built across all attempts, and
        the last returned reason.
    """
    endpoint_class: type[_RecordingEndpoint] = _RecordingEndpoint
    if refuse_topic is not None:
        endpoint_class = type("_Refusing", (_RefusingEndpoint,), {"refuse_topic": refuse_topic})
    built = _install_channel(monkeypatch, "ChannelSubscriber", endpoint_class)

    pending = list(dds_errors)
    monkeypatch.setattr(engine_module, "ensure_dds", lambda _interface: pending.pop(0) if pending else None)

    # Stand in for the IDL lookup unconditionally. The real one imports the SDK's
    # generated ``unitree_go`` modules, which an ordinary runner does not have,
    # so resolving for real would fail the first topic and every cell below would
    # grade a ladder that never got started.
    calls = {"n": 0}

    def _resolve(class_path: tuple[str, str]) -> Any:
        calls["n"] += 1
        if resolve_fails_on_call is not None and calls["n"] == resolve_fails_on_call:
            return "transient: IDL module not loaded yet"
        return type(class_path[1], (), {})

    monkeypatch.setattr(go2_module, "_resolve_message_class", _resolve)

    driver = Go2Driver(network_interface="eth0")
    reason: str | None = None
    for _attempt in range(attempts):
        reason = driver.connect_eagerly()
    return driver, built, reason


def _open(endpoints: list[_RecordingEndpoint]) -> list[_RecordingEndpoint]:
    """The endpoints nothing has closed."""
    return [endpoint for endpoint in endpoints if endpoint.closes == 0]


class _RecordingSubscriberSet:
    """Stands in for the recorded ``_subs`` handle, counting closes.

    Teardown reads ``self._subs`` and calls ``close()`` on it; that is the whole
    contract these cells need from it, and building a real set here would say
    nothing extra about ``cleanup``.
    """

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def blocked_rollout(
    stub_unitree_sdk: None,  # noqa: F811 - the imported fixture, requested by name
) -> Iterator[tuple[Go2Driver, _RecordingPublisher, _RecordingSubscriberSet, threading.Event]]:
    """A driver whose rollout is parked inside its policy, past the join budget.

    The policy blocks on an event rather than sleeping, so the loop is provably
    still inside the step when a teardown runs and the release is exact rather
    than timed. Teardown of the fixture releases it either way, so no cell can
    leave a daemon thread parked for thirty seconds.

    Yields:
        The driver, its publisher recorder, its subscriber-set recorder, and the
        event that lets the parked policy return.
    """
    del stub_unitree_sdk
    driver, pub = _released_driver()
    subs = _RecordingSubscriberSet()
    driver._subs = subs  # type: ignore[assignment]
    entered = threading.Event()
    release = threading.Event()

    def policy(_state: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        release.wait(_LONGER_THAN_THE_JOIN_BUDGET)
        return {"FL_thigh_joint": 0.0}

    started = driver.run_policy(policy, n_steps=1000, duration=_LONGER_THAN_THE_JOIN_BUDGET)
    assert started["status"] == "success", started
    assert entered.wait(5.0), "the rollout never reached the policy"
    try:
        yield driver, pub, subs, release
    finally:
        release.set()
        thread = getattr(driver._loop, "_thread", None)
        if thread is not None:
            thread.join(timeout=5.0)


class TestABringUpThatGivesUpReleasesWhatItBuilt:
    """Every failure exit past the subscriber set closes it."""

    def test_a_refused_subscriber_does_not_leave_the_earlier_topic_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        topics = _topics()
        driver, built, reason = _bring_up(monkeypatch, refuse_topic=topics[-1])
        assert reason is not None and topics[-1] in reason
        assert len(built) == len(topics), "the ladder should have reached the last topic"
        assert _open(built) == [], "subscribers from the topics that did succeed were left open"
        assert driver._subs is None
        assert driver._connect_error == reason
        assert driver._connected is False

    def test_an_unresolvable_idl_class_does_not_leave_the_earlier_topic_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, built, reason = _bring_up(monkeypatch, resolve_fails_on_call=len(_topics()))
        assert reason == "transient: IDL module not loaded yet"
        assert len(built) == len(_topics()) - 1, "the ladder should have subscribed the earlier topics"
        assert _open(built) == [], "subscribers from the topics that did succeed were left open"
        assert driver._subs is None
        assert driver._connect_error == reason
        assert driver._connected is False

    def test_a_publisher_that_will_not_start_rolls_the_readers_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Readers up, writer down: one connect failure rather than a half-open driver."""
        driver, built, reason = _bring_up(monkeypatch, dds_errors=(None, "dds init failed: no such interface"))
        assert reason == "dds init failed: no such interface"
        assert len(built) == len(_topics()), "every topic subscribes before the publisher is started"
        assert _open(built) == [], "the readers were left running after the writer failed"
        assert driver._subs is None and driver._pubs is None
        assert driver._connect_error == reason
        assert driver._connected is False

    def test_a_dds_init_failure_reports_before_anything_is_subscribed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, built, reason = _bring_up(monkeypatch, dds_errors=("dds init failed: no such interface",))
        assert reason == "dds init failed: no such interface"
        assert built == [], "nothing should be subscribed once start() has failed"
        assert driver._connect_error == reason

    def test_the_retry_the_docstring_invites_does_not_duplicate_a_subscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, built, reason = _bring_up(monkeypatch, resolve_fails_on_call=len(_topics()), attempts=2)
        assert reason is None, "the second attempt should have connected"
        assert driver._connected is True
        assert [endpoint.topic for endpoint in _open(built)] == list(_topics())

    def test_a_second_connect_on_a_connected_driver_subscribes_nothing_new(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The idempotence guard: a no-op success, not a rebuilt set that orphans the old one."""
        driver, built, reason = _bring_up(monkeypatch, attempts=2)
        assert reason is None
        assert len(built) == len(_topics())
        assert _open(built) == built, "a successful connect closed its own subscribers"
        assert driver._subs is not None

    def test_cleanup_closes_the_set_a_successful_connect_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, built, _reason = _bring_up(monkeypatch)
        driver.cleanup()
        assert _open(built) == [], "a subscriber outlived cleanup()"
        assert driver._subs is None
        assert driver._connected is False


class TestTeardownWaitsForTheLoopBeforeReleasingTheWire:
    """A loop that outlasted the join budget keeps the publisher it still writes to."""

    def test_stop_task_reports_a_loop_that_did_not_join(
        self, blocked_rollout: tuple[Go2Driver, _RecordingPublisher, _RecordingSubscriberSet, threading.Event]
    ) -> None:
        """An error envelope whose payload agrees with it: nothing claims a stop that has not happened."""
        driver, _pub, _subs, _release = blocked_rollout
        result = driver.stop_task()
        assert result["status"] == "error"
        payload = result["content"][0]["json"]
        assert payload["stopped"] is False
        assert payload["running"] is True
        assert "did not join" in str(payload["reason"])

    def test_cleanup_keeps_the_publisher_the_live_loop_still_needs(
        self,
        blocked_rollout: tuple[Go2Driver, _RecordingPublisher, _RecordingSubscriberSet, threading.Event],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The headline: the zero-torque frame reaches the wire *after* cleanup returned.

        Releasing ``_pubs`` here is the fall ``cleanup``'s docstring says the
        path exists to prevent - the loop's own soft stop would be dropped by the
        teardown meant to make it safe.
        """
        driver, pub, subs, release = blocked_rollout
        with caplog.at_level(logging.ERROR, logger=_DRIVER_LOGGER):
            driver.cleanup()

        assert driver._pubs is pub, "cleanup released the publisher under a live loop"
        assert pub.close_calls == 0
        assert "keeping the publisher" in caplog.text
        # The set is closed either way: nothing writes through a subscriber.
        assert subs.close_calls == 1
        assert driver._subs is None
        assert driver._connected is False
        assert pub.writes == [], "the parked policy published nothing before teardown"

        release.set()
        driver._loop._thread.join(timeout=5.0)  # type: ignore[union-attr]
        assert len(pub.writes) == 1, "the soft-stop frame did not reach the retained publisher"
        topic, _cls, final = pub.writes[-1]
        assert topic == "rt/lowcmd"
        for slot in GO2_JOINT_INDEX.values():
            assert final.motor_cmd[slot].kp == pytest.approx(0.0)

    def test_stop_reports_the_loop_that_outlasted_the_budget(
        self,
        blocked_rollout: tuple[Go2Driver, _RecordingPublisher, _RecordingSubscriberSet, threading.Event],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``stop`` carries no envelope, so the log is the only place it can say so."""
        driver, pub, _subs, _release = blocked_rollout
        with caplog.at_level(logging.ERROR, logger=_DRIVER_LOGGER):
            asyncio.run(driver.stop())
        assert "still holds the wire" in caplog.text
        assert driver._pubs is pub, "stop() must leave the publisher open for the soft stop"
        assert pub.close_calls == 0

    def test_a_loop_that_did_join_gives_its_endpoints_up(self, stub_unitree_sdk: None) -> None:  # noqa: F811
        """The other half of the same rule, and cleanup stays idempotent after it."""
        del stub_unitree_sdk
        driver, pub = _released_driver()
        subs = _RecordingSubscriberSet()
        driver._subs = subs  # type: ignore[assignment]
        assert driver.run_policy(lambda _state: {"FL_thigh_joint": 0.0}, n_steps=1)["status"] == "success"
        driver._loop._thread.join(timeout=5.0)  # type: ignore[union-attr]

        driver.cleanup()
        assert driver._pubs is None and pub.close_calls == 1
        assert driver._subs is None and subs.close_calls == 1

        driver.cleanup()
        assert pub.close_calls == 1, "a second cleanup closed the publisher twice"
        assert subs.close_calls == 1
