"""The teleop loop refuses a rate or a horizon it cannot honor.

``teleoperate(hz=..., duration=...)`` consumes both knobs only inside the
control loop - ``1 / hz`` is the period, ``start + duration`` the deadline - and
that loop runs on a background thread. An unusable value therefore never failed
where the caller could see it:

* ``hz=0`` / ``-10`` / ``nan`` / ``inf`` left the period at ``0`` and spun the
  loop as fast as the host allowed, polling the leader (and, on a hardware
  ``Robot``, writing the servo bus) thousands of times per second while the call
  reported ``status="success"`` at "0Hz" / "nanHz".
* ``duration=0`` was read by truthiness, so the one value that most obviously
  means "stop now" meant "never stop".
* a non-numeric ``hz`` / ``duration`` raised on the loop thread, leaving a
  started-and-dead session behind.

The same rate knob reaches the mesh publish loop through
``Robot.start_teleop_publish`` and :class:`InputPublisher`, so all three entry
points share one domain: :func:`strands_robots.utils.positive_finite_number_error`.

All three are driven here, including the hardware entry point in the middle of
that chain. It is the only guard a direct caller passes through - the
``teleoperate(publish=True)`` tests reach the publisher through a stand-in host
whose ``start_teleop_publish`` validates nothing - and it sits ahead of the
teardown of any publisher already registered under that device name, so a
refused rate must leave a live stream alone rather than stopping it first.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from strands_robots.hardware_robot import Robot as HardwareRobot
from strands_robots.hardware_robot import RobotTaskState
from strands_robots.mesh.input import InputPublisher
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import positive_finite_number_error
from tests.test_teleop import FakeHost, FakePublishHost, FakeTeleop, _spin_until

#: Values no rate or time span can be built from. ``True`` is included because
#: it is an ``int`` subclass that would otherwise act as a silent 1.
UNUSABLE = [0, 0.0, -1, -10.5, float("nan"), float("inf"), float("-inf"), "30", None, [50], True]


def _attached(host: FakeHost) -> FakeTeleop:
    dev = FakeTeleop({"a.pos": 1.0})
    host.attach_teleop(dev, name="lead")
    return dev


class TestTeleoperateRefusesAnUnusableRate:
    """``hz`` is refused at the call, before any hardware is touched."""

    @pytest.mark.parametrize("hz", UNUSABLE)
    def test_unusable_hz_is_refused(self, hz):
        host = FakeHost()
        _attached(host)
        res = host.teleoperate(hz=hz)
        assert res["status"] == "error"
        assert "hz must be > 0" in res["content"][0]["text"]
        assert repr(hz) in res["content"][0]["text"]

    @pytest.mark.parametrize("hz", UNUSABLE)
    def test_a_refused_rate_starts_no_loop_and_touches_no_device(self, hz):
        """The guard runs before connect(), so a rejection has no side effects."""
        host = FakeHost()
        dev = _attached(host)
        host.teleoperate(hz=hz)
        assert dev.connect_calls == 0
        assert dev.get_action_calls == 0
        assert dev.is_connected is False
        assert host.sent == []
        assert host.get_teleoperate_status()["content"][1]["json"]["running"] is False

    @pytest.mark.parametrize("hz", [1, 50.0, 62.5, np.float32(200.0), np.int64(30)])
    def test_a_usable_rate_still_runs(self, hz):
        host = FakeHost()
        dev = _attached(host)
        res = host.teleoperate(hz=hz)
        try:
            assert res["status"] == "success"
            assert _spin_until(lambda: dev.get_action_calls > 0)
        finally:
            host.stop_teleoperate()


class TestTeleoperateRefusesAnUnusableDuration:
    """``duration`` is refused unless it is omitted, which means "run on"."""

    # ``None`` is dropped: for ``duration`` it is the documented "run until
    # stopped", not a mistake.
    @pytest.mark.parametrize("duration", [v for v in UNUSABLE if v is not None])
    def test_unusable_duration_is_refused(self, duration):
        host = FakeHost()
        _attached(host)
        res = host.teleoperate(duration=duration)
        assert res["status"] == "error"
        assert "duration must be > 0" in res["content"][0]["text"]

    def test_zero_duration_is_refused_not_read_as_absent(self):
        """``duration=0`` used to be falsy, so it meant "run forever"."""
        host = FakeHost()
        dev = _attached(host)
        res = host.teleoperate(duration=0.0)
        assert res["status"] == "error"
        assert "duration must be > 0" in res["content"][0]["text"]
        assert dev.connect_calls == 0

    def test_duration_none_still_runs_until_stopped(self):
        host = FakeHost()
        dev = _attached(host)
        res = host.teleoperate(hz=200, duration=None)
        try:
            assert res["status"] == "success"
            assert _spin_until(lambda: dev.get_action_calls > 2)
        finally:
            host.stop_teleoperate()

    def test_a_usable_duration_stops_the_loop_on_its_own(self):
        host = FakeHost()
        dev = _attached(host)
        res = host.teleoperate(hz=200, duration=0.05, block=True)
        assert res["status"] == "success"
        assert dev.get_action_calls >= 1
        assert host.get_teleoperate_status()["content"][1]["json"]["running"] is False


class TestARefusedRateNeverReachesTheMeshPublisher:
    """``publish=True`` forwards ``hz`` on, so the guard must precede it."""

    def test_refused_rate_starts_no_publisher(self):
        host = FakePublishHost()
        dev = _attached(host)
        res = host.teleoperate(hz=0, publish=True)
        assert res["status"] == "error"
        assert host.publish_calls == []
        assert dev.is_connected is False

    @pytest.mark.parametrize("hz", [0, -5, float("nan"), float("inf"), "30", True, None])
    def test_input_publisher_refuses_it_at_construction(self, hz):
        """The publish loop divides by hz on a background thread."""
        with pytest.raises(ValueError, match="hz must be > 0"):
            InputPublisher(mesh=object(), teleoperator=object(), hz=hz)

    def test_input_publisher_accepts_a_usable_rate(self):
        pub = InputPublisher(mesh=object(), teleoperator=object(), hz=np.float32(12.5))
        assert float(pub.hz) == pytest.approx(12.5)


class TestOneDomainForEveryRateAndDurationKnob:
    """The shared helper and the sim's rollout knobs cannot diverge.

    The parity is over the VALUE domain: which durations and rates are usable at
    all. ``_validate_duration`` then applies one further condition the shared
    helper cannot express, because it is not a property of the value - a
    duration shorter than one control period resolves to zero steps at that
    rate. Every value below is checked at a rate that can execute it, so this
    class pins the shared domain alone; the rate-dependent refusal is pinned in
    ``tests/simulation/test_rollout_duration_must_produce_a_control_step.py``.
    """

    #: A rate at which every usable duration below is many control steps, so
    #: the horizon condition cannot mask a value-domain disagreement.
    RATE = 50.0

    @pytest.mark.parametrize("value", UNUSABLE + [1, 50.0, 62.5, np.float32(2.5)])
    def test_the_sim_rollout_knobs_agree_with_the_shared_domain(self, value):
        shared_rejects = positive_finite_number_error(value, "x", "m") is not None
        assert (SimEngine._validate_duration(value, "run_policy", self.RATE) is not None) is shared_rejects
        assert (SimEngine._validate_positive_frequency(value, "run_policy") is not None) is shared_rejects

    def test_the_sim_rollout_messages_are_unchanged(self):
        """Callers (and tests) pin this exact text."""
        assert SimEngine._validate_duration(0, "run_policy", self.RATE)["content"][0]["text"] == (
            "run_policy: duration must be > 0, got 0."
        )
        assert SimEngine._validate_positive_frequency(math.nan, "eval_policy")["content"][0]["text"] == (
            "eval_policy: control_frequency must be > 0, got nan."
        )

    def test_a_numpy_rate_is_usable_but_a_bool_is_not(self):
        assert positive_finite_number_error(np.float32(50.0), "hz", "teleoperate") is None
        assert positive_finite_number_error(True, "hz", "teleoperate") == ("teleoperate: hz must be > 0, got True.")


class _FakeMesh:
    """Minimal live mesh: a peer_id plus a publish that records payloads."""

    def __init__(self, peer_id: str = "leader-1") -> None:
        self.peer_id = peer_id
        self.alive = True
        self.published: list[tuple[str, Any]] = []

    def publish(self, topic: str, payload: Any) -> None:
        self.published.append((topic, payload))


class _LivePublisher:
    """Stand-in for a running publisher that records being stopped."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {}


def _hardware_robot() -> Any:
    """A hardware ``Robot`` carrying only the teleop state the guard reads."""
    hw = HardwareRobot.__new__(HardwareRobot)
    hw.tool_name_str = "rate_guard_arm"
    hw.mesh = _FakeMesh()
    hw.peer_id = "leader-1"
    hw.robot = object()
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rate_guard")
    hw._shutdown_event = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    return hw


class TestTheHardwarePublishEntryPointSharesTheRateDomain:
    """``Robot.start_teleop_publish`` refuses a rate its publish loop cannot honor.

    The third entry point named in this module's docstring, driven directly. The
    ``teleoperate(publish=True)`` tests above reach the mesh publisher through
    :class:`~tests.test_teleop.FakePublishHost`, whose ``start_teleop_publish``
    records the call and returns success without validating anything, so the real
    method's own refusal never ran. That refusal is the only one a direct caller
    passes through: the mixin validates ``hz`` before forwarding it, and
    :class:`InputPublisher` raises rather than reporting, which a caller holding
    an agent-tool envelope cannot use.
    """

    @pytest.mark.parametrize("hz", UNUSABLE)
    def test_unusable_rate_is_refused_in_the_shared_domains_words(self, hz):
        """One rule, one wording - the entry point adds only its own name."""
        hw = _hardware_robot()
        result = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), hz=hz)
        assert result["status"] == "error"
        assert result["content"][0]["text"] == positive_finite_number_error(hz, "hz", "start_teleop_publish")

    @pytest.mark.parametrize("hz", [0, float("nan"), True])
    def test_a_refused_rate_registers_no_publisher(self, hz):
        """Nothing is constructed, so no loop thread divides by the bad rate."""
        hw = _hardware_robot()
        assert hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), hz=hz)["status"] == "error"
        assert getattr(hw, "_input_publishers", {}) == {}
        assert hw.mesh.published == []

    def test_a_refused_rate_leaves_a_live_publisher_running(self):
        """The guard precedes the teardown, so a rejected call loses no stream."""
        hw = _hardware_robot()
        live = _LivePublisher()
        hw._input_publishers = {"leader": live}
        result = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), device_name="leader", hz=0)
        assert result["status"] == "error"
        assert live.stopped is False
        assert hw._input_publishers == {"leader": live}

    def test_a_usable_rate_replaces_the_live_publisher(self):
        """The mirror: the teardown the guard precedes does happen when accepted."""
        hw = _hardware_robot()
        live = _LivePublisher()
        hw._input_publishers = {"leader": live}
        result = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), device_name="leader", hz=50.0)
        try:
            assert result["status"] == "success"
            assert live.stopped is True
            assert hw._input_publishers["leader"] is not live
            assert hw._input_publishers["leader"].hz == 50.0
        finally:
            hw._input_publishers["leader"].stop()

    @pytest.mark.parametrize("hz", UNUSABLE + [1, 50.0, np.float32(2.5)])
    def test_the_entry_point_and_the_constructor_agree(self, hz):
        """Neither surface may accept a rate the other refuses."""
        hw = _hardware_robot()
        result = hw.start_teleop_publish(teleoperator=FakeTeleop({"a.pos": 1.0}), hz=hz)
        entry_refused = result["status"] == "error"
        try:
            InputPublisher(mesh=_FakeMesh(), teleoperator=object(), hz=hz)
        except ValueError:
            constructor_refused = True
        else:
            constructor_refused = False
        assert entry_refused is constructor_refused
        if not entry_refused:
            hw._input_publishers["leader"].stop()
