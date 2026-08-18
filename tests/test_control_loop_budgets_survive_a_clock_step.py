# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A hardware control loop's time budget is honored across a wall-clock step.

Two loops command physical actuators for a caller-supplied number of seconds:
``Robot._execute_task_async`` (a policy rollout, bounded by ``duration``) and
``TeleopMixin._teleop_loop`` (a teleoperation session, bounded by the same
parameter). Both budgets used to be measured against ``time.time()``, which is
not a clock but the current opinion about the date: an NTP correction, a
``date -s`` or a resume from suspend moves it by an arbitrary amount, and both
loops moved with it.

The harm is not symmetric with the step's sign, and neither direction was
reported. Driving the real loops with a stub driver, a 1.0s budget and a clock
that takes one step mid-rollout::

    rollout budget 1.0s          seconds commanding the arm   actions   reported
    no step (control)                          0.980               49     1.001
    wall clock steps +30s                      0.000                1    30.021
    wall clock steps +1h                       0.000                1  3600.021
    wall clock steps -2s                       2.990              147     1.010

    teleop budget 1.0s        seconds driving the follower   frames   reported
    no step (control)                          1.005              50      1.005
    wall clock steps +30s                      0.040               2     30.040
    wall clock steps -2s                       3.016             150      1.016

Every row reported ``status="success"``.

Forward, the rollout is abandoned after its first action and the arm is left
parked at that pose mid-task, while the record claims the task ran for the size
of the step. Backward, the loop keeps commanding the servo bus for the budget
*plus* the step - three times the authorized window here - and the reported
duration under-reports it by exactly the step, so the overrun leaves no trace.

That budget is the only thing bounding a rollout on hardware.
``Robot._duration_error`` exists because a value the comparison cannot honor
means "the loop commands the servo bus indefinitely", and it validates the
value; a clock step defeats the same guarantee from outside the value's domain.

These tests pin the contract on behavior rather than on which clock is called:
each drives a real loop through a clock double that takes a known step, and
asserts the loop spent the budget it was given, applied a rollout's worth of
commands, and reported the time that actually elapsed. The safety subsystem
settled the same boundary three times (``tests/mesh/test_replay_cache_monotonic.py``,
``tests/mesh/test_bridge_dedup.py::TestDedupClock``,
``tests/mesh/test_corroboration_clock_domain.py``): a duration is local
bookkeeping and belongs on a monotonic clock, while an absolute stamp a reader
correlates with other logs stays on the wall clock.

No serial/USB hardware is touched: the driver and the leader are in-memory
fakes and the policy is a structural stub.
"""

from __future__ import annotations

import importlib
import threading
import time as real_time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from strands_robots import hardware_robot as hardware_robot_module
from strands_robots import teleop_mixin as teleop_mixin_module
from strands_robots.hardware_robot import Robot as HardwareRobot
from strands_robots.hardware_robot import RobotTaskState, TaskStatus
from tests.test_hardware_control_loop_rate_guard import _FakeArm
from tests.test_teleop import FakeHost, FakeTeleop

# The teleop loop lazily imports strands_robots.mesh.security on its first tick.
# That import is heavy, and paying it inside a short session would eat the very
# budget under test, so it is resolved here instead - which is what a process
# that has already teleoperated once looks like.
importlib.import_module("strands_robots.mesh.security")

#: Rollout/session budget every test hands the loop, in seconds. Short enough
#: to keep the suite quick, long enough that a loop honoring it applies many
#: commands rather than a handful.
BUDGET = 0.4

#: Command period installed on the loops, so ``BUDGET`` covers ~80 commands.
PERIOD = 0.005

#: Steps a wall clock really takes: a routine NTP correction, an hour of clock
#: skew or a timezone-shaped jump, and a backward correction.
FORWARD_STEPS = [30.0, 3600.0]
BACKWARD_STEPS = [-2.0]


class SteppingWallClock:
    """A ``time`` stand-in whose ``time()`` takes one step, mid-wait.

    Every other member delegates to the real module, so a loop that measures a
    duration on ``monotonic()`` sees an untouched clock and a loop that
    measures it on ``time()`` sees the step. ``reads`` counts ``time()`` calls
    and ``stepped`` records whether the step was ever handed out - a loop on
    the monotonic clock never reads ``time()`` at all, so neither is used as a
    precondition for the behavior under test.
    """

    def __init__(self, step_by: float, after_reads: int = 3, after_seconds: float = 0.0) -> None:
        self.step_by = step_by
        #: The step is withheld until this many reads have happened, so it
        #: always lands after the loop has taken the base it will subtract.
        #: A step that moved the base too would shift both sides of every
        #: comparison equally and change nothing.
        self.after_reads = after_reads
        #: ...and until this long into the run, for a comparison whose base is
        #: rewritten as it goes (the telemetry throttle): the step has to land
        #: between two of its readings to be visible at all.
        self.after_seconds = after_seconds
        self.reads = 0
        self.stepped = False
        self._created = real_time.monotonic()

    def time(self) -> float:
        self.reads += 1
        due = self.reads > self.after_reads and real_time.monotonic() - self._created >= self.after_seconds
        if due and self.step_by:
            self.stepped = True
            return real_time.time() + self.step_by
        return real_time.time()

    def monotonic(self) -> float:
        return real_time.monotonic()

    def perf_counter(self) -> float:
        return real_time.perf_counter()

    def sleep(self, seconds: float) -> None:
        real_time.sleep(seconds)


class _RecordingArm(_FakeArm):
    """A fake arm that records when each command reached it."""

    def __init__(self) -> None:
        super().__init__()
        self.command_times: list[float] = []

    def send_action(self, action: dict[str, Any]) -> None:
        self.command_times.append(real_time.monotonic())
        super().send_action(action)


class _Policy:
    """Structural stand-in for a policy: the members the loop reads."""

    supports_rtc = False
    execution_horizon = 1

    def set_control_frequency(self, hz: float) -> None:
        return None

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        return None

    def reset(self, seed: int | None = None) -> None:
        return None

    async def get_actions(self, observation: Any, instruction: str) -> list[dict[str, Any]]:
        return [{"j0.pos": 0.1}]


class _RecordingMesh:
    """Records every ``publish_step`` the control loop's throttle lets through."""

    def __init__(self) -> None:
        self.publish_times: list[float] = []

    def publish_step(self, step: int, observation: Any, action: Any, **kwargs: Any) -> None:
        self.publish_times.append(real_time.monotonic())

    def stop(self) -> None:
        return None


def _hardware_robot(mesh: Any = None, stream_min_period: float = 0.05) -> HardwareRobot:
    """A ``Robot`` wired to an in-memory arm, with the connect path stubbed."""
    robot = HardwareRobot.__new__(HardwareRobot)
    robot.tool_name_str = "test_arm"
    robot.action_horizon = 1
    robot.data_config = None
    robot.control_frequency = 1.0 / PERIOD
    robot.action_sleep_time = PERIOD
    robot._task_state = RobotTaskState()
    robot._executor = ThreadPoolExecutor(max_workers=1)
    robot._shutdown_event = threading.Event()
    robot._stop_requested = threading.Event()
    robot._task_admission = threading.Lock()
    robot._task_claimed = False
    robot.mesh = mesh
    robot.peer_id = "probe" if mesh is not None else None
    robot.robot = _RecordingArm()
    robot._last_stream_pub = float("-inf")
    robot._stream_min_period = stream_min_period

    async def _connected() -> tuple[bool, str]:
        return (True, "")

    async def _ready() -> bool:
        return True

    def _init_policy(policy: Any) -> Any:
        return _ready()

    def _no_telemetry(observation: dict[str, Any], *, skip_images: bool = False) -> None:
        return None

    robot._connect_robot = _connected  # type: ignore[method-assign]
    robot._initialize_policy = _init_policy  # type: ignore[method-assign]
    robot._publish_ros_telemetry = _no_telemetry  # type: ignore[method-assign]
    return robot


def _run_rollout(
    monkeypatch: pytest.MonkeyPatch,
    step_by: float,
    *,
    mesh: Any = None,
    stream_min_period: float = 0.05,
    step_after_seconds: float = 0.0,
) -> dict[str, Any]:
    """Drive one real rollout with ``BUDGET`` seconds and a stepping clock."""
    clock = SteppingWallClock(step_by, after_seconds=step_after_seconds)
    monkeypatch.setattr(hardware_robot_module, "time", clock)
    robot = _hardware_robot(mesh=mesh, stream_min_period=stream_min_period)
    try:
        started = real_time.monotonic()
        result = robot.run_policy(policy_object=cast(Any, _Policy()), instruction="probe", duration=BUDGET)
        wall_span = real_time.monotonic() - started
        times = robot.robot.command_times
        return {
            "status": result["status"],
            "commanded_span": (times[-1] - times[0]) if len(times) > 1 else 0.0,
            "commands": len(times),
            "reported_duration": robot._task_state.duration,
            "call_span": wall_span,
            "clock": clock,
        }
    finally:
        robot._shutdown_event.set()
        robot._task_state.status = TaskStatus.STOPPED
        robot._executor.shutdown(wait=False)


def _run_teleop_session(monkeypatch: pytest.MonkeyPatch, step_by: float) -> dict[str, Any]:
    """Drive one real teleop session with ``BUDGET`` seconds and a stepping clock."""
    clock = SteppingWallClock(step_by)
    monkeypatch.setattr(teleop_mixin_module, "time", clock)
    host = FakeHost()
    leader = FakeTeleop({"a.pos": 1.0})
    host.attach_teleop(leader, name="lead")
    try:
        started = real_time.monotonic()
        result = host.teleoperate(hz=1.0 / PERIOD, duration=BUDGET, block=True)
        wall_span = real_time.monotonic() - started
        telemetry = [block["json"] for block in result["content"] if "json" in block][0]
        return {
            "status": result["status"],
            "frames": len(host.sent),
            "reported_elapsed": telemetry["elapsed_s"],
            "call_span": wall_span,
            "clock": clock,
        }
    finally:
        host.stop_teleoperate()


class TestTheClockDoubleIsFaithful:
    """The double really moves the wall clock, so a clean run means something."""

    @pytest.mark.parametrize("step_by", [*FORWARD_STEPS, *BACKWARD_STEPS])
    def test_the_double_hands_out_the_step_it_advertises(self, step_by: float):
        clock = SteppingWallClock(step_by, after_reads=1)

        real_before = real_time.time()
        reported_before = clock.time()
        reported_after = clock.time()
        real_after = real_time.time()

        assert clock.stepped is True
        # Relative: what the double added beyond the real clock's own advance
        # is the step, whatever the real clock did meanwhile.
        drift = (reported_after - reported_before) - (real_after - real_before)
        assert drift == pytest.approx(step_by, abs=0.05)

    def test_a_zero_step_never_moves_the_clock(self):
        clock = SteppingWallClock(0.0, after_reads=1)
        for _ in range(5):
            clock.time()
        assert clock.stepped is False


class TestTheRolloutSpendsTheBudgetItWasGiven:
    """``duration`` bounds the rollout regardless of what the date does."""

    def test_no_step_is_the_control(self, monkeypatch: pytest.MonkeyPatch):
        run = _run_rollout(monkeypatch, 0.0)
        assert run["status"] == "success"
        assert run["commanded_span"] >= BUDGET * 0.5
        assert run["commands"] >= 5

    @pytest.mark.parametrize("step_by", FORWARD_STEPS)
    def test_a_forward_step_does_not_abandon_the_rollout(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        """The arm is not left parked mid-task by a clock correction."""
        run = _run_rollout(monkeypatch, step_by)

        assert run["commanded_span"] >= BUDGET * 0.5, (
            f"a +{step_by}s wall-clock step ended the rollout after "
            f"{run['commanded_span']:.3f}s of a {BUDGET}s budget "
            f"({run['commands']} commands)"
        )
        assert run["commands"] >= 5

    @pytest.mark.parametrize("step_by", BACKWARD_STEPS)
    def test_a_backward_step_does_not_extend_the_rollout(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        """The servo bus is not commanded past the authorized window.

        Bounded at half the step so the assertion distinguishes "spent the
        budget" from "spent the budget plus the step" without asserting how
        fast the host runs the loop.
        """
        limit = BUDGET + abs(step_by) / 2
        run = _run_rollout(monkeypatch, step_by)

        assert run["commanded_span"] < limit, (
            f"a {step_by}s wall-clock step kept the arm commanded for "
            f"{run['commanded_span']:.3f}s on a {BUDGET}s budget"
        )

    @pytest.mark.parametrize("step_by", [*FORWARD_STEPS, *BACKWARD_STEPS])
    def test_the_reported_duration_is_the_time_that_elapsed(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        """A record of the rollout that a clock step cannot rewrite."""
        run = _run_rollout(monkeypatch, step_by)

        assert run["reported_duration"] == pytest.approx(run["call_span"], abs=0.2), (
            f"a {step_by}s wall-clock step made a {run['call_span']:.3f}s rollout "
            f"report itself as {run['reported_duration']:.3f}s"
        )


class TestTheTeleopSessionSpendsTheBudgetItWasGiven:
    """``duration`` bounds a teleoperation session on the same terms."""

    def test_no_step_is_the_control(self, monkeypatch: pytest.MonkeyPatch):
        run = _run_teleop_session(monkeypatch, 0.0)
        assert run["status"] == "success"
        assert run["frames"] >= 5

    @pytest.mark.parametrize("step_by", FORWARD_STEPS)
    def test_a_forward_step_does_not_end_the_session_early(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        run = _run_teleop_session(monkeypatch, step_by)

        assert run["call_span"] >= BUDGET * 0.5, (
            f"a +{step_by}s wall-clock step ended the session after "
            f"{run['call_span']:.3f}s of a {BUDGET}s budget ({run['frames']} frames)"
        )
        assert run["frames"] >= 5

    @pytest.mark.parametrize("step_by", BACKWARD_STEPS)
    def test_a_backward_step_does_not_extend_the_session(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        limit = BUDGET + abs(step_by) / 2
        run = _run_teleop_session(monkeypatch, step_by)

        assert run["call_span"] < limit, (
            f"a {step_by}s wall-clock step kept the follower driven for {run['call_span']:.3f}s on a {BUDGET}s budget"
        )

    @pytest.mark.parametrize("step_by", [*FORWARD_STEPS, *BACKWARD_STEPS])
    def test_the_reported_elapsed_is_the_time_that_elapsed(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        run = _run_teleop_session(monkeypatch, step_by)

        assert run["reported_elapsed"] == pytest.approx(run["call_span"], abs=0.2), (
            f"a {step_by}s wall-clock step made a {run['call_span']:.3f}s session "
            f"report itself as {run['reported_elapsed']:.3f}s"
        )


class TestTheStepTelemetryThrottleHoldsItsRate:
    """The rollout's ``publish_step`` throttle is a duration too.

    It is best-effort telemetry rather than a command, so it is the least
    dangerous of the three sites - but it is measured the same way, and it
    carries its own base forward as it goes: each publish records when it
    happened, and the next one is due a period later.

    That makes it visible only to a step landing *between* two publishes. A
    step landing before the first one moves the base and every later comparison
    together, which changes nothing - so this test asks for the step part-way
    into the rollout rather than at its start.
    """

    @pytest.mark.parametrize("step_by", BACKWARD_STEPS)
    def test_a_backward_step_does_not_stall_the_stream(self, monkeypatch: pytest.MonkeyPatch, step_by: float):
        """A step must not silence the stream until the date catches up."""
        mesh = _RecordingMesh()
        period = 0.05
        _run_rollout(monkeypatch, step_by, mesh=mesh, stream_min_period=period, step_after_seconds=BUDGET / 3)

        assert len(mesh.publish_times) >= 2, "the stream published at most once"
        stamps = mesh.publish_times
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        assert max(gaps) < abs(step_by) / 2, f"a {step_by}s wall-clock step stalled the stream for {max(gaps):.3f}s"


class TestTheThrottleStartsDue:
    """A monotonic base is only meaningful relative to another reading.

    ``_last_stream_pub`` therefore starts at ``-inf`` rather than ``0.0``: the
    first tick of a rollout is due wherever this platform's monotonic epoch
    happens to sit, instead of depending on it being far from zero.
    """

    def test_a_fresh_robot_owes_its_first_publish(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(HardwareRobot, "_initialize_robot", lambda self, robot, cameras, **kw: _FakeArm())
        monkeypatch.setattr(HardwareRobot, "_migrate_legacy_calibration", lambda self: None)
        robot = HardwareRobot(tool_name="test_arm", robot="fake_arm")
        try:
            assert real_time.monotonic() - robot._last_stream_pub >= robot._stream_min_period
        finally:
            robot.cleanup()

    def test_a_fresh_task_state_has_no_start_reading(self):
        assert RobotTaskState().start_mono == 0.0
