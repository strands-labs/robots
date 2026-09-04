# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A sim rollout's step-telemetry throttle survives a wall-clock step.

``MuJoCoSimEngine._make_run_policy_hook`` rate-limits ``Mesh.publish_step`` so a
rollout stepping at the control frequency does not exceed the transport caps.
The throttle is an elapsed interval that carries its own base forward as it
goes: each publish records when it happened, and the next one is due a period
later. That base used to be ``time.time()``, which is not a clock but the
current opinion about the date -- an NTP correction, a ``date -s`` or a resume
from suspend moves it by an arbitrary amount.

Driving a real ``so101`` rollout for 0.9s at 50 Hz with a 0.05s throttle period
and a wall clock that takes one step 0.3s in::

    clock event            publishes   largest gap   silent at the end   reported
    no step (control)         15         0.062s            0.06s         success
    wall clock steps -2s       5         0.061s            0.66s         success
    wall clock steps -10s      5         0.061s            0.66s         success

A backward step makes ``now - last`` negative, so the throttle refuses every
later step until the date catches up -- for longer than the rollout lasts, in
both rows. Two thirds of the stream is gone and nothing says so: the publishes
that did land stay correctly spaced, so the largest gap between consecutive
stamps reads ~0.06s in every row, because the publishes the step suppressed
leave no gap behind them. The count is the only thing that moved. The consumers
``publish_step`` exists for (``robot_mesh`` watch, dashboards) see a stream that
stops, nothing raises, and every row reports ``status="success"``.

The hardware control loop throttles the same publish against the same period
resolved by the same :func:`~strands_robots.mesh.session.stream_min_period_from_env`,
and reads ``time.monotonic()`` for it. This is the sim half of that contract, and
the boundary the library settled for its agent-callable tools, its hardware
control loops and its mesh: a duration is local bookkeeping and belongs on a
monotonic clock, while an absolute stamp a reader correlates with other logs
stays on the wall clock. The ``TrajectoryStep.timestamp`` written by the same
hook is such a stamp and is deliberately untouched.

These tests pin the contract on behaviour rather than on which clock is called:
each drives a real MuJoCo rollout through a clock double whose *wall* clock
takes a known step mid-rollout while its monotonic clock does not, and asserts
the stream was still publishing when the rollout ended.
"""

from __future__ import annotations

import math
import time as real_time
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.simulation.mujoco import simulation as mujoco_simulation_module

#: Rollout length. Long enough that the step lands with many steps on each side.
BUDGET = 0.9
#: Rollout control rate, so the hook runs ~45 times.
CONTROL_HZ = 50.0
#: Throttle period under test, and the rate ``STRANDS_MESH_STREAM_HZ`` names.
THROTTLE_PERIOD = 0.05
#: How far into the rollout the wall-clock step lands. It has to fall *between*
#: two publishes to be visible at all: a step before the first one moves the
#: base and every later comparison together, which changes nothing.
STEP_AT = BUDGET / 3
#: Backward steps. Both exceed the rollout's remaining time, which is the point:
#: the stream never recovers within the rollout it was meant to describe.
BACKWARD_STEPS = [-2.0, -10.0]
#: Publishes a clean rollout is due. The hook runs at the control rate, so a
#: publish lands on the first control step at or past each period boundary --
#: derived from both rates rather than from the period alone, which would
#: over-count by the difference and make the threshold below meaningless.
PUBLISHES_DUE = BUDGET / (math.ceil(THROTTLE_PERIOD * CONTROL_HZ) / CONTROL_HZ)


class _SteppingWallClock:
    """A ``time`` stand-in whose ``time()`` takes one step, mid-rollout.

    Every other member delegates to the real module, so a throttle that
    measures its interval on ``monotonic()`` sees an untouched clock and one
    that measures it on ``time()`` sees the step. ``stepped`` records whether
    the step was ever handed out -- a throttle on the monotonic clock never
    reads ``time()`` at all, so it is not used as a precondition for the
    behaviour under test.
    """

    def __init__(self, step_by: float, after_seconds: float = 0.0) -> None:
        self.step_by = step_by
        self.after_seconds = after_seconds
        self.reads = 0
        self.stepped = False
        self._created = real_time.monotonic()

    def time(self) -> float:
        self.reads += 1
        if self.step_by and real_time.monotonic() - self._created >= self.after_seconds:
            self.stepped = True
            return real_time.time() + self.step_by
        return real_time.time()

    def monotonic(self) -> float:
        return real_time.monotonic()

    def perf_counter(self) -> float:
        return real_time.perf_counter()

    def sleep(self, seconds: float) -> None:
        real_time.sleep(seconds)


class _NearZeroMonotonicClock:
    """A ``time`` stand-in whose monotonic epoch sits next to zero.

    ``time.monotonic``'s reference point is undefined; on the platforms this
    runs on it happens to be boot time, which is far from zero. A throttle whose
    "never published" sentinel is ``0.0`` is due on its first tick only because
    of that accident, so this double removes it.
    """

    def __init__(self) -> None:
        self._base = real_time.monotonic()

    def monotonic(self) -> float:
        return real_time.monotonic() - self._base

    def time(self) -> float:
        return real_time.time()

    def perf_counter(self) -> float:
        return real_time.perf_counter()

    def sleep(self, seconds: float) -> None:
        real_time.sleep(seconds)


class _RecordingMesh:
    """Records every ``publish_step`` the rollout's throttle lets through."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.stamps: list[float] = []

    def publish_step(self, step: int, observation: Any, action: Any, **kwargs: Any) -> None:
        self.steps.append(step)
        self.stamps.append(real_time.monotonic())

    def stop(self) -> None:
        return None


def _run_rollout(
    monkeypatch: pytest.MonkeyPatch, build_clock: Callable[[], Any], stream_hz: str | None = None
) -> dict[str, Any]:
    """Drive one real ``so101`` rollout with a clock standing in for ``time``.

    The clock is built after the world is, so ``after_seconds`` is measured from
    the rollout rather than from however long loading the model took.
    """
    import strands_robots

    monkeypatch.setenv("STRANDS_MESH_STREAM_HZ", stream_hz or str(1.0 / THROTTLE_PERIOD))
    mesh = _RecordingMesh()
    sim = strands_robots.Robot("so101", mode="sim", mesh=False)
    sim.mesh = mesh
    clock = build_clock()
    monkeypatch.setattr(mujoco_simulation_module, "time", clock)
    try:
        result = sim.run_policy(
            robot_name="so101",
            policy_provider="mock",
            instruction="probe",
            duration=BUDGET,
            control_frequency=CONTROL_HZ,
            action_horizon=1,
        )
        ended = real_time.monotonic()
    finally:
        try:
            sim.cleanup()
        except Exception:  # noqa: BLE001 - teardown must not mask the assertion
            pass
    return {
        "status": result["status"],
        "steps": mesh.steps,
        "stamps": mesh.stamps,
        "tail_silence": (ended - mesh.stamps[-1]) if mesh.stamps else float("inf"),
        "clock": clock,
    }


class TestTheClockDoubleIsFaithful:
    """The double really moves the wall clock, so a clean run means something."""

    @pytest.mark.parametrize("step_by", BACKWARD_STEPS)
    def test_the_double_hands_out_the_step_it_advertises(self, step_by: float) -> None:
        clock = _SteppingWallClock(step_by)

        real_before = real_time.time()
        reported_before = clock.time()
        reported_after = clock.time()
        real_after = real_time.time()

        assert clock.stepped is True
        # Relative: what the double added beyond the real clock's own advance is
        # the step, whatever the real clock did meanwhile.
        drift = (reported_after - reported_before) - (real_after - real_before)
        assert drift == pytest.approx(0.0, abs=0.05)
        assert reported_after - real_after == pytest.approx(step_by, abs=0.05)

    def test_a_zero_step_never_moves_the_clock(self) -> None:
        clock = _SteppingWallClock(0.0)
        for _ in range(5):
            clock.time()
        assert clock.stepped is False


class TestTheStreamHoldsItsRateAcrossAClockStep:
    """A wall-clock step must not silence per-step telemetry."""

    def test_no_step_is_the_control(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rollout with a steady clock streams for its whole length."""
        run = _run_rollout(monkeypatch, lambda: _SteppingWallClock(0.0, after_seconds=STEP_AT))

        assert run["status"] == "success"
        assert len(run["steps"]) >= 2, "the control run published at most once"
        assert run["tail_silence"] < 5 * THROTTLE_PERIOD

    @pytest.mark.parametrize("step_by", BACKWARD_STEPS)
    def test_a_backward_step_does_not_silence_the_rest_of_the_rollout(
        self, monkeypatch: pytest.MonkeyPatch, step_by: float
    ) -> None:
        """The stream must still be publishing when the rollout ends."""
        run = _run_rollout(monkeypatch, lambda: _SteppingWallClock(step_by, after_seconds=STEP_AT))

        assert run["status"] == "success"
        assert run["tail_silence"] < 5 * THROTTLE_PERIOD, (
            f"a {step_by}s wall-clock step left the stream silent for the last "
            f"{run['tail_silence']:.3f}s of the rollout"
        )

    @pytest.mark.parametrize("step_by", BACKWARD_STEPS)
    def test_a_backward_step_does_not_cost_the_rollout_its_publishes(
        self, monkeypatch: pytest.MonkeyPatch, step_by: float
    ) -> None:
        """The count is what shows the shortfall; the surviving gaps do not.

        Every gap between consecutive publishes stays at the throttle period
        whether the step landed or not, because the ones the step suppressed
        leave no gap behind. So the number that changes is how many arrived.
        """
        run = _run_rollout(monkeypatch, lambda: _SteppingWallClock(step_by, after_seconds=STEP_AT))

        assert len(run["steps"]) >= 0.6 * PUBLISHES_DUE, (
            f"a {step_by}s wall-clock step cost the rollout "
            f"{PUBLISHES_DUE - len(run['steps']):.0f} of its {PUBLISHES_DUE:.0f} publishes"
        )


class TestTheThrottleStartsDue:
    """A monotonic reading is only meaningful relative to another one.

    The "never published" sentinel is therefore ``-inf`` rather than ``0.0``:
    the first step of a rollout is due wherever this platform's monotonic epoch
    happens to sit, instead of depending on it being far from zero. The
    hardware loop settled the same boundary for ``_last_stream_pub``.
    """

    def test_a_rollout_publishes_its_first_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        run = _run_rollout(monkeypatch, _NearZeroMonotonicClock)

        assert run["status"] == "success"
        assert run["steps"], "a rollout on a near-zero monotonic epoch published nothing"
        assert run["steps"][0] == 0, (
            "the first step of the rollout was not published; the throttle's "
            f"sentinel is not below every monotonic reading (first published "
            f"step was {run['steps'][0]})"
        )


class TestAnOperatorsOptOutStaysOff:
    """``STRANDS_MESH_STREAM_HZ=0`` means no step telemetry, including step 0.

    The opt-out is spelled as an infinite period, on the reasoning that no
    elapsed time reaches it. A sentinel below every reading breaks that: the
    gate's subtraction is ``inf`` on the first step, and ``inf >= inf`` holds,
    so the period alone stops being sufficient and the hook reads it directly.
    One escaped publish is a whole observation, action and instruction on the
    mesh, which is what the operator turned off.
    """

    @pytest.mark.parametrize("raw", ["0", "-1", "fast"])
    def test_a_rollout_publishes_nothing(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        run = _run_rollout(monkeypatch, lambda: _SteppingWallClock(0.0), stream_hz=raw)

        assert run["status"] == "success"
        assert run["steps"] == [], (
            f"STRANDS_MESH_STREAM_HZ={raw!r} turns step telemetry off, and the rollout published steps {run['steps']}"
        )
