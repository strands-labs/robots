"""Teleop input frames may not command a joint faster than it can travel.

``InputReceiver._on_input`` bounds each inbound frame on four axes before it
reaches ``send_action()``: who sent it (the subscribed key expression), how
fresh it is, how densely frames may arrive (``STRANDS_MESH_INPUT_MAX_HZ``) and
how large a single value may be (``MAX_INPUT_VALUE_ABS``). Each of those judges
a frame in isolation, so a stream inside every one of them could still reverse a
joint full-scale on every frame - a commanded speed an order of magnitude past
what the leader's own servos can travel, which is the overcurrent / gear-strip
trajectory the rate cap exists to prevent in the time domain.

These tests pin the per-joint slew bound that closes that gap:
:func:`~strands_robots.mesh.security.input_frame_slew_violation`, the per-joint
baseline :func:`~strands_robots.mesh.security.merge_slew_baseline` maintains, and
the receiver guard that applies both. They also pin the properties the design
rests on - a physical leader arm at full speed is never refused, a refused stream
resumes on its own without a resync handshake, and the shape of the frames cannot
change the verdict, since the sender chooses it - plus the composition with the
apply-rate cap that keeps a batched delivery from being mistaken for a high-speed
command.

The receiver tests drive a fake monotonic clock rather than sleeping, so the
interval charged to each frame is exact on any host.
"""

from __future__ import annotations

import logging
import math
import time

import pytest

from strands_robots.mesh import input as mesh_input
from strands_robots.mesh import security
from strands_robots.mesh.input import InputReceiver
from strands_robots.mesh.security import (
    DEFAULT_INPUT_SLEW_ABS,
    input_frame_slew_violation,
    merge_slew_baseline,
)

from .test_input_stream_lifecycle import _make_receiver, _RecvMesh

#: A 50 Hz frame period - the rate InputPublisher streams at by default.
FRAME_S = 1.0 / mesh_input.INPUT_HZ_DEFAULT

#: The teleop envelope is sized in the unit the leader driver publishes, and an
#: SO leader publishes degrees (``use_degrees=True`` by default). The magnitudes
#: below were chosen relative to that bound, so they are stated in frame units
#: via this factor; scaling them together preserves every verdict here.
FRAME_UNITS_PER_RADIAN = 180.0 / math.pi

#: Feetech STS3215 no-load speed at 12 V is ~6.5 rad/s, so a leader arm on an
#: SO-100 class follower travels at least this far in one 50 Hz frame, and any
#: bound that refuses it would refuse legitimate teleoperation. It is a floor,
#: not a ceiling - the leader is back-driven by hand, and recorded teleop reaches
#: 2.4x it (``TestTheSpeedAxisIsSizedOnRecordedTeleop`` in
#: ``tests/mesh/test_input_envelope_units.py``).
LEADER_MAX_STEP = 6.5 * FRAME_UNITS_PER_RADIAN * FRAME_S


class _Clock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@pytest.fixture
def clock(monkeypatch):
    """Replace the monotonic clock the receiver reads for rate and slew."""
    c = _Clock()
    monkeypatch.setattr(mesh_input.time, "perf_counter", c)
    return c


@pytest.fixture(autouse=True)
def _default_slew_bound(monkeypatch):
    """Pin the bound to its default so an operator env var cannot skew a test."""
    monkeypatch.delenv("STRANDS_MESH_INPUT_SLEW_ABS", raising=False)


def _frame(action: dict[str, float], seq: int = 0) -> dict:
    return {"peer_id": "leader-1", "device": "leader", "t": time.time(), "seq": seq, "action": action}


def _base(values: dict[str, float], at: float = 0.0) -> dict[str, tuple[float, float]]:
    """A per-joint baseline whose joints were all last applied at *at*."""
    return {key: (value, at) for key, value in values.items()}


def _timed_receiver(clock: _Clock) -> tuple[InputReceiver, list[tuple[float, dict[str, float]]]]:
    """A receiver that records when each frame was applied, not just what."""
    applied: list[tuple[float, dict[str, float]]] = []
    recv = InputReceiver(
        # A structural stand-in for the mesh: the receiver only subscribes and
        # reads the e-stop lockout, both of which the stub provides.
        mesh=_RecvMesh(),  # type: ignore[arg-type]
        robot=object(),
        source_peer_id="leader-1",
        apply_fn=lambda robot, action: applied.append((clock.now, dict(action))),
    )
    recv._running = True
    return recv, applied


def _max_commanded_speed(applied: list[tuple[float, dict[str, float]]]) -> float:
    """Fastest speed any one joint was actually commanded to travel at.

    Measured across the frames that reached the robot, which is the quantity the
    bound exists to cap - independent of how the stream split joints across
    frames.
    """
    previous: dict[str, tuple[float, float]] = {}
    worst = 0.0
    for applied_at, action in applied:
        for key, value in action.items():
            if key in previous:
                prev_at, prev_value = previous[key]
                dt = applied_at - prev_at
                if dt > 0:
                    worst = max(worst, abs(value - prev_value) / dt)
            previous[key] = (applied_at, value)
    return worst


# --- env knob resolution -------------------------------------------------


class TestInputSlewAbs:
    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_INPUT_SLEW_ABS", raising=False)
        assert security._input_slew_abs() == DEFAULT_INPUT_SLEW_ABS

    def test_valid_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_SLEW_ABS", "400")
        assert security._input_slew_abs() == 400.0

    def test_zero_falls_back_to_default(self, monkeypatch):
        # Unlike the frame-rate cap, this is a safety envelope on actuator
        # travel: it can be widened but not switched off, matching its sibling
        # magnitude bound MAX_INPUT_VALUE_ABS.
        monkeypatch.setenv("STRANDS_MESH_INPUT_SLEW_ABS", "0")
        assert security._input_slew_abs() == DEFAULT_INPUT_SLEW_ABS

    def test_bad_value_falls_back(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_SLEW_ABS", "not-a-number")
        assert security._input_slew_abs() == DEFAULT_INPUT_SLEW_ABS

    def test_negative_falls_back(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_SLEW_ABS", "-5")
        assert security._input_slew_abs() == DEFAULT_INPUT_SLEW_ABS

    def test_default_is_above_leader_servo_speed(self):
        # The bound must sit above what the leader hardware itself can produce,
        # or legitimate teleoperation trips it -- measured in the unit the frames
        # carry. The Feetech STS3215 tops out near 6.5 rad/s at 12 V, and an SO
        # leader publishes degrees (``use_degrees=True`` by default), so the
        # speed to clear is that figure converted, not the radian figure itself.
        no_load_deg_s = 6.5 * 180.0 / math.pi  # ~372 deg/s
        assert DEFAULT_INPUT_SLEW_ABS > no_load_deg_s


# --- the decision function ----------------------------------------------


class TestInputFrameSlewViolation:
    def test_move_within_bound_allowed(self):
        assert input_frame_slew_violation({"j0": 0.1}, _base({"j0": 0.0}), FRAME_S, 0.0) is None

    def test_full_scale_reversal_refused(self):
        reason = input_frame_slew_violation(
            {"j0": -0.9 * FRAME_UNITS_PER_RADIAN}, _base({"j0": 0.9 * FRAME_UNITS_PER_RADIAN}), FRAME_S, 0.0
        )
        assert reason is not None
        assert "j0" in reason
        # 1.8 rad-equivalent units traversed in 0.02 s, in frame units.
        assert f"{1.8 * FRAME_UNITS_PER_RADIAN / FRAME_S:.1f}" in reason

    def test_no_baseline_allowed(self):
        # The first frame of a stream has nothing to measure against.
        assert input_frame_slew_violation({"j0": 9.9}, {}, FRAME_S, 0.0) is None

    def test_joint_absent_from_baseline_allowed(self):
        # A joint appearing mid-stream has no prior command of its own; the
        # magnitude bound still applies to it.
        assert input_frame_slew_violation({"j1": 9.9}, _base({"j0": 0.0}), FRAME_S, 0.0) is None

    def test_unchanged_pose_allowed(self):
        assert (
            input_frame_slew_violation(
                {"j0": 0.9 * FRAME_UNITS_PER_RADIAN}, _base({"j0": 0.9 * FRAME_UNITS_PER_RADIAN}), 1e-9, 0.0
            )
            is None
        )

    def test_zero_interval_with_movement_refused(self):
        reason = input_frame_slew_violation({"j0": 0.5}, _base({"j0": 0.0}), 0.0, 0.0)
        assert reason is not None
        assert "no elapsed time" in reason

    def test_negative_interval_with_movement_refused(self):
        assert input_frame_slew_violation({"j0": 0.5}, _base({"j0": 0.0}), -1.0, 0.0) is not None

    def test_zero_interval_without_movement_allowed(self):
        assert input_frame_slew_violation({"j0": 0.5}, _base({"j0": 0.5}), 0.0, 0.0) is None

    def test_bound_is_a_speed_not_a_step(self):
        # The same displacement is refused over one frame and allowed once
        # enough time has passed for it to be travelled safely. This is what
        # makes a refusal self-healing instead of a permanent stall.
        jump = {"j0": 3.0 * FRAME_UNITS_PER_RADIAN}
        base = _base({"j0": 0.0})
        assert input_frame_slew_violation(jump, base, FRAME_S, 0.0) is not None
        assert (
            input_frame_slew_violation(jump, base, 3.0 * FRAME_UNITS_PER_RADIAN / DEFAULT_INPUT_SLEW_ABS + 0.01, 0.0)
            is None
        )

    def test_each_joint_is_measured_against_its_own_last_command(self):
        # j0 last moved a second ago, j1 a frame ago. The same displacement is
        # reachable safely for the first and not for the second, so a shared
        # interval could only be wrong for one of them.
        base = {"j0": (0.0, 0.0), "j1": (0.0, 0.98)}
        assert input_frame_slew_violation({"j0": 3.0 * FRAME_UNITS_PER_RADIAN}, base, 1.0, 0.0) is None
        reason = input_frame_slew_violation({"j1": 3.0 * FRAME_UNITS_PER_RADIAN}, base, 1.0, 0.0)
        assert reason is not None and "j1" in reason

    def test_interval_is_floored_at_the_minimum_apply_interval(self):
        # Two frames delivered with no time between them are the rate cap's
        # business; the move is charged the interval the cap guarantees.
        move, base = {"j0": 0.1}, _base({"j0": 0.0})
        assert input_frame_slew_violation(move, base, 0.0, 0.0) is not None
        assert input_frame_slew_violation(move, base, 0.0, FRAME_S) is None

    def test_reports_worst_offender_regardless_of_key_order(self):
        base = _base({"a": 0.0, "b": 0.0, "c": 0.0})
        fast = {"a": 1.0, "b": 5.0 * FRAME_UNITS_PER_RADIAN, "c": 2.0}
        reason = input_frame_slew_violation(fast, base, FRAME_S, 0.0)
        assert reason is not None and "'b'" in reason
        # Same frame, keys inserted in a different order -> same verdict.
        reordered = {"c": 2.0, "b": 5.0 * FRAME_UNITS_PER_RADIAN, "a": 1.0}
        assert input_frame_slew_violation(reordered, base, FRAME_S, 0.0) == reason

    def test_equally_fast_joints_report_deterministically(self):
        # Equal speeds are broken by joint name, so neither insertion order nor
        # a tie can change the message.
        base = _base({"a": 0.0, "b": 0.0})
        forward = input_frame_slew_violation(
            {"a": 5.0 * FRAME_UNITS_PER_RADIAN, "b": 5.0 * FRAME_UNITS_PER_RADIAN}, base, FRAME_S, 0.0
        )
        backward = input_frame_slew_violation(
            {"b": 5.0 * FRAME_UNITS_PER_RADIAN, "a": 5.0 * FRAME_UNITS_PER_RADIAN}, base, FRAME_S, 0.0
        )
        assert forward is not None and forward == backward

    def test_one_fast_joint_refuses_the_frame(self):
        reason = input_frame_slew_violation(
            {"j0": 0.01, "j1": 9.9 * FRAME_UNITS_PER_RADIAN}, _base({"j0": 0.0, "j1": 0.0}), FRAME_S, 0.0
        )
        assert reason is not None and "j1" in reason

    def test_explicit_bound_honoured(self):
        # A move refused at the default bound is allowed under a wider one.
        move, base = {"j0": 1.0 * FRAME_UNITS_PER_RADIAN}, _base({"j0": 0.0})
        assert input_frame_slew_violation(move, base, FRAME_S, 0.0) is not None
        assert input_frame_slew_violation(move, base, FRAME_S, 0.0, max_slew=1e6) is None

    def test_env_widens_the_bound(self, monkeypatch):
        move, base = {"j0": 1.0 * FRAME_UNITS_PER_RADIAN}, _base({"j0": 0.0})
        assert input_frame_slew_violation(move, base, FRAME_S, 0.0) is not None
        monkeypatch.setenv("STRANDS_MESH_INPUT_SLEW_ABS", "1000000")
        assert input_frame_slew_violation(move, base, FRAME_S, 0.0) is None

    def test_reason_names_the_bound_and_the_measured_speed(self):
        # This many units in 1 s is that many units/s, just over the bound.
        step = 30.0 * FRAME_UNITS_PER_RADIAN
        reason = input_frame_slew_violation({"j0": step}, _base({"j0": 0.0}), 1.0, 0.0)
        assert reason is not None
        assert f"{step:.1f}" in reason and f"{DEFAULT_INPUT_SLEW_ABS:g}" in reason
        assert "1.0000s" in reason  # the offending joint's own interval

    def test_leader_at_max_servo_speed_allowed(self):
        assert input_frame_slew_violation({"j0": LEADER_MAX_STEP}, _base({"j0": 0.0}), FRAME_S, 0.0) is None

    def test_non_finite_baseline_does_not_crash(self):
        # validate_input_frame rejects non-finite values, so the guard never
        # sees one from the wire; a direct caller must still get an answer
        # rather than an exception.
        assert input_frame_slew_violation({"j0": 0.0}, _base({"j0": math.inf}), FRAME_S, 0.0) is not None


# --- the per-joint baseline ---------------------------------------------


class TestMergeSlewBaseline:
    def test_applied_joints_are_stamped(self):
        merged = merge_slew_baseline({}, {"j0": 0.5}, 10.0)
        assert merged == {"j0": (0.5, 10.0)}

    def test_omitted_joints_keep_their_previous_entry(self):
        # The bypass this closes: a frame carrying one joint must not erase the
        # baseline of the joints it does not mention.
        previous = {"j0": (0.9 * FRAME_UNITS_PER_RADIAN, 10.0), "j1": (0.1, 10.0)}
        merged = merge_slew_baseline(previous, {"j1": 0.2}, 10.02)
        assert merged["j0"] == (0.9 * FRAME_UNITS_PER_RADIAN, 10.0)
        assert merged["j1"] == (0.2, 10.02)

    def test_an_entry_that_can_still_refuse_a_frame_is_kept(self):
        # A quarter of the way to its horizon, this entry still refuses a
        # full-envelope command, so it must survive.
        horizon = security.MAX_INPUT_VALUE_ABS / DEFAULT_INPUT_SLEW_ABS
        merged = merge_slew_baseline({"j0": (0.0, 0.0)}, {"j1": 0.0}, horizon / 4)
        assert "j0" in merged

    def test_an_entry_that_can_no_longer_refuse_anything_is_dropped(self):
        # Past its horizon the widest permissible command is reachable within
        # the bound, so the entry cannot change a verdict and is not retained -
        # which is what keeps a baseline keyed by sender-chosen names bounded.
        horizon = security.MAX_INPUT_VALUE_ABS / DEFAULT_INPUT_SLEW_ABS
        merged = merge_slew_baseline({"j0": (0.0, 0.0)}, {"j1": 0.0}, horizon + 0.01)
        assert "j0" not in merged
        # Dropping it changes no verdict: the move it would have been measured
        # against is within the bound over that interval anyway.
        widest = {"j0": security.MAX_INPUT_VALUE_ABS}
        assert input_frame_slew_violation(widest, {"j0": (0.0, 0.0)}, horizon + 0.01, 0.0) is None

    def test_a_far_baseline_is_retained_longer_than_a_near_one(self):
        # A joint parked at the edge of the envelope can be commanded twice as
        # far as one at the origin, so its entry stays relevant twice as long.
        near, far = 0.0, security.MAX_INPUT_VALUE_ABS
        at = 1.5 * security.MAX_INPUT_VALUE_ABS / DEFAULT_INPUT_SLEW_ABS
        merged = merge_slew_baseline({"near": (near, 0.0), "far": (far, 0.0)}, {}, at)
        assert "near" not in merged
        assert "far" in merged

    def test_explicit_bounds_honoured(self):
        # A wider speed bound shortens the horizon; a wider envelope lengthens it.
        previous = {"j0": (0.0, 0.0)}
        assert merge_slew_baseline(previous, {}, 1.0, max_slew=1e6, value_abs=1.0) == {}
        assert merge_slew_baseline(previous, {}, 1.0, max_slew=1.0, value_abs=1e6) == previous

    def test_values_are_normalised_to_floats(self):
        merged = merge_slew_baseline({}, {"j0": 1}, 10.0)
        assert isinstance(merged["j0"][0], float)

    def test_previous_is_not_mutated(self):
        previous = {"j0": (0.9 * FRAME_UNITS_PER_RADIAN, 10.0)}
        merge_slew_baseline(previous, {"j0": 0.1}, 10.02)
        assert previous == {"j0": (0.9 * FRAME_UNITS_PER_RADIAN, 10.0)}


# --- the receiver guard --------------------------------------------------


class TestReceiverRefusesOverSpeedFrames:
    def test_sustained_full_scale_reversals_never_move_the_joint(self, clock):
        # The reproduction: every frame is correctly scoped, fresh, inside the
        # 100 Hz rate cap and inside the magnitude bound, yet reverses the
        # joint full-scale at 50 Hz.
        recv, applied = _make_receiver()
        for i in range(60):
            clock.advance(FRAME_S)
            recv._on_input(
                recv.topic,
                _frame({"j0": 0.9 * FRAME_UNITS_PER_RADIAN if i % 2 == 0 else -0.9 * FRAME_UNITS_PER_RADIAN}, seq=i),
            )

        assert recv.stats["slew_rejected"] == 30
        # Only the opening pose and its repeats survive, so the commanded pose
        # never actually changes: the joint holds still instead of oscillating.
        assert {a["j0"] for a in applied} == {0.9 * FRAME_UNITS_PER_RADIAN}

    def test_first_frame_is_applied(self, clock):
        recv, applied = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 5.0 * FRAME_UNITS_PER_RADIAN}))
        assert applied == [{"j0": 5.0 * FRAME_UNITS_PER_RADIAN}]
        assert recv.stats["slew_rejected"] == 0

    def test_leader_at_max_servo_speed_is_never_refused(self, clock):
        recv, applied = _make_receiver()
        pos = 0.0
        for i in range(60):
            pos += LEADER_MAX_STEP
            clock.advance(FRAME_S)
            recv._on_input(recv.topic, _frame({"j0": pos}, seq=i))

        assert recv.stats["slew_rejected"] == 0
        assert len(applied) == 60

    def test_refused_frame_never_reaches_the_robot(self, clock):
        recv, applied = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 9.0 * FRAME_UNITS_PER_RADIAN}, seq=1))

        assert applied == [{"j0": 0.0}]
        assert recv.stats["frames_received"] == 1
        assert recv.stats["slew_rejected"] == 1

    def test_refused_frame_does_not_become_the_baseline(self, clock):
        # Otherwise a refused excursion would silently license the next one.
        recv, applied = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 9.0 * FRAME_UNITS_PER_RADIAN}, seq=1))  # refused
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 9.05 * FRAME_UNITS_PER_RADIAN}, seq=2))  # near the refused pose

        assert recv.stats["slew_rejected"] == 2
        assert applied == [{"j0": 0.0}]

    def test_stream_resumes_without_a_resync(self, clock):
        # The allowance grows while nothing is applied, so the follower catches
        # up to the leader on its own once the pose is reachable safely.
        recv, applied = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 3.0 * FRAME_UNITS_PER_RADIAN}, seq=1))
        assert recv.stats["slew_rejected"] == 1

        clock.advance(3.0 * FRAME_UNITS_PER_RADIAN / DEFAULT_INPUT_SLEW_ABS + 0.01)
        recv._on_input(recv.topic, _frame({"j0": 3.0 * FRAME_UNITS_PER_RADIAN}, seq=2))

        assert applied == [{"j0": 0.0}, {"j0": 3.0 * FRAME_UNITS_PER_RADIAN}]
        assert recv.stats["slew_rejected"] == 1

    def test_stats_reports_slew_rejections(self, clock):
        recv, _ = _make_receiver()
        assert recv.stats["slew_rejected"] == 0
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 9.0 * FRAME_UNITS_PER_RADIAN}, seq=1))
        assert recv.stats["slew_rejected"] == 1

    def test_warning_is_rate_limited(self, clock, caplog):
        recv, _ = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        # A constant far target inside the magnitude envelope, with the clock
        # advanced little enough that the growing allowance never reaches it:
        # 12 units over at most 0.3 s is 40 units/s, still over the bound.
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.input"):
            for i in range(20):
                clock.advance(0.015)
                recv._on_input(recv.topic, _frame({"j0": 12.0 * FRAME_UNITS_PER_RADIAN}, seq=i + 1))

        assert recv.stats["slew_rejected"] == 20
        refusals = [r for r in caplog.records if "slew" in r.getMessage()]
        assert len(refusals) == 5

    def test_slew_rejection_is_counted_separately_from_other_refusals(self, clock):
        # ``rejected`` covers the E-stop / freshness / validation refusals; a
        # slew refusal gets its own counter, the way ``rate_dropped`` does.
        recv, _ = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 9.0 * FRAME_UNITS_PER_RADIAN}, seq=1))

        assert recv.stats["slew_rejected"] == 1
        assert recv.stats["rejected"] == 0
        assert recv.stats["rate_dropped"] == 0

    def test_value_outside_magnitude_bound_is_still_a_validation_refusal(self, clock):
        # The slew guard runs after validate_input_frame, so an out-of-envelope
        # value is reported as a validation refusal rather than a slew one.
        recv, _ = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": security.MAX_INPUT_VALUE_ABS * 10}, seq=1))

        assert recv.stats["rejected"] == 1
        assert recv.stats["slew_rejected"] == 0


class TestSlewBoundComposesWithRateCap:
    def test_burst_of_small_steps_is_applied_when_the_rate_cap_is_off(self, clock, monkeypatch):
        # An operator may disable the rate cap on a trusted network, and a
        # batched delivery then arrives with no time between frames. Those
        # commands supersede each other before an actuator can act on them, so
        # charging them an unbounded speed would refuse a harmless burst.
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, applied = _make_receiver()
        for i in range(5):
            recv._on_input(recv.topic, _frame({"j0": 0.1 * i}, seq=i))  # clock never advances

        assert len(applied) == 5
        assert recv.stats["slew_rejected"] == 0

    def test_burst_of_full_scale_reversals_is_still_refused(self, clock, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, applied = _make_receiver()
        for i in range(20):
            recv._on_input(
                recv.topic,
                _frame({"j0": 0.9 * FRAME_UNITS_PER_RADIAN if i % 2 == 0 else -0.9 * FRAME_UNITS_PER_RADIAN}, seq=i),
            )

        assert recv.stats["slew_rejected"] == 10
        assert {a["j0"] for a in applied} == {0.9 * FRAME_UNITS_PER_RADIAN}

    def test_frames_closer_than_the_cap_are_the_cap_s_business(self, clock, monkeypatch):
        # With the cap on, a frame arriving inside the minimum interval is
        # shed by the cap; the slew bound does not also charge it.
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "10")  # 100 ms minimum
        recv, applied = _make_receiver()
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0}, seq=0))
        clock.advance(0.001)
        recv._on_input(recv.topic, _frame({"j0": 9.0 * FRAME_UNITS_PER_RADIAN}, seq=1))

        assert recv.stats["rate_dropped"] == 1
        assert recv.stats["slew_rejected"] == 0
        assert applied == [{"j0": 0.0}]


class TestFrameShapeCannotChangeTheVerdict:
    """The sender chooses how joints are split across frames, so the bound may
    not depend on that choice. A baseline replaced wholesale on every apply
    would: a stream interleaving single-joint frames would leave each joint
    absent from the baseline exactly when it is about to be reversed, so every
    frame would arrive with no reference and none would ever be refused.
    """

    def test_interleaved_single_joint_frames_cannot_reverse_a_joint_every_frame(self, clock):
        # Two joints, one per frame, each reversing full-scale every time it
        # appears: every frame is correctly scoped, fresh, inside the rate cap
        # and inside the magnitude bound, and no frame ever repeats a joint.
        recv, applied = _timed_receiver(clock)
        joints = ("j0", "j1")
        for i in range(60):
            clock.advance(FRAME_S)
            value = 0.9 * FRAME_UNITS_PER_RADIAN if (i // 2) % 2 == 0 else -0.9 * FRAME_UNITS_PER_RADIAN
            recv._on_input(recv.topic, _frame({joints[i % 2]: value}, seq=i))

        assert recv.stats["slew_rejected"] > 0
        # The property the bound exists to guarantee, measured on what actually
        # reached the robot: no joint was ever commanded past the bound.
        assert _max_commanded_speed(applied) <= DEFAULT_INPUT_SLEW_ABS

    def test_a_frame_that_omits_a_joint_does_not_clear_its_baseline(self, clock):
        recv, applied = _timed_receiver(clock)
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.9 * FRAME_UNITS_PER_RADIAN, "j1": 0.0}, seq=0))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j1": 0.01}, seq=1))  # j0 not mentioned
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": -0.9 * FRAME_UNITS_PER_RADIAN}, seq=2))  # full-scale reversal

        assert recv.stats["slew_rejected"] == 1
        assert [action for _, action in applied] == [{"j0": 0.9 * FRAME_UNITS_PER_RADIAN, "j1": 0.0}, {"j1": 0.01}]

    def test_a_flood_of_unseen_joint_names_does_not_dislodge_a_live_joint(self, clock):
        # A frame may carry up to MAX_INPUT_FRAME_KEYS joints, so a baseline
        # bounded by eviction could be cleared in a single frame of names the
        # follower does not even have. Retention is by relevance, not capacity.
        recv, applied = _timed_receiver(clock)
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.9 * FRAME_UNITS_PER_RADIAN}, seq=0))
        clock.advance(FRAME_S)
        flood = {f"f{n}": 0.0 for n in range(security.MAX_INPUT_FRAME_KEYS - 1)}
        recv._on_input(recv.topic, _frame(flood, seq=1))
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": -0.9 * FRAME_UNITS_PER_RADIAN}, seq=2))

        assert recv.stats["slew_rejected"] == 1
        assert [action for _, action in applied] == [{"j0": 0.9 * FRAME_UNITS_PER_RADIAN}, flood]

    def test_a_paused_joint_is_not_over_refused_when_it_moves_again(self, clock):
        # The flip side: measuring every joint against the stream's last apply
        # instead of its own would refuse a joint that held still while others
        # moved, which is ordinary teleoperation, not an attack.
        recv, applied = _timed_receiver(clock)
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 0.0, "j1": 0.0}, seq=0))
        for i in range(15):  # 0.3 s of j1-only traffic; j0 holds still
            clock.advance(FRAME_S)
            recv._on_input(recv.topic, _frame({"j1": 0.01 * i}, seq=i + 1))

        # j0's baseline is still there - a move too fast for the elapsed 0.32 s
        # is still refused ...
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 12.0 * FRAME_UNITS_PER_RADIAN}, seq=100))
        assert recv.stats["slew_rejected"] == 1
        # ... but one reachable within the bound over that interval is applied.
        clock.advance(FRAME_S)
        recv._on_input(recv.topic, _frame({"j0": 6.0}, seq=101))
        assert applied[-1][1] == {"j0": 6.0}
        assert recv.stats["slew_rejected"] == 1
        assert _max_commanded_speed(applied) <= DEFAULT_INPUT_SLEW_ABS
