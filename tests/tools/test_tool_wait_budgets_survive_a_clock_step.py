"""Every agent-callable tool measures a wait budget on a clock that cannot step.

An agent-callable tool publishes its time budget as a tool parameter
(``use_ros(timeout=...)``, ``lerobot_camera(preview_duration=...)``), so the
budget is part of the tool's contract with its caller: ask for five seconds and
the tool waits five seconds. ``time.time()`` cannot honour that, because it is
not a clock -- it is the current opinion about the date, and an NTP correction,
a ``date -s``, or a VM resume moves it by an arbitrary amount mid-wait. A
forward step ends the wait early with the work still in flight, so an arriving
message is reported as a timeout; a backward step runs past the budget by the
size of the step.

The mesh subsystem settled this contract three times already, each time with a
pin test of its own -- ``tests/mesh/test_replay_cache_monotonic.py``,
``tests/mesh/test_bridge_dedup.py::TestDedupClock``,
``tests/mesh/test_corroboration_clock_domain.py`` -- and drew the boundary the
same way each time: a *duration* is local bookkeeping and belongs on
``time.monotonic()``, while an *absolute stamp* stays on the wall clock because
something off this machine correlates it. That boundary is why
``serial_tool``'s five-second monitor window moves while the per-chunk
``timestamp`` in the records it returns does not, and it is pinned below in both
directions.

The source scan that grades the idiom lives in
``tests/test_expiry_gates_survive_a_clock_step.py``, which walks the whole
package: its walk root used to be ``strands_robots/tools`` alone, which is
narrower than the shape it grades, and it read clean while the one offender in
the tree sat in the dashboard's challenge store. What remains here is the
behaviour, driven through the real ``spin_for``.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import pytest

import strands_robots.tools.serial_tool as serial_mod
import strands_robots.tools.use_ros as ros_mod


def test_every_deadline_in_the_ros_tool_is_built_on_one_clock() -> None:
    """One logical deadline, one clock.

    ``_action_send_goal`` governs discovery, acceptance and result delivery with
    a single ``time.monotonic()`` deadline and hands each remaining slice to
    :meth:`_RosBackend.spin_for`. While ``spin_for`` re-based that slice on
    ``time.time()``, the one deadline was measured on two clocks, and the outer
    monotonic guard could not correct the inner loop because it is only consulted
    between ``spin_for`` calls.
    """
    source = pathlib.Path(ros_mod.__file__).read_text(encoding="utf-8")
    built = [line.strip() for line in source.splitlines() if "deadline = time." in line]
    assert built, "premise: no deadline construction found in the use_ros module"
    wall = [line for line in built if "time.monotonic()" not in line]
    assert not wall, f"these deadlines are not on the clock the callers measure their budget with: {wall}"


# ---------------------------------------------------------------------------
# Behaviour: the real spin_for under a stepping wall clock
# ---------------------------------------------------------------------------
class _SteppingWallClock:
    """A wall clock that takes a single step, the way an NTP correction does.

    Elapsed real time comes from ``time.monotonic()``, so the fake advances
    exactly as the true wall clock would, plus one discontinuity of ``step_by``
    seconds once ``step_after`` seconds of real time have passed.
    """

    def __init__(self, step_after: float, step_by: float) -> None:
        self._epoch = 1_700_000_000.0
        self._origin = time.monotonic()
        self._step_after = step_after
        self._step_by = step_by
        self._stepped = False

    def __call__(self) -> float:
        elapsed = time.monotonic() - self._origin
        if not self._stepped and elapsed >= self._step_after:
            self._stepped = True
            self._epoch += self._step_by
        return self._epoch + elapsed


class _CountingExecutor:
    """rclpy executor double that records how often it was pumped."""

    def __init__(self) -> None:
        self.spins = 0

    def spin_once(self, timeout_sec: float = 0.05) -> None:
        self.spins += 1
        time.sleep(0.005)


def _spin_for(
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: float,
    clock: Any = None,
    predicate: Any = None,
) -> tuple[float, int, float]:
    """Drive the real ``spin_for``.

    Returns the true elapsed seconds, how often the executor was pumped, and how
    far the *wall* clock moved across the same window. The last one carries the
    premise: it is a property of the clock double, so it holds whether or not the
    code under test reads that clock -- which a correct implementation does not.
    """
    # Annotated Any so installing the executor double reads as the injection it
    # is: the real attribute is typed for rclpy's SingleThreadedExecutor.
    backend: Any = ros_mod._RosBackend()
    executor = _CountingExecutor()
    backend._executor = executor
    wall = clock or time.time
    if clock is not None:
        monkeypatch.setattr(ros_mod.time, "time", clock)
    wall_before = wall()
    started = time.monotonic()
    backend.spin_for(predicate or (lambda: False), budget)
    elapsed = time.monotonic() - started
    return elapsed, executor.spins, wall() - wall_before


def test_the_stepping_wall_clock_double_takes_the_step_it_advertises() -> None:
    """Pin the double: with no real discontinuity the tests below prove nothing."""
    clock = _SteppingWallClock(step_after=0.05, step_by=+30.0)
    before = clock()
    time.sleep(0.15)
    after = clock()
    assert after - before > 29.0, f"the double advanced {after - before:.3f}s, so it never stepped"


def test_spin_for_waits_its_whole_budget_with_no_clock_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: absent a step, the wait is unchanged and still pumps rclpy."""
    elapsed, spins, wall_moved = _spin_for(monkeypatch, budget=0.4)
    assert wall_moved - elapsed == pytest.approx(0.0, abs=0.2), "premise: this case must not step the clock"
    assert elapsed == pytest.approx(0.4, abs=0.2), f"waited {elapsed:.3f}s for a 0.4s budget"
    assert spins > 0, "the executor was never pumped, so nothing could arrive"


def test_spin_for_returns_as_soon_as_the_predicate_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: a satisfied predicate still short-circuits the budget."""
    elapsed, _, _ = _spin_for(monkeypatch, budget=5.0, predicate=lambda: True)
    assert elapsed < 0.5, f"a satisfied predicate should return at once, waited {elapsed:.3f}s"


def test_spin_for_honours_its_budget_across_a_forward_clock_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forward step must not end the wait with the work still in flight."""
    clock = _SteppingWallClock(step_after=0.1, step_by=+30.0)
    elapsed, _, wall_moved = _spin_for(monkeypatch, budget=0.6, clock=clock)
    assert wall_moved - elapsed == pytest.approx(30.0, abs=0.3), (
        f"premise: the wall clock gained {wall_moved - elapsed:.3f}s on real time, not the +30s step"
    )
    assert elapsed == pytest.approx(0.6, abs=0.2), (
        f"a +30s wall-clock step cut a 0.6s wait to {elapsed:.3f}s, so a message that was "
        "arriving is reported as a timeout"
    )


def test_spin_for_honours_its_budget_across_a_backward_clock_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backward step must not extend the wait past the caller's budget."""
    clock = _SteppingWallClock(step_after=0.1, step_by=-2.0)
    elapsed, _, wall_moved = _spin_for(monkeypatch, budget=0.6, clock=clock)
    assert wall_moved - elapsed == pytest.approx(-2.0, abs=0.3), (
        f"premise: the wall clock lost {elapsed - wall_moved:.3f}s against real time, not the -2s step"
    )
    assert elapsed == pytest.approx(0.6, abs=0.2), (
        f"a -2s wall-clock step stretched a 0.6s wait to {elapsed:.3f}s, holding the caller for the size of the step"
    )


def test_a_step_does_not_starve_the_cancel_that_follows_a_timed_out_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout-cancel fail-safe needs the executor pumped to transmit.

    ``_action_send_goal`` cancels before surfacing a timeout so a timed-out call
    never leaves a robot pursuing an orphaned goal, and waits on the cancel
    future for two seconds. A cancel request only leaves the process while the
    executor is spun, so a wait truncated by a clock step reports a cancel it may
    not have sent.
    """
    clock = _SteppingWallClock(step_after=0.05, step_by=+30.0)
    elapsed, spins, wall_moved = _spin_for(monkeypatch, budget=1.0, clock=clock)
    assert wall_moved - elapsed == pytest.approx(30.0, abs=0.3), (
        f"premise: the wall clock gained {wall_moved - elapsed:.3f}s on real time, not the +30s step"
    )
    assert spins >= 20, (
        f"the cancel wait pumped the executor only {spins} times in {elapsed:.3f}s of a 1.0s "
        "budget, so the cancel request may never have been transmitted while the timeout "
        "reported it was"
    )


def test_a_monitor_record_timestamp_stays_on_the_wall_clock() -> None:
    """The other half of the boundary: an absolute stamp is not a duration.

    ``serial_tool``'s monitor window is a duration and moved to
    ``time.monotonic()``; the ``timestamp`` on each returned record is an
    absolute stamp a reader correlates with other logs, so it must not move with
    it. This fails if the module is swept clock-blind rather than per value.
    """
    source = pathlib.Path(serial_mod.__file__).read_text(encoding="utf-8")
    assert '"timestamp": time.time(),' in source, (
        "the monitor record's timestamp must stay on the wall clock: it is an absolute stamp, "
        "and a time.monotonic() value (seconds of process uptime) is meaningless to a reader"
    )
