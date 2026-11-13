"""The pacer's hybrid wait: sleep to just before the target, spin the rest, so a late OS wake is not a late write."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from strands_robots.robot.clock import MONOTONIC
from strands_robots.robot.run.pace import SPIN, Pacer, spin_for


class LateWakeClock:
    """A host clock: every sleep of more than 0 s wakes ``late`` s after asking; reading it costs ``tick`` s."""

    def __init__(self, late: float, tick: float = 50e-6) -> None:
        self.t, self.late, self.tick, self.sleeps = 1000.0, late, tick, []

    def now(self) -> float:
        self.t += self.tick
        return self.t

    def sleep(self, seconds: float) -> bool:
        self.sleeps.append(seconds)
        if seconds > 0:
            self.t += seconds + self.late
        return False


def _writes(p: Pacer, clock: LateWakeClock, n: int = 200, work: float = 0.003) -> np.ndarray:
    out = []
    for _ in range(n):
        p.start()
        clock.t += work
        p.before_write()
        out.append(clock.t)
        p.after_write()
    return np.diff(out)[1:]  # the first write has no period before it


@pytest.mark.parametrize("late", [0.0005, 0.0015])
def test_a_late_wake_inside_the_spin_does_not_move_the_write(late):
    """100 Hz with 3 ms of work and a sleep that wakes up to 1.5 ms late: with the spin every write lands on its
    one-period grid (within one clock read); with a plain sleep each lands ``late`` after it."""
    dt = 0.01
    spun, plain = LateWakeClock(late), LateWakeClock(late)
    gaps = _writes(Pacer(dt, now=spun.now, sleep=spun.sleep, spin=SPIN), spun)
    assert np.abs(gaps - dt).max() < 3 * spun.tick
    lagged = _writes(Pacer(dt, now=plain.now, sleep=plain.sleep), plain)
    assert np.abs(lagged - dt).max() > late / 2  # without the spin the late wake reaches the bus


def test_the_spin_asks_the_sleep_each_turn_and_an_engaged_latch_ends_the_wait():
    """``sleep(0)`` answering True (the e-stop latch is engaged) returns at once, short of the target."""
    clock = LateWakeClock(0.0)
    answers = iter([False, False, True])
    p = Pacer(0.01, now=clock.now, sleep=lambda s: clock.sleep(s) or (s == 0 and next(answers)), spin=SPIN)
    p.start()
    p.before_write()
    p.after_write()
    t0 = clock.t
    p.start()  # waits for the next grid point: a long sleep, then the spin, cut on its third turn
    assert clock.sleeps.count(0) == 3 and clock.t < t0 + 0.01


def test_without_a_spin_the_wait_is_one_sleep_as_before():
    clock = LateWakeClock(0.001)
    p = Pacer(0.02, now=clock.now, sleep=clock.sleep)
    for _ in range(3):
        p.start()
        p.before_write()
        p.after_write()
    assert 0 not in clock.sleeps and all(s > 0 for s in clock.sleeps)


def test_the_spin_is_never_more_than_half_a_period_and_never_negative():
    clock = LateWakeClock(0.0)
    assert Pacer(0.001, now=clock.now, sleep=clock.sleep, spin=SPIN)._spin == pytest.approx(0.0005)
    with pytest.raises(ValueError, match="spin"):
        Pacer(0.01, spin=-0.001)


def test_only_the_host_monotonic_clock_spins():
    """A clock of the caller's own (a test's, a world's) may only move when slept: a spin on it would never end."""
    assert spin_for(MONOTONIC) == SPIN > 0
    assert spin_for(LateWakeClock(0.0)) == 0.0


def test_a_hardware_run_paces_with_the_spin_and_a_custom_clock_without(monkeypatch):
    from test_kernel import CountingDriver, VirtualClock

    from strands_robots import Robot

    loop, made = importlib.import_module("strands_robots.robot.run.loop"), []

    class Recording(Pacer):
        def __init__(self, *a, **kw) -> None:
            made.append(kw.get("spin"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(loop, "Pacer", Recording)
    Robot("so101", driver=CountingDriver(), twin=False, clock=MONOTONIC).run(lambda obs: {}, steps=3, hz=100)
    Robot("so101", driver=CountingDriver(), twin=False, clock=VirtualClock()).run(lambda obs: {}, steps=3, hz=100)
    assert made == [SPIN, 0.0]
