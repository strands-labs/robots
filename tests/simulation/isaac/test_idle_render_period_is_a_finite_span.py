# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SO101_IDLE_RENDER_PERIOD`` cannot resolve to a period no refresh satisfies.

The Isaac idle live-preview gate in :meth:`IsaacSimulation.run_pump_forever` is
``now_mono - last_render_mono >= self._idle_render_period``, and
:func:`~strands_robots.simulation.isaac.simulation._env_float` resolved that
period with ``float()`` behind a ``v > 0`` bound. ``inf > 0`` is ``True``, so
every infinite spelling passed through as the period - and no elapsed span
satisfies the gate against it. Driving the real loop for 120 idle iterations
(6.0 s of virtual time at the loop's own 0.05 s granularity):

===================================  ==========  =========================
``SO101_IDLE_RENDER_PERIOD``         refreshes   at (s)
===================================  ==========  =========================
``inf`` / ``Infinity`` / ``1e999``   1 (was)     0.0 - then never again
``inf`` / ``Infinity`` / ``1e999``   6 (now)     0.0, 1.0, 2.0, ... 5.0
``nan`` / ``-inf``                   6           unchanged: ``nan > 0`` and
                                                 ``-inf > 0`` are both False
``2.5``                              3           unchanged: 0.0, 2.5, 5.0
unset                                6           unchanged: the 1.0 default
===================================  ==========  =========================

One refresh at t=0 and none after is a viewport frozen on the first frame for
the life of the process, while ``pump()`` keeps draining the app - which reads as
a stalled simulation rather than as a misconfigured variable. That is why the
fallback is now reported: the operator has to be able to tell those apart.

``nan`` reaching the default before this change was not the bound working, it was
``nan > 0`` being ``False`` - the same route a typo takes. The assertions below
are stated over the *achieved refresh timeline* rather than over the resolved
number wherever the timeline is what the caller observes, so none of them can be
satisfied by renaming a call.

Solver-free: the engine is the skeleton ``__new__`` shape
``test_isaac_durations_survive_a_clock_step.py`` uses and ``pump`` is faked on the
instance, so no Isaac Sim Kit runtime, GL context or GPU is touched.
"""

from __future__ import annotations

import queue
import time
from typing import Any

import pytest

from strands_robots.simulation.isaac import simulation as isaac_module
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _env_float

#: The knob under test and the default its resolver falls back to.
IDLE_PERIOD_ENV = "SO101_IDLE_RENDER_PERIOD"
DEFAULT_PERIOD = 1.0

#: Idle iterations to drive. At the loop's 0.05 s sleep this is 6.0 s of virtual
#: time - long enough that a large *finite* period still refreshes more than
#: once, so "refreshed once" identifies an infinite period rather than a slow one.
PUMP_ITERATIONS = 120
IDLE_SLEEP = 0.05

#: Spellings ``float()`` accepts and the ``> 0`` bound cannot refuse.
INFINITE_SPELLINGS = ["inf", "Inf", "Infinity", "  inf  ", "1e999"]

#: Spellings the positivity bound already sent to the default. Kept as controls:
#: they must keep resolving to the default, and now for a stated reason.
ALREADY_REFUSED = ["nan", "NaN", "-inf", "-1e999", "-1", "0", "junk", ""]


class _VirtualClock:
    """A ``time`` double whose sleeps advance a virtual clock instead of blocking.

    The gate reads ``monotonic()`` once per idle iteration; ``sleep`` moves the
    clock by the requested span, so the achieved refresh timeline is exact and
    the case runs in milliseconds. Every other attribute delegates to real
    :mod:`time`.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self._virtual = start
        self.start = start

    def monotonic(self) -> float:
        return self._virtual

    def time(self) -> float:
        return self._virtual

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._virtual += seconds

    def elapsed(self) -> float:
        """Virtual seconds since the start, as the test's own instrumentation."""
        return self._virtual - self.start

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _StopAfter:
    """A ``threading.Event``-shaped flag ending the loop after ``n`` checks."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks > self.n


def _refresh_timeline(monkeypatch: pytest.MonkeyPatch, iterations: int = PUMP_ITERATIONS) -> list[float]:
    """Drive the real idle gate and return the virtual times it refreshed at.

    The period is resolved exactly as ``__init__`` resolves it, from the
    environment through ``_env_float``, so the timeline is the consequence of the
    resolver rather than of a number the test chose.
    """
    clock = _VirtualClock()
    monkeypatch.setattr(isaac_module, "time", clock)
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._idle_render_period = _env_float(IDLE_PERIOD_ENV, DEFAULT_PERIOD)
    engine._main_jobs = queue.Queue()
    engine._action_q = queue.Queue()
    engine._pump_running = False

    refreshes: list[float] = []

    def _pump(render: bool = True) -> None:
        if render:
            refreshes.append(round(clock.elapsed(), 3))

    engine.pump = _pump  # type: ignore[method-assign]
    engine.run_pump_forever(_StopAfter(iterations))
    return refreshes


@pytest.fixture(autouse=True)
def _no_inherited_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knob must not be inherited from the runner running these tests."""
    monkeypatch.delenv(IDLE_PERIOD_ENV, raising=False)


class TestAnInfinitePeriodDoesNotBecomeThePreviewCadence:
    """The defect: a period the gate can never satisfy, applied in silence."""

    @pytest.mark.parametrize("raw", INFINITE_SPELLINGS)
    def test_the_preview_keeps_refreshing(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        """The timeline, not the number: one refresh is a frozen viewport."""
        monkeypatch.setenv(IDLE_PERIOD_ENV, raw)
        refreshes = _refresh_timeline(monkeypatch)
        assert refreshes == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], f"{raw!r} gave {refreshes}"

    @pytest.mark.parametrize("raw", INFINITE_SPELLINGS)
    def test_the_resolved_period_is_the_default(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(IDLE_PERIOD_ENV, raw)
        assert _env_float(IDLE_PERIOD_ENV, DEFAULT_PERIOD) == DEFAULT_PERIOD

    @pytest.mark.parametrize("raw", INFINITE_SPELLINGS)
    def test_the_fallback_names_the_variable_and_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
    ) -> None:
        """A frozen preview and a stalled simulation look identical; only the
        log distinguishes them, so a silent substitution is not a fix.
        """
        monkeypatch.setenv(IDLE_PERIOD_ENV, raw)
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            _env_float(IDLE_PERIOD_ENV, DEFAULT_PERIOD)
        messages = [record.getMessage() for record in caplog.records]
        assert any(IDLE_PERIOD_ENV in m and "finite" in m for m in messages), messages


class TestTheAcceptedDomainIsUnchanged:
    """Over-reach controls: only the non-finite verdict moved."""

    @pytest.mark.parametrize(("raw", "expected"), [("0.5", 0.5), ("2.5", 2.5), ("1e-3", 0.001)])
    def test_a_finite_positive_period_is_honored(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
    ) -> None:
        monkeypatch.setenv(IDLE_PERIOD_ENV, raw)
        assert _env_float(IDLE_PERIOD_ENV, DEFAULT_PERIOD) == expected

    def test_a_large_finite_period_still_refreshes_more_than_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Separates "refreshed once" from "was configured slowly"."""
        monkeypatch.setenv(IDLE_PERIOD_ENV, "2.5")
        assert _refresh_timeline(monkeypatch) == [0.0, 2.5, 5.0]

    @pytest.mark.parametrize("raw", ALREADY_REFUSED)
    def test_a_value_the_bound_already_refused_still_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv(IDLE_PERIOD_ENV, raw)
        assert _env_float(IDLE_PERIOD_ENV, DEFAULT_PERIOD) == DEFAULT_PERIOD

    def test_an_unset_variable_resolves_to_the_default_without_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Not setting a knob is not a misconfiguration, so it must stay quiet."""
        with caplog.at_level("WARNING", logger=isaac_module.__name__):
            assert _env_float(IDLE_PERIOD_ENV, DEFAULT_PERIOD) == DEFAULT_PERIOD
        assert [r.getMessage() for r in caplog.records] == []

    def test_the_unset_default_cadence_is_the_documented_one_second(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _refresh_timeline(monkeypatch) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


class TestTheIntResolverNeedsNoSuchBound:
    """The control: ``int()`` refuses every non-finite spelling itself."""

    @pytest.mark.parametrize("raw", ["inf", "Infinity", "1e999", "nan"])
    def test_the_int_resolver_already_falls_back(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv("SO101_IDLE_CONVERGE", raw)
        assert isaac_module._env_int("SO101_IDLE_CONVERGE", 4) == 4
