"""``mesh.pacing.Ticker`` must be accurate, stoppable, and honest about both.

These tests assert the two properties the publish loops depend on:
the achieved rate is the requested rate even in a process tree where
``Event.wait`` is inflated by ~145ms, and a stop is honoured within a slice
rather than within a period.

Every timing assertion here is written to hold on BOTH kinds of machine -- a
terminal-started shell where sleeps are accurate, and a daemon-descended tree
where they are not -- by calibrating against
:func:`strands_robots.mesh.pacing.sleep_penalty_s` instead of picking a number
that happens to pass where it was written. A test that only passes in one of the
two environments is exactly the failure mode that hides this bug.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import selectors
import socket
import threading
import time
from typing import Any

import numpy as np
import pytest

import strands_robots
from strands_robots.mesh import pacing
from strands_robots.mesh.pacing import Ticker, sleep_penalty_s
from strands_robots.utils import positive_finite_number_error


class TestARefusedPeriodCannotBusySpinAHardwareLoop:
    @pytest.mark.parametrize("bad", [0, 0.0, -1, -0.001, float("nan"), float("inf")])
    def test_a_period_that_would_spin_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="period"):
            Ticker(bad)

    @pytest.mark.parametrize("bad", [0, -0.5, float("nan")])
    def test_a_slice_that_would_spin_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="slice_s"):
            Ticker(0.1, slice_s=bad)

    def test_a_slice_longer_than_the_period_is_clamped_not_rejected(self) -> None:
        # A 100Hz loop with the default 10ms slice must not wait 10ms per 10ms
        # tick and then some: the slice is capped at the period.
        with Ticker(0.005) as ticker:
            assert ticker.slice_s == pytest.approx(0.005)


class TestTheAchievedRateIsTheRequestedRate:
    def test_ten_hz_stays_ten_hz_where_event_wait_would_not(self) -> None:
        """The headline claim, measured against Event.wait on the same machine."""
        penalty = sleep_penalty_s()
        period = 0.05

        stop = threading.Event()
        ticks = 0
        with Ticker(period, stop) as ticker:
            start = time.perf_counter()
            while time.perf_counter() - start < 0.6:
                ticks += 1
                if ticker.wait():
                    break
            ticker_hz = ticks / (time.perf_counter() - start)

        # The floor is generous (75% of nominal) because CI machines are noisy,
        # but the point is the COMPARISON below, which is what a wrong pacing
        # implementation cannot satisfy.
        assert ticker_hz > 0.75 / period, f"Ticker only achieved {ticker_hz:.1f}Hz asking for {1 / period:.0f}Hz"

        if penalty < 0.01:
            pytest.skip(
                f"sleeps are accurate here (penalty {penalty * 1000:.1f}ms), so Event.wait pacing is "
                "not degraded and there is nothing to out-perform"
            )
        event_ticks = 0
        event_start = time.perf_counter()
        idle = threading.Event()
        while time.perf_counter() - event_start < 0.6:
            idle.wait(period)
            event_ticks += 1
        event_hz = event_ticks / (time.perf_counter() - event_start)
        assert ticker_hz > event_hz * 1.5, (
            f"Ticker {ticker_hz:.1f}Hz vs Event.wait {event_hz:.1f}Hz at a nominal {1 / period:.0f}Hz "
            f"(sleep penalty {penalty * 1000:.0f}ms) - the selector timer is supposed to sidestep the penalty"
        )

    def test_work_inside_the_tick_is_subtracted_from_the_period_not_added(self) -> None:
        """The period is a deadline: a 20ms tick body inside a 50ms period keeps 50ms."""
        period = 0.05
        elapsed: list[float] = []
        with Ticker(period) as ticker:
            for _ in range(6):
                start = time.perf_counter()
                busy_until = start + 0.02  # busy-wait: unaffected by the sleep penalty
                while time.perf_counter() < busy_until:
                    pass
                ticker.wait()
                elapsed.append(time.perf_counter() - start)
        mid = sorted(elapsed)[len(elapsed) // 2]
        assert mid < period * 1.6, (
            f"a 20ms body inside a {period * 1000:.0f}ms period produced {mid * 1000:.0f}ms ticks - "
            "the work is being ADDED to the period instead of subtracted from it"
        )

    def test_an_overrunning_tick_does_not_fire_a_catch_up_burst(self) -> None:
        """Missed deadlines are dropped, not chased.

        A publish loop that chases lost time emits several frames back to back
        with near-identical timestamps, which reads downstream as a rate spike
        rather than as the stall it actually was.
        """
        period = 0.02
        with Ticker(period) as ticker:
            ticker.wait()
            overrun_until = time.perf_counter() + 0.15  # 7+ periods
            while time.perf_counter() < overrun_until:
                pass
            waits = []
            for _ in range(3):
                start = time.perf_counter()
                ticker.wait()
                waits.append(time.perf_counter() - start)
        assert all(w > period * 0.4 for w in waits), (
            f"after a 150ms overrun the next waits returned immediately ({[round(w * 1000) for w in waits]}ms) - "
            "that is a catch-up burst"
        )


class TestAStopIsHonouredWithinASliceNotWithinAPeriod:
    def test_wait_returns_true_when_the_event_is_already_set(self) -> None:
        stop = threading.Event()
        stop.set()
        with Ticker(10.0, stop) as ticker:
            start = time.perf_counter()
            assert ticker.wait() is True
            assert time.perf_counter() - start < 1.0, "a set stop event must not wait out the period"

    def test_a_stop_set_mid_wait_is_seen_within_a_slice(self) -> None:
        stop = threading.Event()
        penalty = sleep_penalty_s()
        with Ticker(5.0, stop, slice_s=0.01) as ticker:

            def stopper() -> None:
                time.sleep(0.05)
                stop.set()
                ticker.wake()

            threading.Thread(target=stopper, daemon=True).start()
            start = time.perf_counter()
            assert ticker.wait() is True
            took = time.perf_counter() - start
        # The stopper's own sleep(0.05) is subject to the machine's penalty, so
        # the budget is 0.05 + penalty + a slice + slack -- never the 5s period.
        budget = 0.05 + penalty + 0.01 + 0.5
        assert took < budget, f"stop took {took:.3f}s (budget {budget:.3f}s) out of a 5s period"

    def test_a_spurious_wake_does_not_shorten_the_tick(self) -> None:
        """wake() without a stop must not turn a 50ms tick into a 5ms one.

        Otherwise any code that pokes the ticker to be helpful silently doubles
        the publish rate of a hardware loop.
        """
        stop = threading.Event()
        with Ticker(0.05, stop) as ticker:
            threading.Thread(target=ticker.wake, daemon=True).start()
            start = time.perf_counter()
            assert ticker.wait() is False
            assert time.perf_counter() - start > 0.04, "a wake() with no stop cut the tick short"

    def test_wake_and_close_are_safe_in_any_order_and_repeatedly(self) -> None:
        ticker = Ticker(0.01, threading.Event())
        ticker.wake()
        ticker.close()
        ticker.close()  # idempotent
        ticker.wake()  # after close: a no-op, never an exception on a shutdown path
        with pytest.raises(RuntimeError, match="after close"):
            ticker.wait()


class TestTheCalibrationHelperIsUsableByOtherTests:
    def test_it_reports_a_non_negative_extra_cost(self) -> None:
        assert sleep_penalty_s(0.01) >= 0.0

    def test_it_refuses_a_sample_count_it_cannot_take_a_median_of(self) -> None:
        with pytest.raises(ValueError, match="samples"):
            sleep_penalty_s(0.01, samples=0)

    def test_one_sample_is_not_the_default_because_the_first_sleep_can_be_accurate(self) -> None:
        """Regression for a flake I created and measured.

        The first ``time.sleep`` after CPU-bound work came back 3.3ms late on this
        machine while every following one was ~145ms late, so a one-shot probe
        says "sleeps are fine here" often enough to silently skip the comparison
        this module exists to make. The default must therefore be > 1 sample.
        """
        import inspect

        default = inspect.signature(sleep_penalty_s).parameters["samples"].default
        assert default >= 3, "a single-sample calibration is flaky in the direction that hides the bug"

    def test_it_measures_the_extra_not_the_total(self) -> None:
        # A 10ms sleep on an accurate machine is ~0 extra; the total would be
        # ~0.01. Anything that returns the total is broken in the direction that
        # makes every calibrated ceiling too generous.
        assert sleep_penalty_s(0.01) < 1.0


#: Values that do not name a cadence, one per way of failing to be one. Only
#: ``True``, ``"0.05"``, ``None``, a list and a value past the float64 range were
#: new here: the rest a bare positivity test already refused. The point is that
#: this list is not maintained here at all -- it is the same set
#: ``tests/test_wait_budget_domain.py`` holds for the bridges' ``poll_period``,
#: because it is the same quantity.
NOT_A_CADENCE: list[Any] = [
    0,
    0.0,
    -1,
    -0.001,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,  # int subclass: a bare positivity test admits it as a 1-second period
    False,
    "0.05",  # a numeric string is not a real number
    None,
    [0.05],
    10**400,  # positive and finite, but past the float64 range
]

#: Values that do name a cadence, spanning the fractional and whole cases.
A_CADENCE: list[float] = [0.001, 0.02, 0.5, 1, 2.5]


class TestThePeriodIsHeldToTheSharedDomain:
    """One domain decides which values pace a loop, wherever the value enters.

    A period reaching this constructor has usually already been validated by the
    surface that produced it -- ``HardwareRtpsBridge.poll_period``, or an ``hz``
    a loop inverts -- against
    :func:`~strands_robots.utils.positive_finite_number_error`. The check written
    here instead was narrower than the values that reach it, and narrower in the
    direction that pages nobody: ``True`` became a silent one-second period on a
    camera loop, and ``None`` or ``10**400`` got a bare ``TypeError`` /
    ``OverflowError`` out of the conversion rather than the ``ValueError`` this
    constructor documents.
    """

    @pytest.mark.parametrize("value", NOT_A_CADENCE, ids=repr)
    def test_a_period_that_names_no_cadence_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="period"):
            Ticker(value)

    @pytest.mark.parametrize("value", NOT_A_CADENCE, ids=repr)
    def test_a_slice_that_names_no_cadence_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="slice_s"):
            Ticker(0.1, slice_s=value)

    @pytest.mark.parametrize("value", NOT_A_CADENCE, ids=repr)
    def test_the_verdict_is_the_shared_domains_verdict(self, value: Any) -> None:
        """Parity, so the two cannot come to disagree about one quantity.

        Asserted as an equivalence rather than as two lists: a value this
        constructor refuses and the domain accepts would be just as much a
        divergence as the one that was here.
        """
        assert positive_finite_number_error(value, "period", "Ticker") is not None
        with pytest.raises(ValueError):
            Ticker(value)

    @pytest.mark.parametrize("value", A_CADENCE, ids=repr)
    def test_a_period_that_paces_a_loop_is_still_accepted(self, value: float) -> None:
        assert positive_finite_number_error(value, "period", "Ticker") is None
        with Ticker(value) as ticker:
            assert ticker.period == pytest.approx(float(value))

    def test_a_numpy_scalar_is_accepted_and_stored_as_a_builtin_float(self) -> None:
        """Why the conversion has to happen after the guard, not before it.

        The shared domain accepts any real scalar, so a ``np.float32`` read from
        a config array is a usable period -- but it is not what
        :meth:`selectors.BaseSelector.select` can be handed, so the value has to
        be converted once it is known to be usable.
        """
        # Bound through ``Any`` for the same reason the bridges' own numpy case
        # is: ``period`` is annotated ``float``, and the subject here is what the
        # runtime does with a scalar the shared domain accepts, which is not what
        # the annotation describes.
        value: Any = np.float32(0.02)
        assert positive_finite_number_error(value, "period", "Ticker") is None
        with Ticker(value) as ticker:
            assert type(ticker.period) is float
            assert type(ticker.slice_s) is float

    def test_the_refusal_still_says_what_an_unusable_period_would_do(self) -> None:
        """The domain names the value; this constructor names the consequence."""
        with pytest.raises(ValueError, match="busy-spins the loop") as refused:
            Ticker(0)
        assert "period must be" in str(refused.value)


def _package_root() -> pathlib.Path:
    """The installed package directory, derived from an imported symbol."""
    return pathlib.Path(inspect.getfile(strands_robots)).parent


def _ticker_constructions(source: str) -> list[tuple[int, bool]]:
    """``(lineno, acquired_with_with)`` for every ``Ticker(...)`` in ``source``."""
    tree = ast.parse(source)
    acquired = {
        id(item.context_expr)
        for node in ast.walk(tree)
        if isinstance(node, ast.With | ast.AsyncWith)
        for item in node.items
    }
    return [
        (node.lineno, id(node) in acquired)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Ticker"
    ]


class TestEveryPacedLoopAcquiresItsTickerWithWith:
    """A ticker owns a selector and a self-pipe, so its release is structural.

    Every one of these loops first spelled the release as ``try: ... finally:
    ticker.close()``, which is correct exactly as long as each of the seven
    writes it -- and this module exists because that is the kind of rule six
    loops get right and the seventh does not. ``with`` moves it from a
    discipline to the language, and it is checked here rather than left to
    review because the loops live in five files.
    """

    def test_no_paced_loop_hand_rolls_the_release(self) -> None:
        adrift = [
            f"{path.relative_to(_package_root())}:{lineno}"
            for path in sorted(_package_root().rglob("*.py"))
            for lineno, with_stmt in _ticker_constructions(path.read_text())
            if not with_stmt
        ]
        assert not adrift, (
            "these tickers are constructed outside a ``with``, so their selector and "
            f"self-pipe are released only if the loop remembers to: {adrift}"
        )

    def test_the_scan_finds_every_paced_loop(self) -> None:
        """Non-vacuity: a scan that reached nothing would report a clean sweep."""
        found = sum(len(_ticker_constructions(path.read_text())) for path in sorted(_package_root().rglob("*.py")))
        assert found >= 7, f"expected the mesh, teleop and RTPS publish loops, found {found}"

    def test_the_scan_reports_a_hand_rolled_release(self) -> None:
        """A scanner that matched nothing would pass the sweep vacuously."""
        planted = (
            "def loop(self):\n"
            "    ticker = Ticker(0.02, self._stop)\n"
            "    try:\n"
            "        pass\n"
            "    finally:\n"
            "        ticker.close()\n"
        )
        assert _ticker_constructions(planted) == [(2, False)]

    def test_the_scan_accepts_an_acquired_ticker(self) -> None:
        planted = "def loop(self):\n    with Ticker(0.02, self._stop) as ticker:\n        pass\n"
        assert _ticker_constructions(planted) == [(2, True)]

    def test_the_documented_usage_shows_the_shape_the_loops_use(self) -> None:
        """The class docstring taught the hand-rolled release it was flagged for.

        A reader following the example wrote the shape this guard bans, which is
        how it arrived at seven call sites at once.
        """
        example = Ticker.__doc__ or ""
        assert "with Ticker(period, stop_event) as ticker:" in example
        assert "finally:" not in example


class TestTheDoorbellIsSomethingEverySelectorAccepts:
    """The wake object has to be a socket, or Windows loses every paced thread.

    ``selectors.DefaultSelector`` is ``SelectSelector`` on Windows, and its
    WinSock ``select()`` accepts sockets only: an ``os.pipe()`` fd raises
    ``OSError`` (WSAENOTSOCK, 10038) out of the first :meth:`Ticker.wait`. That
    call sits outside the ``try`` in every paced loop, so each publish thread
    would die on its first tick while the mesh still looked up -- a robot that
    joins the fleet and streams nothing.

    That refusal cannot be reproduced on a POSIX host, where ``select()`` takes
    any descriptor, so these pin the mechanism that makes the Windows selector
    accept the doorbell rather than the Windows symptom. ``asyncio`` builds its
    own self-pipe from a socketpair for this reason, which is also what makes
    this module's claim to wait on the same primitive true.
    """

    def test_both_ends_of_the_doorbell_are_sockets(self) -> None:
        with Ticker(0.01) as ticker:
            assert isinstance(ticker._wake_r, socket.socket)
            assert isinstance(ticker._wake_w, socket.socket)

    def test_the_module_does_not_build_the_doorbell_from_a_pipe(self) -> None:
        """A source guard, because the POSIX-only shape passes every test here."""
        source = pathlib.Path(inspect.getfile(pacing)).read_text()
        piped = [
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and ast.unparse(node.func) in {"os.pipe", "pipe"}
        ]
        assert not piped, (
            f"the doorbell is built from a pipe at {piped}: WinSock select() takes only sockets, "
            "so every paced loop's thread dies on its first wait() under Windows"
        )

    def test_the_doorbell_is_accepted_by_the_selector_windows_resolves_to(self) -> None:
        """``SelectSelector`` is what ``DefaultSelector`` is on Windows.

        Exercised directly rather than through ``DefaultSelector``, which is
        epoll here. On POSIX this passes for a pipe too -- what it establishes is
        that the doorbell is an object ``select()`` takes, which is the whole of
        what Windows refuses about the pipe.
        """
        with Ticker(5.0, threading.Event()) as ticker:
            with selectors.SelectSelector() as selector:
                selector.register(ticker._wake_r, selectors.EVENT_READ)
                assert selector.select(0) == [], "no wake has been sent yet"
                ticker.wake()
                assert selector.select(1.0), "a wake must be visible to a select()-based selector"

    def test_a_wake_still_survives_a_drain_and_a_close(self) -> None:
        """The socket swap must not change what the doorbell guarantees."""
        stop = threading.Event()
        ticker = Ticker(0.01, stop)
        ticker.wake()
        ticker._drain()
        ticker.wake()
        ticker.close()
        ticker.wake()  # after close: a no-op on a shutdown path, never an exception
        ticker.close()  # idempotent
