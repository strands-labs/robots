# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac's preview cadence and reported recording duration survive a clock step.

Two durations in the Isaac backend were measured by subtracting two readings of
``time.time()``, which is not a clock but the current opinion about the date - an
NTP correction, a ``date -s``, a resume from suspend moves it by an arbitrary
amount - so a step landing between the readings changed the answer:

**The idle-render gate** in :meth:`IsaacSimulation.run_pump_forever` decides,
once per iteration, whether to refresh the live preview. Driving the real loop at
a 0.2 s period through a clock double that steps once, with the loop's own
``sleep(0.05)`` setting the granularity:

    clock event              refreshes   gaps (s)   last refresh
    no step (control)               10   0.20       t = 1.80
    wall clock steps +30s           11   0.15, 0.20 t = 1.95
    wall clock steps -2s             3   0.20       t = 0.40
    wall clock steps +1h            11   0.15, 0.20 t = 1.95

Backward is the damaging direction: the preview stopped refreshing for the size
of the step - 1.55 s of that 1.95 s run - while ``pump()`` kept draining the app,
so the viewport sat frozen on a stale frame and nothing said so. Forward fired
the gate once early and then re-based onto the stepped clock, so it cost one
spurious refresh rather than a run of them. Fixed, all four cases produce the
control's timeline exactly: 10 refreshes, 0.20 s apart, the last at t = 1.80.

**The recording duration** ``stop_cameras_recording`` reports came from the same
clock as the base ``start_cameras_recording`` stamped, so a step anywhere in the
recording corrupted it. A 3.0 s capture reported ``after 33.0s``, ``after 1.0s``
and ``after 3603.0s`` for the same three steps.

The MuJoCo backend's two recorders already hold this base on
``time.monotonic()`` under the name ``started_mono``, and
``test_rendering_pacers_survive_a_clock_step.py`` pins that they carry no
``started_at`` key because "a duration base named as a wall-clock stamp invites
the next reader to subtract ``time.time()`` from it" - which is what Isaac's
third recorder did. These tests extend that contract to it.

This is the boundary the library settled for its agent-callable tools (#2404),
its hardware control loops (#2406), its mesh (#2408), its sim rollouts (#2413)
and its rendering pacers (#2425), and the rule AGENTS.md states as "a duration is
measured, a stamp is recorded", which names a frame pacer explicitly.

Every assertion is on the *achieved* refresh timeline or the *reported* duration
rather than on which clock is called, so none of them can be satisfied by
renaming a call. The clock double counts reads of either clock, so the step lands
at the same loop position whichever clock the loop consults. No Isaac Sim Kit
runtime, GL context or MP4 encoder is touched: the engine is the skeleton
``__new__`` fixture shape ``test_cameras_recording_preflight_guards.py`` uses and
``pump`` is faked on the instance.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import pytest

from strands_robots.simulation.isaac import simulation as isaac_module
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState

#: Idle-render period under test, and therefore the gap the gate must hold.
IDLE_PERIOD = 0.2
#: Idle iterations to run. Long enough that a backward step's suppression window
#: closes inside the run, so the timeline shows the recovery too.
PUMP_ITERATIONS = 40
#: Loop position the wall-clock step lands at: one read per iteration either
#: way, so this is the iteration index. Between the refresh due at 10 and at 15.
PUMP_STEP_AT_READ = 12
#: The loop's own idle sleep, and therefore the resolution the gate can hold the
#: period to: it refreshes on the first iteration at or past the period.
PUMP_GRANULARITY = 0.05
#: Simulated recording length, in seconds, for the reported-duration tests.
RECORDING_SECONDS = 3.0
#: Wall-clock steps exercised. Forward and backward, and one large enough that
#: the forward case has visibly saturated.
STEPS = (30.0, -2.0, 3600.0)
STEP_IDS = ["forward_30s", "backward_2s", "forward_1h"]


class _SteppingClock:
    """A ``time`` double whose wall clock steps once and whose monotonic does not.

    ``time()`` and ``monotonic()`` share a read counter, so ``step_after_reads``
    names a position in the loop rather than a particular clock: a loop reading
    ``time()`` once per iteration and a loop reading ``monotonic()`` once per
    iteration both take the step at the same iteration. That is what lets one
    stimulus reach the pre-fix and post-fix loops identically.

    Sleeps advance the virtual clock instead of blocking, so the achieved gap is
    exact and the suite stays fast. ``apply``ing the step moves only the wall
    clock, which is what an NTP correction does. Every other attribute delegates
    to the real :mod:`time`, so a module that imports this double for
    ``perf_counter`` or ``strftime`` is unaffected.
    """

    def __init__(self, wall_step: float, step_after_reads: int | None, *, start: float = 1_000_000.0) -> None:
        self._virtual = start
        self._wall_offset = 0.0
        self._wall_step = wall_step
        self._step_after_reads = step_after_reads
        self.reads = 0
        self.sleeps: list[float] = []
        self.step_applied_at_read: int | None = None

    def _read(self) -> None:
        self.reads += 1
        if self._step_after_reads is not None and self.reads == self._step_after_reads:
            self._wall_offset += self._wall_step
            self.step_applied_at_read = self.reads

    def time(self) -> float:
        self._read()
        return self._virtual + self._wall_offset

    def monotonic(self) -> float:
        self._read()
        return self._virtual

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if seconds > 0:
            self._virtual += seconds

    def now(self) -> float:
        """Read the virtual clock *without* counting it as a read.

        The tests stamp each refresh to recover the achieved timeline, and that
        stamp is the test's own instrumentation rather than something the loop
        does. Counting it would shift ``step_after_reads`` off the loop position
        it names.
        """
        return self._virtual

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _StopAfter:
    """A ``threading.Event``-shaped stop flag that ends the loop after ``n`` checks."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks > self.n


def _pump_engine(period: float = IDLE_PERIOD) -> Any:
    """Skeleton engine carrying only what ``run_pump_forever`` reads."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._idle_render_period = period
    engine._main_jobs = queue.Queue()
    engine._action_q = queue.Queue()
    engine._pump_running = False
    return engine


def _refresh_timeline(
    clock: _SteppingClock,
    monkeypatch: pytest.MonkeyPatch,
    *,
    period: float = IDLE_PERIOD,
    iterations: int = PUMP_ITERATIONS,
    queue_action: bool = False,
    main_job: Any = None,
) -> tuple[list[float], list[bool]]:
    """Drive the real ``run_pump_forever`` and return when the preview refreshed.

    Returns ``(refresh_times, render_flags)``: the virtual times at which
    ``pump(render=True)`` was called, and every ``render`` flag the loop passed,
    so a caller can assert on the refreshes *and* on the pumps that did not
    refresh.
    """
    monkeypatch.setattr(isaac_module, "time", clock)
    engine = _pump_engine(period)
    if queue_action:
        engine._action_q.put(object())
    if main_job is not None:
        engine._main_jobs.put(main_job)

    refreshes: list[float] = []
    flags: list[bool] = []

    def pump(render: bool = False) -> None:
        flags.append(bool(render))
        if render:
            refreshes.append(clock.now())

    engine.pump = pump
    IsaacSimulation.run_pump_forever(engine, stop_event=_StopAfter(iterations))
    assert engine._pump_running is False, "the loop must clear its running flag on the way out"
    return refreshes, flags


def _gaps(times: list[float]) -> list[float]:
    """Gaps between consecutive refreshes, rounded past float noise."""
    return [round(b - a, 6) for a, b in zip(times, times[1:], strict=False)]


# --------------------------------------------------------------------------- #
# 1. The idle-render gate
# --------------------------------------------------------------------------- #


def test_idle_refresh_cadence_without_a_step_is_the_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: with no step the gate holds one refresh per period.

    Passes before and after the clock swap. It is what makes the stepped cases
    below a statement about the step rather than about the loop's granularity,
    and it fails if the gate is ever made unconditional.
    """
    refreshes, flags = _refresh_timeline(_SteppingClock(0.0, step_after_reads=None), monkeypatch)

    gaps = _gaps(refreshes)
    assert refreshes, "premise: the loop never refreshed the preview at all"
    assert len(set(gaps)) == 1, f"the unstepped cadence must be uniform; achieved gaps {gaps}"
    # The gate fires on the first iteration at or past the period, and the loop
    # advances in 0.05s sleeps, so the achieved gap is the period rounded up to
    # that granularity.
    assert IDLE_PERIOD <= gaps[0] < IDLE_PERIOD + PUMP_GRANULARITY, (
        f"the gate must refresh once per {IDLE_PERIOD}s period at the loop's "
        f"{PUMP_GRANULARITY}s granularity; achieved gaps {gaps}"
    )
    assert len(flags) == PUMP_ITERATIONS, "every iteration pumps, refreshing or not"


@pytest.mark.parametrize("wall_step", STEPS, ids=STEP_IDS)
def test_idle_refresh_cadence_survives_a_wall_clock_step(wall_step: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wall-clock step must not change when the live preview is refreshed.

    The step lands between the refresh due at iteration 10 and the one due at 15.
    On ``time.time()`` a backward step of S suppressed every refresh for S -
    ``pump()`` kept draining the app, so the viewport sat frozen on a stale frame
    with nothing reporting it - while a forward step fired the gate once early.

    Asserted against the unstepped timeline rather than against a tolerance, so
    the contract is "a step changes nothing", which no rename can satisfy.
    """
    control, _ = _refresh_timeline(_SteppingClock(0.0, step_after_reads=None), monkeypatch)

    clock = _SteppingClock(wall_step, step_after_reads=PUMP_STEP_AT_READ)
    stepped, _ = _refresh_timeline(clock, monkeypatch)

    assert clock.step_applied_at_read is not None, "premise: the step never reached the loop"
    assert stepped == control, (
        f"a {wall_step:+.0f}s wall-clock step moved the preview refreshes: "
        f"{len(stepped)} refreshes at gaps {_gaps(stepped)} where the unstepped loop "
        f"made {len(control)} at gaps {_gaps(control)}"
    )


def test_the_first_idle_iteration_refreshes_whatever_the_clock_epoch_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preview comes up on the first idle iteration, not one period later.

    ``time.monotonic()``'s reference point is unspecified - seconds since boot on
    Linux - so a numeric "never refreshed" sentinel would suppress the first
    refresh for the first ``_idle_render_period`` seconds of uptime. The sentinel
    is therefore ``None``, which says what it means whatever the clock reads.
    """
    clock = _SteppingClock(0.0, step_after_reads=None, start=IDLE_PERIOD / 4)

    refreshes, flags = _refresh_timeline(clock, monkeypatch, iterations=1)

    assert flags == [True], (
        f"the first idle iteration must refresh the preview even when the clock's "
        f"epoch reads {IDLE_PERIOD / 4}s, below the {IDLE_PERIOD}s period; pumped {flags}"
    )
    assert refreshes == [IDLE_PERIOD / 4]


def test_a_busy_pump_never_refreshes_the_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued worker action runs the episode at full speed: pump, never render.

    Passes before and after the clock swap. It pins that the gate was not made
    unconditional, and it is the branch that never consults a clock at all.
    """
    clock = _SteppingClock(0.0, step_after_reads=None)

    _, flags = _refresh_timeline(clock, monkeypatch, iterations=6, queue_action=True)

    assert flags == [False] * 6, f"a busy pump must not spend time rendering; pumped {flags}"
    assert clock.reads == 0, "the busy branch reads no clock"


def test_a_main_job_forces_the_next_idle_iteration_to_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole-job submission freezes the preview, so the next idle pass refreshes.

    Passes before and after the clock swap. It fails if the reset the job path
    performs is dropped, which would leave the preview showing the pre-job frame
    for up to one period after a record or plan finished.
    """
    ran: list[str] = []

    _, flags = _refresh_timeline(
        _SteppingClock(0.0, step_after_reads=None),
        monkeypatch,
        iterations=3,
        main_job=lambda: ran.append("job"),
    )

    assert ran == ["job"], "premise: the submitted job never ran"
    assert flags[0] is True, f"the idle iteration after a job must refresh the preview; pumped {flags}"


# --------------------------------------------------------------------------- #
# 2. The duration ``stop_cameras_recording`` reports
# --------------------------------------------------------------------------- #


def _recording_engine() -> Any:
    """Skeleton engine with a running camera recording and empty buffers.

    Empty buffers keep the flush away from ``encode_clip``, so no MP4 encoder or
    ffmpeg import runs under the clock double.
    """
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._cameras = {"front": _CameraState(name="front", prim_path="/World/Cameras/front", width=32, height=24)}
    engine._cams_rec_state = None
    return engine


def _record_and_report(
    clock: _SteppingClock, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, seconds: float = RECORDING_SECONDS
) -> str:
    """Start a recording, let ``seconds`` pass, and return the stop line.

    Both calls run with the double installed as the module a fresh
    ``import time`` resolves to, because each resolves ``time`` inside its own
    body.
    """
    # Bind the encoder module on the real clock: ``stop_cameras_recording``
    # imports it, and a module first imported under the double would keep
    # reading the double for the rest of the session.
    from strands_robots.rendering.video import encode_clip  # noqa: F401

    monkeypatch.setitem(sys.modules, "time", clock)
    engine = _recording_engine()
    started = engine.start_cameras_recording(cameras=["front"], output_dir=str(tmp_path), fps=10, name="cap")
    assert started["status"] == "success", started["content"][0]["text"]
    clock.sleep(seconds)
    stopped = engine.stop_cameras_recording()
    assert stopped["status"] == "success", stopped["content"][0]["text"]
    return str(stopped["content"][0]["text"]).splitlines()[0]


def test_reported_recording_duration_without_a_step_is_the_elapsed_time(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: with no step the reported duration is the time that passed."""
    line = _record_and_report(_SteppingClock(0.0, step_after_reads=None), monkeypatch, tmp_path)

    assert f"after {RECORDING_SECONDS:.1f}s" in line, line


@pytest.mark.parametrize("wall_step", STEPS, ids=STEP_IDS)
def test_reported_recording_duration_survives_a_wall_clock_step(
    wall_step: float, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step during the recording must not move the duration it reports.

    The base and the reading shared ``time.time()``, so a step anywhere in the
    recording - not only in some window - moved the answer. The buffers carry no
    per-frame timestamp, so the reported duration is the only thing that says how
    much wall time the frames span.
    """
    clock = _SteppingClock(wall_step, step_after_reads=2)

    line = _record_and_report(clock, monkeypatch, tmp_path)

    assert clock.step_applied_at_read is not None, "premise: the step never reached the recording"
    assert f"after {RECORDING_SECONDS:.1f}s" in line, (
        f"a {wall_step:+.0f}s wall-clock step moved the duration reported for a "
        f"{RECORDING_SECONDS:.1f}s recording: {line!r}"
    )


def test_the_recorder_names_the_clock_its_duration_base_holds(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isaac's recorder carries the same duration-base spelling as MuJoCo's two.

    ``test_rendering_pacers_survive_a_clock_step.py`` pins that the MuJoCo
    recorders' state carries ``started_mono`` and no ``started_at``, because a
    duration base named as a wall-clock stamp invites the next reader to subtract
    ``time.time()`` from it. This backend's recorder is a third one behind the
    same tool pair, so it holds the same contract.
    """
    monkeypatch.setitem(sys.modules, "time", _SteppingClock(0.0, step_after_reads=None))
    engine = _recording_engine()
    started = engine.start_cameras_recording(cameras=["front"], output_dir=str(tmp_path), fps=10, name="cap")
    assert started["status"] == "success", started["content"][0]["text"]

    keys = set(engine._cams_rec_state)

    assert "started_mono" in keys
    assert "started_at" not in keys, (
        "a duration base named as a wall-clock stamp invites the next reader to subtract time.time() from it"
    )
