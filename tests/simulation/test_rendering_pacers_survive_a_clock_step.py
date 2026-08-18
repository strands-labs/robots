# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The rendering pacers' rate and reported duration survive a wall-clock step.

Two rendering loops pace themselves by reading a base, doing the frame's work,
and sleeping for whatever is left of the interval: the MJPEG stream generator
that feeds a browser ``<img src=...>`` view, and the multi-camera recorder
thread that fills the buffers ``stop_cameras_recording`` flushes to one MP4 per
camera. Both bases used to be ``time.time()``, which is not a clock but the
current opinion about the date - an NTP correction, a ``date -s``, a resume from
suspend moves it by an arbitrary amount - and the sleep was computed from two
readings of it, so a step landing between them changed the rate:

    MJPEG stream, 12 fps (83.3 ms nominal), step inside one frame's window

    clock event                achieved interval for that frame
    no step (control)                        83.3 ms
    wall clock steps +30s                     4.0 ms   (pacing skipped)
    wall clock steps -2s                   2083.3 ms   (one stall)
    wall clock steps +1h                      4.0 ms   (saturates: sleep skipped)

The recorder is the same shape with a dataset rather than a viewer downstream. A
real 2-camera 10 fps capture over 2.85 s of simulated motion, with the step
landing in the render window, wrote **9 frames instead of 29** - and because the
buffers carry no per-frame timestamp, the MP4 that came out declared a 0.9 s
duration for that 2.85 s of motion. Nothing raised; the call returned
``status="success"``.

How often the step lands in that window is the render's share of the cycle,
measured on this recorder:

    recorder configuration            render/cycle   interval   vulnerable
    2 cams @ 160x120, 10 fps                6.4 ms   100.0 ms          6%
    2 cams @ 320x240, 30 fps               17.3 ms    33.3 ms         52%
    4 cams @ 640x480, 30 fps               97.8 ms    33.3 ms        100%

so the configuration ``start_cameras_recording``'s own docstring sizes ("a 2s /
4-cam / 320x240 / 15fps rollout") is squarely in the range where the loop is
render-bound and any step lands where it changes what was captured.

The duration each recorder *reports* shared the clock and is corrupted whenever
a step lands anywhere in the recording, not only in the render window: the same
3.01 s capture reported ``Stopped 'fwd30' after 33.4s`` and ``after 1.4s``.

These tests pin the contract on behaviour rather than on which clock is called.
The clock double counts reads of *either* clock, so the step lands at the same
position in the loop whichever clock the loop consults, and each assertion is on
the *achieved* interval - the pacing the viewer or the dataset actually got -
rather than on the value of the computed sleep, so it cannot be satisfied by
renaming a call.

This is the boundary the library settled for its agent-callable tools (#2404),
its hardware control loops (#2406), its mesh (#2408) and its sim rollouts
(#2413), and the rule AGENTS.md states as "a duration is measured, a stamp is
recorded", which names a frame pacer explicitly. No GL context, dataset or robot
is touched: ``render`` is faked on the instance, as in
``test_daemon_camera_recording.py``.
"""

from __future__ import annotations

import functools
import io
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed - pip install strands-robots[sim-mujoco]")
imageio = pytest.importorskip("imageio", reason="imageio not installed - pip install imageio imageio-ffmpeg")

from strands_robots.rendering import video as video_module  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402

#: Stream rate under test, and therefore the interval the pacer must hold.
MJPEG_FPS = 12.0
#: Frames the stream emits. Enough that the step lands mid-stream with frames
#: on either side of it.
MJPEG_FRAMES = 20
#: Frame index whose ``base .. elapsed`` window the wall-clock step lands in.
MJPEG_STEP_AT_FRAME = 10
#: Simulated cost of one ``frame_fn()`` + JPEG encode, in seconds. Well under
#: one interval, so a loop that paces correctly always sleeps.
MJPEG_WORK = 0.004

#: Capture rate under test for the multi-camera recorder.
RECORDER_FPS = 10
#: Simulated cost of rendering one camera, in seconds.
RECORDER_WORK = 0.006
#: Recorder cycle whose render window the wall-clock step lands in.
RECORDER_STEP_AT_CYCLE = 6
#: Frames to capture per camera before stopping.
RECORDER_FRAMES = 16

#: Wall-clock steps exercised. Forward and backward, and one large enough that
#: the forward case has visibly saturated.
STEPS = (30.0, -2.0, 3600.0)


class _SteppingClock:
    """A ``time`` double whose wall clock steps once and whose monotonic does not.

    ``time()`` and ``monotonic()`` share a read counter, so ``step_after_reads``
    names a position in the loop rather than a particular clock: a loop reading
    ``time()`` twice per cycle and a loop reading ``monotonic()`` twice per cycle
    both take the step at the same cycle. That is what lets one stimulus reach
    the pre-fix and post-fix loops identically.

    Sleeps are recorded and advance the virtual clock instead of blocking, so the
    achieved interval is exact and the suite stays fast. ``apply_step`` moves only
    the wall clock, which is what an NTP correction does. Every other attribute
    delegates to the real :mod:`time`, so a module that imports this double for
    ``perf_counter`` or ``strftime`` is unaffected.
    """

    def __init__(self, wall_step: float, step_after_reads: int | None) -> None:
        self._virtual = 1_000_000.0
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

    def advance(self, seconds: float) -> None:
        """Advance both clocks, standing in for work the loop did."""
        self._virtual += seconds

    def now(self) -> float:
        """Read the virtual clock *without* counting it as a read.

        The tests stamp each frame to recover the achieved interval, and that
        stamp is the test's own instrumentation rather than something the loop
        does. Counting it would shift ``step_after_reads`` off the loop position
        it names.
        """
        return self._virtual

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 ndarray as PNG bytes (render()'s wire format)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _gradient(width: int = 32, height: int = 24) -> np.ndarray:
    """A frame ``_extract_frame_ndarray`` keeps and the recorder calls warm.

    The recorder's cold-context warmup thresholds on *per-column* std-dev, so
    the frame has to vary down each column as well as across each row - a
    gradient that only varies along the width scores 0 and leaves the warmup
    burning its full 30-attempt budget before the capture loop starts.
    """
    cols = np.linspace(0, 200, width)
    rows = np.linspace(0, 55, height)
    arr = (rows[:, None] + cols[None, :]).astype(np.uint8)
    return np.stack([arr, arr[::-1], arr[:, ::-1]], axis=-1).astype(np.uint8)


def _achieved_intervals(stamps: list[float]) -> np.ndarray:
    """Intervals between consecutive frames, in milliseconds."""
    return np.diff(np.asarray(stamps)) * 1000.0


@functools.cache
def _warm_media_stack() -> None:
    """Import everything the recording path imports lazily, on the real clock.

    ``stop_cameras_recording`` encodes each buffer to an MP4, and that pulls in
    ``imageio``'s ffmpeg plugin, ``imageio_ffmpeg``, ``subprocess``, ``logging``
    and ``zipfile`` the first time it runs. A module first imported while a
    ``time`` double is installed in ``sys.modules`` binds the double for the rest
    of the session, so the ffmpeg writer would keep reading a stepped clock long
    after this module's tests finished and a later test writing an MP4 would fail
    on unreadable metadata. Doing that first import here, before any double is
    installed, is what keeps the double confined to the loop under test.
    """
    from strands_robots.rendering.video import encode_clip

    frames = [_gradient() for _ in range(3)]
    with tempfile.TemporaryDirectory() as scratch:
        clip = encode_clip(frames, Path(scratch) / "warm.mp4", fps=RECORDER_FPS)
        assert len(list(iio.imiter(clip))) == len(frames)


def _install_time_double(monkeypatch: pytest.MonkeyPatch, clock: _SteppingClock) -> None:
    """Make ``clock`` the module a fresh ``import time`` resolves to."""
    _warm_media_stack()
    monkeypatch.setitem(sys.modules, "time", clock)


def _assert_double_did_not_leak(clock: _SteppingClock) -> None:
    """No module may be left holding the double once the recording is over.

    Guards the confinement ``_warm_media_stack`` buys: if the recording path
    grows a new lazy import, this fails here rather than as unreadable MP4
    metadata in whichever unrelated test happens to run next.
    """
    leaked = []
    for name, module in list(sys.modules.items()):
        if module is None or name == "time":
            continue
        try:
            if getattr(module, "time", None) is clock:
                leaked.append(name)
        except Exception:  # noqa: BLE001 - a module whose attribute access raises is not a holder
            continue
    assert not leaked, f"the clock double was bound by {sorted(leaked)} and would outlive this test"


# --------------------------------------------------------------------------- #
# 1. MJPEG stream pacer
# --------------------------------------------------------------------------- #


def _run_mjpeg(clock: _SteppingClock, monkeypatch: pytest.MonkeyPatch) -> tuple[int, np.ndarray]:
    """Drive the real ``mjpeg_frames`` generator through ``clock``."""
    monkeypatch.setattr(video_module, "time", clock)
    stamps: list[float] = []

    def frame_fn() -> np.ndarray:
        clock.advance(MJPEG_WORK)
        stamps.append(clock.now())
        return _gradient(8, 8)

    emitted = sum(1 for _ in video_module.mjpeg_frames(frame_fn, fps=MJPEG_FPS, max_frames=MJPEG_FRAMES))
    return emitted, _achieved_intervals(stamps)


@pytest.mark.parametrize("wall_step", STEPS, ids=["forward_30s", "backward_2s", "forward_1h"])
def test_mjpeg_pacing_survives_a_wall_clock_step(wall_step: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wall-clock step must not change how long the client waits for a frame.

    The step lands inside one frame's ``base .. elapsed`` window - two clock
    reads per frame, so the second read of frame ``MJPEG_STEP_AT_FRAME``. On
    ``time.time()`` that frame was emitted after ``MJPEG_WORK`` with the pacing
    skipped (forward) or held for ``interval + |step|`` (backward).
    """
    nominal_ms = 1000.0 / MJPEG_FPS
    # The generator's own frame_fn() read is not a clock read, so the loop takes
    # exactly two: the base and the elapsed. Land on the elapsed read, or the
    # step moves the base as well and both sides of the subtraction shift alike.
    clock = _SteppingClock(wall_step, step_after_reads=2 * MJPEG_STEP_AT_FRAME + 2)
    emitted, intervals = _run_mjpeg(clock, monkeypatch)

    assert emitted == MJPEG_FRAMES
    assert clock.step_applied_at_read is not None, "premise: the step never reached the loop"
    assert intervals.max() == pytest.approx(nominal_ms, abs=1.0), (
        f"a {wall_step:+.0f}s wall-clock step held one frame for {intervals.max():.1f} ms "
        f"against a {nominal_ms:.1f} ms stream rate"
    )
    assert intervals.min() == pytest.approx(nominal_ms, abs=1.0), (
        f"a {wall_step:+.0f}s wall-clock step emitted one frame after {intervals.min():.1f} ms "
        f"against a {nominal_ms:.1f} ms stream rate - the pacing was skipped"
    )


def test_mjpeg_pacing_without_a_step_is_the_nominal_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the pacer holds the stream rate when the clock behaves.

    Fixes the reference the stepped runs are compared against, so a change that
    broke the pacing outright could not pass the tests above by making every
    interval equally wrong.
    """
    nominal_ms = 1000.0 / MJPEG_FPS
    clock = _SteppingClock(0.0, step_after_reads=None)
    emitted, intervals = _run_mjpeg(clock, monkeypatch)

    assert emitted == MJPEG_FRAMES
    assert intervals.min() == pytest.approx(nominal_ms, abs=1.0)
    assert intervals.max() == pytest.approx(nominal_ms, abs=1.0)


def test_mjpeg_pacing_ignores_a_step_taken_while_it_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step outside the ``base .. elapsed`` window was always harmless.

    Passes before and after the change: it bounds the fix to the window that was
    actually vulnerable rather than claiming every step mispaced a frame.
    """
    nominal_ms = 1000.0 / MJPEG_FPS
    # An odd read index is a frame's base: the later elapsed read carries the
    # same offset, so the subtraction is unaffected even on the wall clock.
    clock = _SteppingClock(-2.0, step_after_reads=2 * MJPEG_STEP_AT_FRAME + 1)
    emitted, intervals = _run_mjpeg(clock, monkeypatch)

    assert emitted == MJPEG_FRAMES
    assert intervals.max() == pytest.approx(nominal_ms, abs=1.0)


# --------------------------------------------------------------------------- #
# 2. Multi-camera recorder
# --------------------------------------------------------------------------- #


def _sim_with_fake_render(clock: _SteppingClock, stamps: list[float]) -> Simulation:
    """Real ``Simulation`` with two cameras and a GL-free ``render`` that pays
    ``RECORDER_WORK`` of virtual time per camera and stamps the monotonic clock."""
    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", data_config="so101", position=[0.0, 0.0, 0.0])
    sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
    sim.add_camera("cam_b", position=[0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])

    payload = _png_bytes(_gradient())
    assert float(_gradient().std(axis=0).mean()) > 5.0, "premise: the frame must read as warm"

    def _fake_render(camera_name: str, width: int | None = None, height: int | None = None, **_kw: Any):
        clock.advance(RECORDER_WORK)
        # The recorder sets ``ready`` between its warmup and its capture loop, so
        # this excludes the warmup renders from the measured intervals without
        # having to guess how many of them there were.
        st = getattr(sim, "_cams_rec_state", None)
        if camera_name == "cam_a" and st is not None and st["ready"].is_set():
            stamps.append(clock.now())
        return {
            "status": "success",
            "content": [{"text": "32x24"}, {"image": {"format": "png", "source": {"bytes": payload}}}],
        }

    sim.render = _fake_render  # type: ignore[assignment,method-assign]
    return sim


def _capture(
    clock: _SteppingClock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the real recorder thread through ``clock`` until the cap is reached."""
    stamps: list[float] = []
    sim = _sim_with_fake_render(clock, stamps)
    # ``start_cameras_recording`` binds its clock with a function-local
    # ``import time``, so the double is installed in ``sys.modules`` rather than
    # patched onto a module attribute. The recorder thread is the only reader.
    _install_time_double(monkeypatch, clock)
    try:
        started = sim.start_cameras_recording(
            cameras=["cam_a", "cam_b"],
            output_dir=str(tmp_path),
            fps=RECORDER_FPS,
            width=32,
            height=24,
            name="pacing",
            max_frames_per_camera=RECORDER_FRAMES,
        )
        assert started["status"] == "success", started
        state = sim._cams_rec_state
        deadline = time.monotonic() + 20.0  # real clock: this module's `time`
        while len(state["buffers"]["cam_a"]) < RECORDER_FRAMES and time.monotonic() < deadline:
            time.sleep(0.01)
        stopped = sim.stop_cameras_recording()
    finally:
        sim.cleanup()
    assert stopped["status"] == "success", stopped
    assert len(stamps) >= RECORDER_FRAMES // 2, f"premise: only {len(stamps)} frames captured"
    _assert_double_did_not_leak(clock)
    return _achieved_intervals(stamps), stopped


@pytest.mark.parametrize("wall_step", STEPS, ids=["forward_30s", "backward_2s", "forward_1h"])
def test_recorder_capture_rate_survives_a_wall_clock_step(
    wall_step: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wall-clock step must not change how much of the rollout a buffer sampled.

    The recorder reads its base, renders every camera, then sleeps for the rest
    of the interval. On ``time.time()`` a step landing in the render window made
    that cycle capture at a rate the caller never asked for - and the buffered
    frames carry no per-frame timestamp, so the MP4 that comes out is
    indistinguishable afterwards from one paced correctly.
    """
    nominal_ms = 1000.0 / RECORDER_FPS
    # The state dict's duration base is read once before the loop, so the loop's
    # two reads per cycle land on odd/even from read 2: cycle k's second read is
    # ``3 + 2k``.
    clock = _SteppingClock(wall_step, step_after_reads=3 + 2 * RECORDER_STEP_AT_CYCLE)
    intervals, _ = _capture(clock, tmp_path, monkeypatch)

    assert clock.step_applied_at_read is not None, "premise: the step never reached the loop"
    assert intervals.max() == pytest.approx(nominal_ms, abs=2.0), (
        f"a {wall_step:+.0f}s wall-clock step stalled the capture for {intervals.max():.1f} ms "
        f"against a {nominal_ms:.1f} ms period, so the buffer skipped "
        f"{intervals.max() / nominal_ms - 1:.0f} frames of the rollout"
    )
    assert intervals.min() == pytest.approx(nominal_ms, abs=2.0), (
        f"a {wall_step:+.0f}s wall-clock step captured a frame after {intervals.min():.1f} ms "
        f"against a {nominal_ms:.1f} ms period - the pacing was skipped"
    )


def test_recorder_capture_rate_without_a_step_is_the_nominal_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the recorder captures at the requested rate when the clock behaves."""
    nominal_ms = 1000.0 / RECORDER_FPS
    clock = _SteppingClock(0.0, step_after_reads=None)
    intervals, _ = _capture(clock, tmp_path, monkeypatch)

    assert intervals.min() == pytest.approx(nominal_ms, abs=2.0)
    assert intervals.max() == pytest.approx(nominal_ms, abs=2.0)


# --------------------------------------------------------------------------- #
# 3. The duration each recorder reports
# --------------------------------------------------------------------------- #


def _sync_recording(clock: _SteppingClock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Simulation:
    """A synchronous-mode recording, whose only clock reading is its duration base.

    Synchronous mode spawns no thread, so it isolates the reported duration from
    the pacing: the base is read once at start and subtracted once at stop.
    """
    sim = _sim_with_fake_render(clock, [])
    _install_time_double(monkeypatch, clock)
    started = sim.start_cameras_recording_synchronous(
        cameras=["cam_a"],
        output_dir=str(tmp_path),
        fps=RECORDER_FPS,
        width=32,
        height=24,
        name="reported",
    )
    assert started["status"] == "success", started
    return sim


@pytest.mark.parametrize("wall_step", STEPS, ids=["forward_30s", "backward_2s", "forward_1h"])
def test_stop_reports_the_duration_that_actually_elapsed(
    wall_step: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stop_cameras_recording``'s elapsed line must be the time that elapsed.

    Unlike the pacing, this is corrupted by a step landing anywhere in the
    recording: the base is read once at the start, so every later step is still
    in the subtraction at stop.
    """
    clock = _SteppingClock(wall_step, step_after_reads=2)
    sim = _sync_recording(clock, tmp_path, monkeypatch)
    try:
        clock.advance(4.0)  # four seconds of recording
        stopped = sim.stop_cameras_recording()
    finally:
        sim.cleanup()

    _assert_double_did_not_leak(clock)
    assert stopped["status"] == "success", stopped
    assert clock.step_applied_at_read is not None, "premise: the step never reached the recording"
    text = stopped["content"][0]["text"]
    assert "after 4.0s" in text, (
        f"a {wall_step:+.0f}s wall-clock step moved the reported duration of a 4.0s recording: {text.splitlines()[0]!r}"
    )


@pytest.mark.parametrize("wall_step", STEPS, ids=["forward_30s", "backward_2s", "forward_1h"])
def test_status_reports_the_duration_that_actually_elapsed(
    wall_step: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_cameras_recording_status`` reads the same base and must agree."""
    clock = _SteppingClock(wall_step, step_after_reads=2)
    sim = _sync_recording(clock, tmp_path, monkeypatch)
    try:
        clock.advance(4.0)
        status = sim.get_cameras_recording_status()
    finally:
        sim.stop_cameras_recording()
        sim.cleanup()

    _assert_double_did_not_leak(clock)
    assert clock.step_applied_at_read is not None, "premise: the step never reached the recording"
    text = status["content"][0]["text"]
    assert "for 4.0s" in text, (
        f"a {wall_step:+.0f}s wall-clock step moved the elapsed time reported for an "
        f"ongoing 4.0s recording: {text.splitlines()[0]!r}"
    )


def test_both_recorders_name_the_clock_their_duration_base_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two recorders keep one spelling for their duration base.

    ``stop_cameras_recording`` and ``get_cameras_recording_status`` are shared by
    the daemon-thread and synchronous recorders, so a base named for one clock in
    one of them and the other clock in the other would leave a reader of the
    shared code unable to tell which it is holding.
    """
    clock = _SteppingClock(0.0, step_after_reads=None)
    stamps: list[float] = []
    sim = _sim_with_fake_render(clock, stamps)
    _install_time_double(monkeypatch, clock)
    try:
        sim.start_cameras_recording_synchronous(
            cameras=["cam_a"], output_dir=str(tmp_path), fps=RECORDER_FPS, width=32, height=24, name="sync"
        )
        sync_keys = set(sim._cams_rec_state)
        sim.stop_cameras_recording()

        sim.start_cameras_recording(
            cameras=["cam_a"],
            output_dir=str(tmp_path),
            fps=RECORDER_FPS,
            width=32,
            height=24,
            name="daemon",
            max_frames_per_camera=2,
        )
        daemon_keys = set(sim._cams_rec_state)
        sim.stop_cameras_recording()
    finally:
        sim.cleanup()

    assert "started_mono" in sync_keys
    assert "started_mono" in daemon_keys
    assert "started_at" not in sync_keys | daemon_keys, (
        "a duration base named as a wall-clock stamp invites the next reader to subtract time.time() from it"
    )
