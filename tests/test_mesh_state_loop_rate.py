"""The state loop must publish at STATE_HZ on THIS machine, not at 40% of it.

The mesh's publish loops paced themselves with ``self._stop_event.wait(period)``.
That is a delay where a rate needs a deadline: the time ``_read_state`` spends on
the bus was added to the period instead of subtracted from it, so the loop ran at
``1 / (period + read)`` and every counter reported that as the rate the robot
managed. On a host that also inflates ``Event.wait`` (see
:mod:`strands_robots.mesh.pacing`) the two costs stack.

This test measures the loop's ACHIEVED rate through the real ``Mesh._state_loop``
with the transport mocked, and calibrates its floor against the machine it is
running on (:func:`sleep_penalty_s`) instead of asserting a number that happens
to pass in a terminal. That calibration is the whole point: the regression it
guards against is invisible in an interactive shell and severe under a daemon.

The session is mocked exactly the way tests/mesh/test_mesh.py mocks it, so no
real zenoh session is ever built here: a test that cannot prove transport
isolation must not run at all, and that applies to anything touching a Mesh.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.core import Mesh
from strands_robots.mesh.pacing import sleep_penalty_s
from strands_robots.mesh.session import STATE_HZ


class _StatefulRobot:
    """Duck-typed robot whose state read is cheap and always succeeds."""

    tool_name_str = "pacingbot"

    def __init__(self) -> None:
        self._world = MagicMock()
        self._world._data.time = 1.0
        self._world.robots = {"arm0": object()}


def _run_state_loop_for(seconds: float) -> tuple[int, float]:
    """Drive the real _state_loop for a wall-clock window; count state publishes."""
    mesh = Mesh(_StatefulRobot(), peer_id="pace-1", peer_type="sim")
    published: list[str] = []
    mesh.publish = lambda topic, _payload=None, **_kw: published.append(topic)  # type: ignore[method-assign]
    mesh._running = True
    stop = mesh._stop_event
    started = threading.Event()

    def loop() -> None:
        started.set()
        mesh._state_loop()

    thread = threading.Thread(target=loop, daemon=True)
    with patch("strands_robots.mesh.core.put"):
        thread.start()
        started.wait(2.0)
        start = time.perf_counter()
        # SLEEP the window, do not spin it. A Python busy-wait holds the GIL and
        # starves the loop thread whose rate is being measured -- measured: a
        # 30Hz camera loop read as 21.0fps against a spinning main thread. The
        # sleep being inflated by the penalty under test is harmless here because
        # the rate is computed against the MEASURED elapsed time, not the
        # requested window.
        time.sleep(seconds)
        elapsed = time.perf_counter() - start
        ticks = len([t for t in published if t.endswith("/state")])
        mesh._running = False
        stop.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive(), "the state loop did not stop within 5s of the stop event"
    return ticks, elapsed


class TestTheCameraLoopHitsItsNominalRate:
    """The camera loop is where the old pacing cost the most.

    Grabbing a frame is slow and the penalty was added on top of it, so a
    nominal 30Hz ran at about 6Hz -- and that is the rate a recorded dataset's
    video was actually captured at while the run reported 30.
    """

    def test_thirty_hz_of_cheap_frames_is_thirty_hz(self) -> None:
        mesh = Mesh(_StatefulRobot(), peer_id="pace-cam", peer_type="sim")
        mesh._running = True
        frames: list[float] = []
        with (
            patch("strands_robots.mesh.core.put"),
            patch.object(mesh, "_publish_cameras_once", side_effect=lambda: frames.append(time.perf_counter())),
        ):
            thread = threading.Thread(target=mesh._camera_loop, args=(30.0,), daemon=True)
            thread.start()
            start = time.perf_counter()
            time.sleep(1.0)  # sleep, never spin: a spinning main thread starves the loop (GIL)
            elapsed = time.perf_counter() - start
            mesh._running = False
            mesh._stop_event.set()
            thread.join(timeout=5.0)
        assert not thread.is_alive(), "the camera loop did not stop within 5s"
        achieved = len(frames) / elapsed
        assert achieved > 21.0, (
            f"camera loop achieved {achieved:.1f}fps asking for 30 "
            f"(sleep penalty here {sleep_penalty_s() * 1000:.0f}ms) - see mesh.pacing"
        )

    def test_a_dropped_deadline_does_not_publish_a_burst_of_near_identical_frames(self) -> None:
        """A gap is a better lie about a camera than a burst.

        Frames stamped microseconds apart tell a consumer the camera sped up,
        when what actually happened is that it stalled.
        """
        mesh = Mesh(_StatefulRobot(), peer_id="pace-cam2", peer_type="sim")
        mesh._running = True
        stamps: list[float] = []
        calls = {"n": 0}

        def grab() -> None:
            calls["n"] += 1
            if calls["n"] == 2:  # one slow frame worth ~7 periods
                deadline = time.perf_counter() + 0.15
                while time.perf_counter() < deadline:
                    pass
            stamps.append(time.perf_counter())

        with patch("strands_robots.mesh.core.put"), patch.object(mesh, "_publish_cameras_once", side_effect=grab):
            thread = threading.Thread(target=mesh._camera_loop, args=(50.0,), daemon=True)
            thread.start()
            time.sleep(0.6)
            mesh._running = False
            mesh._stop_event.set()
            thread.join(timeout=5.0)
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        instant = [g for g in gaps if g < 0.002]
        assert len(instant) <= 1, (
            f"{len(instant)} of {len(gaps)} frame gaps were under 2ms at a nominal 20ms period - "
            "the loop is chasing the deadlines it missed during the slow frame"
        )


class TestTheStateLoopHitsItsNominalRate:
    def test_achieved_rate_is_close_to_state_hz_even_where_sleeps_are_taxed(self) -> None:
        penalty = sleep_penalty_s()
        window = 1.2
        ticks, elapsed = _run_state_loop_for(window)
        achieved = ticks / elapsed

        floor = STATE_HZ * 0.7
        assert achieved >= floor, (
            f"state loop achieved {achieved:.1f}Hz against STATE_HZ={STATE_HZ} "
            f"(sleep penalty on this machine: {penalty * 1000:.0f}ms). Below {floor:.1f}Hz means the loop is "
            "paced by an inflated blocking wait again - see mesh.pacing."
        )

        if penalty >= 0.01:
            # On a taxed machine the OLD pacing could not have passed: 1/(0.1 +
            # penalty) is at most ~5Hz. Pin that gap so the test is known to be
            # measuring the thing it claims to measure.
            old_style_ceiling = 1.0 / (1.0 / STATE_HZ + penalty)
            assert achieved > old_style_ceiling * 1.4, (
                f"achieved {achieved:.1f}Hz is within noise of what Event.wait pacing would give "
                f"({old_style_ceiling:.1f}Hz) - the conversion may not be in effect"
            )

    def test_the_loop_stops_promptly_rather_than_at_the_end_of_a_tick(self) -> None:
        """A stop must not wait out the period, whatever the pacing.

        _run_state_loop_for joins with a 5s timeout, so a loop that only checked
        its stop event once per period would still pass that; this asserts the
        stop is fast in absolute terms with a slow nominal rate in play.
        """
        mesh = Mesh(_StatefulRobot(), peer_id="pace-2", peer_type="sim")
        mesh.publish = lambda *_a, **_kw: None  # type: ignore[method-assign]
        mesh._running = True
        with patch("strands_robots.mesh.core.put"), patch.object(mesh, "_read_state", return_value=None):
            thread = threading.Thread(target=mesh._state_loop, daemon=True)
            thread.start()
            time.sleep(0.05)  # let it enter the wait
            start = time.perf_counter()
            mesh._running = False
            mesh._stop_event.set()
            thread.join(timeout=3.0)
            took = time.perf_counter() - start
        assert not thread.is_alive()
        budget = 1.0 / STATE_HZ + sleep_penalty_s() + 0.3
        assert took < budget, f"stop took {took:.3f}s (budget {budget:.3f}s)"

    def test_a_slow_state_read_is_subtracted_from_the_period_not_added(self) -> None:
        """20ms of bus time inside a 100ms period must still tick at ~10Hz."""
        mesh = Mesh(_StatefulRobot(), peer_id="pace-3", peer_type="sim")
        published: list[float] = []

        def slow_read() -> dict[str, float]:
            deadline = time.perf_counter() + 0.02
            while time.perf_counter() < deadline:
                pass
            return {"t": time.perf_counter()}

        mesh.publish = lambda *_a, **_kw: published.append(time.perf_counter())  # type: ignore[method-assign]
        mesh._running = True
        with patch("strands_robots.mesh.core.put"), patch.object(mesh, "_read_state", side_effect=slow_read):
            thread = threading.Thread(target=mesh._state_loop, daemon=True)
            thread.start()
            time.sleep(1.0)
            mesh._running = False
            mesh._stop_event.set()
            thread.join(timeout=5.0)
        gaps = [b - a for a, b in zip(published, published[1:], strict=False)]
        assert gaps, "no state publishes to measure"
        median_gap = sorted(gaps)[len(gaps) // 2]
        nominal = 1.0 / STATE_HZ
        assert median_gap < nominal * 1.5, (
            f"median gap {median_gap * 1000:.0f}ms against a {nominal * 1000:.0f}ms period - "
            "the 20ms read is being added to the period instead of subtracted from it"
        )


@pytest.mark.parametrize("attr", ["_state_loop", "_heartbeat_loop", "_camera_loop"])
def test_the_converted_loop_no_longer_paces_on_the_stop_event(attr: str) -> None:
    """Pin the conversion in source, so a later edit cannot quietly revert it.

    The rate tests above are the real proof, but they are timing tests: on a
    heavily loaded machine their floors could in principle be met by luck. This
    one cannot be satisfied by luck.
    """
    import inspect

    func = getattr(Mesh, attr)
    source = inspect.getsource(func)
    # Scan the CODE, not the prose: the docstring of the converted loop explains
    # what it stopped doing and therefore contains the very string this test
    # bans. My first version failed on its own explanation - a source-scanning
    # test that reads comments is a test that punishes documentation.
    doc = func.__doc__
    if doc:
        source = source.replace(doc, "")
    assert "_stop_event.wait(" not in source, (
        f"Mesh.{attr} is pacing on _stop_event.wait again - that wait adds the tick's work to "
        "the period, and is inflated further in a daemon-descended tree; use mesh.pacing.Ticker"
    )
    assert "Ticker(" in source, f"Mesh.{attr} should pace on a Ticker"


@pytest.mark.parametrize(
    "loop",
    ["_pose_loop", "_health_loop", "_imu_loop", "_odom_loop", "_lidar_loop", "_hand_loop", "_map_info_loop"],
)
def test_every_sensor_loop_paces_through_the_shared_ticker_generator(loop: str) -> None:
    """All seven sensor loops must pace in ONE place.

    They differ only in what they read, so pacing them individually is how six
    get the ownership rules right and the seventh leaks a selector. The generator
    also has to be the ONLY pacing wait in the module: a loop that quietly kept
    ``_stop_event.wait(period)`` would run at 40% of its rate in a daemon-hosted
    robot while the other six were fixed, which is harder to notice than all
    seven being slow.
    """
    import inspect

    from strands_robots.mesh import sensors as mesh_sensors

    func = getattr(mesh_sensors.SensorLoopsMixin, loop)
    source = inspect.getsource(func)
    assert "self._paced(" in source, f"{loop} does not pace through SensorLoopsMixin._paced"
    assert "_stop_event.wait(" not in source, f"{loop} paces on the inflated Event.wait again - see mesh.pacing"


def test_only_the_shared_generator_owns_a_ticker_in_the_sensors_module() -> None:
    import inspect

    from strands_robots.mesh import sensors as mesh_sensors

    module_source = inspect.getsource(mesh_sensors)
    # Strip docstrings' mention of the old call by counting real code lines only.
    code_hits = [
        line
        for line in module_source.splitlines()
        if "_stop_event.wait(" in line and not line.lstrip().startswith(("#", '"', "`"))
    ]
    assert not code_hits, f"pacing waits left in sensors.py: {code_hits}"
    assert module_source.count("Ticker(") == 1, (
        "exactly one Ticker construction belongs in this module - the one inside _paced"
    )


def test_no_publish_loop_in_the_mesh_still_paces_on_an_inflated_wait() -> None:
    """The inventory check: this is only cured if NO pacer was missed.

    This scans for the SHAPE rather than for one spelling, because a spelling has
    already slipped past a narrower check: the teleop apply loop was found only
    after the other eleven had been converted, because its event is named
    ``_teleop_stop_event`` and the grep the inventory came from looked for
    ``_stop_event``. A pacer missed while its siblings are fixed is harder to
    notice than all of them being slow, since the stream that is still late looks
    like the one sensor that is genuinely slow.

    So the shape is "``.wait(...)`` on an attribute whose name reads as a stop
    flag", in any statement position, with or without ``timeout=``. The keyword
    form is included because a walk over the files is not enough on its own - the
    regex is the part that misses things, not the ``rglob``.

    Waits that are NOT pacing (a shutdown join, a settle window) are allowed:
    they run once, so the cost is paid once rather than on every tick of a
    stream. They are listed explicitly, so adding one is a deliberate act.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "strands_robots"
    allowed = {
        # (file, fragment) -> why this wait's inflation does not matter here.
        # Each reason carries the NUMBER it rests on, so a later change of that
        # number (a poll interval dropped to 100ms, say) invalidates the excuse
        # visibly instead of quietly.
        ("mesh/core.py", "self._stop_event.wait(timeout=timeout)"): "one-shot shutdown wait, not a loop tick",
        ("hardware_ros_bridge.py", "self._stop.wait(self._spin_period)"): (
            "the EXCEPTION path only: a backoff after spin_once raised, where waiting longer "
            "than spin_period is the intent. The happy path has no wait at all"
        ),
    }
    # Any statement position (`while not ...`, `if ...`, bare), any attribute
    # whose name reads as a stop flag (_stop_event, _teleop_stop_event,
    # _stop_evt, _shutdown_event), keyword form included.
    pattern = re.compile(r"self\._[a-z_0-9]*(?:stop|shutdown|halt)[a-z_0-9]*\.wait\(\s*(?:timeout\s*=\s*)?([^)]*)\)")
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "pacing.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            match = pattern.search(line)
            if not match:
                continue
            # Prose is not a pacer. Several modules now DESCRIBE this bug in a
            # comment or docstring quoting the offending call, and a scanner
            # that flags its own documentation trains the reader to ignore it.
            before = line[: match.start()]
            if before.lstrip().startswith("#") or "``" in before or '"""' in before:
                continue
            rel = str(path.relative_to(root))
            if any(rel == f and frag in line for (f, frag) in allowed):
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these waits pace a loop, so the tick's work is added to its period; "
        f"pace them with mesh.pacing.Ticker or add them to `allowed` with a reason: {offenders}"
    )
