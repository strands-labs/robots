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

import ast
import inspect
import re
import textwrap
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


# Every scan below reads the loops for the wait they were converted away from,
# and every converted loop's docstring EXPLAINS that conversion by quoting the
# very call being banned. So the scan has to tell code from prose, and doing that
# textually does not work: these graders used to strip ``__doc__`` out of
# ``inspect.getsource``, which assumes the two are byte-identical. Python 3.13
# removes a docstring's common leading indentation at compile time, so ``__doc__``
# stopped being a substring of the source, the strip removed nothing, and
# ``_state_loop`` was reported as pacing on a wait it does not contain -- on a
# supported interpreter, with the loop unchanged, and with a message telling the
# reader to use the Ticker the loop already uses. The textual form also never
# covered comments, which ``getsource`` includes and ``__doc__`` never did.
#
# ``ast`` draws the line where the compiler draws it: a docstring is a constant
# expression and a comment is not in the tree at all, so neither can be a hit on
# any interpreter, and a call split over several lines still is one.

_STOP_FLAG = re.compile(r"^_[a-z_0-9]*(?:stop|shutdown|halt)[a-z_0-9]*$")


def _code(source: str) -> ast.Module:
    """Parse module source, or one function's source (still indented)."""
    return ast.parse(textwrap.dedent(source))


def _pacing_waits(source: str) -> list[int]:
    """Line numbers of ``.wait(...)`` calls on a stop-flag attribute.

    Matched on the shape rather than one spelling, because a spelling has
    slipped past a narrower check before: the teleop apply loop was found last
    of the twelve because its event is named ``_teleop_stop_event``. So any
    attribute whose name reads as a stop flag counts, in any statement position,
    with or without ``timeout=``.

    Args:
        source: Module source, or the source of a single function as
            ``inspect.getsource`` returns it.

    Returns:
        The 1-based line of each such call within ``source``, in tree order.
        Empty when the only mentions are in docstrings or comments.
    """
    return [
        node.lineno
        for node in ast.walk(_code(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "wait"
        and isinstance(node.func.value, ast.Attribute)
        and _STOP_FLAG.match(node.func.value.attr)
    ]


def _calls_named(source: str, name: str) -> list[int]:
    """Line numbers of calls whose callee ends in ``name``.

    ``Ticker(...)`` and ``pacing.Ticker(...)`` are one construction, so the match
    is on the last segment of the callee.

    Args:
        source: Module source, or one function's source.
        name: The trailing segment to match, e.g. ``"Ticker"`` or ``"_paced"``.

    Returns:
        The 1-based line of each matching call within ``source``.
    """
    return [
        node.lineno
        for node in ast.walk(_code(source))
        if isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] == name
    ]


@pytest.mark.parametrize("attr", ["_state_loop", "_heartbeat_loop", "_camera_loop"])
def test_the_converted_loop_no_longer_paces_on_the_stop_event(attr: str) -> None:
    """Pin the conversion in source, so a later edit cannot quietly revert it.

    The rate tests above are the real proof, but they are timing tests: on a
    heavily loaded machine their floors could in principle be met by luck. This
    one cannot be satisfied by luck.
    """
    source = inspect.getsource(getattr(Mesh, attr))
    waits = _pacing_waits(source)
    assert not waits, (
        f"Mesh.{attr} is pacing on _stop_event.wait again (line {waits[0]} of its definition) - that "
        "wait adds the tick's work to the period, and is inflated further in a daemon-descended tree; "
        "use mesh.pacing.Ticker"
    )
    assert _calls_named(source, "Ticker"), f"Mesh.{attr} should pace on a Ticker"


@pytest.mark.parametrize("storage", ["dedented", "absent"])
def test_the_pacing_scan_reads_the_code_whatever_the_docstring_is(
    storage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verdict cannot depend on how the interpreter stores ``__doc__``.

    The scan above used to separate code from prose by removing ``__doc__`` from
    ``inspect.getsource`` textually. Python 3.13 dedents docstrings at compile
    time, so on that interpreter the removal matched nothing, ``_state_loop``'s
    own explanation of the conversion was read as code, and the grader failed on
    documentation while the loop it grades was correct. Both storage shapes are
    reproduced here on whatever interpreter runs the suite, so the environment
    cannot decide the verdict and a textual strip cannot come back unnoticed.
    """
    doc = Mesh._state_loop.__doc__
    assert doc and "_stop_event.wait(" in doc, (
        "this pin rests on _state_loop's docstring quoting the banned call, which is the "
        "documentation the scan must not read. If that prose is gone, so is the hazard."
    )
    if storage == "dedented":  # what Python 3.13 stores
        first, sep, rest = doc.partition("\n")
        stored: str | None = first + sep + textwrap.dedent(rest)
        assert stored != doc, "the docstring has no indented continuation, so this shape proves nothing"
    else:
        stored = None

    monkeypatch.setattr(Mesh._state_loop, "__doc__", stored)
    test_the_converted_loop_no_longer_paces_on_the_stop_event("_state_loop")


@pytest.mark.parametrize(
    ("label", "planted", "is_a_pacer"),
    [
        (
            "a docstring quoting the call it stopped making",
            'def loop(self):\n    """Paced by a Ticker now, not by self._stop_event.wait(period)."""\n'
            "    with Ticker(1.0, self._stop_event) as ticker:\n        ticker.wait()\n",
            False,
        ),
        (
            "prose on its own line, with no backticks to hide behind",
            'def loop(self):\n    """Was paced by\n\n    self._stop_event.wait(period), now a Ticker.\n    """\n'
            "    with Ticker(1.0, self._stop_event) as ticker:\n        ticker.wait()\n",
            False,
        ),
        (
            "a comment quoting the call",
            "def loop(self):\n    # was: self._stop_event.wait(period)\n"
            "    with Ticker(1.0, self._stop_event) as ticker:\n        ticker.wait()\n",
            False,
        ),
        (
            "the ticker's own wait, which is the cure",
            "def loop(self):\n    with Ticker(1.0, self._stop_event) as ticker:\n        ticker.wait()\n",
            False,
        ),
        (
            "the real call on one line",
            "def loop(self):\n    while not self._stop_event.wait(0.1):\n        pass\n",
            True,
        ),
        (
            "the real call split over three lines",
            "def loop(self):\n    while not self._stop_event.wait(\n        period,\n    ):\n        pass\n",
            True,
        ),
        (
            "the real call under a differently named stop flag",
            "def loop(self):\n    while not self._teleop_stop_event.wait(timeout=0.1):\n        pass\n",
            True,
        ),
    ],
)
def test_only_a_wait_the_interpreter_would_execute_is_a_pacing_hit(label: str, planted: str, is_a_pacer: bool) -> None:
    """Both halves of the scan's job, on source planted for the purpose.

    The two rows the previous line-by-line reading got wrong are the point: a
    docstring line with no backticks was flagged as a pacer, and a real call
    split across lines was not flagged at all. A scan that reports documentation
    trains its reader to ignore it, and one that misses a wrapped call leaves
    exactly the loop it exists to find.
    """
    hits = _pacing_waits(planted)
    assert bool(hits) is is_a_pacer, (
        f"{label}: the scan {'missed' if is_a_pacer else 'flagged'} it (hits at {hits}) in:\n{planted}"
    )


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
    from strands_robots.mesh import sensors as mesh_sensors

    source = inspect.getsource(getattr(mesh_sensors.SensorLoopsMixin, loop))
    assert _calls_named(source, "_paced"), f"{loop} does not pace through SensorLoopsMixin._paced"
    waits = _pacing_waits(source)
    assert not waits, f"{loop} paces on the inflated Event.wait again (line {waits[0]}) - see mesh.pacing"


def test_only_the_shared_generator_owns_a_ticker_in_the_sensors_module() -> None:
    """One pacer for the whole mixin, and no wait left beside it."""
    from strands_robots.mesh import sensors as mesh_sensors

    module_source = inspect.getsource(mesh_sensors)
    waits = _pacing_waits(module_source)
    assert not waits, f"pacing waits left in sensors.py at lines {waits}"
    tickers = _calls_named(module_source, "Ticker")
    assert len(tickers) == 1, (
        "exactly one Ticker construction belongs in this module - the one inside _paced - "
        f"but {len(tickers)} were built, at lines {tickers}"
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
    flag", in any statement position, with or without ``timeout=``. It is read
    out of the parsed tree rather than matched per line, because a walk over the
    files is not the part that misses things: a per-line pattern cannot see a
    call wrapped across lines, and it has to guess at which mentions are prose.

    Waits that are NOT pacing (a shutdown join, a settle window) are allowed:
    they run once, so the cost is paid once rather than on every tick of a
    stream. They are listed explicitly, so adding one is a deliberate act.
    """
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
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "pacing.py":
            continue
        lines = path.read_text().splitlines()
        # Prose is not a pacer, and it does not need excluding by hand: several
        # modules now DESCRIBE this bug in a comment or a docstring quoting the
        # offending call, and neither is a call in the parsed tree.
        for lineno in _pacing_waits("\n".join(lines)):
            line = lines[lineno - 1]
            rel = str(path.relative_to(root))
            if any(rel == f and frag in line for (f, frag) in allowed):
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these waits pace a loop, so the tick's work is added to its period; "
        f"pace them with mesh.pacing.Ticker or add them to `allowed` with a reason: {offenders}"
    )
