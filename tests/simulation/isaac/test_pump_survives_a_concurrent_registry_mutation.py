"""``pump()`` snapshots the registries it walks, so a worker thread may mutate them.

``pump`` is the main-thread half of this backend's threading contract, and its own
docstring states the other half:

    A web UI calls ``get_observation``/``send_action`` from worker threads where
    Isaac's renderer / physics deadlock. Those calls instead enqueue actions and
    read cached frames; this pump (run on the owning main thread) is the single
    place that actually advances the sim and renders the cameras.

So concurrent access is the designed usage rather than an edge case - and
``add_robot``, ``remove_robot``, ``add_camera`` and ``destroy`` all mutate
``_robots`` / ``_cameras`` under ``self._lock``. ``pump`` walked those dicts
**live and unlocked**, so an agent adding a robot from a worker thread while the
pump ran raised

    RuntimeError: dictionary changed size during iteration

measured on 6 of 6 trials with 200 robots and a worker adding 60 more mid-walk.

Two things make it worse than a lost tick. The exception comes from the ``for``
statement's own call to the iterator, which sits **outside** the per-item
``try``/``except (RuntimeError, ...)`` in the body - so the handler that looks
like it covers this cannot see it. And it lands on the **main** thread, which is
the thread that owns Kit and runs the pump loop, so it takes the application down
rather than degrading one frame.

The fix copies each registry under the lock and iterates the copy. The lock is
held only for the copy, deliberately not across the body: the joint read and the
frame grab both reach into Kit, and holding it across them would serialize the
pump against every tool call for the length of a render.

Nothing here needs Isaac Sim - the articulation and camera handles are stand-ins.
What is exercised is the real ``pump``.
"""

from __future__ import annotations

import queue
import threading
import time
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: Enough robots that a worker mutating partway through lands inside the walk.
_ROBOT_COUNT = 200


class _SlowArticulation:
    """Reads slowly enough that a concurrent mutation lands mid-iteration."""

    def __init__(self, delay: float = 0.002) -> None:
        self._delay = delay

    def get_joint_positions(self) -> list[float]:
        import time

        time.sleep(self._delay)
        return [0.0]


class _MutatingArticulation:
    """Runs a callback from inside the joint read - i.e. inside the walk itself."""

    def __init__(self, on_read) -> None:
        self._on_read = on_read

    def get_joint_positions(self) -> list[float]:
        self._on_read()
        return [0.0]

    def set_joint_positions(self, arr: Any) -> None:
        return None

    def set_joint_velocities(self, arr: Any) -> None:
        return None


class _Camera:
    def __init__(self) -> None:
        self.handle = object()


def _no_converge(n: int = 1) -> None:
    """``_converge_render`` with the render loop taken out.

    Carries a default because the method it replaces has one and mypy rejects the
    substitution without it; the *value* is not load-bearing, since mypy erases it
    to ``...`` and the only caller passes ``_idle_converge`` positionally. Measured:
    4 calls, none of them zero-argument.
    """
    return None


def _slow_grab(cname: str, cam: Any) -> Any:
    """``_grab_frame`` taking real time, for a camera walk that nothing else paces.

    Parameter names match the method it replaces, so the substitution is
    signature-compatible. This default is a backstop rather than the stub the
    camera tests race against: measured, it is called 0 times, because every test
    that turns cameras on installs its own grab (see the two below). Kept so a
    future camera test that does not override still walks a paced loop rather than
    an instant one.
    """
    time.sleep(0.002)
    return None


def _engine(*, cameras: bool = False) -> Any:
    """A skeleton engine holding exactly what ``pump`` touches."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world_created = True
    engine._world = types.SimpleNamespace()
    engine._action_q = queue.Queue()
    engine._robots = {}
    engine._cameras = {}
    engine._joint_cache = {}
    engine._frame_cache = {}
    engine._pump_cameras = cameras
    engine._idle_converge = 1
    engine._converge_render = _no_converge  # type: ignore[method-assign]
    # A frame grab that takes real time, because an instant one lets the camera walk
    # finish before any mutation can land - and then the camera tests pass with the
    # snapshot REMOVED (measured: 15/15 either way). The robot walk races only
    # because _SlowArticulation sleeps; the camera walk needs the same. Each camera
    # test installs its own grab over this one, so this assignment is the default
    # for a camera test that does not, not the pacing those tests rely on.
    engine._grab_frame = _slow_grab  # type: ignore[method-assign]
    for i in range(_ROBOT_COUNT):
        engine._robots[f"r{i}"] = types.SimpleNamespace(  # type: ignore[assignment]
            articulation=_SlowArticulation(), joint_names=["j"]
        )
        if cameras:
            engine._cameras[f"c{i}"] = _Camera()  # type: ignore[assignment]
    return engine


def _mutate_after(engine: Any, attr: str, make, *, delay: float = 0.01) -> threading.Thread:
    """Add entries to ``attr`` from another thread, once the walk has started."""

    def _run() -> None:
        import time

        time.sleep(delay)
        registry = getattr(engine, attr)
        for i in range(_ROBOT_COUNT, _ROBOT_COUNT + 60):
            registry[f"new{i}"] = make()

    thread = threading.Thread(target=_run)
    thread.start()
    return thread


class TestAWorkerMayAddARobotMidPump:
    """The measured crash: 6/6 before, 0/6 after."""

    @pytest.mark.parametrize("trial", range(6))
    def test_pump_does_not_raise(self, trial: int) -> None:
        engine = _engine()
        thread = _mutate_after(
            engine,
            "_robots",
            lambda: types.SimpleNamespace(articulation=_SlowArticulation(), joint_names=["j"]),
        )
        try:
            engine.pump(render=False)
        finally:
            thread.join()

    def test_a_removal_mid_pump_does_not_raise(self) -> None:
        """Removal is the other direction and the one ``remove_robot`` performs."""
        engine = _engine()

        def _remove() -> None:
            import time

            time.sleep(0.01)
            for i in range(_ROBOT_COUNT // 2):
                engine._robots.pop(f"r{i}", None)

        thread = threading.Thread(target=_remove)
        thread.start()
        try:
            engine.pump(render=False)
        finally:
            thread.join()


class TestAWorkerMayAddACameraMidPump:
    """The second walk, reached on the idle render path.

    The mutation is triggered from INSIDE the frame grab rather than from a timed
    worker thread. A timed thread cannot reach this walk: pump() does the robot
    walk first, and with 200 slow articulations that takes ~400 ms, so a mutator
    sleeping 10 ms has finished long before the camera loop starts. Measured, that
    made the whole class pass with the camera snapshot REMOVED - it advertised a
    pin it did not hold. Mutating from the grab callback lands the change during
    the iteration by construction, with no timing assumption at all.
    """

    def _engine_mutating_during_the_camera_walk(self) -> Any:
        engine = _engine(cameras=True)
        # One robot, so the robot walk does not dominate the run.
        engine._robots = {"r0": types.SimpleNamespace(articulation=_SlowArticulation(0.0), joint_names=["j"])}
        added = {"done": False}

        def _grab(name: str, handle: Any) -> Any:
            if not added["done"]:
                added["done"] = True
                for i in range(20):
                    engine._cameras[f"late{i}"] = _Camera()
            return None

        engine._grab_frame = _grab
        return engine

    def test_pump_does_not_raise(self) -> None:
        engine = self._engine_mutating_during_the_camera_walk()

        engine.pump(render=True)

    def test_the_mutation_really_happened(self) -> None:
        """Otherwise the test above passes because nothing was added at all."""
        engine = self._engine_mutating_during_the_camera_walk()
        before = len(engine._cameras)

        engine.pump(render=True)

        assert len(engine._cameras) == before + 20

    def test_a_removal_during_the_camera_walk_does_not_raise(self) -> None:
        engine = _engine(cameras=True)
        engine._robots = {"r0": types.SimpleNamespace(articulation=_SlowArticulation(0.0), joint_names=["j"])}
        removed = {"done": False}

        def _grab(name: str, handle: Any) -> Any:
            if not removed["done"]:
                removed["done"] = True
                for i in range(_ROBOT_COUNT // 2):
                    engine._cameras.pop(f"c{i}", None)
            return None

        engine._grab_frame = _grab

        engine.pump(render=True)


class TestTheConvergeRenderWalkIsSnapshottedToo:
    """pump() calls ``_converge_render`` at step 2, and it walks ``_robots`` too.

    This is the walk a snapshot added only inside ``pump`` misses: it sits two
    lines ABOVE the loops that were fixed, in a helper, on the idle preview path
    ``run_pump_forever`` takes by default. The class above stubs
    ``_converge_render`` to a no-op - correct for testing pump's own loops, and
    exactly why it cannot see this one - so these drive the REAL helper.

    The mutation is triggered from inside ``world.step``, which
    ``_converge_render`` calls once per convergence tick, so it lands during the
    walk by construction rather than by timing.
    """

    def _engine_mutating_during_converge(self, *, remove: bool = False) -> Any:
        engine = _engine()
        engine._converge_render = IsaacSimulation._converge_render.__get__(engine, IsaacSimulation)
        engine._idle_converge = 4
        fired = {"done": False}

        def _mutate() -> None:
            if fired["done"]:
                return
            fired["done"] = True
            if remove:
                for i in range(_ROBOT_COUNT // 2):
                    engine._robots.pop(f"r{i}", None)
            else:
                for i in range(60):
                    engine._robots[f"late{i}"] = types.SimpleNamespace(
                        articulation=_MutatingArticulation(lambda: None), joint_names=["j"]
                    )

        # Fired from INSIDE the walk, not from world.step. _converge_render's
        # shape is `for _ in range(n): for r in <registry>: ...; world.step()`,
        # so a mutation at world.step lands BETWEEN outer iterations - and an
        # unsnapshotted `self._robots.values()` is a fresh view each iteration,
        # which never sees a size change mid-walk and never raises. Measured: the
        # world.step version passed 17/17 with the snapshot removed. The per-robot
        # read is the only hook actually inside the iteration.
        first = _MutatingArticulation(_mutate)
        engine._robots = {"r0": first}
        engine._robots = {
            f"r{i}": types.SimpleNamespace(articulation=first if i == 0 else _SlowArticulation(0.0), joint_names=["j"])
            for i in range(_ROBOT_COUNT)
        }
        engine._world = types.SimpleNamespace(step=lambda render=True: None)
        return engine, fired

    def test_an_add_during_converge_does_not_raise(self) -> None:
        engine, fired = self._engine_mutating_during_converge()

        engine.pump(render=True)

        assert fired["done"], "world.step never ran, so nothing was mutated mid-walk"

    def test_a_removal_during_converge_does_not_raise(self) -> None:
        engine, fired = self._engine_mutating_during_converge(remove=True)

        engine.pump(render=True)

        assert fired["done"]

    def test_the_helper_is_the_real_one(self) -> None:
        """A stub here would make both tests above vacuous."""
        engine, _ = self._engine_mutating_during_converge()

        assert engine._converge_render.__func__ is IsaacSimulation._converge_render


class TestThePumpStillDoesItsWork:
    """Controls. A snapshot that walked nothing would pass every test above."""

    def test_the_joint_cache_is_populated(self) -> None:
        engine = _engine()

        engine.pump(render=False)

        assert len(engine._joint_cache) == _ROBOT_COUNT
        assert engine._joint_cache["r0"] == {"j": 0.0}

    def test_the_frame_cache_is_populated_on_the_idle_path(self) -> None:
        engine = _engine(cameras=True)

        def _named_grab(cname: str, cam: Any) -> Any:
            time.sleep(0.001)
            return f"frame-{cname}"

        engine._grab_frame = _named_grab

        engine.pump(render=True)

        assert len(engine._frame_cache) == _ROBOT_COUNT
        assert engine._frame_cache["c0"] == "frame-c0"

    def test_a_robot_added_mid_pump_is_picked_up_on_the_next_tick(self) -> None:
        """A snapshot is a point-in-time read, so the new robot is served by the
        following tick rather than dropped. Pinned so the fix cannot be mistaken
        for one that silently forgets late arrivals."""
        engine = _engine()
        engine.pump(render=False)
        engine._robots["late"] = types.SimpleNamespace(articulation=_SlowArticulation(), joint_names=["j"])

        assert "late" not in engine._joint_cache

        engine.pump(render=False)

        assert "late" in engine._joint_cache

    def test_the_lock_is_not_held_across_the_body(self) -> None:
        """Holding it across the joint read would serialize the pump against
        every tool call for the length of a render. Observed from inside the
        walk, on the thread the articulation read runs on."""
        engine = _engine()
        held: list[bool] = []

        def _probe() -> bool:
            """Whether the lock is currently held, asked from another thread.

            An RLock is re-entrant for its owner, so the main thread could take
            it again even while holding it - the question is only answerable from
            a thread that does not own it. Acquire and release happen in that
            same thread: releasing an RLock from a thread that does not own it
            raises RuntimeError, and pump's per-item handler would swallow it,
            leaving this assertion silently unevaluated.
            """
            got: list[bool] = []

            def _try() -> None:
                acquired = engine._lock.acquire(blocking=False)
                got.append(acquired)
                if acquired:
                    engine._lock.release()

            thread = threading.Thread(target=_try)
            thread.start()
            thread.join()
            return not got[0]

        class _Observing:
            def get_joint_positions(self) -> list[float]:
                held.append(_probe())
                return [0.0]

        engine._robots = {"only": types.SimpleNamespace(articulation=_Observing(), joint_names=["j"])}

        engine.pump(render=False)

        assert held == [False], "the lock was held while the articulation was read"
