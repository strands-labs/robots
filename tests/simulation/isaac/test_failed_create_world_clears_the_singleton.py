"""A failed ``create_world`` tears the World down instead of dropping the reference.

``World`` registers itself as ``SimulationContext.instance()``, a process-wide
singleton, so ``self._world = None`` releases nothing. ``create_world``'s failure
path did only that, while ``destroy`` does the full teardown - ``stop()``,
``clear_instance()``, then drop.

That asymmetry is not a leak that costs memory; it is a leak that returns a wrong
answer to the caller's obvious next move. Measured on an A10G under Isaac Sim
6.0.1:

    World(physics_dt=1/60)      -> registered as SimulationContext.instance()
    del w1; gc.collect()        -> instance STILL alive
    World(physics_dt=1/120)     -> the SAME object; get_physics_dt() is 1/60
    clear_instance()
    World(physics_dt=1/120)     -> a fresh world; get_physics_dt() is 1/120

So a caller whose ``create_world`` failed, who corrected the config and called it
again, got a world running the **failed attempt's** physics - and the new result
echoed the new value, because ``world_info`` is built from the config and the
arguments and never read back off the world. Silent, and in the direction that
looks fine.

The failure path now performs the same teardown as ``destroy``, under the same
narrow handler: ``stop()`` and ``clear_instance()`` can themselves raise on a
half-built world, and this path already has a failure to report, so a cleanup
error is logged rather than allowed to replace it.

Nothing here needs Isaac Sim - a fake ``World`` records the teardown calls and can
be made to fail at a chosen stage.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.simulation.isaac import simulation as isaac_simulation
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation


class _PhysicsContext:
    def __init__(self, fail: bool) -> None:
        self._fail = fail
        self.device = "cuda:0"

    def set_gravity(self, magnitude: object) -> None:
        if self._fail:
            # A real cause: Isaac Sim 5.1's set_gravity rejects a non-scalar.
            raise RuntimeError("set_gravity failed on a half-built physics context")


class _FakeScene:
    def add_default_ground_plane(self) -> None:
        return None


class _FakeWorld:
    """A SINGLETON, because that is the mechanism under test.

    Modelling this is not optional. Measured against the real
    ``isaacsim.core.api.World`` on an A10G, a second construction returns the
    same object and *keeps the first call's arguments* - ``World(physics_dt=1/60)``
    then ``World(physics_dt=1/120)`` reports ``get_physics_dt()`` of 1/60 - and
    only ``clear_instance()`` releases it. A fake that handed out a fresh object
    per call would let every retry assertion below pass on the broken code, since
    the retry would get a clean world for the wrong reason.
    """

    #: The live instance, or None. Mirrors ``SimulationContext.instance()``.
    instance: _FakeWorld | None = None
    #: Every object actually constructed, so a reuse is distinguishable.
    built: list[_FakeWorld] = []
    #: Set by the fixture to make ``create_world`` fail after the World exists.
    fail_on_gravity: bool = False

    #: Set by a test to make the CONSTRUCTOR itself raise, after registration.
    fail_in_init: bool = False

    # Annotation-only, so no class attribute is created and the runtime shape is
    # unchanged: both are bound per instance, and both are bound in a method that
    # runs before - or is defined above - the one that reads them.
    _initialized: bool
    instance_cleared: bool

    def __new__(cls, **kwargs: Any) -> _FakeWorld:
        if cls.instance is not None:
            return cls.instance
        obj = super().__new__(cls)
        # Registered in __new__, exactly as SimulationContext does - which is why
        # a teardown gated on the caller's `self._world` (bound only after
        # __init__ returns) cannot see an instance whose __init__ raised.
        cls.instance = obj
        cls.built.append(obj)
        obj._initialized = False
        return obj

    @classmethod
    def clear_instance(cls) -> None:  # type: ignore[override]
        """A classmethod, as on the real World, so it is reachable with no ref."""
        if cls.instance is not None:
            cls.instance.instance_cleared = True
        cls.instance = None

    def __init__(self, **kwargs: Any) -> None:
        # Only the first construction binds; a reuse keeps its original kwargs,
        # which is exactly the measured behaviour that made the leak a wrong
        # answer rather than merely wasted memory.
        if self._initialized:
            return
        self._initialized = True
        if _FakeWorld.fail_in_init:
            raise RuntimeError("physics_dt the integrator cannot honour")
        self.kwargs = kwargs
        self.stopped = False
        self.instance_cleared = False
        self.physics_context = _PhysicsContext(fail=_FakeWorld.fail_on_gravity)
        self.scene = _FakeScene()

    def get_physics_context(self) -> _PhysicsContext:
        return self.physics_context

    def stop(self) -> None:
        self.stopped = True

    def reset(self) -> None:
        return None


@pytest.fixture()
def fake_isaacsim(monkeypatch):
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP", None)
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP_LAUNCH", None)
    _FakeWorld.built = []
    _FakeWorld.instance = None
    _FakeWorld.fail_on_gravity = False
    _FakeWorld.fail_in_init = False
    mods = {}
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.api"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        mods[name] = module
    mods["isaacsim"].SimulationApp = lambda launch=None: types.SimpleNamespace(launch=launch)
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].api = mods["isaacsim.core.api"]
    mods["isaacsim.core.api"].World = _FakeWorld
    return mods


class TestAFailedCreateWorldTearsTheWorldDown:
    def test_the_world_is_stopped_and_the_instance_cleared(self, fake_isaacsim) -> None:
        _FakeWorld.fail_on_gravity = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

        result = sim.create_world()

        assert result["status"] == "error", result
        assert len(_FakeWorld.built) == 1
        world = _FakeWorld.built[0]
        assert world.stopped, "stop() was not called, so the singleton stayed live"
        assert world.instance_cleared, "clear_instance() was not called, so a retry reuses it"

    def test_the_reference_is_still_dropped(self, fake_isaacsim) -> None:
        _FakeWorld.fail_on_gravity = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

        sim.create_world()

        assert sim._world is None

    def test_the_world_is_not_marked_created(self, fake_isaacsim) -> None:
        _FakeWorld.fail_on_gravity = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

        sim.create_world()

        assert sim._world_created is False

    def test_a_retry_builds_a_new_world(self, fake_isaacsim) -> None:
        """The consequence the teardown exists for: the retry must not inherit
        the failed attempt's world. With the real singleton, reusing it silently
        kept the first call's physics_dt."""
        _FakeWorld.fail_on_gravity = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))
        assert sim.create_world()["status"] == "error"

        _FakeWorld.fail_on_gravity = False
        result = sim.create_world()

        assert result["status"] == "success", result
        assert len(_FakeWorld.built) == 2, "the retry reused the failed world"
        assert _FakeWorld.built[1] is not _FakeWorld.built[0]

    def test_the_retry_gets_its_own_timestep(self, fake_isaacsim) -> None:
        """Read off the world the retry actually built, not off the config."""
        _FakeWorld.fail_on_gravity = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))
        sim.create_world(timestep=1.0 / 60)

        _FakeWorld.fail_on_gravity = False
        assert sim.create_world(timestep=1.0 / 120)["status"] == "success"

        assert _FakeWorld.built[1].kwargs["physics_dt"] == pytest.approx(1.0 / 120)


class TestTheConstructorItselfRaising:
    """The likeliest failure, and the one a ``self._world`` guard cannot see.

    ``self._world`` is bound only after ``World(...)`` RETURNS, while the singleton
    registers inside ``SimulationContext.__new__``. So a failure raised by the
    constructor - an unusable ``physics_dt``, a device the host cannot provide -
    left a registered instance that a ``if self._world is not None`` teardown
    skipped entirely, which is the exact leak the fix is about.
    """

    def test_the_singleton_is_cleared_even_though_self_world_was_never_bound(self, fake_isaacsim) -> None:
        _FakeWorld.fail_in_init = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

        result = sim.create_world()

        assert result["status"] == "error", result
        assert sim._world is None, "the constructor raised, so this was never bound"
        assert len(_FakeWorld.built) == 1, "the instance WAS registered in __new__"
        assert _FakeWorld.instance is None, "the registered singleton was not cleared"

    def test_a_retry_after_a_constructor_failure_builds_a_new_world(self, fake_isaacsim) -> None:
        _FakeWorld.fail_in_init = True
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))
        assert sim.create_world()["status"] == "error"

        _FakeWorld.fail_in_init = False
        assert sim.create_world()["status"] == "success", "the retry inherited the failed singleton"

        assert len(_FakeWorld.built) == 2


class TestACleanupFailureDoesNotReplaceTheRealError:
    """The original failure is what the caller needs; cleanup is best-effort."""

    def test_a_raising_stop_does_not_skip_clear_instance(self, fake_isaacsim) -> None:
        """``stop()`` is the call that raises on a half-built world - destroy()'s
        own comment says so. With both in one ``try``, a raising stop skipped the
        ``clear_instance()`` that is the entire point, silently restoring the leak.
        """
        _FakeWorld.fail_on_gravity = True

        def _raising_stop(self: Any) -> None:
            raise RuntimeError("stop() on a half-built world")

        _FakeWorld.stop = _raising_stop  # type: ignore[method-assign]
        try:
            sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

            result = sim.create_world()

            assert result["status"] == "error", result
            assert "set_gravity" in result["content"][0]["text"]
            assert _FakeWorld.instance is None, "a raising stop() skipped clear_instance()"
            assert sim._world is None
        finally:
            del _FakeWorld.stop

    def test_a_raising_clear_instance_still_reports_the_original_failure(self, fake_isaacsim) -> None:
        _FakeWorld.fail_on_gravity = True
        original = _FakeWorld.__dict__["clear_instance"]

        def _raising_clear(cls: Any) -> None:
            raise AttributeError("clear_instance missing on this SDK version")

        _FakeWorld.clear_instance = classmethod(_raising_clear)  # type: ignore[assignment]
        try:
            sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

            result = sim.create_world()

            assert result["status"] == "error", result
            assert "set_gravity" in result["content"][0]["text"]
            assert sim._world is None, "the reference must be dropped even when cleanup raised"
        finally:
            _FakeWorld.clear_instance = original  # type: ignore[assignment]

    def test_a_raising_stop_still_reports_the_original_failure(self, fake_isaacsim) -> None:
        _FakeWorld.fail_on_gravity = True

        def _raising_stop(self: Any) -> None:
            raise RuntimeError("stop() on a half-built world")

        _FakeWorld.stop = _raising_stop  # type: ignore[method-assign]
        try:
            result = IsaacSimulation(config=IsaacConfig(render_mode="headless")).create_world()

            assert "set_gravity" in result["content"][0]["text"]
            assert "stop() on a half-built world" not in result["content"][0]["text"]
        finally:
            del _FakeWorld.stop


class TestASuccessfulCreateWorldIsUntouched:
    """The control: tearing down on the success path would break everything."""

    def test_the_world_is_neither_stopped_nor_cleared(self, fake_isaacsim) -> None:
        sim = IsaacSimulation(config=IsaacConfig(render_mode="headless"))

        assert sim.create_world()["status"] == "success"

        world = _FakeWorld.built[0]
        assert not world.stopped
        assert not world.instance_cleared
        assert sim._world is world
        assert sim._world_created is True
