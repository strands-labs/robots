"""Regression pins for the Isaac Sim 6.0.x deferred-physics + camera-warmup fixes.

Both defects were found running the LIBERO Isaac driver (now
an Isaac driver) end-to-end on
the pip ``isaacsim`` wheels (verified on 6.0.0.1; 6.0.1.0 fails identically by
mechanism) and each has an isolated live repro recorded in the fixing PR:

1. **Deferred-physics guard defeated.** ``timeline.stop()`` invalidates the
   physics-tensor view *asynchronously* on Isaac Sim 6.0.x:
   ``SimulationManager.get_physics_sim_view()`` is still non-``None`` when
   ``stop()`` returns, so ``RigidPrim.__init__``'s eager velocity query still
   fires for a prim outside the view and raises the bare
   ``Exception("Failed to get rigid body velocities from backend")`` the #159
   guard exists to prevent - killing every LIBERO ``load_scene``. The fix makes
   ``_stop_timeline_for_deferred_physics`` ALSO call
   ``SimulationManager.invalidate_physics()`` (upstream's documented
   manual-invalidation entry point), which tears the view down synchronously.

2. **Camera warm-up under a stopped timeline.** After the deferred-physics
   window the timeline stays stopped and ``world.step()`` does not resume
   play, so a freshly added camera's RTX render product never accumulates a
   frame and ``_warmup_camera`` burns its whole budget stepping a dead
   renderer (the ``video.wrist_image``-missing failure on the LIBERO GR00T
   path). The fix re-asserts ``timeline.play()`` before EVERY warm-up
   iteration - per-iteration, not once, because stop/play land asynchronously
   on kit update ticks and a queued ``stop()`` can undo a single pre-loop
   resume.

The tests drive the REAL ``IsaacSimulation`` methods on a ``__new__`` skeleton
(the pattern of ``test_dataset_recording.py``) with fake ``omni.timeline`` /
``isaacsim.core.simulation_manager`` module trees injected via
``monkeypatch.setitem(sys.modules, ...)`` (the pattern of
``test_backend_parity.py``), so they run with or without a real kit install.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation


class _FakeTimeline:
    """Stub ``omni.timeline`` interface with async-landing stop/play semantics.

    ``stop()`` / ``play()`` only *queue* the state change; ``tick()`` applies
    it - mirroring the kit behaviour observed live on 6.0.x where
    ``is_playing()`` reports stale state right after a queued command.
    """

    def __init__(self, playing: bool = True) -> None:
        self._playing = playing
        self._queued: list[bool] = []
        self.stop_calls = 0
        self.play_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self._queued.append(False)

    def play(self) -> None:
        self.play_calls += 1
        self._queued.append(True)

    def is_playing(self) -> bool:
        return self._playing

    def tick(self) -> None:
        """Land every queued state change (one kit update)."""
        while self._queued:
            self._playing = self._queued.pop(0)


class _FakeSimulationManager:
    """Stub ``SimulationManager`` whose view outlives ``timeline.stop()``.

    The live 6.0.x behaviour this pins: stopping the timeline does NOT clear
    the view synchronously; only ``invalidate_physics()`` does.
    """

    view: object | None = object()
    invalidate_calls: int = 0

    @classmethod
    def get_physics_sim_view(cls) -> object | None:
        return cls.view

    @classmethod
    def invalidate_physics(cls) -> None:
        cls.invalidate_calls += 1
        cls.view = None

    @classmethod
    def reset(cls, *, view_live: bool) -> None:
        cls.view = object() if view_live else None
        cls.invalidate_calls = 0


@pytest.fixture
def fake_timeline(monkeypatch) -> _FakeTimeline:
    """Inject a fake ``omni.timeline`` module tree; return the timeline stub."""
    timeline = _FakeTimeline(playing=True)
    omni = types.ModuleType("omni")
    omni_timeline = types.ModuleType("omni.timeline")
    setattr(omni_timeline, "get_timeline_interface", lambda: timeline)  # noqa: B010 - ModuleType has no such attr statically
    setattr(omni, "timeline", omni_timeline)  # noqa: B010
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.timeline", omni_timeline)
    return timeline


@pytest.fixture
def fake_simulation_manager(monkeypatch) -> type[_FakeSimulationManager]:
    """Inject a fake ``isaacsim.core.simulation_manager`` module tree."""
    _FakeSimulationManager.reset(view_live=True)
    mods = {}
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.simulation_manager"):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    setattr(mods["isaacsim.core.simulation_manager"], "SimulationManager", _FakeSimulationManager)  # noqa: B010
    setattr(mods["isaacsim"], "core", mods["isaacsim.core"])  # noqa: B010
    setattr(mods["isaacsim.core"], "simulation_manager", mods["isaacsim.core.simulation_manager"])  # noqa: B010
    return _FakeSimulationManager


class TestStopTimelineForDeferredPhysics:
    """The #159 guard must tear the physics view down SYNCHRONOUSLY."""

    def test_invalidates_live_view_after_timeline_stop(self, fake_timeline, fake_simulation_manager) -> None:
        """Live view + stop() alone would leave RigidPrim's eager query armed.

        Pre-fix behaviour: the guard called ``timeline.stop()`` and returned
        with ``get_physics_sim_view()`` still non-``None`` (stop lands async),
        so constructing a ``Dynamic*`` prim raised the bare backend Exception.
        Post-fix: the guard calls ``invalidate_physics()`` and the view is
        ``None`` when the guard returns - no tick needed.
        """
        IsaacSimulation._stop_timeline_for_deferred_physics()

        assert fake_timeline.stop_calls == 1
        assert fake_simulation_manager.invalidate_calls == 1
        assert fake_simulation_manager.get_physics_sim_view() is None

    def test_no_view_means_no_invalidate_call(self, fake_timeline, fake_simulation_manager) -> None:
        """A dead view is left alone (scene-build path before any reset())."""
        fake_simulation_manager.reset(view_live=False)

        IsaacSimulation._stop_timeline_for_deferred_physics()

        assert fake_timeline.stop_calls == 1
        assert fake_simulation_manager.invalidate_calls == 0

    def test_missing_simulation_manager_is_best_effort(self, fake_timeline, monkeypatch) -> None:
        """Older/partial installs without the manager API must not raise."""
        for name in ("isaacsim", "isaacsim.core", "isaacsim.core.simulation_manager"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(
            "builtins.__import__",
            _blocking_import("isaacsim"),
        )

        IsaacSimulation._stop_timeline_for_deferred_physics()  # must not raise

        assert fake_timeline.stop_calls == 1


def _blocking_import(prefix: str):
    """An ``__import__`` that raises ImportError for ``prefix``-rooted modules."""
    real_import = __import__

    def _import(name, *args, **kwargs):
        if name == prefix or name.startswith(prefix + "."):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    return _import


class _WarmupWorld:
    """Stub Isaac ``World`` that renders only while the timeline plays.

    Mirrors the live failure: with the timeline stopped, stepping never feeds
    the RTX render product, so the camera never yields a frame.
    """

    def __init__(self, timeline: _FakeTimeline, frames_until_ready: int = 2) -> None:
        self._timeline = timeline
        self._remaining = frames_until_ready
        self.render_steps_while_playing = 0

    def step(self, render: bool = False) -> None:  # noqa: ARG002 - parity
        # Kit update: queued timeline commands land on the step boundary.
        self._timeline.tick()
        if self._timeline.is_playing():
            self.render_steps_while_playing += 1
            if self._remaining > 0:
                self._remaining -= 1

    @property
    def camera_ready(self) -> bool:
        return self._remaining == 0


class _FakeApp:
    """Stub ``SimulationApp`` whose ``update()`` is render-only.

    ``_refresh_all_render_products`` prefers ``SimulationApp.update()`` and
    falls back to ``world.step(render=True)``. The real ``update()`` flushes
    the secondary render products WITHOUT advancing physics, so this double
    only counts calls: it must not tick the timeline or advance camera
    readiness, or a secondary flush would masquerade as a warm-up step.
    """

    def __init__(self) -> None:
        self.update_calls = 0

    def update(self) -> None:
        self.update_calls += 1


def _skeleton_sim(
    world: _WarmupWorld,
    *,
    camera_names: tuple[str, ...] = ("front",),
) -> IsaacSimulation:
    """``__new__`` skeleton carrying every attribute ``_warmup_camera`` reads.

    ``camera_names`` populates the camera registry, which the loop *sizes* to
    decide whether the secondary render products need an explicit flush
    (``len(self._cameras) > 1``). Pass the real two-camera shape - a
    pre-existing ``image`` plus the LIBERO adapter's late ``wrist_image`` - to
    exercise that branch; a single name keeps the primary-product-only path.

    Every attribute here is one the real ``__init__`` always sets, so the
    skeleton stays a faithful stand-in rather than a shape the production code
    is bent to accept: the loop reads ``_cameras``, ``_refresh_all_render_products``
    reads ``_world_created`` and ``_app``, and ``__repr__`` reads
    ``_world_created`` (an incomplete skeleton makes every assertion failure in
    this file carry a misleading ``AttributeError`` from ``repr()``).
    """
    sim = IsaacSimulation.__new__(IsaacSimulation)
    sim._config = IsaacConfig(headless=True, num_envs=1)
    sim._world = world
    sim._world_created = True
    sim._sim_time = 0.0
    sim._step_count = 0
    # Values are irrelevant - the loop only takes len() - but the keys mirror
    # the live registry so a reader sees which scenario is under test.
    sim._cameras = dict.fromkeys(camera_names, None)  # type: ignore[assignment]
    sim._app = _FakeApp()  # type: ignore[assignment]

    def _render(camera_name: str = "default", width=None, height=None):  # noqa: ANN001, ARG001
        if world.camera_ready:
            return {"status": "success", "content": []}
        return {"status": "error", "content": [{"text": "render product not accumulated"}]}

    sim.render = _render  # type: ignore[method-assign]
    return sim


class TestWarmupCameraResumesTimeline:
    """``_warmup_camera`` must not step a dead renderer (stopped timeline)."""

    def test_resumes_stopped_timeline_and_warms_up(self, fake_timeline) -> None:
        """The load_scene aftermath: timeline stopped, camera just added.

        Pre-fix behaviour: every warm-up step ran with the timeline stopped,
        the render product never accumulated, and the loop exhausted its
        budget (the ``video.wrist_image``-missing GR00T failure). Post-fix
        the loop resumes play and the camera warms up within budget.
        """
        fake_timeline._playing = False
        world = _WarmupWorld(fake_timeline, frames_until_ready=2)
        sim = _skeleton_sim(world, camera_names=("image", "wrist_image"))

        assert sim._warmup_camera("wrist_image", n_steps=5) is True
        assert fake_timeline.play_calls >= 1
        assert world.render_steps_while_playing >= 2
        # The late second camera is exactly the #1802 shape, so the loop must
        # also have flushed the secondary render products.
        assert sim._app.update_calls >= 1

    def test_reasserts_play_when_queued_stop_lands_mid_loop(self, fake_timeline) -> None:
        """A stale ``is_playing() == True`` with a stop in flight.

        This is the observed 6.0.x trap: right after the deferred-physics
        guard, ``is_playing()`` still reports ``True`` while the queued
        ``stop()`` lands on the next kit update - undoing any single pre-loop
        resume. The per-iteration re-assert converges anyway.
        """
        fake_timeline._playing = True
        fake_timeline.stop()  # queued; lands on the first world.step tick
        world = _WarmupWorld(fake_timeline, frames_until_ready=2)
        sim = _skeleton_sim(world, camera_names=("image", "wrist_image"))

        assert sim._warmup_camera("wrist_image", n_steps=5) is True
        assert fake_timeline.play_calls >= 1

    def test_playing_timeline_needs_no_resume(self, fake_timeline) -> None:
        """Steady-state add_camera: timeline already playing -> no play() call."""
        fake_timeline._playing = True
        world = _WarmupWorld(fake_timeline, frames_until_ready=1)
        sim = _skeleton_sim(world)

        assert sim._warmup_camera("front", n_steps=3) is True
        assert fake_timeline.play_calls == 0


class _RaisingWorld(_WarmupWorld):
    """``_WarmupWorld`` whose ``step`` raises on the ``raise_on``-th call.

    Models the failure the loop's own comment anticipates - "stepping /
    rendering a partially-initialised stage can raise on surface drift" - so
    the loop breaks with budget left over.
    """

    def __init__(self, timeline: _FakeTimeline, *, raise_on: int, exc: Exception) -> None:
        super().__init__(timeline, frames_until_ready=10**6)
        self._raise_on = raise_on
        self._exc = exc
        self.step_calls = 0

    def step(self, render: bool = False) -> None:
        self.step_calls += 1
        if self.step_calls == self._raise_on:
            raise self._exc
        super().step(render=render)


def _warnings(caplog) -> str:  # noqa: ANN001 - pytest fixture
    """Every WARNING message the warm-up emitted, joined."""
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


class TestWarmupCameraFailureReportNamesTheCause:
    """A ``False`` must say WHICH failure happened - the remedies differ.

    An exhausted budget means the render product never accumulated, so
    stepping/waiting longer can help. An early abort means the loop stopped on
    an exception with budget left, so waiting cannot help until that exception
    is fixed. Reporting an abort with the exhaustion text (the pre-fix
    behaviour, with the exception itself DEBUG-only) points the operator at
    the renderer for what is really a surface or attribute error.
    """

    def test_an_abort_names_the_step_reached_and_the_cause(self, fake_timeline, caplog) -> None:
        """Abort on step 1 of 5: report the step reached and the exception."""
        fake_timeline._playing = True
        world = _RaisingWorld(fake_timeline, raise_on=1, exc=AttributeError("no attribute '_cameras'"))
        sim = _skeleton_sim(world)

        with caplog.at_level(logging.WARNING):
            assert sim._warmup_camera("front", n_steps=5) is False

        text = _warnings(caplog)
        assert "aborted on step 1 of 5" in text
        assert "AttributeError" in text
        assert "no attribute '_cameras'" in text
        # It must NOT claim the render product was given the whole budget.
        assert "did not produce a valid frame" not in text

    def test_the_reported_step_count_is_the_steps_taken(self, fake_timeline, caplog) -> None:
        """A mid-loop abort reports where it stopped, not the budget."""
        fake_timeline._playing = True
        world = _RaisingWorld(fake_timeline, raise_on=3, exc=RuntimeError("stage surface drift"))
        sim = _skeleton_sim(world)

        with caplog.at_level(logging.WARNING):
            assert sim._warmup_camera("front", n_steps=6) is False

        assert "aborted on step 3 of 6" in _warnings(caplog)

    def test_an_exhausted_budget_still_points_at_the_render_product(self, fake_timeline, caplog) -> None:
        """No exception, never ready: the original exhaustion report stands."""
        fake_timeline._playing = True
        world = _WarmupWorld(fake_timeline, frames_until_ready=10**6)
        sim = _skeleton_sim(world)

        with caplog.at_level(logging.WARNING):
            assert sim._warmup_camera("front", n_steps=3) is False

        text = _warnings(caplog)
        assert "did not produce a valid frame after 3 warm-up step(s)" in text
        assert "aborted" not in text
        assert world.render_steps_while_playing == 3, "the whole budget must really have been spent"


class TestWarmupCameraSecondaryRenderProducts:
    """The ``len(self._cameras) > 1`` flush from #1802, exercised.

    ``world.step(render=True)`` reliably refreshes only the PRIMARY render
    product, so a camera added next to an existing one needs an explicit
    secondary flush before the frame check. The registry size is what selects
    that branch.
    """

    def test_a_second_camera_flushes_the_secondary_render_products(self, fake_timeline) -> None:
        """Two registered cameras -> the secondary products are flushed."""
        fake_timeline._playing = True
        world = _WarmupWorld(fake_timeline, frames_until_ready=2)
        sim = _skeleton_sim(world, camera_names=("image", "wrist_image"))

        assert sim._warmup_camera("wrist_image", n_steps=5) is True
        assert sim._app.update_calls >= 1

    def test_a_single_camera_needs_no_secondary_flush(self, fake_timeline) -> None:
        """One registered camera -> stepping alone refreshes it, no flush."""
        fake_timeline._playing = True
        world = _WarmupWorld(fake_timeline, frames_until_ready=2)
        sim = _skeleton_sim(world, camera_names=("front",))

        assert sim._warmup_camera("front", n_steps=5) is True
        assert sim._app.update_calls == 0
