"""``get_contacts``' per-step cache does not survive a rewind of the clock it keys on.

``get_contacts`` caches its translated contact list against ``_step_count`` so
several predicate evaluations in one step ask PhysX once. That is correct, and it is
pinned elsewhere. What was missing is that **``_step_count`` is not monotonic**:
``create_world``, ``reset`` and ``destroy`` all set it back to 0. A cache written at
step N before a rewind is then indistinguishable from one written at step N after
it, and the stale list is served whenever the two indices coincide.

Three ways that happens, all demonstrated:

* the cache was last written at step 0, so the very next query after a ``reset()``
  hits it;
* the cache was written at step N, and the first post-reset step to reach index N
  hits it - which for a policy-runner episode that ended on its **first** contact
  query is the *next* episode's first query;
* across ``destroy()`` and a new ``create_world()``, where a world holding no
  objects at all can report the previous world's object-ground pair without PhysX
  ever being asked.

The last is reachable in shipped code: ``PolicyRunner`` calls ``sim.reset()`` per
episode and evaluates a resolved success criterion after each step, and
``success_fn="contact"`` routes through the predicate DSL into ``get_contacts``. A
false contact success is a silently passing episode.

The fix is an **epoch** bumped by ``_rewind_clock()``, the one owner of that rewind,
with the cache keyed on ``(epoch, step_count)``. An epoch rather than a
``_contact_cache = None`` at each of the four rewind sites, because the epoch makes
it structural: a rewind added later inherits the invalidation by calling the owner,
instead of needing to remember a second line.

The MuJoCo sibling documents the same hazard from the other side -
``mujoco/rendering.py`` runs ``mj_forward`` first so contacts are fresh "even
immediately after ``reset``... (without this, stale contacts from the previous step
can appear as phantom penetrations at t=0)".
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: A contact record from a world that no longer exists.
_STALE = [{"geom1": "cube", "geom2": "CollisionPlane", "dist": -0.001, "pos": [0.0, 0.0, 0.0], "active": True}]


def _engine() -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world_created = True
    engine._world = types.SimpleNamespace(
        reset=lambda: None, stop=lambda: None, clear_instance=lambda: None, step=lambda **k: None
    )
    engine._robots = {}
    engine._cameras = {}
    engine._objects = {}
    engine._prim_registry = []
    engine._action_controllers = {}
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._num_envs_active = 1
    engine._replicated = False
    engine._applied_wrenches = {}
    engine._obs_noise = {}
    engine._obs_noise_rng = None
    engine._dr_base = {}
    engine._frame_cache = {}
    engine._joint_cache = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._revive_articulations_after_reset = lambda: None  # type: ignore[method-assign]
    engine._flush_open_episode_before_reset = lambda: None  # type: ignore[method-assign]
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    return engine


def _cache_would_hit(engine: Any) -> bool:
    """Whether ``get_contacts`` would serve the cached list at the current clock."""
    cache = getattr(engine, "_contact_cache", None)
    return cache is not None and cache[0] == (engine._contact_epoch, engine._step_count)


def _seed(engine: Any, step: int) -> None:
    engine._step_count = step
    engine._contact_cache = ((engine._contact_epoch, step), _STALE)


class TestARewindInvalidatesTheCache:
    def test_a_reset_then_the_same_step_index_does_not_hit(self) -> None:
        """The subtle one: the counter comes back around to the cached index."""
        engine = _engine()
        _seed(engine, 1)
        engine._step_count = 7

        assert engine.reset()["status"] == "success"
        engine._step_count = 1

        assert not _cache_would_hit(engine)

    def test_a_reset_with_the_cache_at_step_zero_does_not_hit(self) -> None:
        """The immediate one: step 0 before and step 0 after."""
        engine = _engine()
        _seed(engine, 0)
        engine._step_count = 42

        engine.reset()

        assert not _cache_would_hit(engine)

    def test_a_destroy_and_a_new_world_does_not_hit(self) -> None:
        """The worst one: a world holding no objects reporting a previous world's."""
        engine = _engine()
        _seed(engine, 0)

        engine.destroy()
        engine._world_created = True
        engine._world = types.SimpleNamespace()
        engine._objects = {}

        assert not _cache_would_hit(engine)

    @pytest.mark.parametrize("cached_at", [0, 1, 5, 99])
    def test_no_cached_index_survives_a_reset(self, cached_at: int) -> None:
        """Swept, because the defect was index-dependent - a single index could
        pass by luck."""
        engine = _engine()
        _seed(engine, cached_at)

        engine.reset()
        engine._step_count = cached_at

        assert not _cache_would_hit(engine)


class TestTheCacheStillWorksWithinAStep:
    """Controls. Invalidating always would satisfy every test above and undo the
    reason the cache exists - several predicates in one step asking PhysX once."""

    def test_the_same_step_hits(self) -> None:
        engine = _engine()
        _seed(engine, 3)

        assert _cache_would_hit(engine)

    def test_the_next_step_misses(self) -> None:
        engine = _engine()
        _seed(engine, 3)
        engine._step_count = 4

        assert not _cache_would_hit(engine)


class TestTheEpochHasOneOwner:
    """The rewind and the invalidation cannot be separated by a later edit."""

    def test_every_clock_rewind_goes_through_the_owner(self) -> None:
        """A raw ``self._step_count = 0`` anywhere else is a rewind that forgot to
        bump the epoch, which is exactly the bug this fixes."""
        import inspect

        source = inspect.getsource(IsaacSimulation)
        raw = [
            line.strip()
            for line in source.split("\n")
            if line.strip() == "self._step_count = 0" and "_rewind_clock" not in line
        ]
        owner_body = inspect.getsource(IsaacSimulation._rewind_clock)

        # The only raw rewind allowed is the one inside the owner itself.
        assert len(raw) == owner_body.count("self._step_count = 0") == 1, (
            f"{len(raw)} raw `self._step_count = 0` assignments; only _rewind_clock() may "
            f"perform one, so the epoch cannot be skipped"
        )

    def test_the_owner_bumps_the_epoch(self) -> None:
        engine = _engine()
        before = engine._contact_epoch

        engine._rewind_clock()

        assert engine._contact_epoch == before + 1

    def test_the_owner_also_zeroes_the_clock(self) -> None:
        engine = _engine()
        engine._sim_time = 1.25
        engine._step_count = 30

        engine._rewind_clock()

        assert engine._sim_time == 0.0
        assert engine._step_count == 0

    def test_the_epoch_is_a_class_default(self) -> None:
        """Skeleton engines built with ``__new__`` never run ``__init__``, and the
        cache comparison reads this - an instance-only attribute would raise
        ``AttributeError`` inside ``get_contacts`` for 20-plus test modules."""
        assert IsaacSimulation._contact_epoch == 0

    def test_the_destroy_drift_guard_cannot_see_this_kind_of_clear(self) -> None:
        """Recorded because it is a real limit of a sibling guard, not a defect here.

        ``test_destroy_clears_the_state_that_outlives_a_world`` derives its sets by
        matching ``self._x.clear()`` and ``self._x = {}``. An epoch bump is neither,
        so that guard is blind to this invalidation - which is why the epoch is
        pinned by the tests in this file rather than left to it.
        """
        import inspect

        guard = inspect.getsource(IsaacSimulation.destroy)

        assert "_rewind_clock" in guard, "destroy must still route its rewind through the owner"
        assert "_contact_cache" not in guard, "the invalidation is by epoch, not by clearing here"
