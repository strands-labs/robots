"""``destroy()`` clears the per-world state that would otherwise reach the next world.

``destroy()`` cleared seven registries - ``_robots``, ``_cameras``, ``_objects``,
``_prim_registry``, ``_action_controllers``, ``_cams_rec_state`` and
``_recording_state_dict`` - and left six other pieces of per-world state in place.
Measured on a torn-down engine, every one of these survived:

    _applied_wrenches   _obs_noise   _obs_noise_rng   _dr_base
    _frame_cache        _joint_cache

All of them are keyed by an object / robot / camera name or a prim path, and those
names are reused across worlds as a matter of course: a second world with a
``"cube"`` in it is the ordinary case, not a collision someone has to engineer. So
the next ``create_world()`` inherited configuration for a scene it was never given:

* a latched ``apply_force`` replayed onto the new world's body of the same name -
  an external force nobody applied, in a fresh world;
* ``set_obs_noise``'s per-robot sigma kept perturbing ``get_observation``, so a
  deliberately clean run silently carried noise from a previous one;
* ``randomize()``'s ``_dr_base`` first-touch baseline - which exists precisely to
  stop scaling compounding - anchored the new world's randomization to a pose from
  a world that no longer exists;
* ``_frame_cache`` held a full RTX frame and ``_joint_cache`` a joint snapshot from
  the old stage, readable as though current.

``_applied_wrenches`` is the sharpest case, because ``reset()`` already clears it -
the shared contract is "reset() clears every latched wrench in the world". So
``destroy()``, the stronger boundary of the two, was the one that did not.

The clears use ``getattr`` guards. ``destroy()`` runs from ``__del__``, and many
test modules build a skeleton engine with ``__new__`` seeding only the attributes
they exercise, so a missing attribute here would raise during garbage collection
and mask whatever the test was actually about.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: The state that must not outlive a world, and a non-empty value for each.
PER_WORLD_STATE: dict[str, dict[str, Any]] = {
    "_applied_wrenches": {"cube": {"force": [0.0, 0.0, 40.0], "torque": [0.0, 0.0, 0.0]}},
    "_obs_noise": {"arm": 0.05},
    "_dr_base": {"cube": [0.1, 0.2, 0.3]},
    "_frame_cache": {"cam": "a-frame-from-the-old-stage"},
    "_joint_cache": {"arm": {"j0": 1.23}},
}


def _engine() -> Any:
    """A torn-down-able engine seeded with per-world state in every registry."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world_created = True
    engine._world = types.SimpleNamespace(stop=lambda: None, clear_instance=lambda: None)
    engine._robots = {}
    engine._cameras = {}
    engine._objects = {}
    engine._prim_registry = []
    engine._action_controllers = {}
    engine._cams_rec_state = None
    engine._recording_state_dict = {}
    engine._num_envs_active = 1
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._replicated = False
    for name, value in PER_WORLD_STATE.items():
        setattr(engine, name, dict(value))
    engine._obs_noise_rng = object()
    return engine


class TestNoPerWorldStateSurvivesDestroy:
    @pytest.mark.parametrize("registry", sorted(PER_WORLD_STATE))
    def test_the_registry_is_emptied(self, registry: str) -> None:
        engine = _engine()
        assert getattr(engine, registry), "the fixture must seed it, or this proves nothing"

        assert engine.destroy()["status"] == "success"

        assert not getattr(engine, registry), f"{registry} survived destroy()"

    def test_the_noise_generator_is_dropped(self) -> None:
        """Not a mapping - a seeded stream that belongs to the destroyed world."""
        engine = _engine()

        engine.destroy()

        assert engine._obs_noise_rng is None

    def test_nothing_at_all_survives(self) -> None:
        """One assertion over the whole set, so adding a seventh registry to the
        fixture without adding it to destroy() fails here."""
        engine = _engine()

        engine.destroy()

        survivors = [name for name in PER_WORLD_STATE if getattr(engine, name, None)]
        assert survivors == [], f"{survivors} survived destroy()"


class TestTheRegistriesDestroyAlreadyClearedStillClear:
    """Controls: the new clears must not have displaced the old ones."""

    @pytest.mark.parametrize(
        "registry",
        ["_robots", "_cameras", "_objects", "_prim_registry", "_action_controllers"],
    )
    def test_the_original_registry_is_still_emptied(self, registry: str) -> None:
        engine = _engine()
        seeded = {"x": object()} if registry != "_prim_registry" else ["/World/x"]
        setattr(engine, registry, seeded)

        engine.destroy()

        assert not getattr(engine, registry)

    def test_the_world_is_still_torn_down(self) -> None:
        engine = _engine()

        engine.destroy()

        assert engine._world is None
        assert engine._world_created is False


class TestASkeletonEngineDoesNotRaise:
    """``destroy()`` runs from ``__del__``, so a missing attribute must not raise.

    Test modules build engines with ``__new__`` and seed only what they exercise.
    An unguarded ``self._applied_wrenches.clear()`` would raise ``AttributeError``
    during garbage collection, surfacing as noise attributed to whatever test was
    running rather than to the clear.
    """

    def test_destroy_succeeds_with_none_of_the_new_state_present(self) -> None:
        engine = _engine()
        for name in list(PER_WORLD_STATE) + ["_obs_noise_rng"]:
            delattr(engine, name)

        assert engine.destroy()["status"] == "success"

    @pytest.mark.parametrize("registry", sorted(PER_WORLD_STATE))
    def test_destroy_succeeds_with_any_one_of_them_absent(self, registry: str) -> None:
        engine = _engine()
        delattr(engine, registry)

        assert engine.destroy()["status"] == "success"


class TestDestroyIsNotWeakerThanReset:
    """Anything ``reset()`` clears, ``destroy()`` must clear too.

    A drift guard rather than a fixed list, and branch-independent: it derives both
    sides from the source. ``destroy()`` is the stronger boundary of the two - it
    tears the world down, where ``reset()`` only rewinds it - so a registry that
    ``reset`` wipes and ``destroy`` retains is incoherent whichever registry it is.

    That asymmetry is exactly how ``_applied_wrenches`` came to leak: the
    cross-backend contract is "reset() clears every latched wrench in the world",
    ``reset`` honoured it, and ``destroy`` did not - so a caller who trusted the
    weaker boundary got less from the stronger one.
    """

    #: Registries whose lifetime is deliberately wider than one world.
    _NOT_PER_WORLD = frozenset(
        {
            # The process-wide SimulationApp outlives destroy() by design, and the
            # renderer/pump configuration that describes it is construction state,
            # not scene state.
            "_config",
        }
    )

    def _cleared_by(self, method) -> set[str]:
        import inspect
        import re

        source = inspect.getsource(method)
        found = set()
        for match in re.finditer(r"self\.(_[a-z_]+)\.clear\(\)|self\.(_[a-z_]+) = \{\}", source):
            found.add(match.group(1) or match.group(2))
        # The loop-driven clears name their registries as string literals.
        for match in re.finditer(r'"(_[a-z_]+)"', source):
            found.add(match.group(1))
        return found - self._NOT_PER_WORLD

    def test_destroy_clears_everything_reset_clears(self) -> None:
        reset_clears = self._cleared_by(IsaacSimulation.reset)
        destroy_clears = self._cleared_by(IsaacSimulation.destroy)

        only_reset = sorted(reset_clears - destroy_clears)

        assert only_reset == [], (
            f"{only_reset} is cleared by reset() but not by destroy(). destroy() is the "
            f"stronger boundary, so it cannot wipe less - that asymmetry is how a latched "
            f"wrench survived into the next world."
        )

    def test_the_guard_can_actually_see_a_registry(self) -> None:
        """Otherwise a regex that matched nothing would make the guard vacuous."""
        assert self._cleared_by(IsaacSimulation.destroy), "the guard found no clears at all"
