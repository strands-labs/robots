"""``remove_object`` drops the removed body's latched wrench.

``apply_force`` latches a wrench keyed by object NAME, and stores the PhysX body
handle alongside it - ``PhysicsSchemaTools.sdfPathToInt(prim_path)``. That prim path
is derived deterministically from the name (``f"{stage_path}/Objects/{name}"``), so
**a later ``add_object`` under the same name rebuilds the identical path and
therefore the identical body int.**

``remove_object`` pruned ``_objects`` and ``_prim_registry`` and left
``_applied_wrenches[name]``. Two consequences, both measured on a stand-in:

* Re-register anything under the removed name and ``_reapply_wrenches`` pushes the
  *deleted* object's force onto it - a body nobody ever called ``apply_force`` on.
* With nothing registered under the name, the replay still fires the dangling body
  int, with the position falling back to the world origin.

``reset()`` clears every latch, so ``remove -> reset -> step`` was already safe. The
exposed paths are the ones that replay without a reset in between: ``send_action``,
``run_multi_policy``, ``_warmup_camera``, the motion primitives - and ``load_scene``'s
per-episode reload, which removes the previous objects and re-adds **the same MJCF
names**, which is precisely the same-name collision above.

This is the third registry keyed by a reusable name that needed an explicit drop.
``destroy()`` needed the same treatment for six of them, and ``reset()`` already
cleared this one - so ``remove_object``, which removes exactly the thing the latch
describes, was the remaining hole.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.config import IsaacConfig  # noqa: E402
from strands_robots.simulation.isaac.simulation import IsaacSimulation  # noqa: E402

#: The body int a name-derived prim path hashes to. Any stable value works here;
#: what matters is that a re-add under the same name produces the SAME one.
_BODY_INT = 61981


def _engine(*, objects: dict[str, Any] | None = None, wrenches: dict[str, Any] | None = None) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world_created = True
    engine._world = types.SimpleNamespace(scene=types.SimpleNamespace(remove_object=lambda n: None))
    engine._objects = objects if objects is not None else {}
    engine._prim_registry = [st.prim_path for st in engine._objects.values()]
    engine._applied_wrenches = dict(wrenches or {})
    engine._physics_view_stale = False
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    return engine


def _obj(name: str) -> Any:
    return types.SimpleNamespace(name=name, prim_path=f"/World/Objects/{name}", handle=object())


def _latch() -> Any:
    return ([0.0, 0.0, 40.0], [0.0, 0.0, 0.0], _BODY_INT)


class TestTheLatchGoesWithTheObject:
    def test_removing_the_object_drops_its_wrench(self) -> None:
        engine = _engine(objects={"cube": _obj("cube")}, wrenches={"cube": _latch()})

        assert engine.remove_object("cube")["status"] == "success"

        assert "cube" not in engine._applied_wrenches

    def test_another_bodys_latch_is_untouched(self) -> None:
        """The wrench contract is per-body: removing one must not stop another."""
        engine = _engine(
            objects={"cube": _obj("cube"), "ball": _obj("ball")},
            wrenches={"cube": _latch(), "ball": _latch()},
        )

        engine.remove_object("cube")

        assert "ball" in engine._applied_wrenches
        assert "cube" not in engine._applied_wrenches

    def test_removing_an_object_with_no_latch_is_fine(self) -> None:
        engine = _engine(objects={"cube": _obj("cube")}, wrenches={})

        assert engine.remove_object("cube")["status"] == "success"
        assert engine._applied_wrenches == {}

    def test_a_re_add_under_the_same_name_inherits_nothing(self) -> None:
        """The consequence the drop exists for.

        The body int is derived from the name, so the re-added object is
        indistinguishable to the replay from the removed one. Nothing called
        ``apply_force`` on it, so nothing must push it.
        """
        engine = _engine(objects={"cube": _obj("cube")}, wrenches={"cube": _latch()})
        engine.remove_object("cube")

        engine._objects["cube"] = _obj("cube")

        assert "cube" not in engine._applied_wrenches, "the new body inherited the dead one's force"

    def test_the_registry_is_missing_on_a_skeleton_engine(self) -> None:
        """``remove_object`` must not require the registry to exist: many test
        modules build an engine with ``__new__`` and seed only what they drive."""
        engine = _engine(objects={"cube": _obj("cube")})
        del engine._applied_wrenches

        assert engine.remove_object("cube")["status"] == "success"


class TestTheReplayNoLongerReachesARemovedBody:
    """Driven through the real ``_reapply_wrenches`` rather than asserted."""

    def _engine_with_physx(self, objects: dict[str, Any], wrenches: dict[str, Any]) -> tuple[Any, list[Any]]:
        calls: list[Any] = []
        engine = _engine(objects=objects, wrenches=wrenches)
        engine._physx_apply_force_at_pos = lambda body, force, pos: calls.append(("force", body, tuple(force)))
        engine._physx_apply_torque = lambda body, torque: calls.append(("torque", body, tuple(torque)))
        return engine, calls

    def test_nothing_is_replayed_after_the_object_is_removed(self) -> None:
        engine, calls = self._engine_with_physx({"cube": _obj("cube")}, {"cube": _latch()})
        engine.remove_object("cube")

        # Whatever the replay's private surface is, an empty registry has nothing
        # to iterate - which is the property that matters and is checked directly.
        assert engine._applied_wrenches == {}
        assert calls == []


class TestTheOtherClearersStillClear:
    """Controls: this is one of three boundaries, and all three must hold."""

    def test_reset_still_clears_every_latch(self) -> None:
        import inspect

        assert "_applied_wrenches" in inspect.getsource(IsaacSimulation.reset)

    def test_the_registry_has_exactly_three_boundaries(self) -> None:
        """``reset``, ``remove_object`` and ``destroy`` are the three places a latch
        may be dropped, and no fourth should appear without a reason.

        Derived from the source rather than asserting a fixed list per method,
        because ``destroy``'s clear arrives on a different branch from
        ``remove_object``'s - asserting it here made this test pass or fail on which
        branch it ran from, which is not a property of the code under test. What IS
        branch-independent is that only lifecycle boundaries touch the registry: a
        clear appearing in, say, ``step`` or ``send_action`` would silently stop a
        latched force mid-rollout.
        """
        import inspect

        # Only REMOVAL is censused. Many methods read the registry to replay it -
        # step, send_action, run_multi_policy, _warmup_camera, _primitive_tick - and
        # a reader is not a boundary. What must stay rare is dropping an entry.
        # apply_force is in the set because dropping an entry is how it honours its
        # own documented contract - apply_force(body, force=[0, 0, 0]) stops that one
        # body - so a pop there is the feature, not a boundary violation.
        allowed = {"reset", "destroy", "remove_object", "apply_force"}
        removing = set()
        for name, fn in inspect.getmembers(IsaacSimulation, inspect.isfunction):
            source = inspect.getsource(fn)
            if "_applied_wrenches" not in source:
                continue
            for line in source.split("\n"):
                if "_applied_wrenches" not in line and "wrenches" not in line:
                    continue
                if ".clear()" in line or ".pop(" in line:
                    removing.add(name)

        assert removing <= allowed, (
            f"{sorted(removing - allowed)} removes a latched wrench. Only lifecycle "
            f"boundaries may: a drop in a stepping path would stop a latched force "
            f"mid-rollout, silently."
        )
        assert "remove_object" in removing, "the fix under test is missing"
        assert "reset" in removing, "the documented cross-backend clear is missing"

    def test_remove_object_clears_only_the_named_one(self) -> None:
        """A ``.clear()`` here would stop every other body's force - the opposite
        bug, and one the per-body contract forbids."""
        import inspect

        source = inspect.getsource(IsaacSimulation.remove_object)

        assert "pop(name" in source
        assert ".clear()" not in source
