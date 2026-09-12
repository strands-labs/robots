"""``replicate`` clones the scene, and refuses rather than reporting a fleet it did not build.

The method was a stub that reported success for doing nothing. Its body was a
comment between two clock reads:

```python
t0 = time.perf_counter()
# In full implementation: use omni.isaac.cloner.Cloner
# to replicate the scene N times
self._replicated = True
self._num_envs_active = n
elapsed = time.perf_counter() - t0
```

so the "Build time" it quoted was the duration of two assignments. Measured on
live Isaac Sim 6.0.1 (A10G), `replicate(64)` reported::

    Replicated to 64 environments. Build time: 0ms. Device: cuda:0.

with the stage's prim count unchanged at 69, `get_state()` reporting
`num_envs: 64`, and `add_robot` refused from then on with "Cannot add robots
after replicate(). Call destroy() first." - so the no-op also locked the caller
out of the scene it had not replicated.

The module it named does not exist on 6.x. `omni.isaac.cloner` raises
`ModuleNotFoundError`; the cloner is `isaacsim.core.cloner`, exposing `Cloner`
and `GridCloner`. Cloning through it works: in the same session a
`GridCloner(spacing=1.5)` clone of one robot into four environment paths created
**45 prims** and returned the grid offsets it applied.

Three properties are pinned here, and the third is the one that matters most:

* the clone actually happens, with the source prims and target paths the caller's
  scene implies;
* every refusal leaves the simulation **un-replicated**, so a retry is possible
  and `add_robot` is not refused on the strength of a failed clone;
* a cloner that returns without error and grows the stage by nothing is
  **refused**, not reported. That is the exact shape the stub had, so a future
  regression to it fails here rather than shipping.

Scope: the Kit leaves are stood in. The vendor cloner has a signature but no
behavioural contract this repository can pin, so the real one is exercised on GPU;
what is graded here is what this method does with what the cloner reports.
"""

from __future__ import annotations

import math
import sys
import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _ObjectState,
    _RobotState,
)


def _engine(objects: bool = False) -> Any:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig()
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._robots = {"arm": _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])}
    engine._objects = {}
    if objects:
        engine._objects["cube"] = _ObjectState(
            name="cube", prim_path="/World/Objects/cube", shape="box", is_static=False
        )
    engine._cameras = {}
    engine._action_controllers = {}
    engine._replicated = False
    engine._num_envs_active = 1
    return engine


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


class TestTheCloneActuallyHappens:
    def test_the_cloner_is_driven_with_the_scene_prims(self, cloner) -> None:
        engine = _engine(objects=True)

        result = engine.replicate(4)

        assert result["status"] == "success", result
        assert len(cloner.instances) == 1
        instance = cloner.instances[0]
        # Both the robot and the object are cloned: an environment is the scene,
        # not just its articulation.
        cloned_sources = sorted(call["source_prim_path"] for call in instance.clones)
        assert cloned_sources == ["/World/Objects/cube", "/World/Robots/arm"]
        # env_0 is the scene already on the stage, so only 1..n-1 are built.
        for call in instance.clones:
            leaf = call["source_prim_path"].rsplit("/", 1)[-1]
            assert call["prim_paths"] == [f"/World/envs/env_{i}/{leaf}" for i in (1, 2, 3)]
        assert instance.base_envs == ["/World/envs"]

    def test_physics_is_replicated_and_collisions_filtered(self, cloner) -> None:
        """A fleet whose environments collide is not N independent episodes."""
        engine = _engine()

        result = engine.replicate(4)

        instance = cloner.instances[0]
        assert all(call["replicate_physics"] is True for call in instance.clones)
        assert instance.filters, "inter-environment collisions were never filtered"
        assert _payload(result)["physics_replicated"] is True
        assert _payload(result)["collisions_filtered"] is True

    def test_each_environment_scope_is_defined_before_cloning_into_it(self, cloner) -> None:
        """The ordering the whole thing turns on, measured rather than assumed.

        The real cloner writes a clone only where the target's parent scope already
        exists, and does nothing - without raising - when it does not. Across six
        flag combinations (with and without ``replicate_physics``,
        ``copy_from_source``, ``base_env_path`` and ``root_path``) every call
        returned cleanly and created zero environment prims until the scopes were
        defined first. So this is not tidiness: reversing it silently produces an
        empty fleet.
        """
        engine = _engine()

        assert engine.replicate(4)["status"] == "success"

        stage = cloner.stage
        assert stage is not None
        defined = [path for path, _type in stage.defined]
        assert defined == ["/World/envs/env_1", "/World/envs/env_2", "/World/envs/env_3"], defined
        # And every scope existed before the clone that targets it.
        instance = cloner.instances[0]
        assert instance.clones, "nothing was cloned"
        for call in instance.clones:
            for target in call["prim_paths"]:
                scope = target.rsplit("/", 1)[0]
                assert scope in defined, f"cloned into {target} whose scope {scope} was never defined"

    def test_the_spacing_reaches_the_cloner(self, cloner) -> None:
        engine = _engine()

        assert engine.replicate(4, spacing=2.75)["status"] == "success"

        assert cloner.instances[0].spacing == pytest.approx(2.75)

    def test_the_report_counts_the_prims_that_appeared(self, cloner) -> None:
        """Not the number it was handed: the stub's whole defect was echoing back
        an argument as though it were an outcome."""
        cloner.per_clone = 5
        engine = _engine()

        payload = _payload(engine.replicate(4))

        # 3 clone targets x 5 prims each = 15, plus the 3 environment scopes this
        # method defines so the cloner has a parent to write under = 18. The field
        # is the measured stage delta, which is the point: it is not derivable from
        # the argument, so it cannot be echoed back the way the stub echoed n.
        assert payload["prims_created"] == 18
        assert payload["clones_created"] == 3
        assert payload["num_envs"] == 4
        assert payload["source_prims"] == 1

    def test_the_state_is_marked_replicated(self, cloner) -> None:
        engine = _engine()

        assert engine.replicate(4)["status"] == "success"

        assert engine._replicated is True
        assert engine._num_envs_active == 4

    def test_the_report_states_what_it_does_not_do(self, cloner) -> None:
        """The clones advance under physics but cannot be driven individually, and
        a caller reading ``num_envs: 64`` would otherwise assume they can."""
        engine = _engine()

        result = engine.replicate(4)

        text = _text(result)
        assert "get_observation" in text
        assert "send_action" in text
        assert _payload(result)["per_env_action_api"] is False

    def test_the_build_time_is_a_finite_non_negative_duration(self, cloner) -> None:
        """Named for what it grades, which is less than it might appear.

        A fake cloner returns in microseconds, so this cannot tell a real build
        time from the stub's two-assignment measurement - and it passes on the
        pre-fix code for exactly that reason. Whether the number reflects real
        work is settled on GPU, where the same call takes hundreds of
        milliseconds. What is worth pinning here is only that the field is a
        usable duration rather than absent, negative or NaN.
        """
        engine = _engine()

        payload = _payload(engine.replicate(4))

        assert payload["build_time_ms"] >= 0.0
        assert math.isfinite(payload["build_time_ms"])


class TestOneEnvironmentIsAnHonestNoOp:
    """The scene is already one environment, so there is nothing to clone."""

    def test_it_succeeds_without_cloning(self, cloner) -> None:
        engine = _engine()

        result = engine.replicate(1)

        assert result["status"] == "success", result
        assert cloner.instances == [], "a no-op reached the cloner"
        assert _payload(result)["clones_created"] == 0
        assert _payload(result)["prims_created"] == 0

    def test_it_does_not_lock_out_add_robot(self, cloner) -> None:
        """The load-bearing half. The stub set ``_replicated`` unconditionally, so
        a call that cloned nothing refused every later ``add_robot``."""
        engine = _engine()

        assert engine.replicate(1)["status"] == "success"

        assert engine._replicated is False
        assert engine._num_envs_active == 1

    def test_the_config_default_of_one_is_the_same_no_op(self, cloner) -> None:
        engine = _engine()

        assert engine.replicate()["status"] == "success"

        assert engine._replicated is False
        assert cloner.instances == []


class TestTheCountAndSpacingDomains:
    @pytest.mark.parametrize("bad", [0, -1, -64, 2.7, float("nan"), float("inf"), True, "4", [4]])
    def test_an_unusable_count_is_refused(self, cloner, bad: Any) -> None:
        """``IsaacConfig`` validates its own ``num_envs``; this argument bypassed
        that entirely, so a negative count reported success having built nothing
        and a bool passed as a count of one."""
        engine = _engine()

        result = engine.replicate(bad)

        assert result["status"] == "error", (bad, result)
        assert "num_envs" in _text(result)
        # Refused before any stage work, so nothing is half-built.
        assert cloner.instances == []
        assert engine._replicated is False
        assert engine._num_envs_active == 1

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "1.5", None])
    def test_an_unusable_spacing_is_refused(self, cloner, bad: Any) -> None:
        engine = _engine()

        result = engine.replicate(4, spacing=bad)

        assert result["status"] == "error", (bad, result)
        assert "spacing" in _text(result)
        assert cloner.instances == []
        assert engine._replicated is False

    def test_none_defers_to_the_config_rather_than_being_refused(self, cloner) -> None:
        """``None`` is the documented "use ``config.num_envs``" spelling, so it is
        not an unusable count - it is how the signature's own default is written.
        With a config of 16 it must clone 15, not refuse."""
        engine = _engine()
        engine._config = IsaacConfig(num_envs=16)

        result = engine.replicate(None)

        assert result["status"] == "success", result
        assert _payload(result)["num_envs"] == 16
        assert _payload(result)["clones_created"] == 15

    @pytest.mark.parametrize("good", [2, 64, 1024])
    def test_a_usable_count_is_accepted(self, cloner, good: int) -> None:
        engine = _engine()

        result = engine.replicate(good)

        assert result["status"] == "success", result
        assert _payload(result)["num_envs"] == good
        assert _payload(result)["clones_created"] == good - 1


class TestEveryFailureLeavesTheSimUnreplicated:
    """The stub's damage was as much the state it set as the text it printed."""

    def test_an_absent_cloner_is_refused(self, cloner, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "isaacsim.core.cloner", None)
        engine = _engine()

        result = engine.replicate(4)

        assert result["status"] == "error"
        text = _text(result)
        assert "isaacsim.core.cloner" in text
        assert "Kit extension" in text
        assert engine._replicated is False
        assert engine._num_envs_active == 1

    def test_a_raising_cloner_is_refused(self, cloner) -> None:
        cloner.fail_clone = RuntimeError("fabric out of memory")
        engine = _engine()

        result = engine.replicate(4)

        assert result["status"] == "error"
        assert "fabric out of memory" in _text(result)
        assert engine._replicated is False
        assert engine._num_envs_active == 1

    def test_a_cloner_that_creates_nothing_is_refused(self, cloner) -> None:
        """The stub exactly: no error, no clone, and a success envelope quoting the
        count it was handed. This is the cell that fails if it comes back.

        It is also the shape the real cloner produces when a target's parent scope
        is absent, measured across six flag combinations - so this is a live
        failure mode rather than a hypothetical one. And note what the guard cannot
        be: a stage-count comparison. This method defines an environment scope per
        clone, so the stage grows by 63 here while not one clone exists.
        """
        cloner.per_clone = 0
        engine = _engine()

        result = engine.replicate(64)

        assert result["status"] == "error", result
        text = _text(result)
        assert "not on the stage" in text
        assert "63" in text, "the refusal should name the clones that do not exist"
        # The paths are named, because a cloner that raised nothing gives a caller
        # no other handle on what went wrong.
        assert "/World/envs/env_1/arm" in text
        assert engine._replicated is False
        assert engine._num_envs_active == 1

    def test_a_failed_collision_filter_is_reported_not_hidden(self, cloner) -> None:
        """Not fatal - the clones exist - but a fleet whose environments collide
        is a materially different thing, so the caller is told which they got."""
        cloner.fail_filter = RuntimeError("no physics scene")
        engine = _engine()

        result = engine.replicate(4)

        assert result["status"] == "success", result
        assert _payload(result)["collisions_filtered"] is False
        assert "WARNING" in _text(result)
        assert "collide" in _text(result)


class TestThePreconditionsAreUnchanged:
    """Controls: the two refusals this method already had still fire, and first."""

    def test_no_world_is_still_refused(self, cloner) -> None:
        engine = _engine()
        engine._world_created = False

        result = engine.replicate(4)

        assert result["status"] == "error"
        assert "No world created." in _text(result)
        assert cloner.instances == []

    def test_no_robot_is_still_refused(self, cloner) -> None:
        engine = _engine()
        engine._robots = {}

        result = engine.replicate(4)

        assert result["status"] == "error"
        assert "at least one robot" in _text(result)
        assert cloner.instances == []


class TestThePhysicsSceneIsFoundNotAssumed:
    """A wrong physics-scene path degrades the fleet silently.

    ``filter_collisions`` raises ``RuntimeError: Accessed schema on invalid prim``
    for a path that is not a ``UsdPhysics.Scene``, and that failure is deliberately
    non-fatal - so a hardcoded path that is wrong does not fail the clone, it
    quietly produces a fleet whose environments push each other around. Measured on
    Isaac Sim 6.0.1: the scene is at ``/physicsScene``, at the STAGE ROOT, and
    ``/World/physicsScene`` - the intuitive spelling under the configured
    ``stage_path``, and the one this method shipped with - is invalid.
    """

    def test_the_scene_is_discovered_from_the_stage(self, cloner) -> None:
        from strands_robots.simulation.isaac.simulation import _physics_scene_path

        class _PhysPrim:
            def __init__(self, path: str, is_scene: bool) -> None:
                self._path, self._is_scene = path, is_scene

            def GetPath(self) -> Any:  # noqa: N802 - USD API spelling
                return types.SimpleNamespace(pathString=self._path)

            def IsA(self, schema: Any) -> bool:  # noqa: N802 - USD API spelling
                return self._is_scene

        prims = [_PhysPrim("/World", False), _PhysPrim("/somewhere/else/myScene", True)]
        found = _physics_scene_path(types.SimpleNamespace(Traverse=lambda: prims))

        # pxr may be absent in the unit environment, in which case the documented
        # fallback is what is returned; either way it must never be the wrong
        # ``{stage_path}/physicsScene`` guess.
        assert found in ("/somewhere/else/myScene", "/physicsScene")
        assert found != "/World/physicsScene"

    def test_the_fallback_is_the_measured_location(self) -> None:
        from strands_robots.simulation.isaac.simulation import (
            _DEFAULT_PHYSICS_SCENE_PATH,
            _physics_scene_path,
        )

        assert _DEFAULT_PHYSICS_SCENE_PATH == "/physicsScene"
        # A stage that cannot be traversed falls back rather than raising: this runs
        # inside the collision-filter try block, so raising here would turn a
        # degradation into a failed clone.
        assert _physics_scene_path(object()) == "/physicsScene"

    def test_the_source_does_not_hardcode_the_stage_path_spelling(self) -> None:
        import inspect

        source = inspect.getsource(IsaacSimulation.replicate)
        assert "_physics_scene_path(stage)" in source
        assert "stage_path}/physicsScene" not in source


class TestTheStaleClonerNameIsGone:
    def test_the_source_does_not_reach_for_the_4x_module(self) -> None:
        """``omni.isaac.cloner`` does not exist on Isaac Sim 6.x - measured,
        ``ModuleNotFoundError``. A mention in prose explaining that is fine; an
        import is not."""
        import inspect

        source = inspect.getsource(IsaacSimulation.replicate)
        assert "from isaacsim.core.cloner import" in source
        assert "import omni.isaac.cloner" not in source
