"""Contract tests for what a *rejected* ``patch_scene_mjcf`` batch leaves behind.

``patch_scene_mjcf`` applies a batch of structured ops to the live ``MjSpec``
atomically: the batch mutates the live spec and a snapshot taken beforehand is
put back if any op raises, so the caller is left with a usable world rather than
a half-patched one.

``tests/simulation/mujoco/test_patch_scene_mjcf.py`` pins that the *compiled
model* is unchanged after a rejected batch. That assertion is nearly free for a
batch rejected by one of its *ops*, so the property that actually makes the world
usable again is unpinned by it: the **live spec** must be the clean snapshot. Nothing else undoes the ops that already ran, and the spec is
what the next mutation recompiles from, so a batch that only restored the
compiled model would hand the next ``add_object`` an orphan body.

A batch can also be rejected by the *compiler* rather than by an op: every op
applies, and the model they add up to is one MuJoCo refuses. This module used to
record the assumption that "a rejected batch never reaches the recompile", which
was true of every rejection it constructed and false of the class -- that
recompile ran outside the rollback, so the whole batch stayed on the live spec
and every later mutation recompiled it, turning one refused batch into a world
that never worked again. ``{"op": "add_site", "size": [0.0]}`` is such a batch:
the size is finite, so it clears the op-level domain, and MuJoCo refuses it at
compile time.

This module pins what a caller can observe after a rejection of either kind:

1. The message names **which** op failed, so a batch is debuggable.
2. The live spec carries none of the ops that already ran, and a subsequent
   mutation therefore recompiles cleanly.
3. The world is still steppable.
4. A batch the compiler refuses is rolled back on the same terms, and the
   recompile that refused it runs inside the rollback's scope.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

pytest.importorskip("mujoco")


from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A batch whose first op is valid and whose second is not, so a rejection has
#: something half-applied to roll back.
REJECTED_BATCH: list[dict[str, Any]] = [
    {"op": "add_body", "name": "doomed", "pos": [0, 0, 1]},
    {"op": "totally_made_up", "name": "whatever"},
]


#: A batch whose ops all apply and whose compiled model MuJoCo then refuses. A
#: zero ``size`` is finite, so it passes the op-level ``size`` domain and reaches
#: the compiler, which refuses a zero-radius geom. Every test using it asserts
#: that it really was the compiler that refused, so an op-level domain added to
#: ``add_site`` later turns these into loud failures rather than quiet passes.
COMPILE_REFUSED_BATCH: list[dict[str, Any]] = [
    {"op": "add_site", "body": "world", "name": "unbuildable", "size": [0.0]},
]


def _own_scope_calls(func: ast.FunctionDef) -> list[ast.Call]:
    """Calls made in ``func``'s own scope, not inside a nested function."""
    nested = {
        id(n)
        for outer in ast.walk(func)
        if isinstance(outer, ast.FunctionDef) and outer is not func
        for n in ast.walk(outer)
    }
    return [n for n in ast.walk(func) if isinstance(n, ast.Call) and id(n) not in nested]


def _functions_recompiling_a_spec_directly() -> list[str]:
    """``scene_ops`` functions that call ``spec.recompile(...)`` themselves."""
    source = pathlib.Path(scene_ops.__file__).read_text(encoding="utf-8")
    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in _own_scope_calls(node):
            func = call.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "recompile"
                and isinstance(func.value, ast.Name)
                and func.value.id == "spec"
            ):
                out.append(node.name)
                break
    return out


def _patch_scene_mjcf_ast() -> ast.FunctionDef:
    source = pathlib.Path(scene_ops.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "patch_scene_mjcf":
            return node
    raise AssertionError("patch_scene_mjcf vanished from scene_ops")


def _functions_calling(symbol: str) -> list[str]:
    """Names of the ``scene_ops`` functions whose body mentions ``symbol``."""
    source = pathlib.Path(scene_ops.__file__).read_text(encoding="utf-8")
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and symbol in (ast.get_source_segment(source, node) or "")
    ]


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch_rejected_batch", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestRejectedBatchIsDebuggable:
    def test_the_message_names_which_op_failed(self, sim: Simulation) -> None:
        """A batch is only debuggable if the rejection says which op broke it."""
        sim.create_world()

        result = sim.patch_scene_mjcf(list(REJECTED_BATCH))

        assert result["status"] == "error"
        message = result["content"][0]["text"].lower()
        assert "patch op #2" in message, f"the failing op is not identified: {message}"


class TestRejectedBatchRestoresTheLiveSpec:
    """The spec -- not just the compiled model -- is what must come back clean.

    A rejected batch never recompiles, so ``_model`` is trivially unchanged. The
    spec is the mutable object the ops wrote to and the one the next mutation
    recompiles from, so it is the only place a half-applied body can survive.
    """

    def test_the_live_spec_carries_no_half_applied_body(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        spec_before = sorted(b.name for b in sim._world._backend_state["spec"].bodies)

        result = sim.patch_scene_mjcf(list(REJECTED_BATCH))
        assert result["status"] == "error"

        assert sim._world is not None
        spec_after = sorted(b.name for b in sim._world._backend_state["spec"].bodies)
        assert spec_after == spec_before, "op #1's body survived the rejection on the live spec"

    def test_a_later_mutation_recompiles_without_the_orphan(self, sim: Simulation) -> None:
        """The consequence the restore exists for: the next mutation stays clean."""
        sim.create_world()
        assert sim._world is not None
        mj = sim._mj
        nbody_before = sim._world._model.nbody

        assert sim.patch_scene_mjcf(list(REJECTED_BATCH))["status"] == "error"

        # This recompiles from the live spec, so a surviving op #1 would be
        # baked into the model here rather than at the rejected batch.
        added = sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])
        assert added["status"] == "success", added

        assert sim._world is not None
        model = sim._world._model
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "crate") >= 0, "the later mutation did not land"
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "doomed") == -1, (
            "the rejected batch's body was recompiled into the model by a later mutation"
        )
        assert model.nbody == nbody_before + 1


class TestRejectedBatchLeavesTheWorldSteppable:
    def test_the_world_still_steps(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        nbody_before = sim._world._model.nbody

        assert sim.patch_scene_mjcf(list(REJECTED_BATCH))["status"] == "error"

        assert sim._world is not None
        assert sim._world._model.nbody == nbody_before
        sim.step(1)


class TestTheRestoreIsAnObjectSwap:
    """Pins the mechanism, because this module used to test a different one.

    The snapshot moved from a ``spec.to_xml()`` round trip to ``MjSpec.copy``
    (see :func:`strands_robots.simulation.mujoco.scene_ops._snapshot_spec`,
    whose docstring records why the round trip was abandoned). The rollback is
    now a plain reassignment of the cached spec, so ``SpecBuilder.from_mjcf_string``
    is unreachable from the patch path -- a test that stubs it to force a
    "restore itself failed" branch intercepts nothing and passes on the ordinary
    rollback instead.
    """

    def test_the_patch_path_does_not_rebuild_the_spec_from_mjcf(self) -> None:
        callers = _functions_calling("from_mjcf_string")
        assert callers, "from_mjcf_string vanished from scene_ops; this guard needs re-reading"
        assert "patch_scene_mjcf" not in callers, (
            f"patch_scene_mjcf now rebuilds the spec from MJCF (callers: {callers}); the rollback "
            "is no longer a plain object swap and this module's contract needs re-reading"
        )


def _refuse_at_compile(sim: Simulation) -> str:
    """Run the compile-refused batch and return its message.

    Asserts the refusal came from the *compiler* rather than from an op, so a
    future op-level domain on ``add_site``'s ``size`` makes every caller of this
    helper fail loudly instead of silently testing the op-level path instead.
    """
    result = sim.patch_scene_mjcf([dict(op) for op in COMPILE_REFUSED_BATCH])
    assert result["status"] == "error", result
    message = result["content"][0]["text"]
    assert "patch op #" not in message, (
        f"premise: this batch is meant to clear every op-level check and be refused by the "
        f"compiler, but an op rejected it: {message}"
    )
    return message


class TestACompileRefusedBatchIsRolledBackToo:
    """A batch every op accepts and the compiler refuses costs only that batch.

    The recompile used to sit outside the rollback, so this rejection took
    neither branch: the snapshot was dropped and the live spec kept the whole
    batch. That spec is what every later mutation recompiles from, so one
    refused batch made the world permanently unusable -- an unrelated
    ``add_object`` afterwards failed too, naming the leftover element from a
    batch the caller had been told had failed.
    """

    def test_the_live_spec_carries_none_of_the_batch(self, sim: Simulation) -> None:
        sim.create_world()
        assert sim._world is not None
        sites_before = sorted(s.name for s in sim._world._backend_state["spec"].sites)

        _refuse_at_compile(sim)

        assert sim._world is not None
        sites_after = sorted(s.name for s in sim._world._backend_state["spec"].sites)
        assert sites_after == sites_before, "the compiler-refused batch survived on the live spec"

    def test_a_later_unrelated_mutation_still_succeeds(self, sim: Simulation) -> None:
        """The consequence: the refusal costs the batch, not the world."""
        sim.create_world()

        _refuse_at_compile(sim)

        added = sim.add_object(name="crate", shape="box", size=[0.1] * 3, position=[0.4, 0, 0.05])
        assert added["status"] == "success", f"an unrelated add_object inherited the refused batch: {added}"
        assert (
            sim.patch_scene_mjcf([{"op": "add_body", "parent": "world", "name": "later", "pos": [1, 0, 1]}])["status"]
            == "success"
        )

    def test_the_compiled_model_and_cached_xml_are_untouched(self, sim: Simulation) -> None:
        """Nothing but the spec needs restoring, and nothing else was written."""
        sim.create_world()
        assert sim._world is not None
        model_before = sim._world._model
        sizes_before = (model_before.nbody, model_before.nsite, model_before.nq)
        xml_before = sim._world._backend_state.get("xml")

        _refuse_at_compile(sim)

        assert sim._world is not None
        model_after = sim._world._model
        assert model_after is model_before, "the refused batch replaced the compiled model"
        assert (model_after.nbody, model_after.nsite, model_after.nq) == sizes_before
        assert sim._world._backend_state.get("xml") == xml_before, (
            "the cached XML was re-synced from the spec the compiler refused"
        )

    def test_the_world_still_steps(self, sim: Simulation) -> None:
        sim.create_world()

        _refuse_at_compile(sim)

        assert sim.step(1)["status"] == "success"

    def test_the_refusal_carries_the_compilers_reason(self, sim: Simulation) -> None:
        """A compile refusal names what the compiler could not build.

        It must NOT claim an op index: every op applied, so there is no single op
        at fault and a count would just be ``len(ops)``. This holds before the
        fix too - the message was never the defect, the leftover spec was.
        """
        sim.create_world()

        message = _refuse_at_compile(sim)

        assert "size" in message.lower(), f"the compiler's reason did not travel: {message}"


class TestTheBatchRecompilesThroughTheSharedHelper:
    """The root cause, pinned structurally.

    ``_recompile_preserving_state`` is where a scene rebuild's state carry, its
    refusal rung and its per-robot id re-discovery live. Every other mutation
    goes through it; this batch called ``spec.recompile`` itself, so it inherited
    none of that and its refusal landed past the rollback. Keying the guard on
    "only the shared helper recompiles a spec" is what stops the next path that
    needs a recompile from re-deriving one.
    """

    def test_only_the_shared_helper_recompiles_a_spec(self) -> None:
        callers = _functions_recompiling_a_spec_directly()
        assert callers, "no scene_ops function recompiles a spec any more; this guard needs re-reading"
        assert callers == ["_recompile_preserving_state"], (
            f"a scene_ops function recompiles the spec itself instead of through the shared "
            f"helper, so it keeps none of the state that helper carries: {callers}"
        )

    def test_the_batch_recompile_is_inside_the_rollbacks_scope(self) -> None:
        """The refusal must be caught by the handler that puts the spec back."""
        guarded = [
            node
            for node in ast.walk(_patch_scene_mjcf_ast())
            if isinstance(node, ast.Try)
            and any(
                isinstance(call.func, ast.Name) and call.func.id == "_recompile_preserving_state"
                for stmt in node.body
                for call in ast.walk(stmt)
                if isinstance(call, ast.Call)
            )
        ]
        assert len(guarded) == 1, (
            "patch_scene_mjcf's recompile is not inside a try, so a refused compile leaves the batch on the live spec"
        )
        restores = [
            target
            for handler in guarded[0].handlers
            for stmt in ast.walk(handler)
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Subscript) and "_backend_state" in ast.unparse(target)
        ]
        assert restores, "the recompile's handler does not put the pre-patch spec back"
