"""Contract tests for ``patch_scene_mjcf`` rollback when the restore itself fails.

``patch_scene_mjcf`` applies a batch of structured ops to the live ``MjSpec``
atomically: if any op raises, the mutated spec is discarded and the world is
rolled back to a pre-patch XML snapshot. The happy rollback (snapshot restores
cleanly) is covered elsewhere; this module pins the *degraded* branch where the
rollback's own ``SpecBuilder.from_mjcf_string(backup_xml)`` restore raises.

The contract in that branch is twofold:

1. The ORIGINAL op failure is surfaced to the caller - a restore failure must
   not mask why the patch was rejected in the first place.
2. The live compiled model/data are left intact (the batch never recompiled),
   so the world remains steppable even though the cached spec could not be
   swapped back to the clean snapshot.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")


from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_patch_rollback", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


class TestPatchRollbackRestoreFailure:
    def test_restore_failure_surfaces_original_op_error(self, sim: Simulation, monkeypatch) -> None:
        """A failing rollback-restore must not hide the original op error."""
        sim.create_world()
        assert sim._world is not None

        def _boom(_xml: str):
            raise RuntimeError("snapshot round-trip unavailable")

        # Only the rollback path calls from_mjcf_string during a patch; the
        # backup snapshot itself is taken via spec.to_xml(), not this function.
        monkeypatch.setattr(scene_ops.SpecBuilder, "from_mjcf_string", staticmethod(_boom))

        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "doomed", "pos": [0, 0, 1]},
                {"op": "totally_made_up", "name": "whatever"},
            ]
        )

        assert result["status"] == "error"
        message = result["content"][0]["text"].lower()
        # The caller sees which op failed and why - not the restore failure.
        assert "patch op #2" in message
        assert "snapshot round-trip unavailable" not in message

    def test_restore_failure_leaves_live_model_steppable(self, sim: Simulation, monkeypatch) -> None:
        """Even when restore fails, the pre-patch model is never recompiled away."""
        sim.create_world()
        assert sim._world is not None
        mj = sim._mj
        nbody_before = sim._world._model.nbody

        monkeypatch.setattr(
            scene_ops.SpecBuilder,
            "from_mjcf_string",
            staticmethod(lambda _xml: (_ for _ in ()).throw(RuntimeError("no restore"))),
        )

        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "doomed", "pos": [0, 0, 1]},
                {"op": "totally_made_up", "name": "whatever"},
            ]
        )
        assert result["status"] == "error"

        # The half-applied batch never recompiled, so the live model is unchanged
        # and the doomed body from op #1 is absent.
        assert sim._world is not None
        assert sim._world._model.nbody == nbody_before
        assert mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "doomed") == -1

        # And the world still steps without raising.
        sim.step(1)
