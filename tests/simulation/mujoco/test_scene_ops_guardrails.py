"""Guardrail behavior of the MuJoCo scene-mutation layer (``scene_ops``).

The happy paths of ``patch_scene_mjcf`` / ``replace_scene_mjcf`` are exercised
elsewhere (``test_patch_scene_mjcf``, ``test_replace_scene_mjcf``). This module
pins the *defensive* contract of
:mod:`strands_robots.simulation.mujoco.scene_ops`: the structured-op validators
that reject malformed agent input with an actionable message, the optional-field
branches of each op, and the "no compiled world yet" early returns that every
inject/eject helper makes before touching the spec. These are the boundaries an
autonomous agent hits first when it drives the scene API blind, so they must
fail loudly and predictably rather than crash mid-mutation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.models import (  # noqa: E402
    SimCamera,
    SimObject,
    SimRobot,
    SimWorld,
)
from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_scene_guard", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


# A minimal single-joint arm so attach-based namespacing can be exercised.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


class TestInjectEjectRequireCompiledWorld:
    """Every mutation helper returns a clean ``False`` (never crashes) when the
    world has no spec/model yet - the agent called a scene edit before
    ``create_world``."""

    def test_inject_robot_without_spec_returns_false(self) -> None:
        world = SimWorld()
        ok = scene_ops.inject_robot_into_scene(world, SimRobot(name="r", urdf_path="x.xml"), "x.xml")
        assert ok is False

    def test_inject_object_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.inject_object_into_scene(world, SimObject(name="o", shape="box")) is False

    def test_inject_camera_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.inject_camera_into_scene(world, SimCamera(name="c")) is False

    def test_eject_body_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.eject_body_from_scene(world, "foo") is False

    def test_eject_robot_without_spec_returns_false(self) -> None:
        world = SimWorld()
        assert scene_ops.eject_robot_from_scene(world, "r") is False


class TestInjectFailuresReturnFalse:
    """A spec mutation that raises (bad shape, unreadable URDF) is caught and
    surfaced as ``False`` so the caller can emit an error dict, leaving the
    already-compiled world intact."""

    def test_inject_object_with_unsupported_shape_returns_false(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        nbody_before = world._model.nbody
        assert scene_ops.inject_object_into_scene(world, SimObject(name="bad", shape="not_a_shape")) is False
        # The failed add must not have grown the compiled model.
        assert world._model.nbody == nbody_before

    def test_inject_robot_with_unreadable_urdf_returns_false(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        ok = scene_ops.inject_robot_into_scene(
            world, SimRobot(name="rr", urdf_path="/no/such/file.xml"), "/no/such/file.xml"
        )
        assert ok is False


class TestEjectMissingBodyIsConsistent:
    def test_eject_unknown_body_returns_true_without_changing_model(self, sim: Simulation) -> None:
        """Ejecting a body that isn't in the spec is a no-op that still reports
        success: the caller has already dropped the Python-side entry, so the
        scene stays consistent."""
        sim.create_world()
        world = sim._world
        assert world is not None
        nbody_before = world._model.nbody
        assert scene_ops.eject_body_from_scene(world, "does_not_exist") is True
        assert world._model.nbody == nbody_before


class TestRepositionBodyGuards:
    """``reposition_body_in_scene`` is the static-fixture sibling of the
    inject/eject helpers and makes the same defensive promises: a call before
    ``create_world`` and a call naming a body absent from the spec both return
    ``False`` (logged) without mutating the compiled model, so ``move_object``'s
    static branch surfaces a clean error instead of a silent no-op. The happy
    path (a welded fixture actually relocates) is pinned alongside so the guard
    tests cannot pass by simply disabling the feature."""

    def test_reposition_without_spec_returns_false(self) -> None:
        assert scene_ops.reposition_body_in_scene(SimWorld(), "x", position=[0, 0, 1]) is False

    def test_reposition_unknown_body_returns_false_without_changing_model(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        nbody_before = world._model.nbody
        assert scene_ops.reposition_body_in_scene(world, "does_not_exist", position=[0, 0, 1]) is False
        # A body that is not in the spec is a no-op: the compiled model is untouched.
        assert world._model.nbody == nbody_before

    def test_reposition_treats_raising_spec_lookup_as_missing(self, sim: Simulation, monkeypatch) -> None:
        """Some MuJoCo builds raise ``KeyError``/``ValueError`` from
        ``spec.body(name)`` for an unknown body instead of returning ``None``.
        The helper catches that and treats it as a missing body (returns
        ``False``) rather than letting the exception escape the scene edit."""
        sim.create_world()
        world = sim._world
        assert world is not None

        class _RaisingSpec:
            def body(self, name: str):  # noqa: ANN202 - stub mirrors mjSpec.body
                raise ValueError(f"no such body: {name}")

        monkeypatch.setattr(scene_ops, "_get_spec", lambda _world: _RaisingSpec())
        assert scene_ops.reposition_body_in_scene(world, "whatever", position=[0, 0, 1]) is False

    def test_reposition_relocates_static_fixture(self, sim: Simulation) -> None:
        """A welded static body (no freejoint) is moved by editing its spec pose
        and recompiling - the qpos path cannot touch a DOF-less body, so this is
        the only route that actually relocates a static fixture."""
        sim.create_world()
        world = sim._world
        assert world is not None
        assert (
            sim.add_object(
                name="fixture", shape="box", size=[0.05, 0.05, 0.05], position=[0.2, 0.0, 0.1], is_static=True
            )["status"]
            == "success"
        )
        assert scene_ops.reposition_body_in_scene(world, "fixture", position=[0.4, 0.1, 0.3]) is True
        mj = sim._mj
        bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, "fixture")
        assert bid >= 0
        assert pytest.approx(list(world._model.body_pos[bid]), abs=1e-6) == [0.4, 0.1, 0.3]


class TestSnapshotRestoreWithoutModel:
    def test_snapshot_empty_world_returns_empty_dict(self) -> None:
        assert scene_ops._snapshot_joint_state(SimWorld()) == {}

    def test_restore_empty_world_restores_nothing(self) -> None:
        assert scene_ops._restore_joint_state(SimWorld(), {}) == 0


class TestPatchOpValidation:
    """``_apply_patch_op`` rejects malformed ops through the public
    ``patch_scene_mjcf`` entry point with an actionable, op-specific message and
    rolls the whole batch back (atomic)."""

    @pytest.fixture
    def world_sim(self, sim: Simulation) -> Simulation:
        sim.create_world()
        return sim

    def test_non_dict_op_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([42])  # type: ignore[list-item]
        assert result["status"] == "error"
        assert "must be a dict" in result["content"][0]["text"]

    def test_add_body_unknown_parent_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_body", "parent": "ghost", "name": "x"}])
        assert result["status"] == "error"
        assert "parent 'ghost' not found" in result["content"][0]["text"]

    def test_add_geom_requires_body(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_geom", "type": "box"}])
        assert result["status"] == "error"
        assert "add_geom requires 'body'" in result["content"][0]["text"]

    def test_add_geom_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_geom", "body": "ghost", "type": "box"}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_add_site_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_site", "body": "world"}])
        assert result["status"] == "error"
        assert "add_site requires 'name'" in result["content"][0]["text"]

    def test_add_site_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "add_site", "body": "ghost", "name": "s"}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_set_body_pos_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_pos"}])
        assert result["status"] == "error"
        assert "set_body_pos requires 'name'" in result["content"][0]["text"]

    def test_set_body_pos_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "ghost", "pos": [0, 0, 1]}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_set_body_quat_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_quat"}])
        assert result["status"] == "error"
        assert "set_body_quat requires 'name'" in result["content"][0]["text"]

    def test_set_body_quat_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "ghost", "quat": [1, 0, 0, 0]}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]

    def test_delete_body_requires_name(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "delete_body"}])
        assert result["status"] == "error"
        assert "delete_body requires 'name'" in result["content"][0]["text"]

    def test_delete_body_unknown_body_rejected(self, world_sim: Simulation) -> None:
        result = world_sim.patch_scene_mjcf([{"op": "delete_body", "name": "ghost"}])
        assert result["status"] == "error"
        assert "body 'ghost' not found" in result["content"][0]["text"]


class TestPatchOpOptionalFields:
    """The optional-attribute branches of the add ops (geom name/pos/quat,
    site size/rgba, body quat) are honored and compile into the model."""

    def test_add_geom_with_name_pos_quat(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "host", "pos": [0, 0, 0.5]},
                {
                    "op": "add_geom",
                    "body": "host",
                    "name": "shell",
                    "type": "sphere",
                    "size": [0.05],
                    "pos": [0, 0, 0.1],
                    "quat": [1, 0, 0, 0],
                },
            ]
        )
        assert result["status"] == "success", result
        mj = sim._mj
        assert mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_GEOM, "shell") >= 0

    def test_add_site_with_size_and_rgba(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        result = sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "anchor", "pos": [0, 0, 0.3]},
                {
                    "op": "add_site",
                    "body": "anchor",
                    "name": "tip",
                    "pos": [0, 0, 0.1],
                    "size": [0.02],
                    "rgba": [1, 0, 0, 1],
                },
            ]
        )
        assert result["status"] == "success", result
        mj = sim._mj
        assert mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_SITE, "tip") >= 0

    def test_set_body_quat_updates_orientation(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        sim.patch_scene_mjcf(
            [
                {"op": "add_body", "name": "spinner", "pos": [0, 0, 0.5]},
                {"op": "add_geom", "body": "spinner", "type": "box", "size": [0.05, 0.05, 0.05]},
            ]
        )
        result = sim.patch_scene_mjcf([{"op": "set_body_quat", "name": "spinner", "quat": [0, 1, 0, 0]}])
        assert result["status"] == "success", result
        mj = sim._mj
        bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, "spinner")
        assert bid >= 0
        assert pytest.approx(list(world._model.body_quat[bid]), abs=1e-6) == [0.0, 1.0, 0.0, 0.0]


class TestFindBodyScansAttachedRobotBodies:
    def test_namespaced_attached_body_is_patchable(self, sim: Simulation, tmp_path) -> None:
        """A body introduced via ``spec.attach`` (namespaced under the robot
        name) is not visible through ``spec.body(name)`` until the next compile,
        so ``_find_body`` falls back to scanning ``spec.bodies``. Referencing the
        namespaced body in a patch op must resolve through that fallback."""
        arm_path = tmp_path / "arm.xml"
        arm_path.write_text(_ARM_XML)
        sim.create_world()
        sim.add_robot(name="arm1", urdf_path=str(arm_path))
        result = sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "arm1/link0", "pos": [0, 0, 0.2]}])
        assert result["status"] == "success", result
        world = sim._world
        assert world is not None
        mj = sim._mj
        bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, "arm1/link0")
        assert bid >= 0
        assert pytest.approx(list(world._model.body_pos[bid]), abs=1e-6) == [0.0, 0.0, 0.2]


# A scene with one joint of every variable-width type so the per-name snapshot/
# restore logic exercises its free (7/6), ball (4/3) and hinge (1/1) branches.
_MIXED_JOINT_XML = """
<mujoco model="mixed_joints">
  <compiler angle="radian"/>
  <worldbody>
    <body name="floater" pos="0 0 1">
      <freejoint name="free_j"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="baller" pos="0.3 0 1">
      <joint name="ball_j" type="ball"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="hinger" pos="0.6 0 0.2">
      <joint name="hinge_j" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


class TestSnapshotRestoreJointWidths:
    """``_snapshot_joint_state`` / ``_restore_joint_state`` slice each joint at
    the width MuJoCo actually uses (free 7/6, ball 4/3, hinge/slide 1/1).

    ``eject_robot_from_scene`` relies on this to carry surviving robots and
    object freejoints across a scene rebuild by *name* - a width bug would
    silently mis-assign DOFs (the exact failure mode the per-name approach
    exists to prevent), so the per-type widths are pinned here end to end.
    """

    @pytest.fixture
    def mixed_world(self, sim: Simulation) -> SimWorld:
        sim.create_world()
        result = sim.replace_scene_mjcf(_MIXED_JOINT_XML)
        assert result["status"] == "success", result
        world = sim._world
        assert world is not None
        return world

    def test_snapshot_uses_per_joint_type_widths(self, mixed_world: SimWorld) -> None:
        """Each joint is captured at its type-correct qpos/qvel width."""
        snap = scene_ops._snapshot_joint_state(mixed_world)
        assert (len(snap["free_j"][0]), len(snap["free_j"][1])) == (7, 6)
        assert (len(snap["ball_j"][0]), len(snap["ball_j"][1])) == (4, 3)
        assert (len(snap["hinge_j"][0]), len(snap["hinge_j"][1])) == (1, 1)

    def test_snapshot_restore_round_trips_all_joint_types(self, mixed_world: SimWorld) -> None:
        """A snapshot restored back into the same model touches every joint and
        leaves the recorded values byte-for-byte intact across all widths."""
        # Seed distinctive state so a width mis-slice would corrupt values.
        data = mixed_world._data
        data.qpos[:] = [float(i) * 0.01 for i in range(len(data.qpos))]
        data.qvel[:] = [float(i) * 0.02 for i in range(len(data.qvel))]
        snap = scene_ops._snapshot_joint_state(mixed_world)

        # Clobber state, then restore from the snapshot by name.
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        restored = scene_ops._restore_joint_state(mixed_world, snap)

        assert restored == 3, "free + ball + hinge all restored"
        re_snap = scene_ops._snapshot_joint_state(mixed_world)
        for name in ("free_j", "ball_j", "hinge_j"):
            assert re_snap[name] == snap[name]

    def test_restore_skips_width_mismatched_joint(self, mixed_world: SimWorld) -> None:
        """A snapshot entry whose width no longer matches the joint type (e.g.
        a same-named joint changed type across a rebuild) is skipped, not
        force-written - corrupt DOFs are never silently injected."""
        snap = scene_ops._snapshot_joint_state(mixed_world)
        # Forge a free-joint-width payload under the hinge joint's name.
        snap["hinge_j"] = ([0.0] * 7, [0.0] * 6)
        restored = scene_ops._restore_joint_state(mixed_world, snap)
        # free + ball restored; the mismatched hinge entry is dropped.
        assert restored == 2


class TestInjectCameraFailureReturnsFalse:
    """A camera whose mount point cannot be resolved makes
    ``inject_camera_into_scene`` return ``False`` (caught ``ValueError``)
    while leaving the already-compiled model untouched."""

    def test_inject_camera_unknown_parent_body_returns_false(self, sim: Simulation) -> None:
        sim.create_world()
        world = sim._world
        assert world is not None
        nbody_before = world._model.nbody
        cam = SimCamera(name="wrist", parent_body="no_such_body")
        assert scene_ops.inject_camera_into_scene(world, cam) is False
        # The failed add must not have grown or mutated the compiled model.
        assert world._model.nbody == nbody_before


class TestPatchSceneRequiresCompiledWorld:
    """``patch_scene_mjcf`` raises before touching any op when the world was
    never compiled - the agent edited the scene before ``create_world``."""

    def test_patch_without_spec_raises_runtime_error(self) -> None:
        world = SimWorld()
        with pytest.raises(RuntimeError, match="no spec"):
            scene_ops.patch_scene_mjcf(world, [{"op": "add_body", "name": "x"}])
