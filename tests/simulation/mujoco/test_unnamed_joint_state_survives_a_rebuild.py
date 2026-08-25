"""An unnamed joint keeps its state when a scene rebuild renumbers the model.

``eject_robot_from_scene`` rebuilds the compiled model, which renumbers every
joint id, so the dynamic state is snapshotted under a name-based key and written
back afterwards. :func:`~strands_robots.simulation.mujoco.scene_ops._joint_key`
builds that key, and it used to return ``None`` for any unnamed non-free joint,
on the grounds that "its body may carry several, so the body does not single it
out". A key of ``None`` is skipped, so the joint's ``qpos``, ``qvel`` and
``qfrc_applied`` were dropped and it came back at its fresh-compile value while
the operation reported success.

An unnamed ``<joint>`` is the ordinary MJCF spelling -- a door hinge or a drawer
slide is rarely named -- so removing one robot silently reset the rest of the
scene's articulation. The premise was sound but the conclusion was too strong:
the body does not single the joint out, yet the body *plus its position among
that body's joints* does. MuJoCo stores a body's joints contiguously from
``body_jntadr`` in declaration order, so the ordinal is a stable handle across a
rebuild, and no scene op inserts a joint into an existing body (the patch
vocabulary is ``add_body`` / ``add_geom`` / ``add_site`` / ``set_body_pos`` /
``set_body_quat`` / ``delete_body``), so nothing shifts it.

Two joints of one body are the case that makes the ordinal load-bearing rather
than decorative: a hinge and a slide both use width 1/1, so the downstream width
check in ``_restore_scene_state`` cannot tell them apart, and keying both on the
body alone would swap their values.

Still unmatched, deliberately: a joint whose body is *also* unnamed. Nothing
identifies either end across the rebuild, so it is reported rather than guessed
at, and that is pinned here too.

GL-free: ``mesh=False`` and no rendering, so this runs without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# An arm whose only joint is named, so it is carried by the pre-existing
# ``("joint", name)`` key. It is the robot the tests eject.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.05">
      <geom type="box" size="0.04 0.04 0.05"/>
      <body name="link" pos="0 0 0.1">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# The furniture that must survive the eject untouched: a door on a single
# unnamed hinge, and a drawer carrying two unnamed joints on one body.
_FURNITURE_XML = """
<mujoco model="furniture">
  <compiler angle="radian"/>
  <worldbody>
    <body name="door_frame" pos="0.5 0 0.2">
      <geom type="box" size="0.02 0.02 0.2"/>
      <body name="door_panel" pos="0 0 0">
        <joint type="hinge" axis="0 0 1" range="-3 3" damping="0.5"/>
        <geom type="box" size="0.01 0.15 0.18" pos="0 0.15 0"/>
      </body>
    </body>
    <body name="drawer" pos="-0.5 0 0.2">
      <joint type="hinge" axis="0 0 1" range="-3 3" damping="0.5"/>
      <joint type="slide" axis="1 0 0" range="-1 1" damping="0.5"/>
      <geom type="box" size="0.06 0.06 0.06"/>
    </body>
  </worldbody>
</mujoco>
"""

#: Distinctive values, one per unnamed degree of freedom. They differ from each
#: other so a swap between two joints of one body is visible, and none is zero
#: so a value lost to a fresh compile is visible too.
_DOOR_STATE = (0.70, -1.25)
_DRAWER_HINGE_STATE = (0.31, 0.44)
_DRAWER_SLIDE_STATE = (-0.17, 2.05)


def _model_from(xml: str) -> Any:
    """Compile ``xml`` to a model, for the key helpers that need no world."""
    return mujoco.MjModel.from_xml_string(xml)


def _joint_id(model: Any, name: str) -> int:
    """The id of the joint called ``name``."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _unnamed_joint_ids(model: Any) -> list[int]:
    """Every joint id the model carries no name for, in id order."""
    return [jid for jid in range(int(model.njnt)) if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)]


def _body_of(model: Any, jid: int) -> str:
    """The name of the body owning joint ``jid``."""
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])))


def _read(world: Any, jid: int) -> tuple[float, float]:
    """The ``(qpos, qvel)`` of the single-dof joint ``jid``."""
    qpos_adr = int(world._model.jnt_qposadr[jid])
    dof_adr = int(world._model.jnt_dofadr[jid])
    return float(world._data.qpos[qpos_adr]), float(world._data.qvel[dof_adr])


def _write(world: Any, jid: int, state: tuple[float, float]) -> None:
    """Seed the ``(qpos, qvel)`` of the single-dof joint ``jid``."""
    qpos_adr = int(world._model.jnt_qposadr[jid])
    dof_adr = int(world._model.jnt_dofadr[jid])
    world._data.qpos[qpos_adr], world._data.qvel[dof_adr] = state


@pytest.fixture
def scene(tmp_path: Path) -> Any:
    """A world holding the arm plus the furniture, with the furniture seeded.

    Yields the ``Simulation`` and a ``{body name: {ordinal: state}}`` map of what
    every unnamed joint was seeded with, read back through the compiled model so
    the expectations are the model's own view rather than a hand-kept list.
    """
    (tmp_path / "arm.xml").write_text(_ARM_XML)
    (tmp_path / "furniture.xml").write_text(_FURNITURE_XML)

    sim = Simulation(tool_name="test_unnamed_joint_state_survives_a_rebuild", mesh=False)
    sim.create_world(gravity=[0, 0, -9.81])
    assert sim.add_robot(name="arm", urdf_path=str(tmp_path / "arm.xml"))["status"] == "success"
    assert sim.add_robot(name="fur", urdf_path=str(tmp_path / "furniture.xml"))["status"] == "success"

    world = sim._world
    assert world is not None
    model = world._model
    unnamed = _unnamed_joint_ids(model)
    # The premise: the furniture really did compile to three unnamed joints, so a
    # rename upstream cannot quietly turn these tests into no-ops.
    assert len(unnamed) == 3, [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]

    seeded: dict[str, dict[int, tuple[float, float]]] = {}
    for jid in unnamed:
        body = _body_of(model, jid)
        ordinal = jid - int(model.body_jntadr[int(model.jnt_bodyid[jid])])
        state = {
            (0, True): _DOOR_STATE,
            (0, False): _DRAWER_HINGE_STATE,
            (1, False): _DRAWER_SLIDE_STATE,
        }[(ordinal, body.endswith("door_panel"))]
        _write(world, jid, state)
        seeded.setdefault(body, {})[ordinal] = state
    mujoco.mj_forward(model, world._data)

    yield sim, seeded
    sim.cleanup()


def _survivors(sim: Any) -> dict[str, dict[int, tuple[float, float]]]:
    """The current ``(qpos, qvel)`` of every unnamed joint, keyed as seeded."""
    world = sim._world
    model = world._model
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for jid in _unnamed_joint_ids(model):
        ordinal = jid - int(model.body_jntadr[int(model.jnt_bodyid[jid])])
        out.setdefault(_body_of(model, jid), {})[ordinal] = _read(world, jid)
    return out


class TestAnUnnamedJointSurvivesTheEject:
    """The regression: removing one robot must not reset the rest of the scene."""

    def test_every_unnamed_joint_keeps_the_state_it_was_seeded_with(self, scene: Any) -> None:
        sim, seeded = scene
        assert sim.remove_robot("arm")["status"] == "success"
        assert _survivors(sim) == seeded

    def test_the_single_unnamed_hinge_keeps_its_position_and_velocity(self, scene: Any) -> None:
        """The headline shape: a door on an unnamed hinge, spelled as MJCF spells it."""
        sim, _ = scene
        sim.remove_robot("arm")
        door = next(body for body in _survivors(sim) if body.endswith("door_panel"))
        assert _survivors(sim)[door][0] == _DOOR_STATE

    def test_the_restore_counts_the_unnamed_joints(self, scene: Any) -> None:
        """A skipped key is not merely invisible - it is absent from the count."""
        sim, _ = scene
        world = sim._world
        snapshot = scene_ops._snapshot_scene_state(world)
        assert scene_ops._restore_scene_state(world, snapshot) == int(world._model.njnt)


class TestTheOrdinalTellsTwoJointsOfOneBodyApart:
    """Two unnamed joints on one body: the case a body-only key would swap."""

    def test_the_two_joints_of_one_body_keep_their_own_values(self, scene: Any) -> None:
        sim, _ = scene
        sim.remove_robot("arm")
        drawer = next(body for body in _survivors(sim) if body.endswith("drawer"))
        by_ordinal = _survivors(sim)[drawer]
        assert by_ordinal[0] == _DRAWER_HINGE_STATE
        assert by_ordinal[1] == _DRAWER_SLIDE_STATE

    def test_the_two_joints_are_indistinguishable_by_width(self, scene: Any) -> None:
        """The premise for the test above: the downstream width check cannot help.

        ``_restore_scene_state`` skips an entry whose width no longer matches the
        joint type. A hinge and a slide are both 1/1, so a swap between them
        passes that check - only the key can keep them apart.
        """
        sim, _ = scene
        model = sim._world._model
        drawer_joints = [j for j in _unnamed_joint_ids(model) if _body_of(model, j).endswith("drawer")]
        assert len(drawer_joints) == 2
        widths = {scene_ops._joint_state_widths(int(model.jnt_type[j]), mujoco) for j in drawer_joints}
        assert widths == {(1, 1)}, widths

    def test_the_two_joints_get_distinct_keys(self, scene: Any) -> None:
        sim, _ = scene
        model = sim._world._model
        drawer_joints = [j for j in _unnamed_joint_ids(model) if _body_of(model, j).endswith("drawer")]
        keys = [scene_ops._joint_key(model, j, mujoco) for j in drawer_joints]
        assert len(set(keys)) == len(keys), keys


class TestWhichHandleEachJointGets:
    """The key forms, on a model compiled straight from MJCF."""

    def test_an_unnamed_joint_is_keyed_by_its_body_and_position(self) -> None:
        model = _model_from(_FURNITURE_XML)
        drawer = [j for j in _unnamed_joint_ids(model) if _body_of(model, j) == "drawer"]
        assert [scene_ops._joint_key(model, j, mujoco) for j in drawer] == [
            ("body_joint", "drawer", 0),
            ("body_joint", "drawer", 1),
        ]

    def test_a_named_joint_is_still_keyed_by_its_name(self) -> None:
        model = _model_from(_ARM_XML)
        assert scene_ops._joint_key(model, _joint_id(model, "pan"), mujoco) == ("joint", "pan")

    def test_an_unnamed_free_joint_is_still_keyed_by_its_body_alone(self) -> None:
        """The free-joint form is unchanged: one free joint per body names it."""
        model = _model_from(
            """
            <mujoco><worldbody>
              <body name="crate" pos="0 0 1"><freejoint/><geom type="box" size="0.05 0.05 0.05"/></body>
            </worldbody></mujoco>
            """
        )
        assert scene_ops._joint_key(model, 0, mujoco) == ("body", "crate")

    def test_a_joint_whose_body_is_also_unnamed_stays_unmatched(self) -> None:
        """Nothing identifies either end, so it is reported rather than guessed at."""
        model = _model_from(
            """
            <mujoco><worldbody>
              <body pos="0 0 1"><joint type="hinge" axis="0 0 1"/><geom type="box" size="0.05 0.05 0.05"/></body>
            </worldbody></mujoco>
            """
        )
        assert scene_ops._joint_key(model, 0, mujoco) is None


class TestTheOrdinalHandleResolvesOnlyWhatItNames:
    """``_resolve_joint_key`` must not claim a joint the handle does not mean."""

    def test_it_resolves_the_joint_at_that_position(self) -> None:
        model = _model_from(_FURNITURE_XML)
        expected = [j for j in _unnamed_joint_ids(model) if _body_of(model, j) == "drawer"]
        assert [
            scene_ops._resolve_joint_key(model, ("body_joint", "drawer", ordinal), mujoco) for ordinal in (0, 1)
        ] == expected

    def test_an_ordinal_past_the_body_s_joints_resolves_to_nothing(self) -> None:
        model = _model_from(_FURNITURE_XML)
        assert scene_ops._resolve_joint_key(model, ("body_joint", "drawer", 2), mujoco) == -1

    def test_a_vanished_body_resolves_to_nothing(self) -> None:
        model = _model_from(_FURNITURE_XML)
        assert scene_ops._resolve_joint_key(model, ("body_joint", "gone", 0), mujoco) == -1

    def test_a_joint_that_gained_a_name_is_left_to_its_own_key(self) -> None:
        """A named joint is carried by ``("joint", name)``; the ordinal must not
        also claim it, or one joint would be restored from two entries."""
        model = _model_from(
            """
            <mujoco><worldbody>
              <body name="drawer" pos="0 0 1">
                <joint name="now_named" type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.05 0.05 0.05"/>
              </body>
            </worldbody></mujoco>
            """
        )
        assert scene_ops._resolve_joint_key(model, ("body_joint", "drawer", 0), mujoco) == -1


class TestTheEjectStillDoesWhatItDidBefore:
    """Over-reach controls: carrying more state must not carry the wrong state."""

    def test_the_named_joint_of_a_surviving_robot_still_survives(self, tmp_path: Path) -> None:
        (tmp_path / "a.xml").write_text(_ARM_XML)
        sim = Simulation(tool_name="test_unnamed_joint_named_control", mesh=False)
        try:
            sim.create_world(gravity=[0, 0, -9.81])
            sim.add_robot(name="a", urdf_path=str(tmp_path / "a.xml"))
            sim.add_robot(name="b", urdf_path=str(tmp_path / "a.xml"))
            world = sim._world
            assert world is not None
            keep = _joint_id(world._model, "b/pan")
            _write(world, keep, (0.55, -0.9))
            mujoco.mj_forward(world._model, world._data)
            assert sim.remove_robot("a")["status"] == "success"
            rebuilt = sim._world
            assert rebuilt is not None
            assert _read(rebuilt, _joint_id(rebuilt._model, "b/pan")) == (0.55, -0.9)
        finally:
            sim.cleanup()

    def test_the_ejected_robot_leaves_no_joint_behind(self, scene: Any) -> None:
        sim, _ = scene
        sim.remove_robot("arm")
        model = sim._world._model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(int(model.njnt))]
        assert "arm/pan" not in names, names

    def test_the_furniture_is_reported_present_after_the_eject(self, scene: Any) -> None:
        sim, _ = scene
        sim.remove_robot("arm")
        assert "fur" in sim.list_robots()
