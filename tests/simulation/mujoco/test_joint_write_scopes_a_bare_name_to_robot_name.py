"""Regression tests: a bare joint name is scoped to ``robot_name``.

``set_joint_positions`` / ``set_joint_velocities`` resolve each dict key through
:meth:`~strands_robots.simulation.mujoco.physics.PhysicsMixin._resolve_mj_name`,
which tries a name verbatim and then under every robot namespace in turn,
returning the first hit. That is deliberate for the read paths that share it, but
for a *write* it meant a bare key ignored ``robot_name`` entirely: two robots
each declaring ``j1`` / ``j2`` compile to ``alice/j1`` ... ``bob/j2``, and::

    sim.set_joint_positions({"j1": 0.9, "j2": -0.9}, robot_name="bob")
    # -> success: "Set 2/2 joint positions, FK updated"

moved *alice* (the first robot attached), left ``bob`` at rest, and reported the
requested count for the robot that never moved. The method's own docstring
recommended the dict form as "safest in multi-robot scenes" and named
``robot_name`` as whose namespace "resolves an unqualified joint name", so the
documented safe form was the one that crossed robots (#2453).

The ordered (list) form shares the defect rather than escaping it, contrary to
what #2453 assumed: it normalises to ``dict(zip(robot.joint_names, values))``
and ``SimRobot.joint_names`` holds *short* names, so a 3-vector addressed to
``bob`` wrote ``alice/j1``, ``alice/j2`` and ``bob/shoulder`` - bound to bob's
joint order, written to alice's addresses. Scoping in the resolver repairs both
forms at once.

These pin the repair and its blast radius: an unqualified key resolves inside
``robot_name``'s namespace first, and every spelling that resolved before still
resolves - a qualified key, a bare key with no ``robot_name``, and a bare key
that names no joint of ``robot_name`` but does name one elsewhere in the scene.
"""

import os
import tempfile
from typing import Any

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Two robots that both declare ``j1`` / ``j2``. ``alice`` carries a freejoint so
# the two robots do not share a qpos layout, which is what makes a cross-robot
# write land at a visibly wrong address rather than a coincidentally equal one.
ALICE_XML = """
<mujoco model="alice">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="g0" type="box" size="0.05 0.05 0.05"/>
      <body name="l1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="g1" type="capsule" size="0.02 0.05"/>
        <body name="l2" pos="0 0 0.1">
          <joint name="j2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="g2" type="capsule" size="0.02 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# ``shoulder`` is declared by ``bob`` only, so it exercises the fallback: a bare
# name that is not a joint of the addressed robot must still resolve.
BOB_XML = """
<mujoco model="bob">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0.5 0 0.1">
      <body name="l1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="g1" type="capsule" size="0.02 0.05"/>
        <body name="l2" pos="0 0 0.1">
          <joint name="j2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="g2" type="capsule" size="0.02 0.05"/>
          <body name="l3" pos="0 0 0.1">
            <joint name="shoulder" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
            <geom name="g3" type="capsule" size="0.02 0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _write(xml: str, name: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), f"{name}.xml")
    with open(path, "w") as handle:
        handle.write(xml)
    return path


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_joint_write_robot_scoping", mesh=False)
    s.create_world()
    # ``alice`` is attached first, so it is the robot the unscoped first-match
    # lookup used to hand a bare ``j1`` to.
    assert s.add_robot(name="alice", urdf_path=_write(ALICE_XML, "alice"))["status"] == "success"
    assert s.add_robot(name="bob", urdf_path=_write(BOB_XML, "bob"))["status"] == "success"
    yield s
    s.cleanup()


def _qpos(sim, joint: str) -> float:
    model, data = sim._world._model, sim._world._data
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jid >= 0, f"fixture must compile a joint named {joint!r}"
    return float(data.qpos[model.jnt_qposadr[jid]])


def _qvel(sim, joint: str) -> float:
    model, data = sim._world._model, sim._world._data
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jid >= 0, f"fixture must compile a joint named {joint!r}"
    return float(data.qvel[model.jnt_dofadr[jid]])


def test_fixture_shares_short_joint_names(sim):
    """Guard the premise: without a shared short name nothing below is a test."""
    names = {mj.mj_id2name(sim._world._model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(sim._world._model.njnt)}
    assert {"alice/j1", "alice/j2", "bob/j1", "bob/j2"} <= names


def test_bare_name_writes_the_addressed_robot(sim):
    """The reported robot is the one that moves."""
    result = sim.set_joint_positions({"j1": 0.9, "j2": -0.9}, robot_name="bob")
    assert result["status"] == "success", result
    assert _qpos(sim, "bob/j1") == pytest.approx(0.9)
    assert _qpos(sim, "bob/j2") == pytest.approx(-0.9)


def test_bare_name_leaves_the_unaddressed_robot_alone(sim):
    """The bug wrote alice; a robot nobody addressed must not move."""
    result = sim.set_joint_positions({"j1": 0.9, "j2": -0.9}, robot_name="bob")
    assert result["status"] == "success", result
    assert _qpos(sim, "alice/j1") == pytest.approx(0.0)
    assert _qpos(sim, "alice/j2") == pytest.approx(0.0)


def test_reported_count_is_honest_for_the_addressed_robot(sim):
    """A reported 2/2 has to mean two of bob's joints, not two writes anywhere."""
    result = sim.set_joint_positions({"j1": 0.9, "j2": -0.9}, robot_name="bob")
    assert "2/2" in result["content"][0]["text"]
    moved = [j for j in ("bob/j1", "bob/j2") if _qpos(sim, j) != 0.0]
    assert len(moved) == 2, f"count claimed 2 but only {moved} moved"


def test_the_other_robot_is_reachable_by_name_too(sim):
    """Scoping must not pin every bare name to one robot."""
    assert sim.set_joint_positions({"j1": 0.4}, robot_name="alice")["status"] == "success"
    assert _qpos(sim, "alice/j1") == pytest.approx(0.4)
    assert _qpos(sim, "bob/j1") == pytest.approx(0.0)


def test_qualified_key_still_wins(sim):
    """An explicit ``<robot>/<joint>`` spelling is not re-scoped."""
    result = sim.set_joint_positions({"alice/j1": 0.5}, robot_name="bob")
    assert result["status"] == "success", result
    assert _qpos(sim, "alice/j1") == pytest.approx(0.5)
    assert _qpos(sim, "bob/j1") == pytest.approx(0.0)


def test_bare_name_absent_from_the_addressed_robot_still_resolves(sim):
    """The cross-robot fallback is preserved, so no working call is refused."""
    # ``shoulder`` is bob's; addressing alice must still reach it rather than
    # becoming an error the moment scoping is tried first.
    result = sim.set_joint_positions({"shoulder": 0.3}, robot_name="alice")
    assert result["status"] == "success", result
    assert _qpos(sim, "bob/shoulder") == pytest.approx(0.3)


def test_bare_name_without_robot_name_is_unchanged(sim):
    """No ``robot_name`` means the pre-existing first-match lookup, untouched."""
    result = sim.set_joint_positions({"j1": 0.7})
    assert result["status"] == "success", result
    assert _qpos(sim, "alice/j1") == pytest.approx(0.7)


def test_unknown_robot_name_falls_back_rather_than_raising(sim):
    """A robot_name no robot carries cannot scope, and must not crash the write."""
    result = sim.set_joint_positions({"j1": 0.2}, robot_name="nobody")
    assert result["status"] == "success", result
    assert _qpos(sim, "alice/j1") == pytest.approx(0.2)


def test_unresolvable_name_is_still_all_or_nothing(sim):
    """Scoping must not weaken the all-or-nothing guard."""
    before = sim._world._data.qpos.copy()
    result = sim.set_joint_positions({"j1": 0.5, "nope": 0.1}, robot_name="bob")
    assert result["status"] == "error", result
    assert np.array_equal(sim._world._data.qpos, before), "partial write leaked into qpos"


def test_velocities_scope_to_the_addressed_robot(sim):
    """The sibling setter shares the resolver, so it shares the contract."""
    result = sim.set_joint_velocities({"j1": 1.5}, robot_name="bob")
    assert result["status"] == "success", result
    assert _qvel(sim, "bob/j1") == pytest.approx(1.5)
    assert _qvel(sim, "alice/j1") == pytest.approx(0.0)


def test_list_form_writes_the_addressed_robot(sim):
    """The ordered form shared the defect, because it binds to short names.

    ``SimRobot.joint_names`` holds unqualified names (``['j1', 'j2',
    'shoulder']``), so the ordered form's ``dict(zip(joint_names, positions))``
    normalisation produced exactly the bare keys the resolver then sent to
    whichever robot declared them first. Measured on the unfixed tree, a
    3-vector for ``bob`` wrote ``alice/j1``, ``alice/j2`` and ``bob/shoulder``
    -- positionally bound to bob's joint *order* and then written to alice's
    joint *addresses*. Binding to ``robot_name`` in the resolver fixes both
    forms at once, which is why this is asserted rather than assumed.
    """
    joints = sim.robot_joint_names("bob")
    assert joints == ["j1", "j2", "shoulder"], joints
    result = sim.set_joint_positions([0.1] * len(joints), robot_name="bob")
    assert result["status"] == "success", result
    for joint in ("bob/j1", "bob/j2", "bob/shoulder"):
        assert _qpos(sim, joint) == pytest.approx(0.1), joint
    for joint in ("alice/j1", "alice/j2"):
        assert _qpos(sim, joint) == pytest.approx(0.0), joint


# A robot_name that cannot key the robot registry, one per unhashable builtin a
# caller might plausibly pass by mistake (a single-element list is what wrapping
# a name in brackets produces; a dict is what a half-built kwargs mapping looks
# like). Spelled out rather than derived from the types so the values are the
# ones a caller would actually pass, and to cover the same set as
# tests/simulation/test_unhashable_entity_name_is_reported.py.
UNHASHABLE_ROBOT_NAMES: list[tuple[str, Any]] = [
    ("list", ["bob"]),
    ("dict", {"bob": 1}),
    ("set", {"bob"}),
    ("bytearray", bytearray(b"bob")),
]


@pytest.mark.parametrize("kind,robot_name", UNHASHABLE_ROBOT_NAMES, ids=[k for k, _ in UNHASHABLE_ROBOT_NAMES])
def test_dict_form_reports_a_robot_name_that_cannot_be_a_key(sim, kind, robot_name):
    """Scoping made robot_name a registry lookup, which must stay total.

    The namespace lookup here is the dict form's *first* use of ``robot_name``:
    before scoping, the dict form ignored the argument outright, so no lookup
    existed to be partial. A bare ``self._world.robots.get(robot_name)`` is not
    total -- for a name that cannot be a key (a list, a dict, a set) the lookup
    itself raises ``TypeError: unhashable type``, which escapes the
    ``{"status", "content"}`` envelope this method documents as its only failure
    channel. Measured on the unscoped-lookup tree: all three names raised out of
    the dict form while the list form reported each one, so one argument had two
    answers depending on the form. Routing through
    :func:`~strands_robots.simulation.models.registry_entry` makes the name
    resolve to "no such robot", which is the fall-back path
    :func:`test_unknown_robot_name_falls_back_rather_than_raising` already pins.
    """
    result = sim.set_joint_positions({"j1": 0.4}, robot_name=robot_name)
    assert isinstance(result, dict) and "status" in result, f"{kind} name escaped the envelope: {result!r}"
    assert result["status"] == "success", result
    assert _qpos(sim, "alice/j1") == pytest.approx(0.4)


@pytest.mark.parametrize("kind,robot_name", UNHASHABLE_ROBOT_NAMES, ids=[k for k, _ in UNHASHABLE_ROBOT_NAMES])
def test_the_list_form_still_refuses_such_a_name(sim, kind, robot_name):
    """The control: the ordered form already reported it, and still must.

    Without this, the test above could be satisfied by making both forms raise.
    """
    result = sim.set_joint_positions([0.1, 0.2, 0.3], robot_name=robot_name)
    assert result["status"] == "error", result
    assert "not found" in result["content"][0]["text"], result


@pytest.mark.parametrize("kind,robot_name", UNHASHABLE_ROBOT_NAMES, ids=[k for k, _ in UNHASHABLE_ROBOT_NAMES])
def test_velocities_share_the_total_lookup(sim, kind, robot_name):
    """The sibling setter shares the resolver, so it shares the contract."""
    result = sim.set_joint_velocities({"j1": 1.0}, robot_name=robot_name)
    assert isinstance(result, dict) and "status" in result, f"{kind} name escaped the envelope: {result!r}"
    assert result["status"] == "success", result
    assert _qvel(sim, "alice/j1") == pytest.approx(1.0)
