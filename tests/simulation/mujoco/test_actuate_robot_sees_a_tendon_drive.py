"""``actuate_robot`` refuses to double-actuate a joint a tendon already drives.

The guard's job is stated in its own refusal: "refusing to double-actuate".
It decided that by asking :func:`actuator_joint_id` which joint each existing
actuator drives - a function that answers ``-1`` for a tendon, because a tendon
is not a joint. A gripper whose fingers are coupled by one fixed tendon is the
standard MJCF idiom (``robotiq_2f85`` and ``robotiq_2f85_v4`` are driven that
way and by nothing else), so on those robots the guard saw no driven joint,
reported ``success``, and added a position servo per finger on top of the drive
that was already there. The two then fight, and the shipped gripper interface
stops moving the fingers.

The same package already resolves a tendon to the joints it wraps, on the
action-application path (``_actuator_for_joint``, issue #318). This suite pins
the guard against that same rule, and pins the parts a wider rule must NOT
change: a tendon wrapping a *different* robot's joints still must not refuse
(the id-space collision the narrower gate was introduced for), and a site
transmission still must not, because it moves a frame rather than commanding a
joint coordinate.

Every model here is inline MJCF loaded through ``add_robot(urdf_path=...)``, so
none of it downloads an asset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.manipulation import _MAX_NAMED_JOINTS  # noqa: E402
from strands_robots.simulation.mujoco.scene_ops import (  # noqa: E402
    actuator_driven_joint_ids,
    actuator_joint_id,
    tendon_joint_ids,
)
from tests.simulation.mujoco.test_actuator_ownership_by_transmission import (  # noqa: E402
    ARM_XML,
    GRIPPER_XML,
    UNACTUATED_ARM_XML,
    _write,
)

# A three-hinge arm with actuators on only two of them. The third is left free
# so the refusal has a "partly driven" case to word differently from a robot
# driven throughout.
PARTIAL_ARM_XML = """
<mujoco model="partial">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.04 0.04 0.04"/>
      <body name="link1" pos="0 0 0.08">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.02"/>
        <body name="link2" pos="0.14 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.018"/>
          <body name="link3" pos="0.12 0 0">
            <joint name="j3" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="1"/>
            <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.016"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j1_act" joint="j1" kp="30" dampratio="1"/>
    <position name="j2_act" joint="j2" kp="30" dampratio="1"/>
  </actuator>
</mujoco>
"""

# A hinge with no joint drive, plus a thrust actuator on a site. A site
# transmission applies a wrench at a frame, so the hinge is still free for
# ``actuate_robot`` to claim.
SITE_DRIVE_XML = """
<mujoco model="thruster">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hull">
      <geom type="box" size="0.05 0.05 0.02"/>
      <site name="nozzle" pos="0 0 -0.02"/>
      <body name="vane" pos="0 0 0.03">
        <joint name="vane_hinge" type="hinge" axis="0 1 0" range="-1 1" limited="true" damping="1"/>
        <geom type="box" size="0.01 0.03 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="thrust" site="nozzle" gear="0 0 1 0 0 0" ctrlrange="0 1"/>
  </actuator>
</mujoco>
"""

# A tendon gripper authored for a dynamics measurement rather than for id
# bookkeeping: explicit masses, real joint damping and a ctrlrange in tendon
# length units. The sibling suite's GRIPPER_XML carries near-zero mass, so any
# servo force there drives straight through the joint limit - fine for the id
# assertions it was written for, useless for reading travel off. Gravity is off
# so the fingers settle on the drive alone.
TENDON_GRIPPER_XML = """
<mujoco model="tgrip">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="palm">
      <geom type="box" size="0.03 0.03 0.02" mass="1.0"/>
      <body name="fl" pos="0.02 0 0.03">
        <joint name="finger1" type="slide" axis="1 0 0" range="0 0.04" limited="true" damping="4"/>
        <geom type="box" size="0.005 0.01 0.02" mass="0.05"/>
      </body>
      <body name="fr" pos="-0.02 0 0.03">
        <joint name="finger2" type="slide" axis="-1 0 0" range="0 0.04" limited="true" damping="4"/>
        <geom type="box" size="0.005 0.01 0.02" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="grip">
      <joint joint="finger1" coef="1"/>
      <joint joint="finger2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="finger_act" tendon="grip" kp="8" ctrlrange="0 0.08"/>
  </actuator>
</mujoco>
"""

# Six hinges, every one driven, so the refusal has more driven joints than it
# spells out. Generated rather than written so the count cannot drift from the
# assertion that reads it.
_WIDE_N = 6
WIDE_ARM_XML = (
    '<mujoco model="wide">\n  <compiler angle="radian"/>\n  <worldbody>\n    <body name="base">\n'
    '      <geom type="box" size="0.03 0.03 0.03"/>\n'
    + "".join(
        f'      <body name="l{i}" pos="0 0 {0.06 + 0.05 * i:.2f}">\n'
        f'        <joint name="w{i}" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="1"/>\n'
        f'        <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.012"/>\n'
        for i in range(_WIDE_N)
    )
    + "      </body>\n" * _WIDE_N
    + "    </body>\n  </worldbody>\n  <actuator>\n"
    + "".join(f'    <position name="w{i}_act" joint="w{i}" kp="30" dampratio="1"/>\n' for i in range(_WIDE_N))
    + "  </actuator>\n</mujoco>\n"
)

# One hinge driven BOTH by its own position actuator and by a tendon it shares
# with a neighbour. No stock registry asset wires a joint both ways, so the
# priority between the two passes is only observable on a model like this.
DOUBLE_WIRED_XML = """
<mujoco model="doublewired">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <geom type="box" size="0.03 0.03 0.02"/>
      <body name="a" pos="0.02 0 0.03">
        <joint name="ja" type="slide" axis="1 0 0" range="0 0.04" limited="true" damping="2"/>
        <geom type="box" size="0.005 0.01 0.02"/>
      </body>
      <body name="b" pos="-0.02 0 0.03">
        <joint name="jb" type="slide" axis="-1 0 0" range="0 0.04" limited="true" damping="2"/>
        <geom type="box" size="0.005 0.01 0.02"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="shared">
      <joint joint="ja" coef="1"/>
      <joint joint="jb" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="shared_act" tendon="shared" kp="40" ctrlrange="0 255"/>
    <position name="ja_act" joint="ja" kp="40"/>
  </actuator>
</mujoco>
"""


# ``sim`` is left un-annotated in the helpers: the engine types ``_world`` as
# ``SimWorld | None``, so an annotated helper would need a narrowing assert
# before every model read. Same convention as the sibling suites.


def _lone(tmp_path: Path, name: str, xml: str):
    """Build a world holding exactly one robot compiled from ``xml``."""
    from strands_robots.simulation import create_simulation

    sim = create_simulation("mujoco", tool_name="tendon_guard_sim", mesh=False)
    assert sim.create_world()["status"] == "success"
    path = _write(tmp_path, f"{name}.xml", xml)
    assert sim.add_robot(name=name, urdf_path=path)["status"] == "success"
    return sim


def _joint_value(sim, full_name: str) -> float:
    model, data = sim._world._model, sim._world._data
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
    assert jnt >= 0, full_name
    return float(data.qpos[int(model.jnt_qposadr[jnt])])


def _close_the_gripper(sim) -> tuple[float, float]:
    """Command the pre-existing tendon drive shut and report the finger travel.

    ``0.75`` is a close fraction: a small value on a tendon drive is mapped onto
    the actuator's ctrlrange, which is the documented way to address a coupled
    gripper without knowing its tendon units.
    """
    start = (_joint_value(sim, "grip/finger1"), _joint_value(sim, "grip/finger2"))
    assert sim.send_action({"finger_act": 0.75}, robot_name="grip")["status"] == "success"
    assert sim.step(1200)["status"] == "success"
    end = (_joint_value(sim, "grip/finger1"), _joint_value(sim, "grip/finger2"))
    return (end[0] - start[0], end[1] - start[1])


# -- the guard sees the tendon ----------------------------------------------


def test_a_robot_driven_only_through_a_tendon_is_refused(tmp_path: Path) -> None:
    """Both fingers are already driven, so there is nothing to add."""
    sim = _lone(tmp_path, "grip", GRIPPER_XML)
    try:
        nu_before = int(sim._world._model.nu)
        result = sim.actuate_robot(robot_name="grip", kp=40.0)
        assert result["status"] == "error", result["content"][0]["text"]
        assert int(sim._world._model.nu) == nu_before, "a refused call must add no actuator"
    finally:
        sim.cleanup()


def test_the_existing_tendon_drive_keeps_the_travel_it_had(tmp_path: Path) -> None:
    """The point of refusing: the shipped gripper interface still shuts the fingers.

    The reference is the same gripper's own travel measured before the call, so
    the assertion carries no hand-written magic number.
    """
    sim = _lone(tmp_path, "grip", TENDON_GRIPPER_XML)
    try:
        reference = _close_the_gripper(sim)
        assert reference[0] > 0.005, f"premise: the gripper must close on its own drive, got {reference}"
        assert reference[0] < 0.04, f"premise: the fingers must stay inside their limit, got {reference}"
        assert sim.reset()["status"] == "success"

        sim.actuate_robot(robot_name="grip", kp=40.0)

        after = _close_the_gripper(sim)
        assert after[0] == pytest.approx(reference[0], abs=1e-3), (
            f"the tendon drive moved finger1 {reference[0]:.4f} rad before actuate_robot "
            f"and {after[0]:.4f} rad after it"
        )
        assert after[1] == pytest.approx(reference[1], abs=1e-3)
    finally:
        sim.cleanup()


def test_the_refusal_names_each_driven_joint_and_the_actuator_driving_it(tmp_path: Path) -> None:
    """A caller cannot act on "already has actuators" without knowing which."""
    sim = _lone(tmp_path, "grip", GRIPPER_XML)
    try:
        text = sim.actuate_robot(robot_name="grip", kp=40.0)["content"][0]["text"]
        assert "finger1" in text and "finger2" in text, text
        assert "finger_act" in text, text
    finally:
        sim.cleanup()


def test_a_robot_driven_throughout_says_it_needs_no_added_actuators(tmp_path: Path) -> None:
    """Every joint driven: the useful next step is to command what is there."""
    sim = _lone(tmp_path, "arm", ARM_XML)
    try:
        text = sim.actuate_robot(robot_name="arm", kp=40.0)["content"][0]["text"]
        assert "2 of the 2" in text, text
        assert "already drivable" in text, text
    finally:
        sim.cleanup()


def test_a_partly_driven_robot_says_why_the_rest_was_left_alone(tmp_path: Path) -> None:
    """Naming the count is what tells a caller the remainder is not an oversight."""
    sim = _lone(tmp_path, "partial", PARTIAL_ARM_XML)
    try:
        text = sim.actuate_robot(robot_name="partial", kp=40.0)["content"][0]["text"]
        assert "2 of the 3" in text, text
        assert "remaining 1" in text, text
        assert "already drivable" not in text, text
    finally:
        sim.cleanup()


def test_the_refusal_summarizes_once_past_the_named_cap(tmp_path: Path) -> None:
    """A dexterous hand carries dozens; the verdict must not drown in the list."""
    sim = _lone(tmp_path, "wide", WIDE_ARM_XML)
    try:
        text = sim.actuate_robot(robot_name="wide", kp=40.0)["content"][0]["text"]
        listed = text.split("existing actuator (", 1)[1].split(")", 1)[0]
        named = [part for part in listed.split(", ") if " <- " in part]
        assert len(named) == _MAX_NAMED_JOINTS, listed
        assert f"and {_WIDE_N - _MAX_NAMED_JOINTS} more" in listed, listed
        assert f"{_WIDE_N} of the {_WIDE_N}" in text, text
    finally:
        sim.cleanup()


# -- what a wider rule must not change --------------------------------------


def test_a_robot_carrying_no_actuator_still_actuates(tmp_path: Path) -> None:
    """The capability itself is untouched: an unactuated arm still gets servos."""
    sim = _lone(tmp_path, "arm", UNACTUATED_ARM_XML)
    try:
        nu_before = int(sim._world._model.nu)
        result = sim.actuate_robot(robot_name="arm", kp=40.0)
        assert result["status"] == "success", result["content"][0]["text"]
        assert int(sim._world._model.nu) == nu_before + 2
    finally:
        sim.cleanup()


def test_a_site_transmission_does_not_block_actuation(tmp_path: Path) -> None:
    """A thruster moves a frame; the hinge beside it still has no drive of its own."""
    sim = _lone(tmp_path, "thruster", SITE_DRIVE_XML)
    try:
        result = sim.actuate_robot(robot_name="thruster", kp=40.0)
        assert result["status"] == "success", result["content"][0]["text"]
    finally:
        sim.cleanup()


# -- the shared rule ---------------------------------------------------------


def test_actuator_driven_joint_ids_reports_every_joint_a_tendon_wraps(tmp_path: Path) -> None:
    """The wider question resolves the tendon; the narrower one still says "no joint"."""
    sim = _lone(tmp_path, "grip", GRIPPER_XML)
    try:
        model = sim._world._model
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip/finger_act")
        wrapped = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"grip/{j}") for j in ("finger1", "finger2")}
        assert actuator_driven_joint_ids(model, act, mujoco) == wrapped
        assert actuator_joint_id(model, act, mujoco) == -1, "the per-ctrl question is unchanged"
    finally:
        sim.cleanup()


def test_actuator_driven_joint_ids_is_empty_for_a_site_transmission(tmp_path: Path) -> None:
    """A frame wrench commands no joint coordinate."""
    sim = _lone(tmp_path, "thruster", SITE_DRIVE_XML)
    try:
        model = sim._world._model
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thruster/thrust")
        assert actuator_driven_joint_ids(model, act, mujoco) == frozenset()
    finally:
        sim.cleanup()


def test_tendon_joint_ids_refuses_an_out_of_range_tendon(tmp_path: Path) -> None:
    """A caller holding a stale id gets an empty answer, not an out-of-bounds read."""
    sim = _lone(tmp_path, "grip", GRIPPER_XML)
    try:
        model = sim._world._model
        assert tendon_joint_ids(model, -1, mujoco) == frozenset()
        assert tendon_joint_ids(model, int(model.ntendon), mujoco) == frozenset()
    finally:
        sim.cleanup()


def test_a_joints_own_drive_keeps_priority_over_a_tendon_it_shares(tmp_path: Path) -> None:
    """Writing a joint addresses its own ctrl, not a tendon coupling its neighbour.

    Both actuators reach ``ja``, and the tendon is declared first, so a lookup
    that stopped at the first match would command the pair through the tendon
    and drag ``jb`` along with it.
    """
    sim = _lone(tmp_path, "dw", DOUBLE_WIRED_XML)
    try:
        model = sim._world._model
        ja = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dw/ja")
        shared = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "dw/shared_act")
        own = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "dw/ja_act")
        assert shared < own, "premise: the tendon drive must come first in model order"
        assert ja in actuator_driven_joint_ids(model, shared, mujoco)

        assert sim.send_action({"ja": 0.02}, robot_name="dw")["status"] == "success"
        assert float(sim._world._data.ctrl[own]) == pytest.approx(0.02)
        assert float(sim._world._data.ctrl[shared]) == pytest.approx(0.0)
    finally:
        sim.cleanup()
