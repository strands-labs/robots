"""A tool-point site outranks the end-effector body, including a same-named one.

:func:`strands_robots.simulation.ik.discover_ee_frame` resolves the Cartesian IK
target frame that ``move_to`` servos and that eef-delta policies decode against.
Its ladder is documented site-first - "a TCP-like site, else a hand/tool body,
else the chain's leaf body" - because a site is placed at the tool point while a
body origin sits at the link's mount, several centimetres behind it.

Rung 1 only searched TCP-specific spellings (``attachment_site`` / ``grasp`` /
``tcp`` / ...), while rung 2 matched a wider end-effector vocabulary on bodies
(``gripper`` / ``hand`` / ``tool`` / ...). A model that publishes its tool point
as a site named for the end effector therefore skipped rung 1 and was answered
from rung 2 with a link origin - silently, and with the correct frame present in
the same model. Shipped models do exactly this: ``so101`` names both its tool
site and its jaw body ``gripper`` (98 mm apart), ``aloha`` pairs the ``gripper``
site with the ``gripper_link`` body (130 mm), and ``toddlerbot`` pairs the
``left_hand_center`` site with the ``left_hand`` body (61 mm). Nothing reported
the substitution, so a solved ``move_to`` left the fingertips short of the
target by that offset.

Rung 1 now searches the TCP spellings first and then rung 2's vocabulary, so a
site wins whenever one names the end effector - the same-name case included.
Pinned on the frame's resolved world position, not only on its name, because the
offset is the whole defect.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.ik import discover_ee_frame  # noqa: E402

# A jaw body and its tool site share the bare name ``gripper``, as ``so101``
# ships it. The site sits 98 mm out along the link, matching that model's
# measured site-to-body-origin offset.
_TOOL_OFFSET_M = 0.098

_SAME_NAME_ARM = f"""
<mujoco><worldbody>
  <body name="arm/base">
    <joint name="arm/j0" type="hinge"/><geom type="box" size=".05 .05 .05"/>
    <body name="arm/gripper" pos="0 0 .2">
      <joint name="arm/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
      <site name="arm/gripper" pos="0 0 {_TOOL_OFFSET_M}"/>
    </body>
  </body>
</worldbody></mujoco>
"""


def _framed(xml: str, namespace: str) -> tuple[tuple[str, str], np.ndarray]:
    """Resolve the ee-frame and return it with its world position."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = discover_ee_frame(model, namespace)
    assert frame is not None
    name, kind = frame
    obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
    index = mujoco.mj_name2id(model, obj, name)
    position = data.site_xpos[index] if kind == "site" else data.xpos[index]
    return frame, np.asarray(position, dtype=float).copy()


def test_tool_site_wins_over_the_body_that_shares_its_name() -> None:
    """The site is chosen, and the frame lands at the tool point not the mount."""
    frame, position = _framed(_SAME_NAME_ARM, "arm/")

    assert frame == ("arm/gripper", "site")
    # The jaw body origin is at z=0.2; the tool point is _TOOL_OFFSET_M beyond it.
    assert position[2] == pytest.approx(0.2 + _TOOL_OFFSET_M, abs=1e-9)


def test_end_effector_named_site_wins_over_a_differently_named_body() -> None:
    """A site named for the hand outranks the hand body (the toddlerbot shape)."""
    xml = """
    <mujoco><worldbody>
      <body name="bot/torso">
        <joint name="bot/j0" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        <body name="bot/left_hand" pos="0 0 .3">
          <joint name="bot/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
          <site name="bot/left_hand_center" pos="0 0 .061"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    frame, position = _framed(xml, "bot/")

    assert frame == ("bot/left_hand_center", "site")
    assert position[2] == pytest.approx(0.361, abs=1e-9)


def test_a_tcp_named_site_still_outranks_an_end_effector_named_site() -> None:
    """Rung 1 keeps its own spellings ahead of rung 2's vocabulary.

    A model carrying both a purpose-built TCP site and a site named for the hand
    must resolve the TCP one, whichever order they are declared in.
    """
    xml = """
    <mujoco><worldbody>
      <body name="a/l0">
        <joint name="a/j0" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        <site name="a/gripper" pos="0 0 .1"/>
        <site name="a/grasp_point" pos="0 0 .2"/>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(mujoco.MjModel.from_xml_string(xml), "a/") == (
        "a/grasp_point",
        "site",
    )


def test_a_site_less_model_still_resolves_to_the_end_effector_body() -> None:
    """Rung 2 stays reachable: widening rung 1 must not strand a site-less model."""
    xml = """
    <mujoco><worldbody>
      <body name="arm/base">
        <joint name="arm/j0" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        <body name="arm/gripper" pos="0 0 .2">
          <joint name="arm/j1" type="hinge"/><geom type="box" size=".05 .05 .05"/>
        </body>
      </body>
    </worldbody></mujoco>
    """
    assert discover_ee_frame(mujoco.MjModel.from_xml_string(xml), "arm/") == (
        "arm/gripper",
        "body",
    )
