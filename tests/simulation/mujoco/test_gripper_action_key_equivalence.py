"""A gripper command must not depend on which name the action key spells.

``send_action`` accepts an action key as EITHER the actuator name
(``actuator8`` on the Panda) or a joint name the actuator drives
(``finger_joint1``). Both resolve to the same actuator, so both must write the
same ``data.ctrl`` value.

They did not. Only the joint-name branch ran ``_scale_ctrl_for_actuator``, which
maps a logical ``[0, 1]`` open/close fraction onto a tendon gripper's wide
ctrlrange (the Panda's ``[0, 255]``). The actuator-name branch wrote the value
verbatim, so a normalised ``1.0`` meant "fully open" through the joint name and
"0.4% of range - closed" through the actuator name.

The actuator name is the spelling the engine advertises: ``robot_action_keys()``
returns actuator names, a policy receives them via ``set_robot_state_keys``, and
a positional action vector (``send_action([...])``, ``replay_episode``) binds to
them in order. So every policy and dataset-replay path took the unscaled branch
and could not open a tendon gripper at all.

These tests pin the equivalence on a synthetic model (no asset download): one
direct-joint actuator plus one tendon actuator with a Panda-like ``[0, 255]``
ctrlrange.
"""

from __future__ import annotations

import logging

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402

# One hinge driven directly ("arm_act") + two finger slides wrapped by a
# "split" tendon driven by "grip_act" with the Panda's remapped [0, 255] range.
_XML = """
<mujoco model="key_equivalence">
  <worldbody>
    <body name="link">
      <joint name="arm_joint" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0 0 0.2"/>
      <body name="hand" pos="0 0 0.2">
        <geom type="box" size="0.03 0.03 0.02"/>
        <body name="finger1" pos="0.03 0 0">
          <joint name="finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
          <geom type="box" size="0.005 0.01 0.02"/>
        </body>
        <body name="finger2" pos="-0.03 0 0">
          <joint name="finger_joint2" type="slide" axis="-1 0 0" range="0 0.04"/>
          <geom type="box" size="0.005 0.01 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="split">
      <joint joint="finger_joint1" coef="1"/>
      <joint joint="finger_joint2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="arm_act" joint="arm_joint" ctrlrange="-3 3"/>
    <position name="grip_act" tendon="split" ctrlrange="0 255" kp="1"/>
  </actuator>
</mujoco>
"""

# A tendon whose ctrlrange spans zero is already the normalised command space,
# so a symmetric command passes through clamped instead of being re-mapped.
_XML_SYMMETRIC = """
<mujoco model="key_equivalence_symmetric">
  <worldbody>
    <body name="hand">
      <body name="f1" pos="0.03 0 0">
        <joint name="fj1" type="slide" axis="1 0 0" range="-1 1"/>
        <geom type="box" size="0.005 0.01 0.02"/>
      </body>
      <body name="f2" pos="-0.03 0 0">
        <joint name="fj2" type="slide" axis="-1 0 0" range="-1 1"/>
        <geom type="box" size="0.005 0.01 0.02"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="sym_split">
      <joint joint="fj1" coef="1"/>
      <joint joint="fj2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="sym_act" tendon="sym_split" ctrlrange="-1 1" kp="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def _aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _ctrl_after(model, action):
    """Apply ``action`` to a fresh MjData and return the ctrl vector."""
    data = mujoco.MjData(model)
    RenderingMixin()._apply_action_by_name(model, data, action, "", mujoco)
    return data.ctrl.copy()


@pytest.mark.parametrize("command", [0.0, 0.25, 0.5, 1.0])
def test_actuator_name_and_joint_name_write_the_same_gripper_ctrl(model, command):
    """The same logical gripper command must land identically for both spellings.

    Pre-fix the actuator-name key wrote ``command`` verbatim (1.0 of a
    [0, 255] range = closed) while the joint-name key wrote the mapped
    fraction (255 = open) - the same call, opposite physical outcome.
    """
    grip = _aid(model, "grip_act")
    by_actuator = _ctrl_after(model, {"grip_act": command})[grip]
    by_joint = _ctrl_after(model, {"finger_joint1": command})[grip]
    assert by_actuator == pytest.approx(by_joint)
    assert by_actuator == pytest.approx(command * 255.0)


def test_actuator_name_open_command_fully_opens_tendon_gripper(model):
    """A normalised fully-open command through the actuator name reaches hi.

    This is the key spelling ``robot_action_keys()`` advertises and that a
    policy action vector binds to positionally, so it is the path every policy
    rollout and dataset replay takes.
    """
    grip = _aid(model, "grip_act")
    assert _ctrl_after(model, {"grip_act": 1.0})[grip] == pytest.approx(255.0)


def test_actuator_name_in_range_tendon_command_still_passes_verbatim(model):
    """An already-in-ctrlrange tendon command is trusted, not re-mapped."""
    grip = _aid(model, "grip_act")
    assert _ctrl_after(model, {"grip_act": 128.0})[grip] == pytest.approx(128.0)


def test_actuator_name_direct_joint_command_stays_raw(model):
    """A direct-transmission actuator keeps writing the raw value.

    The unit mapping applies to tendon transmissions only, so joint-position
    and torque actuators addressed by actuator name are unchanged.
    """
    arm = _aid(model, "arm_act")
    assert _ctrl_after(model, {"arm_act": 0.5})[arm] == pytest.approx(0.5)


def test_actuator_name_symmetric_tendon_command_passes_verbatim():
    """A tendon ctrlrange spanning zero is itself the normalised space."""
    m = mujoco.MjModel.from_xml_string(_XML_SYMMETRIC)
    sym = _aid(m, "sym_act")
    data = mujoco.MjData(m)
    RenderingMixin()._apply_action_by_name(m, data, {"sym_act": -0.5}, "", mujoco)
    assert data.ctrl[sym] == pytest.approx(-0.5)


def test_direct_actuator_name_out_of_range_still_warns(model, caplog):
    """The no-silent-clamp warning still fires for direct actuators."""
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.rendering"):
        _ctrl_after(model, {"arm_act": 99.0})
    assert any("outside" in r.message and "ctrlrange" in r.message for r in caplog.records)


def test_tendon_actuator_name_does_not_warn_about_clamping(model, caplog):
    """A mapped tendon command must not report a spurious clamp.

    ``_scale_ctrl_for_actuator`` deliberately maps the command into the
    ctrlrange, so warning that MuJoCo "will clamp it" would be noise at every
    control step - and it was emitted for the actuator-name spelling only.
    """
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.rendering"):
        _ctrl_after(model, {"grip_act": 1.0})
    assert not [r for r in caplog.records if "ctrlrange" in r.message]
