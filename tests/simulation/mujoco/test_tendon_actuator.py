"""Regression tests for issue #318: MuJoCo tendon-transmission actuators.

``_apply_action_by_name`` previously resolved a joint-name action key only to
*direct* joint-transmission actuators (``actuator_trnid[ai, 0] == jnt_id``).
Tendon-transmission actuators - most notably the Franka/Panda gripper, whose
``split`` actuator drives ``finger_joint1`` / ``finger_joint2`` through a
tendon - were silently dropped, so the gripper never actuated.

These tests build a tiny synthetic model (no asset download) with one direct
joint actuator and one tendon actuator wrapping two finger joints, and pin:

1. ``_actuator_for_joint`` resolves a finger joint to the tendon actuator.
2. ``_actuator_for_joint`` still resolves a direct joint to its actuator.
3. ``_scale_ctrl_for_actuator`` maps a logical [0, 1] command onto the
   tendon actuator's wide ctrlrange, and leaves direct-joint commands raw.
4. End-to-end ``_apply_action_by_name`` writes a non-zero ctrl for a finger
   joint key (the bug was a silent no-op).
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402

# A minimal arm: one hinge "arm_joint" (direct actuator) + two finger slides
# wrapped by a "split" tendon driven by a single tendon actuator with a wide
# ctrlrange (mirrors the Panda gripper's [0, 255]).
_XML = """
<mujoco model="tendon_test">
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


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _aid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def test_actuator_for_joint_resolves_tendon_gripper(model):
    """A finger joint must resolve to the tendon actuator (issue #318)."""
    fj = _jid(model, "finger_joint1")
    grip = _aid(model, "grip_act")
    assert RenderingMixin._actuator_for_joint(model, fj, mujoco) == grip


def test_actuator_for_joint_resolves_second_finger(model):
    """The other tendon-wrapped joint resolves to the same actuator."""
    fj2 = _jid(model, "finger_joint2")
    grip = _aid(model, "grip_act")
    assert RenderingMixin._actuator_for_joint(model, fj2, mujoco) == grip


def test_actuator_for_joint_resolves_direct_joint(model):
    """Direct joint transmission keeps working (no regression)."""
    aj = _jid(model, "arm_joint")
    arm = _aid(model, "arm_act")
    assert RenderingMixin._actuator_for_joint(model, aj, mujoco) == arm


def test_actuator_for_joint_unknown_returns_negative(model):
    """A joint that drives no actuator returns -1."""
    assert RenderingMixin._actuator_for_joint(model, 999, mujoco) == -1


def test_scale_maps_logical_open_to_full_range(model):
    """A logical 1.0 (fully open) maps to the tendon ctrlrange max."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 1.0, mujoco)
    assert out == pytest.approx(255.0)


def test_scale_maps_logical_close_to_range_min(model):
    """A logical 0.0 (closed) maps to the ctrlrange min."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 0.0, mujoco)
    assert out == pytest.approx(0.0)


def test_scale_passes_through_clear_in_range_command(model):
    """A clearly in-range tendon command (e.g. 128) is trusted verbatim."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 128.0, mujoco)
    assert out == pytest.approx(128.0)


def test_scale_leaves_direct_joint_value_raw(model):
    """Direct JOINT actuators are never rescaled (positions/torques)."""
    arm = _aid(model, "arm_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, arm, 0.5, mujoco)
    assert out == pytest.approx(0.5)


def test_apply_action_by_name_drives_tendon_gripper(model):
    """End-to-end: a finger-joint key writes a non-zero tendon ctrl.

    Before the fix this was a silent no-op (data.ctrl stayed 0).
    """
    data = mujoco.MjData(model)
    mixin = RenderingMixin()
    mixin._apply_action_by_name(model, data, {"finger_joint1": 1.0}, "", mujoco)
    grip = _aid(model, "grip_act")
    assert data.ctrl[grip] == pytest.approx(255.0)


def test_apply_action_by_name_direct_joint_unscaled(model):
    """End-to-end: a direct joint key writes the raw command."""
    data = mujoco.MjData(model)
    mixin = RenderingMixin()
    mixin._apply_action_by_name(model, data, {"arm_joint": 0.5}, "", mujoco)
    arm = _aid(model, "arm_act")
    assert data.ctrl[arm] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# #367 v0.4.1 polish bundle                                                    #
# --------------------------------------------------------------------------- #
# A tendon actuator whose ctrlrange spans zero ([-1, 1]) -- the #367 item-1a
# case. The scale helper only special-cases TENDON transmissions, so the
# symmetric-range pin must use a tendon actuator (not the direct arm joint).
_XML_SYMMETRIC_TENDON = """
<mujoco model="symmetric_tendon">
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


def test_scale_symmetric_tendon_ctrlrange_passes_negative_verbatim():
    """#367 item 1a: a TENDON ctrlrange spanning zero ([-1, 1]) is itself the
    normalised command space -- a symmetric negative command must pass through
    clamped, NOT be re-mapped onto [lo, hi]. Pre-fix (lo<0 with value<lo+1)
    fell into the fraction branch and clipped -0.5 toward lo=-1.0."""
    m = mujoco.MjModel.from_xml_string(_XML_SYMMETRIC_TENDON)
    sym = _aid(m, "sym_act")
    out_neg = RenderingMixin._scale_ctrl_for_actuator(m, sym, -0.5, mujoco)
    assert out_neg == pytest.approx(-0.5), f"symmetric -0.5 must pass verbatim, got {out_neg}"
    out_pos = RenderingMixin._scale_ctrl_for_actuator(m, sym, 0.5, mujoco)
    assert out_pos == pytest.approx(0.5)
    # Out-of-range clamps to the symmetric bounds.
    assert RenderingMixin._scale_ctrl_for_actuator(m, sym, -5.0, mujoco) == pytest.approx(-1.0)
    assert RenderingMixin._scale_ctrl_for_actuator(m, sym, 5.0, mujoco) == pytest.approx(1.0)


def test_scale_epsilon_boundary_treats_one_plus_noise_as_fraction(model):
    """#367 item 1b: a normalised 1.0 + FP noise must still map to hi (255),
    not slip into the verbatim branch and write ~1.0 onto the [0, 255] range
    (a nearly-closed gripper when fully-open was intended)."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 1.0 + 1e-9, mujoco)
    assert out == pytest.approx(255.0), f"1.0+noise should map to hi, got {out}"


def test_scale_clamps_above_hi(model):
    """#367 item 3a: a value above hi clamps to hi (characterization pin)."""
    grip = _aid(model, "grip_act")
    out = RenderingMixin._scale_ctrl_for_actuator(model, grip, 300.0, mujoco)
    assert out == pytest.approx(255.0)


def test_apply_action_warns_once_on_unresolved_key(model, caplog):
    """#367 item 2: an action key resolving to neither actuator nor joint is
    no longer silently dropped -- it emits a WARNING (once per key)."""
    import logging

    data = mujoco.MjData(model)
    mixin = RenderingMixin()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.rendering"):
        mixin._apply_action_by_name(model, data, {"nonexistent_joint": 1.0}, "", mujoco)
        # Second call with the same key must NOT add another warning.
        mixin._apply_action_by_name(model, data, {"nonexistent_joint": 1.0}, "", mujoco)
    warns = [r.getMessage() for r in caplog.records if "nonexistent_joint" in r.getMessage()]
    assert len(warns) == 1, f"expected exactly one warn-once for the unresolved key, got {warns}"
