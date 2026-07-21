"""Regression: warn when an action value is clamped by a ctrl-limited actuator.

The direct-actuator branch of ``_apply_action_by_name`` writes the action
value verbatim to ``data.ctrl``. When the actuator is ``ctrllimited`` and the
value is outside its ``ctrlrange``, MuJoCo clamps it inside ``mj_step`` - so
the commanded trajectory is silently NOT reproduced for that actuator while the
call still reports success. This is the failure mode of replaying a dataset
whose action units differ from the robot's actuator ctrl units (e.g. a
normalized gripper action in ``[0.19, 1.12]`` replayed onto a joint-position
gripper whose ctrlrange is ``[0.002, 0.037]``): every value clamps to the max,
pinning the gripper open and destroying the grasp channel.

These tests build a tiny synthetic model (no asset download) with a wide-range
arm actuator, a small-range ("gripper") ctrl-limited actuator, and an
unlimited actuator, and pin that ``_apply_action_by_name`` warns exactly when a
value is meaningfully outside a ctrl-limited actuator's range.
"""

from __future__ import annotations

import logging

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.rendering import RenderingMixin  # noqa: E402

_LOGGER = "strands_robots.simulation.mujoco.rendering"

# arm_act: wide ctrlrange; grip_act: small ctrlrange mirroring the ALOHA
# gripper ([0.002, 0.037]); free_act: no ctrlrange -> ctrllimited=0.
_XML = """
<mujoco model="clamp_test">
  <worldbody>
    <body name="link">
      <joint name="arm_joint" type="hinge" axis="0 0 1" range="-3 3"/>
      <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0 0 0.2"/>
      <body name="finger" pos="0 0 0.2">
        <joint name="grip_joint" type="slide" axis="1 0 0" range="0 0.041"/>
        <geom type="box" size="0.01 0.01 0.02"/>
      </body>
      <body name="spin" pos="0.1 0 0">
        <joint name="free_joint" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.01 0.01 0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="arm_act" joint="arm_joint" ctrlrange="-3 3"/>
    <position name="grip_act" joint="grip_joint" ctrlrange="0.002 0.037" kp="10"/>
    <position name="free_act" joint="free_joint" kp="1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def _apply(model, action, mixin=None):
    data = mujoco.MjData(model)
    mixin = mixin or RenderingMixin()
    mixin._apply_action_by_name(model, data, action, "", mujoco)
    return mixin, data


def test_warns_when_gripper_value_outside_ctrlrange(model, caplog):
    """A normalized gripper action (0.5) far above ctrlrange [0.002, 0.037] warns."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 0.5})
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    hit = [m for m in msgs if "grip_act" in m and "ctrlrange" in m]
    assert hit, f"expected a clamp warning naming grip_act; got {msgs}"
    # Names the actuator's actual ctrlrange so the user can self-correct.
    assert "0.002" in hit[0] and "0.037" in hit[0]


def test_no_warning_when_gripper_value_in_range(model, caplog):
    """An in-range gripper command (0.02) does not warn."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 0.02})
    assert not [r for r in caplog.records if "grip_act" in r.getMessage() and "ctrlrange" in r.getMessage()]


def test_no_warning_for_in_range_arm_value(model, caplog):
    """A wide-range arm actuator commanded within range does not warn."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"arm_act": 1.0})
    assert not [r for r in caplog.records if "arm_act" in r.getMessage() and "ctrlrange" in r.getMessage()]


def test_no_warning_for_unlimited_actuator(model, caplog):
    """An actuator with no ctrlrange (ctrllimited=0) never clamps -> no warning."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"free_act": 999.0})
    assert not [r for r in caplog.records if "free_act" in r.getMessage() and "ctrlrange" in r.getMessage()]


def test_warn_once_dedup(model, caplog):
    """Two out-of-range commands for the same key warn only once (no 50Hz spam)."""
    mixin = RenderingMixin()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 0.5}, mixin)
        _apply(model, {"grip_act": 0.8}, mixin)
    hits = [r for r in caplog.records if "grip_act" in r.getMessage() and "ctrlrange" in r.getMessage()]
    assert len(hits) == 1, f"expected exactly one dedup'd warning, got {len(hits)}"


def test_no_warning_for_degenerate_ctrlrange(model, caplog):
    """A ctrl-limited actuator whose range is degenerate ([v, v]) never clamps
    meaningfully, so an out-of-range command must not warn.

    A ``[0, 0]`` (or any ``lo >= hi``) ctrlrange is a sentinel, not a real
    limit: MuJoCo would pin every command to the single point, but the warning
    is about *unit mismatch*, not about a legitimately degenerate actuator.
    Emitting it here would be noise. The range is forced on the compiled model
    (the MJCF compiler auto-clears ``ctrllimited`` for a zero range) to mirror
    the sentinel state seen in the wild.
    """
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip_act")
    model.actuator_ctrlrange[grip_id] = [5.0, 5.0]
    model.actuator_ctrllimited[grip_id] = 1
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _apply(model, {"grip_act": 100.0})
    assert not [r for r in caplog.records if "grip_act" in r.getMessage() and "ctrlrange" in r.getMessage()]


def test_no_crash_for_stale_actuator_id(model, caplog):
    """A stale/out-of-range actuator id must be swallowed, not crash the loop.

    After a scene recompile an actuator id can outlive the model it indexed
    (fewer actuators than before). Indexing ``actuator_ctrllimited`` with it
    raises ``IndexError``; the 50Hz control loop must survive that -- the warn
    is best-effort diagnostics, never a hard dependency. No warning is emitted
    and no exception escapes.
    """
    mixin = RenderingMixin()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        mixin._warn_ctrl_clamp(model, model.nu + 99, "", "grip_act", 100.0, mujoco)
    assert not [r for r in caplog.records if "ctrlrange" in r.getMessage()]
