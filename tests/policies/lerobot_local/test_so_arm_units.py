"""Regression tests for the SO-arm degrees<->radians units conversion.

MolmoAct2 SO-100/101 (and other lerobot SO-arm checkpoints) emit joint actions
in the driver's MotorNormMode: arm joints in DEGREES, gripper in RANGE_0_100
(see lerobot/robots/so_follower/so_follower.py). The MuJoCo sim joints are
RADIANS. Without conversion the raw degree values saturate the radian joint
limits and the arm freezes -- this pins the fix.
"""

import math

import numpy as np

from strands_robots.policies.lerobot_local.embodiment import (
    EmbodimentMap,
    _convert_joint_vector,
    load_embodiment,
)

# so101 sim gripper joint range (robotstudio_so101/so101_new_calib.xml joint 6).
GRIPPER_RANGE = [-0.175, 1.745]


def _so_arm_map() -> EmbodimentMap:
    return EmbodimentMap(
        name="test_so",
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        state_units="degrees",
        action_units="degrees",
        gripper_index=5,
        gripper_joint_range=GRIPPER_RANGE,
    )


def test_action_degrees_to_radians_arm():
    emb = _so_arm_map()
    # 90 deg arm joints -> pi/2 rad; gripper handled separately below.
    out = emb.model_action_to_sim([90.0, 90.0, 90.0, 90.0, 90.0, 50.0])
    for v in out[:5]:
        assert math.isclose(v, math.pi / 2, abs_tol=1e-6), out


def test_action_gripper_range_0_100_to_joint():
    emb = _so_arm_map()
    lo, hi = GRIPPER_RANGE
    # 0 -> lo, 100 -> hi, 50 -> midpoint.
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 0])[5], lo, abs_tol=1e-6)
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 100])[5], hi, abs_tol=1e-6)
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 50])[5], (lo + hi) / 2, abs_tol=1e-6)


def test_state_radians_to_degrees_round_trip():
    emb = _so_arm_map()
    sim_state = [0.5, -1.0, 1.2, -0.3, 2.0, 0.4]
    to_model = emb.sim_state_to_model(sim_state)
    back = emb.model_action_to_sim(to_model)
    assert np.allclose(back, sim_state, atol=1e-6), (back, sim_state)


def test_native_units_is_noop():
    emb = EmbodimentMap(name="t", action_units="native", state_units="native")
    vals = [125.0, -270.0, 10.0]
    assert emb.model_action_to_sim(vals) == vals
    assert emb.sim_state_to_model(vals) == vals


def test_degree_action_stays_in_so101_joint_range():
    """A realistic MolmoAct2 degree action (mean joint2 ~125 deg) must land
    inside the so101 radian joint limits after conversion -- the whole point of
    the fix (raw degrees saturate; converted radians fit)."""
    emb = _so_arm_map()
    # so101 joint ranges (rad): 1:+/-1.92 2:+/-1.745 3:[-1.745,1.571] 4:+/-1.658 5:+/-2.793
    deg_action = [40.0, 95.0, 80.0, 50.0, -60.0, 50.0]  # within trained quantiles + joint limits
    rad = emb.model_action_to_sim(deg_action)
    limits = [(-1.92, 1.92), (-1.745, 1.745), (-1.745, 1.571), (-1.658, 1.658), (-2.793, 2.793), tuple(GRIPPER_RANGE)]
    for v, (lo, hi) in zip(rad, limits, strict=True):
        assert lo - 1e-6 <= v <= hi + 1e-6, (v, lo, hi)


def test_so101_embodiment_declares_degrees():
    emb = load_embodiment("so101")
    assert emb.state_units == "degrees"
    assert emb.action_units == "degrees"
    assert emb.gripper_index == 5
    assert emb.gripper_joint_range == [-0.175, 1.745]


def test_so_real_embodiment_stays_native():
    """Real hardware speaks the driver units already -- must NOT double-convert."""
    emb = load_embodiment("so_real")
    assert emb.state_units == "native"
    assert emb.action_units == "native"


def test_convert_helper_does_not_mutate_input():
    src = [10.0, 20.0, 30.0]
    _convert_joint_vector(src, to_model=False)
    assert src == [10.0, 20.0, 30.0]


# Mid-point centering (BUG-3): LeRobot MotorNormMode.DEGREES is mid-centered.
# Ground truth lerobot/motors/motors_bus.py: reported degrees =
# (raw - mid) * 360 / max_res with mid = (range_min + range_max) / 2.
# The sim expresses absolute joint angles, so without the per-joint mid the
# packed observation.state is offset from the training distribution. These pin
# the joint_mids mechanism that carries the calibration mid into the conversion.

MIDS = [10.0, -20.0, 30.0, -5.0, 15.0, 0.0]  # per-joint mid offsets (degrees)


def _so_arm_map_with_mids() -> EmbodimentMap:
    return EmbodimentMap(
        name="test_so_mids",
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        state_units="degrees",
        action_units="degrees",
        gripper_index=5,
        gripper_joint_range=GRIPPER_RANGE,
        joint_mids=MIDS,
    )


def test_joint_mids_subtracts_calibration_midpoint():
    """sim -> model must subtract the per-joint mid (mid-centered degrees),
    matching motors_bus, rather than emitting the absolute joint angle."""
    emb = _so_arm_map_with_mids()
    # 90 deg arm joints (pi/2 rad). Model state = 90 - mid per joint.
    sim_state = [math.pi / 2] * 5 + [0.5]  # gripper handled separately
    out = emb.sim_state_to_model(sim_state)
    for i in range(5):
        assert math.isclose(out[i], 90.0 - MIDS[i], abs_tol=1e-6), (i, out)


def test_joint_mids_round_trip():
    """sim -> model -> sim recovers the original sim state with mids applied."""
    emb = _so_arm_map_with_mids()
    sim_state = [0.5, -1.0, 1.2, -0.3, 2.0, 0.4]
    back = emb.model_action_to_sim(emb.sim_state_to_model(sim_state))
    assert np.allclose(back, sim_state, atol=1e-6), (back, sim_state)


def test_joint_mids_gripper_column_exempt():
    """The gripper column uses RANGE_0_100 and must ignore its mid entry even
    when joint_mids supplies a (non-zero) value at that index."""
    mids = [0.0, 0.0, 0.0, 0.0, 0.0, 999.0]  # bogus gripper mid -> must be ignored
    emb = EmbodimentMap(
        name="t",
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        state_units="degrees",
        action_units="degrees",
        gripper_index=5,
        gripper_joint_range=GRIPPER_RANGE,
        joint_mids=mids,
    )
    lo, hi = GRIPPER_RANGE
    # 50 -> midpoint regardless of the bogus gripper mid entry.
    assert math.isclose(emb.model_action_to_sim([0, 0, 0, 0, 0, 50])[5], (lo + hi) / 2, abs_tol=1e-6)


def test_joint_mids_matches_motors_bus_degrees_formula():
    """Parity with lerobot motors_bus DEGREES mode for a 1:1-geared servo
    (encoder full turn = 360 deg). For a physical angle and calibration mid,
    motors_bus reports (raw - mid) * 360 / max_res == angle_deg - mid_deg, which
    is exactly what the mid-centered conversion must produce."""
    emb = _so_arm_map_with_mids()
    max_res = 4095  # STS3215 12-bit resolution - 1
    for joint, angle_deg in enumerate([12.0, -47.0, 88.0, -3.0, 61.0]):
        mid_deg = MIDS[joint]
        raw = angle_deg * max_res / 360.0
        mid_ticks = mid_deg * max_res / 360.0
        motors_bus_degrees = (raw - mid_ticks) * 360.0 / max_res
        sim_rad = math.radians(angle_deg)
        ours = emb.sim_state_to_model([sim_rad] * 6)[joint]
        assert math.isclose(ours, motors_bus_degrees, abs_tol=1e-6), (joint, ours, motors_bus_degrees)


def test_empty_joint_mids_preserves_prior_behavior():
    """Default (no joint_mids) must be the prior absolute-degrees behavior:
    deg = rad * 180/pi with no offset."""
    emb = _so_arm_map()  # no joint_mids
    assert emb.joint_mids == []
    out = emb.sim_state_to_model([math.pi / 2] * 5 + [0.5])
    for i in range(5):
        assert math.isclose(out[i], 90.0, abs_tol=1e-6), out


def test_convert_helper_joint_mids_does_not_mutate_input():
    src = [10.0, 20.0, 30.0]
    _convert_joint_vector(src, to_model=True, joint_mids=[1.0, 2.0, 3.0])
    assert src == [10.0, 20.0, 30.0]


# Degenerate gripper range: a robot whose gripper joint has a zero-width range
# (``min == max``, e.g. a welded/fixed gripper joint or a malformed asset) must
# not crash the units conversion. ``_convert_joint_vector`` normalises the
# gripper column via ``(v - lo) / span``; a zero span would raise
# ZeroDivisionError and take down every observation/action pass. The guard
# treats a zero-span gripper as a pass-through (value unchanged) in both
# directions, so the rest of the vector still converts.


def _zero_span_map() -> EmbodimentMap:
    return EmbodimentMap(
        name="test_zero_span",
        state_keys=["1", "2", "3", "4", "5", "6"],
        action_keys=["1", "2", "3", "4", "5", "6"],
        state_units="degrees",
        action_units="degrees",
        gripper_index=5,
        gripper_joint_range=[0.5, 0.5],  # zero span
    )


def test_zero_span_gripper_does_not_divide_by_zero_to_model():
    """sim -> model with a zero-span gripper must not raise and must leave the
    gripper column untouched while arm joints still convert rad -> deg."""
    emb = _zero_span_map()
    out = emb.sim_state_to_model([math.pi / 2] * 5 + [1.23])
    assert out[5] == 1.23  # gripper passed through unchanged
    for i in range(5):
        assert math.isclose(out[i], 90.0, abs_tol=1e-6), out


def test_zero_span_gripper_does_not_divide_by_zero_from_model():
    """model -> sim with a zero-span gripper must not raise and must leave the
    gripper column untouched while arm joints still convert deg -> rad."""
    emb = _zero_span_map()
    out = emb.model_action_to_sim([90.0] * 5 + [1.23])
    assert out[5] == 1.23  # gripper passed through unchanged
    for i in range(5):
        assert math.isclose(out[i], math.pi / 2, abs_tol=1e-6), out
