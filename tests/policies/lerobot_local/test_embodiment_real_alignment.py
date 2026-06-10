"""Regression tests pinning *_real embodiment keys 1:1 to the LeRobot drivers.

The ``*_real`` embodiments must declare ``state_keys``/``action_keys`` whose
names AND order match the corresponding LeRobot robot driver's
``observation_features``/``action_features`` exactly. ``PackStateProcessorStep``
packs ``observation.state`` in the declared ``state_keys`` order, so a wrong
name silently zero-pads that column (``dim_policy="pad"``) and a wrong order
mis-indexes every joint after the divergence point -- a policy trained on the
driver's native layout then receives scrambled values.

These cases were found by cross-comparing ``embodiments.json`` against the
installed LeRobot source:

* ``lekiwi_real``: the mobile base is exposed by ``LeKiwi._state_ft`` as
  body-frame velocities ``x.vel``/``y.vel``/``theta.vel`` (the driver converts
  wheel<->body internally); it is NOT three wheel ``.pos`` channels.
* ``reachy2_real``: ``Reachy2._generate_joints_dict`` builds the ordered key
  dict as neck -> l_arm -> r_arm -> antennas. The map must use that order, not
  neck -> antennas -> r_arm -> l_arm.
"""

from strands_robots.policies.lerobot_local.embodiment import load_embodiment


def test_lekiwi_real_base_is_body_velocity_not_wheel_pos():
    em = load_embodiment("lekiwi_real")
    assert em.state_keys == em.action_keys
    assert em.state_keys == [
        "arm_shoulder_pan.pos",
        "arm_shoulder_lift.pos",
        "arm_elbow_flex.pos",
        "arm_wrist_flex.pos",
        "arm_wrist_roll.pos",
        "arm_gripper.pos",
        "x.vel",
        "y.vel",
        "theta.vel",
    ]
    # The old (wrong) wheel-position keys must never come back.
    assert not any("wheel" in k for k in em.state_keys)


def test_reachy2_real_order_matches_generate_joints_dict():
    em = load_embodiment("reachy2_real")
    assert em.state_keys == em.action_keys
    # neck(3) -> l_arm(8) -> r_arm(8) -> antennas(2), per _generate_joints_dict.
    assert em.state_keys == [
        "neck_yaw.pos",
        "neck_pitch.pos",
        "neck_roll.pos",
        "l_shoulder_pitch.pos",
        "l_shoulder_roll.pos",
        "l_elbow_yaw.pos",
        "l_elbow_pitch.pos",
        "l_wrist_roll.pos",
        "l_wrist_pitch.pos",
        "l_wrist_yaw.pos",
        "l_gripper.pos",
        "r_shoulder_pitch.pos",
        "r_shoulder_roll.pos",
        "r_elbow_yaw.pos",
        "r_elbow_pitch.pos",
        "r_wrist_roll.pos",
        "r_wrist_pitch.pos",
        "r_wrist_yaw.pos",
        "r_gripper.pos",
        "l_antenna.pos",
        "r_antenna.pos",
    ]
    assert len(em.state_keys) == 21
    # Antennas live at the tail (idx 19-20), arms in the middle -- the old
    # ordering put antennas at idx 3-4 and swapped the arms.
    assert em.state_keys[3].startswith("l_shoulder")
    assert em.state_keys[19:] == ["l_antenna.pos", "r_antenna.pos"]
