"""D1/D2: a LeRobot leader arm drives a strands robot in one line; units come from the leader's own bus declaration;
a LeRobot-format ``.pos`` dict given to ``act()`` is refused, naming the units problem and the fix."""

from __future__ import annotations

import numpy as np
import pytest
from test_kernel import CountingDriver, VirtualClock  # noqa: E402

from strands_robots import Description, Robot
from strands_robots.units import lerobot_units

lm = pytest.importorskip("lerobot.motors")
from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

D = Description.load("so101")
KEYS = D.action_keys


class FakeLeader:
    """A LeRobot SO leader without a serial port: its REAL FeetechMotorsBus normalises scripted encoder counts."""

    def __init__(self, use_degrees: bool) -> None:
        body = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
        motors = {k: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if k == "gripper" else body)
                  for i, k in enumerate(KEYS)}  # fmt: skip
        cal = {k: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=1000, range_max=3000)
               for i, k in enumerate(KEYS)}  # fmt: skip
        self.bus = FeetechMotorsBus(port="/dev/null-leader", motors=motors, calibration=cal)
        self.id = "v1_leader"
        self.t = 0
        self.sent: list[dict[str, float]] = []

    def get_action(self) -> dict[str, float]:
        self.t += 1
        counts = {i + 1: 1500 + 40 * self.t + 100 * i for i in range(len(KEYS))}
        norm = self.bus._normalize(counts)
        a = {f"{k}.pos": float(norm[i + 1]) for i, k in enumerate(KEYS)}
        self.sent.append(a)
        return a


@pytest.mark.parametrize("units", ["auto", "lerobot"])  # "lerobot" names the scheme the leader declares: its modes
@pytest.mark.parametrize("use_degrees", [False, True])
def test_d2_arm_run_leader_is_one_line_and_converts_exactly(use_degrees, units):
    clock = VirtualClock()
    drv = CountingDriver(clock)
    arm = Robot("so101", driver=drv, twin=False, clock=clock)
    leader = FakeLeader(use_degrees)
    res = arm.run(leader, hz=30, steps=10, units=units)  # the whole teleop program
    assert res.steps == 10 and res.deadline_misses == 0
    norm = {k: ("range_0_100" if k == "gripper" else ("degrees" if use_degrees else "range_m100_100")) for k in KEYS}
    conv = lerobot_units(D, norm)
    for (_t, written), sent in zip(drv.writes, leader.sent, strict=True):
        want = arm.coerce(conv.to_rad(np.array([sent[f"{k}.pos"] for k in KEYS])))
        np.testing.assert_allclose(written, want, atol=1e-12)


def test_d1_a_source_without_declared_units_is_refused_with_the_fix():
    class NoBus:
        def get_action(self):
            return {f"{k}.pos": 0.0 for k in KEYS}

    arm = Robot("so101", driver=CountingDriver(), twin=False)
    with pytest.raises(TypeError, match="not a policy"):
        arm.run(NoBus(), steps=1, hz=30)


def test_d1_a_lerobot_dict_given_to_act_is_refused_naming_the_units_problem():
    arm = Robot("so101", driver=CountingDriver(), twin=False)
    with pytest.raises(KeyError, match="LeRobot"):
        arm.act({"shoulder_pan.pos": 12.0})


def test_f12_a_degrees_leader_drives_a_range_mode_follower_through_the_followers_own_calibration():
    """The real SO-101 pair: LeRobot's leader in degrees, a follower whose registry default is range_m100_100 and
    which has its OWN bus calibration. Strands converts in the LEADER's units (its declaration) with the FOLLOWER's
    calibration -- exactly what LeRobot's teleoperate does (follower._unnormalize(leader._normalize(counts)))."""
    from test_drivers import make_driver

    from strands_robots.robot.drivers import feetech as ft

    d, serial = make_driver()  # the follower: its own calibration (tests/fixtures), real feetech driver, fake wire
    arm = Robot("so101", driver=d, twin=False)
    leader = FakeLeader(use_degrees=True)
    arm.run(leader, hz=30, steps=3)
    # what LeRobot would write on the follower: its own bus un-normalising the leader's degrees
    motors = {k: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if k == "gripper" else MotorNormMode.DEGREES)
              for i, k in enumerate(KEYS)}  # fmt: skip
    cal = {
        k: MotorCalibration(id=c.id, drive_mode=c.drive_mode, homing_offset=c.homing_offset, range_min=c.range_min,
                            range_max=c.range_max)
        for k, c in d.calibration.joints.items()
    }  # fmt: skip
    follower_bus = FeetechMotorsBus(port="/dev/null-follower", motors=motors, calibration=cal)
    want = follower_bus._unnormalize({i + 1: leader.sent[-1][f"{k}.pos"] for i, k in enumerate(KEYS)})
    regs = serial.regs
    for i, k in enumerate(KEYS):
        got = int.from_bytes(regs[d.ids[i]][ft.REG_GOAL_POSITION :][:2], "little")
        c = d.calibration.joints[k]
        assert abs(got - min(c.range_max, max(c.range_min, want[i + 1]))) <= 1, (k, got, want[i + 1])
