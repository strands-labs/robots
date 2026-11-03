"""A joint an Action leaves out keeps its last commanded target; it does not re-target wherever gravity took it."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import JointState, Robot, Simulation
from strands_robots.description import Description

KEYS = Description.load("so101").action_keys
LIFT, ELBOW = KEYS.index("shoulder_lift"), KEYS.index("elbow_flex")


class SaggingDriver:
    """A hardware body under load: every joint reads 0.1 rad below the goal it was last given."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self) -> None:
        self.names, self.goal, self.holds = KEYS, np.zeros(len(KEYS)), 0

    def read(self) -> JointState:
        z = np.zeros(len(self.names))
        return JointState(self.names, self.goal - 0.1, z, z.copy())

    def write(self, targets) -> None:
        self.goal = np.asarray(targets, dtype=float).copy()

    def hold(self) -> None:
        self.holds += 1
        self.goal = self.goal - 0.1  # Goal_Position := Present_Position

    def estop(self) -> None:
        self.hold()

    def pose(self):
        return None

    def images(self):
        return {}

    def extras(self):
        return {}

    def calibrate(self, *, interactive: bool = True):
        return None

    def close(self) -> None: ...


def _hw() -> tuple[Robot, SaggingDriver]:
    drv = SaggingDriver()
    return Robot("so101", driver=drv, twin=False), drv


def test_a_partial_action_holds_the_joints_it_does_not_name_against_gravity():
    arm = Robot("so101", sim=Simulation())
    lift0 = arm.observe().joints.positions[LIFT]
    for _ in range(200):
        obs = arm.act({"elbow_flex": 0.4})
    assert abs(obs.joints.positions[ELBOW] - 0.4) < 0.05
    assert abs(obs.joints.positions[LIFT] - lift0) < 0.05  # before: sagged 0.46 rad, never asked to move


def test_an_omitted_joint_keeps_the_last_target_written_not_the_measured_pose():
    arm, drv = _hw()
    arm.act(np.full(len(KEYS), 0.3))
    arm.act({"elbow_flex": 0.4})
    want = np.full(len(KEYS), 0.3)
    want[ELBOW] = 0.4
    np.testing.assert_allclose(drv.goal, want)  # measured is 0.2: re-targeting it is what made the sim arm sag


def test_an_empty_mapping_is_a_hold_at_the_last_targets():
    arm, drv = _hw()
    arm.act(np.full(len(KEYS), 0.3))
    for _ in range(5):
        arm.act({})
    np.testing.assert_allclose(drv.goal, np.full(len(KEYS), 0.3))


@pytest.mark.parametrize(
    "forget",
    [
        pytest.param(lambda arm: arm.hold(), id="hold"),
        pytest.param(lambda arm: (arm.estop(), arm.resume(approved_by="op", stop=arm.estop_latch.count)), id="estop"),
        pytest.param(lambda arm: arm.calibrate(interactive=False), id="calibrate"),
    ],
)
def test_after_a_hold_an_estop_or_a_calibration_an_omitted_joint_holds_the_measured_pose(forget):
    arm, drv = _hw()
    arm.act(np.full(len(KEYS), 0.3))
    forget(arm)
    measured = np.array(arm.observe().joints.positions)
    arm.act({"elbow_flex": 0.4})
    others = [i for i in range(len(KEYS)) if i != ELBOW]
    np.testing.assert_allclose(drv.goal[others], measured[others])  # never back toward the target before it


def test_after_a_reset_an_omitted_joint_holds_the_spawn_pose_not_the_old_target():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    spawn = np.array(arm.observe().joints.positions)
    arm.act(np.full(len(KEYS), 0.5))
    sim.reset()
    target = arm.coerce({"elbow_flex": 0.4})
    others = [i for i in range(len(KEYS)) if i != ELBOW]
    np.testing.assert_allclose(target[others], spawn[others], atol=1e-6)


def test_a_stop_landing_mid_write_forgets_the_target_that_write_carried():
    arm, drv = _hw()
    arm.act(np.full(len(KEYS), 0.3))
    write = drv.write

    def write_then_stop(targets) -> None:  # the stop lands while the sync-write is on the bus
        write(targets)
        arm.estop()

    drv.write = write_then_stop  # type: ignore[method-assign]
    with pytest.raises(PermissionError):
        arm.act(np.full(len(KEYS), 0.9))
    drv.write = write  # type: ignore[method-assign]
    arm.resume(approved_by="op", stop=arm.estop_latch.count)
    measured = np.array(arm.observe().joints.positions)
    arm.act({"elbow_flex": 0.4})
    others = [i for i in range(len(KEYS)) if i != ELBOW]
    np.testing.assert_allclose(drv.goal[others], measured[others])  # never the 0.9 the stopped write carried
