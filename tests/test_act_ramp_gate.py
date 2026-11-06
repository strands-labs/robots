"""``confirm_act(..., seconds=)`` checks every point of its ramp, not only the target: a joint-space line between two
poses inside a Cartesian box can leave it, and an interlock can change while the arm moves."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Robot
from strands_robots.robot.safety import RequestRefused, WorkspaceBoundsValidator

pytest.importorskip("mujoco")


def test_a_ramp_that_leaves_the_workspace_between_two_allowed_poses_is_refused_mid_way():
    arm = Robot("so101")
    # x = 0.3 sin(2 q0): q0 = 0 and q0 = pi/2 are both at x = 0, the way between them reaches x = 0.3
    box = WorkspaceBoundsValidator(
        lambda q: np.array([0.3 * np.sin(2 * q[0]), 0.0, 0.2]), lo=(-0.2, -0.2, 0), hi=(0.2, 0.2, 0.4)
    )
    arm.validators.append(box)
    arm.confirm_act({"shoulder_pan": 0.0}, seconds=0.2)
    with pytest.raises(RequestRefused, match="outside the workspace"):
        arm.confirm_act({"shoulder_pan": np.pi / 2}, seconds=0.5)
    pan = arm.observe(images=False)["shoulder_pan"]
    assert 0.0 < pan < np.pi / 4 + 0.05, pan  # it stopped where the box ends, short of the target
    arm.close()


def test_an_interlock_is_asked_on_every_ramp_point_and_stops_the_move_when_it_trips():
    arm = Robot("so101")
    asked = []

    def door(robot, request):
        asked.append(float(request.targets[0]))
        if len(asked) > 5:
            raise RequestRefused("the cell door opened")

    arm.interlocks.append(door)
    with pytest.raises(RequestRefused, match="door opened"):
        arm.confirm_act({"shoulder_pan": 0.8}, seconds=1.0)
    assert len(asked) == 6  # the target once, then each ramp point until the door opened
    assert arm.observe(images=False)["shoulder_pan"] < 0.8 - 0.1, "the arm did not finish the move"
    arm.close()


def test_a_ramp_that_stays_allowed_still_reaches_its_pose():
    arm = Robot("so101")
    obs = arm.confirm_act({"shoulder_pan": 0.4}, seconds=0.5)
    assert obs["shoulder_pan"] == pytest.approx(0.4, abs=0.05)
    with pytest.raises(ValueError, match="steps= or seconds="):
        arm.confirm_act({"shoulder_pan": 0.4}, seconds=0.5, steps=3)
    arm.close()
