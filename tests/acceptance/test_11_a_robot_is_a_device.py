"""Check 11: a Robot is a device: one manifest, a safety chain that refuses by name, an estop that latches.

Graded on a sim SO-101: the manifest round-trips and declares one control per joint; a write outside a
control's range is refused naming the joint; ``estop()`` from another thread stops a running policy with
``stopped_by="estop"``; the latch holds until ``resume(approved_by=)``.
"""

from __future__ import annotations

import threading

import pytest
from checks import needs_module

pytest.importorskip("mujoco")


def test_a_robot_is_a_device():
    needs_module("strands_robots.robot.device", "part 5.5: a Robot is a device")
    from strands_robots import Robot
    from strands_robots.robot.device import Manifest
    from strands_robots.robot.safety import RequestRefused

    arm = Robot("so101")
    m = arm.manifest
    assert Manifest.from_dict(m.to_dict()) == m
    d = arm.describe()  # the manifest, plus the device's live status
    assert {k: d[k] for k in m.to_dict()} == m.to_dict()
    assert sorted(c.name for c in m.controls) == sorted(f"{k}.pos" for k in arm.action_keys)

    lo, hi = arm.description.limits["shoulder_pan"]
    with pytest.raises(RequestRefused, match="shoulder_pan"):
        arm.confirm_act({"shoulder_pan": hi + 1.0})

    timer = threading.Timer(0.3, arm.estop)
    timer.start()
    r = arm.run(lambda o: {"shoulder_pan": 0.5}, hz=30, seconds=30, realtime=True)
    timer.join()
    assert r.stopped_by == "estop" and r.steps <= 0.3 * 30 + 3, r  # within a tick of the estop

    with pytest.raises(RequestRefused):
        arm.act({"shoulder_pan": 0.0})  # latched
    with pytest.raises((RequestRefused, TypeError, ValueError)):
        arm.resume(approved_by=None)
    arm.resume(approved_by="operator")
    arm.act({"shoulder_pan": 0.0})
