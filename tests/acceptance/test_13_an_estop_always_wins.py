"""Check 13: an e-stop always wins: one that lands between a command's checks and its write stops the write.

The interleaving is made exact, not raced: ``estop()`` is called from inside the command, after the
action was coerced and before the body is written. No target may reach the body with the latch engaged.
"""

from __future__ import annotations

import pytest
from checks import needs_module

pytest.importorskip("mujoco")


def test_an_estop_between_check_and_write_stops_the_write():
    needs_module("strands_robots.robot.safety", "part 5.5: a Robot is a device")
    from strands_robots import Robot

    arm = Robot("so101")
    latched_writes: list[bool] = []
    write, coerce = arm.body.write, arm.coerce

    def spy_write(targets):
        latched_writes.append(bool(arm.estopped))
        return write(targets)

    def coerce_then_estop(action, **kw):
        out = coerce(action, **kw)
        arm.estop()
        return out

    arm.body.write, arm.coerce = spy_write, coerce_then_estop
    try:
        arm.act({"shoulder_pan": 0.5})
    except PermissionError:  # RequestRefused / EStopped are refusals; refusing is the right outcome
        pass
    assert True not in latched_writes, latched_writes
