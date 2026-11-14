"""The ``Move`` a device's operator reads names who sent it: a mesh peer's act says ``mesh``, never ``agent``."""

from __future__ import annotations

import pytest

from strands_robots import Robot
from strands_robots.robot.run import Move
from strands_robots.robot.safety import RequestRefused


def test_a_peers_act_asks_the_devices_operator_with_a_mesh_move(mesh_pair):
    a, b = mesh_pair
    asked: list[Move] = []
    arm = Robot("so101", id="arm", driver="memory", twin=False, operator=lambda r: asked.append(r) or False)
    a.mount(arm, watchdog_ms=2000)
    assert b.discover("arm", min_count=1, timeout=3.0)
    with pytest.raises(RequestRefused, match="the operator gate refused move arm from mesh"):
        b.invoke("arm", "act", action={"shoulder_pan": 0.2})
    assert [m.source for m in asked] == ["mesh"] and arm.body.writes == []
    a.close()
    arm.close()
