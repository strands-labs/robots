"""mesh.mount(sim) mounts a whole world; Robot(..., mesh=) mounts at construction."""

from __future__ import annotations

import threading

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh


def test_mount_a_whole_world_in_one_call():
    sim = Simulation()
    Robot("so101", sim=sim, id="left")
    Robot("so101", sim=sim, id="right")
    with Mesh("mem://world", tenant="t-world") as m:
        devices = m.mount(sim, labels={"bench": "a"}, watchdog_ms=None)
        assert sorted(d.id for d in devices) == ["left", "right"]
        found = m.discover("device(bench:a)", min_count=2, timeout=2.0)
        assert sorted(d.id for d in found) == ["left", "right"]
        m.invoke("left", "act", action={"shoulder_pan": 0.3})
        assert sim.robots["left"].observe().joints["shoulder_pan"] > 0.01
    sim.close()


def test_robot_mesh_is_sugar_for_mount():
    sim = Simulation()
    with Mesh("mem://sugar", tenant="t-sugar") as m:
        arm = Robot("so101", sim=sim, id="arm", mesh=m)
        assert [d.id for d in m.discover("arm", min_count=1, timeout=2.0)] == ["arm"]
        assert m.invoke("arm", "describe")["device_id"] == arm.id
    sim.close()


@pytest.mark.parametrize(
    "use",
    [
        lambda m, arm: m.mount(arm, device_id="late"),
        lambda m, arm: m.connect("late"),
        lambda m, arm: m.invoke("late", "describe"),
        lambda m, arm: m.broadcast("hold"),
        lambda m, arm: m.subscribe("late", "state", print),
    ],
    ids=["mount", "connect", "invoke", "broadcast", "subscribe"],
)
def test_a_closed_mesh_refuses_to_be_used_and_starts_no_thread(use):
    sim = Simulation()
    arm = Robot("so101", sim=sim, id="late")
    m = Mesh("mem://closed", tenant="t-closed")
    m.close()
    before = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="closed"):
        use(m, arm)
    assert set(threading.enumerate()) - before == set()
    m.close()  # still idempotent
    sim.close()
