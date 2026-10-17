"""The watchdog over the mesh: an invoker that moved a device and then vanishes or goes quiet triggers on_disconnect."""

from __future__ import annotations

import time

import pytest

from strands_robots import Robot


@pytest.fixture
def device(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, device_id="arm", watchdog_ms=300)
    assert b.discover("arm", min_count=1, timeout=3.0)
    yield arm, holds, a, b
    a.close()
    arm.close()


def _wait(pred, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.005)
    return pred()


def test_a_quiet_invoker_after_motion_triggers_on_disconnect_once(device):
    _, holds, _, b = device
    b.invoke("arm", "act", action={"shoulder_pan": 0.2})
    t0 = time.monotonic()
    assert _wait(lambda: holds, 2.0)
    assert 0.25 <= holds[0] - t0 <= 0.3 + 0.2
    time.sleep(0.5)
    assert len(holds) == 1, "on_disconnect applies once per lapse"


def test_steady_commands_keep_the_watchdog_quiet(device):
    _, holds, _, b = device
    end = time.monotonic() + 0.8
    while time.monotonic() < end:
        b.invoke("arm", "act", action={"shoulder_pan": 0.1})
        time.sleep(0.03)
    assert holds == []


def test_reads_do_not_arm_the_watchdog(device):
    _, holds, _, b = device
    b.invoke("arm", "observe")
    b.invoke("arm", "describe")
    time.sleep(0.6)
    assert holds == []


def test_an_invoker_that_leaves_triggers_on_disconnect_before_the_timeout(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, device_id="arm", watchdog_ms=5000)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "act", action={"shoulder_pan": 0.2})
    t0 = time.monotonic()
    b.close()
    assert _wait(lambda: holds, 3.0), "the invoker's presence vanished, so the device holds"
    assert holds[0] - t0 < 1.0
    a.close()
    arm.close()


def test_watchdog_off(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    holds: list[int] = []
    arm.body.hold = lambda: holds.append(1)
    a.mount(arm, device_id="arm", watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.invoke("arm", "act", action={"shoulder_pan": 0.2})
    time.sleep(0.6)
    assert holds == []
    a.close()
    arm.close()
