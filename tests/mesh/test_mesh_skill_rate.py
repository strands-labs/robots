"""The point of device skills, measured: over a real network transport (Zenoh loopback here), a policy run ON the
device keeps its own rate, while the same policy driven from the caller pays an observe and an act rpc per control
step. Over AWS IoT Core the caller-driven path measured 5.7 steps per second for a 50 Hz walk (35.2 s for a 4 s skill);
this test holds the ordering on a transport CI has."""

from __future__ import annotations

import time

import pytest

from strands_robots import Robot
from strands_robots.mesh import Mesh

pytest.importorskip("zenoh")


def test_a_device_skill_runs_far_faster_than_the_same_policy_driven_from_the_caller(port, tenant):
    policy = lambda o: {"shoulder_pan": 0.25}  # noqa: E731 - the one policy both paths run
    a = Mesh(f"zenoh://?listen=tcp/127.0.0.1:{port}&trusted=1", tenant=tenant)
    b = Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant)
    arm = Robot("so101", id="arm")
    a.mount(arm, skills={"pan": policy}, watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=5.0)

    t0 = time.monotonic()
    b.invoke("arm", "skill", name="pan", seconds=2.0)
    end = time.monotonic() + 20
    while b.invoke("arm", "skill_status")["state"] == "running" and time.monotonic() < end:
        time.sleep(0.02)
    st = b.invoke("arm", "skill_status")
    on_device = st["steps"] / (time.monotonic() - t0)

    remote = b.connect("arm")  # the caller-driven path: one observe and one act rpc per step
    remote.operator = lambda plan: True
    t0 = time.monotonic()
    res = remote.run(policy, seconds=2.0)
    from_caller = res.steps / (time.monotonic() - t0)

    assert st["state"] == "done" and res.steps > 0
    assert on_device > 3 * from_caller, (on_device, from_caller)
    print(f"steps per wall second: on the device {on_device:.0f}, driven from the caller {from_caller:.0f}")
    b.close()
    a.close()
    arm.close()
