"""The watchdog over the mesh: an invoker that moved a device and then vanishes or goes quiet triggers on_disconnect."""

from __future__ import annotations

import time

import numpy as np
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


def test_a_motion_rpc_without_a_peer_is_refused_where_a_watchdog_guards(tenant):
    from strands_robots.mesh import Mesh, wire
    from strands_robots.mesh.transports import open_transport

    url = f"mem://a4-{tenant}"
    dev, arm = Mesh(url, tenant=tenant), Robot("so101")
    dev.mount(arm, device_id="arm", watchdog_ms=200)
    raw = open_transport(url)
    env = wire.envelope(tenant, "arm", "rpc", "act", seq=1, args={"action": {"shoulder_pan": 0.3}})
    reply = wire.decode_envelope(raw.query(wire.key("rpc", tenant, "arm", "act"), wire.encode(env), 1.0)[0])
    assert reply["kind"] == "error" and "peer id" in reply["error"]["message"]
    assert abs(float(arm.observe().joints["shoulder_pan"])) < 0.05, "nothing moved"
    raw.close()
    dev.close()
    arm.close()


class _LimpArm:
    """A hardware Body (speaks real JointState frames); counts holds and goal writes."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self, names):
        self.names, self.holds = names, 0
        self.q = np.zeros(len(names))

    def read(self):
        from strands_robots import JointState

        return JointState(self.names, self.q, self.q * 0, self.q * 0)

    def write(self, targets):
        self.q = np.asarray(targets, dtype=float)

    def hold(self):
        self.holds += 1

    pose = lambda self: None  # noqa: E731
    images = extras = lambda self: {}  # noqa: E731
    estop = close = lambda self: None  # noqa: E731

    def calibrate(self, *, interactive=True):
        raise RuntimeError("no calibration")


def _real_arm():
    names = Robot("so101").description.action_keys
    bus = _LimpArm(names)
    # the operator approves every write: only the missing watchdog may refuse it
    return Robot("so101", driver=bus, confirm=lambda targets: True, twin=False), bus


def test_a_hardware_device_without_a_watchdog_is_served_observe_only(mesh_pair):
    from strands_robots.mesh import RequestRefused

    a, b = mesh_pair
    real, bus = _real_arm()
    a.mount(real, device_id="arm", watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=3.0)
    # reads still work: it is a device, just observe-only
    assert list(b.invoke("arm", "observe").joints.names) == list(real.description.action_keys)
    assert b.invoke("arm", "describe")["name"] == "so101"
    # every motion verb is refused by name, and the bus never moved
    for rpc, args in (("act", {"action": {"shoulder_pan": 0.3}}), ("hold", {})):
        with pytest.raises(RequestRefused, match="observe-only"):
            b.invoke("arm", rpc, **args)
    assert bus.holds == 0 and float(real.observe().joints["shoulder_pan"]) == 0.0
    a.close()
    real.close()


def test_a_simulated_device_without_a_watchdog_still_moves(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")  # a simulated body: harmless without a watchdog, unchanged
    a.mount(arm, device_id="sim-arm", watchdog_ms=None)
    assert b.discover("sim-arm", min_count=1, timeout=3.0)
    b.invoke("sim-arm", "act", action={"shoulder_pan": 0.2}, steps=100)
    assert float(arm.observe().joints["shoulder_pan"]) > 0.1  # act served, the joint moved toward the target
    a.close()
    arm.close()


def test_a_reach_in_progress_is_not_a_quiet_invoker(device):
    arm, holds, _, b = device
    real = arm.confirm_act
    arm.confirm_act = lambda *a, **k: (time.sleep(0.6), real(*a, **k))[1]  # a reach longer than watchdog_ms=300
    t0 = time.monotonic()
    b.invoke("arm", "act", action={"shoulder_pan": 0.2}, seconds=0.6)
    t1 = time.monotonic()
    assert not holds and t1 - t0 >= 0.6
    assert _wait(lambda: holds, 2.0)
    assert holds[0] - t1 >= 0.25  # the quiet clock started when the motion ended
