"""The ``control`` shadow is a write path: off by default, refused on an observe-only mount, watched by the watchdog."""

from __future__ import annotations

import json
import time
import uuid

import numpy as np

from strands_robots import JointState, Robot, Simulation
from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import IotOptions
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink

DELTA = "$aws/things/arm/shadow/name/control/update/delta"
CONTROL = "$aws/things/arm/shadow/name/control/update"


class _Bus:
    """A hardware body that records every goal write."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self, names):
        self.names, self.writes, self.q = names, [], np.zeros(len(names))

    def read(self):
        return JointState(self.names, self.q, self.q * 0, self.q * 0)

    def write(self, targets):
        self.writes.append(list(targets))
        self.q = np.asarray(targets, dtype=float)

    hold = pose = estop = close = lambda self: None  # noqa: E731
    images = extras = lambda self: {}  # noqa: E731

    def calibrate(self, *, interactive=True):
        raise RuntimeError("no calibration")


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while not pred() and time.monotonic() < end:
        time.sleep(0.01)
    return pred()


def _mesh():
    name = f"g-{uuid.uuid4().hex[:8]}"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    seen: list[dict] = []
    dash.subscribe(CONTROL, lambda t, raw: seen.append(json.loads(raw)))
    return Mesh(f"iot://fake?broker={name}", tenant="lab"), dash, seen


def _delta(dash, value=0.25):
    body = {"state": {"shoulder_pan.pos": value}, "metadata": {}, "version": 3, "timestamp": int(time.time())}
    dash.publish(DELTA, json.dumps(body).encode())


def _real_arm():
    bus = _Bus(Robot("so101").description.action_keys)
    return Robot("so101", driver=bus, confirm=lambda targets: True, twin=False), bus


def test_the_control_shadow_is_off_by_default_and_a_default_mount_never_moves_on_a_delta():
    assert IotOptions().control is False
    mesh, dash, seen = _mesh()
    arm, bus = _real_arm()
    mesh.mount(arm, device_id="arm", watchdog_ms=None, iot=IotOptions(shadow_hz=1.0))
    _delta(dash)
    time.sleep(0.5)
    assert bus.writes == []
    mesh.close()


def test_an_observe_only_hardware_mount_refuses_every_control_target_by_name():
    mesh, dash, seen = _mesh()
    arm, bus = _real_arm()
    mesh.mount(arm, device_id="arm", watchdog_ms=None, iot=IotOptions(shadow_hz=1.0, control=True))
    _delta(dash)
    assert _wait(lambda: any((s["state"].get("reported") or {}).get("refused") for s in seen))
    refused = next(s["state"]["reported"]["refused"] for s in seen if (s["state"].get("reported") or {}).get("refused"))
    assert "observe-only" in refused["shoulder_pan.pos"]
    assert bus.writes == []
    mesh.close()


def test_a_shadow_target_arms_the_watchdog_and_silence_applies_on_disconnect():
    mesh, dash, _ = _mesh()
    arm = Robot("so101", sim=Simulation())
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    mesh.mount(arm, device_id="arm", watchdog_ms=300, iot=IotOptions(shadow_hz=1.0, control=True))
    _delta(dash)
    assert _wait(lambda: holds, 3.0), "a shadow mover that goes quiet is recalled like a mesh mover"
    time.sleep(0.5)
    assert len(holds) == 1
    mesh.close()
    arm.close()
