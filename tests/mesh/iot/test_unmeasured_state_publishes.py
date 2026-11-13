"""A default Feetech arm reads position only, so its velocity and effort are unmeasured (NaN in the driver).

The IoT ``state`` shadow and Basic Ingest telemetry encode strict JSON; before ``state_payload`` sent unmeasured
values as ``None`` both raised in the sampling thread and published nothing for such an arm (one ``fault``, no state).
"""

from __future__ import annotations

import json
import time
import uuid

from strands_robots import Robot
from strands_robots.mesh import wire
from strands_robots.mesh.iot import ShadowMirror, rules, shadow_topic
from strands_robots.mesh.record import record
from strands_robots.mesh.serve import ARGS
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink
from tests.test_drivers import FakeSerial, make_driver


def _feetech_arm() -> Robot:
    return Robot("so101", driver=make_driver(serial=FakeSerial())[0], twin=False)


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while not pred() and time.monotonic() < end:
        time.sleep(0.01)
    assert pred()


def test_the_state_shadow_publishes_for_a_position_only_feetech_arm():
    broker = FakeIotBroker.named(f"un-{uuid.uuid4().hex[:8]}")
    arm = _feetech_arm()
    faults: list[dict] = []
    arm.subscribe("fault", faults.append)
    device, operator = FakeLink(broker, "arm"), FakeLink(broker, "op")
    seen: list[dict] = []
    operator.subscribe(shadow_topic("arm", "state", "update"), lambda t, raw: seen.append(json.loads(raw)))
    mirror = ShadowMirror(device, arm, record(arm, "arm", {"role": "arm"}, "lab", tuple(ARGS)), shadow_hz=1.0)
    mirror.start()
    try:
        _wait(lambda: mirror.updates["state"] >= 1)
    finally:
        mirror.stop()
        device.close()
        operator.close()
        arm.close()
    doc = seen[-1]["state"]["reported"]
    assert doc["shoulder_pan.vel"] is None and doc["shoulder_pan.effort"] is None
    assert isinstance(doc["shoulder_pan.pos"], float)
    assert faults == []


def test_telemetry_publishes_for_a_position_only_feetech_arm():
    broker = FakeIotBroker.named(f"ut-{uuid.uuid4().hex[:8]}")
    arm = _feetech_arm()
    faults: list[dict] = []
    arm.subscribe("fault", faults.append)
    device, sink = FakeLink(broker, "arm"), FakeLink(broker, "sink")
    got: list[dict] = []
    sink.subscribe("$aws/rules/+/strands/lab/arm/state", lambda t, raw: got.append(wire.decode_envelope(raw)))
    undo = rules.telemetry(device, "lab", "arm", arm, 1.0)
    try:
        _wait(lambda: got)
    finally:
        undo()
        device.close()
        sink.close()
        arm.close()
    assert "shoulder_pan.pos" in got[-1]["result"]
    assert faults == []
