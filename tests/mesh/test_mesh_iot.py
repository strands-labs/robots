"""The IoT transport's own promises: Direct Messaging to an offline client is a 404 and nothing is queued."""

from __future__ import annotations

import time
import uuid

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh
from strands_robots.mesh.transports import Offline
from strands_robots.mesh.transports.iot_link import FakeIotBroker


def test_estop_to_an_offline_device_is_a_404_and_is_never_replayed():
    name = f"b-{uuid.uuid4().hex[:8]}"
    url = f"iot://fake?broker={name}"
    tenant = "t-iot"
    arm = Robot("so101", sim=Simulation())
    device = Mesh(url, tenant=tenant)
    device.mount(arm, device_id="arm", watchdog_ms=None)
    controller = Mesh(url, tenant=tenant)
    assert controller.discover("arm", min_count=1, timeout=3.0)
    controller.invoke("arm", "act", action={"shoulder_pan": 0.2})

    device.transports[0].link.close()  # the device drops off the network without a goodbye
    with pytest.raises(Offline, match="404"):
        controller.invoke("arm", "estop")
    broker = FakeIotBroker.named(name)
    assert broker.direct[-1][2] == 404

    device.close()
    back = Mesh(url, tenant=tenant)  # the device comes back: nothing is waiting for it
    back.mount(arm, device_id="arm", watchdog_ms=None)
    time.sleep(0.3)
    assert not arm.estopped
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        try:
            controller.invoke("arm", "describe")
            break
        except Offline:
            time.sleep(0.05)
    controller.invoke("arm", "estop")
    assert arm.estopped
    for m in (controller, back):
        m.close()
    arm.sim.close()


def test_a_device_that_stops_beating_leaves_discover_within_the_ttl():
    name = f"b-{uuid.uuid4().hex[:8]}"
    url = f"iot://fake?broker={name}"
    device, controller = Mesh(url, tenant="t-ttl", presence_ttl=1.0), Mesh(url, tenant="t-ttl", presence_ttl=1.0)

    class Arm:
        def describe(self):
            return {"name": "arm", "device_type": "arm"}

    device.mount(Arm(), device_id="arm")
    assert controller.discover("arm", min_count=1, timeout=2.0)
    device.transports[0].link.close()
    t0 = time.monotonic()
    while controller.discover("arm") and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    left = time.monotonic() - t0
    assert controller.discover("arm") == []
    assert left <= 1.0 + 0.5
    controller.close()
    device.close()


def _lab(name):
    url = f"iot://fake?broker={name}"
    arm = Robot("so101", sim=Simulation())
    dev = Mesh(url, tenant="lab")
    dev.mount(arm, device_id="arm1", watchdog_ms=None)
    ctl = Mesh(url, tenant="lab")
    assert ctl.discover("arm1", min_count=1, timeout=3.0)
    return url, arm, dev, ctl


def test_a_forged_heartbeat_cannot_reroute_a_devices_rpcs():
    import json

    from strands_robots.mesh import wire
    from strands_robots.mesh.transports.iot import IotTransport

    name = f"b-{uuid.uuid4().hex[:8]}"
    url, arm, dev, ctl = _lab(name)
    evil = IotTransport(f"{url}&client=evil")
    key = wire.key("presence", "lab", "arm1")
    evil.link.publish(key, json.dumps({"alive": True, "client": "evil"}).encode())
    time.sleep(0.2)
    ctl.invoke("arm1", "estop", timeout=2.0)
    assert arm.estopped, "the e-stop reached the real device, not the forger"
    evil.link.publish(key, json.dumps({"alive": False, "client": "evil"}).encode())
    time.sleep(0.2)
    assert [d.id for d in ctl.discover()] == ["arm1"], "a goodbye from another client is ignored"
    evil.close()
    for m in (ctl, dev):
        m.close()
    arm.sim.close()


def test_a_targeted_estop_also_goes_out_on_the_tenant_estop_key():
    name = f"b-{uuid.uuid4().hex[:8]}"
    _, arm, dev, ctl = _lab(name)
    ctl.transports[0].query = lambda key, payload, timeout: []  # the targeted path is lost on the way
    with pytest.raises(TimeoutError):
        ctl.invoke("arm1", "estop", timeout=0.5)
    end = time.monotonic() + 2.0
    while not arm.estopped and time.monotonic() < end:
        time.sleep(0.01)
    assert arm.estopped
    for m in (ctl, dev):
        m.close()
    arm.sim.close()


def test_a_stale_boto3_without_send_direct_message_fails_with_a_clear_floor_hint(monkeypatch):
    """A boto3 older than 1.43.17 has no iot-data SendDirectMessage; the transport must say so, not AttributeError."""
    import boto3

    from strands_robots.mesh.transports.iot_link import AwsLink

    class _StaleIotData:  # an iot-data client from boto3 < 1.43.17: no send_direct_message
        pass

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _StaleIotData())
    link = object.__new__(AwsLink)  # skip the live MQTT connect in __init__
    link._boto = None
    link.region, link.endpoint = "us-east-1", "example-ats.iot.us-east-1.amazonaws.com"
    with pytest.raises(ImportError, match="1.43.17"):
        link.send_direct("c", "strands/x", b"{}", timeout=5.0)
