"""Which path an rpc takes (MAPPING 4.1, 4.2, 4.5): Direct Message for operators, MQTT request/response for
certificate-only devices; the device answers only on its tenant's reply keys."""

from __future__ import annotations

import json
import time
import uuid

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import direct
from strands_robots.mesh.transports.iot import IotTransport
from strands_robots.mesh.transports.iot_link import FakeIotBroker


def test_mode_defaults_follow_the_credentials_and_the_url_wins():
    assert direct.rpc_mode({}) == "direct"  # IAM operator: the 404 in one round trip
    assert (
        direct.rpc_mode({"cert": "c.pem", "key": "k.pem"}) == "mqtt"
    )  # a certificate cannot call HTTPS SendDirectMessage
    assert direct.rpc_mode({"cert": "c.pem", "key": "k.pem", "rpc": "direct"}) == "direct"
    with pytest.raises(ValueError, match="rpc mode"):
        direct.rpc_mode({"rpc": "smoke"})
    assert direct.shared("dash", "strands/lab/+/state") == "$share/dash/strands/lab/+/state"


def test_reply_topic_must_be_the_tenants_reply_key():
    assert direct.reply_ok("strands/lab/@reply/op-1", "lab")
    assert not direct.reply_ok("strands/other/@reply/op-1", "lab")
    assert not direct.reply_ok("strands/lab/arm/state", "lab")
    assert not direct.reply_ok("strands/lab/@reply/", "lab")


def test_request_response_mode_runs_the_whole_mesh_and_an_offline_device_is_silence_not_404():
    name = f"rr-{uuid.uuid4().hex[:8]}"
    url = f"iot://fake?broker={name}"
    arm = Robot("so101", sim=Simulation())
    device = Mesh(url, tenant="lab")
    device.mount(arm, device_id="arm", watchdog_ms=None)
    ctl = Mesh(f"{url}&rpc=mqtt", tenant="lab")
    assert ctl.transports[0].rpc == "mqtt" and device.transports[0].rpc == "direct"
    assert ctl.discover("arm", min_count=1, timeout=3.0)
    t0 = time.perf_counter()
    ctl.invoke("arm", "act", action={"shoulder_pan": 0.2})
    rtt_ms = (time.perf_counter() - t0) * 1000
    assert ctl.invoke("arm", "describe")["name"] == "so101"
    assert not [d for d in FakeIotBroker.named(name).direct if "/rpc/" in d[1]], "no rpc went as a Direct Message"
    ctl.invoke("arm", "estop")
    assert arm.estopped
    device.transports[0].link.close()  # gone without a goodbye
    with pytest.raises(TimeoutError):  # silence, by design: presence (3 s heartbeat) is the offline signal
        ctl.invoke("arm", "describe", timeout=0.5)
    print(f"\nfake-broker rr act rtt {rtt_ms:.1f} ms")
    ctl.close()
    device.close()
    arm.sim.close()


def test_a_request_with_a_foreign_reply_address_is_never_answered():
    name = f"rr-{uuid.uuid4().hex[:8]}"
    url = f"iot://fake?broker={name}"
    arm = Robot("so101", sim=Simulation())
    device = Mesh(url, tenant="lab")
    device.mount(arm, device_id="arm", watchdog_ms=None)
    evil = IotTransport(f"{url}&client=evil")
    heard: list = []
    evil.link.subscribe("strands/other/@reply/evil", lambda t, raw: heard.append(raw))
    evil.link.subscribe("strands/lab/@reply/evil", lambda t, raw: heard.append(raw))
    evil.link.subscribe("strands/lab/arm/state", lambda t, raw: heard.append(raw) if b'"id"' in raw else None)
    from strands_robots.mesh import wire

    env = wire.encode(wire.envelope("lab", "arm", "rpc", "describe", seq=1, args={}))
    for reply in ("strands/other/@reply/evil", "strands/lab/arm/state", "evil"):
        body = json.dumps({"id": "x", "reply": reply, "body": env.decode()}).encode()
        evil.link.publish(wire.key("rpc", "lab", "arm", "describe"), body)
    good = json.dumps({"id": "y", "reply": "strands/lab/@reply/evil", "body": env.decode()}).encode()
    evil.link.publish(
        wire.key("rpc", "lab", "arm", "describe"), good
    )  # sent last: its answer proves the others were seen
    end = time.monotonic() + 5.0
    while not heard and time.monotonic() < end:
        time.sleep(0.02)
    time.sleep(0.2)
    assert [json.loads(h)["id"] for h in heard] == ["y"], "only the request with a tenant reply key was answered"
    evil.close()
    device.close()
    arm.sim.close()
