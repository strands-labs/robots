"""``Mesh.mount(robot, iot=IotOptions(...))``: shadows appear on the IoT transport, nothing else changes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import IotOptions
from strands_robots.mesh.transports.iot_link import FakeIotBroker, FakeLink


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while not pred() and time.monotonic() < end:
        time.sleep(0.01)
    assert pred()


def test_mount_with_iot_options_mirrors_shadows_and_a_shadow_target_moves_the_arm_through_confirm_act():
    name = f"m-{uuid.uuid4().hex[:8]}"
    url, tenant = f"iot://fake?broker={name}", "lab"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    seen: dict[str, list] = {}
    dash.subscribe(
        "$aws/things/arm/shadow/name/+/update", lambda t, raw: seen.setdefault(t, []).append(json.loads(raw))
    )
    arm = Robot("so101", sim=Simulation())
    device = Mesh(url, tenant=tenant)
    dev = device.mount(arm, device_id="arm", watchdog_ms=None, iot=IotOptions(shadow_hz=1.0))
    assert dev.id == "arm"
    manifest = "$aws/things/arm/shadow/name/manifest/update"
    _wait(lambda: manifest in seen)
    assert seen[manifest][0]["state"]["reported"]["device_id"] == "arm"  # the compact document

    controller = Mesh(url, tenant=tenant)  # the mesh itself is unchanged: discover + invoke still work
    assert controller.discover("arm", min_count=1, timeout=3.0)
    controller.invoke("arm", "act", action={"shoulder_pan": 0.1})

    calls = []
    real = arm.confirm_act
    arm.confirm_act = lambda action, **kw: (calls.append(action), real(action, **kw))[1]
    delta = {"state": {"shoulder_pan.pos": 0.25}, "metadata": {}, "version": 3, "timestamp": int(time.time())}
    dash.publish("$aws/things/arm/shadow/name/control/update/delta", json.dumps(delta).encode())
    _wait(lambda: calls)
    assert calls == [{"shoulder_pan": 0.25}]
    _wait(lambda: "$aws/things/arm/shadow/name/state/update" in seen)
    assert seen["$aws/things/arm/shadow/name/state/update"][-1]["state"]["reported"]["lockout"]["engaged"] is False

    device.unmount("arm")
    time.sleep(0.2)
    n = len(seen.get("$aws/things/arm/shadow/name/state/update", []))
    time.sleep(1.2)
    assert len(seen["$aws/things/arm/shadow/name/state/update"]) == n  # reporting stopped with the device
    controller.close()
    device.close()
    dash.close()
    arm.sim.close()


def test_iot_options_without_an_iot_transport_is_refused_and_a_plain_mount_is_untouched():
    arm = Robot("so101", sim=Simulation())
    mesh = Mesh(f"mem://{uuid.uuid4().hex[:8]}", tenant="lab")
    with pytest.raises(ValueError, match="no transport of this Mesh is iot://"):
        mesh.mount(arm, device_id="arm", watchdog_ms=None, iot={"shadows": True})
    assert mesh.mount(arm, device_id="arm", watchdog_ms=None).id == "arm"
    mesh.close()
    arm.sim.close()


def test_importing_the_mesh_loads_no_aws_sdk():
    code = "import json, sys, strands_robots.mesh.iot; print(json.dumps(list(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert not {"boto3", "botocore", "awsiot", "awscrt", "cryptography"} & {m.split(".")[0] for m in json.loads(out)}


def test_the_state_shadow_reports_a_local_estop_without_waiting_for_an_rpc():
    """``robot.estop()`` from a button or a fault handler latches locally; the ``state`` shadow is the surface an
    operator who is not on the mesh reads, so it must say ``engaged: true`` by itself, as the ``lockout`` rpc does."""
    name = f"m-{uuid.uuid4().hex[:8]}"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    states: list[dict] = []
    dash.subscribe("$aws/things/arm/shadow/name/state/update", lambda t, raw: states.append(json.loads(raw)))
    arm = Robot("so101", sim=Simulation())
    device = Mesh(f"iot://fake?broker={name}", tenant="lab")
    device.mount(arm, device_id="arm", watchdog_ms=None, iot=IotOptions(shadow_hz=1.0))
    try:
        _wait(lambda: states)
        assert states[-1]["state"]["reported"]["lockout"]["engaged"] is False
        arm.estop()  # local: no rpc, no broadcast
        n = len(states)
        _wait(lambda: len(states) > n, timeout=3.0)
        lock = states[-1]["state"]["reported"]["lockout"]
        assert lock["engaged"] is True and lock["epoch"], lock
        arm.resume(approved_by="alice")
        n = len(states)
        _wait(lambda: len(states) > n, timeout=3.0)
        assert states[-1]["state"]["reported"]["lockout"]["engaged"] is False
    finally:
        device.close()
        dash.close()
        arm.sim.close()


def _shadow_threads():
    import threading

    return [t for t in threading.enumerate() if t.name.startswith("shadow-") and t.is_alive()]


def test_a_bad_iot_option_refuses_before_anything_starts_and_a_refused_mount_leaves_nothing_behind(monkeypatch):
    """Yin on #4784: telemetry_hz was validated by telemetry() AFTER the mirror had written manifest, started the
    state stream and subscribed to control deltas; the mount raised, the device was never in _mounted, and the
    orphan mirror kept applying deltas with no handle to stop it (STEER K2: validate, then mutate; release on every
    exit)."""
    name = f"m-{uuid.uuid4().hex[:8]}"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    seen: dict[str, list] = {}
    dash.subscribe(
        "$aws/things/arm/shadow/name/+/update", lambda t, raw: seen.setdefault(t, []).append(json.loads(raw))
    )
    arm = Robot("so101", sim=Simulation())
    mesh = Mesh(f"iot://fake?broker={name}", tenant="lab")
    live_before = len(_shadow_threads())
    with pytest.raises(ValueError, match="telemetry_hz"):
        IotOptions(telemetry_hz=2.0)  # refused at construction
    with pytest.raises(ValueError, match="telemetry_hz"):
        mesh.mount(arm, device_id="arm", watchdog_ms=None, iot={"telemetry_hz": 2.0})
    assert list(mesh._mounted) == [] and len(_shadow_threads()) == live_before
    assert "$aws/things/arm/shadow/name/manifest/update" not in seen, "nothing started before the refusal"
    for bad in ({"shadow_hz": 2.0}, {"stale_s": 0.1}):
        with pytest.raises(ValueError):
            IotOptions(**bad)

    # A failure AFTER the mirror started (here: the telemetry rule) undoes the mirror before the error travels.
    import strands_robots.mesh.iot.rules as rules

    def boom(*a, **kw):
        raise RuntimeError("no telemetry rule")

    monkeypatch.setattr(rules, "telemetry", boom)
    with pytest.raises(RuntimeError, match="no telemetry rule"):
        mesh.mount(arm, device_id="arm", watchdog_ms=None, iot={"shadows": True, "telemetry_hz": 0.5})
    time.sleep(0.3)
    assert list(mesh._mounted) == [] and len(_shadow_threads()) == live_before, "the started mirror was stopped"
    calls = []
    real = arm.confirm_act
    arm.confirm_act = lambda action, **kw: (calls.append(action), real(action, **kw))[1]
    delta = json.dumps({"state": {"shoulder_pan.pos": 0.2}}).encode()
    dash.publish("$aws/things/arm/shadow/name/control/update/delta", delta)
    time.sleep(0.5)
    assert calls == [], "a delta to the robot the API said is not mounted reaches nothing"
    mesh.close()
    dash.close()
    arm.sim.close()


@pytest.mark.parametrize("step", ["watchdog", "serve", "announce"])
def test_a_mount_that_fails_after_the_mirror_started_stops_it(monkeypatch, step):
    """Yin on #4784: a failure in mount() after _iot_attach (a bad watchdog_ms, serve or announce raising) left the
    mirror applying control deltas to a robot the API said is not mounted; mount now undoes what it started."""
    import strands_robots.mesh.mesh as mesh_mod

    name = f"m-{uuid.uuid4().hex[:8]}"
    broker = FakeIotBroker.named(name)
    dash = FakeLink(broker, "dashboard")
    arm = Robot("so101", sim=Simulation())
    mesh = Mesh(f"iot://fake?broker={name}", tenant="lab")
    import threading

    def dogs():
        return sum(t.name == "mesh-watchdog" and t.is_alive() for t in threading.enumerate())

    live_before, dogs_before = len(_shadow_threads()), dogs()
    kw: dict = {"watchdog_ms": 500.0}  # serve/announce fail after the watchdog started: it stops too
    if step == "watchdog":
        kw["watchdog_ms"] = 0
    else:

        def boom(*a, **k):
            raise RuntimeError(f"{step} failed")

        monkeypatch.setattr(mesh_mod, step, boom)
    with pytest.raises((ValueError, RuntimeError)):
        mesh.mount(arm, device_id="arm", iot=IotOptions(shadow_hz=1.0), **kw)
    time.sleep(0.3)
    assert list(mesh._mounted) == [] and len(_shadow_threads()) == live_before, "the started mirror was stopped"
    assert dogs() == dogs_before, "the started watchdog was stopped"
    calls = []
    arm.confirm_act = lambda action, **k: calls.append(action)
    dash.publish(
        "$aws/things/arm/shadow/name/control/update/delta", json.dumps({"state": {"shoulder_pan.pos": 0.25}}).encode()
    )
    time.sleep(0.5)
    assert calls == [], "a delta to the robot the API said is not mounted reaches nothing"
    mesh.close()
    dash.close()
    arm.sim.close()
