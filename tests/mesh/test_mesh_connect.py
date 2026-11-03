"""connect(id): a Robot whose Body lives on another peer; observe/act/run work unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh, RemoteBody


def policy(obs):
    """Deterministic, state-dependent: the trajectory differs if any observation differs."""
    q = obs.joints.positions
    return {"shoulder_pan": 0.5 * np.sin(q[1] + 0.3), "shoulder_lift": 0.4 - 0.5 * q[0], "elbow_flex": 0.3}


def test_remote_run_matches_the_local_trajectory(mesh_pair):
    a, b = mesh_pair
    device = Robot("so101", sim=Simulation())
    a.mount(device, device_id="arm", watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=3.0)
    remote = b.connect("arm")
    assert isinstance(remote.body, RemoteBody) and remote.id == "arm"
    assert remote.supports_position_hold is bool(device.manifest.position_hold)

    local = Robot("so101", sim=Simulation())
    local_traj, remote_traj = [], []
    local.run(lambda o: (local_traj.append(o.joints.positions.copy()), policy(o))[1], steps=200, hz=50)
    remote.run(lambda o: (remote_traj.append(o.joints.positions.copy()), policy(o))[1], steps=200, hz=50)
    assert len(local_traj) == len(remote_traj) == 200
    np.testing.assert_allclose(np.array(remote_traj), np.array(local_traj), atol=1e-9)
    np.testing.assert_allclose(remote.observe().joints.positions, local.observe().joints.positions, atol=1e-9)
    a.close()
    for r in (device, local):
        r.sim.close()


def test_a_remote_refusal_reaches_the_remote_robot_verbatim(mesh_pair):
    a, b = mesh_pair
    device = Robot("so101", sim=Simulation())
    a.mount(device, device_id="arm", watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=3.0)
    remote = b.connect("arm")
    from strands_robots.mesh import RequestRefused

    remote.act({"shoulder_pan": 99.0})  # developer code: clipped to the limits locally, as in-process
    assert remote.observe().joints["shoulder_pan"] == pytest.approx(device.observe().joints["shoulder_pan"])
    device.estop()
    with pytest.raises(RequestRefused, match="e-stop is engaged") as remote_err:
        remote.act({"shoulder_pan": 0.1})
    with pytest.raises(RequestRefused) as local_err:
        device.act({"shoulder_pan": 0.1})
    assert str(remote_err.value) == str(local_err.value)
    a.close()
    device.sim.close()


def test_a_remote_run_stops_with_estop_when_the_device_latches(mesh_pair):
    import threading
    import time

    a, b = mesh_pair
    device = Robot("so101", sim=Simulation())
    a.mount(device, device_id="arm", watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=3.0)
    remote = b.connect("arm")
    threading.Timer(0.3, lambda: b.broadcast("estop")).start()
    t0 = time.monotonic()
    result = remote.run(lambda obs: {"shoulder_pan": 0.2}, hz=50, seconds=5)
    assert result.stopped_by == "estop" and time.monotonic() - t0 < 2.0
    a.close()
    device.sim.close()


def test_connect_to_an_absent_device_is_a_lookup_error():
    with Mesh("mem://connect-absent", tenant="t-absent") as m:
        with pytest.raises(LookupError, match="nobody"):
            m.connect("nobody")


def test_the_record_is_what_the_mesh_says_and_leaks_no_host_paths(mesh_pair):
    a, b = mesh_pair
    device = Robot("so101", sim=Simulation())
    a.mount(device, device_id="arm", labels={"role": "arm"})
    found = b.discover("arm", min_count=1, timeout=3.0)[0]
    m = found.manifest
    assert m["device_id"] == "arm" and m["tenant"] == a.tenant and m["labels"] == {"role": "arm"}
    assert "model" not in m and "model" not in m["description"]
    from strands_robots.mesh.serve import ARGS

    assert {r["name"] for r in m["rpcs"]} <= set(ARGS)
    assert len(repr(found)) < 200
    a.close()
    device.sim.close()


def test_a_connected_robot_trusts_the_registry_not_a_peers_manifest(tenant):
    """A14: a peer that announces a lying manifest cannot widen a device's limits or reorder its keys. The
    operator's Robot takes its limits and action_keys from the local registry by the announced type's name; a
    reordered-keys manifest is refused outright."""
    from strands_robots.description import Description
    from strands_robots.mesh.presence import announce
    from strands_robots.mesh.record import record

    sim = Simulation()
    arm = Robot("so101", sim=sim)
    honest = record(arm, "arm", {}, tenant, ("observe", "act", "hold", "estop"))

    # a manifest that lies about the limits, keys in their real order: connect ignores the manifest's limits
    lie = {**honest, "manifest": {**honest["manifest"]}}
    lie["manifest"]["description"] = {**honest["manifest"]["description"], "limits": {"shoulder_lift": [-3.32, 0.17]}}
    lie["manifest"]["limits"] = {"shoulder_lift": [-3.32, 0.17]}
    bus = f"mem://a14-{tenant}"
    liar = Mesh(bus, tenant=tenant)
    undo = announce(liar.transports[0], tenant, lie, "liar")
    operator = Mesh(bus, tenant=tenant)
    try:
        assert operator.discover("arm", min_count=1, timeout=2.0)
        remote = operator.connect("arm")
        canonical = Description.load("so101").limits["shoulder_lift"]
        assert remote.description.limits["shoulder_lift"] == canonical
        assert remote.description.limits["shoulder_lift"] != (-3.32, 0.17)
    finally:
        undo()
        operator.close()
        liar.close()

    # a manifest that reorders the device's action_keys cannot slip past the body-to-type check
    keys = list(honest["manifest"]["action_keys"])
    reordered = {**honest, "manifest": {**honest["manifest"], "action_keys": keys[::-1]}}
    bus2 = f"mem://a14b-{tenant}"
    liar2 = Mesh(bus2, tenant=tenant)
    undo2 = announce(liar2.transports[0], tenant, reordered, "liar")
    op2 = Mesh(bus2, tenant=tenant)
    try:
        assert op2.discover("arm", min_count=1, timeout=2.0)
        with pytest.raises(ValueError, match="action_keys"):
            op2.connect("arm")
    finally:
        undo2()
        op2.close()
        liar2.close()
        sim.close()


@pytest.mark.parametrize(
    ("robot_type", "manifest_claim", "expected"),
    [
        ("koch", True, False),  # registry leaves position_hold undeclared: a True claim cannot grant it
        ("so101", False, True),  # registry declares it: a False claim cannot take it away
    ],
)
def test_the_registry_not_a_manifest_decides_position_hold(tenant, robot_type, manifest_claim, expected):
    """A14 remainder: ``connect`` trusts the registry's ``hardware.position_hold`` for the announced type, never the
    peer's manifest. An undeclared type whose manifest claims a hold does not get one; a declared type keeps its
    registry answer whatever the manifest says."""
    from strands_robots.mesh.presence import announce
    from strands_robots.mesh.record import record

    sim = Simulation()
    arm = Robot(robot_type, sim=sim)
    honest = record(arm, "arm", {}, tenant, ("observe", "act", "hold", "estop"))
    lie = {**honest, "manifest": {**honest["manifest"], "position_hold": manifest_claim}}
    bus = f"mem://a14hold-{robot_type}-{tenant}"
    liar = Mesh(bus, tenant=tenant)
    undo = announce(liar.transports[0], tenant, lie, "liar")
    operator = Mesh(bus, tenant=tenant)
    try:
        assert operator.discover("arm", min_count=1, timeout=2.0)
        remote = operator.connect("arm")
        assert remote.supports_position_hold is expected
    finally:
        undo()
        operator.close()
        liar.close()
        sim.close()


def test_a_robot_from_the_mesh_reprs_as_remote_not_hardware(mesh_pair):
    a, b = mesh_pair
    device = Robot("so101", sim=Simulation())
    a.mount(device, device_id="arm", watchdog_ms=None)
    assert b.discover("arm", min_count=1, timeout=3.0)
    remote = b.connect("arm")
    assert "remote on" in repr(remote) and "hardware" not in repr(remote)
    a.close()
    device.sim.close()
