"""Devices announce themselves; a discover() sees them, and loses them when they go."""

from __future__ import annotations

import time

import pytest

from strands_robots.mesh import Device, Mesh


class FakeArm:
    """Enough of a Robot for presence: it describes itself."""

    def __init__(self, name="arm"):
        self.name = name

    def describe(self):
        return {"name": self.name, "dof": 6, "device_type": "arm"}


def test_mount_then_discover_by_label(mesh_pair):
    a, b = mesh_pair
    d = a.mount(FakeArm(), device_id="so101-a", labels={"role": "arm", "bench": "desk"})
    assert isinstance(d, Device) and d.id == "so101-a" and d.tenant == a.tenant
    a.mount(FakeArm("cam"), device_id="cam-1", labels={"role": "camera"})
    found = b.discover("device(role:arm)", min_count=1, timeout=3.0)
    assert [x.id for x in found] == ["so101-a"]
    assert found[0].labels == {"role": "arm", "bench": "desk"}
    assert found[0].device_type == "arm"
    assert found[0].manifest["dof"] == 6
    assert found[0].last_seen > 0
    assert {x.id for x in b.discover("device(*)", min_count=2, timeout=3.0)} == {"so101-a", "cam-1"}


def test_unmount_leaves_discover(mesh_pair):
    a, b = mesh_pair
    a.mount(FakeArm(), device_id="gone-1")
    assert b.discover("gone-1", min_count=1, timeout=3.0)
    a.unmount("gone-1")
    end = time.monotonic() + 3.0
    while b.discover("gone-1") and time.monotonic() < end:
        time.sleep(0.02)
    assert b.discover("gone-1") == []


def test_tenants_do_not_see_each_other(mesh_pair):
    a, _ = mesh_pair
    other = Mesh(a.transports[0], tenant=a.tenant + "-other", own_transports=False)
    a.mount(FakeArm(), device_id="mine")
    time.sleep(0.3)
    assert other.discover("device(*)") == []
    other.close()


def test_a_device_id_is_unique_per_mesh(mesh_pair):
    a, _ = mesh_pair
    a.mount(FakeArm(), device_id="dup")
    with pytest.raises(ValueError, match="already mounted"):
        a.mount(FakeArm(), device_id="dup")


def test_a_real_sim_robot_mounts_with_its_description():
    from strands_robots import Robot

    with Mesh("mem://real-robot", tenant="t-real") as m:
        d = m.mount(Robot("so101"), labels={"role": "arm"})
        assert d.id == "so101"
        assert m.discover("device(role:arm)", min_count=1, timeout=1.0)[0].manifest["dof"] == 6


def test_a_peer_answering_for_another_id_never_shadows_a_device():
    from strands_robots.mesh import wire
    from strands_robots.mesh.transports import open_transport

    with Mesh("mem://shadow", tenant="t-shadow") as m:
        m.mount(FakeArm(), device_id="so101-a", labels={"role": "arm"})
        evil = open_transport("mem://shadow")
        fake = {"id": "so101-a", "device_type": "arm", "labels": {"role": "impostor"}, "manifest": {}}
        reply = wire.encode(wire.envelope("t-shadow", "evil", "manifest", "manifest", result=fake))
        evil.queryable(wire.key("manifest", "t-shadow", "evil"), lambda k, p: reply)
        evil.liveliness(wire.key("presence", "t-shadow", "evil"))
        time.sleep(0.3)
        assert [(d.id, d.labels) for d in m.discover()] == [("so101-a", {"role": "arm"})]
        evil.close()


def test_the_record_carries_no_live_status():
    from strands_robots import Robot

    with Mesh("mem://live", tenant="t-live") as m:
        arm = Robot("so101")
        m.mount(arm)
        rec = m.discover("so101", min_count=1, timeout=1.0)[0].manifest
        assert "estop" not in rec and "task" not in rec
        m.close()
        arm.close()


def test_the_default_tenant_is_the_users_own(monkeypatch):
    monkeypatch.setenv("STRANDS_TENANT", "my lab")
    with Mesh("mem://tenant-default") as m:
        assert m.tenant == "my-lab"
