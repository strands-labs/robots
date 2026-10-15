"""broadcast("estop") latches every matching device; resume goes through the device's resume authority."""

from __future__ import annotations

import time

import pytest

from strands_robots import Robot
from strands_robots.mesh import Mesh, RequestRefused, authority


@pytest.fixture
def lab(mesh_pair):
    a, b = mesh_pair
    arms = [Robot("so101", id=f"arm{i}") for i in range(3)]
    for i, arm in enumerate(arms):
        a.mount(arm, device_id=f"arm{i}", labels={"bench": "left" if i < 2 else "right"})
    assert len(b.discover("device(*)", min_count=3, timeout=3.0)) == 3
    yield arms, a, b
    a.close()  # unmount before the robots go
    for arm in arms:
        arm.close()


def test_broadcast_estop_reaches_every_device_and_reports_acks(lab):
    arms, _, b = lab
    t0 = time.perf_counter()
    acked = b.broadcast("estop", selector="device(*)")
    assert time.perf_counter() - t0 < 1.0
    assert sorted(acked) == ["arm0", "arm1", "arm2"]
    assert all(arm.estopped for arm in arms)


def test_broadcast_estop_honours_the_selector(lab):
    arms, _, b = lab
    assert sorted(b.broadcast("estop", selector="device(bench:left)")) == ["arm0", "arm1"]
    assert [arm.estopped for arm in arms] == [True, True, False]


def test_an_undiscovered_peer_still_latches_from_the_tenant_estop(lab):
    arms, a, _ = lab
    late = Mesh(a.transports[0], tenant=a.tenant, own_transports=False)
    late.broadcast("estop", selector="device(*)", deadline=0.0)
    deadline = time.monotonic() + 2.0
    while not all(arm.estopped for arm in arms) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert all(arm.estopped for arm in arms)
    late.close()


def test_resume_by_the_operator_who_stopped(lab):
    arms, _, b = lab
    b.broadcast("estop", selector="arm0")
    b.invoke("arm0", "resume", approved_by=b.operator)
    assert not arms[0].estopped


def test_resume_from_a_non_authority_is_refused(lab):
    arms, a, b = lab
    b.broadcast("estop", selector="arm0")
    with pytest.raises(RequestRefused, match="not a resume authority"):
        b.invoke("arm0", "resume", approved_by="mallory")
    assert arms[0].estopped
    assert b.broadcast("resume", selector="arm0", approved_by="mallory") == []
    assert arms[0].estopped


def test_a_named_resume_authority_may_resume(mesh_pair):
    a, b = mesh_pair
    a.resume_authority = authority.Authority(names=("safety-lead",))
    arm = Robot("so101")
    a.mount(arm, device_id="arm")
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.broadcast("estop", selector="arm")
    assert b.broadcast("resume", selector="arm", approved_by="safety-lead") == ["arm"]
    assert not arm.estopped
    a.close()  # unmount before the robot goes
    arm.close()


def test_with_keys_only_a_signed_fresh_assertion_for_this_lockout_resumes(mesh_pair):
    a, b = mesh_pair
    private, public = authority.keygen()
    a.resume_authority = authority.Authority(public_keys=(public,))
    arm = Robot("so101")
    a.mount(arm, device_id="arm")
    assert b.discover("arm", min_count=1, timeout=3.0)
    b.broadcast("estop", selector="arm")
    with pytest.raises(RequestRefused, match="signed"):
        b.invoke("arm", "resume", approved_by=b.operator)
    b.resume_key, _ = authority.keygen()
    with pytest.raises(RequestRefused, match="signature"):
        b.invoke("arm", "resume", approved_by="op")
    assert arm.estopped
    b.resume_key = private
    b.invoke("arm", "resume", approved_by="op")
    assert not arm.estopped
    assert b.invoke("arm", "lockout")["engaged"] is False


def test_the_mesh_kill_switch(monkeypatch):
    monkeypatch.setenv("STRANDS_MESH", "false")
    with pytest.raises(RuntimeError, match="STRANDS_MESH"):
        Mesh("mem://killed")


def test_claimed_names_are_not_trusted_on_a_real_network():
    from strands_robots.mesh.authority import Authority, Lockout
    from strands_robots.mesh.serve import is_local

    assert is_local("mem://x") and is_local("zenoh://?listen=tcp/127.0.0.1:7447")
    assert not is_local("zenoh://") and not is_local("zenoh://?connect=tcp/10.0.0.5:7447")
    assert not is_local("zenoh://?listen=tcp/127.0.0.1:7447&connect=tcp/10.0.0.5:7447")
    assert not is_local("zenoh://?listen=tcp/0.0.0.0:7447")
    stop = Lockout(True, "e1", "alice")
    with pytest.raises(RequestRefused, match="needs operator keys"):
        Authority().check("arm", stop, "alice", None, local=False)
    assert Authority(trust_names=True).check("arm", stop, "alice", None, local=False) == "alice"
    assert Authority().check("arm", stop, "alice", None, local=True) == "alice"


def test_the_refusal_names_the_ways_out():
    from strands_robots.mesh.authority import Authority, Lockout

    with pytest.raises(RequestRefused) as err:
        Authority().check("arm", Lockout(True, "e1", None), "me", None, local=True)
    msg = str(err.value)
    assert "by=<operator>" in msg and "Authority(names=" in msg and "public_keys" in msg
