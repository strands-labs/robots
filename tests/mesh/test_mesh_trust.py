"""What a peer that is not the operator can do on the mesh: a refused motion arms nothing, a second answer to one
request is refused, and a zenoh mesh can be given a config that says who may join."""

from __future__ import annotations

import json
import logging
import time

import pytest

from strands_robots import Robot
from strands_robots.mesh import wire
from strands_robots.mesh.reply import the_reply
from strands_robots.mesh.transports.base import RemoteError


@pytest.fixture
def device(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, device_id="arm", watchdog_ms=200)
    assert b.discover("arm", min_count=1, timeout=3.0)
    yield arm, holds, a, b
    a.close()
    arm.close()


def _raw_act(mesh, peer: str, args: dict) -> dict:
    """An act as a peer that builds its own envelope would send it (no client-side argument checks)."""
    env = wire.envelope(mesh.tenant, "arm", "rpc", "act", seq=99, args=args, peer=peer)
    replies = mesh.transports[0].query(wire.key("rpc", mesh.tenant, "arm", "act"), wire.encode(env), 2.0)
    return the_reply(replies, mesh.tenant, "arm", "act", 99)


@pytest.mark.parametrize(
    "args",
    [{"action": {"shoulder_pan": 0.1}, "bogus": 1}, {"action": {"no_such_joint": 0.1}}],
    ids=["unknown-argument", "refused-by-the-robot"],
)
def test_a_refused_motion_never_arms_the_watchdog(device, args):
    _, holds, _, b = device
    assert _raw_act(b, "someone", args)["kind"] == "error"
    time.sleep(0.5)
    assert holds == [], "a request that moved nothing must not make the device apply on_disconnect"


def test_a_refused_motion_leaves_the_real_movers_watch_as_it_was(device):
    _, holds, _, b = device
    b.invoke("arm", "act", action={"shoulder_pan": 0.1})
    assert _raw_act(b, "someone-else", {"action": {"no_such_joint": 0.1}})["kind"] == "error"
    t0 = time.monotonic()
    end = t0 + 2.0
    while not holds and time.monotonic() < end:
        time.sleep(0.005)
    assert len(holds) == 1, "the real mover's silence still recalls the arm"


def test_disconnect_actuation_leaves_an_audit_row(device, caplog):
    _, holds, _, b = device
    with caplog.at_level(logging.WARNING, logger="strands_robots.audit"):
        b.invoke("arm", "act", action={"shoulder_pan": 0.1})
        end = time.monotonic() + 2.0
        while not holds and time.monotonic() < end:
            time.sleep(0.005)
    assert holds and any("on_disconnect" in r.getMessage() for r in caplog.records)


def _reply(kind: str, name: str = "observe", **body) -> bytes:
    return wire.encode(wire.envelope("lab", "arm", kind, name, seq=7, **body))


def test_two_different_answers_to_one_request_are_refused():
    real, forged = _reply("result", result={"q": 0.1}), _reply("result", result={"q": 1.5})
    with pytest.raises(RemoteError, match="different answers"):
        the_reply([forged, real], "lab", "arm", "observe", 7)
    assert the_reply([real, real], "lab", "arm", "observe", 7)["result"] == {"q": 0.1}  # the same answer twice
    stops = [_reply("result", "stop", result={"held": True}), _reply("result", "stop", result={"held": False})]
    assert the_reply(stops, "lab", "arm", "stop", 7)["result"] == {"held": True}  # every holder stops it


def test_a_zenoh_config_file_is_the_session_config(tmp_path, monkeypatch):
    zenoh = pytest.importorskip("zenoh")
    from strands_robots.mesh.transports import zenoh as zt

    path = tmp_path / "zenoh.json5"
    path.write_text(json.dumps({"mode": "client", "transport": {"auth": {"usrpwd": {"user": "op", "password": "x"}}}}))
    cfg = zt.session_config(zenoh, f"zenoh://?connect=tcp/10.0.0.2:7447&config={path}", presence_ttl=3.0)
    assert json.loads(cfg.get_json("mode")) == "client"  # the file's word
    assert json.loads(cfg.get_json("transport/auth/usrpwd/user")) == "op"
    assert json.loads(cfg.get_json("connect/endpoints")) == ["tcp/10.0.0.2:7447"]
    assert json.loads(cfg.get_json("transport/link/tx/lease")) == 2000
    monkeypatch.setenv("STRANDS_ZENOH_CONFIG", str(path))
    assert zt.config_file("zenoh://") == str(path)
    monkeypatch.delenv("STRANDS_ZENOH_CONFIG")
    assert zt.config_file("zenoh://") is None


def test_a_real_robot_on_a_mesh_anyone_may_join_is_reported(tenant, caplog):
    import test_drivers

    from strands_robots.mesh import Mesh
    from strands_robots.mesh.transports import open_transport

    t = open_transport(f"mem://open-{tenant}")
    t.open_peers = True  # what a zenoh transport without a config file says
    mesh = Mesh([t], tenant=tenant)
    d, _ = test_drivers.make_driver()
    real = Robot("so101", driver=d, id="real_arm", twin=False)
    with caplog.at_level(logging.WARNING, logger="strands_robots.audit"):
        mesh.mount(real, device_id="real_arm")
    assert any("STRANDS_ZENOH_CONFIG" in r.getMessage() for r in caplog.records)
    mesh.close()
    real.close()
