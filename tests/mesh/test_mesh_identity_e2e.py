"""A mesh with identities over real zenoh: a real robot moves for its operator, and an admitted peer can speak
only as itself (no beats under another peer's id, no answers for a device its certificate does not serve)."""

from __future__ import annotations

import socket
import time

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("cryptography")

from strands_robots import Robot  # noqa: E402
from strands_robots.mesh import Mesh, pki, wire  # noqa: E402
from strands_robots.mesh.identity import Identity  # noqa: E402
from strands_robots.mesh.transports import open_transport  # noqa: E402
from strands_robots.robot.safety import RequestRefused  # noqa: E402


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab")
    ca = pki.make_ca(root / "ca")
    serves = {"device": ["arm"], "operator": [], "insider": ["other-*"]}
    return {n: pki.issue(ca, n, root / n, serves=s) for n, s in serves.items()}


def _ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    [s.close() for s in socks]
    return ports


@pytest.fixture
def world(lab, tenant):
    """The device listens; the insider listens too and links to the device; the operator links to both, so every
    query of the operator reaches the insider's queryables and the insider reaches the device."""
    pd, pi = _ports(2)
    dev = Mesh(f"zenoh://?identity={lab['device']}&listen=tls/127.0.0.1:{pd}", tenant=tenant)
    arm = Robot("so101", driver="memory", confirm=lambda action: True)
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())[1]
    dev.mount(arm, device_id="arm", watchdog_ms=500.0)
    ins = open_transport(f"zenoh://?identity={lab['insider']}&listen=tls/127.0.0.1:{pi}&connect=tls/127.0.0.1:{pd}")
    url = f"zenoh://?identity={lab['operator']}&connect=tls/127.0.0.1:{pd}&connect=tls/127.0.0.1:{pi}"
    op = Mesh(url, tenant=tenant)
    assert op.discover("arm", min_count=1, timeout=5.0)
    yield dev, op, ins, arm, holds
    ins.close(), op.close(), dev.close(), arm.close()


def _rpc(t, tenant, name, peer, seq, sign=True, **args):
    env = wire.envelope(tenant, "arm", "rpc", name, seq=seq, args=args, peer=peer)
    raw = wire.encode(t.identity.seal(env)) if sign else wire.encode(env)
    return [wire.decode_envelope(r) for r in t.query(wire.key("rpc", tenant, "arm", name), raw, 2.0)]


def test_a_real_robot_moves_for_an_operator_with_a_mesh_certificate(world):
    dev, op, _ins, arm, _holds = world
    assert op.peer_id.startswith("operator.") and dev.peer_id.startswith("device.")
    op.invoke("arm", "act", action={"shoulder_pan": 0.2})
    assert arm.observe().joints["shoulder_pan"] == pytest.approx(0.2)
    assert op.broadcast("estop") == ["arm"] and arm.estopped


@pytest.mark.parametrize("how", ["unsigned", "another-peers-id", "replayed"])
def test_an_admitted_peer_cannot_speak_for_another_peer(world, tenant, how):
    _dev, op, ins, _arm, _holds = world
    if how == "replayed":
        env = op.transports[0].identity.seal(wire.envelope(tenant, "arm", "rpc", "observe", seq=4242, peer=op.peer_id))
        raw = wire.encode(env)
        key = wire.key("rpc", tenant, "arm", "observe")
        assert wire.decode_envelope(ins.query(key, raw, 2.0)[0])["kind"] == "result"
        replies = [wire.decode_envelope(r) for r in ins.query(key, raw, 2.0)]
    else:
        replies = _rpc(ins, tenant, "observe", op.peer_id, 9, sign=how != "unsigned")
    assert [r["error"]["type"] for r in replies] == ["Unverified"]


def test_beats_under_the_operators_id_never_keep_the_watchdog_quiet(world, tenant):
    _dev, op, ins, _arm, holds = world
    op.invoke("arm", "act", action={"shoulder_pan": 0.1})
    t0 = time.monotonic()
    for i in range(15):  # the operator stays connected but quiet; the insider beats with its id for 2.25 s
        _rpc(ins, tenant, "observe", op.peer_id, 1000 + i)
        time.sleep(0.15)
    assert holds and holds[0] - t0 < 1.5, "the watchdog must fire on the operator's own silence"


def test_an_answer_from_a_peer_that_does_not_serve_the_device_is_ignored(world, tenant):
    _dev, op, ins, _arm, _holds = world
    ident: Identity = ins.identity
    forged: list[int] = []

    def forge(_key, payload):
        env = wire.decode_envelope(payload)
        forged.append(env["seq"])
        kind, body = ("error", {"error": {"type": "RequestRefused", "message": "forged"}})
        return wire.encode(
            ident.seal(wire.envelope(tenant, "arm", kind, env["name"], seq=env["seq"], peer=env["peer"], **body))
        )

    undo = ins.queryable(f"strands/{tenant}/arm/rpc/**", forge)
    try:
        time.sleep(0.3)
        for _ in range(5):
            assert op.invoke("arm", "observe").joints["shoulder_pan"] == 0.0
            assert op.invoke("arm", "stop") is None
        assert len(forged) >= 10, "the forger must have heard (and answered) every request"
    finally:
        undo()


def test_a_device_announced_by_a_peer_that_may_not_serve_it_stays_out_of_discover(world, tenant):
    _dev, op, ins, _arm, _holds = world
    rec = {"id": "ghost", "device_type": "robot_arm", "tenant": tenant, "labels": {}, "manifest": {"name": "x"}}
    manifest = wire.encode(ins.identity.seal(wire.envelope(tenant, "ghost", "manifest", "manifest", result=rec)))
    undo = [ins.queryable(wire.key("manifest", tenant, "ghost"), lambda k, p: manifest)]
    undo.append(ins.liveliness(wire.key("presence", tenant, "ghost")))
    try:
        deadline = time.monotonic() + 3.0
        while not op._dirs[0].knows("ghost") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert op._dirs[0].knows("ghost"), "the operator must see the insider's presence token"
        assert op.discover("ghost", min_count=1, timeout=1.5) == []
    finally:
        [u() for u in undo]


def test_mounting_a_device_the_certificate_does_not_serve_is_refused(world):
    dev, *_ = world
    with pytest.raises(ValueError, match="may not answer for 'other'"):
        dev.mount(Robot("so101"), device_id="other")


def test_a_real_robot_is_observe_only_without_an_identity_even_with_a_config_file(tenant, tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    path = tmp_path / "z.json5"
    path.write_text('{mode: "peer"}')
    dev = Mesh(f"zenoh://?config={path}&listen=tcp/127.0.0.1:{port}", tenant=tenant)
    op = Mesh(f"zenoh://?connect=tcp/127.0.0.1:{port}", tenant=tenant)
    arm = Robot("so101", driver="memory", confirm=lambda action: True)
    try:
        dev.mount(arm, device_id="arm")
        assert op.discover("arm", min_count=1, timeout=3.0)
        with pytest.raises(RequestRefused, match="observe-only"):
            op.invoke("arm", "act", action={"shoulder_pan": 0.1})
    finally:
        op.close(), dev.close(), arm.close()
