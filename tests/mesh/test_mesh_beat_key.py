"""Where the transport does not prove who sent a request, the watched mover is the session that got the beat key in
the reply to its ``act``: another peer naming the mover's id never keeps the mover's watch alive."""

from __future__ import annotations

import itertools
import time

import pytest

from strands_robots import Robot
from strands_robots.mesh import wire
from strands_robots.mesh.reply import the_reply
from strands_robots.mesh.watchdog import MeshWatchdog

SEQ = itertools.count(1000)


@pytest.fixture
def device(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    holds: list[float] = []
    real_hold = arm.body.hold
    arm.body.hold = lambda: (holds.append(time.monotonic()), real_hold())
    a.mount(arm, device_id="arm", watchdog_ms=200)
    assert b.discover("arm", min_count=1, timeout=3.0)
    yield holds, a, b
    a.close()
    arm.close()


def _observe_as(mesh, peer: str, beat: str | None = None) -> dict:
    """An observe as a peer that builds its own envelope and names ``peer``."""
    seq = next(SEQ)
    env = wire.envelope(mesh.tenant, "arm", "rpc", "observe", seq=seq, args={}, peer=peer, beat=beat)
    replies = mesh.transports[0].query(wire.key("rpc", mesh.tenant, "arm", "observe"), wire.encode(env), 2.0)
    return the_reply(replies, mesh.tenant, "arm", "observe", seq)


def _beat_for(seconds: float, holds: list[float], send) -> float:
    t0 = end = time.monotonic()
    end += seconds
    while not holds and time.monotonic() < end:
        send()
        time.sleep(0.05)
    return t0


def test_a_peer_naming_the_silent_movers_id_never_delays_its_hold(device):
    holds, _, b = device
    b.invoke("arm", "act", action={"shoulder_pan": 0.1})
    t0 = _beat_for(1.5, holds, lambda: _observe_as(b, b.peer_id))  # 7x the timeout, without the key
    assert holds and holds[0] - t0 < 0.6, "the mover went quiet: it is held on time whoever else names it"


def test_the_movers_own_requests_keep_its_watch_alive(device):
    holds, _, b = device
    b.invoke("arm", "act", action={"shoulder_pan": 0.1})
    _beat_for(1.0, holds, lambda: b.invoke("arm", "observe"))
    assert holds == [], "the mover kept talking: no hold"
    end = time.monotonic() + 2.0
    while not holds and time.monotonic() < end:
        time.sleep(0.005)
    assert len(holds) == 1, "and once it goes quiet it is held"


def test_the_beat_key_travels_only_back_to_the_mover(device):
    holds, _, b = device
    b.invoke("arm", "act", action={"shoulder_pan": 0.1})
    key = b._beats["arm"]
    assert _observe_as(b, b.peer_id).get("beat") is None, "an observe reply never hands the key out"
    _beat_for(1.0, holds, lambda: _observe_as(b, b.peer_id, beat=key))
    assert holds == [], "a request that shows the mover's key is the mover's"


def test_without_a_proven_peer_only_the_key_counts():
    arm = Robot("so101")
    dog = MeshWatchdog(arm, 10_000, [], "lab")
    try:
        key = dog.on_rpc("act", "mover", None, False)(True)
        assert isinstance(key, str) and len(key) >= 16
        assert dog.on_rpc("act", "mover", key, False)(True) == key, "the same mover keeps its key"
        before = dog.dog.last
        dog.on_rpc("observe", "mover", "not-the-key", False)
        dog.on_rpc("observe", "mover", None, False)
        assert dog.dog.last == before, "a request without the key is no sign of life"
        dog.on_rpc("observe", "mover", key, False)
        assert dog.dog.last > before
        stolen = dog.on_rpc("act", "mover", None, False)(True)  # moved, but could not show the key
        assert stolen != key, "whoever moved it without the key gets a new key: the old one is no beat any more"
    finally:
        dog.close()
        arm.close()


def test_a_proven_peer_needs_no_key():
    arm = Robot("so101")
    dog = MeshWatchdog(arm, 10_000, [], "lab")
    try:
        dog.on_rpc("act", "op.1")(True)
        before = dog.dog.last
        time.sleep(0.01)
        dog.on_rpc("observe", "op.1")
        assert dog.dog.last > before, "on a transport that proves the peer, its id is enough (mesh identities)"
    finally:
        dog.close()
        arm.close()


@pytest.mark.parametrize("beat", ["", 7, "x" * 65])
def test_an_envelope_with_a_malformed_beat_is_refused(beat):
    env = wire.envelope("lab", "arm", "rpc", "observe", seq=1, args={}, peer="p")
    env["beat"] = beat
    with pytest.raises(ValueError, match="beat"):
        wire.decode_envelope(wire.encode(env))
