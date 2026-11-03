"""M3: an invoke's reply must answer THIS request (tenant, device, rpc, seq), never just ``replies[0]``."""

from __future__ import annotations

import pytest

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh, wire
from strands_robots.mesh.reply import the_reply


def test_the_reply_matches_tenant_device_rpc_and_seq():
    mine = wire.encode(wire.envelope("t", "arm", "result", "act", seq=7, result=1))
    wrong_seq = wire.encode(wire.envelope("t", "arm", "result", "act", seq=6, result=1))
    wrong_dev = wire.encode(wire.envelope("t", "arm2", "result", "act", seq=7, result=1))
    assert the_reply([wrong_seq, mine, wrong_dev], "t", "arm", "act", 7)["seq"] == 7
    with pytest.raises(LookupError, match="none answers"):
        the_reply([wrong_seq, wrong_dev], "t", "arm", "act", 7)
    # a stop both peers of a raced id answer: the first matching ack is this caller's own, not an error
    assert the_reply([mine, mine], "t", "arm", "act", 7)["result"] == 1


def test_a_device_echoes_the_request_seq_so_a_shared_rpc_key_stays_matched():
    """Several callers share one rpc key; the device stamps each reply with the caller's own seq, so a caller whose
    seq has drifted from the device's internal counter still matches its own reply instead of raising."""
    sim = Simulation()
    arm = Robot("so101", sim=sim, id="arm")
    with (
        Mesh("mem://m3-echo", tenant="t-m3") as host,
        Mesh("mem://m3-echo", tenant="t-m3") as a,
        Mesh("mem://m3-echo", tenant="t-m3") as b,
    ):
        host.mount(arm, watchdog_ms=None)
        assert a.discover("arm", min_count=1, timeout=3.0)
        assert b.discover("arm", min_count=1, timeout=3.0)
        a.invoke("arm", "observe")  # device counter -> 1, a.seq -> 1
        b.invoke("arm", "observe")  # device counter -> 2, b.seq -> 1
        # a.seq -> 2 while the device counter is now 3; the echoed seq keeps them matched
        obs = a.invoke("arm", "observe")
        assert obs is not None and obs.joints
    sim.close()
