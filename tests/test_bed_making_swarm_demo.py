"""End-to-end test of the AI-Fabric-orchestrated swarm bed-making demo.

Runs the swarm scenario over the in-process loopback bus (no broker) and
asserts the full peer-to-peer coordination contract from issue #2 feedback:
both equal peers reach the goal "the bed is made", one peer asks another for
help when stuck, the other sees the request and offers help, and the event +
help histories reflect it on both sides.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_g1_driver import BedMakingG1Driver
    from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_swarm_demo import (
        LoopbackBus,
        make_direct_invoker,
        run_scenario,
    )
except ImportError:  # pragma: no cover - optional Device Connect edge runtime
    pytest.skip("device_connect_edge not installed", allow_module_level=True)


def _build_swarm():
    bus = LoopbackBus()
    peers = [
        BedMakingG1Driver(device_id="g1-0", role="peer", auto_offer=True),
        BedMakingG1Driver(device_id="g1-1", role="peer", auto_offer=True),
    ]
    for peer in peers:
        bus.register(peer)
    return peers


def test_loopback_bus_routes_events_to_peer_handlers():
    """A help request from one peer must reach the OTHER peer's @on handler
    over the in-process bus — the same routing NATS would perform.

    Drive it through the public ``askForHelp`` action (an ``@rpc``) rather than
    calling the raw ``@emit`` directly: the action emits ``helpRequested`` via
    the driver's safe-emit path, which the bus fans out to peer handlers. (A
    raw ``@emit`` call requires an attached DeviceRuntime, so it is not the
    supported entry point.)
    """

    peers = _build_swarm()
    p0, p1 = peers

    asyncio.run(p0.askForHelp(corner="A", reason="stuck"))

    # p1 saw the request over the bus (recorded in its help history + peer view).
    assert any(r["from"] == "g1-0" for r in p1.agent.help_history())
    assert p1.agent.peer_list()["g1-0"]["status"] == "needs_help"


def test_swarm_scenario_reaches_goal_and_exchanges_help():
    peers = _build_swarm()
    result = asyncio.run(run_scenario(peers, invoke=make_direct_invoker()))

    # Both equal peers see the shared goal reached (4/4 corners placed).
    assert all(g["bed_made"] for g in result["goals"].values())
    assert all(g["progress"] == "4/4" for g in result["goals"].values())


def test_swarm_scenario_help_is_visible_on_both_sides():
    peers = _build_swarm()
    p0, p1 = peers
    asyncio.run(run_scenario(peers, invoke=make_direct_invoker()))

    # p0 got stuck and asked for help with corner A.
    assert any(r["kind"] == "request" and r["from"] == "g1-0" for r in p0.agent.help_history())
    # p1 saw that request in its own help history (dashboard visibility).
    assert any(r["kind"] == "request" and r["from"] == "g1-0" for r in p1.agent.help_history())
    # p1 offered help; p0 saw the offer.
    assert any(r["kind"] == "offer" and r["from"] == "g1-1" for r in p1.agent.help_history())
    assert any(r["kind"] == "offer" and r["from"] == "g1-1" for r in p0.agent.help_history())


def test_swarm_scenario_peers_see_each_other():
    peers = _build_swarm()
    p0, p1 = peers
    asyncio.run(run_scenario(peers, invoke=make_direct_invoker()))
    assert "g1-1" in p0.agent.peer_list()
    assert "g1-0" in p1.agent.peer_list()


def test_swarm_scenario_event_history_is_ordered_and_has_goal():
    peers = _build_swarm()
    asyncio.run(run_scenario(peers, invoke=make_direct_invoker()))

    combined_has_goal = False
    for peer in peers:
        events = peer.agent.event_history(limit=200)
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs), "event history must be time-ordered"
        if any(e["kind"] == "goal" for e in events):
            combined_has_goal = True
    assert combined_has_goal, "a goal-reached event should be logged once the bed is made"
