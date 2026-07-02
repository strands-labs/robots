"""Tests for the swarm/peer surface of the bed-making Device Connect driver.

These cover the features added per issue #2 feedback (comment 4580165360):
- both robots are equal peers pursuing the goal state "the bed is made",
- callable @rpc actions exposed in the dashboard (askForHelp, offerHelp,
  pickUpBedSheet, walkToNextCorner, putDownBedSheet, ...),
- event history + a dedicated help-request history queryable via @rpc,
- peers see each other's help requests over Device Connect events.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_g1_driver import (
        GOAL_STATE,
        SHEET_TO_BED,
        BedMakingG1Driver,
        SwarmAgent,
    )
except ImportError:  # pragma: no cover - optional Device Connect edge runtime
    pytest.skip("device_connect_edge not installed", allow_module_level=True)


def _run(coro):
    return asyncio.run(coro)


# ── Callable actions are advertised as dashboard functions ─────────────────


def test_swarm_actions_are_advertised_as_callable_functions():
    """Every swarm action must show up as a Device Connect function so the
    portal renders it as an invocable button."""

    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    caps = driver.capabilities
    func_names = {f.name for f in caps.functions}

    expected = {
        "askForHelp",
        "offerHelp",
        "pickUpBedSheet",
        "walkToNextCorner",
        "putDownBedSheet",
        "getStatus",
        "getGoalState",
        "listPeers",
        "getEventHistory",
        "getHelpHistory",
        "emergencyStopAll",
    }
    assert expected <= func_names, f"missing dashboard functions: {expected - func_names}"


def test_action_parameters_have_schema_and_descriptions():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    by_name = {f.name: f for f in driver.capabilities.functions}

    walk = by_name["walkToNextCorner"]
    assert walk.description  # docstring summary survived
    props = walk.parameters["properties"]
    assert props["direction"]["default"] == "counterclockwise"

    ask = by_name["askForHelp"]
    ask_props = ask.parameters["properties"]
    assert {"corner", "reason", "target"} <= set(ask_props)


def test_swarm_events_are_advertised():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    event_names = {e.name for e in driver.capabilities.events}
    assert {"helpRequested", "helpOffered", "walkingTo", "goalReached"} <= event_names


# ── Goal state ─────────────────────────────────────────────────────────────


def test_goal_state_tracks_progress_to_bed_made():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")

    goal = _run(driver.getGoalState())
    assert goal["goal"] == GOAL_STATE
    assert goal["bed_made"] is False
    assert goal["progress"] == "0/4"

    for corner in ("A", "B", "C", "D"):
        result = _run(driver.putDownBedSheet(corner=corner))
    assert result["bed_made"] is True

    goal = _run(driver.getGoalState())
    assert goal["bed_made"] is True
    assert goal["progress"] == "4/4"
    assert goal["remaining"] == 0


def test_put_down_infers_bed_corner_from_sheet_corner():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    result = _run(driver.putDownBedSheet(corner="A"))
    assert result["bed_corner"] == SHEET_TO_BED["A"]


# ── walkToNextCorner steps around the bed ──────────────────────────────────


def test_walk_to_next_corner_counterclockwise_then_clockwise():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")

    # Counter-clockwise from the NW start: NW -> SW -> SE -> NE -> NW.
    seen = [_run(driver.walkToNextCorner())["target_corner"] for _ in range(4)]
    assert seen == ["SW", "SE", "NE", "NW"]

    # Clockwise walks the other way around the ring.
    cw = _run(driver.walkToNextCorner(direction="clockwise"))["target_corner"]
    assert cw in {"NW", "NE", "SE", "SW"}


# ── Pick up / hold ─────────────────────────────────────────────────────────


def test_pickup_sets_held_corner_and_logs_event():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    result = _run(driver.pickUpBedSheet(corner="A"))
    assert result["ok"] is True

    status = _run(driver.getStatus())
    assert status["held_corner"] == "A"

    events = _run(driver.getEventHistory())["events"]
    assert any(e["kind"] == "pickup" and e["detail"]["corner"] == "A" for e in events)


# ── Event + help history ───────────────────────────────────────────────────


def test_event_history_accumulates_and_respects_limit():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    for corner in ("A", "B", "C"):
        _run(driver.pickUpBedSheet(corner=corner))
        _run(driver.putDownBedSheet(corner=corner))

    all_events = _run(driver.getEventHistory(limit=100))["events"]
    assert len(all_events) >= 6  # 3 pickups + 3 placements (+ maybe goal)

    limited = _run(driver.getEventHistory(limit=2))["events"]
    assert len(limited) == 2
    # newest-last ordering: the limited slice is the tail of the full log
    assert limited == all_events[-2:]


def test_help_request_is_recorded_in_help_history():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    result = _run(driver.askForHelp(corner="A", reason="corner drifting", target="g1-1"))
    assert result["kind"] == "request"

    help_log = _run(driver.getHelpHistory())["help"]
    assert len(help_log) == 1
    rec = help_log[0]
    assert rec["from"] == "g1-0"
    assert rec["to"] == "g1-1"
    assert rec["corner"] == "A"
    assert "drifting" in rec["reason"]


def test_offer_help_defaults_to_last_requester():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    # A peer asked us for help over the mesh...
    _run(driver.onPeerHelpRequested("g1-1", "helpRequested", {"corner": "B", "reason": "stuck", "target": ""}))
    # ...we offer help without naming a target; it should resolve to g1-1.
    result = _run(driver.offerHelp(corner="B"))
    assert result["to"] == "g1-1"

    help_log = _run(driver.getHelpHistory())["help"]
    kinds = [(h["kind"], h["from"]) for h in help_log]
    assert ("request", "g1-1") in kinds
    assert ("offer", "g1-0") in kinds


# ── Peer visibility of help requests over the mesh ─────────────────────────


def test_peer_sees_help_request_in_its_own_history():
    """When g1-0 asks for help, g1-1's @on handler must log it so g1-1's
    dashboard shows 'g1-0 asked for help'."""

    peer = BedMakingG1Driver(device_id="g1-1", role="peer")
    _run(peer.onPeerHelpRequested("g1-0", "helpRequested", {"corner": "A", "reason": "drifting", "target": ""}))

    help_log = _run(peer.getHelpHistory())["help"]
    assert len(help_log) == 1
    assert help_log[0]["from"] == "g1-0"

    peers = _run(peer.listPeers())["peers"]
    assert peers["g1-0"]["status"] == "needs_help"


def test_help_request_for_self_is_ignored():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    _run(driver.onPeerHelpRequested("g1-0", "helpRequested", {"corner": "A", "reason": "x", "target": ""}))
    assert _run(driver.getHelpHistory())["help"] == []


def test_auto_offer_triggers_when_free_peer_sees_a_request():
    """With auto_offer, a free peer offers help automatically when it sees a
    request — making the swarm self-organising without an orchestrator."""

    helper = BedMakingG1Driver(device_id="g1-1", role="peer", auto_offer=True)
    _run(helper.onPeerHelpRequested("g1-0", "helpRequested", {"corner": "A", "reason": "stuck", "target": ""}))
    help_log = _run(helper.getHelpHistory())["help"]
    kinds = {h["kind"] for h in help_log}
    assert "request" in kinds  # it recorded the incoming request
    assert "offer" in kinds  # ...and auto-offered help


def test_busy_peer_does_not_auto_offer():
    helper = BedMakingG1Driver(device_id="g1-1", role="peer", auto_offer=True)
    _run(helper.pickUpBedSheet(corner="C"))  # now holding -> not free
    _run(helper.onPeerHelpRequested("g1-0", "helpRequested", {"corner": "A", "reason": "stuck", "target": ""}))
    help_log = _run(helper.getHelpHistory())["help"]
    assert all(h["kind"] != "offer" for h in help_log)


# ── Emergency stop ─────────────────────────────────────────────────────────


def test_emergency_stop_sets_status_and_logs():
    driver = BedMakingG1Driver(device_id="g1-0", role="peer")
    _run(driver.emergencyStopAll(reason="test"))
    status = _run(driver.getStatus())
    assert status["status"] == "stopped"
    events = _run(driver.getEventHistory())["events"]
    assert any(e["kind"] == "estop" for e in events)


# ── SwarmAgent unit behaviour ──────────────────────────────────────────────


def test_swarm_agent_is_bed_made_threshold():
    agent = SwarmAgent(device_id="g1-0")
    for corner in ("A", "B", "C"):
        agent.record_placement(corner)
    assert agent.is_bed_made() is False
    agent.record_placement("D")
    assert agent.is_bed_made() is True


def test_swarm_agent_can_assist_only_when_free():
    agent = SwarmAgent(device_id="g1-0")
    assert agent.can_assist() is True
    agent.set_held_corner("A")
    assert agent.can_assist() is False
    agent.set_held_corner(None)
    assert agent.can_assist() is True
    agent.set_status("stopped")
    assert agent.can_assist() is False
