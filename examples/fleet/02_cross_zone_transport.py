#!/usr/bin/env python3
"""Cross-zone transport: one shared skill, two zones, a gated handoff.

Goal: A site is split into two ownership zones, each run by its own zone
orchestrator - an in-process mesh peer (``init_mesh(..., peer_id="zone-a")``,
the multi-peer pattern from ``notebooks/06_fleet_orchestration.ipynb``) that
owns that zone's robots and nothing else. A fleet coordinator (a third peer,
holding no robot) decomposes a cross-zone transport request into per-zone
legs joined at a handoff dock, selects each leg's robot by running the ONE
shared ``transport`` skill requirement through the suite's ``capabilities.py``
hard-constraint filter - the identical skill artifact serves both zones, so
there is zero per-zone skill code to fork - and dispatches each leg over
``mesh.send`` behind a human-in-the-loop gate. Custody of the payload is
tracked explicitly: leg 2 is never dispatched before leg 1's success reply,
and an aborted handoff reports exactly where the tote physically is.

Dependencies: pip install "strands-robots[sim-mujoco,mesh]"
              (--dry-run needs only the base package: no simulator, no Zenoh)
Expected output: a stock -> etch transport decomposed into two legs handed
                 off at dock-ab, executed by different robot types (a wheeled
                 base in zone-a, a quadruped in zone-b) running the same
                 skill definition; a same-zone request that stays one leg;
                 and an unroutable request refused with a machine-readable
                 reason - never a guess.
Runtime: ~2 seconds with --dry-run; under ~90 seconds live (cached assets).

Note: Set STRANDS_MESH_HITL_ACTIONS=none to auto-approve dispatches (CI /
      smoke mode; logged loudly) - the default is an interactive prompt per
      leg. The read-only fleet dashboard (``dashboard.py``) attaches to the
      same mesh from a separate terminal and shows the handoff live.

Part of the fleet suite (epic #2179); the shared manifest schema lives in
``capabilities.py`` (also consumed by examples 01, 03 and 05).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_FLEET_DIR = Path(__file__).resolve().parent
if str(_FLEET_DIR) not in sys.path:
    # examples/ is not an installed package; the shared schema module lives
    # next to this script.
    sys.path.insert(0, str(_FLEET_DIR))

from capabilities import (  # noqa: E402
    StepRequirement,
    feasible_robots,
    manifest_from_dict,
)

from strands_robots.mesh.audit import log_safety_event  # noqa: E402

COORDINATOR_ID = "fleet-coordinator"
SITE = "site-a"

# The zone topology: each zone owns a set of locations and a set of robots.
# ``locations`` is the coverage the ZONE is responsible for; each robot's
# manifest declares which of them its transport skill actually serves (the
# manifest schema calls per-skill locations ``zones``). The two zones meet at
# a dock - the only place custody of a payload may change hands.
ZONES: dict[str, dict[str, Any]] = {
    "zone-a": {
        "locations": ("stock", "dock-ab"),
        "manifests": [
            {
                "robot": "lekiwi-a1",
                "site": SITE,
                "skills": [
                    {"name": "transport", "payload_kg": 8.0, "fixture": "tote_clamp", "zones": ["stock", "dock-ab"]}
                ],
            },
        ],
    },
    "zone-b": {
        "locations": ("dock-ab", "etch"),
        "manifests": [
            {
                "robot": "go2-b1",
                "site": SITE,
                "skills": [
                    {"name": "transport", "payload_kg": 5.0, "fixture": "tote_clamp", "zones": ["dock-ab", "etch"]}
                ],
            },
        ],
    },
}

# Docks: unordered zone pair -> the shared handoff location.
DOCKS: dict[frozenset[str], str] = {frozenset({"zone-a", "zone-b"}): "dock-ab"}

# Manifest robot id -> registry embodiment, used only by the live sim path.
ROBOT_EMBODIMENT = {"lekiwi-a1": "lekiwi", "go2-b1": "go2"}

# THE shared skill definition - the one artifact both zones execute. Every
# leg requirement is derived from it by ``leg_requirement`` below; neither
# zone carries a variant, an override, or a fork of any of it.
TRANSPORT_SKILL: dict[str, Any] = {"name": "transport", "fixture": "tote_clamp"}

# The requests this example runs: one cross-zone (the point of the example),
# one same-zone (decomposes to a single leg, no handoff), one unroutable
# (no zone covers the destination - refused with the reason, never guessed).
REQUESTS: list[dict[str, Any]] = [
    {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"},
    {"request_id": "X-02", "payload_kg": 3.0, "src": "dock-ab", "dst": "etch"},
    {"request_id": "X-03", "payload_kg": 3.0, "src": "stock", "dst": "cleanroom"},
]


def _audit(event: str, payload: dict[str, Any]) -> None:
    log_safety_event(event, COORDINATOR_ID, payload)


def zones_covering(location: str, zones: dict[str, dict[str, Any]]) -> list[str]:
    """Every zone that owns *location*, sorted by zone id (dock locations
    are owned by both zones they join)."""
    return sorted(zone_id for zone_id in zones if location in zones[zone_id]["locations"])


def decompose_cross_zone(
    request: dict[str, Any], zones: dict[str, dict[str, Any]], docks: dict[frozenset[str], str]
) -> dict[str, Any]:
    """Split one transport request into per-zone legs joined at a dock.

    Deterministic and total: the result is either ``{"legs": [...],
    "handoff": ...}`` or ``{"refused": {...}}`` with a machine-readable
    reason (``code`` plus the offending values). A request some single zone
    can serve end to end - including one starting at a dock both zones own -
    is one leg and no handoff; a genuinely cross-zone request is exactly two
    legs meeting at the pair's dock.
    """
    src, dst = request["src"], request["dst"]
    src_zones = zones_covering(src, zones)
    dst_zones = zones_covering(dst, zones)
    if not src_zones or not dst_zones:
        missing = src if not src_zones else dst
        return {"refused": {"code": "unroutable_location", "location": missing}}
    same_zone = sorted(set(src_zones) & set(dst_zones))
    if same_zone:
        return {"legs": [{"zone": same_zone[0], "src": src, "dst": dst}], "handoff": None}
    src_zone, dst_zone = src_zones[0], dst_zones[0]
    dock = docks.get(frozenset({src_zone, dst_zone}))
    if dock is None:
        return {"refused": {"code": "no_dock_between_zones", "zones": sorted((src_zone, dst_zone))}}
    return {
        "legs": [
            {"zone": src_zone, "src": src, "dst": dock},
            {"zone": dst_zone, "src": dock, "dst": dst},
        ],
        "handoff": {"location": dock, "from_zone": src_zone, "to_zone": dst_zone},
    }


def leg_requirement(request: dict[str, Any], leg: dict[str, Any]) -> StepRequirement:
    """One leg's hard constraints, derived from the SHARED skill definition.

    This is the only place a leg meets the skill artifact, and it is the
    same call for every zone - which is what makes "zero per-zone forks of
    skill code" a structural property rather than a convention.
    """
    return StepRequirement(
        skill=TRANSPORT_SKILL["name"],
        site=SITE,
        payload_kg=request["payload_kg"],
        fixture=TRANSPORT_SKILL["fixture"],
        zones=(leg["src"], leg["dst"]),
    )


def plan_request(
    request: dict[str, Any],
    zones: dict[str, dict[str, Any]],
    docks: dict[frozenset[str], str],
) -> dict[str, Any]:
    """Decompose one request and select each leg's robot zone-side.

    Robot selection runs the shared filter over ONLY the executing zone's
    manifests - the coordinator never sees another zone's robots, and a leg
    no robot in its zone can serve refuses the whole plan with the filter's
    per-robot rejections attached.
    """
    decomposition = decompose_cross_zone(request, zones, docks)
    if "refused" in decomposition:
        return {"request_id": request["request_id"], "refused": decomposition["refused"]}
    plan: list[dict[str, Any]] = []
    for index, leg in enumerate(decomposition["legs"], start=1):
        manifests = [manifest_from_dict(m) for m in zones[leg["zone"]]["manifests"]]
        feasible, rejections = feasible_robots(manifests, leg_requirement(request, leg))
        if not feasible:
            return {
                "request_id": request["request_id"],
                "refused": {"code": "no_capable_robot_in_zone", "zone": leg["zone"], "rejections": rejections},
            }
        robot = feasible[0].robot  # deterministic: feasible is sorted by name
        plan.append(
            {
                "leg": index,
                "zone": leg["zone"],
                "robot": robot,
                "src": leg["src"],
                "dst": leg["dst"],
                "instruction": (
                    f"request {request['request_id']} leg {index}: transport "
                    f"{request['payload_kg']} kg tote_clamp from {leg['src']} to {leg['dst']}"
                ),
            }
        )
    return {"request_id": request["request_id"], "legs": plan, "handoff": decomposition["handoff"]}


def execute_handoff(
    plan: dict[str, Any],
    approve: Callable[[str, str, str], bool],
    send: Callable[[str, dict[str, Any]], dict[str, Any]],
    *,
    n_steps: int = 25,
) -> dict[str, Any]:
    """Run one plan's legs in custody order through the HITL gate.

    ``approve(action, zone, instruction)`` is the HITL seam; ``send(zone,
    cmd)`` is the transport seam (``mesh.send`` live, a loopback in
    --dry-run and the smoke test). The ordering contract is the point: leg
    N+1 is dispatched only after leg N's success reply, because until then
    the payload has not physically reached the handoff dock. Every outcome
    is explicit - ``done`` / ``declined`` / ``failed`` - and an abort names
    the payload's current location, so a tote stranded at a dock is a
    recorded fact rather than a surprise.
    """
    request_id = plan["request_id"]
    summary: dict[str, Any] = {"request_id": request_id, "status": "failed", "legs_done": [], "at": None}
    if "refused" in plan:
        summary["status"] = "refused"
        summary["refusal"] = plan["refused"]
        _audit("handoff_refused", {"request_id": request_id, "refusal": plan["refused"]})
        print(f"  REFUSED {request_id}: {plan['refused']['code']}")
        return summary

    location = plan["legs"][0]["src"]
    summary["at"] = location
    _audit("handoff_dispatch", {"request_id": request_id, "legs": len(plan["legs"]), "handoff": plan["handoff"]})
    for leg in plan["legs"]:
        if not approve("dispatch", leg["zone"], leg["instruction"]):
            summary["status"] = "declined"
            summary["declined_leg"] = leg["leg"]
            _audit("handoff_declined", {"request_id": request_id, "leg": leg["leg"], "at": location})
            print(f"  DECLINED {request_id} leg {leg['leg']}: payload stays at {location}")
            return summary
        reply = send(
            leg["zone"],
            {
                "action": "execute",
                "instruction": leg["instruction"],
                "policy_provider": "mock",
                "robot_name": leg["robot"],
                "n_steps": n_steps,
            },
        )
        result = reply.get("result") if isinstance(reply, dict) else None
        if not (isinstance(result, dict) and result.get("status") == "success"):
            summary["failed_leg"] = leg["leg"]
            summary["reply"] = str(reply)[:300]
            _audit("handoff_aborted", {"request_id": request_id, "leg": leg["leg"], "at": location})
            print(f"  ABORT {request_id} leg {leg['leg']}: payload stranded at {location}")
            return summary
        location = leg["dst"]
        summary["at"] = location
        summary["legs_done"].append(leg["leg"])
        _audit(
            "handoff_leg_done",
            {"request_id": request_id, "leg": leg["leg"], "zone": leg["zone"], "robot": leg["robot"], "at": location},
        )
        print(f"  leg {leg['leg']}: {leg['zone']}/{leg['robot']} done; payload at {location}")
        if plan["handoff"] is not None and location == plan["handoff"]["location"]:
            _audit("handoff_custody", {"request_id": request_id, **plan["handoff"]})
            print(f"  custody: {plan['handoff']['from_zone']} -> {plan['handoff']['to_zone']} at {location}")
    summary["status"] = "done"
    _audit("handoff_complete", {"request_id": request_id, "at": location})
    print(f"  DONE {request_id}: payload at {location}")
    return summary


def make_hitl_gate() -> Callable[[str, str, str], bool]:
    """Human-in-the-loop gate applied BEFORE any leg reaches a zone.

    Interactive by default; ``STRANDS_MESH_HITL_ACTIONS=none`` opts out for
    CI/smoke runs (same env contract as the robot_mesh tool, and just as
    loud about it). A decline is a normal outcome, not an error.
    """
    if os.environ.get("STRANDS_MESH_HITL_ACTIONS", "").strip().lower() == "none":

        def auto_approve(action: str, zone: str, instruction: str) -> bool:
            print(f"  [HITL] auto-approved {action} -> {zone} (STRANDS_MESH_HITL_ACTIONS=none): {instruction!r}")
            return True

        return auto_approve

    def prompt_operator(action: str, zone: str, instruction: str) -> bool:
        answer = input(f"  [HITL] approve {action} -> {zone}: {instruction!r} [y/N] ")
        return answer.strip().lower() in {"y", "yes", "approve", "approved"}

    return prompt_operator


def make_loopback_send() -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Dry-run transport seam: prints each dispatch and reports success."""

    def send(zone: str, cmd: dict[str, Any]) -> dict[str, Any]:
        print(f"  [dry-run] {zone} <- {cmd['robot_name']}: {cmd['instruction']!r}")
        return {"type": "response", "result": {"status": "success", "detail": "dry-run loopback"}}

    return send


class _Coordinator:
    """Minimal mesh owner for the coordinator peer (it holds no robot)."""

    tool_name_str = COORDINATOR_ID


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float, what: str, poll_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    raise RuntimeError(f"timed out after {timeout_s:.0f}s waiting for {what}")


def _build_live_zones() -> tuple[Any, Callable[[], None]]:
    """One sim world per zone, each joined as that zone's mesh peer.

    Returns ``(coordinator_mesh, cleanup)``. The coordinator can only
    address zones: each zone peer owns its robots, and the wire dispatch on
    the zone side resolves ``robot_name`` within that zone's world.
    """
    from strands_robots.mesh import init_mesh
    from strands_robots.simulation import Simulation

    sims: list[Any] = []
    meshes: list[Any] = []
    coordinator: Any | None = None

    def cleanup() -> None:
        if coordinator is not None and coordinator.alive:
            coordinator.stop()
        for mesh in meshes:
            if mesh.alive:
                mesh.stop()
        for sim in sims:
            sim.destroy()

    try:
        for zone_id, zone in ZONES.items():
            sim = Simulation()
            sims.append(sim)
            result = sim.create_world()
            if result.get("status") != "success":
                raise RuntimeError(f"create_world({zone_id}) failed: {result}")
            for manifest in zone["manifests"]:
                robot_id = manifest["robot"]
                result = sim.add_robot(name=robot_id, data_config=ROBOT_EMBODIMENT[robot_id])
                if result.get("status") != "success":
                    raise RuntimeError(f"add_robot({robot_id}) failed: {result}")
            mesh = init_mesh(sim, peer_id=zone_id, peer_type="sim")
            if mesh is None:
                raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
            # A peer whose mesh did not start is not a slow peer: it publishes no
            # presence and discovers none, so the presence wait below can only
            # expire.  Refuse here, where the cause is still known.
            if not mesh.alive:
                raise RuntimeError(
                    f"mesh did not start for peer {zone_id!r} (mesh.alive is False): install the mesh "
                    'extra with pip install "strands-robots[mesh]", or rerun with --dry-run'
                )
            sim.mesh, sim.peer_id = mesh, mesh.peer_id
            meshes.append(mesh)
        coordinator = init_mesh(_Coordinator(), peer_id=COORDINATOR_ID)
        if coordinator is None:
            raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
        if not coordinator.alive:
            raise RuntimeError(
                f"mesh did not start for peer {COORDINATOR_ID!r} (mesh.alive is False): install the mesh "
                'extra with pip install "strands-robots[mesh]", or rerun with --dry-run'
            )
    except BaseException:
        cleanup()
        raise
    return coordinator, cleanup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-steps", type=int, default=25, help="policy steps per dispatched leg (live mode)")
    parser.add_argument("--dry-run", action="store_true", help="no simulator, no mesh: loopback transport seam")
    args = parser.parse_args(argv)
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be a positive number of policy steps")

    for zone_id, zone in ZONES.items():
        robots = ", ".join(m["robot"] for m in zone["manifests"])
        print(f"{zone_id}: locations {'/'.join(zone['locations'])}, robots [{robots}]")

    approve = make_hitl_gate()
    coordinator, cleanup = (None, None)
    if args.dry_run:
        send = make_loopback_send()
    else:
        coordinator, cleanup = _build_live_zones()
        _wait_for(
            lambda: all(zone_id in coordinator.peers_by_id for zone_id in ZONES),
            timeout_s=15.0,
            what="presence discovery of both zone peers",
        )
        coordinator_mesh = coordinator

        def send(zone: str, cmd: dict[str, Any]) -> dict[str, Any]:
            return coordinator_mesh.send(zone, cmd, timeout=60.0)

    summaries = []
    try:
        for request in REQUESTS:
            print(f"\nrequest {request['request_id']}: {request['src']} -> {request['dst']}")
            plan = plan_request(request, ZONES, DOCKS)
            summaries.append(execute_handoff(plan, approve, send, n_steps=args.n_steps))
    finally:
        if cleanup is not None:
            cleanup()

    print(f"\nsummary: {summaries}")
    outcomes = {s["request_id"]: s["status"] for s in summaries}
    if outcomes.get("X-01") == "done" and len(summaries[0]["legs_done"]) != 2:
        raise RuntimeError(f"cross-zone request did not execute as a two-leg handoff: {summaries[0]}")
    if outcomes.get("X-03") not in (None, "refused"):
        raise RuntimeError(f"expected the unroutable request to be refused with a reason: {summaries[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
