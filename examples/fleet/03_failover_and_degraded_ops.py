#!/usr/bin/env python3
"""Peer-loss reassignment and dispatcher-down safety for a robot fleet.

Goal: Show the two failure modes a fleet orchestrator must survive, on the
primitives the mesh already ships. Part 1 - robot failure mid-task: a robot's
heartbeat dies mid-transport, the orchestrator observes the presence timeout
(``PEER_TIMEOUT``, strands_robots.mesh.session), closes that robot's rollout
bookkeeping, and re-dispatches the remaining legs to a capable peer - chosen
by the shared ``capabilities.py`` hard-constraint filter (capability match,
never name match). Part 2 - dispatcher failure: the orchestrator peer dies,
the robots' local control loops keep answering, and ``emergency_stop`` still
propagates peer to peer over ``strands/safety/estop`` with the dispatcher
dead - safety does not depend on the dispatcher. The orchestrator then
restarts and re-syncs its state from presence plus the signed audit log.

Dependencies: pip install "strands-robots[sim-mujoco,mesh]"
              (--dry-run needs only the base package: no simulator, no Zenoh)
Expected output: a 4-leg transport that loses its robot after leg 1 and
                 completes on the failover robot; then a dispatcher outage
                 during which robots still answer status, an estop engages
                 every surviving peer's lockout, and a restarted orchestrator
                 reconstructs the whole story from presence + the audit log.
Runtime: ~5 seconds with --dry-run; under ~90 seconds live (the presence
         timeout is honoured in full, twice).

Note: The live estop drill leaves the surviving robots in safety lockout by
      design; resuming needs the operator override code
      (STRANDS_MESH_OVERRIDE_CODE). Set STRANDS_MESH_AUDIT_PSK to sign audit
      records and STRANDS_MESH_AUDIT_DIR to relocate the log.
      Set STRANDS_MESH_LOCAL_DEV=1 (defaulted below) to skip TLS locally.

Part of the fleet suite (epic #2179); the shared manifest schema lives in
``capabilities.py``. The read-only Rerun fleet dashboard (#2181) attaches to
the same mesh and shows the peer loss and the recovery live once it lands.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_FLEET_DIR = Path(__file__).resolve().parent
if str(_FLEET_DIR) not in sys.path:
    # examples/ is not an installed package; the shared schema module lives
    # next to this script.
    sys.path.insert(0, str(_FLEET_DIR))

from capabilities import (  # noqa: E402
    CapabilityManifest,
    StepRequirement,
    feasible_robots,
    manifest_from_dict,
)

from strands_robots.mesh.audit import log_safety_event, read_audit_log, verify_audit_integrity  # noqa: E402
from strands_robots.mesh.session import PEER_TIMEOUT  # noqa: E402

ORCHESTRATOR_ID = "fleet-orchestrator"

# The fleet: one site, three heterogeneous robots. The quadruped and the
# wheeled base can both transport a tote; the arm cannot - it exists so the
# reassignment below is demonstrably a capability match, not a name match.
FLEET_MANIFESTS: list[dict[str, Any]] = [
    {
        "robot": "go2-a1",
        "site": "site-a",
        "skills": [{"name": "transport", "payload_kg": 5.0, "fixture": "tote_clamp", "zones": ["stock", "etch"]}],
    },
    {
        "robot": "lekiwi-a1",
        "site": "site-a",
        "skills": [
            {"name": "transport", "payload_kg": 8.0, "fixture": "tote_clamp", "zones": ["stock", "etch", "litho"]}
        ],
    },
    {
        "robot": "so101-a1",
        "site": "site-a",
        "skills": [{"name": "handle", "payload_kg": 0.5, "fixture": "smif_pod", "zones": ["etch"]}],
    },
]

# Manifest robot id -> registry embodiment, used only by the live sim path.
ROBOT_EMBODIMENT = {"go2-a1": "go2", "lekiwi-a1": "lekiwi", "so101-a1": "so101"}

# The transport task, split into legs so a mid-transport loss leaves a
# well-defined remainder to re-dispatch. Both transport robots satisfy every
# hard constraint; ``feasible_robots`` sorts by name, so ``go2-a1`` is the
# deterministic first pick - and the robot this example kills mid-task.
TRANSPORT_TASK: dict[str, Any] = {
    "task_id": "T-42",
    "skill": "transport",
    "site": "site-a",
    "payload_kg": 4.0,
    "fixture": "tote_clamp",
    "zones": ("stock", "etch"),
    "legs": 4,
}


def _audit(event: str, payload: dict[str, Any]) -> None:
    log_safety_event(event, ORCHESTRATOR_ID, payload)


def _task_requirement(task: dict[str, Any]) -> StepRequirement:
    return StepRequirement(
        skill=task["skill"],
        site=task["site"],
        payload_kg=task["payload_kg"],
        fixture=task["fixture"],
        zones=tuple(task["zones"]),
    )


def _leg_instruction(task: dict[str, Any], leg: int) -> str:
    return (
        f"task {task['task_id']} leg {leg}/{task['legs']}: {task['skill']} "
        f"{task['payload_kg']} kg {task['fixture']} in "
        f"{'/'.join((task['site'], *task['zones']))}"
    )


def peer_is_lost(peer: dict[str, Any] | None, presence_timeout: float = PEER_TIMEOUT) -> bool:
    """Presence-timeout verdict for one peer snapshot.

    A peer is lost when it is absent from the registry (already pruned by a
    heartbeat tick) or when its last heartbeat is older than the presence
    timeout. A malformed ``age`` counts as lost - a peer whose liveness
    cannot be read must never be treated as alive.
    """
    if peer is None:
        return True
    age = peer.get("age")
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        return True
    return age > presence_timeout


def confirm_peer_loss(
    robot: str,
    get_peer: Callable[[str], dict[str, Any] | None],
    *,
    presence_timeout: float = PEER_TIMEOUT,
    poll_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Distinguish a lost peer from an alive-but-failing one.

    Called after a leg reply fails: poll presence for one full presence-
    timeout window. ``True`` means the peer's heartbeat is confirmably gone
    (fail over); ``False`` means the peer kept heartbeating through the whole
    window, so the failed reply was the robot misbehaving while alive - a
    structured task failure, never a failover.
    """
    deadline = clock() + presence_timeout + 2.0 * poll_s
    while True:
        if peer_is_lost(get_peer(robot), presence_timeout):
            return True
        if clock() >= deadline:
            return False
        sleep(poll_s)


def run_task_with_failover(
    task: dict[str, Any],
    manifests: Sequence[CapabilityManifest],
    get_peer: Callable[[str], dict[str, Any] | None],
    send: Callable[[str, str, int], dict[str, Any]],
    *,
    presence_timeout: float = PEER_TIMEOUT,
    n_steps: int = 25,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_leg_done: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Part 1 core: dispatch a multi-leg task, failing over on peer loss.

    ``get_peer(robot)`` is the presence seam (a mesh peer dict with an
    ``age`` field, or ``None`` once pruned); ``send(robot, instruction,
    n_steps)`` is the transport seam - the live path wires ``mesh.send``, the
    smoke test and ``--dry-run`` wire stubs. Before every leg the executor's
    presence is checked; a leg whose reply fails is confirmed against
    presence before any verdict. On a confirmed loss the lost robot's rollout
    book entry is closed (``interrupted``), the remaining legs - including
    the unconfirmed one, so delivery is at-least-once - are re-dispatched to
    a peer chosen by re-running the capability filter over the surviving
    manifests. No feasible survivor is a structured failure, never a guess.

    Returns a summary: ``status`` (``done`` / ``failed``), the rollout
    ``book`` (one entry per assignment), ``reassignments``, and ``failure``.
    """
    req = _task_requirement(task)
    task_id = task["task_id"]
    summary: dict[str, Any] = {
        "task_id": task_id,
        "status": "failed",
        "book": [],
        "reassignments": [],
        "failure": None,
    }

    excluded: set[str] = set()

    def assign() -> dict[str, Any] | None:
        surviving = [m for m in manifests if m.robot not in excluded]
        feasible, rejections = feasible_robots(surviving, req)
        if not feasible:
            return {"code": "no_feasible_robot", "rejections": rejections, "excluded": sorted(excluded)}
        entry = {"robot": feasible[0].robot, "legs_done": [], "status": "running"}
        summary["book"].append(entry)
        return None

    failure = assign()
    if failure is not None:
        summary["failure"] = failure
        _audit("task_failed", {"task_id": task_id, "reason": failure})
        print(f"  FAIL {task_id}: {failure['code']}")
        return summary

    entry = summary["book"][-1]
    _audit("task_dispatch", {"task_id": task_id, "robot": entry["robot"], "legs": task["legs"]})
    print(f"  task {task_id}: {task['legs']} legs -> {entry['robot']}")

    def fail_over(at_leg: int) -> bool:
        """Close the lost robot's book entry and reassign; False = no survivor."""
        lost = entry["robot"]
        entry["status"] = "interrupted"
        excluded.add(lost)
        _audit("peer_lost", {"task_id": task_id, "robot": lost, "at_leg": at_leg, "legs_done": entry["legs_done"]})
        print(f"  peer lost: {lost} (presence timeout at leg {at_leg}); rollout bookkeeping closed")
        failure = assign()
        if failure is not None:
            failure["code"] = "no_feasible_robot_after_loss"
            summary["failure"] = failure
            _audit("task_failed", {"task_id": task_id, "reason": failure})
            print(f"  FAIL {task_id}: {failure['code']}")
            return False
        _audit(
            "task_reassigned",
            {"task_id": task_id, "from": lost, "to": summary["book"][-1]["robot"], "next_leg": at_leg},
        )
        summary["reassignments"].append({"from": lost, "to": summary["book"][-1]["robot"], "at_leg": at_leg})
        print(f"  task {task_id}: legs {at_leg}..{task['legs']} reassigned -> {summary['book'][-1]['robot']}")
        return True

    leg = 1
    while leg <= task["legs"]:
        entry = summary["book"][-1]
        robot = entry["robot"]
        if peer_is_lost(get_peer(robot), presence_timeout):
            if not fail_over(leg):
                return summary
            continue
        reply = send(robot, _leg_instruction(task, leg), n_steps)
        result = reply.get("result") if isinstance(reply, dict) else None
        if isinstance(result, dict) and result.get("status") == "success":
            entry["legs_done"].append(leg)
            _audit("task_leg_done", {"task_id": task_id, "robot": robot, "leg": leg})
            print(f"  leg {leg}/{task['legs']}: done ({robot})")
            if on_leg_done is not None:
                on_leg_done(robot, leg)
            leg += 1
            continue
        # The leg reply failed. Peer loss and a misbehaving-but-alive robot
        # need opposite verdicts, so consult presence before either.
        if confirm_peer_loss(robot, get_peer, presence_timeout=presence_timeout, clock=clock, sleep=sleep):
            if not fail_over(leg):
                return summary
            continue
        failure = {"code": "dispatch_failed", "robot": robot, "leg": leg, "reply": str(reply)[:300]}
        entry["status"] = "failed"
        summary["failure"] = failure
        _audit("task_failed", {"task_id": task_id, "reason": failure})
        print(f"  FAIL {task_id}: {failure['code']} ({robot}, leg {leg})")
        return summary

    entry["status"] = "done"
    summary["status"] = "done"
    _audit("task_completed", {"task_id": task_id, "book": summary["book"]})
    print(f"  DONE {task_id}")
    return summary


# Part 2 -- dispatcher failure: degraded operations and re-sync.


def assert_degraded_safety(
    probe: Callable[[str, dict[str, Any]], dict[str, Any]],
    robots: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Assert every robot is locked out but alive after a dispatcher-down estop.

    ``probe(robot, cmd)`` sends one command peer to peer (the live path wires
    a surviving robot peer's ``mesh.send`` - the dispatcher is dead). For each
    robot: ``status`` must still answer (the local control loop survived the
    dispatcher) and ``execute`` must be refused (the estop lockout engaged
    without the dispatcher's help). Raises ``RuntimeError`` naming the first
    robot that fails either check - the claim is asserted, never narrated.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    for robot in robots:
        status_reply = probe(robot, {"action": "status"})
        if not isinstance(status_reply, dict) or status_reply.get("type") != "response":
            raise RuntimeError(f"degraded-ops check failed: {robot} did not answer status: {status_reply!r}")
        exec_reply = probe(
            robot,
            # A fully valid command on purpose: a probe the *sender-side*
            # validator rejects would never reach the peer, and the refusal
            # being asserted here is the receiver's lockout, nothing else.
            {"action": "execute", "instruction": "probe: must be refused", "policy_provider": "mock", "n_steps": 1},
        )
        refused = isinstance(exec_reply, dict) and exec_reply.get("type") == "error"
        if not refused:
            raise RuntimeError(f"degraded-ops check failed: {robot} accepted execute during lockout: {exec_reply!r}")
        verdicts[robot] = {"status_answered": True, "execute_refused": True}
        print(f"  {robot}: status answered, execute refused (lockout holds)")
    return verdicts


def resync_after_restart(peers: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Part 2 core: rebuild orchestrator state from presence + the audit log.

    A restarted orchestrator holds no memory of the outage; everything it
    needs is in the two places that survived it. ``peers`` is the presence
    snapshot (who is on the mesh right now); ``records`` is the audit trail
    (``read_audit_log(since=...)``). Returns the reconstruction: peers alive
    now, per-peer lockout state derived from the estop/resume safety events,
    the last estop issuer, and the per-task event chains.
    """
    alive = sorted(str(p["peer_id"]) for p in peers if isinstance(p.get("peer_id"), str))
    lockouts: dict[str, bool] = {}
    estop_issuer: str | None = None
    tasks: dict[str, list[str]] = {}
    for record in records:
        event = record.get("event")
        peer_id = record.get("peer_id")
        payload = record.get("payload")
        if not isinstance(event, str) or not isinstance(peer_id, str):
            continue
        if event == "emergency_stop":
            lockouts[peer_id] = True
            estop_issuer = peer_id
        elif event == "remote_estop_engaged":
            lockouts[peer_id] = True
        elif event in ("resume_ok", "remote_resume_applied"):
            lockouts[peer_id] = False
        if isinstance(payload, dict) and isinstance(payload.get("task_id"), str):
            tasks.setdefault(payload["task_id"], []).append(event)
    return {"alive_peers": alive, "lockouts": lockouts, "estop_issuer": estop_issuer, "tasks": tasks}


# Dry-run seams: a scripted fleet, no simulator, no Zenoh.


class ScriptedFleet:
    """Presence table + loopback transport with a scriptable peer death.

    ``get_peer`` / ``send`` mirror the live seams exactly: a live peer
    answers with a fresh presence dict and a success reply; a killed peer
    disappears from presence and its in-flight leg times out - the same
    observable sequence a real heartbeat death produces.
    """

    def __init__(self, robots: Sequence[str]) -> None:
        self._alive = set(robots)

    def kill(self, robot: str) -> None:
        """Kill *robot*'s heartbeat: gone from presence, deaf on the wire."""
        self._alive.discard(robot)
        print(f"  [scripted] {robot} heartbeat killed")

    def get_peer(self, robot: str) -> dict[str, Any] | None:
        if robot not in self._alive:
            return None
        return {"peer_id": robot, "type": "sim", "age": 0.5}

    def send(self, robot: str, instruction: str, n_steps: int) -> dict[str, Any]:
        if robot not in self._alive:
            return {"status": "timeout"}
        print(f"  [dry-run] {robot} <- {instruction!r} ({n_steps} steps)")
        return {"type": "response", "result": {"status": "success", "detail": "dry-run loopback"}}


class _Orchestrator:
    """Minimal mesh owner for the orchestrator peer (it only sends)."""

    tool_name_str = ORCHESTRATOR_ID


def _build_live_fleet() -> tuple[dict[str, Any], Any, Callable[[], None]]:
    """Bring up one sim world per robot, each as its own mesh peer.

    Returns ``(robot_meshes, orchestrator_mesh, cleanup)``. Per-robot peers
    (rather than 05's single sim peer) are the point of this example: a
    robot's death must be a *peer* death the presence layer can observe.
    """
    from strands_robots.mesh import init_mesh
    from strands_robots.simulation import Simulation

    sims: list[Any] = []
    meshes: dict[str, Any] = {}
    orch_mesh: Any | None = None

    def cleanup() -> None:
        if orch_mesh is not None and orch_mesh.alive:
            orch_mesh.stop()
        for mesh in meshes.values():
            if mesh.alive:
                mesh.stop()
        for sim in sims:
            sim.destroy()

    try:
        for robot_id, embodiment in ROBOT_EMBODIMENT.items():
            sim = Simulation()
            sims.append(sim)
            result = sim.create_world()
            if result.get("status") != "success":
                raise RuntimeError(f"create_world({robot_id}) failed: {result}")
            result = sim.add_robot(name=robot_id, data_config=embodiment)
            if result.get("status") != "success":
                raise RuntimeError(f"add_robot({robot_id}) failed: {result}")
            mesh = init_mesh(sim, peer_id=robot_id, peer_type="sim")
            if mesh is None:
                raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
            sim.mesh, sim.peer_id = mesh, mesh.peer_id
            meshes[robot_id] = mesh
        orch_mesh = init_mesh(_Orchestrator(), peer_id=ORCHESTRATOR_ID)
        if orch_mesh is None:
            raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
    except BaseException:
        cleanup()
        raise
    return meshes, orch_mesh, cleanup


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float, what: str, poll_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    raise RuntimeError(f"timed out after {timeout_s:.0f}s waiting for {what}")


def _run_live(n_steps: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Live choreography: sim + mesh, both failure drills, then the re-sync."""
    from strands_robots.mesh import init_mesh

    run_start = time.time()
    meshes, orch_mesh, cleanup = _build_live_fleet()
    manifests = [manifest_from_dict(m) for m in FLEET_MANIFESTS]
    try:
        _wait_for(
            lambda: all(robot in orch_mesh.peers_by_id for robot in ROBOT_EMBODIMENT),
            timeout_s=15.0,
            what="presence discovery of all robots",
        )

        print("\npart 1 - robot failure mid-task")

        def kill_first_executor_after_leg_1(robot: str, leg: int) -> None:
            if leg == 1 and meshes[robot].alive:
                meshes[robot].stop()
                print(f"  {robot}: heartbeat killed mid-transport")

        def send_leg(robot: str, instruction: str, steps: int) -> dict[str, Any]:
            return orch_mesh.send(
                robot,
                {
                    "action": "execute",
                    "instruction": instruction,
                    "policy_provider": "mock",
                    "robot_name": robot,
                    "n_steps": steps,
                },
                timeout=30.0,
            )

        failover_summary = run_task_with_failover(
            TRANSPORT_TASK,
            manifests,
            orch_mesh.get_peer,
            send_leg,
            n_steps=n_steps,
            on_leg_done=kill_first_executor_after_leg_1,
        )
        if failover_summary["status"] != "done" or not failover_summary["reassignments"]:
            raise RuntimeError(f"failover drill did not complete via reassignment: {failover_summary}")

        print("\npart 2 - dispatcher failure")
        orch_mesh.stop()
        survivors = [robot for robot, mesh in meshes.items() if mesh.alive]
        prober, others = survivors[0], survivors[1:]
        _wait_for(
            lambda: peer_is_lost(meshes[prober].get_peer(ORCHESTRATOR_ID)),
            timeout_s=PEER_TIMEOUT + 10.0,
            what="the dispatcher to drop out of presence",
        )
        print(f"  {ORCHESTRATOR_ID}: dead (presence timeout); robots keep running")
        for robot in others:
            reply = meshes[prober].send(robot, {"action": "status"}, timeout=10.0)
            if reply.get("type") != "response":
                raise RuntimeError(f"{robot} stopped answering after the dispatcher died: {reply!r}")
            print(f"  {robot}: local control loop alive (status answered, no dispatcher)")

        print(f"  {prober}: issuing emergency_stop with the dispatcher dead")
        meshes[prober].emergency_stop()
        assert_degraded_safety(lambda robot, cmd: meshes[prober].send(robot, cmd, timeout=10.0), others)

        print("\n  restarting the orchestrator")
        orch_mesh2 = init_mesh(_Orchestrator(), peer_id=ORCHESTRATOR_ID)
        if orch_mesh2 is None:
            raise RuntimeError("mesh is disabled (STRANDS_MESH=0)")
        try:
            _wait_for(
                lambda: all(robot in orch_mesh2.peers_by_id for robot in survivors),
                timeout_s=15.0,
                what="presence discovery after the orchestrator restart",
            )
            records = read_audit_log(since=run_start - 1.0)
            resync = resync_after_restart(orch_mesh2.peers, records)
        finally:
            orch_mesh2.stop()
        if not any(resync["lockouts"].get(robot) for robot in others):
            raise RuntimeError(f"re-sync did not recover the estop lockout from the audit log: {resync}")
        return failover_summary, resync, records
    finally:
        cleanup()


def _run_dry() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Dry-run choreography: scripted presence + loopback transport.

    Part 1 runs the full failover core. Part 2 runs the re-sync core against
    this run's audit records and the post-outage presence snapshot; the live
    peer-to-peer estop drill needs a real mesh (the smoke test asserts that
    propagation through the real safety handler, dispatcher-free).
    """
    run_start = time.time()
    manifests = [manifest_from_dict(m) for m in FLEET_MANIFESTS]
    fleet = ScriptedFleet(list(ROBOT_EMBODIMENT))

    print("\npart 1 - robot failure mid-task")
    failover_summary = run_task_with_failover(
        TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        sleep=lambda _s: None,
        on_leg_done=lambda robot, leg: fleet.kill(robot) if leg == 1 else None,
    )

    print("\npart 2 - dispatcher restart re-sync (estop drill is live-mode only)")
    peers_now = [fleet.get_peer(robot) for robot in ROBOT_EMBODIMENT]
    records = read_audit_log(since=run_start - 1.0)
    resync = resync_after_restart([p for p in peers_now if p is not None], records)
    return failover_summary, resync, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-steps", type=int, default=25, help="policy steps per dispatched leg (live mode)")
    parser.add_argument("--dry-run", action="store_true", help="no simulator, no mesh: scripted presence + loopback")
    args = parser.parse_args(argv)

    manifests = [manifest_from_dict(m) for m in FLEET_MANIFESTS]
    print(f"fleet: {', '.join(f'{m.robot}@{m.site}' for m in manifests)}")

    if args.dry_run:
        failover_summary, resync, records = _run_dry()
    else:
        failover_summary, resync, records = _run_live(args.n_steps)

    print(f"\nfailover summary: {failover_summary}")
    print("re-sync after restart (presence + audit log):")
    print(f"  alive now: {resync['alive_peers']}")
    print(f"  lockouts:  {resync['lockouts']} (estop issuer: {resync['estop_issuer']})")
    for task_id, chain in sorted(resync["tasks"].items()):
        print(f"  {task_id}: {' -> '.join(chain)}")
    # Attest the records this run reported on, not the developer's whole audit
    # log: an unscoped verify_audit_integrity() would count (and fail on) prior
    # unsigned history that the table above never shows.
    integrity = verify_audit_integrity(records)
    print(f"audit integrity: ok={integrity['ok']} (signed={integrity['signed']}/{integrity['total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
