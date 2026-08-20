#!/usr/bin/env python3
"""Structured work-order ingress mapped onto per-site capability manifests.

Goal: Show the two-stage translation from business vocabulary to robot
vocabulary. Work orders arrive on a JSONL file queue speaking only business
language ({order_id, material, operation, qty, from, to, due}). Stage one is
deterministic: schema validation plus hard-constraint filtering (site, payload,
fixture, zones) against per-robot capability manifests, with an explicit NACK
back onto the queue when no feasible robot exists - a machine-readable reason,
never a silent drop. Stage two is the agent's territory: choosing AMONG the
capable robots and sequencing multi-step orders. The agent can never invent a
capability - a choice outside the feasible set is refused. Dispatch goes over
the mesh behind a human-in-the-loop gate, the order_id is threaded through the
signed audit log end to end, and a structured completion/failure event is
emitted back onto the queue.

The example's boundary is the JSON schema, not the transport: a production
ingress can front the same schema with a queue service or an API. The file
queue keeps this suite air-gapped by design.

Dependencies: pip install "strands-robots[sim-mujoco,mesh]"
              (--dry-run needs only the base package: no simulator, no Zenoh)
Expected output: three orders dispatched (one per site on different robot
                 types, one sequenced across two robots), one order NACKed
                 with a machine-readable reason, and an audit-log
                 reconstruction of order -> dispatch -> action -> completion.
Runtime: ~5 seconds with --dry-run; under ~60 seconds live (cached assets).

Note: Set STRANDS_MESH_HITL_ACTIONS=none to auto-approve dispatches (CI /
      smoke mode; logged loudly). Set STRANDS_MESH_AUDIT_PSK to sign audit
      records and STRANDS_MESH_AUDIT_DIR to relocate the log.
      Set STRANDS_MESH_LOCAL_DEV=1 (defaulted below) to skip TLS locally.

Part of the fleet suite (epic #2179); the shared manifest schema lives in
``capabilities.py`` (also consumed by example 01, #2180).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import random
import re
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
    CapabilityManifest,
    StepRequirement,
    feasible_robots,
    manifest_from_dict,
)

from strands_robots.mesh.audit import (  # noqa: E402
    log_safety_event,
    read_audit_log,
    verify_audit_integrity,
)

DISPATCHER_ID = "fleet-dispatcher"

# ---------------------------------------------------------------------------
# Deterministic business -> robot translation tables (stage one).
# Business vocabulary is the queue's whole surface; everything physical is
# derived here, in code review-able tables, never by the agent.
# ---------------------------------------------------------------------------

MATERIAL_UNIT_MASS_KG = {"wafer_lot": 0.35, "photomask": 0.2, "tote": 4.0}
MATERIAL_FIXTURE = {"wafer_lot": "smif_pod", "photomask": "reticle_pod", "tote": "tote_clamp"}

# operation -> ordered skill steps. "process" is the multi-step case: stage
# the lot at the source tool, then move it on to the destination tool.
OPERATION_SKILLS: dict[str, tuple[str, ...]] = {
    "inspect": ("handle",),
    "transfer": ("transport",),
    "process": ("handle", "transport"),
}

REQUIRED_ORDER_FIELDS = ("order_id", "material", "operation", "qty", "from", "to", "due")

_ORDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LOCATION_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)/([A-Za-z0-9][A-Za-z0-9_.-]*)$")

# The fleet: per robot, per site - the formalization of the shared
# capabilities.py schema. In a deployment these manifests are the robots'
# own declarations; here they are in-example data.
FLEET_MANIFESTS: list[dict[str, Any]] = [
    {
        "robot": "so101-a1",
        "site": "site-a",
        "skills": [{"name": "handle", "payload_kg": 0.5, "fixture": "smif_pod", "zones": ["litho", "etch"]}],
    },
    {
        "robot": "lekiwi-a1",
        "site": "site-a",
        "skills": [
            {"name": "transport", "payload_kg": 8.0, "fixture": "smif_pod", "zones": ["litho", "etch", "stock"]},
            {"name": "transport", "payload_kg": 8.0, "fixture": "tote_clamp", "zones": ["litho", "etch", "stock"]},
        ],
    },
    {
        "robot": "go2-b1",
        "site": "site-b",
        "skills": [{"name": "transport", "payload_kg": 5.0, "fixture": "tote_clamp", "zones": ["receiving", "stock"]}],
    },
]

# Manifest robot id -> registry embodiment, used only by the live sim path.
ROBOT_EMBODIMENT = {"so101-a1": "so101", "lekiwi-a1": "lekiwi", "go2-b1": "go2"}
ROBOT_POSITION = {"so101-a1": [0.0, 0.0, 0.0], "lekiwi-a1": [0.8, 0.0, 0.0], "go2-b1": [0.0, 1.0, 0.0]}


class OrderError(ValueError):
    """A work order that fails schema validation.

    Carries the offending field so the NACK reason stays machine-readable.
    """

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(detail)
        self.field = field


def _parse_location(value: Any, field: str) -> tuple[str, str]:
    """Split ``"site-a/litho"`` into ``("site-a", "litho")`` or raise OrderError."""
    if not isinstance(value, str):
        raise OrderError(field, f"{field} must be a string 'site/zone' (got {type(value).__name__})")
    match = _LOCATION_RE.match(value)
    if not match:
        raise OrderError(field, f"{field} must match 'site/zone' with [A-Za-z0-9_.-] parts (got {value!r})")
    return match.group(1), match.group(2)


def validate_order(raw: Any) -> dict[str, Any]:
    """Stage 1a: schema validation. Business vocabulary in, normalized order out.

    Raises OrderError on any violation - an order that cannot be validated is
    NACKed by the caller, never repaired or partially accepted.
    """
    if not isinstance(raw, dict):
        raise OrderError("order", f"order must be a JSON object (got {type(raw).__name__})")
    missing = [f for f in REQUIRED_ORDER_FIELDS if f not in raw]
    if missing:
        raise OrderError(missing[0], f"order is missing required fields {missing}")
    unknown = sorted(set(raw) - set(REQUIRED_ORDER_FIELDS))
    if unknown:
        raise OrderError(unknown[0], f"order has unknown fields {unknown}; the schema is the boundary")
    order_id = raw["order_id"]
    if not isinstance(order_id, str) or not _ORDER_ID_RE.match(order_id):
        raise OrderError("order_id", f"order_id must match [A-Za-z0-9][A-Za-z0-9_.-]* (got {order_id!r})")
    material = raw["material"]
    if material not in MATERIAL_UNIT_MASS_KG:
        raise OrderError("material", f"unknown material {material!r}; known: {sorted(MATERIAL_UNIT_MASS_KG)}")
    operation = raw["operation"]
    if operation not in OPERATION_SKILLS:
        raise OrderError("operation", f"unknown operation {operation!r}; known: {sorted(OPERATION_SKILLS)}")
    qty = raw["qty"]
    if isinstance(qty, bool) or not isinstance(qty, int) or qty < 1:
        raise OrderError("qty", f"qty must be an integer >= 1 (got {qty!r})")
    from_site, from_zone = _parse_location(raw["from"], "from")
    to_site, to_zone = _parse_location(raw["to"], "to")
    if from_site != to_site:
        raise OrderError("to", f"cross-site orders are not supported (from {from_site!r}, to {to_site!r})")
    due = raw["due"]
    if not isinstance(due, str) or not due:
        raise OrderError("due", f"due must be a non-empty string (got {due!r})")
    return {
        "order_id": order_id,
        "material": material,
        "operation": operation,
        "qty": qty,
        "site": from_site,
        "from_zone": from_zone,
        "to_zone": to_zone,
        "due": due,
        "payload_kg": MATERIAL_UNIT_MASS_KG[material] * qty,
        "fixture": MATERIAL_FIXTURE[material],
    }


def plan_steps(order: dict[str, Any]) -> list[StepRequirement]:
    """Stage 1b: derive the ordered robot-vocabulary steps for one order.

    Deterministic: the operation names the skill sequence, the material names
    the fixture and (with qty) the payload, and the locations name the zones.
    """
    steps: list[StepRequirement] = []
    for skill in OPERATION_SKILLS[order["operation"]]:
        if skill == "transport":
            zones = tuple(dict.fromkeys((order["from_zone"], order["to_zone"])))
        else:
            zones = (order["from_zone"],)
        steps.append(
            StepRequirement(
                skill=skill,
                site=order["site"],
                payload_kg=order["payload_kg"],
                fixture=order["fixture"],
                zones=zones,
            )
        )
    return steps


def select_robot(feasible: list[CapabilityManifest], load: dict[str, int]) -> CapabilityManifest:
    """Deterministic stage-two default: least-loaded, robot name as tie-break."""
    return min(feasible, key=lambda m: (load.get(m.robot, 0), m.robot))


def guard_choice(
    choice: str,
    feasible: list[CapabilityManifest],
    fallback: CapabilityManifest,
) -> CapabilityManifest:
    """The capability floor under any chooser: an out-of-set pick is refused.

    Whatever selects among the capable robots - the deterministic default or
    an LLM agent - the choice is validated against the feasible set computed
    by stage one. A name outside it falls back to the deterministic pick, so
    a chooser can never invent a capability.
    """
    for manifest in feasible:
        if manifest.robot == choice:
            return manifest
    print(f"  [guard] choice {choice!r} is outside the feasible set {[m.robot for m in feasible]}; refused")
    return fallback


def make_agent_chooser() -> Callable[[dict[str, Any], StepRequirement, list[CapabilityManifest]], str]:
    """Optional stage two: a Strands agent chooses among the feasible robots.

    The agent sees only the feasible set and replies with one robot id; the
    reply still passes through :func:`guard_choice`. Any agent failure (no
    credentials, no network) falls back to the deterministic default - the
    example never depends on an LLM being reachable.
    """
    from strands import Agent

    system = (
        "You assign factory work-order steps to robots. Reply with EXACTLY one "
        "robot id copied verbatim from the candidate list. No prose."
    )
    agent = Agent(system_prompt=system, callback_handler=None)

    def choose(order: dict[str, Any], step: StepRequirement, feasible: list[CapabilityManifest]) -> str:
        names = [m.robot for m in feasible]
        prompt = (
            f"Order {order['order_id']}: step '{step.skill}' at {step.site} "
            f"(payload {step.payload_kg} kg, fixture {step.fixture}, zones {list(step.zones)}). "
            f"Candidates: {names}. Reply with one id."
        )
        try:
            reply = str(agent(prompt)).strip()
        except Exception as exc:  # noqa: BLE001 - the agent is optional decoration; an unreachable model must not strand the queue
            print(f"  [agent] unavailable ({type(exc).__name__}); using the deterministic selector")
            return ""
        return reply.split()[-1].strip("'\"`.,") if reply else ""

    return choose


def make_hitl_gate() -> Callable[[str, str, str], bool]:
    """Human-in-the-loop gate applied BEFORE any command goes on the wire.

    Interactive by default; ``STRANDS_MESH_HITL_ACTIONS=none`` opts out for
    CI/smoke runs (same env contract as the robot_mesh tool, and just as
    loud about it). A decline is a normal outcome, not an error: the order
    fails with a structured event and nothing is dispatched.
    """
    if os.environ.get("STRANDS_MESH_HITL_ACTIONS", "").strip().lower() == "none":

        def auto_approve(action: str, target: str, instruction: str) -> bool:
            print(f"  [HITL] auto-approved {action} -> {target} (STRANDS_MESH_HITL_ACTIONS=none): {instruction!r}")
            return True

        return auto_approve

    def prompt_operator(action: str, target: str, instruction: str) -> bool:
        answer = input(f"  [HITL] approve {action} -> {target}: {instruction!r} [y/N] ")
        return answer.strip().lower() in {"y", "yes", "approve", "approved"}

    return prompt_operator


def emit_event(events_path: Path, record: dict[str, Any]) -> None:
    """Append one structured event line back onto the file queue."""
    record = {"ts": time.time(), **record}
    with events_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _audit(event: str, payload: dict[str, Any]) -> None:
    log_safety_event(event, DISPATCHER_ID, payload)


def process_queue(
    orders_path: Path,
    events_path: Path,
    manifests: list[CapabilityManifest],
    send: Callable[[str, str, int], dict[str, Any]],
    approve: Callable[[str, str, str], bool],
    chooser: Callable[[dict[str, Any], StepRequirement, list[CapabilityManifest]], str] | None = None,
    n_steps: int = 25,
) -> dict[str, list[str]]:
    """Drain the ingress queue: validate, filter, choose, dispatch, close out.

    ``send(robot_name, instruction, n_steps)`` is the transport seam - the
    live path wires it to ``mesh.send``; ``--dry-run`` (and the smoke test)
    wires a loopback. Every order leaves a completion, failure, or NACK event
    on ``events_path`` and an order_id-threaded trail in the audit log; no
    outcome is silent.
    """
    summary: dict[str, list[str]] = {"completed": [], "failed": [], "nacked": []}
    load: dict[str, int] = {}

    def nack(order_ref: str | None, reason: dict[str, Any]) -> None:
        emit_event(events_path, {"event": "work_order_nacked", "order_id": order_ref, "reason": reason})
        _audit("work_order_nacked", {"order_id": order_ref, "reason": reason})
        summary["nacked"].append(order_ref or "<unidentified>")
        print(f"  NACK {order_ref or '<unidentified>'}: {reason['code']}")

    def fail(order_ref: str, reason: dict[str, Any]) -> None:
        emit_event(events_path, {"event": "work_order_failed", "order_id": order_ref, "reason": reason})
        _audit("work_order_failed", {"order_id": order_ref, "reason": reason})
        summary["failed"].append(order_ref)
        print(f"  FAIL {order_ref}: {reason['code']}")

    for lineno, line in enumerate(orders_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            # A line that is not JSON has no trustworthy order_id; it is still
            # NACKed onto the queue rather than dropped.
            nack(None, {"code": "invalid_json", "line": lineno, "detail": str(exc)})
            continue
        order_ref = raw.get("order_id") if isinstance(raw, dict) and isinstance(raw.get("order_id"), str) else None
        _audit("work_order_received", {"order_id": order_ref, "line": lineno})
        print(f"order {order_ref or '<unidentified>'} (line {lineno})")

        # Stage 1a: schema validation.
        try:
            order = validate_order(raw)
        except OrderError as exc:
            nack(order_ref, {"code": "invalid_order", "field": exc.field, "detail": str(exc)})
            continue
        order_id = order["order_id"]

        # Stage 1b + 1c: derive steps and hard-filter EVERY step before
        # dispatching any, so an infeasible later step never strands a
        # half-dispatched order.
        steps = plan_steps(order)
        per_step_feasible: list[list[CapabilityManifest]] = []
        infeasible: dict[str, Any] | None = None
        for index, step in enumerate(steps):
            feasible, rejections = feasible_robots(manifests, step)
            if not feasible:
                infeasible = {
                    "code": "no_feasible_robot",
                    "step_index": index,
                    "step": {
                        "skill": step.skill,
                        "site": step.site,
                        "payload_kg": step.payload_kg,
                        "fixture": step.fixture,
                        "zones": list(step.zones),
                    },
                    "rejections": rejections,
                }
                break
            per_step_feasible.append(feasible)
        if infeasible is not None:
            nack(order_id, infeasible)
            continue

        # Stage 2: choose among capable robots, sequence the steps, dispatch.
        dispatched: list[dict[str, str]] = []
        order_failed = False
        for index, (step, feasible) in enumerate(zip(steps, per_step_feasible, strict=True)):
            fallback = select_robot(feasible, load)
            if chooser is not None:
                robot = guard_choice(chooser(order, step, feasible), feasible, fallback)
            else:
                robot = fallback
            instruction = (
                f"work order {order_id} step {index + 1}/{len(steps)}: {step.skill} "
                f"{order['qty']}x {order['material']} ({step.payload_kg} kg, {step.fixture}) "
                f"in {'/'.join((step.site, *step.zones))}, due {order['due']}"
            )
            if not approve("execute", robot.robot, instruction):
                fail(order_id, {"code": "hitl_declined", "step_index": index, "robot": robot.robot})
                order_failed = True
                break
            _audit(
                "work_order_dispatch",
                {
                    "order_id": order_id,
                    "step_index": index,
                    "robot": robot.robot,
                    "site": step.site,
                    "skill": step.skill,
                },
            )
            reply = send(robot.robot, instruction, n_steps)
            result = reply.get("result") if isinstance(reply, dict) else None
            ok = isinstance(result, dict) and result.get("status") == "success"
            _audit(
                "work_order_action",
                {
                    "order_id": order_id,
                    "step_index": index,
                    "robot": robot.robot,
                    "status": "success" if ok else "error",
                    "detail": str(reply)[:300],
                },
            )
            if not ok:
                fail(
                    order_id,
                    {"code": "dispatch_failed", "step_index": index, "robot": robot.robot, "reply": str(reply)[:300]},
                )
                order_failed = True
                break
            load[robot.robot] = load.get(robot.robot, 0) + 1
            dispatched.append({"skill": step.skill, "robot": robot.robot, "site": step.site})
            print(f"  step {index + 1}/{len(steps)}: {step.skill} -> {robot.robot}")
        if order_failed:
            continue

        emit_event(events_path, {"event": "work_order_completed", "order_id": order_id, "steps": dispatched})
        _audit("work_order_completed", {"order_id": order_id, "steps": dispatched})
        summary["completed"].append(order_id)
        print(f"  DONE {order_id}")
    return summary


def reconstruct_audit_trail(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group this run's dispatcher audit records into per-order event chains."""
    chains: dict[str, list[str]] = {}
    for record in records:
        payload = record.get("payload")
        if record.get("peer_id") != DISPATCHER_ID or not isinstance(payload, dict):
            continue
        order_ref = payload.get("order_id") or "<unidentified>"
        chains.setdefault(order_ref, []).append(record.get("event", "?"))
    return chains


def make_loopback_send() -> Callable[[str, str, int], dict[str, Any]]:
    """Dry-run transport: prints the dispatch and reports success."""

    def send(robot_name: str, instruction: str, n_steps: int) -> dict[str, Any]:
        print(f"  [dry-run] {robot_name} <- {instruction!r} ({n_steps} steps)")
        return {"result": {"status": "success", "detail": "dry-run loopback"}}

    return send


class _Dispatcher:
    """Minimal mesh owner for the dispatcher peer (it only sends)."""

    tool_name_str = DISPATCHER_ID


def _build_live_transport() -> tuple[Callable[[str, str, int], dict[str, Any]], Callable[[], None]]:
    """Bring up the sim + two mesh peers and return (send, cleanup).

    One MuJoCo world hosts every robot in the manifests (heterogeneous
    embodiments; sites are logical). The dispatcher must be a SEPARATE peer:
    a peer cannot RPC its own peer_id in-process.
    """
    from strands_robots.mesh import init_mesh
    from strands_robots.simulation import Simulation

    sim = Simulation()
    result = sim.create_world()
    if result.get("status") != "success":
        sim.destroy()
        raise RuntimeError(f"create_world failed: {result}")
    for robot_id, embodiment in ROBOT_EMBODIMENT.items():
        result = sim.add_robot(name=robot_id, data_config=embodiment, position=ROBOT_POSITION[robot_id])
        if result.get("status") != "success":
            sim.destroy()
            raise RuntimeError(f"add_robot({robot_id}) failed: {result}")

    sim_mesh = init_mesh(sim, peer_id="fleet-sites", peer_type="sim")
    if sim_mesh is None:
        sim.destroy()
        raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
    # A peer whose mesh did not start answers every RPC with "mesh not running",
    # so without this the queue below runs to completion and reports each order
    # as a per-robot dispatch failure.  Refuse here, where the cause is known.
    if not sim_mesh.alive:
        sim.destroy()
        raise RuntimeError(
            "mesh did not start for peer 'fleet-sites' (mesh.alive is False): install the mesh "
            'extra with pip install "strands-robots[mesh]", or rerun with --dry-run'
        )
    sim.mesh, sim.peer_id = sim_mesh, sim_mesh.peer_id
    orch_mesh = init_mesh(_Dispatcher(), peer_id=DISPATCHER_ID)
    if orch_mesh is None:
        sim_mesh.stop()
        sim.destroy()
        raise RuntimeError("mesh is disabled (STRANDS_MESH=0); rerun with --dry-run")
    if not orch_mesh.alive:
        sim_mesh.stop()
        sim.destroy()
        raise RuntimeError(
            f"mesh did not start for peer {DISPATCHER_ID!r} (mesh.alive is False): install the mesh "
            'extra with pip install "strands-robots[mesh]", or rerun with --dry-run'
        )

    def send(robot_name: str, instruction: str, n_steps: int) -> dict[str, Any]:
        # "execute" runs the policy synchronously, so multi-step orders are
        # genuinely sequenced: the reply arrives when the step is done.
        return orch_mesh.send(
            "fleet-sites",
            {
                "action": "execute",
                "instruction": instruction,
                "policy_provider": "mock",
                "robot_name": robot_name,
                "n_steps": n_steps,
            },
            timeout=60.0,
        )

    def cleanup() -> None:
        orch_mesh.stop()
        sim_mesh.stop()
        sim.destroy()

    return send, cleanup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=Path, default=_FLEET_DIR / "work_orders.jsonl", help="ingress JSONL queue")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("work_order_events.jsonl"),
        help="outbound JSONL queue for completion/failure/NACK events",
    )
    parser.add_argument("--seed", type=int, default=42, help="world seed (live mode)")
    parser.add_argument("--n-steps", type=int, default=25, help="policy steps per dispatched skill (live mode)")
    parser.add_argument("--dry-run", action="store_true", help="no simulator, no mesh: loopback transport")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="let a Strands agent choose among the capable robots (falls back to the deterministic pick)",
    )
    args = parser.parse_args(argv)
    random.seed(args.seed)

    manifests = [manifest_from_dict(m) for m in FLEET_MANIFESTS]
    print(f"fleet: {', '.join(f'{m.robot}@{m.site}' for m in manifests)}")

    chooser = None
    if args.agent:
        try:
            chooser = make_agent_chooser()
        except Exception as exc:  # noqa: BLE001 - the agent is optional decoration; the deterministic path is authoritative
            print(f"agent chooser unavailable ({type(exc).__name__}: {exc}); using the deterministic selector")

    cleanup: Callable[[], None] | None = None
    if args.dry_run:
        send = make_loopback_send()
    else:
        send, cleanup = _build_live_transport()

    run_start = time.time()
    try:
        summary = process_queue(
            orders_path=args.orders,
            events_path=args.events,
            manifests=manifests,
            send=send,
            approve=make_hitl_gate(),
            chooser=chooser,
            n_steps=args.n_steps,
        )
    finally:
        if cleanup is not None:
            cleanup()

    print(f"\nsummary: {summary}")
    print(f"events queue: {args.events}")

    print("\naudit reconstruction (order -> dispatch -> action -> completion):")
    # One scoped read feeds both the trail and the verdict, so the integrity
    # line attests exactly the records shown - never the developer's whole
    # ~/.strands_robots/mesh_audit.jsonl, which is mostly other runs.
    records = read_audit_log(since=run_start - 1.0)
    for order_ref, chain in sorted(reconstruct_audit_trail(records).items()):
        print(f"  {order_ref}: {' -> '.join(chain)}")
    integrity = verify_audit_integrity(records)
    print(f"audit integrity: ok={integrity['ok']} (signed={integrity['signed']}/{integrity['total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
