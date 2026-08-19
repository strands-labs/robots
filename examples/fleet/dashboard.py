#!/usr/bin/env python3
"""Read-only fleet dashboard: a subscribe-only mesh peer rendered in Rerun.

Goal: One dashboard that serves every example in the fleet suite unchanged,
because it attaches to the MESH, not to a simulator or a backend (epic
#2179, decision D9). It joins as its own peer (``peer_type="dashboard"``),
subscribes to presence, health and safety topics, tails the signed audit
log, and renders two things: a fleet table (peer id, type, presence age,
current task, last safety event) and an event timeline (dispatch / estop /
resume / HITL decisions, as recorded in the audit trail).

READ-ONLY is enforced, not narrated: ``restrict_to_subscribe_only`` replaces
every command-capable method on the peer (``send`` / ``tell`` / ``broadcast``
/ ``emergency_stop`` / ``publish_step``) with a refusal, and confines raw
``publish`` to the peer's own ``strands/{peer_id}/...`` namespace - so a
command, an estop or a resume cannot be published from this peer by accident.
Two write paths are deliberately kept, because the peer's own mesh loops need
them: ``publish`` under that namespace (presence and health, so the dashboard
is visible as a peer) and ``publish_safety_event``, which the mesh's own
safety handlers call to record this peer's lockout transitions. So the
dashboard does append to the signed audit trail it renders - for its own
``peer_id`` only. This is a guard against misuse, not a security boundary:
the refusals are instance attributes, so code that deliberately reaches for
the class attribute is not contained by them (``restrict_to_subscribe_only``
itself does exactly that, via ``type(mesh).publish``, to keep the confined
path working). HITL approvals stay in the operator terminal; a write-capable
UI is an explicit epic non-goal.

Dependencies: pip install "strands-robots[mesh]"
              (rendering upgrades from terminal to Rerun when rerun-sdk is
              installed: pip install rerun-sdk)
Expected output: a live-updating fleet table plus an event timeline, in the
                 Rerun viewer when available and on the terminal otherwise.
Runtime: until Ctrl+C, or --duration seconds.

Note: Run any fleet example in another terminal (e.g.
      ``python examples/fleet/02_cross_zone_transport.py``) and this peer
      shows its zones, dispatches and handoffs live. Two processes need a
      discovery channel: multicast scouting is OFF by default, so set
      ``STRANDS_MESH_MULTICAST=true`` on BOTH processes (trusted networks
      only) or configure connect endpoints. Point STRANDS_MESH_AUDIT_DIR at
      the same directory the examples use so the audit tail reads the trail
      they write.

Part of the fleet suite (epic #2179, dashboard v1 per #2181).
"""

from __future__ import annotations

import os

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")

import argparse
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from strands_robots.mesh import init_mesh
from strands_robots.mesh.audit import read_audit_log
from strands_robots.utils import require_optional

DASHBOARD_ID = "fleet-dashboard"

# The subscribe surface: presence and health for the fleet table, the two
# fleet-wide safety topics plus per-peer safety events for the timeline.
SUBSCRIBE_TOPICS: tuple[str, ...] = (
    "strands/*/presence",
    "strands/*/health",
    "strands/*/safety/event",
    "strands/safety/estop",
    "strands/safety/resume",
)

# Command-capable Mesh methods a read-only peer must not hold. Every one of
# these is reachable only from caller code: the mesh's own loops never call
# them (``tell`` delegates to ``send`` and ``emergency_stop`` to
# ``broadcast``, and both callers are themselves refused), so removing them
# cannot break the peer.
#
# Two write-capable methods are deliberately absent, because the peer's own
# mesh loops DO call them and a refusal would break the peer rather than
# guard it:
#   ``publish``              - confined to the peer's own namespace instead
#                              (the presence/health loops publish there).
#   ``publish_safety_event`` - the mesh's own ``_on_safety_estop`` /
#                              ``_on_safety_resume`` handlers call it to
#                              record this peer's lockout transitions, two of
#                              them outside a try, and the guarded ones catch
#                              only (TypeError, ValueError, OSError). Adding
#                              it here raises inside a Zenoh subscription
#                              callback on the safety path.
_COMMAND_METHODS: tuple[str, ...] = ("send", "tell", "broadcast", "emergency_stop", "publish_step")


def restrict_to_subscribe_only(mesh: Any) -> Any:
    """Turn a started Mesh peer into a subscribe-only surface, in place.

    Every method in :data:`_COMMAND_METHODS` is replaced with a refusal that
    raises ``RuntimeError`` naming the dashboard's contract, and raw
    ``publish`` is confined to the peer's own ``strands/{peer_id}/...``
    namespace - which keeps the presence/health liveness loops honest (the
    dashboard is visible as a peer) while making every command topic
    (another peer's ``cmd``, ``broadcast``, ``safety/estop``,
    ``safety/resume``) unreachable. Returns the same mesh object.

    ``publish_safety_event`` is exempt and stays callable: the mesh's own
    safety handlers use it to record this peer's lockout transitions, so
    refusing it would raise on the safety path rather than guard anything
    (see the note on :data:`_COMMAND_METHODS`). It writes only this peer's
    own ``strands/{peer_id}/safety/event`` topic and audit records carrying
    this peer's own ``peer_id`` - it cannot name another peer - but that does
    mean a caller holding the restricted mesh can append to the signed audit
    trail this dashboard renders. Scoping that write is a mesh-side contract
    question, not something this example decides.
    """

    def _refuse_command(name: str) -> Callable[..., Any]:
        def refused(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(f"read-only dashboard: {name} is disabled on this peer (subscribe-only surface)")

        return refused

    for name in _COMMAND_METHODS:
        setattr(mesh, name, _refuse_command(name))

    own_prefix = f"strands/{mesh.peer_id}/"
    inner_publish = type(mesh).publish

    def confined_publish(key: str, payload: dict[str, Any]) -> None:
        if not key.startswith(own_prefix):
            raise RuntimeError(
                f"read-only dashboard: publish to {key!r} refused (only {own_prefix}* is permitted on this peer)"
            )
        inner_publish(mesh, key, payload)

    mesh.publish = confined_publish
    return mesh


def build_snapshot(
    peers: list[dict[str, Any]],
    health: dict[str, dict[str, Any]],
    safety: dict[str, str],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pure snapshot builder: mesh state in, render-ready rows out.

    ``peers`` is the presence registry (``mesh.peers``), ``health`` the last
    health payload per peer, ``safety`` the last safety event type per peer
    (with the ``"fleet"`` key carrying the fleet-wide estop/resume state),
    and ``events`` the timeline. Rows are sorted by peer id so successive
    renders are stable.
    """
    rows = []
    for peer in sorted(peers, key=lambda p: str(p.get("peer_id"))):
        peer_id = str(peer.get("peer_id"))
        battery = health.get(peer_id, {}).get("battery")
        rows.append(
            {
                "peer": peer_id,
                "type": str(peer.get("type", "?")),
                "age_s": peer.get("age"),
                "task": str(peer.get("task_status") or peer.get("instruction") or "-"),
                "battery": battery if battery is not None else "-",
                "safety": safety.get(peer_id, "-"),
            }
        )
    return {"fleet_safety": safety.get("fleet", "ok"), "peers": rows, "events": list(events)}


class FleetDashboard:
    """Collects mesh + audit state and hands snapshots to a renderer.

    The mesh peer is injected already started (and already restricted -
    :func:`main` composes ``init_mesh`` + :func:`restrict_to_subscribe_only`
    + ``attach``), so everything here is observation: subscription callbacks
    fold wire events into local state, ``poll_audit`` tails the signed audit
    log, and ``tick`` renders one snapshot.
    """

    def __init__(self, mesh: Any, renderer: Any, *, tail_audit: bool = True, max_events: int = 200) -> None:
        self._mesh = mesh
        self._renderer = renderer
        self._tail_audit = tail_audit
        self._health: dict[str, dict[str, Any]] = {}
        self._safety: dict[str, str] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._audit_since = time.time()

    def attach(self) -> None:
        """Declare every read-side subscription; raise if any is refused."""
        for topic in SUBSCRIBE_TOPICS:
            if self._mesh.subscribe(topic, callback=self._on_sample, name=f"dashboard:{topic}") is None:
                raise RuntimeError(f"dashboard could not subscribe to {topic!r}; is the mesh up?")

    def _on_sample(self, key: str, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if key.endswith("/health"):
            peer_id = str(data.get("peer_id", key.split("/")[1]))
            self._health[peer_id] = data
            return
        if key == "strands/safety/estop":
            issuer = str(data.get("peer_id", "?"))
            self._safety["fleet"] = f"LOCKOUT (estop by {issuer})"
            self._events.append({"ts": data.get("t", time.time()), "source": "wire", "event": "estop", "peer": issuer})
            return
        if key == "strands/safety/resume":
            self._safety["fleet"] = "ok (resumed)"
            self._events.append({"ts": data.get("t", time.time()), "source": "wire", "event": "resume", "peer": "-"})
            return
        if key.endswith("/safety/event"):
            peer_id = str(data.get("peer_id", key.split("/")[1]))
            self._safety[peer_id] = str(data.get("type", "?"))
            self._events.append(
                {"ts": data.get("t", time.time()), "source": "wire", "event": str(data.get("type")), "peer": peer_id}
            )
        # presence samples need no folding here: the peer registry the mesh
        # already maintains is the fleet-table source of truth.

    def poll_audit(self) -> None:
        """Fold new signed audit records into the event timeline."""
        if not self._tail_audit:
            return
        records = read_audit_log(since=self._audit_since)
        for record in records:
            ts = record.get("ts")
            if isinstance(ts, (int, float)):
                self._audit_since = max(self._audit_since, float(ts) + 1e-6)
            self._events.append(
                {
                    "ts": ts,
                    "source": "audit",
                    "event": str(record.get("event")),
                    "peer": str(record.get("peer_id")),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(self._mesh.peers, self._health, self._safety, list(self._events))

    def tick(self) -> dict[str, Any]:
        self.poll_audit()
        snapshot = self.snapshot()
        self._renderer.render(snapshot)
        return snapshot


class TerminalRenderer:
    """Fallback renderer: the same snapshot, printed as plain text."""

    def __init__(self, *, max_event_lines: int = 8) -> None:
        self._max_event_lines = max_event_lines
        self._seen_events = 0

    def render(self, snapshot: dict[str, Any]) -> None:
        print(f"\n== fleet ({time.strftime('%H:%M:%S')})  safety: {snapshot['fleet_safety']} ==")
        header = f"{'peer':<20} {'type':<10} {'age_s':>6} {'battery':>8} {'safety':<24} task"
        print(header)
        for row in snapshot["peers"]:
            age = f"{row['age_s']:.1f}" if isinstance(row["age_s"], (int, float)) else "?"
            print(
                f"{row['peer']:<20} {row['type']:<10} {age:>6} {row['battery']!s:>8} {row['safety']:<24} {row['task']}"
            )
        new_events = snapshot["events"][self._seen_events :]
        self._seen_events = len(snapshot["events"])
        for event in new_events[-self._max_event_lines :]:
            print(f"  [{event['source']}] {event['event']} ({event['peer']})")


class RerunRenderer:
    """Rerun renderer: fleet table as a text document, events as a text log."""

    def __init__(self, *, spawn: bool = True) -> None:
        self._rr = require_optional(
            "rerun",
            pip_install="rerun-sdk",
            purpose="the fleet dashboard's Rerun rendering",
        )
        self._rr.init("strands-fleet-dashboard", spawn=spawn)
        self._seen_events = 0

    def render(self, snapshot: dict[str, Any]) -> None:
        lines = [f"fleet safety: {snapshot['fleet_safety']}", ""]
        lines.append("| peer | type | age_s | battery | safety | task |")
        lines.append("|---|---|---|---|---|---|")
        for row in snapshot["peers"]:
            age = f"{row['age_s']:.1f}" if isinstance(row["age_s"], (int, float)) else "?"
            lines.append(
                f"| {row['peer']} | {row['type']} | {age} | {row['battery']} | {row['safety']} | {row['task']} |"
            )
        self._rr.log(
            "fleet/table",
            self._rr.TextDocument("\n".join(lines), media_type=self._rr.MediaType.MARKDOWN),
        )
        new_events = snapshot["events"][self._seen_events :]
        self._seen_events = len(snapshot["events"])
        for event in new_events:
            self._rr.log("fleet/events", self._rr.TextLog(f"[{event['source']}] {event['event']} ({event['peer']})"))


def make_renderer(*, prefer_rerun: bool = True, spawn: bool = True) -> Any:
    """Rerun when available, terminal otherwise - degradation is loud."""
    if prefer_rerun:
        try:
            return RerunRenderer(spawn=spawn)
        except ImportError as exc:
            print(f"rerun unavailable ({exc}); rendering to the terminal instead")
    return TerminalRenderer()


class _DashboardOwner:
    """Minimal mesh owner for the dashboard peer (it holds no robot)."""

    tool_name_str = DASHBOARD_ID


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = until Ctrl+C)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between renders")
    parser.add_argument("--no-rerun", action="store_true", help="force the terminal renderer")
    parser.add_argument("--no-spawn", action="store_true", help="do not spawn the Rerun viewer process")
    parser.add_argument("--no-audit", action="store_true", help="do not tail the signed audit log")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        raise SystemExit("--interval must be a positive number of seconds")

    mesh = init_mesh(_DashboardOwner(), peer_id=DASHBOARD_ID, peer_type="dashboard")
    if mesh is None:
        raise SystemExit("mesh is disabled (STRANDS_MESH=0); the dashboard has nothing to attach to")
    if not mesh.alive:
        raise SystemExit('mesh did not start (is eclipse-zenoh installed? pip install "strands-robots[mesh]")')
    restrict_to_subscribe_only(mesh)

    dashboard = FleetDashboard(
        mesh,
        make_renderer(prefer_rerun=not args.no_rerun, spawn=not args.no_spawn),
        tail_audit=not args.no_audit,
    )
    dashboard.attach()
    print(f"dashboard peer {DASHBOARD_ID} attached (subscribe-only); Ctrl+C to exit")

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        while deadline is None or time.monotonic() < deadline:
            dashboard.tick()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        # Ctrl+C is the documented way to stop the dashboard; announce the
        # shutdown so the interrupt is visibly handled, then fall through to
        # the mesh.stop() cleanup below.
        print("dashboard interrupted; detaching from the mesh")
    finally:
        mesh.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
