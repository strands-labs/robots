"""Device Connect driver for a Unitree G1 swarm peer in the bed-making demo.

Both G1s register with Device Connect as **equal peers**. There is no
control/worker split: each peer independently pursues the shared goal state
``"the bed is made"``, and **AI Fabric acts as the swarm orchestrator** —
every peer can *see* every other peer over Device Connect and can ask any
peer for help or offer help to any peer that asks.

Coordination happens two ways, both visible in the Device Connect dashboard:

1. **Callable actions (``@rpc``).** The peer's high-level behaviours are
   exposed as Device Connect functions so they show up as invocable buttons
   in the portal and can be driven by an AI Fabric orchestrator:
   ``askForHelp``, ``offerHelp``, ``pickUpBedSheet``, ``walkToNextCorner``
   (counter-clockwise by default), ``putDownBedSheet``, ``emergencyStopAll``,
   plus read-only ``getStatus`` / ``getGoalState`` / ``listPeers`` /
   ``getEventHistory`` / ``getHelpHistory``.

2. **Broadcast events (``@emit`` / ``@on``).** When a peer claims, picks up,
   places, or releases a corner — or asks/offers help — it emits a Device
   Connect event. Every other peer subscribes and updates its local view of
   the swarm so it can react (pick a different corner, walk over to support,
   etc.).

Because the Device Connect server keeps **no persistent event history** (the
portal exposes only a live SSE stream), each driver keeps its own bounded,
timestamped ring buffers — an *event history* and a dedicated *help-request
history* — and exposes them as ``@rpc`` so "what happened" and "who asked
whom for help" are queryable from the dashboard at any time.

The driver mutates its bound :class:`SwarmAgent` (passed in as ``agent``)
under a lock. The simulation main thread reads the agent state and applies it
to MuJoCo; the Device Connect asyncio thread mutates it when peer events
arrive.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

# Imported directly from the public Arm Device Connect edge package so this
# example is self-contained and does not depend on strands_robots internals.
from device_connect_edge.drivers import DeviceDriver, emit, on, periodic, rpc
from device_connect_edge.types import DeviceIdentity, DeviceStatus

logger = logging.getLogger(__name__)

# The shared goal every peer in the swarm is trying to reach.
GOAL_STATE = "the bed is made"

# Sheet corners (A-D) map onto bed corners (compass labels). This mirrors the
# placement plan in examples/mujoco_bed_making/bed_making_demo/unitree_g1_bed_making_demo.py.
SHEET_CORNERS: Tuple[str, ...] = ("A", "B", "C", "D")
SHEET_TO_BED: Dict[str, str] = {"A": "NW", "B": "NE", "C": "SE", "D": "SW"}

# Bed corners arranged as rings so ``walkToNextCorner`` can step a peer around
# the bed. Counter-clockwise (the demo default) is NW -> SW -> SE -> NE.
BED_CORNERS_CW: Tuple[str, ...] = ("NW", "NE", "SE", "SW")
BED_CORNERS_CCW: Tuple[str, ...] = ("NW", "SW", "SE", "NE")

# Bound on the per-peer history ring buffers.
HISTORY_MAXLEN = 256


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ring(direction: str) -> Tuple[str, ...]:
    return BED_CORNERS_CW if str(direction).lower().startswith("clock") else BED_CORNERS_CCW


@dataclass
class PeerView:
    """What this agent knows about another swarm peer."""

    device_id: str
    status: str = "online"
    holding_corner: Optional[str] = None
    holding_position: Optional[Tuple[float, float, float]] = None
    walking_to: Optional[str] = None
    last_event: Optional[str] = None


@dataclass
class SwarmState:
    """Mutable, lock-guarded view of one swarm peer's own state."""

    device_id: str
    role: str = "peer"
    status: str = "online"
    held_corner: Optional[str] = None
    held_position: Optional[Tuple[float, float, float]] = None
    placed_corners: set = field(default_factory=set)
    peer_claims: Dict[str, str] = field(default_factory=dict)
    peers: Dict[str, PeerView] = field(default_factory=dict)
    last_event: Optional[str] = None
    last_help_requester: Optional[str] = None
    corner_cursor: int = 0
    rpc_counts: Dict[str, int] = field(default_factory=dict)
    # Bounded, timestamped history. event_history is the full activity log;
    # help_history is the dedicated "who asked/offered help" log.
    event_seq: int = 0
    event_history: Deque[dict] = field(default_factory=lambda: deque(maxlen=HISTORY_MAXLEN))
    help_history: Deque[dict] = field(default_factory=lambda: deque(maxlen=HISTORY_MAXLEN))


class SwarmAgent:
    """Per-robot decision state shared between the DC driver and the sim loop.

    The sim main thread reads ``state`` to decide MuJoCo actions. The DC
    driver mutates ``state`` from the asyncio thread when peer events arrive,
    and the dashboard-facing RPCs mutate it too. All mutations are guarded by
    ``lock``. Both threads hold the lock when reading/writing nested data.

    Helper methods never call one another while holding ``lock`` (the lock is
    non-reentrant), so logging + state mutation are always sequential.
    """

    def __init__(self, *, device_id: str, role: str = "peer") -> None:
        self.lock = threading.Lock()
        self.state = SwarmState(device_id=device_id, role=role)

    # ── History logging ───────────────────────────────────────────────────

    def log_action(self, *, kind: str, actor: str, summary: str, **detail: Any) -> dict:
        """Append a generic activity record to the event history."""
        with self.lock:
            self.state.event_seq += 1
            rec = {
                "seq": self.state.event_seq,
                "ts": _utcnow_iso(),
                "kind": kind,
                "actor": actor,
                "summary": summary,
                "detail": detail,
            }
            self.state.event_history.append(rec)
            self.state.last_event = summary
            return dict(rec)

    def log_help(self, *, kind: str, frm: str, to: str = "", corner: str = "", reason: str = "") -> dict:
        """Record a help request/offer/resolution.

        Appends to BOTH the dedicated help history and the unified event
        history so the dashboard can show either view. ``kind`` is one of
        ``request`` / ``offer`` / ``resolved``.
        """
        if kind == "request":
            summary = (
                f"{frm} asked {to or 'the swarm'} for help with corner "
                f"{corner or '?'} ({reason or 'stuck'})"
            )
        elif kind == "offer":
            summary = f"{frm} offered to help {to or 'a peer'} with corner {corner or '?'}"
        else:
            summary = f"{frm} resolved help for corner {corner or '?'}"
        with self.lock:
            self.state.event_seq += 1
            seq = self.state.event_seq
            ts = _utcnow_iso()
            help_rec = {
                "seq": seq,
                "ts": ts,
                "kind": kind,
                "from": frm,
                "to": to,
                "corner": corner,
                "reason": reason,
                "summary": summary,
            }
            self.state.help_history.append(help_rec)
            self.state.event_history.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "kind": f"help.{kind}",
                    "actor": frm,
                    "summary": summary,
                    "detail": {"to": to, "corner": corner, "reason": reason},
                }
            )
            self.state.last_event = summary
            return dict(help_rec)

    # ── Mutators called from the DC driver's @on handlers (asyncio thread) ──

    def note_peer_claim(self, peer_id: str, corner: str) -> None:
        with self.lock:
            self.state.peer_claims[peer_id] = corner
            self._peer(peer_id).status = "claiming"
            self._peer(peer_id).last_event = f"claim:{corner}"

    def note_peer_held(self, peer_id: str, corner: str, pos: Tuple[float, float, float]) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.holding_corner = corner
            view.holding_position = pos
            view.status = "holding"
            view.last_event = f"hold:{corner}"
            self.state.peer_claims[peer_id] = corner

    def note_peer_placed(self, peer_id: str, corner: str, bed_corner: str) -> None:
        with self.lock:
            self.state.placed_corners.add(corner)
            self.state.peer_claims.pop(peer_id, None)
            view = self._peer(peer_id)
            view.holding_corner = None
            view.holding_position = None
            view.status = "placed"
            view.last_event = f"placed:{corner}@{bed_corner}"

    def note_peer_released(self, peer_id: str, corner: str) -> None:
        with self.lock:
            self.state.peer_claims.pop(peer_id, None)
            view = self._peer(peer_id)
            view.holding_corner = None
            view.holding_position = None
            view.status = "online"
            view.last_event = f"released:{corner}"

    def note_peer_assist(self, peer_id: str, corner: str, reason: str) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.last_event = f"askAssist:{corner}:{reason}"

    def note_peer_state(self, peer_id: str, status: str, holding: Optional[str]) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.status = status
            view.holding_corner = holding

    def note_peer_walking(self, peer_id: str, corner: str, direction: str) -> None:
        with self.lock:
            view = self._peer(peer_id)
            view.walking_to = corner
            view.status = "walking"
            view.last_event = f"walking:{direction}:{corner}"

    def note_incoming_help_request(self, *, peer_id: str, corner: str, reason: str, target: str) -> None:
        """A peer asked for help — record it (visible in this peer's history)."""
        self.log_help(kind="request", frm=peer_id, to=target, corner=corner, reason=reason)
        with self.lock:
            self.state.last_help_requester = peer_id
            view = self._peer(peer_id)
            view.status = "needs_help"
            view.last_event = f"askForHelp:{corner}:{reason}"

    def note_incoming_help_offer(self, *, peer_id: str, corner: str, target: str) -> None:
        self.log_help(kind="offer", frm=peer_id, to=target, corner=corner)
        with self.lock:
            view = self._peer(peer_id)
            view.status = "assisting"
            view.last_event = f"offerHelp:{corner}"

    # ── Mutators called from the sim main thread / RPCs ────────────────────

    def set_status(self, status: str) -> None:
        with self.lock:
            self.state.status = status

    def set_held_corner(self, corner: Optional[str]) -> None:
        with self.lock:
            self.state.held_corner = corner
            if corner is None:
                self.state.held_position = None

    def set_held_position(self, pos: Optional[Tuple[float, float, float]]) -> None:
        with self.lock:
            self.state.held_position = pos

    def record_placement(self, corner: str) -> None:
        with self.lock:
            if corner:
                self.state.placed_corners.add(corner)

    def advance_corner_cursor(self, direction: str = "counterclockwise") -> str:
        ring = _ring(direction)
        with self.lock:
            self.state.corner_cursor = (self.state.corner_cursor + 1) % len(ring)
            return ring[self.state.corner_cursor]

    def current_corner(self, direction: str = "counterclockwise") -> str:
        ring = _ring(direction)
        with self.lock:
            return ring[self.state.corner_cursor % len(ring)]

    def tally_rpc(self, name: str) -> None:
        with self.lock:
            self.state.rpc_counts[name] = self.state.rpc_counts.get(name, 0) + 1

    # ── Read accessors ─────────────────────────────────────────────────────

    def held_position(self) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            return self.state.held_position

    def last_help_requester(self) -> Optional[str]:
        with self.lock:
            return self.state.last_help_requester

    def can_assist(self) -> bool:
        """A peer can offer help if it is not holding a corner and not stopped."""
        with self.lock:
            return self.state.held_corner is None and self.state.status != "stopped"

    def is_bed_made(self) -> bool:
        with self.lock:
            return len(self.state.placed_corners) >= len(SHEET_CORNERS)

    def goal_state(self) -> dict:
        with self.lock:
            placed = sorted(self.state.placed_corners)
        made = len(placed) >= len(SHEET_CORNERS)
        return {
            "goal": GOAL_STATE,
            "bed_made": made,
            "placed_corners": placed,
            "placed_count": len(placed),
            "remaining": max(0, len(SHEET_CORNERS) - len(placed)),
            "progress": f"{len(placed)}/{len(SHEET_CORNERS)}",
        }

    def event_history(self, limit: int = 25) -> List[dict]:
        with self.lock:
            items = list(self.state.event_history)
        return items[-limit:] if limit and limit > 0 else items

    def help_history(self, limit: int = 25) -> List[dict]:
        with self.lock:
            items = list(self.state.help_history)
        return items[-limit:] if limit and limit > 0 else items

    def peer_list(self) -> Dict[str, dict]:
        with self.lock:
            return {pid: dict(vars(view)) for pid, view in self.state.peers.items()}

    def snapshot(self) -> dict:
        with self.lock:
            placed = sorted(self.state.placed_corners)
            snap = {
                "device_id": self.state.device_id,
                "role": self.state.role,
                "status": self.state.status,
                "goal": GOAL_STATE,
                "bed_made": len(placed) >= len(SHEET_CORNERS),
                "held_corner": self.state.held_corner,
                "held_position": self.state.held_position,
                "placed_corners": placed,
                "progress": f"{len(placed)}/{len(SHEET_CORNERS)}",
                "peer_claims": dict(self.state.peer_claims),
                "peers": {pid: dict(vars(view)) for pid, view in self.state.peers.items()},
                "last_event": self.state.last_event,
                "last_help_requester": self.state.last_help_requester,
                "event_count": len(self.state.event_history),
                "help_count": len(self.state.help_history),
                "recent_events": list(self.state.event_history)[-5:],
                "recent_help": list(self.state.help_history)[-5:],
                "rpc_counts": dict(self.state.rpc_counts),
            }
        return snap

    def _peer(self, peer_id: str) -> PeerView:
        view = self.state.peers.get(peer_id)
        if view is None:
            view = PeerView(device_id=peer_id)
            self.state.peers[peer_id] = view
        return view


class BedMakingG1Driver(DeviceDriver):
    """Device Connect driver for one Unitree G1 swarm peer.

    Each instance is bound to a :class:`SwarmAgent` and exposes the
    bed-making swarm surface — callable actions, broadcast events, peer
    subscriptions, and queryable history. ``role`` is a cosmetic label (all
    production peers use ``"peer"``); ``device_id`` is the registry id.

    When ``auto_offer`` is True, a peer that sees another peer ask for help
    will automatically offer help if it is free — this makes the swarm
    self-organising without an external orchestrator. The default is False so
    unit tests (which call the @on handlers directly, with no transport)
    don't try to emit.
    """

    device_type = "unitree_g1_bed_making"

    def __init__(
        self,
        *,
        device_id: str,
        role: str = "peer",
        agent: Optional[SwarmAgent] = None,
        auto_offer: bool = False,
    ) -> None:
        super().__init__()
        self._device_id = device_id
        self._agent = agent if agent is not None else SwarmAgent(device_id=device_id, role=role)
        self._auto_offer = auto_offer
        # Optional async sink invoked with (event_name, payload) for every
        # event this peer emits. An in-process loopback bus uses it to fan
        # events out to the other peers' @on handlers without a broker, so
        # the swarm coordinates offline exactly as it would over NATS.
        self._emit_sink: Optional[Any] = None

    def set_emit_sink(self, sink: Optional[Any]) -> None:
        """Register an ``async (event_name, payload)`` sink for emitted events."""
        self._emit_sink = sink

    @property
    def identity(self) -> DeviceIdentity:
        return DeviceIdentity(
            device_type=self.device_type,
            manufacturer="Unitree",
            model="G1 EDU",
            description=(
                f"Unitree G1 humanoid — swarm peer ({self._agent.state.role}) in the "
                "Strands Robots two-humanoid MuJoCo bed-making demo. Goal: "
                f"{GOAL_STATE}."
            ),
        )

    @property
    def status(self) -> DeviceStatus:
        snap = self._agent.snapshot()
        my_status = snap["status"]
        busy = my_status not in {"online", "available"}
        # The Device Connect portal renders the status pill GREEN with a
        # flashing dot only when availability == "available" (see
        # device-connect-server .../templates/devices/_device_row_pair.html).
        # Any other value renders a grey pill, so use "available" as the
        # default ready state and "busy" while actively working.
        return DeviceStatus(
            availability="busy" if busy else "available",
            busy_score=1.0 if busy else 0.0,
            location="bed_making_simulator",
        )

    async def connect(self) -> None:
        logger.info(
            "BedMakingG1Driver connected device_id=%s role=%s goal=%r",
            self._device_id,
            self._agent.state.role,
            GOAL_STATE,
        )

    async def disconnect(self) -> None:
        logger.info("BedMakingG1Driver disconnected device_id=%s", self._device_id)

    # ── Internal helper ────────────────────────────────────────────────────

    async def _safe_emit(self, name: str, **payload: Any) -> None:
        """Emit a DC event, tolerating a missing/unconnected transport.

        Lets the callable RPCs record state + history and return a result even
        when no broker is attached (e.g. in unit tests)."""
        method = getattr(self, name, None)
        if method is not None:
            try:
                await method(**payload)
            except Exception as exc:  # pragma: no cover - depends on transport
                logger.debug("emit %s failed: %s", name, exc)
        sink = self._emit_sink
        if sink is not None:
            try:
                await sink(name, dict(payload))
            except Exception as exc:  # pragma: no cover - bus is best-effort
                logger.debug("emit sink %s failed: %s", name, exc)

    # ── Callable swarm actions (shown as functions in the portal) ──────────

    @rpc(labels={"direction": "write", "phase": "coordination"})
    async def askForHelp(self, corner: str = "", reason: str = "", target: str = "") -> dict:
        """Ask the swarm (or a specific peer) for help with a sheet corner.

        Args:
            corner: The sheet corner this peer needs help with (A-D).
            reason: Why help is needed (e.g. 'placed corner is drifting').
            target: Optional peer device_id to ask directly; empty = anyone.
        """
        self._agent.tally_rpc("askForHelp")
        rec = self._agent.log_help(
            kind="request", frm=self._device_id, to=target, corner=corner, reason=reason or "stuck"
        )
        await self._safe_emit(
            "helpRequested", corner=corner, reason=reason or "stuck", requester=self._device_id, target=target
        )
        # Legacy event for back-compat with existing subscribers.
        await self._safe_emit("askAssist", corner=corner, reason=reason or "stuck")
        return {"ok": True, "action": "askForHelp", **rec}

    @rpc(labels={"direction": "write", "phase": "coordination"})
    async def offerHelp(self, target: str = "", corner: str = "") -> dict:
        """Offer to help another peer hold/place a corner.

        Args:
            target: The peer to help; empty = the last peer that asked.
            corner: The corner to assist with.
        """
        self._agent.tally_rpc("offerHelp")
        if not target:
            target = self._agent.last_help_requester() or ""
        rec = self._agent.log_help(kind="offer", frm=self._device_id, to=target, corner=corner)
        self._agent.set_status("assisting")
        await self._safe_emit("helpOffered", corner=corner, target=target, helper=self._device_id)
        return {"ok": True, "action": "offerHelp", **rec}

    @rpc(labels={"direction": "write", "phase": "manipulation"})
    async def pickUpBedSheet(self, corner: str = "") -> dict:
        """Pick up a bed-sheet corner.

        Args:
            corner: The sheet corner to pick up (A-D).
        """
        self._agent.tally_rpc("pickUpBedSheet")
        self._agent.set_held_corner(corner)
        self._agent.set_status("busy")
        rec = self._agent.log_action(
            kind="pickup",
            actor=self._device_id,
            summary=f"{self._device_id} picked up sheet corner {corner or '?'}",
            corner=corner,
        )
        pos = self._agent.held_position() or (0.0, 0.0, 0.0)
        await self._safe_emit("claimCorner", corner=corner)
        await self._safe_emit("cornerHeld", corner=corner, x=pos[0], y=pos[1], z=pos[2])
        return {"ok": True, "action": "pickUpBedSheet", **rec}

    @rpc(labels={"direction": "write", "phase": "navigation"})
    async def walkToNextCorner(self, direction: str = "counterclockwise") -> dict:
        """Walk to the next bed corner.

        Args:
            direction: 'counterclockwise' (default) or 'clockwise'.
        """
        self._agent.tally_rpc("walkToNextCorner")
        target = self._agent.advance_corner_cursor(direction)
        self._agent.set_status("busy")
        rec = self._agent.log_action(
            kind="walk",
            actor=self._device_id,
            summary=f"{self._device_id} walking {direction} to bed corner {target}",
            target=target,
            direction=direction,
        )
        await self._safe_emit("walkingTo", corner=target, direction=direction)
        return {"ok": True, "action": "walkToNextCorner", "target_corner": target, **rec}

    @rpc(labels={"direction": "write", "phase": "manipulation"})
    async def putDownBedSheet(self, corner: str = "", bed_corner: str = "") -> dict:
        """Put down / place a sheet corner onto a bed corner.

        Args:
            corner: The sheet corner being placed (A-D).
            bed_corner: The bed corner to place onto; empty = inferred from corner.
        """
        self._agent.tally_rpc("putDownBedSheet")
        if not bed_corner:
            bed_corner = SHEET_TO_BED.get(corner, "")
        self._agent.record_placement(corner)
        self._agent.set_held_corner(None)
        self._agent.set_status("available")
        bed_made = self._agent.is_bed_made()
        rec = self._agent.log_action(
            kind="place",
            actor=self._device_id,
            summary=(
                f"{self._device_id} placed sheet corner {corner or '?'} on bed corner "
                f"{bed_corner or '?'}"
            ),
            corner=corner,
            bed_corner=bed_corner,
            bed_made=bed_made,
        )
        await self._safe_emit("cornerPlaced", corner=corner, bed_corner=bed_corner)
        await self._safe_emit("cornerReleased", corner=corner)
        if bed_made:
            self._agent.log_action(
                kind="goal",
                actor=self._device_id,
                summary=f"GOAL REACHED: {GOAL_STATE}",
            )
            await self._safe_emit("goalReached", goal=GOAL_STATE)
        return {"ok": True, "action": "putDownBedSheet", "bed_corner": bed_corner, "bed_made": bed_made, **rec}

    # ── Read-only RPCs ─────────────────────────────────────────────────────

    @rpc(labels={"direction": "read"})
    async def getStatus(self) -> dict:
        """Return the agent's full swarm snapshot for the dashboard."""
        self._agent.tally_rpc("getStatus")
        return self._agent.snapshot()

    @rpc(labels={"direction": "read"})
    async def getGoalState(self) -> dict:
        """Return the shared goal and this peer's progress toward it."""
        self._agent.tally_rpc("getGoalState")
        return self._agent.goal_state()

    @rpc(labels={"direction": "read"})
    async def listPeers(self) -> dict:
        """List the other swarm peers this robot can currently see."""
        self._agent.tally_rpc("listPeers")
        return {"device_id": self._device_id, "peers": self._agent.peer_list()}

    @rpc(labels={"direction": "read"})
    async def getEventHistory(self, limit: int = 25) -> dict:
        """Return the most recent activity events (newest last).

        Args:
            limit: Maximum number of events to return.
        """
        self._agent.tally_rpc("getEventHistory")
        return {"device_id": self._device_id, "events": self._agent.event_history(limit)}

    @rpc(labels={"direction": "read"})
    async def getHelpHistory(self, limit: int = 25) -> dict:
        """Return the history of help requests and offers (newest last).

        Args:
            limit: Maximum number of help records to return.
        """
        self._agent.tally_rpc("getHelpHistory")
        return {"device_id": self._device_id, "help": self._agent.help_history(limit)}

    @rpc(labels={"direction": "write", "safety": "critical"})
    async def emergencyStopAll(self, reason: str = "") -> dict:
        """Broadcast an emergency stop so all peers go idle."""
        self._agent.tally_rpc("emergencyStopAll")
        self._agent.log_action(
            kind="estop", actor=self._device_id, summary=f"{self._device_id} triggered emergency stop", reason=reason
        )
        await self._safe_emit("emergencyStop", reason=reason)
        self._agent.set_status("stopped")
        return {"ok": True, "action": "emergencyStopAll", "reason": reason}

    # ── Swarm coordination events (emitted by this peer) ──────────────────

    @emit(labels={"category": "coordination"})
    async def helpRequested(self, corner: str = "", reason: str = "", requester: str = "", target: str = "") -> None:
        """This peer is asking the swarm for help with ``corner``."""

    @emit(labels={"category": "coordination"})
    async def helpOffered(self, corner: str = "", target: str = "", helper: str = "") -> None:
        """This peer is offering to help ``target`` with ``corner``."""

    @emit(labels={"category": "navigation"})
    async def walkingTo(self, corner: str = "", direction: str = "") -> None:
        """This peer is walking to bed ``corner``."""

    @emit(labels={"category": "goal"})
    async def goalReached(self, goal: str = "") -> None:
        """This peer reports the swarm goal has been reached."""

    @emit()
    async def claimCorner(self, corner: str = "") -> None:
        """Announce this peer intends to grab ``corner`` next."""

    @emit()
    async def cornerHeld(self, corner: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """Announce this peer is currently holding ``corner`` at (x, y, z)."""

    @emit()
    async def cornerPlaced(self, corner: str = "", bed_corner: str = "") -> None:
        """Announce this peer has placed ``corner`` over ``bed_corner``."""

    @emit()
    async def cornerReleased(self, corner: str = "") -> None:
        """Announce this peer has let go of ``corner`` — it is free again."""

    @emit()
    async def askAssist(self, corner: str = "", reason: str = "") -> None:
        """Legacy assist event (kept for back-compat); prefer ``helpRequested``."""

    @emit()
    async def heartbeat(self, status: str = "online", holding: str = "", progress: str = "") -> None:
        """Periodic state heartbeat broadcast every few seconds."""

    @emit()
    async def emergencyStop(self, reason: str = "") -> None:
        """Network-wide stop signal."""

    # ── Peer event subscribers ────────────────────────────────────────────

    @on(event_name="claimCorner")
    async def onPeerClaim(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        if corner:
            self._agent.note_peer_claim(device_id, corner)
            logger.info("%s saw peer %s claim %s", self._device_id, device_id, corner)

    @on(event_name="cornerHeld")
    async def onPeerHeld(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        if not corner:
            return
        pos = (
            float(payload.get("x", 0.0)),
            float(payload.get("y", 0.0)),
            float(payload.get("z", 0.0)),
        )
        self._agent.note_peer_held(device_id, corner, pos)

    @on(event_name="cornerPlaced")
    async def onPeerPlaced(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        bed_corner = payload.get("bed_corner", "")
        if corner:
            self._agent.note_peer_placed(device_id, corner, bed_corner)
            logger.info("%s saw peer %s place %s on %s", self._device_id, device_id, corner, bed_corner)

    @on(event_name="cornerReleased")
    async def onPeerReleased(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        if corner:
            self._agent.note_peer_released(device_id, corner)

    @on(event_name="askAssist")
    async def onPeerAsk(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        reason = payload.get("reason", "")
        if corner:
            self._agent.note_peer_assist(device_id, corner, reason)

    @on(event_name="helpRequested")
    async def onPeerHelpRequested(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        reason = payload.get("reason", "")
        target = payload.get("target", "")
        self._agent.note_incoming_help_request(peer_id=device_id, corner=corner, reason=reason, target=target)
        logger.info("%s saw peer %s ask for help with %s (%s)", self._device_id, device_id, corner, reason)
        # Self-organising swarm: a free peer offers help automatically.
        if self._auto_offer and (not target or target == self._device_id) and self._agent.can_assist():
            await self.offerHelp(target=device_id, corner=corner)

    @on(event_name="helpOffered")
    async def onPeerHelpOffered(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        corner = payload.get("corner", "")
        target = payload.get("target", "")
        self._agent.note_incoming_help_offer(peer_id=device_id, corner=corner, target=target)

    @on(event_name="walkingTo")
    async def onPeerWalking(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        self._agent.note_peer_walking(device_id, payload.get("corner", ""), payload.get("direction", ""))

    @on(event_name="goalReached")
    async def onPeerGoalReached(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        self._agent.log_action(
            kind="goal",
            actor=device_id,
            summary=f"{device_id} reports GOAL: {payload.get('goal', GOAL_STATE)}",
        )

    @on(event_name="heartbeat")
    async def onPeerHeartbeat(self, device_id: str, event_name: str, payload: dict) -> None:
        if device_id == self._device_id:
            return
        status = payload.get("status", "online")
        holding = payload.get("holding", "") or None
        self._agent.note_peer_state(device_id, status, holding)

    @on(event_name="emergencyStop")
    async def onPeerEmergencyStop(self, device_id: str, event_name: str, payload: dict) -> None:
        logger.warning(
            "device=%s emergency stop received from %s reason=%r",
            self._device_id,
            device_id,
            payload.get("reason"),
        )
        self._agent.log_action(
            kind="estop", actor=device_id, summary=f"{device_id} broadcast emergency stop"
        )
        self._agent.set_status("stopped")

    # ── Periodic heartbeat ────────────────────────────────────────────────

    @periodic(interval=2.0, wait_for_completion=True)
    async def _emitHeartbeat(self) -> None:
        snap = self._agent.snapshot()
        try:
            await self.heartbeat(
                status=snap["status"],
                holding=snap.get("held_corner") or "",
                progress=snap.get("progress", ""),
            )
        except Exception as exc:
            logger.debug("heartbeat emit failed: %s", exc)

    # ── Helpers used by the sim main thread ───────────────────────────────

    @property
    def agent(self) -> SwarmAgent:
        return self._agent

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def role(self) -> str:
        return self._agent.state.role
