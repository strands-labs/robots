"""Autonomous per-robot behaviour for the Isaac Sim bed-making swarm.

Each G1 runs its **own** decision loop. Nothing here is a fixed timeline: every
tick the robot looks at the live world (where the sheet corners actually are,
which corners are placed, whether a placed corner has drifted) and at the swarm
state it sees over Device Connect (which corners peers have claimed, who is
asking for help), and decides its next manipulation command. The order in which
corners get placed, and when a robot breaks off to help its peer, *emerge* from
that state.

The behaviour yields abstract manipulation commands; the executor in
``demo.py`` turns them into G1 arm-IK reaches, cloth grasps, and releases, then
feeds observations back. Device Connect side effects (claim a corner, place it,
ask for / offer help) are issued through the :class:`SwarmCoordinator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Sheet corner -> bed corner placement (matches the swarm driver's SHEET_TO_BED).
SHEET_TO_BED = {"A": "NW", "B": "NE", "C": "SE", "D": "SW"}
# Bed corner each robot stands nearest, used to split work between the two peers.
DEFAULT_ASSIGNMENTS = {0: ["A", "C"], 1: ["B", "D"]}

DRIFT_THRESHOLD_M = 0.18  # a placed corner this far from its bed target = "drifting"


@dataclass
class Command:
    """An abstract manipulation command for the executor to fulfil."""

    kind: str  # reach_grasp | carry_place | hold | release_idle | done
    corner: Optional[str] = None        # sheet corner label (A..D)
    bed_corner: Optional[str] = None     # compass label
    target: Optional[Tuple[float, float, float]] = None  # world xyz
    arm: str = "right"                   # which arm to use


@dataclass
class WorldView:
    """Observation passed to a behaviour each tick (read from the live sim)."""

    sheet_corner_world: Dict[str, Tuple[float, float, float]]  # label -> xyz
    bed_corner_world: Dict[str, Tuple[float, float, float]]    # compass -> xyz
    placed_ok: Dict[str, bool]            # bed corner -> currently within tolerance
    holding: Dict[str, bool]              # bed corner -> a robot is holding it


SHEET_LABEL_TO_COMPASS = SHEET_TO_BED


class RobotBehaviour:
    """State machine for one G1 peer."""

    def __init__(self, robot_idx: int, coordinator, assignments: Optional[List[str]] = None):
        self.idx = robot_idx
        self.coord = coordinator
        self.device_id = coordinator.peers[robot_idx].device_id
        self.queue: List[str] = list(assignments or DEFAULT_ASSIGNMENTS[robot_idx])
        self.phase = "idle"            # idle|reaching|carrying|placing|helping|done
        self.active_corner: Optional[str] = None
        self.helping_corner: Optional[str] = None
        self.placed: List[str] = []    # sheet labels this robot has placed
        self._announced_help_for: set = set()
        self._announced: set = set()

    # ── helpers ───────────────────────────────────────────────────────────
    def _arm_for(self, compass: str) -> str:
        # use the arm on the bed-side the corner is on (east corners -> right)
        return "right" if compass in ("NE", "SE") else "left"

    def _peer_help_request(self) -> Optional[dict]:
        """Return the most recent unresolved help request from another peer."""
        reqs = [h for h in self.coord.help_history(self.idx)
                if h["kind"] == "request" and h["from"] != self.device_id]
        return reqs[-1] if reqs else None

    # ── main decision ─────────────────────────────────────────────────────
    def tick(self, world: WorldView) -> Command:
        # 1) Highest priority: a peer is asking for help and I'm free -> go hold.
        if self.phase in ("idle",) and not self.queue:
            req = self._peer_help_request()
            if req:
                compass = req.get("corner") or ""
                bed_compass = SHEET_LABEL_TO_COMPASS.get(compass, compass)
                if bed_compass in world.bed_corner_world and not world.holding.get(bed_compass):
                    self.phase = "helping"
                    self.helping_corner = compass
                    self.coord.invoke(self.idx, "offerHelp",
                                      to=req["from"], corner=compass)
                    tgt = world.sheet_corner_world.get(compass) or world.bed_corner_world[bed_compass]
                    return Command(kind="hold", corner=compass, bed_corner=bed_compass,
                                   target=tgt, arm=self._arm_for(bed_compass))

        # 2) If I'm mid-task, the executor reports completion by calling
        #    on_command_done(); here we just emit the current intent again is not
        #    needed — tick is only called when ready for the NEXT command.

        # 3) Start the next assigned corner.
        if self.phase in ("idle",) and self.queue:
            label = self.queue[0]
            compass = SHEET_TO_BED[label]
            # Claim it on Device Connect so the peer won't duplicate.
            self.coord.invoke(self.idx, "pickUpBedSheet", corner=label)
            self.active_corner = label
            self.phase = "carrying"
            tgt = world.sheet_corner_world.get(label)
            return Command(kind="reach_grasp", corner=label, bed_corner=compass,
                           target=tgt, arm=self._arm_for(compass))

        # 4) Nothing to do.
        if not self.queue and self.phase == "idle":
            self.phase = "done"
            return Command(kind="done")
        return Command(kind="release_idle")

    # ── executor callbacks ────────────────────────────────────────────────
    def on_reach_grasped(self, world: WorldView) -> Command:
        """Hand has grasped the sheet corner; now carry it to the bed corner."""
        label = self.active_corner
        compass = SHEET_TO_BED[label]
        self.coord.invoke(self.idx, "walkToNextCorner", direction="counterclockwise")
        return Command(kind="carry_place", corner=label, bed_corner=compass,
                       target=world.bed_corner_world[compass], arm=self._arm_for(compass))

    def on_placed(self) -> None:
        """Corner placed + released."""
        label = self.active_corner
        self.coord.invoke(self.idx, "putDownBedSheet", corner=label)
        self.placed.append(label)
        if self.queue and self.queue[0] == label:
            self.queue.pop(0)
        self.active_corner = None
        self.phase = "idle"

    def on_help_done(self) -> None:
        self.helping_corner = None
        self.phase = "idle"

    def ask_for_help(self, sheet_label: str, reason: str) -> None:
        """Owner of a drifting placed corner asks the swarm to hold it."""
        if sheet_label in self._announced:
            return
        self._announced.add(sheet_label)
        self.coord.invoke(self.idx, "askForHelp", corner=sheet_label, reason=reason)
