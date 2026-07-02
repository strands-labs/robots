"""Human-partner coordination for the IRL bed-making G1 (armwaheed/robots#3).

Issue #2 made the bed with **two robots** coordinating over Device Connect. Issue #3 makes it with
**one robot and a human**: the G1 attempts the whole bed on its own and **asks the human for help
only when it gets stuck** — out of reach, the sheet too heavy to pull, or its balance at risk. The
human is reached through the robot's **voice** (speaker + microphone, robotics-connect
`unitree/g1/voice`), and **Device Connect orchestrates** the dialogue and the shared goal state so
the whole human-assisted episode is visible in the dashboard.

Three pieces live here, layered so the core is testable with no robot and no Device Connect:

1. ``CompetenceMonitor`` — the **"ask when stuck" trigger**. Rather than a single naive threshold,
   it reads the execution-time signals (reach excess, grip resistance, balance margin) and decides,
   *before the robot fails*, whether it can finish a corner alone or must ask — the
   competence/failure-prediction idea from BCVA (arXiv:2302.04334) and FAIL-Detect (arXiv:2503.08558),
   applied to a humanoid. Each reason maps to a spoken question + the answers to listen for.

2. The **voice channel** — the real robotics-connect ``VoiceIO`` on the robot (TTS out + mic→ASR),
   or a console/keyboard stub off-robot, behind one small interface. The human's free-speech reply
   is grounded to a choice (KnowNo-style MCQA, arXiv:2307.01928).

3. ``BedMakingHelpDriver`` + ``HumanPartnerCoordinator`` — the Device Connect device (one G1) that
   exposes the help surface as ``@rpc`` and broadcasts the dialogue as ``@emit`` events, and a
   coordinator that runs the DC runtime on a background thread and bridges the synchronous sim loop
   to the (async) DC fabric + the (blocking) voice I/O.

Run a self-contained loopback episode (no robot, no broker):
    python -m examples.isaac_bed_making.human_partner --demo
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

# The shared goal the robot pursues (the same phrase issue #2's swarm used).
GOAL_STATE = "the bed is made"

# The four bed corners to place, in the order a single robot works them (head corners first — the
# pillow-anchored "made" end — then the foot corners), matching the head-of-bed approach.
BED_CORNERS: Tuple[str, ...] = ("NW", "NE", "SW", "SE")

HISTORY_MAXLEN = 256


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ══════════════════════════════════════════════════════════════════════════════════════════════
#  1. The "ask when stuck" trigger — a competence monitor over execution-time signals
# ══════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ReachSignals:
    """The per-corner-attempt signals the robot reads to decide 'can I finish this on my own?'

    These come from the deployed policy + sensors on the real robot (and from the sim in the Isaac
    demo): how far the corner is beyond the trained reach, the sheet tension opposing the pull
    (Brainco fingertip force / measured end-effector load), and the balance margin (how close to
    toppling). `placed` is whether the corner actually landed in tolerance under the robot's own
    effort."""

    corner: str
    target_reach_excess_m: float = 0.0   # metres the corner is BEYOND the trained reach (0 = reachable)
    grip_resistance_n: float = 0.0       # sheet tension opposing the pull, N
    balance_margin: float = 1.0          # 0..1 (1 = solidly upright, 0 = at the topple limit)
    placed: bool = False                 # did the corner land within tolerance on the robot's own?


@dataclass
class HelpDecision:
    need_help: bool
    reason: str = ""           # machine key: out_of_reach | too_heavy | losing_balance | placement_failed
    question: str = ""         # what the robot should say to the human
    choices: Any = "yesno"     # how to ground the reply (VoiceIO choices)
    severity: float = 0.0      # 0..1, how urgent
    detail: str = ""           # human-readable why


class CompetenceMonitor:
    """Decide whether the robot can finish a corner alone, or must ask the human — fired *before*
    the robot fails, not after it topples.

    The limits are tied to what the policy was actually TRAINED to handle, so "stuck" means "outside
    the envelope the RL policy is competent in" rather than an arbitrary number:
      * REACH_LIMIT_M  — the trained hand-target workspace x-extent (bed_reach_env command range).
      * RESISTANCE_LIMIT_N — the grip-slip / sheet-tension force the policy was trained to absorb.
      * BALANCE_FLOOR  — below this balance margin, asking beats pressing on (the headward drag is
        the loco-manipulation balance-under-load failure mode; the behaviour layer's safe move is to
        let go and ask, per RL_WHOLE_BODY_REACH §9).
    """

    REACH_LIMIT_M = 0.55          # trained workspace x-extent (bed_reach_env_cfg pos_x max)
    RESISTANCE_LIMIT_N = 35.0     # the grip-slip force range the policy trained against
    BALANCE_FLOOR = 0.35          # margin below which letting-go + asking is the safe choice

    # reason → (question template, choices). {corner} is filled in per attempt.
    QUESTIONS: Dict[str, Tuple[str, Any]] = {
        "out_of_reach": (
            "I can't quite reach the {corner} corner of the sheet. "
            "Could you tuck that corner in for me? Please say yes or no.",
            "yesno",
        ),
        "too_heavy": (
            "The sheet is caught and I can't pull the {corner} corner without losing my balance. "
            "Could you free it or pull it with me? Please say yes or no.",
            "yesno",
        ),
        "losing_balance": (
            "I'm starting to lose my balance pulling the {corner} corner, so I'm letting go. "
            "Can you take that corner? Please say yes or no.",
            "yesno",
        ),
        "placement_failed": (
            "I tried the {corner} corner but it didn't sit right. "
            "Could you straighten it for me? Please say yes or no.",
            "yesno",
        ),
    }

    def assess(self, sig: ReachSignals) -> HelpDecision:
        """Return a HelpDecision for one corner attempt. Picks the MOST severe tripped signal."""
        candidates: List[Tuple[float, str, str]] = []  # (severity, reason, detail)

        if sig.target_reach_excess_m > 0.0:
            sev = min(1.0, 0.5 + sig.target_reach_excess_m / 0.30)
            candidates.append((sev, "out_of_reach", f"corner is {sig.target_reach_excess_m * 100:.0f} cm beyond reach"))

        if sig.grip_resistance_n > self.RESISTANCE_LIMIT_N:
            over = sig.grip_resistance_n - self.RESISTANCE_LIMIT_N
            sev = min(1.0, 0.5 + over / self.RESISTANCE_LIMIT_N)
            candidates.append((sev, "too_heavy", f"grip load {sig.grip_resistance_n:.0f} N > {self.RESISTANCE_LIMIT_N:.0f} N trained"))

        if sig.balance_margin < self.BALANCE_FLOOR:
            sev = min(1.0, 0.6 + (self.BALANCE_FLOOR - sig.balance_margin))
            candidates.append((sev, "losing_balance", f"balance margin {sig.balance_margin:.2f} < {self.BALANCE_FLOOR:.2f}"))

        # If nothing tripped but the corner still didn't land, that's a (lower-severity) failure ask.
        if not candidates and not sig.placed:
            candidates.append((0.4, "placement_failed", "corner did not reach tolerance"))

        if not candidates:
            return HelpDecision(need_help=False, detail="within competence — robot finishes alone")

        severity, reason, detail = max(candidates, key=lambda c: c[0])
        q, choices = self.QUESTIONS[reason]
        return HelpDecision(
            need_help=True,
            reason=reason,
            question=q.format(corner=sig.corner),
            choices=choices,
            severity=severity,
            detail=detail,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════
#  2. The voice channel — real robotics-connect VoiceIO on the robot, console stub off-robot
# ══════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class VoiceReply:
    transcript: str = ""
    choice: Optional[str] = None
    confidence: float = 0.0
    heard: bool = False


class ConsoleVoice:
    """Off-robot fallback: print what the robot says, read the human's reply from the keyboard, and
    ground it the same way VoiceIO does. Lets the whole ask-when-stuck loop run with no hardware."""

    def __init__(self) -> None:
        self._ground = None
        self._resolve = None
        try:  # reuse the real grounding if the voice module is importable (pure-python parts)
            mod = _import_voice_module()
            self._ground = mod.ground
            self._resolve = mod.VoiceIO._resolve_choices
        except Exception:
            pass

    def announce(self, text: str) -> None:
        print(f'[robot 🔊] "{text}"')

    def ask(self, question: str, choices: Any = "yesno") -> VoiceReply:
        print(f'[robot 🔊] "{question}"')
        try:
            transcript = input("[human 🎤] > ").strip()
        except EOFError:
            transcript = ""
        choice, conf = None, 0.0
        if self._ground and self._resolve and choices is not None:
            opts = self._resolve(choices)
            if opts:
                choice, conf = self._ground(transcript, opts)
        return VoiceReply(transcript=transcript, choice=choice, confidence=conf, heard=bool(transcript))


def _import_voice_module():
    """Import robotics-connect's `g1_voice` from the installed/dev locations, or raise."""
    candidates = [
        os.environ.get("ROBOTICS_CONNECT_DIR", ""),
        "/home/unitree/robotics-connect",
        str(Path.home() / "workspaces/git/robotics-connect-audio"),
        str(Path.home() / "workspaces/git/robotics-connect"),
    ]
    for base in candidates:
        if not base:
            continue
        voice_dir = os.path.join(base, "unitree", "g1", "voice")
        if os.path.isfile(os.path.join(voice_dir, "g1_voice.py")):
            if voice_dir not in sys.path:
                sys.path.insert(0, voice_dir)
            import g1_voice  # type: ignore

            return g1_voice
    raise ImportError("robotics-connect g1_voice not found (set ROBOTICS_CONNECT_DIR)")


def make_voice_channel(iface: Optional[str] = None, prefer_real: bool = True, asr_backend: str = "auto"):
    """Return a voice channel: the real robotics-connect ``VoiceIO`` if available (on the robot),
    else a ``ConsoleVoice`` stub. Both expose ``announce(text)`` and ``ask(question, choices)``;
    ``ask`` returns an object with ``.choice``, ``.transcript``, ``.confidence``, ``.heard``."""
    if prefer_real:
        try:
            mod = _import_voice_module()
            return mod.VoiceIO(iface=iface, asr_backend=asr_backend)
        except Exception as exc:
            print(f"[human-partner] real voice unavailable ({exc!r}); using console voice.")
    return ConsoleVoice()


def _load_rc_module(name: str, path: str):
    """Load a robotics-connect module by file path under a unique name, so its
    ``locomotion`` module can't clash with this demo's own ``locomotion.py``."""
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rc_base() -> Optional[str]:
    """First robotics-connect checkout that has the locomotion module, else None."""
    for base in (os.environ.get("ROBOTICS_CONNECT_DIR", ""),
                 "/home/unitree/robotics-connect",
                 str(Path.home() / "workspaces/git/robotics-connect")):
        if base and os.path.isfile(os.path.join(base, "unitree", "g1", "locomotion", "g1_locomotion.py")):
            return base
    return None


def _import_locomotion_modules():
    """Import robotics-connect's locomotion layer (G1 binding + abstract Sim), or raise."""
    base = _rc_base()
    if base is None:
        raise ImportError("robotics-connect locomotion not found (set ROBOTICS_CONNECT_DIR)")
    loco_lib = _load_rc_module("robotics_connect_locomotion", os.path.join(base, "lib", "locomotion.py"))
    g1mod = _load_rc_module("robotics_connect_g1_locomotion",
                            os.path.join(base, "unitree", "g1", "locomotion", "g1_locomotion.py"))
    return g1mod, loco_lib


def _wire_controller_abort(loco, iface: str) -> None:
    """Wire the handheld controller's any-button abort into ``loco`` (best-effort).

    The watcher halts the *software* routine (stop + hold balance) — it is NOT the
    e-stop; the controller's firmware damping and the robot's power/e-stop are. If the
    controller module is missing the routine still runs and that hardware stop applies."""
    base = _rc_base()
    remote_py = os.path.join(base or "", "unitree", "g1", "controller", "g1_remote.py")
    if not base or not os.path.isfile(remote_py):
        print("[human-partner] controller module unavailable; rely on the hardware e-stop.")
        return
    try:
        g1_remote = _load_rc_module("robotics_connect_g1_remote", remote_py)
        remote = g1_remote.G1Remote(iface=iface, init_dds=False)  # locomotion already inited DDS
        remote.connect()
        remote.wait_until_armed(timeout_s=3.0)
        loco.set_abort_source(remote.aborted)
        loco.remote = remote  # let the routine poll abort between steps
        print("[human-partner] controller abort ARMED — any button halts the routine "
              "(NOT the e-stop; keep the hardware stop in reach).")
    except Exception as exc:
        print(f"[human-partner] controller abort unavailable ({exc!r}); hardware e-stop still applies.")


def make_locomotion(iface: Optional[str] = None, prefer_real: bool = True):
    """Return a locomotion controller from robotics-connect: the real ``G1Locomotion``
    (connected + ready) on the robot, else a dependency-free ``SimLocomotion`` for
    off-robot dry-runs. ``None`` if robotics-connect isn't found. Both controllers
    expose ``stand``/``walk_forward``/``walk_to``/``turn_to``/``stop``."""
    try:
        g1mod, loco_lib = _import_locomotion_modules()
    except ImportError as exc:
        print(f"[human-partner] {exc}; skipping the walk-to-bed approach.")
        return None
    if prefer_real:
        try:
            loco = g1mod.G1Locomotion(iface=iface or "eth0")
            loco.connect()
            _wire_controller_abort(loco, iface or "eth0")
            return loco
        except Exception as exc:
            print(f"[human-partner] real locomotion unavailable ({exc!r}); using sim.")
    return loco_lib.SimLocomotion()


# ══════════════════════════════════════════════════════════════════════════════════════════════
#  3. Device Connect — the single-G1 help device + the coordinator
# ══════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class HelpState:
    """Lock-guarded state of the bed-making robot and its dialogue with the human."""

    device_id: str
    status: str = "online"
    current_corner: Optional[str] = None
    placed_corners: List[str] = field(default_factory=list)         # robot did these alone
    assisted_corners: List[str] = field(default_factory=list)       # human helped with these
    pending_corners: List[str] = field(default_factory=lambda: list(BED_CORNERS))
    last_request: Optional[str] = None
    rpc_counts: Dict[str, int] = field(default_factory=dict)
    event_seq: int = 0
    event_history: Deque[dict] = field(default_factory=lambda: deque(maxlen=HISTORY_MAXLEN))
    help_history: Deque[dict] = field(default_factory=lambda: deque(maxlen=HISTORY_MAXLEN))


class HelpAgent:
    """Per-robot decision state shared between the DC driver and the sim loop, lock-guarded.

    Mirrors swarm_driver.SwarmAgent's surface (snapshot / event_history / help_history) so the
    Device Connect dashboard and any orchestrator read it the same way as the issue #2 peers."""

    def __init__(self, *, device_id: str) -> None:
        self.lock = threading.Lock()
        self.state = HelpState(device_id=device_id)

    # -- history --------------------------------------------------------------------------------
    def log_action(self, *, kind: str, summary: str, **detail: Any) -> dict:
        with self.lock:
            self.state.event_seq += 1
            rec = {"seq": self.state.event_seq, "ts": _utcnow_iso(), "kind": kind, "summary": summary, **detail}
            self.state.event_history.append(rec)
            return rec

    def log_help(self, *, reason: str, corner: str, question: str, transcript: str = "",
                 choice: Optional[str] = None, outcome: str = "asked") -> dict:
        with self.lock:
            rec = {
                "ts": _utcnow_iso(), "reason": reason, "corner": corner, "question": question,
                "human_said": transcript, "grounded": choice, "outcome": outcome,
            }
            self.state.help_history.append(rec)
            self.state.last_request = corner
            return rec

    # -- mutators -------------------------------------------------------------------------------
    def set_status(self, status: str) -> None:
        with self.lock:
            self.state.status = status

    def set_current_corner(self, corner: Optional[str]) -> None:
        with self.lock:
            self.state.current_corner = corner

    def record_corner(self, corner: str, *, assisted: bool) -> None:
        with self.lock:
            if corner in self.state.pending_corners:
                self.state.pending_corners.remove(corner)
            target = self.state.assisted_corners if assisted else self.state.placed_corners
            if corner not in target:
                target.append(corner)

    def tally_rpc(self, name: str) -> None:
        with self.lock:
            self.state.rpc_counts[name] = self.state.rpc_counts.get(name, 0) + 1

    # -- reads ----------------------------------------------------------------------------------
    def is_bed_made(self) -> bool:
        with self.lock:
            return not self.state.pending_corners

    def goal_state(self) -> dict:
        with self.lock:
            done = len(self.state.placed_corners) + len(self.state.assisted_corners)
            return {
                "goal": GOAL_STATE,
                "reached": not self.state.pending_corners,
                "corners_done": done,
                "corners_total": len(BED_CORNERS),
                "placed_by_robot": list(self.state.placed_corners),
                "placed_with_human": list(self.state.assisted_corners),
                "pending": list(self.state.pending_corners),
            }

    def event_history(self, limit: int = 25) -> List[dict]:
        with self.lock:
            return list(self.state.event_history)[-limit:]

    def help_history(self, limit: int = 25) -> List[dict]:
        with self.lock:
            return list(self.state.help_history)[-limit:]

    def snapshot(self) -> dict:
        with self.lock:
            s = self.state
            return {
                "device_id": s.device_id,
                "status": s.status,
                "current_corner": s.current_corner,
                "placed_by_robot": list(s.placed_corners),
                "placed_with_human": list(s.assisted_corners),
                "pending": list(s.pending_corners),
                "last_request": s.last_request,
                "goal": GOAL_STATE,
                "bed_made": not s.pending_corners,
                "rpc_counts": dict(s.rpc_counts),
            }


# The Device Connect driver depends on device_connect_edge; guard the import so the core
# (CompetenceMonitor + voice + HelpAgent) is usable without it (tests / sim-only).
try:
    from device_connect_edge.drivers import DeviceDriver, emit, on, rpc
    from device_connect_edge.types import DeviceIdentity, DeviceStatus

    _HAVE_DC = True
except Exception:  # pragma: no cover - DC optional for the core
    _HAVE_DC = False
    DeviceDriver = object  # type: ignore

    def rpc(*a, **k):  # type: ignore
        def deco(fn):
            return fn

        return deco

    emit = on = rpc  # type: ignore


class BedMakingHelpDriver(DeviceDriver):  # type: ignore[misc]
    """Device Connect device for the single bed-making G1. Exposes the human-help surface as
    callable ``@rpc`` functions (visible/invocable in the portal) and broadcasts the dialogue as
    ``@emit`` events. The actual speaking/listening is performed by the bound voice channel,
    off the DC event loop (run in an executor)."""

    device_type = "unitree_g1_bed_making_irl"

    def __init__(self, *, device_id: str, agent: Optional[HelpAgent] = None,
                 voice: Any = None, monitor: Optional[CompetenceMonitor] = None) -> None:
        if _HAVE_DC:
            super().__init__()
        self._device_id = device_id
        self._agent = agent if agent is not None else HelpAgent(device_id=device_id)
        self._voice = voice
        self._monitor = monitor or CompetenceMonitor()
        self._emit_sink: Optional[Any] = None

    def set_emit_sink(self, sink: Optional[Any]) -> None:
        self._emit_sink = sink

    @property
    def agent(self) -> HelpAgent:
        return self._agent

    # -- DC identity / lifecycle ----------------------------------------------------------------
    @property
    def identity(self) -> "DeviceIdentity":
        return DeviceIdentity(
            device_type=self.device_type,
            manufacturer="Unitree",
            model="G1 EDU (23-DOF, Brainco hands)",
            description=(
                "Unitree G1 EDU making a bed on its own and asking a human partner for help when "
                f"it gets stuck (voice over the G1 speaker + mic). Goal: {GOAL_STATE}."
            ),
        )

    @property
    def status(self) -> "DeviceStatus":
        snap = self._agent.snapshot()
        busy = snap["status"] not in {"online", "available"}
        return DeviceStatus(
            availability="busy" if busy else "available",
            busy_score=1.0 if busy else 0.0,
            location="bed_making_irl",
        )

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _safe_emit(self, name: str, **payload: Any) -> None:
        method = getattr(self, name, None)
        if _HAVE_DC and method is not None:
            try:
                await method(**payload)
            except Exception:
                pass
        if self._emit_sink is not None:
            try:
                await self._emit_sink(name, dict(payload))
            except Exception:
                pass

    # -- callable actions (portal functions) ----------------------------------------------------
    @rpc(labels={"direction": "write", "phase": "coordination"})
    async def requestHumanHelp(self, corner: str = "", reason: str = "stuck", question: str = "",
                               choices: Any = "yesno") -> dict:
        """Ask the human partner for help with a corner — speak the question through the G1's
        speaker and listen for the spoken reply, grounded to a choice. Returns the grounded
        decision so the behaviour layer (or an orchestrator) can act on it."""
        self._agent.tally_rpc("requestHumanHelp")
        q = question or f"I need help with the {corner} corner. Can you help me? Please say yes or no."
        self._agent.log_help(reason=reason, corner=corner, question=q, outcome="asked")
        await self._safe_emit("helpRequested", corner=corner, reason=reason, requester=self._device_id)

        reply = await self._ask_human(q, choices)

        outcome = "accepted" if reply.choice == "yes" else ("declined" if reply.choice == "no" else "unclear")
        self._agent.log_help(reason=reason, corner=corner, question=q, transcript=reply.transcript,
                             choice=reply.choice, outcome=outcome)
        await self._safe_emit("humanResponded", corner=corner, said=reply.transcript,
                              grounded=reply.choice or "", outcome=outcome)
        return {"ok": True, "corner": corner, "reason": reason, "outcome": outcome,
                "choice": reply.choice, "transcript": reply.transcript, "heard": reply.heard}

    @rpc(labels={"direction": "write", "phase": "manipulation"})
    async def reportCornerDone(self, corner: str = "", assisted: bool = False) -> dict:
        """Record a corner as placed (by the robot alone, or with the human's help) and emit it."""
        self._agent.tally_rpc("reportCornerDone")
        self._agent.record_corner(corner, assisted=bool(assisted))
        self._agent.log_action(kind="corner_done", summary=f"{corner} placed ({'with human' if assisted else 'solo'})",
                               corner=corner, assisted=bool(assisted))
        await self._safe_emit("cornerPlaced", corner=corner, assisted=bool(assisted))
        if self._agent.is_bed_made():
            self._agent.set_status("available")
            await self._safe_emit("goalReached", goal=GOAL_STATE)
        return {"ok": True, "corner": corner, "assisted": bool(assisted), **self._agent.goal_state()}

    @rpc(labels={"direction": "write", "phase": "coordination"})
    async def announce(self, text: str = "") -> dict:
        """Speak a status line to the human (no reply expected)."""
        self._agent.tally_rpc("announce")
        await self._run_blocking(self._voice.announce, text) if self._voice else None
        self._agent.log_action(kind="announce", summary=text)
        return {"ok": True, "said": text}

    @rpc(labels={"direction": "read"})
    async def getStatus(self) -> dict:
        return self._agent.snapshot()

    @rpc(labels={"direction": "read"})
    async def getGoalState(self) -> dict:
        return self._agent.goal_state()

    @rpc(labels={"direction": "read"})
    async def getHelpHistory(self, limit: int = 25) -> dict:
        return {"help": self._agent.help_history(limit=limit)}

    @rpc(labels={"direction": "read"})
    async def getEventHistory(self, limit: int = 25) -> dict:
        return {"events": self._agent.event_history(limit=limit)}

    # -- broadcast events -----------------------------------------------------------------------
    @emit(labels={"category": "coordination"})
    async def helpRequested(self, corner: str = "", reason: str = "", requester: str = "") -> None:
        ...

    @emit(labels={"category": "coordination"})
    async def humanResponded(self, corner: str = "", said: str = "", grounded: str = "", outcome: str = "") -> None:
        ...

    @emit(labels={"category": "manipulation"})
    async def cornerPlaced(self, corner: str = "", assisted: bool = False) -> None:
        ...

    @emit(labels={"category": "goal"})
    async def goalReached(self, goal: str = "") -> None:
        ...

    # -- voice plumbing (keep blocking I/O off the DC event loop) --------------------------------
    async def _ask_human(self, question: str, choices: Any) -> VoiceReply:
        if self._voice is None:
            return VoiceReply()
        res = await self._run_blocking(self._voice.ask, question, choices)
        return VoiceReply(transcript=getattr(res, "transcript", ""), choice=getattr(res, "choice", None),
                          confidence=getattr(res, "confidence", 0.0), heard=getattr(res, "heard", False))

    @staticmethod
    async def _run_blocking(fn: Callable, *args) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)


# ── Coordinator: run the DC runtime on a background thread + bridge the sim loop ────────────────
class HumanPartnerCoordinator:
    """Runs the single bed-making G1 as a Device Connect device on a background event loop, with a
    voice channel for the human dialogue. The synchronous sim/behaviour loop calls:

        coord.attempt_corner(corner, signals) -> dict   # assess; ask the human if stuck; record

    which is the whole issue #3 contribution in one call: the competence monitor decides if the
    robot is stuck, the robot speaks + listens if so, the reply is grounded, and the result + the
    dialogue are logged and broadcast over Device Connect.
    """

    def __init__(self, device_id: str = "beta-unitree-g1-edu-0", mode: str = "loopback",
                 iface: Optional[str] = None, voice: Any = None, asr_backend: str = "auto",
                 locomotion: Any = None) -> None:
        self.device_id = device_id
        self.mode = mode
        self.iface = iface
        self._locomotion = locomotion          # robotics-connect controller, lazily made
        self._loco_ready = locomotion is not None
        self.monitor = CompetenceMonitor()
        self.voice = voice or make_voice_channel(iface=iface, asr_backend=asr_backend)
        self.agent = HelpAgent(device_id=device_id)
        self.driver = BedMakingHelpDriver(device_id=device_id, agent=self.agent,
                                          voice=self.voice, monitor=self.monitor)
        self._runtime = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="dc-human-partner")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self) -> str:
        self._thread.start()
        if self.mode == "broker":
            fut = asyncio.run_coroutine_threadsafe(self._async_start_broker(), self._loop)
            fut.result(timeout=30)
        self.agent.log_action(kind="start", summary=f"bed-making G1 online (mode={self.mode})")
        return self.device_id

    async def _async_start_broker(self) -> None:
        from device_connect_edge import DeviceRuntime  # type: ignore

        creds = Path(__file__).resolve().parents[2] / ".credentials" / f"{self.device_id}.creds.json"
        self._runtime = DeviceRuntime(
            driver=self.driver, device_id=self.device_id, messaging_backend="nats",
            messaging_urls=["nats://fabric.deviceconnect.dev:4222"], credentials_file=str(creds),
        )
        self._loop.create_task(self._runtime.run())
        await asyncio.sleep(3.0)

    def _call(self, coro, timeout: float = 30.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def _ensure_locomotion(self):
        if not self._loco_ready:
            self._locomotion = make_locomotion(iface=self.iface, prefer_real=(self.mode == "broker"))
            self._loco_ready = True
        return self._locomotion

    def approach_bed(self, distance_m: float = 1.0, vmax: float = 0.30) -> dict:
        """Walk the robot to the bedside before it starts on the corners.

        First pass: the robot is positioned within a few metres of the bed on a clear path
        (manually, or by the Navigator), so the approach is a short straight ``walk_forward``
        under closed-loop odometry via robotics-connect's locomotion layer. Returns a result
        dict; a skipped no-op if locomotion is unavailable (e.g. off-robot without the layer)."""
        loco = self._ensure_locomotion()
        if loco is None:
            return {"approached": False, "skipped": True}
        self.agent.set_status("approaching")
        self.announce("Walking to the bed.")
        loco.stand()
        result = loco.walk_forward(distance_m, vmax=vmax)
        if result == "aborted":
            self.announce("Aborting.")
            self.agent.set_status("aborted")
        arrived = result is None
        self.agent.log_action(kind="approach", summary=f"walked ~{distance_m:.1f} m to the bedside",
                              arrived=arrived, result=result)
        return {"approached": True, "arrived": arrived, "result": result}

    def aborted(self) -> bool:
        """True once the human presses the controller to abort (real robot only)."""
        remote = getattr(self._locomotion, "remote", None)
        return bool(remote and remote.aborted())

    # -- the synchronous bridge the sim/behaviour loop uses --------------------------------------
    def attempt_corner(self, corner: str, signals: ReachSignals, listen_seconds: float = 6.0) -> dict:
        """Assess the robot's competence for `corner`; if it's stuck, ask the human (speak+listen),
        then record the outcome. Returns a result dict. Blocking (drives the voice I/O)."""
        self.agent.set_current_corner(corner)
        self.agent.set_status("busy")
        decision = self.monitor.assess(signals)

        if not decision.need_help:
            self._call(self.driver.reportCornerDone(corner=corner, assisted=False))
            return {"corner": corner, "asked": False, "assisted": False, "reason": decision.detail}

        res = self._call(
            self.driver.requestHumanHelp(
                corner=corner, reason=decision.reason, question=decision.question, choices=decision.choices
            ),
            timeout=listen_seconds + 30.0,
        )
        accepted = res.get("outcome") == "accepted"
        # Robot-solo policy: if the human helps, the corner gets done with assistance; if they
        # decline or we couldn't hear them, the robot leaves it pending (it can re-approach later).
        if accepted:
            self._call(self.driver.reportCornerDone(corner=corner, assisted=True))
        return {"corner": corner, "asked": True, "assisted": accepted, "reason": decision.reason,
                "outcome": res.get("outcome"), "transcript": res.get("transcript")}

    def announce(self, text: str) -> None:
        self._call(self.driver.announce(text=text))

    def snapshot(self) -> dict:
        return self.agent.snapshot()

    def help_history(self, limit: int = 50) -> List[dict]:
        return self.agent.help_history(limit=limit)

    def goal_state(self) -> dict:
        return self.agent.goal_state()

    def stop(self) -> None:
        async def _shutdown() -> None:
            if self._runtime is not None:
                stop = getattr(self._runtime, "stop", None)
                if stop:
                    res = stop()
                    if asyncio.iscoroutine(res):
                        await res

        try:
            self._call(_shutdown(), timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


# ══════════════════════════════════════════════════════════════════════════════════════════════
#  Loopback demo — a full ask-when-stuck episode with no robot and no broker
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _demo_signals() -> Dict[str, ReachSignals]:
    """A scripted episode: the robot handles two corners alone, and gets stuck on the other two in
    two different ways (out of reach, and balance-at-risk under a heavy pull) — so the human is
    asked exactly when the robot is outside its competence envelope."""
    return {
        "NW": ReachSignals("NW", placed=True, balance_margin=0.8),                       # solo ok
        "NE": ReachSignals("NE", target_reach_excess_m=0.12, balance_margin=0.7),        # out of reach → ask
        "SW": ReachSignals("SW", placed=True, balance_margin=0.6),                       # solo ok
        "SE": ReachSignals("SE", grip_resistance_n=48.0, balance_margin=0.25),           # too heavy + losing balance → ask
    }


def _run_demo() -> int:
    print(f"\n=== IRL bed-making (robot solo, asks when stuck) — loopback demo ===\nGoal: {GOAL_STATE}\n")
    coord = HumanPartnerCoordinator(mode="loopback")
    coord.start()
    coord.announce("Hello. I'm going to make the bed. I'll ask for your help if I get stuck.")
    print(f"\n--- approach ---\n   {coord.approach_bed(distance_m=1.0)}")
    signals = _demo_signals()
    for corner in BED_CORNERS:
        if coord.aborted():
            print("\n!!! ABORTED by controller — halting the routine.")
            break
        print(f"\n--- corner {corner} ---")
        out = coord.attempt_corner(corner, signals[corner])
        verdict = ("solo" if not out["asked"] else f"asked → {out['outcome']}")
        print(f"   result: {verdict}")
    gs = coord.goal_state()
    coord.announce("The bed is made. Thank you for your help!" if gs["reached"] else "I still need a hand to finish.")
    print(f"\n=== goal: {gs} ===")
    print("\n=== help dialogue ===")
    for h in coord.help_history():
        print(f"   [{h['outcome']:8}] {h['corner']} ({h['reason']}): said={h['human_said']!r} → {h['grounded']}")
    coord.stop()
    return 0 if gs["reached"] else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Human-partner bed-making coordination (issue #3).")
    ap.add_argument("--demo", action="store_true", help="Run the loopback ask-when-stuck episode.")
    args = ap.parse_args()
    if args.demo:
        sys.exit(_run_demo())
    print("Use --demo to run the loopback episode. Import HumanPartnerCoordinator to drive the sim.")
