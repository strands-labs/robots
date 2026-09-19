"""The dashboard's agent: a Strands Agent whose hands are the simulator sessions.

One :class:`Console` is one operator conversation. Its tools go through the
same :class:`~strands_robots.dashboard.routes_sim.Safety` object the HTTP
routes use, so the e-stop refuses the agent exactly as it refuses a button,
and every accepted command is the same proof that the lockout is clear.

Anything that moves a robot - here ``sim_set_joints`` - raises the SDK
interrupt the real-hardware hook uses (:mod:`strands_robots.dashboard.agent_hitl`),
so the browser shows a consent card and the same turn resumes on a yes. The
operator can grant one call or the rest of the conversation; the grant lives
in this object and dies with the socket. Stopping is never gated.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from strands import Agent, tool
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)

INTERRUPT_NAME = "sim_motion"
MODEL_ENV = "STRANDS_MODEL_ID"
MAX_PROMPT_CHARS = 8_000

SYSTEM_PROMPT = """You are the strands-robots dashboard agent. You operate simulated robots for an operator
who is watching the same screen. Be brief. Before moving a robot, know which session you are
moving (sim_sessions) and what its joints are (sim_state). Joint positions are radians; joints
are addressed by name or by 1-based index as strings. A move you request may be put to the
operator first; if they decline, say so and stop. Never work around a refusal or an e-stop."""

#: agent tool name -> does it move the robot (and so asks the operator first)?
MOTION_TOOLS: frozenset[str] = frozenset({"sim_set_joints"})


def model_id() -> str:
    """``STRANDS_MODEL_ID`` if the operator set one, else whatever model the SDK defaults to.

    The console holds no model name of its own: a second copy diverges from the
    installed SDK's default the moment either moves, and the operator meets that
    divergence as a ValidationException on their first turn.
    """
    from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID

    return os.environ.get(MODEL_ENV) or DEFAULT_BEDROCK_MODEL_ID


def default_model() -> Any:
    """A Bedrock model on :func:`model_id` (``STRANDS_MODEL_ID``, the variable the CLI honours)."""
    from strands.models import BedrockModel

    return BedrockModel(
        model_id=model_id(),
        region_name=os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1",
    )


@dataclass
class Grants:
    """What the operator has already said yes to, for this conversation only."""

    sessions: set[str] = field(default_factory=set)

    def covers(self, session_id: str) -> bool:
        """Has the operator allowed motion on this session for the rest of the conversation?"""
        return session_id in self.sessions

    def extend(self, session_id: str) -> None:
        """Remember a 'for this conversation' yes."""
        self.sessions.add(session_id)


def response_approves(response: Any) -> tuple[bool, bool]:
    """(approved, for the rest of the conversation). Anything but an explicit yes is a no."""
    if isinstance(response, bool):
        return response, False
    if isinstance(response, Mapping):
        approve = response.get("approve")
        if isinstance(approve, bool):
            return approve, bool(response.get("always", False)) and approve
        return False, False
    if isinstance(response, str):
        return response.strip().lower() in {"yes", "y", "approve", "approved", "ok"}, False
    return False, False


class MotionGate(HookProvider):
    """Interrupt before any motion tool call the operator has not already granted."""

    def __init__(self, grants: Grants) -> None:
        self._grants = grants

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe the gate to every tool call the agent is about to make."""
        registry.add_callback(BeforeToolCallEvent, self._gate)

    def _gate(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = str(tool_use.get("name") or "")
        if name not in MOTION_TOOLS:
            return
        tool_input = dict(tool_use.get("input") or {})
        session_id = str(tool_input.get("session_id") or "")
        if self._grants.covers(session_id):
            return
        reason = {
            "tool": name,
            "session_id": session_id,
            "positions": tool_input.get("positions"),
            "detail": _detail(tool_input),
        }
        response = event.interrupt(INTERRUPT_NAME, reason=reason)
        approved, always = response_approves(response)
        from strands_robots._hitl_audit import log_operator_response

        log_operator_response("dashboard_agent_console", name, session_id, approved=approved, response=response)
        if approved:
            if always:
                self._grants.extend(session_id)
            return
        event.cancel_tool = "The operator declined this motion. Do not retry it; tell them and wait."


def _detail(tool_input: Mapping[str, Any]) -> str:
    positions = tool_input.get("positions")
    if isinstance(positions, Mapping):
        return ", ".join(f"{k} → {float(v):.3f} rad" for k, v in positions.items() if isinstance(v, (int, float)))
    return json.dumps(positions)


def build_tools(safety: Any) -> list[Any]:
    """The agent's hands: every call goes through ``safety`` like a button press would."""

    def _session(session_id: str) -> Any:
        session = safety.store.get(session_id)
        if session is None:
            raise ValueError(f"no session {session_id}")
        return session

    def _snapshot(session: Any) -> dict[str, Any]:
        snap = session.snapshot.as_dict()
        return {k: v for k, v in snap.items() if k != "model_path"}

    def _gate(action: str) -> None:
        # Safety.gate raises HTTPException(423); the agent should read a sentence.
        from fastapi import HTTPException

        try:
            safety.gate(action)
        except HTTPException as exc:
            raise PermissionError(str(exc.detail))

    def _accepted(drop: str | None = None) -> None:
        """Fold the proof this command was accepted, or report the e-stop that beat it.

        Args:
            drop: a session to forget when the e-stop landed. A session that was
                admitted and then refused must not be left in the store: it holds
                one of ``MAX_SESSIONS`` slots and thaws into a running robot on
                resume - a robot the caller was told was refused.

        Raises:
            PermissionError: the lockout latched while this command was in flight.
        """
        from fastapi import HTTPException

        try:
            safety.accepted()
        except HTTPException as exc:
            if drop is not None:
                safety.store.remove(drop)
            raise PermissionError(str(exc.detail))

    @tool
    def robots() -> list[dict[str, Any]]:
        """Robots that can be simulated: name, dof, and whether a session already runs one."""
        from strands_robots.dashboard.fleet import registry_robots

        running = {s.robot for s in safety.store.all()}
        return [{**r, "running": r["name"] in running} for r in registry_robots("sim")]

    @tool
    def sim_sessions() -> list[dict[str, Any]]:
        """Every running simulation: id, robot, state, joint names, current joint positions."""
        return [_snapshot(s) for s in safety.store.all()]

    @tool
    def sim_start(robot: str) -> dict[str, Any]:
        """Start a simulation of a registry robot and return its session (id, joints).

        A start that does not finish is forgotten rather than handed back: the
        state published before the first frame is ``running``, so a session that
        never rendered would be reported as a robot the operator can watch while
        it streams nothing and holds one of the store's slots.
        """
        _gate("create")
        from strands_robots.dashboard import routes_sim
        from strands_robots.registry.robots import get_robot, resolve_name

        entry = get_robot(robot)
        if entry is None or not entry.get("asset"):
            raise ValueError(f"{robot!r} is not a robot with a simulation asset")
        session = safety.store.create(resolve_name(robot))
        timeout = routes_sim.READY_TIMEOUT
        if not session.wait_ready(timeout):
            safety.store.remove(session.id)
            raise RuntimeError(f"{robot} did not render a first frame within {timeout:.0f}s")
        if session.snapshot.state == "error":
            safety.store.remove(session.id)
            raise RuntimeError(f"could not start {robot}: {session.snapshot.error}")
        _accepted(drop=session.id)
        return _snapshot(session)

    @tool
    def sim_state(session_id: str) -> dict[str, Any]:
        """The session's latest snapshot: state, sim time, joint names and positions (radians)."""
        return _snapshot(_session(session_id))

    @tool
    def sim_set_joints(session_id: str, positions: dict[str, float]) -> dict[str, Any]:
        """Move joints of a simulated robot to target positions in radians.

        Args:
            session_id: which simulation (from sim_sessions).
            positions: joint name or 1-based index (as a string) -> radians, e.g. {"2": 1.2}.
        """
        _gate("set_joints")
        if not positions or any(not isinstance(v, (int, float)) for v in positions.values()):
            raise ValueError("positions must map joint -> number")
        result = _session(session_id).command("set_joints", positions=dict(positions))
        if result.get("status") == "error":
            raise ValueError(str(result.get("content")))
        _accepted()
        return dict(result)

    @tool
    def sim_reset(session_id: str) -> dict[str, Any]:
        """Return a simulated robot to its home pose."""
        _gate("reset")
        result = _session(session_id).command("reset")
        _accepted()
        return dict(result)

    @tool
    def sim_stop(session_id: str) -> dict[str, Any]:
        """Stop a simulation and forget it. Never refused."""
        return {"ok": safety.store.remove(session_id)}

    @tool
    def emergency_stop() -> dict[str, Any]:
        """Freeze every simulation and latch the dashboard's lockout. Never refused."""
        return dict(safety.estop(by="agent"))

    return [robots, sim_sessions, sim_start, sim_state, sim_set_joints, sim_reset, sim_stop, emergency_stop]


class Console:
    """One operator conversation."""

    def __init__(self, safety: Any, model: Any | None = None) -> None:
        self.grants = Grants()
        self.agent = Agent(
            model=model if model is not None else default_model(),
            tools=build_tools(safety),
            hooks=[MotionGate(self.grants)],
            system_prompt=SYSTEM_PROMPT,
            callback_handler=None,
        )

    async def run(self, prompt: Any) -> AsyncIterator[dict[str, Any]]:
        """Stream one turn as flat JSON events: text, tool_use, tool_result, interrupt, done, error."""
        try:
            async for event in self.agent.stream_async(prompt):
                for out in _translate(event):
                    yield out
                if "result" in event:
                    result = event["result"]
                    interrupts = list(getattr(result, "interrupts", None) or [])
                    if interrupts:
                        for it in interrupts:
                            yield {"type": "interrupt", "id": it.id, "name": it.name, "reason": it.reason}
                    else:
                        yield {"type": "done", "stop_reason": str(getattr(result, "stop_reason", ""))}
        except Exception as exc:  # noqa: BLE001 - the operator reads it as one line
            logger.warning("agent console turn failed: %s", exc)
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def resume(interrupt_id: str, approve: bool, always: bool = False) -> list[dict[str, Any]]:
        """The prompt that answers an interrupt - the SDK's interruptResponse block."""
        return [
            {
                "interruptResponse": {
                    "interruptId": interrupt_id,
                    "response": {"approve": bool(approve), "always": bool(always)},
                }
            }
        ]


def _translate(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """SDK stream events -> the flat events the browser renders. Text deltas and whole messages only."""
    if "data" in event and isinstance(event["data"], str):
        return [{"type": "text", "text": event["data"]}]
    message = event.get("message")
    out: list[dict[str, Any]] = []
    if isinstance(message, Mapping):
        for block in message.get("content") or []:
            if "toolUse" in block:
                tu = block["toolUse"]
                out.append({"type": "tool_use", "name": tu.get("name"), "input": tu.get("input")})
            elif "toolResult" in block:
                tr = block["toolResult"]
                texts = [str(c.get("text", "")) for c in tr.get("content", []) if isinstance(c, Mapping)]
                out.append(
                    {"type": "tool_result", "status": tr.get("status"), "text": "\n".join(t for t in texts if t)[:2000]}
                )
    return out
