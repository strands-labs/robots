"""Per-peer proxy AgentTools: every fleet robot becomes a NATIVE tool on the dashboard agent.

Users of strands_robots write ``Agent(tools=[Robot('so101')])`` and the robot IS a tool.
The dashboard cannot do that literally: its robots are CHILD PROCESSES holding the
serial buses / sim state, and a second in-process ``Robot('so101')`` would collide on
the bus. So "native" here means a PROXY that is indistinguishable to the agent: for
each fleet peer we build an AgentTool named for it whose tool_spec mirrors what that
peer really is - ``hardware_robot.Robot``'s execute/start/status/stop spec for a real
arm, the MuJoCo published-action spec for a sim - and whose invocation routes over the
mesh rails that already exist (``sim_call`` for sim actions; the validated
execute/start/status/stop command family for robots), via the dashboard bridge's
``send_cmd``.

Gating stays ONE layer: these proxies do NOT gate themselves. ``MotionInterruptHook``
(agent_hitl) gates them by tool name + action, peer-aware through ``peer_is_physical``
 -  which is why :func:`map_invocation` guarantees a ``target`` field is always present
in the reason the hook derives (the proxy binds it). sim actions ask nothing (the peer
is provably a sim), stop/status are never gated (not in MOTION_ACTIONS).

Everything above the wire is a PURE rule in this module, tested without a mesh.
"""

from __future__ import annotations

import json
import keyword
import re
from collections.abc import AsyncGenerator, Callable, Mapping
from pathlib import Path
from typing import Any, cast

# ── classification ──────────────────────────────────────────────────────────

#: Robot kinds a proxy can represent. ``skip`` = build no tool for this peer.
KIND_REAL = "real"
KIND_SIM = "sim"
KIND_HOST = "host"  # a robot process with no joints announced (yet): status/stop only
KIND_SKIP = "skip"

#: Sim actions a ``sim_call`` rail refuses: rollouts must ride execute/start, whose
#: provider/HF-repo/host allowlists would otherwise be bypassed. The proxy
#: spec must not advertise what the wire will refuse. main's ``strands_robots.mesh.security``
#: has no ``sim_call`` action yet, so the list lives here until the rail lands
#: and exports it (then import it from there so the two cannot drift).
SIM_CALL_BLOCKED: frozenset[str] = frozenset({"run_policy", "start_policy", "replay_episode", "eval_policy"})

_SIM_TYPES = ("sim", "simulation", "mujoco")

#: Peer types that coordinate rather than move: no tool at all. Read off
#: ``robot_type`` because that is the field the WIRE carries -
#: ``strands_robots.mesh.core`` builds presence as ``{"robot_id", "robot_type": peer_type,
#: "hostname", "timestamp", ...}`` (``strands_robots.mesh.core``, the presence builder) and
#: ``robot_mesh._gateway_mesh()`` joins with ``peer_type="gateway"``, as does
#: ``mesh_bridge``'s safety peer. Nothing publishes a ``kind`` field: reading
#: ``presence["kind"]`` skipped nothing at all, so throwaway ``gateway-*``
#: sessions became AgentTools described to the model as "Robot peer" and
#: ``fleet_signature`` churned the agent on every probe's birth and death
#: (measured once: 4 of 8 live tools were coordinator debris).
_SKIP_TYPES = ("gateway", "dashboard")

#: Belt #1: a coordinator's peer id. ``_gateway_mesh`` names itself
#: ``gateway-<host>-<hex>``; the dashboard's own peer/safety session likewise.
_SKIP_ID_PREFIXES = ("gateway-", "dashboard-")


def _is_coordinator(peer_id: str, peer: Mapping[str, Any], presence: Mapping[str, Any]) -> bool:
    """Is this peer a mesh coordinator (gateway/dashboard) rather than a robot?

    Three independent reads, so a presence payload that drops or renames its
    type field still cannot mint a motion tool for something with no hardware:
    the published ``robot_type``, the peer id's own prefix, and the topic
    advertisement (a robot-less ``Mesh`` announces ``topics == ["health"]``
    only - ``strands_robots.mesh.core`` appends "health" unconditionally and every other topic
    needs a hardware attribute).
    """
    robot_type = str(presence.get("robot_type") or "").strip().lower()
    kind = str(presence.get("kind") or peer.get("kind") or "").strip().lower()
    if robot_type in _SKIP_TYPES or kind in _SKIP_TYPES:
        return True
    if robot_type or kind:
        return False  # it named itself something else: believe it
    pid = peer_id.strip().lower()
    if pid.startswith(_SKIP_ID_PREFIXES):
        return True
    topics = presence.get("topics")
    if isinstance(topics, (list, tuple)) and [str(t).strip().lower() for t in topics] == ["health"]:
        state = peer.get("state") or {}
        if not (state.get("joints") or presence.get("joints") or peer.get("cameras")):
            return True
    return False


def classify_peer(peer_id: str, peer: Mapping[str, Any] | None) -> str:
    """What kind of tool should represent this peer?

    Mirrors ``agent_motion.peer_is_physical``'s reading of presence, but with
    the opposite default posture: the GATE fails closed (unknown = metal), a
    TOOL FACTORY fails quiet (unknown/gateway/dashboard = no tool at all) -
    a tool for a peer we cannot describe would advertise a spec we invented.
    """
    if not peer_id or peer is None:
        return KIND_SKIP
    presence = peer.get("presence") or {}
    kind = str(presence.get("kind") or peer.get("kind") or "").strip().lower()
    if _is_coordinator(peer_id, peer, presence):
        return KIND_SKIP
    robot_type = str(presence.get("robot_type") or "").strip().lower()
    if robot_type in _SIM_TYPES or presence.get("sim") is True or presence.get("mode") == "sim":
        return KIND_SIM
    # A child peer of a sim world (``<parent>__<robot>``) is itself a sim
    # robot even when its own presence is sparse: core delegates its sim_call
    # to the parent Simulation.
    if "__" in peer_id and str(peer.get("parent") or presence.get("parent") or "").strip():
        return KIND_SIM
    state = peer.get("state") or {}
    joints = state.get("joints") or presence.get("joints") or {}
    n_joints = len(joints) if isinstance(joints, Mapping) else int(joints or 0)
    hw = presence.get("hw")
    if n_joints > 0 or (isinstance(hw, str) and hw.strip()) or peer.get("role"):
        return KIND_REAL
    if kind == "robot" or presence:
        return KIND_HOST
    return KIND_SKIP


# ── naming ───────────────────────────────────────────────────────────────────

_NAME_OK = re.compile(r"[^A-Za-z0-9_]")


def sanitize_tool_name(peer_id: str, taken: frozenset[str] | set[str] = frozenset()) -> str:
    """Peer id -> identifier-safe, unique tool name.

    Peer ids carry dashes (``so101-real-689``); tool names must be
    identifier-safe (``so101_real_689``). Collisions (two peers sanitizing to
    one name) get a numeric suffix - deterministic in iteration order.
    """
    name = _NAME_OK.sub("_", peer_id.strip()) or "peer"
    if name[0].isdigit():
        name = f"p_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_"
    base, n = name, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    return name


# ── tool specs ───────────────────────────────────────────────────────────────

_SIM_SPEC_PATH = Path(__file__).resolve().parents[1] / "simulation" / "mujoco" / "tool_spec.json"
_sim_schema_cache: dict[str, Any] | None = None


def _sim_input_schema() -> dict[str, Any]:
    """The MuJoCo published-action schema, with wire-refused actions removed."""
    global _sim_schema_cache
    if _sim_schema_cache is None:
        raw = json.loads(_SIM_SPEC_PATH.read_text(encoding="utf-8"))
        actions = [a for a in raw["properties"]["action"]["enum"] if a not in SIM_CALL_BLOCKED]
        schema = json.loads(json.dumps(raw))  # deep copy; the file is trusted JSON
        schema["properties"]["action"]["enum"] = actions
        schema["properties"]["action"]["description"] = (
            "Published simulation action to invoke on this sim peer. Policy rollouts "
            "(run_policy/start_policy/replay_episode/eval_policy) are not carried on "
            "this rail. Use the execute/start actions of a robot tool instead."
        )
        _sim_schema_cache = schema
    return _sim_schema_cache


def peer_tool_spec(peer_id: str, kind: str, tool_name: str) -> dict[str, Any] | None:
    """The ToolSpec a proxy presents for this peer - mirrors what the peer IS."""
    if kind == KIND_SIM:
        return {
            "name": tool_name,
            "description": (
                f"Simulation peer '{peer_id}' as a native tool. Invokes the sim's own "
                f"published actions (add_object, add_camera, list_objects, raycast, "
                f"register_urdf, ...) over the mesh sim_call rail - world building and "
                f"inspection, never real hardware. Parameters beyond 'action' are that "
                f"action's own keyword arguments."
            ),
            "inputSchema": {"json": _sim_input_schema()},
        }
    if kind == KIND_REAL:
        return {
            "name": tool_name,
            "description": (
                f"Real robot peer '{peer_id}' as a native tool (routed over the mesh; the "
                f"robot process holds the hardware). Actions: execute (blocking policy "
                f"rollout), start (async), status, stop. execute/start move REAL metal and "
                f"raise a human confirmation; status/stop are never gated."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "execute (blocking), start (async), status, stop",
                            "enum": ["execute", "start", "status", "stop"],
                            "default": "status",
                        },
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (required for execute/start)",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (required for execute/start)",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": (
                                "Which policy backend the peer runs: one of cosmos3, curobo, groot, kimodo, "
                                "lerobot_local, microduck, mock, moveit2, protomotions, remote, rl, wbc, wbc_gait. "
                                "groot (default, needs policy_port) dials a server; lerobot_local runs a local "
                                "checkpoint on the peer. An unknown name is refused by the peer listing its registry."
                            ),
                            "default": "groot",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds (positive, finite)",
                        },
                    },
                    "required": ["action"],
                }
            },
        }
    if kind == KIND_HOST:
        return {
            "name": tool_name,
            "description": (
                f"Robot peer '{peer_id}' (no joints announced yet) as a native tool. "
                f"Only status and stop are offered until it says what it is."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["status", "stop"],
                            "default": "status",
                        }
                    },
                    "required": ["action"],
                }
            },
        }
    return None


# ── invocation -> mesh command (pure) ────────────────────────────────────────

#: Fields the real-robot rail forwards. Everything else is refused by
#: mesh/security.validate_command anyway; dropping them here makes the
#: refusal happen with a better sentence and no wire round trip.
_REAL_FIELDS: dict[str, tuple[str, ...]] = {
    "execute": ("instruction", "policy_port", "policy_host", "policy_provider", "duration"),
    "start": ("instruction", "policy_port", "policy_host", "policy_provider", "duration"),
    "status": (),
    "stop": (),
}


def map_invocation(
    peer_id: str, kind: str, tool_input: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Proxy tool input -> the validated mesh command to send this peer.

    Returns ``(command, error)`` - exactly one is non-None. The command is a
    dict for ``bridge.send_cmd(peer_id, command)``; its shape is what
    ``mesh/security.validate_command`` accepts (execute/start/status/stop for
    robots, the sim_call envelope for sims).
    """
    tool_input = dict(tool_input or {})
    action = str(tool_input.pop("action", "") or "").strip()
    if not action:
        return None, "input needs an 'action'"

    if kind == KIND_SIM:
        if action in SIM_CALL_BLOCKED:
            return None, (
                f"{action!r} is a policy rollout and does not ride the sim_call rail "
                f"(its provider/repo allowlists live on execute/start). Ask the robot "
                f"tool to execute instead."
            )
        params = {k: v for k, v in tool_input.items() if v is not None}
        cmd: dict[str, Any] = {"action": "sim_call", "sim_action": action, "sim_params": params}
        # robot_name is a validated top-level field, not a sim param.
        if "robot_name" in params:
            cmd["robot_name"] = params.pop("robot_name")
        return cmd, None

    if kind in (KIND_REAL, KIND_HOST):
        allowed = _REAL_FIELDS if kind == KIND_REAL else {"status": (), "stop": ()}
        if action not in allowed:
            return None, f"unknown action {action!r} for this robot. Valid: {', '.join(sorted(allowed))}"
        cmd = {"action": action}
        for field in allowed[action]:
            if tool_input.get(field) is not None:
                cmd[field] = tool_input[field]
        return cmd, None

    return None, f"peer kind {kind!r} carries no tool"


# ── the AgentTool proxy ──────────────────────────────────────────────────────


def _agent_tool_base() -> type:
    from strands.types.tools import AgentTool  # local import: keep this module importable without strands

    return cast("type", AgentTool)


#: Verbs that are NEVER refused by the staleness gate. House law: a stale
#: presence read makes stopping MORE urgent, not less (one mesh-ingest blackout marked both REAL
#: arms stale 1430s while they were streaming - a refusal there lands on a
#: possibly MOVING arm). stop-class commands and status reads are attempted and
#: their real outcome reported, with a staleness NOTE attached; only actions
#: that START motion (execute/start) stay refused on stale presence.
NEVER_GATED: frozenset[str] = frozenset({"stop", "emergency_stop", "stop_all", "status"})


def stale_note(peer_id: str, peer: Mapping[str, Any] | None) -> str | None:
    """The staleness caveat attached to a stop/status that was SENT anyway.

    Returns None when presence is fresh. Never a refusal - the command has
    already been (or will be) delivered; this only tells the model the ack may
    not arrive and why, so a timeout reads as a presence gap, not a robot fault.
    """
    if not peer or not peer.get("stale"):
        return None
    age = peer.get("last_seen_age")
    if age is None:
        age = peer.get("age")
    when = f" (no presence for {float(age):.0f}s)" if isinstance(age, (int, float)) else ""
    return (
        f"NOTE: peer '{peer_id}' was STALE on the mesh when this command was sent{when}. "
        "The command was delivered anyway - stop-class and status commands are never refused "
        "on staleness - but the acknowledgement may be missing or delayed. Treat a timeout as "
        "a presence gap, not a robot fault."
    )


def stale_refusal(peer_id: str, peer: Mapping[str, Any] | None) -> str | None:
    """Why a command to this peer must not be sent, or None to proceed.

    DECISION, and it is deliberately NOT the ``fleet`` tool's one. ``fleet`` filtered stale peers
    out of its listing (agent_bridge.py:300, :361), so the honest move here looked like "drop the
    proxy". It is the wrong one: staleness is a fact about PRESENCE DELIVERY, not about the robot
    (a 26-minute mesh-ingest blackout marked both real arms stale for 1430s while they were
    connected and streaming the whole time, and it self-healed with no restart). Dropping tools on
    that would delete the agent's entire arm surface mid-blackout and rebuild it minutes later.

    So the proxy STAYS and the invocation refuses - for MOTION-STARTING actions only (
    stop-class verbs and status reads in ``NEVER_GATED`` are always attempted, with a
    ``stale_note`` attached, because refusing a stop on a stale-but-possibly-moving arm inverts
    the safety direction). A stale peer answers nothing,
    so ``send_cmd`` can only burn its 30s timeout and hand the model a bare timeout, which reads as
    a robot fault. The refusal names presence as the suspect instead.

    Checked at INVOCATION time, never baked into the tool list - a build-time flag would be a claim
    about a moment that has passed (which is also why ``fleet_signature`` still ignores stale).
    """
    if not peer or not peer.get("stale"):
        return None
    age = peer.get("last_seen_age")
    if age is None:
        age = peer.get("age")
    when = f" (no presence for {float(age):.0f}s)" if isinstance(age, (int, float)) else ""
    return (
        f"peer '{peer_id}' is STALE on the mesh{when}: the dashboard has not heard from it, so a "
        "command would only wait out its timeout. This is a PRESENCE fact, not a robot fault - the "
        "device may be fine while mesh delivery is stalled. Check the fleet screen (or /api/health's "
        "forwarded counter) and retry once it reports fresh; do not report the robot as broken."
    )


def build_peer_tools(
    peers: Mapping[str, Mapping[str, Any]],
    send_cmd: Callable[..., dict[str, Any]],
    peer_state: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> list[Any]:
    """One proxy AgentTool per tool-worthy fleet peer, names collision-free.

    ``send_cmd(peer_id, command, timeout=..., source="agent")`` is the
    dashboard bridge's sender - injected so the factory stays pure and the
    proxies stay testable with a fake.

    ``peer_state(peer_id)`` is the LIVE presence reader: the proxy asks it
    on every invocation so a peer that went stale after the tool was built
    refuses with a presence sentence instead of a 30s timeout. Omit it and the
    proxies keep their ungated behaviour - no gate, every call goes to the wire.
    """
    AgentTool = _agent_tool_base()

    class PeerProxyTool(AgentTool):  # type: ignore[misc,valid-type]
        """A fleet peer, presented to the agent as the robot itself."""

        def __init__(self, peer_id: str, kind: str, spec: dict[str, Any]) -> None:
            super().__init__()
            self._peer_id = peer_id
            self._kind = kind
            self._spec = spec

        @property
        def tool_name(self) -> str:
            return cast(str, self._spec["name"])

        @property
        def tool_spec(self) -> dict[str, Any]:
            return self._spec

        @property
        def tool_type(self) -> str:
            return "robot"

        @property
        def peer_id(self) -> str:
            """The fleet peer this proxy is bound to - the motion gate's target."""
            return self._peer_id

        @property
        def peer_kind(self) -> str:
            return self._kind

        async def stream(
            self, tool_use: Mapping[str, Any], invocation_state: dict[str, Any], **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            from strands.types._events import ToolResultEvent

            tool_use_id = tool_use.get("toolUseId", "")
            cmd, err = map_invocation(self._peer_id, self._kind, tool_use.get("input") or {})
            if err is not None:
                yield ToolResultEvent({"toolUseId": tool_use_id, "status": "error", "content": [{"text": err}]})
                return
            staleness_note: str | None = None
            if peer_state is not None:
                try:
                    live = peer_state(self._peer_id)
                except Exception:  # noqa: BLE001 - an unreadable snapshot must not block a command
                    live = None
                # stop-class verbs and status reads are NEVER refused on
                # staleness - the raw requested action decides (a sim's stop maps
                # to sim_call, so cmd["action"] would hide it).
                requested = str((tool_use.get("input") or {}).get("action") or "").strip()
                if requested in NEVER_GATED:
                    staleness_note = stale_note(self._peer_id, live)
                else:
                    refusal = stale_refusal(self._peer_id, live)
                    if refusal is not None:
                        yield ToolResultEvent(
                            {"toolUseId": tool_use_id, "status": "error", "content": [{"text": refusal}]}
                        )
                        return
            try:
                res = send_cmd(self._peer_id, cmd, timeout=30.0, source="agent")
            except Exception as exc:  # noqa: BLE001 - the wire's failure IS the result
                fail = f"mesh send to '{self._peer_id}' failed: {exc}"
                if staleness_note:
                    fail = f"{fail}\n{staleness_note}"
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": fail}],
                    }
                )
                return
            res = res if isinstance(res, dict) else {"result": res}
            status = str(res.get("status") or ("error" if res.get("error") else "success"))
            content = res.get("content")
            if not isinstance(content, list):
                content = [{"text": json.dumps(res, default=str)[:8000]}]
            if staleness_note:
                content = list(content) + [{"text": staleness_note}]
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use_id,
                    "status": "success" if status == "success" else "error",
                    "content": content,
                }
            )

    tools: list[Any] = []
    taken: set[str] = set()
    for peer_id, peer in peers.items():
        kind = classify_peer(peer_id, peer)
        if kind == KIND_SKIP:
            continue
        name = sanitize_tool_name(peer_id, taken)
        spec = peer_tool_spec(peer_id, kind, name)
        if spec is None:
            continue
        taken.add(name)
        tools.append(PeerProxyTool(peer_id, kind, spec))
    return tools


def expected_tool_names(peers: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The proxy tool names this fleet would produce - pure, no strands import.

    Lets agent_status answer honestly BEFORE the agent is lazily built
    (the badge used to hardcode ['fleet'], which lied until the first turn).
    """
    names: list[str] = []
    taken: set[str] = set()
    for peer_id, peer in peers.items():
        kind = classify_peer(peer_id, peer)
        if kind == KIND_SKIP:
            continue
        name = sanitize_tool_name(peer_id, taken)
        taken.add(name)
        names.append(name)
    return names


def fleet_signature(peers: Mapping[str, Mapping[str, Any]]) -> frozenset[tuple[str, str]]:
    """What the proxy surface depends on: the set of (peer_id, kind).

    get_agent compares this at call time against the signature the agent was
    built with - a changed fleet (join/leave/reclassify) rebuilds the agent so
    the tool list follows the mesh. Presence details beyond kind do not
    matter to the tools, so they do not churn the agent.
    """
    out = set()
    for peer_id, peer in peers.items():
        kind = classify_peer(peer_id, peer)
        if kind != KIND_SKIP:
            out.add((peer_id, kind))
    return frozenset(out)


def motion_actions_for(tools: list[Any]) -> dict[str, frozenset[str]]:
    """The MOTION_ACTIONS entries these proxies need - derived, never hand-kept.

    Only REAL-arm proxies appear, and only their motion verbs: sims never
    enter the table (their rail is structurally sim-only and peer_is_physical
    exempts them anyway), host proxies offer no motion verbs, and stop/status
    are never gated. Deriving the table from the built tools means the gate
    and the tool surface cannot drift apart.
    """
    return {t.tool_name: frozenset({"execute", "start"}) for t in tools if getattr(t, "peer_kind", None) == KIND_REAL}
