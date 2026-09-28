"""The dashboard's window on the LIVE mesh fleet.

Everything here reads or drives ``request.app.state.bridge``, a
:class:`~strands_robots.dashboard.mesh_bridge.MeshBridge`: the fleet snapshot,
per-peer teleop and task commands, the two safety rails, mesh endpoints, the
activity trail and the camera frames. HTTP routes live on :data:`router`
(prefix ``/api``); the two websockets (``/ws/mesh`` fan-out and
``/ws/camera/{peer_id}/{cam}`` frame push) live on :data:`ws_router`, which
:func:`attach` mounts because a websocket path carries no ``/api`` prefix.

Two rules the routes share with the rest of the package:

* every route takes ``access.require_session``; a websocket runs the same
  check by hand and is closed with ``4401`` on failure, the way ``/ws/agent``
  does;
* a task that would move real hardware is refused unless the browser
  confirmed it or an operator granted unattended motion
  (:mod:`~strands_robots.dashboard.agent_motion`). The refusal carries a
  ``needs_consent`` block (:func:`consent.attach_consent`) so the page can
  offer the grant instead of showing a wall. Stopping is never gated.

Mesh-backed safety is mounted under ``/api/mesh/safety/*``: ``/api/safety/*``
is the sim-only rail in :mod:`~strands_robots.dashboard.routes_sim`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from typing import Any, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from strands_robots.dashboard import access, agent_motion, consent, lan_hint, settings
from strands_robots.dashboard.mesh_bridge import (
    PEER_STALE_S,
    MeshBridge,
    command_succeeded,
    peer_is_known,
    route_task_target,
    silent_arms,
    stop_outcome,
)
from strands_robots.dashboard.refusals import RefusalTally
from strands_robots.dashboard.routes_auth import _json_body
from strands_robots.dashboard.teleop_health import published_frames, teleop_health
from strands_robots.dashboard.ws_observability import (
    CloseLogThrottle,
    cap_note,
    close_line,
    close_verdict,
    fps_cap,
)
from strands_robots.simulation.models import registry_entry
from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["mesh"])
ws_router = APIRouter(tags=["mesh"])

#: Command keys ``mesh.security.validate_command`` admits for execute/start, minus the ones this
#: module sets itself (action / instruction / policy_provider / duration). validate_command builds
#: its output from a strict per-action allowlist and drops everything else silently, so a body
#: key outside this tuple would look accepted and arrive empty.
WIRE_CMD_KEYS: tuple[str, ...] = (
    "policy_host",
    "policy_port",
    "policy_type",
    "server_address",
    "model_path",
    "pretrained_name_or_path",
    "robot_name",
    "target_pose",
    "target_joints",
    "world_update",
    "control_frequency",
    "action_horizon",
    "fast_mode",
    "n_steps",
)

#: The mesh settings ``POST /api/mesh/config`` may change.
MESH_CONFIG_KEYS: tuple[str, ...] = ("connect", "listen", "port", "backend", "camera_hz", "policy_type_allow")

#: Camera tiles are paced at this rate at most, so wifi phones stay happy.
CAMERA_TICK_S = 1 / 15


# ----------------------------------------------------------------------
# App wiring
# ----------------------------------------------------------------------


def attach(app: FastAPI) -> None:
    """Give *app* the state these routes read, and the lifespan work that joins the mesh.

    Idempotent: a bridge already on ``app.state`` is kept (tests install their
    own), and the websocket router is mounted once.
    """
    if getattr(app.state, "bridge", None) is None:
        app.state.bridge = MeshBridge()
    app.state.mesh_online = False
    if getattr(app.state, "refusals", None) is None:
        app.state.refusals = RefusalTally()
    if getattr(app.state, "camera_close_log", None) is None:
        app.state.camera_close_log = CloseLogThrottle()
    if getattr(app.state, "mesh_ingest_prev", None) is None:
        app.state.mesh_ingest_prev = None

    # A flag rather than a scan of app.routes: FastAPI may hold an included router lazily, so the
    # websocket paths are not visible there until the first request.
    if not getattr(app.state, "mesh_ws_mounted", False):
        app.include_router(ws_router)
        app.state.mesh_ws_mounted = True

    async def _start() -> None:
        loop = asyncio.get_running_loop()
        try:
            app.state.mesh_online = await asyncio.to_thread(app.state.bridge.start, loop)
            app.state.mesh_error = None
        except Exception as exc:  # noqa: BLE001 - the dashboard stays up; the page says why the mesh is not
            app.state.mesh_online = False
            app.state.mesh_error = f"{type(exc).__name__}: {exc}"
            logger.warning("mesh session not started: %s", app.state.mesh_error)

    async def _stop() -> None:
        await asyncio.to_thread(app.state.bridge.stop)

    for name, hook in (("startup_hooks", _start), ("shutdown_hooks", _stop)):
        hooks = getattr(app.state, name, None)
        if hooks is None:
            hooks = []
            setattr(app.state, name, hooks)
        hooks.append(hook)


def _bridge(request: Request | WebSocket) -> MeshBridge:
    return cast("MeshBridge", request.app.state.bridge)


def _devices(request: Request) -> Any | None:
    """The device manager when the devices lane is mounted, else None."""
    return getattr(request.app.state, "devices", None)


def _managed_robots(request: Request) -> dict[str, Any]:
    dm = _devices(request)
    robots = getattr(dm, "robots", None) if dm is not None else None
    return dict(robots) if isinstance(robots, dict) else {}


def require_peer(request: Request, peer_id: str) -> None:
    """404 for a peer that was never in the fleet, before spending the RPC."""
    bridge = _bridge(request)
    managed = _managed_robots(request)
    if peer_is_known(peer_id, getattr(bridge, "peers", None) or {}, managed):
        return
    known = sorted(set(getattr(bridge, "peers", None) or {}) | set(managed))
    raise HTTPException(
        404,
        {
            "error": f"no peer {peer_id!r} in the fleet",
            "hint": "GET /api/fleet lists the peers that can be commanded",
            "known_peers": known,
        },
    )


# ----------------------------------------------------------------------
# Fleet
# ----------------------------------------------------------------------


@router.get("/fleet")
async def fleet(request: Request, mode: str = "all", _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The whole fleet as the mesh sees it, plus whether the mesh is delivering at all.

    Roles ride along inside ``snapshot()`` (``bridge.peer_annotations``), so this
    route and the ``/ws/mesh`` stream cannot disagree about which arm is the
    leader. ``mesh_online`` is the bridge's belief; ``mesh_ingest`` is the
    falsifiable reading (freshest presence age + fan-out since the last poll).
    """
    from strands_robots.dashboard.health_ingest import mesh_ingest

    app = request.app
    bridge = _bridge(request)
    snapshot = bridge.snapshot()
    coalesce = bridge.coalesce_stats()
    # A GIL-atomic copy: ``bridge.peers`` is mutated on the zenoh presence thread, and
    # ``mesh_ingest`` iterates it here on the event loop at the page's poll cadence.
    ingest, app.state.mesh_ingest_prev = mesh_ingest(
        dict(bridge.peers),
        coalesce,
        time.time(),
        getattr(app.state, "mesh_ingest_prev", None),
        stale_after=PEER_STALE_S,
    )
    from strands_robots.dashboard.fleet import mesh_peers, registry_robots
    from strands_robots.registry.robots import LIST_ROBOTS_MODES

    if mode not in LIST_ROBOTS_MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(LIST_ROBOTS_MODES)}")
    robots = registry_robots(mode)
    # ``mesh.status`` is the contract the pre-SPA route published ("on" | "off"):
    # a client reads that one key to learn whether peers can be listed at all.
    # The bridge's richer ``mesh_info()`` rides along under the same key.
    posture = mesh_peers()
    mesh_info = dict(snapshot.get("mesh") or {})
    mesh_info["status"] = "on" if (mesh_info.get("online") or posture["status"] == "on") else "off"
    if "reason" in posture and mesh_info["status"] == "off":
        mesh_info["reason"] = posture["reason"]
    out: dict[str, Any] = {
        **snapshot,
        "mesh": mesh_info,
        "robots": robots,
        "count": len(robots),
        "mesh_online": bool(getattr(app.state, "mesh_online", False)),
        "mesh_error": getattr(app.state, "mesh_error", None),
        "dashboard_peer_id": bridge.peer_id,
        "peer_count": len(snapshot.get("peers") or {}),
        "mesh_coalesce": coalesce,
        "mesh_ingest": ingest,
    }
    joint_streams = silent_arms(bridge.peers)
    if joint_streams is not None:
        out["joint_streams"] = joint_streams
    return out


@router.get("/network/hint")
async def network_hint(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Is the caller on this machine's LAN, and which LAN URLs reach this dashboard?"""
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    client_ip = (fwd.split(",")[0].strip() if fwd else None) or (request.client.host if request.client else None)
    own: list[str] = []
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            own.extend(a.address for a in addrs if a.address)
    except Exception:  # noqa: BLE001 - psutil is optional and platform-specific; the hint degrades
        pass
    port = int(getattr(request.app.state, "port", None) or 8090)
    return cast("dict[str, Any]", lan_hint.hint(client_ip, own, port))


@router.get("/robots/registry")
async def registry(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Every robot the registry knows, as the spawn form lists them."""
    from strands_robots.registry import list_robots

    try:
        robots = await asyncio.to_thread(list_robots)
    except Exception as exc:  # noqa: BLE001 - registry read is best-effort
        raise HTTPException(500, f"registry unavailable: {exc}") from exc
    return {"robots": robots}


@router.get("/activity")
async def activity(request: Request, limit: int = 100, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Recent commands and safety events, newest first."""
    return {"activity": _bridge(request).activity_log(limit=max(1, min(limit, 300)))}


# ----------------------------------------------------------------------
# Teleop
# ----------------------------------------------------------------------


@router.get("/robots/{peer_id}/teleop")
async def teleop_status(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Live teleop health for one peer: publisher/receiver rates, drops, slew rejections.

    The counters alone lie: a follower refusing EVERY frame reports
    ``running: true`` and the reason exists only in its child log. ``health``
    turns the counters (and that log, when the peer is ours) into a sentence, and
    a refusal we can continue from arrives with its consent request attached.
    """
    require_peer(request, peer_id)
    bridge = _bridge(request)
    result = await bridge.send_cmd_async(peer_id, {"action": "teleop_status"}, timeout=10.0)
    inner = result.get("result") if isinstance(result, dict) else None
    log_tail = None
    managed = _managed_robots(request).get(peer_id)
    if managed is not None:
        log_tail = list(getattr(managed, "logs", []) or [])[-40:]
    health = teleop_health(inner if inner is not None else result, log_tail)
    # "Nothing is arriving" is not yet an answer: it could be a leader that never started, or two
    # peers that are not meeting.
    silent = {k: v for k, v in health.get("receivers", {}).items() if v.get("state") == "silent"}
    if silent:
        counted: dict[str, int] = {}
        for key in silent:
            source, _, device = key.partition("/")
            if not source or source == peer_id:
                continue
            try:
                src = await bridge.send_cmd_async(source, {"action": "teleop_status"}, timeout=10.0)
            except Exception as exc:  # noqa: BLE001 - a quiet leader is data too
                logger.debug("could not ask leader %s about its publisher: %r", source, exc)
                continue
            frames = published_frames(src.get("result") if isinstance(src, dict) else src, device or "leader")
            if frames is not None:
                counted[key] = frames
        if counted:
            health = teleop_health(inner if inner is not None else result, log_tail, counted)
    worst = health.get("worst") or {}
    if worst.get("state") == "refusing" and log_tail:
        for line in reversed(log_tail):
            req = consent.classify_refusal(str(line))
            if req is not None:
                health["needs_consent"] = req.as_dict()
                break
    return {"peer_id": peer_id, "result": result, "health": health}


@router.post("/robots/{peer_id}/teleop/publish")
async def teleop_publish(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Start publishing a leader arm's input stream."""
    require_peer(request, peer_id)
    body = await _json_body(request)
    cmd: dict[str, Any] = {"action": "teleop_publish"}
    for field in ("device_name", "hz", "robot_name"):
        if body.get(field) is not None:
            cmd[field] = body[field]
    result = await _bridge(request).send_cmd_async(peer_id, cmd, timeout=30.0)
    return {"peer_id": peer_id, "result": result}


@router.post("/robots/{peer_id}/teleop/receive")
async def teleop_receive(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Point a follower (real or sim twin) at a leader's input stream.

    A SIM twin can follow a REAL leader arm: practice on the twin before metal.
    """
    require_peer(request, peer_id)
    body = await _json_body(request)
    source = str(body.get("source_peer_id") or "").strip()
    if not source:
        raise HTTPException(422, "source_peer_id required")
    # The leader is a peer too: pointing a follower at a stream nobody publishes is a 45 s wait
    # ending in a shrug.
    require_peer(request, source)
    # Same consent rail as ``start_task``: on a real follower this POST is motion (the receiver
    # enables tracking and the arm snaps toward the leader's pose), so it needs the browser's
    # confirmation or the operator's standing grant. ``teleop_stop`` stays ungated.
    bridge = _bridge(request)
    verdict = task_gate(
        bridge.peers.get(peer_id), confirmed=body.get("confirmed"), target=peer_id, action="teleop_receive"
    )
    if not verdict["allowed"]:
        refusal: dict[str, Any] = {"error": verdict["reason"], "peer_id": peer_id, "ok": False, "verdict": verdict}
        consent.attach_consent(refusal, verdict, subject=peer_id)
        bridge.record_activity(
            "api", "teleop_receive", target=peer_id, detail="refused: motion not confirmed", ok=False
        )
        raise HTTPException(403, refusal)
    cmd = {
        "action": "teleop_receive",
        "source_peer_id": source,
        "device_name": body.get("device_name", "leader"),
    }
    # The first declare_subscriber on a peer can take >15 s (zenoh declare + gossip propagation):
    # not a deadlock, just slow.
    result = await bridge.send_cmd_async(peer_id, cmd, timeout=45.0)
    return {"peer_id": peer_id, "result": result}


@router.post("/robots/{peer_id}/teleop/stop")
async def teleop_stop(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Stop a peer's teleop publisher or receiver. Never gated."""
    require_peer(request, peer_id)
    body = await _json_body(request)
    cmd: dict[str, Any] = {"action": "teleop_stop"}
    if body.get("device_name"):
        cmd["device_name"] = body["device_name"]
    result = await _bridge(request).send_cmd_async(peer_id, cmd, timeout=10.0)
    return {"peer_id": peer_id, "result": result}


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------


def task_gate(peer: dict[str, Any] | None, *, confirmed: object, target: str, action: str = "task") -> dict[str, Any]:
    """May this POST start motion on *target*? Same shape as ``agent_motion_allowed``.

    ``action`` is one of :data:`agent_motion.GATED_ACTIONS`: a task, or pointing a follower
    at a leader stream (``teleop_receive``), which moves the follower the moment it lands.

    Three ways through, in order: the peer is provably a sim; the browser
    confirmed the click (``confirmed`` must be a JSON boolean, so a string
    ``"false"`` cannot select the confirmed posture); an operator granted
    unattended motion with :data:`agent_motion.MOTION_ENV` and did not also ask
    for per-task confirmation with :data:`agent_motion.TASK_CONFIRM_ENV`.
    Anything else is refused, and the refusal is one
    :func:`consent.classify_refusal` recognises.
    """
    verdict = agent_motion.agent_motion_allowed(action, peer=peer, target=target)
    if not verdict.get("physical"):
        return verdict
    if confirmed is not None and (text := boolean_flag_error(confirmed, "confirmed", action)):
        return {
            "allowed": False,
            "physical": True,
            "gated": True,
            "granted": False,
            "confirmed": False,
            "reason": (
                f"refused: {text} Nothing was sent. Send a JSON boolean "
                f'("confirmed": true), or press play on that robot\'s card, where the browser confirms.'
            ),
        }
    if confirmed is True:
        return {**verdict, "allowed": True, "confirmed": True, "reason": ""}
    if verdict.get("allowed") and not agent_motion.task_confirm_required():
        return verdict
    if verdict.get("allowed"):
        # Granted for the agent, but the operator also wants each task POST confirmed.
        shown = target or "that robot"
        return {
            "allowed": False,
            "physical": True,
            "gated": True,
            "granted": False,
            "confirmed": False,
            "reason": (
                f"refused: this dashboard is set to require a confirmation before a task starts real "
                f"motion on {shown}, and this request did not carry one. Nothing was sent. Press play on "
                f'{shown}\'s card, or send "confirmed": true in the body. Stopping is never gated.'
            ),
        }
    return verdict


@router.post("/robots/{peer_id}/task")
async def start_task(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Start a policy task on one peer; mirror it to a live ``<peer>-twin``."""
    require_peer(request, peer_id)
    bridge = _bridge(request)
    body = await _json_body(request)
    instruction = str(body.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(422, "instruction required")

    peer = bridge.peers.get(peer_id)
    verdict = task_gate(peer, confirmed=body.get("confirmed"), target=peer_id)
    if not verdict["allowed"]:
        refusal: dict[str, Any] = {"error": verdict["reason"], "peer_id": peer_id, "ok": False, "verdict": verdict}
        consent.attach_consent(refusal, verdict, subject=peer_id)
        bridge.record_activity("api", "task", target=peer_id, detail="refused: motion not confirmed", ok=False)
        raise HTTPException(403, refusal)

    try:
        duration = float(body.get("duration", 30.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "duration must be a number of seconds") from exc
    cmd: dict[str, Any] = {
        # Sim peers accept "start"; hardware peers accept both. "execute" is an explicit opt-in.
        "action": body.get("action") or "start",
        "instruction": instruction,
        "policy_provider": body.get("policy_provider", "mock"),
        "duration": duration,
    }
    for opt in WIRE_CMD_KEYS:
        if body.get(opt) is not None:
            cmd[opt] = body[opt]
    # A child sim peer cannot execute itself: "<parent>__<robot>" is routed to the parent with
    # robot_name, so the card's Run button and every API caller get the same fix.
    target, cmd = route_task_target(peer_id, cmd)
    from strands_robots.dashboard.task_timeout import task_ack_budget, timeout_verdict

    # "start" is answered by an immediate ack; "execute" blocks until the rollout ends.
    timeout_s, timeout_kind = task_ack_budget(cmd["action"], body.get("timeout"), duration)
    twin_id = f"{peer_id}-twin"
    twin = bridge.peers.get(twin_id)
    mirrored = bool(twin and not twin.get("stale"))
    if mirrored:
        t_target, t_cmd = route_task_target(twin_id, dict(cmd))
        # Fire-and-forget; the twin's progress streams on the mesh.
        request.app.state.twin_task = asyncio.create_task(bridge.send_cmd_async(t_target, t_cmd, timeout=timeout_s))
    result = await bridge.send_cmd_async(target, cmd, timeout=timeout_s)

    payload: dict[str, Any] = {
        "peer_id": peer_id,
        "routed_to": target if target != peer_id else None,
        "mirrored_to_twin": mirrored,
        # A response can arrive and still say ok=False. The UI needs one honest boolean.
        "ok": command_succeeded(result),
        "timeout_s": timeout_s,
        "result": result,
    }
    # A timeout is not "nothing happened": the command was delivered, so the robot may be loading
    # a policy and about to move. Say which wait ended.
    if isinstance(result, dict) and not payload["ok"] and str(result.get("error", "")).startswith("timeout"):
        tv = timeout_verdict(timeout_kind, timeout_s, target)
        result.update(tv)
        payload["motion_possible"] = True
        payload["timeout_kind"] = tv["timeout_kind"]
    if not payload["ok"] and isinstance(result, dict):
        # The refusal is inside the peer's answer, under whichever key that peer chose.
        consent.attach_consent(
            payload,
            result,
            result.get("error"),
            result.get("detail"),
            result.get("message"),
            result.get("reason"),
            subject=peer_id,
        )
    return payload


@router.post("/robots/{peer_id}/stop")
async def stop_task(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Stop one peer. Never gated."""
    require_peer(request, peer_id)
    result = await _bridge(request).send_cmd_async(peer_id, {"action": "stop"}, timeout=10.0)
    return {"peer_id": peer_id, **stop_outcome(result), "result": result}


@router.post("/robots/{peer_id}/twin")
async def toggle_twin(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Spawn or despawn a MuJoCo digital twin sim peer named ``<peer>-twin``.

    Tasks started via ``/api/robots/<peer>/task`` are mirrored to a live twin.
    """
    dm = _devices(request)
    if dm is None:
        raise HTTPException(503, "device manager not mounted - twins need the devices lane")
    body = await _json_body(request)
    twin_id = f"{peer_id}-twin"
    existing = registry_entry(dm.robots, twin_id)
    if existing and existing.alive():
        return cast("dict[str, Any]", await asyncio.to_thread(dm.despawn, twin_id))
    robot_name = body.get("robot_name")
    if not robot_name:
        peer = _bridge(request).peers.get(peer_id) or {}
        robot_name = (peer.get("presence") or {}).get("tool_name") or "so101"
    return cast("dict[str, Any]", await asyncio.to_thread(dm.spawn, robot_name, "sim", twin_id))


@router.get("/robots/{peer_id}/policy-fit")
async def policy_fit_route(
    request: Request,
    peer_id: str,
    repo_id: str = "",
    norm_tag: str = "",
    _: dict = Depends(access.require_session),
) -> dict[str, Any]:
    """Does the checkpoint at *repo_id* fit this peer's joints and cameras?"""
    require_peer(request, peer_id)
    from strands_robots.dashboard.checkpoints import declared_features
    from strands_robots.dashboard.policy_fit import policy_fit
    from strands_robots.dashboard.training import PathOutside, contain_checkpoint_ref

    # the same gate /api/checkpoints/features applies: a path-shaped repo_id must
    # sit under the training output or the lerobot cache before anything reads it
    try:
        repo_id = contain_checkpoint_ref(repo_id)
    except PathOutside as exc:
        raise HTTPException(400, exc.refusal()) from exc

    peer = _bridge(request).peers.get(peer_id) or {}
    joints = list((peer.get("state") or {}).get("joints") or {})
    cameras = list(peer.get("cameras") or {})
    feats = await asyncio.to_thread(declared_features, repo_id) if repo_id.strip() else {}
    verdict = policy_fit(
        input_features=feats.get("input_features"),
        output_features=feats.get("output_features"),
        joints=joints,
        cameras=cameras,
        physical=bool((peer.get("presence") or {}).get("hw")) or not peer,
        norm_tag=norm_tag,
        declared_norm_tags=feats.get("norm_tags"),
    )
    verdict["evidence"] = bool(feats) and bool(verdict["checked"])
    verdict["repo_id"] = repo_id
    verdict["policy_type"] = feats.get("policy_type")
    verdict["robot"] = {"joints": joints, "cameras": cameras}
    return cast("dict[str, Any]", verdict)


# ----------------------------------------------------------------------
# Safety (mesh-backed; /api/safety/* is the sim rail)
# ----------------------------------------------------------------------


@router.post("/mesh/safety/estop")
async def estop(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Fleet-wide stop: BOTH rails fire, results reported side by side.

    Per-peer ``stop`` commands ask every live peer to halt; the SIGNED rail
    then engages the fleet-wide lockout on every listening peer, which per-peer
    commands cannot do. ``lockout`` is what this dashboard is entitled to say
    about that lockout afterwards.
    """
    bridge = _bridge(request)
    peers = bridge.live_peers()
    stale = sorted(set(bridge.peers) - set(peers))
    results = await asyncio.gather(
        *(bridge.send_cmd_async(p, {"action": "stop"}, timeout=5.0, source="estop") for p in peers),
        return_exceptions=True,
    )
    per_peer: dict[str, dict[str, Any]] = {}
    for peer, raw in zip(peers, results, strict=True):
        result = raw if isinstance(raw, dict) else {"error": str(raw)}
        per_peer[peer] = {**stop_outcome(result), "result": result}
    counts = {"stopped": 0, "not_stopped": 0, "no_answer": 0}
    for info in per_peer.values():
        counts[info["state"]] = counts.get(info["state"], 0) + 1
    bridge.record_activity(
        "estop",
        "stop_all",
        target="fleet",
        detail=f"{counts['stopped']}/{len(peers)} confirmed stopped",
        ok=counts["stopped"] == len(peers) and bool(peers),
    )
    signed = await asyncio.to_thread(bridge.signed_estop)
    return {
        "targeted": peers,
        "stale_skipped": stale,
        "counts": counts,
        # True only when every live peer confirmed. Anything else and the UI must keep shouting.
        "all_stopped": bool(peers) and counts["stopped"] == len(peers),
        "stopped": per_peer,
        "signed_rail": {k: v for k, v in signed.items() if k != "responses"},
        "responses_received": signed.get("responses_received", 0),
        "peers_not_stopped": signed.get("peers_not_stopped", []),
        "lockout_engaged": bool(signed.get("lockout_engaged")),
        "lockout": bridge._lockout.as_fields(),
    }


@router.post("/mesh/safety/resume")
async def safety_resume(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Clear the fleet e-stop lockout with the operator override code."""
    body = await _json_body(request)
    code = str(body.get("override_code") or "").strip()
    if not code:
        raise HTTPException(422, "override_code required")
    bridge = _bridge(request)
    result = await asyncio.to_thread(bridge.signed_resume, code)
    bridge.record_activity(
        "resume",
        "safety_resume",
        target="fleet",
        detail=result.get("status", result.get("error", "?")),
        ok=result.get("status") == "ok",
    )
    return {**result, "lockout": bridge._lockout.as_fields()}


# ----------------------------------------------------------------------
# Mesh configuration
# ----------------------------------------------------------------------


async def _restart_mesh(request: Request, *, force: bool = False) -> dict[str, Any]:
    """Re-open the mesh session against the current settings."""
    managed = [pid for pid, r in _managed_robots(request).items() if r.alive()]
    if managed and not force:
        raise HTTPException(
            409,
            f"{len(managed)} locally spawned robot(s) hold the mesh session ({', '.join(managed)}). "
            "Despawn them first, or pass force=true to re-point anyway (they will keep using the old endpoints).",
        )
    settings.load(refresh=True)
    bridge = _bridge(request)
    online = await asyncio.to_thread(bridge.restart)
    request.app.state.mesh_online = online
    return {"mesh_online": online, "orphaned": managed if force else [], "mesh": bridge.mesh_info()}


@router.get("/mesh/config")
async def get_mesh_config(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Mesh posture: endpoints, auth mode, peer counts, the stored ``mesh`` settings."""
    return cast("dict[str, Any]", _bridge(request).mesh_info())


@router.post("/mesh/config")
async def post_mesh_config(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Persist mesh endpoints and (by default) re-point the session."""
    if who.get("via") == "loopback" and not access.peer_is_loopback(request):
        raise HTTPException(401, "sign in required")
    body = await _json_body(request)
    mesh = {k: v for k, v in body.items() if k in MESH_CONFIG_KEYS}
    ignored = sorted(k for k in body if k not in MESH_CONFIG_KEYS and k not in ("restart", "force"))
    changed, errors = await asyncio.to_thread(settings.update_strict, {"mesh": mesh})
    if errors:
        raise HTTPException(422, "; ".join(errors))
    result: dict[str, Any] = {"applied": changed, "changed": changed, "ignored": ignored, "errors": []}
    if body.get("restart", True):
        result["mesh_restart"] = await _restart_mesh(request, force=bool(body.get("force")))
    return result


@router.post("/mesh/restart")
async def restart_mesh(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Leave and rejoin the mesh with the settings on disk."""
    body = await _json_body(request)
    return await _restart_mesh(request, force=bool(body.get("force")))


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------


@router.get("/frame/{peer_id}/{cam}")
async def frame(request: Request, peer_id: str, cam: str, _: dict = Depends(access.require_session)) -> Response:
    """The latest JPEG for one camera, for a tile that cannot hold a socket."""
    f = _bridge(request).latest_frame(peer_id, cam)
    if f is None:
        raise HTTPException(404, "no frame yet")
    if not f.get("jpeg"):
        # Raw pixels served as image/jpeg render as a black rectangle, indistinguishable from a dead camera.
        raise HTTPException(415, f.get("error") or "frame is not displayable")
    return Response(content=f["jpeg"], media_type="image/jpeg")


# ----------------------------------------------------------------------
# WebSockets
# ----------------------------------------------------------------------


async def _admit(ws: WebSocket) -> bool:
    """Same admission as every route; a stranger is closed with 4401 and False is returned."""
    try:
        access.caller(ws)  # type: ignore[arg-type]  # WebSocket answers headers like a Request
    except HTTPException:
        await access.refuse_socket(ws, 4401)
        return False
    return True


async def _client_gone(ws: WebSocket) -> None:
    """Park on the socket's inbound channel until the client actually leaves."""
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


@ws_router.websocket("/ws/mesh")
async def ws_mesh(ws: WebSocket) -> None:
    """Fan the bridge's events out to one page: a snapshot first, then every event as it lands."""
    if not await _admit(ws):
        return
    await ws.accept()
    bridge = _bridge(ws)
    q = bridge.attach_queue()
    gone = asyncio.create_task(_client_gone(ws))
    try:
        await ws.send_text(json.dumps(bridge.snapshot()))
        while True:
            getter = asyncio.create_task(q.get())
            done, _ = await asyncio.wait({getter, gone}, return_when=asyncio.FIRST_COMPLETED)
            if gone in done:
                getter.cancel()
                break
            await ws.send_text(json.dumps(getter.result()))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        gone.cancel()
        bridge.detach_queue(q)


def _churn_verdict(ws: WebSocket, peer_id: str, cam: str) -> Any | None:
    """Ask the devices lane's ChurnGuard about this viewer, when it is mounted."""
    guard = getattr(ws.app.state, "camera_churn", None)
    if guard is None:
        return None
    from strands_robots.dashboard.churn_guard import viewer_identity

    token = access.presented_token(ws)  # type: ignore[arg-type]
    # A session token is per-login, so its digest identifies the viewer without the token ever
    # entering a key or a log line.
    subject = hashlib.sha256(token.encode()).hexdigest()[:12] if token else None
    return guard.note_open(
        viewer_identity(subject=subject, host=ws.client.host if ws.client else None, peer_id=peer_id, cam=cam)
    )


@ws_router.websocket("/ws/camera/{peer_id}/{cam}")
async def ws_camera(ws: WebSocket, peer_id: str, cam: str) -> None:
    """Push binary JPEG frames for one tile: only when a newer frame exists, paced at ~15 fps at most."""
    if not await _admit(ws):
        return
    await ws.accept()
    bridge = _bridge(ws)
    last_t: Any = None
    reported: str | None = None
    frames_sent = 0
    bytes_sent = 0
    started_at = time.monotonic()
    churn = _churn_verdict(ws, peer_id, cam)
    cap = fps_cap(ws.query_params.get("max_fps"))
    if churn is not None and churn.cap_fps is not None:
        cap = churn.cap_fps if cap is None else min(cap, churn.cap_fps)
    min_interval = None if cap is None else 1.0 / cap
    if churn is not None and churn.reason:
        # Say it on the tile, not only in the log: a silent throttle is indistinguishable from a slow camera.
        with contextlib.suppress(Exception):
            await ws.send_text(
                json.dumps(
                    {"type": "camera_error", "peer_id": peer_id, "cam": cam, "error": churn.reason, "throttled": True}
                )
            )
    last_sent_at: float | None = None
    gone = asyncio.create_task(_client_gone(ws))
    try:
        while not gone.done():
            if min_interval is not None and last_sent_at is not None:
                waited = time.monotonic() - last_sent_at
                if waited < min_interval:
                    # Skip WITHOUT consuming the frame: the newest one at the end of the wait is what
                    # the viewer wants, not this stale one.
                    await asyncio.sleep(min(min_interval - waited, CAMERA_TICK_S))
                    continue
            f = bridge.latest_frame(peer_id, cam)
            if f is not None and f.get("t") != last_t:
                last_t = f.get("t")
                if f.get("jpeg"):
                    await ws.send_bytes(f["jpeg"])
                    frames_sent += 1
                    bytes_sent += len(f["jpeg"])
                    last_sent_at = time.monotonic()
                elif f.get("error") and f["error"] != reported:
                    # One text frame per distinct problem: the tile can then say "raw frames, cannot
                    # decode" instead of going black.
                    reported = f["error"]
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "camera_error",
                                "peer_id": peer_id,
                                "cam": cam,
                                "error": f["error"],
                                "encoding": f.get("encoding"),
                            }
                        )
                    )
            await asyncio.sleep(CAMERA_TICK_S)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        gone.cancel()
        # Rate-limited per peer/camera, with the suppressed count carried forward, so a storm reads
        # as a storm instead of drowning the log it would explain.
        throttle = getattr(ws.app.state, "camera_close_log", None)
        log_now, suppressed = throttle.should_log(f"{peer_id}/{cam}") if throttle is not None else (True, 0)
        if log_now:
            verdict = close_verdict(
                frames_sent=frames_sent,
                lifetime_s=time.monotonic() - started_at,
                publishing=bridge.latest_frame(peer_id, cam) is not None,
                bytes_sent=bytes_sent,
            )
            churn_note = (
                f" [server churn cap: {churn.opens_in_window} opens/min from this viewer]"
                if churn is not None and churn.throttled
                else ""
            )
            line = close_line(
                peer_id=peer_id, cam=cam, verdict=verdict + cap_note(cap) + churn_note, suppressed=suppressed
            )
            (logger.info if frames_sent else logger.warning)(line)
