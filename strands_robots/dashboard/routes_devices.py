"""``/api/devices`` - serial ports and cameras on this machine, and the robots spawned for them.

The :class:`~strands_robots.dashboard.device_manager.DeviceManager` does the work: it scans
USB serial ports and camera indices, spawns a robot child process per port (a managed child
that joins the mesh), stops it, keeps its log ring buffer, and remembers boards by USB serial
so a replugged arm comes back on its own. These routes are the HTTP face of that roster.

Spawning a physical robot has physical consequences, so the route keeps the branch's rails:
an unknown robot or an unspawnable mode is a 422 before any process exists, a port another
process already holds is refused by ``bus_claim`` inside ``DeviceManager.spawn`` (never
started blind), and every spawn, despawn and camera change lands in the mesh bridge's
activity trail, because "who started this peer" is as unanswerable as "who moved that arm"
without it.

The mesh bridge is optional here: the roster works without it (nothing joins a mesh that is
not running), and the audit and the "which camera is streaming" evidence simply stay empty.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response

from strands_robots.dashboard import access, consent
from strands_robots.dashboard.cameras import CameraUnavailable
from strands_robots.dashboard.churn_guard import ChurnGuard
from strands_robots.dashboard.device_manager import (
    AUTOSPAWN_POLL_S,
    DeviceManager,
    respawn_payload,
    validate_cameras,
    validate_motor_model,
    validate_port,
    validate_spawn,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["devices"])


# ---------------------------------------------------------------------------
# helpers: the bridge is another lane's state, and may be absent
# ---------------------------------------------------------------------------


def _bridge(request: Request) -> Any | None:
    return getattr(request.app.state, "bridge", None)


def _devices(request: Request) -> DeviceManager:
    dm = getattr(request.app.state, "devices", None)
    if dm is None:
        raise HTTPException(503, "device roster not attached (routes_devices.attach was not called)")
    return cast("DeviceManager", dm)


def _audit(bridge: Any | None, action: str, *, target: str, detail: str | None = None, ok: bool) -> None:
    """Land one lifecycle entry in the activity trail; a missing bridge is a missing trail, not a crash."""
    record = getattr(bridge, "record_activity", None)
    if record is None:
        return
    try:
        record("api", action, target=target, detail=detail, ok=ok)
    except Exception as e:  # noqa: BLE001 - the audit must never veto the action it describes
        logger.warning("activity trail refused %s %s: %r", action, target, e)


def _live_camera_names(bridge: Any | None) -> dict[str, list[str]]:
    """peer_id -> camera names the mesh has actually seen frames for."""
    snapshot_fn = getattr(bridge, "snapshot", None)
    if snapshot_fn is None:
        return {}
    try:
        snapshot = snapshot_fn() or {}
    except Exception as e:  # noqa: BLE001 - a broken snapshot means "no evidence", not a 500
        logger.warning("mesh snapshot unavailable for camera liveness: %r", e)
        return {}
    return {
        peer_id: list((entry.get("cameras") or {}).keys())
        for peer_id, entry in (snapshot.get("peers") or {}).items()
        if isinstance(entry, dict)
    }


def _is_up(bridge: Any | None) -> Callable[[str], bool]:
    def is_up(peer_id: str) -> bool:
        return peer_id in (getattr(bridge, "peers", None) or {})

    return is_up


def _camera_http_error(index: int, exc: Exception) -> HTTPException:
    """Turn a camera fault into the status the frontend can act on, with the reason in the body.

    A camera nobody has is a 404 (the request named nothing), one another robot is streaming is a
    409, one macOS will not hand over is a 403, a missing OpenCV is a 501 that names the extra,
    and anything else is a 503.
    """
    if isinstance(exc, PermissionError):
        return HTTPException(409, {"error": str(exc), "index": index})
    if isinstance(exc, ImportError):
        return HTTPException(
            501,
            {
                "error": f"camera preview needs OpenCV ({exc}); install the dashboard extra: "
                "pip install 'strands-robots[dashboard]' (or opencv-python-headless)",
                "index": index,
            },
        )
    if isinstance(exc, CameraUnavailable):
        body = {"error": str(exc), "index": index, "state": exc.state, "reason": exc.reason, "remedy": exc.remedy}
        status = {"absent": 404, "blocked": 403}.get(exc.state, 503)
        return HTTPException(status, body)
    return HTTPException(503, {"error": str(exc) or exc.__class__.__name__, "index": index})


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/devices")
async def devices(request: Request, refresh: bool = False, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Local USB serial ports (servo buses) + cameras + managed robots."""
    # The mesh's frame bookkeeping is the evidence for "in use": a camera in a child's config
    # that never delivered a frame is assigned, not streaming, and the difference is what the
    # operator has to act on.
    dm = _devices(request)
    return cast(
        "dict[str, Any]",
        await asyncio.to_thread(dm.devices, refresh, _live_camera_names(_bridge(request))),
    )


@router.get("/devices/profiles")
async def device_profiles(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Remembered USB device profiles, keyed by board serial number."""
    dm = _devices(request)
    return {
        "profiles": dm.profiles.all(),
        "path": dm.profiles.path,
        "autospawn": getattr(request.app.state, "autospawn_task", None) is not None,
    }


@router.get("/devices/arm-role")
async def arm_role(
    request: Request, port: str, model: str = "sts3215", _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Measure whether the arm on ``port`` is a leader or a follower, and remember it."""
    dm = _devices(request)
    # Both strings reach a child's argv; a bad shape is the caller's error (422), not a bus fault.
    for problem in (validate_port(port), validate_motor_model(model)):
        if problem is not None:
            raise HTTPException(422, problem)
    try:
        # Measures AND remembers (keyed by USB serial, never by /dev name - the OS reassigns those).
        return cast("dict[str, Any]", await asyncio.to_thread(dm.measure_arm_role, port, model))
    except PermissionError as e:
        raise HTTPException(409, str(e)) from e
    except Exception as e:  # noqa: BLE001 - bus faults become HTTP, not tracebacks
        raise HTTPException(503, f"could not read {port}: {e}") from e


@router.get("/devices/camera/{index}/preview")
async def camera_preview(request: Request, index: int, _: dict = Depends(access.require_session)) -> Response:
    """One JPEG frame from an unclaimed camera index."""
    dm = _devices(request)
    try:
        jpeg = await asyncio.to_thread(dm.preview_frame, index, _live_camera_names(_bridge(request)))
    except Exception as e:  # noqa: BLE001 - camera faults become HTTP, not tracebacks
        raise _camera_http_error(index, e) from e
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/devices/camera/{index}/modes")
async def camera_modes(request: Request, index: int, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Resolutions and frame rates the camera at ``index`` actually delivers."""
    dm = _devices(request)
    try:
        return cast(
            "dict[str, Any]",
            await asyncio.to_thread(dm.probe_modes, index, _live_camera_names(_bridge(request))),
        )
    except Exception as e:  # noqa: BLE001 - camera faults become HTTP, not tracebacks
        raise _camera_http_error(index, e) from e


async def _spawn(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """The one spawn path: validation, the bus-claim gate, the settle window, the audit trail."""
    dm = _devices(request)
    bridge = _bridge(request)
    robot_name = body.get("robot_name")
    if not robot_name:
        raise HTTPException(422, "robot_name required")
    mode = body.get("mode", "sim")
    # An unspawnable mode or an unknown robot is a bad REQUEST, answered before any process
    # exists.
    checked = await asyncio.to_thread(validate_spawn, robot_name, mode)
    if isinstance(checked, dict):
        _audit(bridge, "spawn", target=str(robot_name), detail=f"refused: {checked['error']}", ok=False)
        raise HTTPException(422, checked)
    result = await asyncio.to_thread(
        dm.spawn,
        robot_name,
        mode,
        body.get("peer_id"),
        body.get("port"),
        body.get("cameras"),
        body.get("robot_id"),
    )
    # A pid is not a running robot.
    peer_id = result.get("peer_id")
    if peer_id and "error" not in result:
        outcome = await asyncio.to_thread(dm.settle, peer_id, is_up=_is_up(bridge))
        result.update(outcome)
        if outcome.get("status") == "failed":
            # Surface it in the field every caller already reads, so a dead spawn cannot be
            # mistaken for a live one by any client.
            result["error"] = outcome.get("reason") or "the peer did not start"
            consent.attach_consent(result, result["error"], "\n".join(outcome.get("log_tail") or []))

    # Lifecycle lands in the audit trail: the auto-spawn watcher and the UI use this same route.
    _audit(
        bridge,
        "spawn",
        target=str(result.get("peer_id") or robot_name),
        detail=f"{robot_name} mode={mode}" + (f" -> {result['error']}" if result.get("error") else ""),
        ok="error" not in result,
    )
    error = result.get("error") or ""
    if "already running" in error:
        # A conflict with something that exists, not a bad request.
        raise HTTPException(409, result)
    if "port required" in error:
        # mode=real without a port describes no device: a bad request, not a spawn that failed.
        raise HTTPException(422, result)
    if "held by" in error or "in use" in error:
        # Another process owns the servo bus (bus_claim): refused, never started blind.
        raise HTTPException(409, result)
    return cast("dict[str, Any]", result)


@router.post("/devices/spawn")
async def spawn(request: Request, body: dict[str, Any], _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Start a robot child process (sim or real) and wait for it to join the mesh."""
    return await _spawn(request, body)


@router.post("/devices/spawn-remembered")
async def spawn_remembered(
    request: Request, body: dict[str, Any], _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Spawn the robot a port's remembered USB profile describes."""
    dm = _devices(request)
    port = str(body.get("port") or "").strip()
    if not port:
        raise HTTPException(422, "port required")
    profile = await asyncio.to_thread(dm.profile_for_port, port)
    payload = respawn_payload(profile, port)
    if payload.get("error"):
        raise HTTPException(404, payload["error"])
    moved = payload.pop("port_moved", None)
    # One spawn path, not two: the settle window, the consent attachment and the audit trail
    # all live in _spawn, and a second copy of them is a second thing to forget to fix.
    result = await _spawn(request, payload)
    result["respawned_from_profile"] = True
    if moved:
        # Said out loud because it is the operator's evidence that the board they are looking
        # at is the board that came up: same serial, new /dev path.
        result["port_moved"] = moved
    return result


@router.post("/devices/despawn")
async def despawn(request: Request, body: dict[str, Any], _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Stop a managed robot's child process."""
    dm = _devices(request)
    peer_id = body.get("peer_id")
    if not peer_id:
        raise HTTPException(422, "peer_id required")
    result = await asyncio.to_thread(dm.despawn, str(peer_id))
    _audit(_bridge(request), "despawn", target=str(peer_id), ok="error" not in result)
    if "error" in result and "unknown" in str(result["error"]):
        raise HTTPException(404, result)
    return cast("dict[str, Any]", result)


@router.post("/devices/{peer_id}/cameras")
async def reconfigure_cameras(
    request: Request, peer_id: str, body: dict[str, Any], _: dict = Depends(access.require_session)
) -> dict[str, Any]:
    """Respawn a managed robot with a new camera mapping (``null`` detaches all)."""
    dm = _devices(request)
    bridge = _bridge(request)
    if "cameras" not in body:
        raise HTTPException(422, "cameras required (a mapping, or null to detach all)")
    bad = validate_cameras(body.get("cameras"))
    if bad:
        raise HTTPException(422, bad)
    result = await asyncio.to_thread(dm.reconfigure_cameras, peer_id, body.get("cameras"))
    if "error" in result and not result.get("reconfigured"):
        _audit(bridge, "cameras", target=peer_id, detail=f"refused: {result['error']}", ok=False)
        status = 404 if "unknown managed peer" in result["error"] else 409
        raise HTTPException(status, result)
    # Same honesty rail as spawn: a pid is not a running robot.
    outcome = await asyncio.to_thread(dm.settle, peer_id, is_up=_is_up(bridge))
    result.update(outcome)
    if outcome.get("status") == "failed":
        result["error"] = outcome.get("reason") or "the peer did not come back"
    _audit(
        bridge,
        "cameras",
        target=peer_id,
        detail=f"respawned with {len(body.get('cameras') or {})} camera(s)"
        + (f" -> {result['error']}" if result.get("error") else ""),
        ok="error" not in result,
    )
    return cast("dict[str, Any]", result)


@router.get("/devices/logs/{peer_id}")
async def device_logs(request: Request, peer_id: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Child-process output for one managed robot (ring buffer)."""
    dm = _devices(request)
    out = dm.logs(peer_id)
    if "error" in out:
        raise HTTPException(
            404,
            {
                "error": out["error"],
                "hint": "only locally spawned robots keep a log ring buffer",
                "managed_peers": sorted(dm.robots),
            },
        )
    return cast("dict[str, Any]", out)


# ---------------------------------------------------------------------------
# lifespan wiring
# ---------------------------------------------------------------------------


def _audit_autospawn(bridge: Any | None, did: dict[str, Any] | None) -> None:
    """Land the auto-spawn watcher's poll results in the activity trail."""
    if not did:
        return
    for peer_id in did.get("spawned") or []:
        _audit(bridge, "spawn", target=peer_id, detail="USB auto-spawn (board plugged in)", ok=True)
    for peer_id in did.get("despawned") or []:
        _audit(bridge, "despawn", target=peer_id, detail="USB auto-spawn (board unplugged)", ok=True)


async def _autospawn_loop(app: FastAPI, watcher: Any) -> None:
    while True:
        try:
            did = await asyncio.to_thread(watcher.poll)
            _audit_autospawn(getattr(app.state, "bridge", None), did)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - a blind watcher is worse than a loud one: say it, keep polling
            logger.warning("USB auto-spawn poll failed: %r", e)
        await asyncio.sleep(AUTOSPAWN_POLL_S)


def attach(app: FastAPI) -> None:
    """Give ``app`` a device roster and register its lifespan work.

    Sets ``app.state.devices`` and ``app.state.camera_churn``; appends to
    ``app.state.startup_hooks`` the USB auto-spawn start (a board with a saved profile comes up
    on its own, an unplugged one is stopped, unknown boards are only reported) and to
    ``app.state.shutdown_hooks`` the roster shutdown. The coordinator runs those lists from the
    app lifespan. Auto-spawn needs the mesh bridge's peer table to judge "came up"; without
    ``app.state.bridge`` it is skipped, and said so.
    """
    app.state.devices = DeviceManager()
    app.state.camera_churn = ChurnGuard()
    startup: list[Any] = getattr(app.state, "startup_hooks", None) or []
    shutdown: list[Any] = getattr(app.state, "shutdown_hooks", None) or []
    app.state.startup_hooks = startup
    app.state.shutdown_hooks = shutdown

    async def _start_autospawn() -> None:
        bridge = getattr(app.state, "bridge", None)
        if bridge is None:
            logger.info("USB auto-spawn skipped: no mesh bridge on this app")
            return
        watcher = app.state.devices.start_autospawn(peer_ids=lambda: list(getattr(bridge, "peers", None) or {}))
        if watcher is not None:
            app.state.autospawn_task = asyncio.create_task(_autospawn_loop(app, watcher))

    async def _stop_devices() -> None:
        task = getattr(app.state, "autospawn_task", None)
        if task is not None:
            task.cancel()
            app.state.autospawn_task = None
        await asyncio.to_thread(app.state.devices.shutdown)

    startup.append(_start_autospawn)
    shutdown.append(_stop_devices)
