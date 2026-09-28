"""``/api/config`` - one composite document the Settings drawer reads and writes.

``GET`` folds the settings tree, the agent's model and prompt, the mesh session's
facts, the policy catalog and the ``.env`` view into one snapshot so the page has a
single fetch to reconcile. ``POST`` applies a patch through :mod:`config_api`,
which grades every key, never echoes a secret back, and says per field whether
the change is live now or waits for a mesh restart.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from strands_robots.dashboard import access, agent_console, config_api

router = APIRouter(prefix="/api", tags=["config"])


def _agent_status() -> dict[str, Any]:
    return {"model": agent_console.model_id(), "asks_first": sorted(agent_console.MOTION_TOOLS)}


@router.get("/config")
async def get_config(request: Request, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The composite document: settings, agent, voice, mesh, policies, env."""
    bridge = getattr(request.app.state, "bridge", None)
    return await asyncio.to_thread(lambda: config_api.snapshot(bridge=bridge, agent_status=_agent_status()))


@router.post("/config")
async def post_config(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Apply a graded patch; says per key whether it is live now or needs a mesh restart."""
    # Writing settings from a loopback-trusted session is only honoured from the
    # loopback itself, the same rule ``/api/settings`` applies.
    if who.get("via") == "loopback" and not access.peer_is_loopback(request):
        raise HTTPException(401, "sign in required")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    result = await asyncio.to_thread(config_api.apply, body)
    if result["errors"] and not result["applied"] and not result["env_written"]:
        raise HTTPException(422, "; ".join(result["errors"]))
    if result["restart_required"] and body.get("restart_mesh"):
        bridge = getattr(request.app.state, "bridge", None)
        if bridge is not None:
            ok = await asyncio.to_thread(bridge.restart)
            request.app.state.mesh_online = bool(ok)
            result["mesh_restart"] = {"ok": bool(ok), "mesh_online": bool(ok)}
    return result
