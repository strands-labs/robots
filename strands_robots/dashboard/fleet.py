"""``/api/fleet`` - what the registry knows, plus the mesh peers this process has heard.

Two sources, both read without side effects:

* :func:`strands_robots.registry.robots.list_robots` - every registered robot
  with its category, joint count and whether it has a sim asset / a hardware
  backend. ``model_local`` says whether the sim asset is already on this disk
  (``resolve_model`` searches local paths only - it never downloads).
* :func:`strands_robots.mesh.session.get_peers` - the in-process peer table,
  when the mesh extra is importable. The dashboard does not join the mesh
  itself here; it reports what this process already knows, and says ``"off"``
  otherwise. Joining is :mod:`strands_robots.mesh`'s decision (opt-in via
  ``STRANDS_MESH``), not a side effect of opening a web page.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from strands_robots.dashboard import access

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["fleet"])


def registry_robots(mode: str = "all") -> list[dict[str, Any]]:
    """The registry's robots with a ``model_local`` flag per sim-capable entry."""
    from strands_robots.registry.robots import list_robots
    from strands_robots.simulation.model_registry import resolve_model

    out: list[dict[str, Any]] = []
    for entry in list_robots(mode=mode):
        row = dict(entry)
        if row.get("has_sim"):
            try:
                row["model_local"] = resolve_model(row["name"]) is not None
            except Exception:  # a broken asset dir is a row, not an outage
                logger.debug("resolve_model failed for %s", row["name"], exc_info=True)
                row["model_local"] = False
        out.append(row)
    return out


def mesh_peers() -> dict[str, Any]:
    """The peers this process knows, or ``{"status": "off"}`` when there is no mesh."""
    try:
        from strands_robots.mesh import session
    except Exception:
        return {"status": "off", "reason": "mesh extra not installed"}
    try:
        active = session.get_session() is not None
        peers = session.get_peers()
    except Exception:
        logger.debug("mesh peer table unreadable", exc_info=True)
        return {"status": "off", "reason": "peer table unreadable"}
    return {"status": "on" if active else "off", "peers": peers}


@router.get("/fleet")
async def fleet(mode: str = "all", _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Every robot the registry knows and every mesh peer this process has heard."""
    from strands_robots.registry.robots import LIST_ROBOTS_MODES

    if mode not in LIST_ROBOTS_MODES:
        raise HTTPException(400, f"mode must be one of {', '.join(LIST_ROBOTS_MODES)}")
    robots = registry_robots(mode)
    return {"robots": robots, "count": len(robots), "mesh": mesh_peers()}


@router.get("/robots/{name}")
async def robot(name: str, _: dict = Depends(access.require_session)) -> dict[str, Any]:
    """One registry entry in full, with the resolved local model path if any."""
    from strands_robots.registry.robots import get_robot, resolve_name
    from strands_robots.simulation.model_registry import resolve_model

    entry = get_robot(name)
    if entry is None:
        raise HTTPException(404, f"unknown robot: {name}")
    canonical = resolve_name(name)
    model = resolve_model(canonical) if entry.get("asset") else None
    return {"name": canonical, "entry": entry, "model_path": model}
