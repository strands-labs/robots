"""``/api/consent`` - the other half of a consent card.

A refusal anywhere in the dashboard carries a ``needs_consent`` block built by
:func:`consent.build_request`; the card the page shows from it ends here. ``POST``
grants exactly one kind for one subject (nothing else is settable) by writing the
smallest environment change to the ``.env`` file and this process; ``revoke``
takes it back. Both say that a child process started before the change keeps the
environment it was started with, because that is the retry that would otherwise
fail for a reason already known.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from strands_robots.dashboard import access, config_api, consent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["consent"])

_RESPAWN_NOTE = "A robot already running was started with the old environment - respawn it, then retry."


def _write_grant_patch(patch: dict[str, str]) -> list[str]:
    """Write a grant/revoke patch to ``.env`` through the consent allowlist, not the page one."""
    return config_api.upsert_env_file(patch, allowed_keys=consent.GRANT_ENV_KEYS)


def _request_from(body: Any) -> consent.ConsentRequest:
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    request = consent.build_request(str(body.get("kind", "")), body.get("subject"))
    if request is None:
        raise HTTPException(422, f"unknown consent kind; expected one of {', '.join(consent.KINDS)}")
    return request


def _audit(request: Request, scope: str, detail: str) -> None:
    bridge = getattr(request.app.state, "bridge", None)
    if bridge is not None:
        bridge.record_activity("api", "consent", target=scope, detail=detail, ok=True)


@router.get("/consent")
async def get_consent(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """What this machine grants right now, every kind in one place."""
    return {**consent.granted_state(os.environ), "env_file": str(config_api.ENV_FILE)}


@router.post("/consent")
async def post_consent(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Approve one refusal, by kind + subject."""
    if who.get("via") == "loopback" and not access.peer_is_loopback(request):
        raise HTTPException(401, "sign in required")
    req = _request_from(await request.json())
    patch = consent.env_patch(req, os.environ)
    if not patch:
        return {
            "granted": False,
            "scope": req.scope,
            "already_granted": True,
            "note": f"nothing to change - this is already allowed here. {_RESPAWN_NOTE}",
            "respawn_required": True,
        }
    # The grant variables are gate-bearing by design, so the page allowlist refuses them;
    # this route writes through its own closed set, derived from the refusal contract.
    for key, value in patch.items():
        problem = config_api.env_entry_error(key, value, allowed_keys=consent.GRANT_ENV_KEYS)
        if problem:
            raise HTTPException(422, problem)
    written = await asyncio.to_thread(_write_grant_patch, patch)
    os.environ.update(patch)
    try:  # the mesh caches its parsed allowlist per value, so re-read it
        from strands_robots.mesh import security as _mesh_security

        _mesh_security._hf_repo_allowlist()
    except Exception:  # noqa: BLE001 - a cache warm-up must not fail a grant
        logger.debug("consent: allowlist warm-up failed", exc_info=True)
    _audit(request, req.scope, f"approved: {', '.join(req.grants)}")
    return {
        "granted": True,
        "scope": req.scope,
        "kind": req.kind,
        "env_written": written,
        "grants": list(req.grants),
        "respawn_required": True,
        "note": f"granted for new processes. {_RESPAWN_NOTE}",
    }


@router.post("/consent/revoke")
async def revoke_consent(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Take one grant back."""
    if who.get("via") == "loopback" and not access.peer_is_loopback(request):
        raise HTTPException(401, "sign in required")
    req = _request_from(await request.json())
    patch = consent.revoke_patch(req, os.environ)
    if not patch:
        return {
            "revoked": False,
            "scope": req.scope,
            "note": "nothing to revoke - this machine does not grant that (an org-wide entry may still cover it).",
        }
    for key, value in patch.items():
        problem = config_api.env_entry_error(key, value, allowed_keys=consent.GRANT_ENV_KEYS)
        if problem:
            raise HTTPException(422, problem)
    # The live process FIRST: a revocation must take effect here even if the file write
    # fails afterwards, so a standing hardware-motion grant can never outlive the click.
    for key, value in patch.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    try:
        written = await asyncio.to_thread(_write_grant_patch, patch)
    except (OSError, ValueError) as exc:
        _audit(request, req.scope, f"revoked in this process; .env not updated: {exc}")
        raise HTTPException(
            500,
            {
                "error": f"revoked for this process, but the .env file could not be updated: {exc}",
                "revoked": True,
                "env_written": [],
                "scope": req.scope,
            },
        ) from exc
    _audit(request, req.scope, "revoked")
    return {
        "revoked": True,
        "scope": req.scope,
        "env_written": written,
        "respawn_required": True,
        "note": "revoked for new processes. A robot already running kept the permission it started with.",
    }
