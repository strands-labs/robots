"""The passkey routes, on top of :mod:`strands_robots.dashboard.auth`.

Nothing here decides who may enrol or sign in: every verdict is the auth
module's, this file only shapes HTTP around it. Two decisions ARE routing
decisions and live here because they concern which route is open:

* ``/api/auth/status`` is public. The login screen reads it before anyone is
  signed in; it says whether setup is required and which proof it needs, never
  the proof itself and never the enrolled passkeys - a route that answers
  whoever the socket lets through publishes the named fields below, not
  whatever the auth module happens to return.
* A second enrolment needs a session; the first needs the bootstrap proof the
  auth module checks (``STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN`` or the ``0600``
  file it minted beside the store).

A finished ceremony sets the ``strands_dash`` cookie so the browser carries the
session on every request, ``HttpOnly`` so page scripts cannot read it and
``SameSite=Strict`` so no other origin can ride it. ``Secure`` follows the
connection: on ``http://localhost`` the cookie must still be settable.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from strands_robots.dashboard import access, auth

router = APIRouter(prefix="/api/auth", tags=["auth"])

#: What a caller with no session may learn from ``/api/auth/status``: whether
#: setup is required, which proof the first enrolment needs, and the login
#: screen's relying-party hints. Named rather than subtracted, so a field added
#: to :func:`auth.status` is private until someone puts it here on purpose.
PUBLIC_STATUS_FIELDS = frozenset(
    {
        "enabled",
        "setup_required",
        "bootstrap_required",
        "bootstrap_source",
        "rp_id",
        "secure_context",
        "rpid_usable",
        "warning",
    }
)


async def _json_body(request: Request) -> dict[str, Any]:
    """The request body as a JSON object, or the 4xx that says why not.

    A body has to declare itself ``application/json``. That is not pedantry:
    a cross-site ``fetch`` may send ``text/plain`` without a preflight, and a
    parser that accepts it turns any page on the web into a caller of every
    write route. Declaring JSON makes the request non-simple, so the browser
    asks first - and this app never says yes.
    """
    raw = await request.body()
    if not raw:
        return {}
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise HTTPException(415, "body must be application/json")
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "body is not JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    return body


def _set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        access.COOKIE,
        token,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    """What the login screen may know before sign-in, plus whether THIS caller is in.

    The enrolled passkeys are not part of it. ``/api/auth/credentials`` serves
    those to a session; this route answers whoever the socket lets through, and
    on a sealed dashboard that includes a page the operator's own browser was
    told to load. ``setup_required`` is that same list reduced to the one bit the
    login screen reads, so the screen loses nothing by not seeing the list.
    """
    out = {k: v for k, v in auth.status(request).items() if k in PUBLIC_STATUS_FIELDS}
    out["authenticated"] = access.session_claims(request) is not None
    out["open_posture"] = access.open_posture(request)
    return out


@router.post("/register/begin")
async def register_begin(request: Request) -> dict[str, Any]:
    """Start an enrolment: the first needs the bootstrap proof, later ones a session."""
    body = await _json_body(request)
    if auth.has_credentials() and access.session_claims(request) is None:
        raise HTTPException(401, "sign in to add another passkey")
    label = str(body.get("label") or "passkey")[:64]
    bootstrap = str(body.get("bootstrap") or "")
    return auth.begin_registration(request, label=label, bootstrap=bootstrap)


@router.post("/register/finish")
async def register_finish(request: Request, response: Response) -> dict[str, Any]:
    """Verify the registration ceremony, enrol the key, set the session cookie."""
    body = await _json_body(request)
    challenge_id = str(body.get("challenge_id") or "")
    credential = body.get("credential")
    if not challenge_id or not isinstance(credential, dict):
        raise HTTPException(400, "challenge_id and credential are required")
    out = auth.finish_registration(request, challenge_id, credential)
    _set_session_cookie(response, request, out["token"])
    return out


@router.post("/login/begin")
async def login_begin(request: Request) -> dict[str, Any]:
    """Start a sign-in ceremony for the enrolled passkeys."""
    return auth.begin_authentication(request)


@router.post("/login/finish")
async def login_finish(request: Request, response: Response) -> dict[str, Any]:
    """Verify the sign-in ceremony and set the session cookie."""
    body = await _json_body(request)
    challenge_id = str(body.get("challenge_id") or "")
    credential = body.get("credential")
    if not challenge_id or not isinstance(credential, dict):
        raise HTTPException(400, "challenge_id and credential are required")
    out = auth.finish_authentication(request, challenge_id, credential)
    _set_session_cookie(response, request, out["token"])
    return out


@router.post("/logout")
async def logout(response: Response) -> dict[str, Any]:
    """Drop the session cookie. The token itself simply expires."""
    response.delete_cookie(access.COOKIE, path="/")
    return {"ok": True}


@router.get("/credentials")
async def credentials(_: dict = Depends(access.require_session)) -> dict[str, Any]:
    """The enrolled passkeys (ids and labels, never key material)."""
    return {"credentials": auth.list_credentials()}


@router.delete("/credentials/{cred_id}")
async def delete_credential(cred_id: str, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Remove a passkey. Only a passkey session may do this, never the static token."""
    if who.get("via") != "passkey":
        raise HTTPException(403, "only a passkey session may remove a passkey")
    return auth.delete_credential(cred_id)


@router.post("/handoff")
async def handoff(request: Request, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """A short-lived token to open the dashboard on another device of the same owner."""
    if who.get("via") != "passkey":
        raise HTTPException(403, "only a passkey session may mint a handoff")
    return auth.issue_handoff(who)


@router.post("/renew")
async def renew(request: Request, response: Response, who: dict = Depends(access.require_session)) -> dict[str, Any]:
    """Renew the presented session if it is due; the absolute cap is the auth module's."""
    if who.get("via") != "passkey":
        return {"renewed": False}
    fresh = auth.renew_if_due(access.presented_token(request))
    if fresh:
        _set_session_cookie(response, request, fresh)
    return {"renewed": bool(fresh)}
