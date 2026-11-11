"""Signed session tokens: ``<payload>.<HMAC-SHA256>``, both base64url, keyed by the store's secret.

The payload is ``{"sub": <passkey id or "token:<name>">, "exp": <unix seconds>}``; a browser session also carries
``"aud": <the Host it was issued on>`` and is refused on any other.

A script sends ``Authorization: Bearer <token>``: nothing sends that header on its own. A browser session has two
halves, neither enough alone. The token rides in an HttpOnly, SameSite=Strict cookie named after this dashboard's
port, which a page script cannot read; but a browser sends a localhost cookie to EVERY localhost port, so the token
alone is refused. Its other half is the :func:`proof` the sign-in answer hands the page, which the page keeps in its
own origin's storage (scoped by port, sent nowhere by the browser) and adds to each request as :data:`PROOF_HEADER`,
and to a socket as the subprotocol ``strands-session.<proof>``, since a browser cannot set a socket's headers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Iterable, MutableMapping
from typing import Any, NamedTuple

COOKIE = "strands_robots_session"
PROOF_HEADER = "x-strands-session"
PROTOCOL = "strands-session."


class Carried(NamedTuple):
    """What a request carries: a token, whether it came in the cookie, the page's proof (or None), its Host."""

    token: str | None
    cookie: bool
    proof: str | None
    host: str = ""


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(secret: bytes, text: str) -> str:
    return _b64(hmac.new(secret, text.encode(), hashlib.sha256).digest())


def mint(secret: bytes, sub: str, ttl: float, *, aud: str | None = None, now: float | None = None) -> str:
    claims: dict[str, Any] = {"sub": sub, "exp": int((now or time.time()) + ttl)}
    if aud is not None:
        claims["aud"] = aud
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode())
    return f"{payload}.{_sign(secret, payload)}"


def verify(secret: bytes, token: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The claims of a token this key signed and that has not expired, else None."""
    payload, _, sig = token.partition(".")
    if not sig or not hmac.compare_digest(sig, _sign(secret, payload)):
        return None
    try:
        claims = json.loads(_unb64(payload))
    except ValueError:
        return None
    if not isinstance(claims, dict) or not isinstance(claims.get("sub"), str):
        return None
    if not isinstance(claims.get("exp"), int) or claims["exp"] <= (now or time.time()):
        return None
    if "aud" in claims and not isinstance(claims["aud"], str):
        return None
    return claims


def proof(secret: bytes, token: str) -> str:
    """The page-held half of the browser session ``token``."""
    return _sign(secret, f"proof.{token}")


def proves(secret: bytes, token: str, given: str | None) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), proof(secret, token).encode())


def admits(secret: bytes, got: Carried, claims: dict[str, Any]) -> bool:
    """A script token only as a bearer; a browser session only on its own Host and with the page's proof beside it."""
    if got.token is None:
        return False
    if "aud" not in claims:  # only a script token is aud-less: a passkey session from before aud is refused
        return not got.cookie and claims["sub"].startswith("token:")
    return claims["aud"] == got.host and proves(secret, got.token, got.proof)


def cookie_name(host: str) -> str:
    """``strands_robots_session_<port>``: two dashboards on one machine never overwrite each other's session."""
    port = host.rsplit(":", 1)[-1] if ":" in host.rsplit("]", 1)[-1] else ""
    return f"{COOKIE}_{port}" if port.isdigit() else COOKIE


def carried(headers: Iterable[tuple[bytes, bytes]]) -> Carried:
    """The bearer token, else this port's session cookie, and the page's proof, from raw ASGI headers."""
    h: dict[str, list[str]] = {}
    for k, v in headers:
        h.setdefault(k.decode("latin-1").lower(), []).append(v.decode("latin-1"))
    given: str | None = h[PROOF_HEADER][0] if PROOF_HEADER in h else None
    for offered in h.get("sec-websocket-protocol", []):
        for p in (p.strip() for p in offered.split(",")):
            if p.startswith(PROTOCOL):
                given = p[len(PROTOCOL) :]
    host = (h.get("host") or [""])[0]
    for value in h.get("authorization", []):
        if value[:7].lower() == "bearer ":
            return Carried(value[7:].strip() or None, False, given, host)
    for value in h.get("cookie", []):
        for part in value.split(";"):
            key, _, val = part.strip().partition("=")
            if key == cookie_name(host) and val:
                return Carried(val, True, given, host)
    return Carried(None, False, given, host)


def from_scope(scope: MutableMapping[str, Any]) -> Carried:
    return carried(scope.get("headers", []))


def set_cookie(token: str, ttl: float, host: str) -> str:
    return f"{cookie_name(host)}={token}; Path=/; Max-Age={int(ttl)}; HttpOnly; SameSite=Strict"


def clear_cookie(host: str) -> str:
    return f"{cookie_name(host)}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
