"""Signed session tokens: ``<payload>.<HMAC-SHA256>``, both base64url, keyed by the store's secret.

The payload is ``{"sub": <passkey id or "token:<name>">, "exp": <unix seconds>}``. A browser carries the token in an
HttpOnly, SameSite=Strict cookie (a page script cannot read it, another site's request does not send it); a script
sends ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Iterable, MutableMapping
from typing import Any

COOKIE = "strands_robots_session"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def mint(secret: bytes, sub: str, ttl: float, *, now: float | None = None) -> str:
    payload = _b64(json.dumps({"sub": sub, "exp": int((now or time.time()) + ttl)}, separators=(",", ":")).encode())
    return f"{payload}.{_b64(hmac.new(secret, payload.encode(), hashlib.sha256).digest())}"


def verify(secret: bytes, token: str, *, now: float | None = None) -> dict[str, Any] | None:
    """The claims of a token this key signed and that has not expired, else None."""
    payload, _, sig = token.partition(".")
    want = _b64(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    if not sig or not hmac.compare_digest(sig, want):
        return None
    try:
        claims = json.loads(_unb64(payload))
    except ValueError:
        return None
    if not isinstance(claims, dict) or not isinstance(claims.get("sub"), str):
        return None
    if not isinstance(claims.get("exp"), int) or claims["exp"] <= (now or time.time()):
        return None
    return claims


def from_headers(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    """The bearer token, else the session cookie, from raw ASGI headers."""
    cookie = None
    for k, v in headers:
        name, value = k.decode("latin-1").lower(), v.decode("latin-1")
        if name == "authorization" and value[:7].lower() == "bearer ":
            return value[7:].strip() or None
        if name == "cookie":
            for part in value.split(";"):
                key, _, val = part.strip().partition("=")
                if key == COOKIE and val:
                    cookie = val
    return cookie


def from_scope(scope: MutableMapping[str, Any]) -> str | None:
    return from_headers(scope.get("headers", []))


def set_cookie(token: str, ttl: float) -> str:
    return f"{COOKIE}={token}; Path=/; Max-Age={int(ttl)}; HttpOnly; SameSite=Strict"


def clear_cookie() -> str:
    return f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
