"""``Auth``: who may use this dashboard. Passkeys sign a person in; a minted token signs a script in.

First run: no passkey exists, so the server holds a one-time setup code (printed when it starts) and enrolment needs
it; any other local process could otherwise enrol first. Every further passkey needs a signed-in session, and the
last one cannot be removed.
"""

from __future__ import annotations

import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from . import ceremony, session
from .store import Store

log = logging.getLogger(__name__)

DEFAULT_TTL = 12 * 3600.0
_CHALLENGE_TTL = 300.0


class _HasUrl(Protocol):
    @property
    def url(self) -> str: ...


_Served = TypeVar("_Served", bound=_HasUrl)


class Auth:
    """Sign-in for one dashboard. ``ttl`` is the session length in seconds (``$STRANDS_ROBOTS_DASHBOARD_SESSION_TTL``,
    default 12 h); ``rp_id`` / ``origin`` override ``localhost`` / ``http://localhost:<port>`` behind a hostname."""

    def __init__(
        self,
        *,
        store: str | Path | Store | None = None,
        ttl: float | None = None,
        rp_id: str = "localhost",
        origin: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.store = store if isinstance(store, Store) else Store(store)
        self.ttl = float(
            ttl if ttl is not None else os.environ.get("STRANDS_ROBOTS_DASHBOARD_SESSION_TTL", DEFAULT_TTL)
        )
        if self.ttl <= 0:
            raise ValueError(f"ttl must be > 0 seconds, got {self.ttl}")
        self.rp_id, self.origin = rp_id, origin
        self._setup: str | None = None
        self._challenges: dict[str, tuple[str, bytes, float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def token(self, name: str = "local") -> str:
        """A session token for a script, notebook or agent (``Authorization: Bearer <token>``), valid ``ttl``."""
        return session.mint(self.store.secret, f"token:{name}", self.ttl)

    def claims(self, token: str | None) -> dict[str, Any] | None:
        """A valid token's claims; a passkey's session ends when that passkey is removed."""
        if not token:
            return None
        claims = session.verify(self.store.secret, token)
        if claims is None or claims["sub"].startswith("token:"):
            return claims
        return claims if any(c["id"] == claims["sub"] for c in self.store.credentials()) else None

    def signed_in(self, scope: Any) -> dict[str, Any] | None:
        """The session of this request. A browser session (it names the Host it was issued on) counts only on that
        Host and only with the page's proof beside it; the cookie carries nothing else."""
        if not self.enabled:
            return {"sub": "sign-in off"}
        got = session.from_scope(scope)
        claims = self.claims(got.token)
        return claims if claims is not None and session.admits(self.store.secret, got, claims) else None

    def proof(self, token: str) -> str:
        """The half of a browser session the page keeps (``X-Strands-Session``); the cookie holds the other."""
        return session.proof(self.store.secret, token)

    @property
    def setup_required(self) -> bool:
        return self.enabled and not self.store.credentials()

    @property
    def setup_code(self) -> str | None:
        """The one-time code that allows the first enrolment (None once a passkey exists)."""
        if not self.setup_required:
            self._setup = None
            return None
        if self._setup is None:
            self._setup = secrets.token_urlsafe(12)
        return self._setup

    def announce(self, server: _Served) -> _Served:
        """Print the first-time setup link for ``server`` (to stderr) when no passkey exists; return ``server``."""
        url = self.setup_url(server.url) if self.enabled else None
        if url is not None:
            print(f"dashboard: no passkey yet; open {url} to create one (one-time link)", file=sys.stderr)
        return server

    def setup_url(self, base: str) -> str | None:
        code = self.setup_code
        return None if code is None else f"{base.rstrip('/')}/?setup={code}"

    def _rp(self, host: str) -> tuple[str, str]:
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if self.origin is None and name != self.rp_id:
            port = host.rsplit(":", 1)[-1]
            raise ValueError(f"passkeys work on http://{self.rp_id}:{port}/, not on {host}: open that address")
        return self.rp_id, self.origin or f"http://{host}"

    def _stash(self, kind: str, challenge: bytes, extra: dict[str, Any]) -> str:
        cid = secrets.token_urlsafe(16)
        now = time.monotonic()
        with self._lock:
            for k in [k for k, v in self._challenges.items() if now - v[2] > _CHALLENGE_TTL]:
                del self._challenges[k]
            while len(self._challenges) >= 64:
                del self._challenges[next(iter(self._challenges))]
            self._challenges[cid] = (kind, challenge, now, extra)
        return cid

    def _take(self, cid: Any, kind: str) -> tuple[bytes, dict[str, Any]]:
        with self._lock:
            rec = self._challenges.pop(cid, None) if isinstance(cid, str) else None
        if rec is None or rec[0] != kind or time.monotonic() - rec[2] > _CHALLENGE_TTL:
            raise PermissionError("unknown or expired challenge: start again")
        return rec[1], rec[3]

    def begin_enrol(self, host: str, *, setup: Any = None, scope: Any = None, label: Any = None) -> dict[str, Any]:
        """Options for ``navigator.credentials.create``: the first passkey needs the setup code, more need a session."""
        if self.setup_required:
            code = self.setup_code or ""
            if not isinstance(setup, str) or not secrets.compare_digest(setup.encode(), code.encode()):
                raise PermissionError("the first passkey needs the setup code the dashboard printed when it started")
        elif self.signed_in(scope) is None:
            raise PermissionError("sign in first: adding a passkey needs a signed-in session")
        rp_id, origin = self._rp(host)
        doc = self.store.read()
        challenge, options = ceremony.enrol_options(
            rp_id=rp_id, user_id=doc["user_id"], exclude=[c["id"] for c in doc["credentials"]]
        )
        name = label.strip()[:64] if isinstance(label, str) and label.strip() else "passkey"
        cid = self._stash("enrol", challenge, {"origin": origin, "label": name, "first": self.setup_required})
        return {"challenge_id": cid, "options": options}

    def finish_enrol(self, host: str, challenge_id: Any, credential: Any) -> str:
        challenge, extra = self._take(challenge_id, "enrol")
        rp_id, origin = self._rp(host)
        if origin != extra["origin"]:
            raise PermissionError("the ceremony began on another origin")
        cid, key, count = ceremony.verify_enrol(credential, challenge=challenge, rp_id=rp_id, origin=origin)

        def add(doc: dict[str, Any]) -> None:
            if extra["first"] and doc["credentials"]:
                raise PermissionError("a passkey was enrolled meanwhile: sign in with it")
            if any(c["id"] == cid for c in doc["credentials"]):
                raise RuntimeError("this passkey is already enrolled")
            entry = {"id": cid, "public_key": key, "sign_count": count, "name": extra["label"], "created": time.time()}
            doc["credentials"].append(entry)

        self.store.update(add)
        self._setup = None
        log.warning("dashboard: passkey %r enrolled (%s)", extra["label"], self.store.path)
        return session.mint(self.store.secret, cid, self.ttl, aud=host)

    def begin_login(self, host: str) -> dict[str, Any]:
        rp_id, origin = self._rp(host)
        if self.setup_required:
            raise RuntimeError("no passkey yet: enrol one with the setup code first")
        challenge, options = ceremony.login_options(rp_id=rp_id, allow=[c["id"] for c in self.store.credentials()])
        return {"challenge_id": self._stash("login", challenge, {"origin": origin}), "options": options}

    def finish_login(self, host: str, challenge_id: Any, credential: Any) -> str:
        challenge, extra = self._take(challenge_id, "login")
        rp_id, origin = self._rp(host)
        cid = credential.get("id") if isinstance(credential, dict) else None
        if origin != extra["origin"] or not isinstance(cid, str):
            raise PermissionError("sign-in refused: wrong origin or no credential id")

        def check(doc: dict[str, Any]) -> None:
            entry = next((c for c in doc["credentials"] if c["id"] == cid), None)
            if entry is None:
                raise PermissionError("this passkey is not enrolled on this dashboard")
            entry["sign_count"] = ceremony.verify_login(
                credential, challenge=challenge, rp_id=rp_id, origin=origin, **_key(entry)
            )

        self.store.update(check)
        return session.mint(self.store.secret, cid, self.ttl, aud=host)

    def passkeys(self) -> list[dict[str, Any]]:
        return [{"id": c["id"], "name": c["name"], "created": c.get("created")} for c in self.store.credentials()]

    def rename(self, cid: str, name: Any) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('body must be {"name": "<new name>"}')

        def edit(doc: dict[str, Any]) -> dict[str, Any]:
            entry = _find(doc, cid)
            entry["name"] = name.strip()[:64]
            return {"id": cid, "name": entry["name"]}

        return self.store.update(edit)  # type: ignore[no-any-return]

    def remove(self, cid: str) -> dict[str, Any]:
        def drop(doc: dict[str, Any]) -> dict[str, Any]:
            _find(doc, cid)
            if len(doc["credentials"]) == 1:
                raise RuntimeError("this is the last passkey: add another before removing it")
            doc["credentials"] = [c for c in doc["credentials"] if c["id"] != cid]
            return {"removed": cid, "remaining": len(doc["credentials"])}

        return self.store.update(drop)  # type: ignore[no-any-return]


def _find(doc: dict[str, Any], cid: str) -> dict[str, Any]:
    entry = next((c for c in doc["credentials"] if c["id"] == cid), None)
    if entry is None:
        raise KeyError(f"no passkey {cid!r}")
    return entry  # type: ignore[no-any-return]


def _key(entry: dict[str, Any]) -> dict[str, Any]:
    return {"public_key": entry["public_key"], "sign_count": int(entry.get("sign_count", 0))}


def configure(auth: Any, *, hardware: bool, insist: bool = False) -> Auth:
    """``True`` (the default) is passkey sign-in; an ``Auth`` is used as given; ``False`` turns sign-in off, which is
    refused while a real robot is attached unless ``insist`` (``Dashboard(unauthenticated_hardware=True)``)."""
    if isinstance(auth, Auth):
        return auth
    if auth is True:
        return Auth()
    if auth is not False:
        raise TypeError(f"auth must be True, False or an Auth, got {auth!r}")
    if hardware and not insist:
        raise ValueError(
            "auth=False with a real robot attached: any process that reaches this port could move it. "
            "Keep sign-in on (use dashboard.auth.token() for scripts), or pass unauthenticated_hardware=True"
        )
    log.warning("dashboard sign-in is OFF: any local process that reaches this port can use every route")
    return Auth(enabled=False)
