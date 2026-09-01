"""WebAuthn (passkey) authentication for the dashboard. Why this exists: the dashboard commands
real hardware (SO-101 arms).

A ceremony is verified against two expectations, and neither may come from the
caller: ``expected_rp_id`` from the challenge record this process stashed, and
``expected_origin`` from the connection the request actually arrived on (see
:func:`origin_verdict`). What the request ASSERTS about itself -- its ``Origin``
header, ``x-forwarded-proto``, ``x-forwarded-for`` -- is a claim, and a claim is
only ever compared against an expectation, never promoted into one.

Configuration:
    ``STRANDS_DASH_AUTH_ORIGIN``: the origin the dashboard is served at, e.g.
        ``https://robots.example``. Unset by default, in which case it is read off
        the connection: the transport's own scheme plus the ``Host`` header. Set
        it when a proxy rewrites ``Host`` or ``Origin``, or when TLS is terminated
        upstream and the server is not configured to forward that fact (uvicorn's
        ``--proxy-headers`` with ``--forwarded-allow-ips``, which is where the
        trusted-proxy decision belongs). WebAuthn compares origins byte-for-byte,
        so it must carry the scheme and any non-default port.
    ``STRANDS_DASH_AUTH_RP_ID``: pins the relying-party id when the hostname
        legitimately changed. See :func:`rp_id_verdict`.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jwt  # PyJWT
from fastapi import HTTPException
from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

_ENV = "STRANDS_DASH_AUTH_"

# Both directions are spelled out. A value outside either vocabulary must not
# resolve to auth-OFF: this gate fronts routes that command real hardware, so a
# typo silently dropping it is the one misparse direction that cannot be
# tolerated. An unrecognized value is reported and the store decides instead.
_ENABLED_TRUE = ("1", "true", "yes", "on")
_ENABLED_FALSE = ("0", "false", "no", "off")


def auth_enabled() -> bool:
    """Whether passkey auth guards the API. The STORE is the source of truth: the moment a passkey is
    enrolled, auth is ON.

    ``STRANDS_DASH_AUTH_ENABLED`` overrides the store only when it is spelled as
    a recognized boolean. Anything else is logged and ignored, so an enrolled
    passkey still guards the API rather than being dropped by a misspelling.
    """
    raw = os.getenv(_ENV + "ENABLED", "").strip().lower()
    if raw in _ENABLED_TRUE:
        return True
    if raw in _ENABLED_FALSE:
        return False
    if raw:
        logging.getLogger(__name__).warning(
            "%sENABLED=%r is not a recognized boolean (true: %s; false: %s); ignoring the override "
            "and reading the credential store instead, so an enrolled passkey still guards the API.",
            _ENV,
            raw,
            ", ".join(_ENABLED_TRUE),
            ", ".join(_ENABLED_FALSE),
        )
    return has_credentials()


def _store_path() -> Path:
    default = Path.home() / ".strands_dashboard" / "auth.json"
    return Path(os.getenv(_ENV + "STORE", str(default))).expanduser().resolve()


def _rp_name() -> str:
    return os.getenv(_ENV + "RP_NAME", "strands robots dashboard")


def _token_ttl() -> int:
    try:
        return int(os.getenv(_ENV + "TOKEN_TTL", "86400"))
    except ValueError:
        return 86400


def _bootstrap_token() -> str:
    return os.getenv(_ENV + "BOOTSTRAP_TOKEN", "").strip()


def _forced_rp_id() -> str:
    return os.getenv(_ENV + "RP_ID", "").strip()


def _forced_origin() -> str:
    return os.getenv(_ENV + "ORIGIN", "").strip()


# --- store: one JSON file, thread-safe, hot-reloaded on mtime change -------

_lock = threading.Lock()

# The store, cached under the identity of the file it was read from. A value
# global beside a parallel key global is an invariant maintained by hand at every
# write - and the two can disagree, at which point a stale hit is
# indistinguishable from a fresh one. Keyed this way they cannot: the key is the
# dict's key, so a value is only reachable through the identity it was read
# under. ``mesh/_acl_config.py`` keys its ACL cache on a file identity tuple for
# the same reason. Holds at most one entry - there is one store path per process
# - so an operator (or an attacker) rewriting the store cannot grow it.
_cache: dict[tuple, dict[str, Any]] = {}


def _default_store() -> dict[str, Any]:
    return {
        "jwt_secret": secrets.token_urlsafe(48),
        "credentials": [],  # {id, public_key, sign_count, name, created}
        "created": time.time(),
    }


# Set when a store on disk could not be parsed: the backup path plus why.
_corrupt: dict[str, str] | None = None


def store_corruption() -> dict[str, str] | None:
    """The unreadable store this process rescued, if any: {'backup': path, 'reason': str}."""
    return dict(_corrupt) if _corrupt else None


def _preserve_corrupt(path: Path, exc: Exception) -> None:
    """Move an unparseable store aside instead of clobbering it, and remember that we did."""
    global _corrupt
    backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    try:
        os.replace(path, backup)
        where = str(backup)
    except OSError:
        where = ""
    _corrupt = {"backup": where, "reason": f"{type(exc).__name__}: {exc}"}
    logging.getLogger(__name__).warning(
        "dashboard auth store at %s is unreadable (%s); kept as %s. Enrollment is limited to "
        "this machine until a passkey exists again.",
        path,
        _corrupt["reason"],
        where or "<could not move it>",
    )


def _store_identity(path: Path) -> tuple | None:
    """Return ``(path, mtime_ns, size)`` for the store, or None if it cannot be stat-ed.

    None is deliberately not a cache key. It means "re-read", which is the safe
    direction to fail in for the file that decides whether this dashboard is
    sealed: serving memory under a key that describes nothing is how a store
    replaced underneath the process goes unnoticed.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _remember_locked(identity: tuple | None, store: dict[str, Any]) -> None:
    """Make ``store`` the single cached entry, reachable only under ``identity``.

    Caller must hold :data:`_lock`. A None identity caches nothing, so the next
    :func:`_load` reads the file again instead of trusting memory.
    """
    _cache.clear()
    if identity is not None:
        _cache[identity] = store


def _load() -> dict[str, Any]:
    """Read the store, re-reading the file whenever it changes on disk."""
    path = _store_path()
    with _lock:
        identity = _store_identity(path)
        if identity is not None:
            cached = _cache.get(identity)
            if cached is not None:
                return cached
            try:
                store: dict[str, Any] = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                _preserve_corrupt(path, exc)
            else:
                _remember_locked(identity, store)
                return store
        store = _default_store()
        _save_locked(store)
        return store


def _save_locked(store: dict[str, Any]) -> None:
    """Replace the store atomically, so no interrupted write can truncate it.

    This is the deployment's only credential record, and this is its
    highest-frequency writer: every successful authentication persists
    ``sign_count`` through here, as does every enrollment and the corruption
    re-seed. Writing the path in place therefore made this function the most
    likely *producer* of the unparseable store :func:`_preserve_corrupt`
    exists to rescue - a kill or power loss inside the write window leaves
    exactly the truncated JSON that path handles. That rescue bounds the
    security damage, not the loss: the passkey records and the ``jwt_secret``
    are gone for good, every session dies with them, and the operator is put
    through the machine-local re-seal.

    So the payload lands in a sibling temp file and is moved into position
    with ``os.replace`` - the same primitive :func:`_preserve_corrupt` uses a
    few lines up, and atomic within a directory, so a concurrent reader sees
    either the whole previous store or the whole new one and never a prefix
    of either. Creating it through ``mkstemp`` closes a second, smaller gap
    as a side effect: ``mkstemp`` opens at ``0o600``, whereas writing the
    path directly created a new store at the umask default and only then
    chmod-ed it, which left a fresh ``jwt_secret`` briefly world-readable.
    The replace carries those bits onto the store, so a store left at ``0o644``
    by an older build is tightened the next time it is written.

    :func:`strands_robots.simulation.safe_output.atomic_write_bytes`
    implements this same sequence and is deliberately *not* imported: it
    would make the ``[dashboard]`` extra pull in the simulation package to
    save a passkey. Lift that helper into ``strands_robots.utils`` if a third
    caller ever wants it.
    """
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(store, indent=2))
        os.replace(tmp, path)
    except BaseException:
        # Leave no debris behind a failed save: the store on disk is still
        # the previous good one, and a stray .tmp beside it would be read by
        # nothing but would outlive the process that abandoned it.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    # Re-key on the file just written. A store that vanished between the replace
    # and this stat has no identity, so nothing is cached and the next _load
    # re-reads - see _store_identity.
    _remember_locked(_store_identity(path), store)


def _save(store: dict[str, Any]) -> None:
    with _lock:
        _save_locked(store)


def _jwt_secret() -> str:
    return cast(str, _load()["jwt_secret"])


def has_credentials() -> bool:
    return len(_load().get("credentials", [])) > 0


def list_credentials() -> list[dict[str, Any]]:
    return [
        {"id": c["id"], "name": c.get("name", "passkey"), "created": c.get("created")}
        for c in _load().get("credentials", [])
    ]


def delete_credential(cred_id: str) -> dict[str, Any]:
    """Revoke a passkey. Refuses to remove the LAST one (would re-open the
    dashboard to anyone via the setup flow)."""
    store = _load()
    creds = store.get("credentials", [])
    if not any(c["id"] == cred_id for c in creds):
        raise HTTPException(404, "credential not found")
    if len(creds) <= 1:
        raise HTTPException(409, "cannot remove the last passkey - enroll another first")
    store["credentials"] = [c for c in creds if c["id"] != cred_id]
    _save(store)
    return {"ok": True, "removed": cred_id, "remaining": len(store["credentials"])}


# --- relying-party id / origin derivation -----------------------------------


def _host_only(host: str) -> str:
    return host.split(":")[0]


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def rpid_is_usable(host_only: str) -> bool:
    """WebAuthn rpId must be a registrable domain or 'localhost'; a raw IP is
    rejected by browsers before the ceremony starts."""
    if host_only == "localhost":
        return True
    if not host_only or _is_ip(host_only):
        return False
    return True


def _headers(request_or_ws: Any) -> Any:
    return request_or_ws.headers


#: Hostnames that are always acceptable as a relying-party id: a browser on this
#: machine is the operator, and local dev must never depend on remote config.
_LOOPBACK_RP_IDS = frozenset({"localhost", "127.0.0.1", "::1"})


def known_rp_ids(store: dict | None = None) -> set:
    """Every rp_id this deployment has PROVEN it uses."""
    s = store if store is not None else _load()
    return {c["rp_id"] for c in s.get("credentials", []) if c.get("rp_id")}


def rp_id_verdict(host_rp_id: str, forced: str = "", known: set | None = None) -> tuple:
    """Decide the rp_id for a ceremony: ``(rp_id, reason)``, or ``(None, reason)``."""
    # Loopback outranks even the pin, and that ordering is deliberate: a browser at
    # http://localhost:8090 CANNOT use 'robots.cagatay.my' as an rp_id -- the spec requires the
    # rp_id to be a registrable suffix of the page's origin, so honouring the pin here would make
    # the browser refuse the ceremony before it starts, and a passkey bound to the domain could
    # not be used from localhost anyway.
    if host_rp_id in _LOOPBACK_RP_IDS:
        return (host_rp_id, "loopback")
    if forced:
        return (forced, "forced by STRANDS_DASH_AUTH_RP_ID")
    known = known_rp_ids() if known is None else known
    if host_rp_id in known:
        return (host_rp_id, "matches an enrolled credential")
    if not known:
        return (host_rp_id, "legacy: no rp_id recorded yet, binding on first use")
    return (None, f"host {host_rp_id!r} is not one of the enrolled {sorted(known)}")


def _derive_rp_id(request_or_ws: Any) -> str:
    host = _host_only(_headers(request_or_ws).get("host", "localhost"))
    rp_id, reason = rp_id_verdict(host, _forced_rp_id())
    if rp_id is None:
        logger.warning("refused WebAuthn ceremony: %s", reason)
        raise HTTPException(
            400,
            {
                "error": "this host cannot be used for a passkey ceremony",
                "detail": reason,
                "hint": "reach the dashboard on its enrolled hostname, or set "
                "STRANDS_DASH_AUTH_RP_ID if it legitimately changed",
            },
        )
    return cast(str, rp_id)


#: How a connection scheme is spelled by the page origin it belongs to. An operator
#: page served over https opens its sockets as ``wss``, and the browser still spells
#: that page's ``Origin`` ``https://`` -- so the socket spelling is never the answer.
_PAGE_SCHEME = {"http": "http", "https": "https", "ws": "http", "wss": "https"}


def origin_verdict(offered: str, expected: str) -> tuple:
    """Decide the origin a ceremony is verified against: ``(origin, reason)``, or ``(None, reason)``.

    ``expected`` is where this deployment is reachable, which is a fact about the
    connection (or a value the operator configured). ``offered`` is the request's
    own ``Origin`` header, which is the caller's claim about itself and therefore
    never the answer: handing it to the WebAuthn library as ``expected_origin``
    tells the library to expect whatever the caller said, and a comparison
    against the caller's own claim cannot fail. What that comparison exists to
    refuse -- an ``http://`` downgrade, a sibling subdomain -- is precisely what
    adopting the header lets through, because ``rp_id`` binds the registrable
    domain but nothing below it.

    Args:
        offered: The request's ``Origin`` header, or ``""`` when it sent none.
        expected: The origin this deployment is actually reachable at.

    Returns:
        ``(origin, reason)`` with the origin to verify against, or
        ``(None, reason)`` when the header contradicts the connection.
    """
    if not offered:
        # A browser sends Origin on the ceremony POSTs, so its absence is odd --
        # but absence is not a reason to widen the expectation. The connection's
        # own origin stands, and the authenticator's signed clientData still has
        # to match it, which is the check that was being bypassed.
        return (expected, "no Origin header; the connection's own origin stands")
    if offered.rstrip("/") == expected:
        return (expected, "Origin matches the connection")
    return (None, f"Origin {offered!r} is not this deployment's origin {expected!r}")


def _connection_scheme(request_or_ws: Any) -> str:
    """The scheme this deployment was actually reached over, spelled as a page origin.

    Read off the ASGI connection rather than from ``x-forwarded-proto``, for the
    same reason :func:`_socket_peer` ignores ``x-forwarded-for``: that header is
    set by whoever is calling, so a stranger can spell it ``https`` and choose
    the scheme half of an expectation that grants a session. A TLS-terminating
    proxy is still honoured -- by the SERVER, under the trusted-proxy allowlist
    the operator configured there (uvicorn's ``--proxy-headers`` with
    ``--forwarded-allow-ips``), which rewrites the scheme before the app is
    reached. A deployment that cannot do that sets ``STRANDS_DASH_AUTH_ORIGIN``.

    Raises:
        HTTPException: 400 when the transport reports no usable scheme, so an
            expectation is refused rather than guessed.
    """
    scheme = str(getattr(getattr(request_or_ws, "url", None), "scheme", "")).lower()
    page = _PAGE_SCHEME.get(scheme)
    if page is None:
        logger.warning("refused WebAuthn ceremony: transport reported scheme %r", scheme)
        raise HTTPException(
            400,
            {
                "error": "this connection cannot be used for a passkey ceremony",
                "detail": f"the transport reported no usable scheme ({scheme!r}), so the origin "
                "a ceremony must be verified against cannot be determined",
                "hint": "set STRANDS_DASH_AUTH_ORIGIN to the origin the dashboard is served at",
            },
        )
    return page


def _served_origin(request_or_ws: Any) -> str:
    """The origin this deployment is reachable at: configured, or the connection's own.

    The ``Host`` header supplies the authority half, which is the same source
    :func:`_derive_rp_id` already binds through :func:`rp_id_verdict` -- so the two
    expectations agree by construction, and a host a stranger made up is refused
    there rather than reappearing here as a different answer.
    """
    forced = _forced_origin()
    if forced:
        # Normalised because WebAuthn compares origins byte-for-byte: a trailing
        # slash in the env var would otherwise fail every ceremony.
        return forced.rstrip("/")
    return f"{_connection_scheme(request_or_ws)}://{_headers(request_or_ws).get('host', 'localhost:8090')}"


def _derive_origin(request_or_ws: Any) -> str:
    """The origin a ceremony is verified against, refusing a caller that claims another."""
    expected = _served_origin(request_or_ws)
    if _forced_origin():
        # The operator decided it, and the installs that need this pin are the ones
        # whose proxy rewrites Host/Origin -- so consulting either header here would
        # refuse exactly the deployment the pin exists for.
        return expected
    origin, reason = origin_verdict(_headers(request_or_ws).get("origin", ""), expected)
    if origin is None:
        logger.warning("refused WebAuthn ceremony: %s", reason)
        raise HTTPException(
            400,
            {
                "error": "this Origin cannot be used for a passkey ceremony",
                "detail": reason,
                "hint": "reach the dashboard on the origin it is served at, or set "
                "STRANDS_DASH_AUTH_ORIGIN if a proxy rewrites Host or Origin",
            },
        )
    return cast(str, origin)


def _rpid_error(rp_id: str) -> HTTPException:
    return HTTPException(
        400,
        f"WebAuthn cannot use '{rp_id}' as the relying-party id (needs a "
        "hostname or domain, not a raw IP). Open the dashboard via a hostname "
        "or set STRANDS_DASH_AUTH_RP_ID.",
    )


# --- challenge cache (short-lived, in-memory) --------------------------------

logger = logging.getLogger(__name__)

_challenges: dict[str, dict[str, Any]] = {}
_chal_lock = threading.Lock()
_CHAL_TTL = 300.0

# : Caps on the challenge table. Both are per-process and generous: a challenge : measures
# ~0.5KB, so 512 of them is ~256KB.
_CHAL_MAX = int(os.getenv("STRANDS_DASH_AUTH_CHAL_MAX", "512"))
# : The property that actually matters: no single client may fill the table and : push out the
# operator's pending login.
_CHAL_MAX_PER_IP = int(os.getenv("STRANDS_DASH_AUTH_CHAL_MAX_PER_IP", "16"))


def _evict_oldest(where: dict[str, dict[str, Any]], keep: int, ip: str | None = None) -> int:
    """Drop the oldest entries (optionally only one ip's) until ``keep`` remain."""
    pool = [(v["t"], k) for k, v in where.items() if ip is None or v.get("ip") == ip]
    dropped = 0
    for _t, k in sorted(pool)[: max(0, len(pool) - keep)]:
        where.pop(k, None)
        dropped += 1
    return dropped


def _stash_challenge(
    kind: str,
    challenge: bytes,
    extra: dict | None = None,
    ip: str | None = None,
) -> str:
    cid = secrets.token_urlsafe(16)
    now = time.time()
    with _chal_lock:
        for k in [k for k, v in _challenges.items() if now - v["t"] > _CHAL_TTL]:
            _challenges.pop(k, None)
        # Evict the flooder's OWN oldest entries first, so one noisy client
        # cannot cost anybody else their in-flight ceremony.
        if ip:
            evicted = _evict_oldest(_challenges, _CHAL_MAX_PER_IP - 1, ip=ip)
            if evicted:
                logger.warning("challenge cap: dropped %d stale challenge(s) from %s", evicted, ip)
        if len(_challenges) >= _CHAL_MAX:
            _evict_oldest(_challenges, _CHAL_MAX - 1)
            logger.warning("challenge table full (%d); evicted oldest", _CHAL_MAX)
        _challenges[cid] = {
            "kind": kind,
            "challenge": challenge,
            "t": now,
            "extra": extra or {},
            "ip": ip,
        }
    return cid


def _client_ip(request_or_ws: Any) -> str | None:
    """Best-effort client identity for the per-ip cap only -- NEVER for trust."""
    try:
        h = _headers(request_or_ws)
        fwd = h.get("cf-connecting-ip") or h.get("x-forwarded-for") or h.get("x-real-ip")
        if fwd:
            return fwd.split(",")[0].strip()[:64] or None
        client = getattr(request_or_ws, "client", None)
        return getattr(client, "host", None)
    except Exception:
        return None


def _socket_peer(request_or_ws: Any) -> str | None:
    """The peer address of the connection itself. This is the one safe to trust.

    Deliberately does not consult ``cf-connecting-ip`` / ``x-forwarded-for`` /
    ``x-real-ip`` the way :func:`_client_ip` does: those are set by whoever is
    calling, so a stranger can spell any of them ``127.0.0.1``. Only the socket
    peer is a fact about who actually connected, so a decision that grants
    something -- rather than merely accounting for it -- reads this.
    """
    try:
        return getattr(getattr(request_or_ws, "client", None), "host", None)
    except Exception:
        return None


def _pop_challenge(cid: str, kind: str) -> dict[str, Any]:
    with _chal_lock:
        rec = _challenges.pop(cid, None)
    if not rec or rec["kind"] != kind:
        raise HTTPException(400, "invalid or expired challenge")
    if time.time() - rec["t"] > _CHAL_TTL:
        raise HTTPException(400, "challenge expired")
    return rec


# --- JWT sessions ------------------------------------------------------------


def issue_token(
    subject: str,
    name: str = "",
    iat0: int | None = None,
    exp: int | None = None,
    via: str | None = None,
) -> str:
    """A session token. `iat0` is the ORIGINAL sign-in, carried unchanged through every
    renewal so the absolute cap in renewal_verdict() cannot be reset by re-issuing.
    `via` marks how the token was minted (e.g. "handoff") for later forensics."""
    now = int(time.time())
    payload = {
        "sub": subject,
        "name": name,
        "iat": now,
        "iat0": int(iat0) if iat0 else now,
        "exp": int(exp) if exp else now + _token_ttl(),
    }
    if via:
        payload["via"] = via
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _session_max_age() -> int:
    """Absolute lifetime of a session, however often it is renewed (default 30 days)."""
    try:
        return int(os.getenv(_ENV + "SESSION_MAX_AGE", "2592000"))
    except ValueError:
        return 2592000


def renewal_verdict(
    claims: Mapping[str, Any] | None,
    now: float,
    ttl: int | None = None,
    max_age: int | None = None,
) -> dict[str, Any]:
    """Should this session be handed a fresh token?"""
    ttl = _token_ttl() if ttl is None else ttl
    max_age = _session_max_age() if max_age is None else max_age
    if not isinstance(claims, Mapping):
        return {"renew": False, "reason": "no session claims to renew", "exp": None, "iat0": None}
    try:
        exp = float(claims["exp"])
    except (KeyError, TypeError, ValueError):
        return {"renew": False, "reason": "session has no expiry to extend", "exp": None, "iat0": None}
    if exp <= now:
        return {"renew": False, "reason": "session already expired - sign in again", "exp": None, "iat0": None}
    # The original sign-in: `iat0` once a session has been renewed, `iat` the first time, and `exp
    # - ttl` for a token issued before this claim existed (a session already in a phone's storage
    # must not be treated as brand new, which would restart its cap).
    try:
        iat0 = float(claims.get("iat0") or claims.get("iat") or (exp - ttl))
    except (TypeError, ValueError):
        iat0 = exp - ttl
    hard_deadline = iat0 + max_age
    if now >= hard_deadline:
        return {
            "renew": False,
            "reason": "this session has reached its maximum age - sign in with your passkey again",
            "exp": None,
            "iat0": int(iat0),
        }
    if now < exp - ttl / 2:
        return {"renew": False, "reason": "session still fresh", "exp": int(exp), "iat0": int(iat0)}
    # Never past the cap, and never SHORTER than what the client already holds: a renewal
    # that shaved time off would be a downgrade the client cannot refuse.
    new_exp = int(min(now + ttl, hard_deadline))
    if new_exp <= exp:
        return {"renew": False, "reason": "renewal would not extend this session", "exp": int(exp), "iat0": int(iat0)}
    return {"renew": True, "reason": "past half-life, extended", "exp": new_exp, "iat0": int(iat0)}


def verify_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "session expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid session")


def renew_if_due(token: str, now: float | None = None) -> str | None:
    if not token:
        return None
    try:
        claims = verify_token(token)
    except HTTPException:
        return None  # an expired or forged token is a login problem, not a renewal one
    verdict = renewal_verdict(claims, time.time() if now is None else now)
    if not verdict.get("renew"):
        return None
    return issue_token(
        str(claims.get("sub") or ""),
        str(claims.get("name") or ""),
        iat0=verdict.get("iat0"),
        exp=verdict.get("exp"),
    )


def session_is_valid(token: str) -> bool:
    """Non-raising check for the ASGI middleware."""
    if not token:
        return False
    try:
        verify_token(token)
        return True
    except HTTPException:
        return False


# --- LAN handoff tokens -------------------------------------------------------


def handoff_ttl() -> int:
    """Lifetime of a handoff token (default 5 minutes). It rides in a URL, so it must be
    short: URLs land in history, logs and screenshots."""
    try:
        return int(os.getenv(_ENV + "HANDOFF_TTL", "300"))
    except ValueError:
        return 300


def handoff_verdict(
    claims: Mapping[str, Any] | None,
    now: float,
    ttl: int | None = None,
) -> dict[str, Any]:
    """May this session be copied into a short-lived URL token, and until when?
    The handoff never outlives the session it came from."""
    ttl = handoff_ttl() if ttl is None else ttl
    if not isinstance(claims, Mapping):
        return {"ok": False, "reason": "no session claims to hand off"}
    try:
        exp = float(claims["exp"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": "session has no expiry"}
    if exp <= now:
        return {"ok": False, "reason": "session already expired - sign in again"}
    return {"ok": True, "exp": int(min(now + ttl, exp))}


def issue_handoff(claims: Mapping[str, Any], now: float | None = None) -> dict[str, Any]:
    """Mint the short-lived token handoff_verdict() approved, carrying the session's
    identity (sub/name/iat0) so renewal caps survive the copy."""
    now = time.time() if now is None else now
    verdict = handoff_verdict(claims, now)
    if not verdict.get("ok"):
        raise HTTPException(401, verdict.get("reason", "cannot mint a handoff token"))
    token = issue_token(
        str(claims.get("sub") or ""),
        str(claims.get("name") or ""),
        iat0=claims.get("iat0") or claims.get("iat"),
        exp=verdict["exp"],
        via="handoff",
    )
    return {"token": token, "exp": verdict["exp"], "expires_in": max(0, int(verdict["exp"] - now))}


def client_is_loopback(client_host: str | None) -> bool:
    """True when the connecting client is this machine. Used so that
    auth-disabled means LOCAL-ONLY rather than open to the network."""
    if not client_host:
        return False
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return client_host == "localhost"


# --- WebAuthn ceremonies ------------------------------------------------------


def begin_registration(request: Any, label: str = "passkey", bootstrap: str = "") -> dict[str, Any]:
    """Start a passkey enrollment. The FIRST enrollment seals the dashboard;
    later ones require a valid session (enforced by the route)."""
    store = _load()
    first_time = len(store.get("credentials", [])) == 0
    required = _bootstrap_token()
    if first_time and required:
        if not secrets.compare_digest(bootstrap or "", required):
            raise HTTPException(403, "bootstrap token required for first enrollment")

    rp_id = _derive_rp_id(request)
    if not rpid_is_usable(rp_id):
        raise _rpid_error(rp_id)

    # The first enrollment seals the dashboard, so it is the one request that hands out
    # ownership of the fleet rather than merely using it. With no bootstrap token configured
    # there is nothing to check it against, so it is limited to the machine itself: whoever is
    # at the keyboard is the only party who can be presumed to be the owner. A disk error is
    # one way to arrive here and a genuinely new install is the other; the second is the
    # commoner one and the more valuable to seize, so both are gated and only the wording
    # differs. The socket peer is deliberate -- see _socket_peer, and note that an unknown
    # peer is NOT the machine.
    damage = store_corruption()
    if first_time and not required:
        if not client_is_loopback(_socket_peer(request)):
            if damage:
                raise HTTPException(
                    403,
                    "the credential store was unreadable and has been kept as "
                    f"{damage['backup'] or 'a backup'} ({damage['reason']}). Enrolling a new passkey "
                    "is limited to the machine itself until one exists again - open the dashboard on "
                    "that machine, or set STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN and pass it.",
                )
            raise HTTPException(
                403,
                "the first passkey enrolled becomes the owner of this dashboard, and no "
                "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN is set for it to be checked against, so it is "
                "limited to the machine itself - open the dashboard on that machine, or set "
                "STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN and pass it.",
            )

    user_id = store.get("user_id")
    if not user_id:
        user_id = bytes_to_base64url(secrets.token_bytes(16))
        store["user_id"] = user_id
        _save(store)

    exclude = [PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"])) for c in store.get("credentials", [])]
    opts = generate_registration_options(
        rp_id=rp_id,
        rp_name=_rp_name(),
        user_id=base64url_to_bytes(user_id),
        user_name="dashboard-admin",
        user_display_name="Dashboard Admin",
        exclude_credentials=exclude or None,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
    )
    cid = _stash_challenge("reg", opts.challenge, {"label": label, "rp_id": rp_id}, ip=_client_ip(request))
    return {"challenge_id": cid, "options": json.loads(options_to_json(opts))}


def finish_registration(request: Any, challenge_id: str, credential: dict) -> dict[str, Any]:
    rec = _pop_challenge(challenge_id, "reg")
    verification = verify_registration_response(
        credential=credential,
        expected_challenge=rec["challenge"],
        expected_rp_id=rec["extra"]["rp_id"],
        expected_origin=_derive_origin(request),
    )
    store = _load()
    cred_id = bytes_to_base64url(verification.credential_id)
    if any(c["id"] == cred_id for c in store.get("credentials", [])):
        raise HTTPException(409, "credential already registered")
    store.setdefault("credentials", []).append(
        {
            "id": cred_id,
            "public_key": bytes_to_base64url(verification.credential_public_key),
            "sign_count": verification.sign_count,
            "name": rec["extra"].get("label", "passkey"),
            "created": time.time(),
            # The binding, recorded: from here on the Host header cannot introduce a
            # different rp_id (see rp_id_verdict).
            "rp_id": rec["extra"]["rp_id"],
        }
    )
    _save(store)
    token = issue_token(cred_id, name=rec["extra"].get("label", "passkey"))
    return {"ok": True, "token": token, "credential_id": cred_id}


def begin_authentication(request: Any) -> dict[str, Any]:
    store = _load()
    if not store.get("credentials"):
        raise HTTPException(400, "no credentials enrolled - setup required")
    rp_id = _derive_rp_id(request)
    if not rpid_is_usable(rp_id):
        raise _rpid_error(rp_id)
    allow = [PublicKeyCredentialDescriptor(id=base64url_to_bytes(c["id"])) for c in store["credentials"]]
    opts = generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=allow,
        user_verification=UserVerificationRequirement.PREFERRED,
    )
    cid = _stash_challenge("auth", opts.challenge, {"rp_id": rp_id}, ip=_client_ip(request))
    return {"challenge_id": cid, "options": json.loads(options_to_json(opts))}


def finish_authentication(request: Any, challenge_id: str, credential: dict) -> dict[str, Any]:
    rec = _pop_challenge(challenge_id, "auth")
    store = _load()
    cred_id = credential.get("id") or credential.get("rawId")
    match = next((c for c in store.get("credentials", []) if c["id"] == cred_id), None)
    if not match:
        raise HTTPException(404, "unknown credential")
    verification = verify_authentication_response(
        credential=credential,
        expected_challenge=rec["challenge"],
        expected_rp_id=rec["extra"]["rp_id"],
        expected_origin=_derive_origin(request),
        credential_public_key=base64url_to_bytes(match["public_key"]),
        credential_current_sign_count=match.get("sign_count", 0),
        require_user_verification=False,
    )
    match["sign_count"] = verification.new_sign_count
    # Self-heal the binding for credentials enrolled before rp_ids were recorded: this
    # authentication VERIFIED against rec["extra"]["rp_id"], which is proof, not a guess.
    if not match.get("rp_id") and rec["extra"].get("rp_id"):
        match["rp_id"] = rec["extra"]["rp_id"]
        logger.info("recorded rp_id %r for credential %s", match["rp_id"], match.get("name"))
    _save(store)
    token = issue_token(cast(str, cred_id), name=match.get("name", "passkey"))
    return {"ok": True, "token": token, "credential_id": cred_id}


def status(request: Any = None) -> dict[str, Any]:
    store = _load()
    out: dict[str, Any] = {
        "enabled": auth_enabled(),
        "setup_required": len(store.get("credentials", [])) == 0,
        "credentials": list_credentials(),
        "bootstrap_required": bool(_bootstrap_token()) and len(store.get("credentials", [])) == 0,
    }
    if request is not None:
        # The rp_id block is advisory: it tells the login screen which relying-party
        # id this origin can use and why it might not work. It reports the origin the
        # dashboard is SERVED at rather than the one a ceremony would accept, because
        # a diagnostic that refuses a mismatched Origin would remove the login
        # screen's hints exactly when a misconfigured proxy makes them worth
        # reading. It is derived from the
        # request, so any transport that answers `headers` or `url` differently than
        # expected can make it raise - and a diagnostic that raises would take the
        # login screen down with it, which is strictly worse than a screen missing
        # its hints. Hence the broad catch.
        #
        # It is assembled into its own dict and merged only once complete, so a
        # failure halfway cannot leave a caller with an `rp_id` and no verdict on
        # whether it is usable: the fields arrive together or not at all. An
        # undiscoverable rp_id is therefore absent, never guessed.
        advisory: dict[str, Any] = {}
        try:
            host = _host_only(request.headers.get("host", ""))
            origin = _served_origin(request)
            forced = _forced_rp_id()
            advisory["rp_id"] = forced or host
            advisory["secure_context"] = origin.startswith("https://") or host == "localhost"
            advisory["rpid_usable"] = True if forced else rpid_is_usable(host)
            if not advisory["secure_context"]:
                advisory["warning"] = "This origin is not a secure context. WebAuthn needs HTTPS or http://localhost."
            elif not advisory["rpid_usable"]:
                advisory["warning"] = (
                    f"'{host}' cannot be a WebAuthn rpId - use a hostname or set STRANDS_DASH_AUTH_RP_ID."
                )
        except Exception:
            # Attributable rather than silent: an operator looking at a login screen
            # with no rp_id hint has no other way to learn that deriving it failed.
            logger.debug("could not derive the rp_id advisory for this request", exc_info=True)
        else:
            out.update(advisory)
    return out
