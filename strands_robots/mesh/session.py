"""Shared Zenoh session and peer registry for the mesh networking layer.

This module provides a single, ref-counted :func:`zenoh.open` session per process
and a thread-safe registry of discovered peers.  It is the lowest layer of the
mesh stack - higher-level constructs (``Mesh``, presence, RPC) build on top.

The Zenoh dependency is **lazy**: ``import strands_robots.mesh_session`` does not
import ``zenoh`` at module level.  The first call to :func:`get_session` triggers
the real import.  If ``eclipse-zenoh`` is not installed the function returns
``None`` and all publish helpers become safe no-ops.

Connection strategy (when no explicit endpoint is configured):

1. Try to **listen** on ``tcp/127.0.0.1:{STRANDS_MESH_PORT}`` - this makes the
   first process the local router.
2. If the port is already bound, fall back to **client** mode and connect to the
   same endpoint.
3. Zenoh scouting (multicast) handles LAN discovery automatically.

Environment variables
---------------------
``ZENOH_CONNECT``
    Comma-separated remote endpoint(s) - e.g. ``tcp/10.0.0.1:7447``.
``ZENOH_LISTEN``
    Comma-separated listen endpoint(s).
``STRANDS_MESH_PORT``
    Local auto-mesh port (default ``7447``).
``STRANDS_MESH``
    Set to ``false`` to disable mesh globally.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Session singleton - one ``zenoh.Session`` per process, ref-counted


_SESSION: Any | None = None  # zenoh.Session when open, else None
_SESSION_LOCK = threading.Lock()
_SESSION_REFS: int = 0


# Constants


#: Default heartbeat frequency (Hz).  Presence payloads are published at this rate.
HEARTBEAT_HZ: float = 2.0

#: Default state-publishing frequency (Hz).
STATE_HZ: float = 10.0

#: Default camera-publishing frequency (Hz).  ``0`` disables the camera
#: loop - opt-in via the ``STRANDS_MESH_CAMERA_HZ`` environment variable
#: because frames are large and bandwidth-heavy.
CAMERA_HZ: float = 0.0

#: Seconds without a heartbeat before a peer is considered dead.
PEER_TIMEOUT: float = 10.0

# Operator-tunable via ``STRANDS_MESH_MAX_PEERS``.
# A real fleet is tens-to-low-hundreds of robots; 1024 leaves generous
# headroom while bounding the flood.
MAX_PEERS_DEFAULT: int = 1024


def _max_peers() -> int:
    """Resolve ``STRANDS_MESH_MAX_PEERS`` (lazy, restart-free).

    Bad / missing / non-positive input falls back to the default cap.
    """
    raw = os.getenv("STRANDS_MESH_MAX_PEERS")
    if raw is None:
        return MAX_PEERS_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return MAX_PEERS_DEFAULT
    return val if val > 0 else MAX_PEERS_DEFAULT


#: Pose publishing frequency (Hz).  Publishes SE(3) pose when a pose
#: provider (SLAM, odometry, VIO) is available on the robot.
POSE_HZ: float = 10.0

#: IMU publishing frequency (Hz).  Downsampled from hardware rate.
IMU_HZ: float = 10.0

#: Odometry publishing frequency (Hz).
ODOM_HZ: float = 10.0

#: Health/fleet-monitoring publishing frequency (Hz).
HEALTH_HZ: float = 0.5

#: LiDAR summary publishing frequency (Hz).
LIDAR_SUMMARY_HZ: float = 5.0

#: LiDAR state publishing frequency (Hz).
LIDAR_STATE_HZ: float = 1.0

#: Hand/end-effector state publishing frequency (Hz).
HAND_HZ: float = 50.0

#: Map info publishing frequency (Hz).
MAP_INFO_HZ: float = 0.2


# Backend selection helpers - when STRANDS_MESH_BACKEND is "iot" or "bridge",
# get_session() / put() / current_session() / session_alive() delegate to the
# transport factory instead of opening a Zenoh session directly. The "zenoh"
# default keeps the historical behaviour byte-for-byte so the 200+ existing
# mesh tests pass unmodified.


def _backend_choice() -> str:
    """Read STRANDS_MESH_BACKEND. Defaults to ``zenoh``. Unknown values fall
    back to ``zenoh`` (matches strands_robots.mesh.transport.factory)."""
    raw = os.getenv("STRANDS_MESH_BACKEND", "zenoh").strip().lower()
    if raw not in ("zenoh", "iot", "bridge"):
        return "zenoh"
    return raw


def _is_transport_backend() -> bool:
    """True when the backend is anything other than the legacy zenoh path."""
    return _backend_choice() in ("iot", "bridge")


# PeerInfo


@dataclass
class PeerInfo:
    """A discovered peer on the Zenoh mesh.

    Attributes:
        peer_id: Unique identifier for this peer (e.g. ``"so100-a1b2"``).
        peer_type: One of ``"robot"``, ``"sim"``, or ``"agent"``.
        hostname: The hostname the peer reported.
        last_seen: :func:`time.time` of the most recent heartbeat.
        caps: Arbitrary capability dictionary broadcast in the presence payload.
    """

    peer_id: str
    peer_type: str = "robot"
    hostname: str = ""
    last_seen: float = 0.0
    caps: dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Seconds since the last heartbeat."""
        return time.time() - self.last_seen

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "peer_id": self.peer_id,
            "type": self.peer_type,
            "hostname": self.hostname,
            "age": round(self.age, 1),
            **self.caps,
        }

    def __repr__(self) -> str:
        return f"PeerInfo(peer_id={self.peer_id!r}, type={self.peer_type!r}, age={self.age:.1f}s)"


# Peer registry - shared across all Mesh instances in the same process


_PEERS: dict[str, PeerInfo] = {}
_PEERS_VERSION: int = 0
_PEERS_LOCK = threading.Lock()


def update_peer(peer_id: str, peer_type: str, hostname: str, caps: dict[str, Any]) -> bool:
    """Insert or update a peer.  Returns ``True`` when the peer is new."""
    global _PEERS_VERSION  # noqa: PLW0603 - module-level singleton by design
    with _PEERS_LOCK:
        is_new = peer_id not in _PEERS
        # When a NEW peer would push us over the cap, evict the oldest
        # peer (smallest last_seen) to make room. Updates to EXISTING peers
        # never trigger eviction (they don't grow the dict). This bounds the
        # phantom-peer flood DoS while still admitting genuine new robots.
        if is_new:
            cap = _max_peers()
            while len(_PEERS) >= cap and _PEERS:
                oldest_id = min(_PEERS, key=lambda pid: _PEERS[pid].last_seen)
                del _PEERS[oldest_id]
                _PEERS_VERSION += 1
                logger.warning(
                    "Mesh: peer registry at cap (%d); evicted oldest peer %s",
                    cap,
                    oldest_id,
                )
        _PEERS[peer_id] = PeerInfo(
            peer_id=peer_id,
            peer_type=peer_type,
            hostname=hostname,
            last_seen=time.time(),
            caps=caps,
        )
        if is_new:
            _PEERS_VERSION += 1
        return is_new


def prune_peers(timeout: float = PEER_TIMEOUT) -> list[str]:
    """Remove peers that have not sent a heartbeat within *timeout* seconds.

    Returns:
        List of pruned peer IDs (may be empty).
    """
    global _PEERS_VERSION  # noqa: PLW0603
    now = time.time()
    pruned: list[str] = []
    with _PEERS_LOCK:
        stale = [pid for pid, p in _PEERS.items() if now - p.last_seen > timeout]
        for pid in stale:
            del _PEERS[pid]
            _PEERS_VERSION += 1
            pruned.append(pid)
    for pid in pruned:
        logger.info("Mesh: peer %s timed out", pid)
    return pruned


def get_peers() -> list[dict[str, Any]]:
    """Return all known peers as plain dicts."""
    with _PEERS_LOCK:
        return [p.to_dict() for p in _PEERS.values()]


def get_peer(peer_id: str) -> dict[str, Any] | None:
    """Return a single peer by *peer_id*, or ``None`` if unknown."""
    with _PEERS_LOCK:
        p = _PEERS.get(peer_id)
        return p.to_dict() if p else None


def peer_count() -> int:
    """Number of currently known (non-stale) peers."""
    with _PEERS_LOCK:
        return len(_PEERS)


def clear_peers() -> None:
    """Remove **all** peers.  Intended for tests only."""
    global _PEERS_VERSION  # noqa: PLW0603
    with _PEERS_LOCK:
        _PEERS.clear()
        _PEERS_VERSION += 1


# Session lifecycle


# Endpoint scheme validation. Under
# ``STRANDS_MESH_AUTH_MODE=mtls`` the wire-config builder restricts
# transports to TLS via ``link_protocols_block``; an operator who sets
# ``ZENOH_LISTEN=tcp/...`` (the documented format) gets a confusing
# zenoh runtime failure instead of a loud ``ValueError`` at config-build
# time. Validate the scheme up-front so the misconfig surfaces at the
# same loud-on-misconfig boundary as ``_float_env`` / ``_load_acl_file``
# / ``resolve_auth_mode``.
# #309: the predicate is "this scheme carries TLS bytes", so the constant is
# named for that intent. Zenoh 1.x TLS-bearing transports are tls, quic,
# wss (WebSocket-over-TLS, used in browser-bridge / ingress fleets) and
# unixsock (local-only but TLS-bearing in the Zenoh transport taxonomy).
# See https://zenoh.io/docs/manual/configuration/ (link protocols).
_TLS_BEARING_SCHEMES: tuple[str, ...] = ("tls", "quic", "wss", "unixsock")
# Backwards-compatible alias (the old name read as "valid under mtls").
_MTLS_OK_SCHEMES: tuple[str, ...] = _TLS_BEARING_SCHEMES
_NONE_OK_SCHEMES: tuple[str, ...] = ("tcp", "udp", "tls", "quic")


def _validate_endpoint_schemes(endpoints_raw: str | None, env_name: str, auth_mode: str) -> None:
    """Reject endpoints whose scheme is incompatible with ``auth_mode``.

    Args:
        endpoints_raw: Comma-separated endpoint string from env, or None.
        env_name: Name of the env var (for the error message).
        auth_mode: ``"mtls"`` or ``"none"``.

    Raises:
        ValueError: If ANY endpoint in the list uses a scheme blocked
            under the current ``auth_mode``.
    """
    if not endpoints_raw:
        return
    if auth_mode == "mtls":
        ok = _MTLS_OK_SCHEMES
    elif auth_mode == "none":
        ok = _NONE_OK_SCHEMES
    else:
        # Unknown auth_mode -- let resolve_auth_mode() raise downstream.
        return
    for ep in (e.strip() for e in endpoints_raw.split(",")):
        if not ep:
            continue
        scheme = ep.split("/", 1)[0].lower()
        if scheme not in ok:
            raise ValueError(
                f"{env_name}={endpoints_raw!r} contains endpoint {ep!r} with "
                f"scheme {scheme!r} -- under STRANDS_MESH_AUTH_MODE={auth_mode!r} "
                f"only {ok} schemes are accepted (the wire-config builder "
                f"restricts transports via link_protocols_block). Use a "
                f"compatible scheme or set STRANDS_MESH_AUTH_MODE=none for "
                f"the development posture (insecure)."
            )


def _build_config() -> Any:
    """Create a ``zenoh.Config`` from environment variables.

    The returned config layers (in order):

    1. Explicit endpoints from ``ZENOH_CONNECT`` / ``ZENOH_LISTEN``.
    2. Fleet namespace (:func:`_zenoh_config.namespace_block`).
    3. Scouting policy (gossip on, multicast off by default).
    4. Transport DoS bounds (max sessions, adminspace lockdown).
    5. Per-key-expression rate caps (``downsampling`` block).
    6. Per-message size caps (``low_pass_filter`` block).
    7. mTLS terminator + ACL when ``STRANDS_MESH_AUTH_MODE=mtls``
       (the default); skipped when explicitly set to ``none``.

    Returns:
        A ``zenoh.Config`` instance.

    Raises:
        ImportError: If ``eclipse-zenoh`` is not installed.
        ValueError: If env-var clamps are violated or
            ``STRANDS_MESH_AUTH_MODE`` is set to an unknown value.
        FileNotFoundError: If ``STRANDS_MESH_AUTH_MODE=mtls`` and any
            of the referenced cert/key/CA files do not exist.
    """
    import zenoh

    from strands_robots.mesh import _acl_config, _zenoh_config

    config = zenoh.Config()

    # Explicit endpoints from env vars (legacy ZENOH_CONNECT / ZENOH_LISTEN).
    # Validate endpoint schemes against
    # auth_mode BEFORE inserting them, so an operator who set
    # ``ZENOH_LISTEN=tcp/0.0.0.0:7447`` under the default
    # ``STRANDS_MESH_AUTH_MODE=mtls`` posture gets a loud
    # ``ValueError`` instead of a confusing zenoh runtime failure
    # (transport restricted to TLS by ``link_protocols_block``). Mirrors
    # the loud-on-misconfig discipline of ``_float_env``,
    # ``_load_acl_file``, ``resolve_auth_mode``.
    # Resolve auth_mode ONCE for the entire ``_build_config`` call so
    # endpoint validation and the later mTLS/none branch selection see
    # the SAME value, even when no ``Mesh.start`` thread-local is in
    # play (direct ``get_session()`` callers, integration tests).
    # Two independent reads of
    # ``os.environ['STRANDS_MESH_AUTH_MODE']`` between scheme
    # validation and block selection used to allow a concurrent test
    # fixture / plugin mutating env to put the two halves of the
    # builder out of sync (mtls scheme check vs none-block emission).
    _stashed_mode = _acl_config._get_thread_auth_mode()
    auth_mode = _stashed_mode if _stashed_mode is not None else _zenoh_config.resolve_auth_mode()

    connect = os.getenv("ZENOH_CONNECT")
    listen = os.getenv("ZENOH_LISTEN")
    _validate_endpoint_schemes(connect, "ZENOH_CONNECT", auth_mode)
    _validate_endpoint_schemes(listen, "ZENOH_LISTEN", auth_mode)
    if connect:
        endpoints = [e.strip() for e in connect.split(",")]
        config.insert_json5("connect/endpoints", json.dumps(endpoints))
    if listen:
        endpoints = [e.strip() for e in listen.split(",")]
        config.insert_json5("listen/endpoints", json.dumps(endpoints))

    # Fleet hardening, applied unconditionally.
    namespace = _zenoh_config.resolve_namespace()
    blocks: list[tuple[str, str]] = [
        _zenoh_config.namespace_block(),
        *_zenoh_config.scouting_block(),
        *_zenoh_config.transport_caps_block(),
        _zenoh_config.adminspace_block(),
        _zenoh_config.downsampling_block(),
        _zenoh_config.low_pass_filter_block(),
    ]

    # mTLS + ACL when auth_mode=mtls. The "none" mode emits everything
    # above except the auth + ACL blocks; it is dev-only. ``auth_mode``
    # was resolved once at the top of ``_build_config`` so endpoint
    # validation and block selection share
    # the same value.
    if auth_mode == "mtls":
        blocks.append(_zenoh_config.link_protocols_block())
        blocks.append(_zenoh_config.tls_block())
        # Issue #218: take ONE snapshot of the
        # ACL state and thread it through both the wire-config-build
        # path AND the refuse-to-start shape gate at Mesh.start. The
        # previous two-call pattern (``acl_block`` +
        # ``is_default_acl_in_use``) AND the cache-keyed-on-mtime
        # variant both had a TOCTOU window where an attacker rewriting
        # the ACL file between calls could bypass the shape gate while
        # feeding a malicious ACL into the wire config.
        # ``snapshot_acl`` consults a thread-local single-flight first,
        # so when ``Mesh.start`` has already resolved the ACL and
        # stashed it, this call returns THAT exact dict without
        # touching the filesystem.
        is_permissive, resolved_acl = _acl_config.snapshot_acl(namespace)
        blocks.append(_acl_config.acl_block_from(resolved_acl))
        # in mtls mode the ACL is the third line of
        # defence after the handshake. When the operator did not supply
        # STRANDS_MESH_ACL_FILE, the built-in default is permissive
        # (any CA-signed peer publishes/subscribes anywhere). Surface a
        # WARNING on every session open so operators who forgot the env
        # var hear about it -- parallel to the auth_mode=none warning
        # below.
        # only emit this WARNING when the
        # operator has NOT explicitly opted into the dev/lab posture.
        # Mesh.start emits a more-specific INFO/ERROR breadcrumb with
        # the opt-in context; emitting both fires two log lines about
        # the same thing on every session open AND has the WARNING
        # contradict the operator's explicit acknowledgement.
        accept_permissive = os.getenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if is_permissive and not accept_permissive:
            logger.warning(
                "STRANDS_MESH_ACL_FILE unset -- using PERMISSIVE built-in "
                "default ACL. Any CA-signed peer can publish/subscribe "
                "on any key. For production fleets supply an operator "
                "ACL enumerating each peer's cert CN; see "
                "examples/mesh_acl_example.json5."
            )
    else:
        logger.error(
            "[mesh] WIRE SECURITY DISABLED -- STRANDS_MESH_AUTH_MODE=none. "
            "Both the mTLS terminator AND the ACL block are off. "
            "Operator opted in via STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1. "
            "This mode is for development on trusted networks only."
        )

    for path, value in blocks:
        config.insert_json5(path, value)

    return config


def current_session() -> Any | None:
    """Return the existing session/transport without bumping the refcount.

    Backend-aware: returns the active transport singleton when
    ``STRANDS_MESH_BACKEND`` is ``iot`` / ``bridge``, otherwise the raw
    Zenoh session (legacy behaviour).
    """
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import current_transport

        return current_transport()

    with _SESSION_LOCK:
        return _SESSION


def get_session() -> Any | None:
    """Acquire the shared mesh transport (lazy, ref-counted).

    Backend selection comes from ``STRANDS_MESH_BACKEND``:

    - ``zenoh`` (default) - open / reuse a ``zenoh.Session`` exactly as before.
      Returned object is the raw session; callers can ``.declare_subscriber()``
      on it.
    - ``iot`` / ``bridge`` - delegate to
      :mod:`strands_robots.mesh.transport.factory`; the returned object is an
      :class:`~strands_robots.mesh.transport.IotMqttTransport` or
      :class:`~strands_robots.mesh.transport.BridgeTransport` which **also**
      exposes ``put()`` / ``declare_subscriber()`` / ``close()`` so existing
      Mesh code works unchanged.

    Returns:
        Backend-dependent: ``zenoh.Session``, ``IotMqttTransport``,
        ``BridgeTransport``, or ``None`` if the chosen backend is unavailable.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    if _is_transport_backend():
        # Delegate to the transport factory. The factory holds its own
        # refcount independently of _SESSION_REFS - that's fine, callers
        # that release_session() will see the matching release_transport().
        from strands_robots.mesh.transport.factory import get_transport

        return get_transport()

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import zenoh  # noqa: F811 - lazy import
        except ImportError:
            logger.debug("eclipse-zenoh not installed - mesh disabled")
            return None

        # STRANDS_MESH_PORT is read at session-open time so a process can be
        # configured via env vars without re-importing.  Bad input falls back
        # to the default and warns once - never raises (the default behaviour
        # is to keep the mesh quietly off rather than crash the host robot).
        port_env = os.getenv("STRANDS_MESH_PORT", "7447")
        try:
            mesh_port = int(port_env)
            if not (1 <= mesh_port <= 65535):
                raise ValueError(f"port {mesh_port} out of range")
        except ValueError as exc:
            logger.warning(
                "Invalid STRANDS_MESH_PORT=%r (%s) - falling back to 7447",
                port_env,
                exc,
            )
            mesh_port = 7447

        connect_env = os.getenv("ZENOH_CONNECT")
        listen_env = os.getenv("ZENOH_LISTEN")

        # When no explicit endpoints are set, try to become the local router.
        # both the auto-listener AND the client
        # fallback below MUST go through ``_build_config()`` -- the
        # threat-coverage table claims namespace + mTLS + ACL +
        # downsampling + low_pass_filter + max_sessions + adminspace
        # lockdown apply on every Zenoh path, and earlier revisions, the auto-
        # listener path used a bare ``zenoh.Config()`` and silently
        # bypassed all of them. The default deployment shape (no
        # ZENOH_CONNECT / ZENOH_LISTEN, first peer in the process) is
        # exactly what most operators hit on first run; the security
        # claim was therefore false on the most common code path.
        # Compose mTLS-aware endpoints (``tls/...`` when auth_mode=mtls,
        # plain ``tcp/...`` otherwise) so ``transport/link/protocols``
        # restriction does not produce an unusable session.
        if not connect_env and not listen_env:
            from strands_robots.mesh import _acl_config
            from strands_robots.mesh._zenoh_config import resolve_auth_mode

            # Loud-on-misconfig: if STRANDS_MESH_AUTH_MODE is set to
            # anything other than "mtls"/"none", let resolve_auth_mode()
            # raise ValueError here. Mesh.start crashes with a clear
            # stacktrace instead of silently falling back to "mtls"
            # and emitting three confusing fallback warnings later
            # (the prior try/except was dead -- _build_config() below
            # invokes resolve_auth_mode() again unconditionally).
            # Aligns with the loud-on-misconfig posture of _float_env
            # and _load_acl_file. Addressed in PR-224 R1.
            #
            # Prefer the thread-local
            # ``auth_mode`` stash from ``Mesh.start``. This is the same
            # one-resolve-per-Mesh.start invariant ``_build_config``
            # already honours at line 328-329; without it, the listener
            # endpoint scheme (composed here) and the wire-config block
            # (composed inside ``_build_config``) can disagree if
            # ``STRANDS_MESH_AUTH_MODE`` flips between the two reads
            # (concurrent test fixture, plugin mutating ``os.environ``,
            # or ``Mesh.start`` clearing the snapshot mid-call). Direct
            # callers of ``get_session()`` without ``Mesh.start``
            # priming the snapshot fall through to ``resolve_auth_mode``
            # (the legacy contract).
            _stashed_mode = _acl_config._get_thread_auth_mode()
            _auth_mode = _stashed_mode if _stashed_mode is not None else resolve_auth_mode()
            scheme = "tls" if _auth_mode == "mtls" else "tcp"
            local_ep = f"{scheme}/127.0.0.1:{mesh_port}"

            # Build config OUTSIDE the listener try so a bad ACL /
            # TLS configuration (ValueError from _build_config) propagates
            # loudly to Mesh.start rather than being silently downgraded
            # to client-mode as if it were a port-already-bound error.
            cfg = _build_config()
            cfg.insert_json5("listen/endpoints", json.dumps([local_ep]))
            cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
            # Resolve zenoh.ZError dynamically -- when tests mock the
            # zenoh module, ``zenoh.ZError`` is a MagicMock (not a class)
            # and including it directly in the except tuple raises
            # TypeError. Fall back to a benign placeholder when zenoh is
            # mocked or ZError is not a real class.
            _ZError = getattr(zenoh, "ZError", None)
            _ZError = _ZError if isinstance(_ZError, type) and issubclass(_ZError, BaseException) else RuntimeError
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (listener on %s)", local_ep)
                return _SESSION
            except (RuntimeError, OSError, ConnectionError, _ZError) as exc:
                # Narrow tuple per AGENTS.md > Review Learnings (#86):
                # ``RuntimeError`` / ``OSError`` / ``ConnectionError`` /
                # ``zenoh.ZError`` cover the realistic transport-side
                # failures (port-bound, bad iface, broker drop) without
                # masking config-shape ``ValueError`` raised by
                # ``_build_config`` upstream (which is now outside the
                # try anyway -- belt-and-braces).
                # Port already bound (the most common case) is not an error.
                logger.debug(
                    "Zenoh listener on %s unavailable (%s) - trying client mode",
                    local_ep,
                    exc,
                )

            # Fall back to client mode - connect to the existing listener.
            # Build cfg OUTSIDE the try so a config-shape ValueError
            # (NaN env clamp, missing TLS file, bad ACL) propagates
            # loudly to Mesh.start instead of being silently downgraded
            # to "session unavailable".
            cfg = _build_config()
            cfg.insert_json5("mode", '"client"')
            cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (client → %s)", local_ep)
                return _SESSION
            except (RuntimeError, OSError, ConnectionError, _ZError) as exc:
                # Narrow tuple per AGENTS.md > Review Learnings (#86):
                # transport-level failures only; config-shape ValueError
                # propagates to caller so misconfigured mTLS surfaces loudly.
                logger.warning("Zenoh session open failed (client mode): %s", exc)
                return None

        # Explicit endpoints provided via env vars.
        # Build cfg outside the try (same loud-on-misconfig discipline
        # as the auto-listener path).
        cfg = _build_config()
        # Re-resolve _ZError under the explicit-endpoints branch (reached
        # when _LOCAL_LISTEN env var is unset, so the listener-block
        # binding above never executed).
        _ZError = getattr(zenoh, "ZError", None)
        _ZError = _ZError if isinstance(_ZError, type) and issubclass(_ZError, BaseException) else RuntimeError
        try:
            _SESSION = zenoh.open(cfg)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened")
            return _SESSION
        except (RuntimeError, OSError, ConnectionError, _ZError) as exc:
            logger.warning("Zenoh session open failed: %s", exc)
            return None


def _get_zenoh_session_directly() -> Any | None:
    """Open/reuse the Zenoh session directly, bypassing transport-backend routing.

    This is used by :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
    when it is instantiated as part of a :class:`BridgeTransport`. In that scenario,
    ``get_session()`` would re-enter the factory's ``_LOCK`` (since
    ``_is_transport_backend()`` returns True for bridge mode) causing a deadlock.

    This function always goes through the raw Zenoh path regardless of
    ``STRANDS_MESH_BACKEND``. It shares the same ``_SESSION`` singleton and
    ``_SESSION_LOCK``.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import zenoh
        except ImportError:
            logger.debug("eclipse-zenoh not installed - mesh disabled")
            return None

        port_env = os.getenv("STRANDS_MESH_PORT", "7447")
        try:
            mesh_port = int(port_env)
            if not (1 <= mesh_port <= 65535):
                raise ValueError(f"port {mesh_port} out of range")
        except ValueError as exc:
            logger.warning(
                "Invalid STRANDS_MESH_PORT=%r (%s) - falling back to 7447",
                port_env,
                exc,
            )
            mesh_port = 7447

        connect_env = os.getenv("ZENOH_CONNECT")
        listen_env = os.getenv("ZENOH_LISTEN")

        if not connect_env and not listen_env:
            # (See get_session above for full rationale.)
            from strands_robots.mesh import _acl_config
            from strands_robots.mesh._zenoh_config import resolve_auth_mode

            # Loud-on-misconfig: if STRANDS_MESH_AUTH_MODE is set to
            # anything other than "mtls"/"none", let resolve_auth_mode()
            # raise ValueError here. Mesh.start crashes with a clear
            # stacktrace instead of silently falling back to "mtls"
            # and emitting three confusing fallback warnings later
            # (the prior try/except was dead -- _build_config() below
            # invokes resolve_auth_mode() again unconditionally).
            # Aligns with the loud-on-misconfig posture of _float_env
            # and _load_acl_file. Addressed in PR-224 R1.
            #
            # Prefer the thread-local
            # ``auth_mode`` stash. Mirrors the same fix at the
            # ``get_session`` boundary upstairs and the
            # ``_build_config`` boundary at line 328-329. See full
            # rationale on the upstream copy.
            _stashed_mode = _acl_config._get_thread_auth_mode()
            _auth_mode = _stashed_mode if _stashed_mode is not None else resolve_auth_mode()
            scheme = "tls" if _auth_mode == "mtls" else "tcp"
            local_ep = f"{scheme}/127.0.0.1:{mesh_port}"

            # Build cfg outside the listener try so config-shape
            # ValueError surfaces loudly.
            cfg = _build_config()
            cfg.insert_json5("listen/endpoints", json.dumps([local_ep]))
            cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
            _ZError = getattr(zenoh, "ZError", None)
            _ZError = _ZError if isinstance(_ZError, type) and issubclass(_ZError, BaseException) else RuntimeError
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (listener on %s)", local_ep)
                return _SESSION
            except (RuntimeError, OSError, ConnectionError, _ZError) as exc:
                # Narrow tuple mirroring the narrowing applied in
                # get_session() upstairs. Config-shape
                # ValueError now propagates instead of being swallowed at DEBUG.
                logger.debug(
                    "Zenoh listener on %s unavailable (%s) - trying client mode",
                    local_ep,
                    exc,
                )

            cfg = _build_config()
            cfg.insert_json5("mode", '"client"')
            cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (client → %s)", local_ep)
                return _SESSION
            except (RuntimeError, OSError, ConnectionError, _ZError) as exc:
                logger.warning("Zenoh session open failed (client mode): %s", exc)
                return None

        cfg = _build_config()
        _ZError = getattr(zenoh, "ZError", None)
        _ZError = _ZError if isinstance(_ZError, type) and issubclass(_ZError, BaseException) else RuntimeError
        try:
            _SESSION = zenoh.open(cfg)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened")
            return _SESSION
        except (RuntimeError, OSError, ConnectionError, _ZError) as exc:
            logger.warning("Zenoh session open failed: %s", exc)
            return None


def release_session() -> None:
    """Release one reference to the shared mesh session.

    Delegates to the transport factory when the active backend is
    ``iot`` / ``bridge``; otherwise falls back to the legacy Zenoh refcount.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import release_transport

        release_transport()
        return

    with _SESSION_LOCK:
        if _SESSION_REFS <= 0:
            return
        _SESSION_REFS -= 1
        if _SESSION_REFS <= 0 and _SESSION is not None:
            try:
                _SESSION.close()
            except Exception:
                pass
            _SESSION = None
            _SESSION_REFS = 0
            logger.info("Zenoh mesh session closed")


def session_alive() -> bool:
    """Return ``True`` if the current backend's session/transport is open."""
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import current_transport

        t = current_transport()
        return t is not None and t.is_alive()

    with _SESSION_LOCK:
        return _SESSION is not None


# Publish helper


def put(key: str, data: dict[str, Any]) -> None:
    """Publish a JSON payload to the mesh.

    Fire-and-forget. No-op when no session/transport is open.

    Backend-aware: delegates to the active transport's ``put()`` when
    running under ``STRANDS_MESH_BACKEND=iot`` / ``bridge``; otherwise
    encodes JSON and pushes to the Zenoh session directly (legacy path).
    """
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import current_transport

        t = current_transport()
        if t is None:
            return
        try:
            t.put(key, data)
        except Exception as exc:
            logger.debug("Mesh transport put error on %s: %s", key, exc)
        return

    if _SESSION is None:
        return
    try:
        _SESSION.put(key, json.dumps(data).encode())
    except Exception as exc:
        logger.debug("Zenoh put error on %s: %s", key, exc)


# Process cleanup


def _atexit_cleanup() -> None:
    """Best-effort session teardown on process exit."""
    global _SESSION, _SESSION_REFS  # noqa: PLW0603
    with _SESSION_LOCK:
        if _SESSION is not None:
            try:
                _SESSION.close()
            except Exception:
                pass
            _SESSION = None
            _SESSION_REFS = 0


atexit.register(_atexit_cleanup)


def _session_alive_directly() -> bool:
    """Return ``True`` if the raw Zenoh session is open, bypassing backend routing.

    Used by :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
    to avoid recursion when operating inside a :class:`BridgeTransport`.
    """
    with _SESSION_LOCK:
        return _SESSION is not None


def _current_zenoh_session_directly() -> Any | None:
    """Return the raw Zenoh session without bumping refcount, bypassing backend routing.

    Used by :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport`
    to avoid recursion when operating inside a :class:`BridgeTransport`.
    """
    with _SESSION_LOCK:
        return _SESSION
