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
3. Zenoh gossip scouting propagates peers reachable through those endpoints.
   Multicast scouting is **disabled by default** (LAN discovery attack
   surface); operators on a controlled LAN can opt in with
   ``STRANDS_MESH_MULTICAST=true``. Cross-host peers otherwise need explicit
   ``ZENOH_CONNECT`` endpoints.

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
``STRANDS_MESH_MULTICAST``
    ``true`` to opt into LAN multicast scouting (logs a warning).
    Default ``false``.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from strands_robots._mesh_switch import MESH_ENV_VAR
from strands_robots.mesh._backend_select import select_backend
from strands_robots.utils import partial_construction_repr

logger = logging.getLogger(__name__)


# Session singleton - one ``zenoh.Session`` per process, ref-counted


_SESSION: Any | None = None  # zenoh.Session when open, else None
_SESSION_LOCK = threading.Lock()
_SESSION_REFS: int = 0

# One-shot guard so an absent ``eclipse-zenoh`` is reported at most once per
# process.  Both session-open paths re-attempt the import on every call (nothing
# is cached when it fails), so a fleet of N robots in one process would otherwise
# emit N copies of an identical, static fact.  A set (mutated via .add, never
# reassigned) avoids a `global` rebind so static analysis sees it as used -- the
# same shape as ``_software_render_warned`` in the MuJoCo backend.  Mutated only
# under ``_SESSION_LOCK``.
_zenoh_missing_warned: set[str] = set()

# One-shot-per-topic guard for a payload that cannot be JSON-encoded.  Unlike a
# transient wire failure, an unencodable payload fails identically on every
# retry, so a 50 Hz publisher would otherwise emit one report per tick forever.
# Keyed on the topic because the payload builder is per-topic: a broken builder
# is broken for every tick of ITS key and says nothing about any other.  A set
# (mutated via .add, never reassigned) avoids a `global` rebind so static
# analysis sees it as used.  Read and mutated WITHOUT ``_SESSION_LOCK``, because
# :func:`_put_zenoh_directly` deliberately does not take it (a 50 Hz teleop loop
# must not serialise on the session lock); the worst race is two threads each
# reporting the same topic once.
_unencodable_topics_warned: set[str] = set()


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


#: Mode used by every session that is NOT the machine's hub (the process that
#: won the ``STRANDS_MESH_PORT`` listener). See :func:`_apply_fallback_topology`
#: for why this is ``client`` and not ``peer``.
FALLBACK_MODE_DEFAULT = "client"

#: Backoff for the background connect-retry on a non-hub session. Deliberately
#: gentle: the hub is on loopback, and a restarted hub only has to be noticed
#: within a few seconds for a robot to rejoin.
FALLBACK_RETRY = {"period_init_ms": 1000, "period_max_ms": 8000, "period_increase_factor": 2}


def _fallback_mode() -> str:
    """Resolve ``STRANDS_MESH_FALLBACK_MODE`` (``client`` | ``peer``).

    Anything else warns and behaves as the default rather than raising: a
    typo in this variable must not take a robot's mesh offline, and the
    correct value is the one that works.
    """
    raw = (os.getenv("STRANDS_MESH_FALLBACK_MODE") or "").strip().lower()
    if not raw:
        return FALLBACK_MODE_DEFAULT
    if raw in ("client", "peer"):
        return raw
    logger.warning(
        "Invalid STRANDS_MESH_FALLBACK_MODE=%r (expected 'client' or 'peer') - using %r",
        raw,
        FALLBACK_MODE_DEFAULT,
    )
    return FALLBACK_MODE_DEFAULT


def _apply_fallback_topology(cfg: Any, local_ep: str, scheme: str) -> str:
    """Configure a non-hub session and return the mode it will open in.

    The machine's first process listens on ``STRANDS_MESH_PORT`` and everyone
    after it lands here, connecting to that hub. The mode chosen here decides
    whether two *children* can hear each other at all, which is not obvious
    and was measured rather than assumed (three throwaway sessions: hub,
    publisher, then a LATE subscriber, hub as the only configured endpoint,
    counting frames on ``strands/<peer>/input/<device>``):

    ==============  ============  ==============
    hub mode        child mode    frames arrived
    ==============  ============  ==============
    peer            peer          **0 of 62**
    router          peer          **0 of 62**
    peer            client        42 of 62
    router          client        42 of 62
    ==============  ============  ==============

    A Zenoh 1.x **peer** assumes a full mesh and will not take traffic relayed
    by an intermediary - not even by a router. ``routing/peer/mode`` (the knob
    that used to make peers route for each other) no longer exists in 1.10:
    ``insert_json5`` raises ``ZError("unknown key")``. So with peer-mode
    children every child-to-child topic silently delivers nothing, while every
    child-to-hub topic works, because that is the link the child opened
    itself. That is exactly the teleop bug this fixes: a leader published 209
    frames to a follower whose counters all read zero.

    A **client** delegates routing to whatever it is connected to, so the hub
    relays for it. The property the previous peer-mode fallback was chosen for
    is kept: with ``connect/exit_on_failure=false`` plus ``connect/retry``, a
    client re-links to a restarted hub on its own (measured: 30 frames before
    the hub was killed, 0 during a 6s outage, 63 after it came back - no
    process restart).

    ``STRANDS_MESH_FALLBACK_MODE=peer`` restores the old topology for an
    operator who wants direct peer links (and who must then arrange them:
    a peer only hears publishers it is directly linked to).

    Args:
        cfg: The ``zenoh.Config`` to mutate (already built by ``_build_config``,
            so namespace / mTLS / ACL / downsampling stay applied).
        local_ep: The hub endpoint to dial, e.g. ``tcp/127.0.0.1:7447``.
        scheme: ``tcp`` or ``tls``, matching the resolved auth mode.

    Returns:
        The mode the session will open in: ``"client"`` or ``"peer"``.
    """
    mode = _fallback_mode()
    if mode == "peer":
        # Ephemeral listener: a peer needs its own listener for another peer
        # to dial it. Port 0 = let the OS choose.
        cfg.insert_json5("listen/endpoints", json.dumps([f"{scheme}/127.0.0.1:0"]))
    else:
        cfg.insert_json5("mode", json.dumps("client"))
    cfg.insert_json5("connect/endpoints", json.dumps([local_ep]))
    # Never give up on the hub endpoint; retry with gentle backoff.
    cfg.insert_json5("connect/exit_on_failure", "false")
    cfg.insert_json5("connect/retry", json.dumps(FALLBACK_RETRY))
    return mode


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

#: Per-step telemetry publishing frequency (Hz) -- the ``publish_step`` throttle
#: on the hardware control loop and the simulation ``run_policy`` hook. Used
#: when ``STRANDS_MESH_STREAM_HZ`` is unset; see
#: :func:`stream_min_period_from_env`.
STREAM_HZ: float = 10.0


def hz_from_env(name: str) -> tuple[float | None, str | None]:
    """Read a loop rate (Hz) held in an environment variable.

    Every mesh loop rate is operator-tunable through an environment variable,
    and each reader has its own documented fallback for a value it cannot use:
    a sensor loop keeps its built-in rate, the camera loop stays off, the
    teleop apply ceiling reverts to its default. What they share is which
    values are usable at all.
    Every consumer turns the rate into a period with ``1.0 / hz``, and
    ``float()`` accepts ``"inf"``, overflows ``"1e999"`` to ``inf`` and accepts
    ``"nan"`` -- none of which survives that division. ``inf`` yields a zero
    period, so a loop that meant to wait between ticks never waits; ``nan``
    yields a period that compares ``False`` against every bound, so a cap
    built from it never trips. Both read as "no rate limit" rather than as the
    misconfiguration they are, which is why non-finite input is reported here
    instead of being passed on.

    Args:
        name: Environment variable to read.

    Returns:
        ``(hz, None)`` when *name* holds a finite number, ``(None, None)`` when
        it is unset or blank, and ``(None, reason)`` when it holds a value no
        loop can honor. *reason* names the variable and the offending value so
        a caller can log it alongside whichever fallback it documents; callers
        decide the fallback, this decides only what is usable.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None, None
    try:
        hz = float(raw)
    except (TypeError, ValueError):
        return None, f"{name}={raw!r} is not a number"
    if not math.isfinite(hz):
        return None, f"{name}={raw!r} is not a finite rate"
    return hz, None


def stream_min_period_from_env() -> float:
    """Resolve the minimum period between per-step telemetry publishes.

    Two call sites throttle ``publish_step`` against a monotonic clock -- the
    hardware control loop (``HardwareRobot.__init__``) and the simulation
    ``run_policy`` hook -- and both did it by dividing
    ``STRANDS_MESH_STREAM_HZ`` straight from the environment. That raises
    ``ZeroDivisionError`` on ``0`` and ``ValueError`` on any non-numeric value,
    and one of those divisions runs in a constructor, so the failure is not
    "telemetry degraded" but "every ``Robot(..., mode="real")`` construction
    raises". ``0`` is a realistic input rather than an adversarial one: the
    sibling knob ``STRANDS_MESH_CAMERA_HZ`` advertises non-positive as off, so
    an operator disabling step telemetry the same way bricked hardware
    bring-up.

    A period rather than a rate is returned because that is what both callers
    hold, and it lets "off" be expressed as ``inf``: no *finite* elapsed time
    reaches an infinite period, so the throttle refuses every step it is asked
    about.

    That is a property of the elapsed time as much as of the period, so it does
    not survive every base. A caller starting its throttle base at ``-inf`` --
    which is what a monotonic base wants to be, a reading being meaningful only
    relative to another one, so that a rollout's first step is due wherever the
    platform's epoch sits -- computes an infinite elapsed time on that first
    step, and ``inf >= inf`` is true. The subtraction alone then lets exactly
    one publish past the opt-out. Such a caller tests this value for finiteness
    once, where it resolves it, and gates the publish on that: the returned
    period alone is not the whole opt-out.

    The fallback for an unusable value follows :meth:`Mesh._resolve_camera_hz`,
    the other opt-out publishing loop: warn naming the variable and the
    offending value, then leave publishing off rather than substitute a rate
    the operator did not ask for. Step telemetry is observability, not control,
    so nothing downstream misbehaves when it stops -- and the warning is what
    makes "no step tiles" diagnosable.

    Returns:
        ``1.0 / STREAM_HZ`` when ``STRANDS_MESH_STREAM_HZ`` is unset or blank,
        ``1.0 / hz`` when it names a positive finite rate, and ``math.inf`` --
        publishing off -- when it is non-positive or holds a value no loop can
        honor.
    """
    hz, reason = hz_from_env("STRANDS_MESH_STREAM_HZ")
    if reason is not None:
        logger.warning(
            "[mesh] %s; per-step telemetry publishing is OFF. "
            "Set STRANDS_MESH_STREAM_HZ to a positive rate to enable it.",
            reason,
        )
        return math.inf
    if hz is None:
        return 1.0 / STREAM_HZ
    if hz <= 0:
        # Explicit operator opt-out, same spelling as STRANDS_MESH_CAMERA_HZ.
        return math.inf
    return 1.0 / hz


# Backend selection helpers - when STRANDS_MESH_BACKEND is "iot" or "bridge",
# get_session() / put() / current_session() / session_alive() delegate to the
# transport factory instead of opening a Zenoh session directly. The "zenoh"
# default keeps the historical behaviour byte-for-byte so the 200+ existing
# mesh tests pass unmodified.


def _backend_choice() -> str:
    """Read STRANDS_MESH_BACKEND. Defaults to ``zenoh``. Unknown values fall
    back to ``zenoh`` with a report.

    Resolved by :func:`strands_robots.mesh._backend_select.select_backend`
    rather than re-read here, so the accepted values and the report of an
    unrecognized one have a single owner. This resolver is the gate the
    transport factory sits behind, so a report living only in the factory could
    never reach an operator who mistyped the variable.
    """
    return select_backend()


def _is_transport_backend() -> bool:
    """True when the backend is anything other than the legacy zenoh path."""
    return _backend_choice() in ("iot", "bridge")


def zenoh_error_types() -> tuple[type[BaseException], ...]:
    """Exception types a best-effort Zenoh lifecycle op may raise.

    Covers ``open`` / ``declare_subscriber`` / ``declare_publisher`` /
    ``undeclare`` / ``close`` failures on the realistic transport-side paths
    (port already bound, bad interface, broker drop, entity already released):

      - ``zenoh.ZError`` -- the native Zenoh error. On real ``eclipse-zenoh`` it
        subclasses ``Exception`` directly (NOT ``RuntimeError``), so it must be
        named explicitly or a genuine transport failure escapes best-effort
        cleanup. When the ``zenoh`` module is mocked in tests, ``zenoh.ZError``
        is a ``MagicMock`` (not a class) and cannot go in an ``except`` tuple,
        so it is dropped and the benign builtins below still apply.
      - ``OSError`` / ``ConnectionError`` -- socket / broker transport faults.
      - ``RuntimeError`` -- internal binding faults.

    Programmer errors (``TypeError`` / ``AttributeError`` / ``MemoryError``) are
    deliberately excluded so they surface loudly instead of being swallowed by a
    best-effort cleanup path.
    """
    base: tuple[type[BaseException], ...] = (RuntimeError, OSError, ConnectionError)
    try:
        import zenoh
    except ImportError:
        return base
    zerr = getattr(zenoh, "ZError", None)
    if isinstance(zerr, type) and issubclass(zerr, BaseException):
        return (*base, zerr)
    return base


# PeerInfo


@dataclass
class PeerInfo:
    """A discovered peer on the Zenoh mesh.

    Attributes:
        peer_id: Unique identifier for this peer (e.g. ``"so100-a1b2"``).
        peer_type: One of ``"robot"``, ``"sim"``, or ``"agent"``.
        hostname: The hostname the peer reported.
        last_seen_mono: :func:`time.monotonic` reading taken when this process
            last saw a heartbeat from the peer. Monotonic because every reader
            subtracts it from a later reading to get an age, and an age decides
            whether the peer is still alive - see :meth:`age` and
            :func:`prune_peers`. It is a local observation, never a stamp the
            peer sent, and it is not serialised: :meth:`to_dict` reports ``age``.
        caps: Arbitrary capability dictionary broadcast in the presence payload.
    """

    peer_id: str
    peer_type: str = "robot"
    hostname: str = ""
    last_seen_mono: float = 0.0
    caps: dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Seconds since the last heartbeat."""
        return time.monotonic() - self.last_seen_mono

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly).

        ``caps`` merges *first* so the four locally-decided fields win a name
        collision. ``caps`` is the peer's own presence payload, and every field
        below it is something this process decided about the peer rather than
        something the peer reported about itself: ``age`` is the observation
        described on :attr:`last_seen_mono`, and ``peer_id`` is the key the
        registry files it under.

        Spread last, a payload carrying any of those four names replaced the
        local reading. An ``age`` the sender chooses defeats every staleness
        verdict read from it, and a ``peer_id`` the sender chooses is the key
        ``Mesh.peers_by_id`` and ``Mesh.get_peer`` look the peer up by.
        """
        return {
            **self.caps,
            "peer_id": self.peer_id,
            "type": self.peer_type,
            "hostname": self.hostname,
            "age": round(self.age, 1),
        }

    def __repr__(self) -> str:
        try:
            return f"PeerInfo(peer_id={self.peer_id!r}, type={self.peer_type!r}, age={self.age:.1f}s)"
        except AttributeError:
            return partial_construction_repr(self)


#: Presence ``robot_type`` values that name a simulation rather than metal.
#: The two in-tree publishers set ``peer_type="sim"`` for a
#: :class:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine` and
#: ``"robot"`` for hardware; ``"simulation"`` / ``"mujoco"`` are accepted so a
#: third-party publisher spelling the same fact differently is still read.
_SIM_PEER_TYPES: frozenset[str] = frozenset({"sim", "simulation", "mujoco"})


def peer_is_physical(peer: Mapping[str, Any] | None) -> tuple[bool, str]:
    """Is this peer metal? Returns ``(physical, why)``, failing closed.

    Reads one entry of the :func:`get_peers` snapshot - the flat dict
    :meth:`PeerInfo.to_dict` returns, which spreads the peer's presence payload
    at the top level rather than nesting it under a ``presence`` key. Every
    marker below is a key some publisher in
    :meth:`~strands_robots.mesh.core.Mesh._build_presence` really sets, so no
    rung is dead on arrival.

    Fail-closed means: physical unless the peer can be SHOWN to be a sim. An
    absent peer, a peer whose presence carries no sim marker, and a marker this
    function cannot read are all metal. The direction of each rung follows from
    that, and the two directions are deliberately different:

    * ``hw`` is a positive METAL marker, so it is read permissively (any
      non-empty string) and checked first - a peer reporting real hardware is
      metal whatever else it claims.
    * ``robot_type`` and ``world`` are positive SIM markers, so they are read
      strictly (an exact token; ``world is True``, not merely truthy). A value
      this function cannot read falls through to metal instead of being taken
      for a sim.

    The verdict is a description, never an authorisation: ``robot_type`` and
    ``world`` arrive over the wire from the peer itself and
    :meth:`~strands_robots.mesh.core.Mesh._on_presence` authenticates neither,
    so a peer can claim to be a sim. That is fit to tell an operator what the
    peer says about itself, and unfit to stand in for the operator.

    Args:
        peer: One entry of the :func:`get_peers` snapshot, or ``None`` when the
            peer is not on it.

    Returns:
        ``(physical, why)`` - the verdict, and the reason it was reached, phrased
        to read after the peer's name ("it reports real hardware (...)").
    """
    if not peer:
        return True, "it is not on the fleet snapshot, so it cannot be shown to be a sim"

    hw = peer.get("hw")
    if isinstance(hw, str) and hw.strip():
        return True, f"it reports real hardware ({hw.strip()})"

    declared = str(peer.get("robot_type") or peer.get("type") or "").strip().lower()
    if declared in _SIM_PEER_TYPES:
        return False, f"it reports itself as {declared}"

    sim_robots = peer.get("sim_robots")
    if peer.get("world") is True or (isinstance(sim_robots, (list, tuple)) and len(sim_robots) > 0):
        return False, "it reports a simulation world"

    return True, "it did not report itself as a simulation"


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
        # peer (smallest last_seen_mono) to make room. Updates to EXISTING peers
        # never trigger eviction (they don't grow the dict). This bounds the
        # phantom-peer flood DoS while still admitting genuine new robots.
        if is_new:
            cap = _max_peers()
            while len(_PEERS) >= cap and _PEERS:
                oldest_id = min(_PEERS, key=lambda pid: _PEERS[pid].last_seen_mono)
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
            last_seen_mono=time.monotonic(),
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
    now = time.monotonic()
    pruned: list[str] = []
    with _PEERS_LOCK:
        stale = [pid for pid, p in _PEERS.items() if now - p.last_seen_mono > timeout]
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
                "examples/mesh/mesh_acl_example.json5."
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


def _report_zenoh_missing() -> None:
    """Report an absent ``eclipse-zenoh`` at WARNING, once per process.

    Every other way a session open can end with no session -- a refused
    endpoint, a failed open -- is reported at WARNING, and even a bad
    ``STRANDS_MESH_PORT`` that still yields a working mesh warns.  A missing
    dependency is the most total of those outcomes: nothing publishes, nothing
    is discovered, and ``Mesh.alive`` stays ``False`` for the whole process.
    Reporting it below the level a default-configured consumer sees left the
    first observable symptom to whatever downstream wait expired first.

    Warning rather than raising keeps the documented contract: a robot that
    does not need the mesh still runs, and the mesh stays off without taking
    the host process with it.

    Callers must hold ``_SESSION_LOCK``; the once-guard is read and mutated
    without further synchronisation.
    """
    if _zenoh_missing_warned:
        return
    _zenoh_missing_warned.add("warned")
    logger.warning(
        "eclipse-zenoh is not installed, so the mesh stays off: this process "
        "publishes no presence and discovers no peers, and Mesh.alive is "
        "False. Install it with: pip install 'strands-robots[mesh]'"
    )


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
    # STRANDS_MESH=false is documented as a hard kill switch: no Zenoh session
    # and no presence on the fleet. Every path that can OPEN one has to answer
    # it, and this is the only path that opens one -- the two Mesh constructors
    # (init_mesh, robot_mesh._gateway_mesh) each asked separately, so a caller
    # that acquires the transport directly, as ZenohTransport and the bridge
    # factory do, reached zenoh.open with the switch engaged. What arrived was
    # not a quiet extra peer: with no explicit endpoints this path LISTENS, so
    # the process the operator disabled the mesh on became the machine's hub,
    # and every later process on the box connected to it as a client.
    #
    # Asked before the backend branch because the switch is about presence, not
    # about Zenoh: an IoT/bridge transport publishes this robot to the fleet
    # just as a Zenoh session does, and one gate covers every backend.
    #
    # Imported inside the function, not at module scope: core imports this
    # module (twice) while mesh/__init__ loads it, so a top-level import would
    # be a genuine cycle. Reaching the mesh package lazily from inside the
    # function that needs it is the technique _mesh_switch's docstring already
    # describes for the same reason -- strands_robots.robot does it so that
    # importing Robot does not eagerly pull in the Zenoh-backed session. The
    # constant above comes from _mesh_switch instead, which imports nothing
    # from the package and so is reachable at module scope.
    from strands_robots.mesh.core import mesh_disabled_by_env

    if mesh_disabled_by_env():
        logger.debug(
            "Mesh transport not acquired: STRANDS_MESH=%r is a hard kill switch",
            os.getenv(MESH_ENV_VAR, ""),
        )
        return None

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
            _report_zenoh_missing()
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
            # and _load_acl_file.
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
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (listener on %s)", local_ep)
                return _SESSION
            except zenoh_error_types() as exc:
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

            # Not the hub: connect to it, in the mode that actually receives
            # relayed traffic. ``_apply_fallback_topology`` documents the
            # measurement - a peer-mode child hears NOTHING a sibling child
            # publishes, which is why teleop's receiver sat at zero frames
            # while its leader published hundreds.
            # Build cfg OUTSIDE the try so a config-shape ValueError
            # (NaN env clamp, missing TLS file, bad ACL) propagates
            # loudly to Mesh.start instead of being silently downgraded
            # to "session unavailable".
            cfg = _build_config()
            fallback_mode = _apply_fallback_topology(cfg, local_ep, scheme)
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info(
                    "Zenoh mesh session opened (%s, retrying -> %s)",
                    fallback_mode,
                    local_ep,
                )
                return _SESSION
            except zenoh_error_types() as exc:
                # Narrow tuple per AGENTS.md > Review Learnings (#86):
                # transport-level failures only; config-shape ValueError
                # propagates to caller so misconfigured mTLS surfaces loudly.
                logger.warning("Zenoh session open failed (%s fallback): %s", fallback_mode, exc)
                return None

        # Explicit endpoints provided via env vars.
        # Build cfg outside the try (same loud-on-misconfig discipline
        # as the auto-listener path).
        cfg = _build_config()
        try:
            _SESSION = zenoh.open(cfg)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened")
            return _SESSION
        except zenoh_error_types() as exc:
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
    # The kill switch, for the same reason as ``get_session`` upstairs: this
    # door exists to skip the transport factory, not to skip the switch, and
    # it reaches the same ``zenoh.open``.
    from strands_robots.mesh.core import mesh_disabled_by_env

    if mesh_disabled_by_env():
        logger.debug(
            "Zenoh session not opened: STRANDS_MESH=%r is a hard kill switch",
            os.getenv(MESH_ENV_VAR, ""),
        )
        return None

    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import zenoh
        except ImportError:
            _report_zenoh_missing()
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
            # and _load_acl_file.
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
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info("Zenoh mesh session opened (listener on %s)", local_ep)
                return _SESSION
            except zenoh_error_types() as exc:
                # Narrow tuple mirroring the narrowing applied in
                # get_session() upstairs. Config-shape
                # ValueError now propagates instead of being swallowed at DEBUG.
                logger.debug(
                    "Zenoh listener on %s unavailable (%s) - trying client mode",
                    local_ep,
                    exc,
                )

            # Same topology as ``get_session`` upstairs, through the one
            # helper. Before this, the two fallbacks disagreed: this site
            # opened a client with no ``connect/retry`` and no
            # ``exit_on_failure=false``, so when the hub process died the
            # bridge-transport peer went permanently dark - no reconnect
            # loop, no surfaced error. Both now retry the hub endpoint in
            # the background (measured for client mode: delivery resumed
            # after a 6s hub outage with no process restart).
            cfg = _build_config()
            fallback_mode = _apply_fallback_topology(cfg, local_ep, scheme)
            try:
                _SESSION = zenoh.open(cfg)
                _SESSION_REFS = 1
                logger.info(
                    "Zenoh mesh session opened (%s, retrying -> %s)",
                    fallback_mode,
                    local_ep,
                )
                return _SESSION
            except zenoh_error_types() as exc:
                logger.warning("Zenoh session open failed (%s fallback): %s", fallback_mode, exc)
                return None

        cfg = _build_config()
        try:
            _SESSION = zenoh.open(cfg)
            _SESSION_REFS = 1
            logger.info("Zenoh mesh session opened")
            return _SESSION
        except zenoh_error_types() as exc:
            logger.warning("Zenoh session open failed: %s", exc)
            return None


def release_session() -> None:
    """Release one reference to the shared mesh session.

    Delegates to the transport factory when the active backend is
    ``iot`` / ``bridge``; otherwise falls back to the legacy Zenoh refcount.

    On the Zenoh path the final release closes the session. That close is
    fail-soft over the surface :func:`zenoh_error_types` documents - a broker
    drop or socket teardown race is logged at WARNING and the reference is
    still dropped, because nothing can retry a close once the only handle to
    the session is gone. A failure outside that surface (a ``TypeError`` or
    ``AttributeError``, i.e. a bug rather than a transport fault) propagates,
    matching how :meth:`strands_robots.mesh.core.Mesh.stop` treats its
    ``undeclare`` calls. The "session closed" INFO line is emitted only when
    the close actually completed.
    """
    if _is_transport_backend():
        from strands_robots.mesh.transport.factory import release_transport

        release_transport()
        return

    _release_zenoh_session_directly()


def _release_zenoh_session_directly() -> None:
    """Release one reference to the raw Zenoh session, bypassing backend routing.

    The teardown companion to :func:`_get_zenoh_session_directly`, and the
    function :func:`release_session` itself calls once the backend branch is
    not taken - so the legacy refcount and its close contract (documented on
    :func:`release_session`) live in exactly one place.

    :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport` must
    release through here rather than :func:`release_session` for the same
    reason it acquires through :func:`_get_zenoh_session_directly`: under
    ``STRANDS_MESH_BACKEND=bridge`` the backend-aware :func:`release_session`
    delegates to the transport factory, whose last release closes the
    :class:`~strands_robots.mesh.transport.bridge_transport.BridgeTransport`
    that owns that very ``ZenohTransport``, re-entering the factory's
    non-reentrant lock from the thread already holding it.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603

    with _SESSION_LOCK:
        if _SESSION_REFS <= 0:
            return
        _SESSION_REFS -= 1
        if _SESSION_REFS <= 0 and _SESSION is not None:
            closed = True
            try:
                _SESSION.close()
            except zenoh_error_types() as exc:
                # Narrow surface per :func:`zenoh_error_types`, whose docstring
                # names ``close`` among the operations it covers and excludes
                # programmer errors so they surface loudly instead of being
                # swallowed by a best-effort teardown. The session reference is
                # dropped below either way, so nothing can retry the close and
                # this record is the only evidence it did not complete -
                # WARNING (not the DEBUG a per-entity cleanup uses) because the
                # INFO line below otherwise reports a clean close, and a record
                # must be at least as loud as the claim it contradicts.
                closed = False
                logger.warning("Zenoh mesh session close failed: %s", exc)
            _SESSION = None
            _SESSION_REFS = 0
            if closed:
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

    _put_zenoh_directly(key, data)


def _report_unencodable_payload(transport: str, key: str, exc: BaseException) -> None:
    """Report a payload that can never reach *key*, at ERROR, once per topic.

    Every transport's ``put`` is fire-and-forget and
    :meth:`strands_robots.mesh.transport.base.MeshTransport.put`
    scopes that tolerance to a TRANSIENT failure - a closed session, a dropped broker, a
    socket-level write - which the next tick retries. A payload the JSON encoder
    refuses is not transient: it fails identically forever, so the message never
    goes out at all and no retry can change that.

    Reporting it at DEBUG left the two halves of one call disagreeing.
    :meth:`strands_robots.mesh.sensors.SensorLoopsMixin.publish_safety_event`
    writes the event to the wire AND to the local audit log; on an unencodable
    payload the audit half records a ``sig="SERIALISE_FAILED"`` poison record and
    logs at ERROR (naming the peer and the reason), while the wire half dropped
    the event silently. A default-configured operator therefore saw a forensic
    trail asserting that a safety event had been raised, with no peer having
    received it and nothing above DEBUG to say so.

    Once per topic, because a payload builder is per-topic: the same builder runs
    on every tick of ITS key, so a 50 Hz publisher would otherwise emit one
    report per tick, while a different key says nothing about it. Per TOPIC and
    not per (topic, transport): under the bridge backend both legs encode the
    same payload with the same encoder, so the second leg's report would restate
    one fact. The leg that noticed is named in the line rather than keyed on,
    which is why the wording does not claim the other leg is unaffected.

    Args:
        transport: Which leg could not encode, for the log line (``"Zenoh"`` /
            ``"MQTT"``). The wording is shared so a reader grepping the log for
            one transport's report finds the other's.
        key: The topic the payload was addressed to; also the once-guard key.
        exc: What the encoder raised, quoted as the reason.
    """
    if key in _unencodable_topics_warned:
        return
    _unencodable_topics_warned.add(key)
    logger.error(
        "put on %s: the payload cannot be JSON-encoded, so this message and every "
        "later one built the same way is dropped before it reaches the wire "
        "(noticed on the %s leg; reported once for this topic): %s",
        key,
        transport,
        exc,
    )


def _put_zenoh_directly(key: str, data: dict[str, Any]) -> None:
    """Publish to the raw Zenoh session, bypassing transport-backend routing.

    The publish companion to :func:`_get_zenoh_session_directly`, and the
    function :func:`put` itself calls once the backend branch is not taken - so
    the raw JSON encode-and-publish lives in exactly one place.

    :class:`~strands_robots.mesh.transport.zenoh_transport.ZenohTransport` must
    publish through here rather than :func:`put`: under
    ``STRANDS_MESH_BACKEND=bridge`` the backend-aware :func:`put` resolves the
    active transport, which is the
    :class:`~strands_robots.mesh.transport.bridge_transport.BridgeTransport`
    that owns that very ``ZenohTransport`` - so the payload would route back
    into the bridge instead of onto the wire.

    Fire-and-forget, and it reads ``_SESSION`` without taking
    ``_SESSION_LOCK`` exactly as :func:`put` always has: a 50 Hz teleop loop
    must not serialise on the session lock.

    The JSON encode is attempted OUTSIDE the handler that absorbs a wire
    failure, because the two outcomes are not the same kind of failure. A wire
    failure is transient and the next tick retries it, so it stays at DEBUG. A
    payload the encoder refuses can never be published, so it is reported
    through :func:`_report_unencodable_payload` instead of being absorbed by the
    same DEBUG line. Neither raises: the ``put`` contract is fire-and-forget
    either way. The encode catch is deliberately wide because an arbitrary
    payload object can carry an arbitrary ``__reduce__`` / ``keys`` / ``default``
    hook, so the encoder's raise set is not enumerable here (``json.dumps``
    itself raises ``TypeError`` for an unsupported type and ``ValueError`` for a
    circular reference).
    """
    if _SESSION is None:
        return
    try:
        encoded = json.dumps(data).encode()
    except Exception as exc:  # noqa: BLE001 - see the encode note in the docstring
        _report_unencodable_payload("Zenoh", key, exc)
        return
    try:
        _SESSION.put(key, encoded)
    except Exception as exc:
        logger.debug("Zenoh put error on %s: %s", key, exc)


# Process cleanup


def _atexit_cleanup() -> None:
    """Best-effort session teardown on process exit.

    Same close contract as :func:`release_session`, logged at DEBUG: this path
    reports nothing on success, so there is no claim for the record to
    contradict. A programmer error still propagates rather than being swallowed
    at interpreter shutdown.
    """
    global _SESSION, _SESSION_REFS  # noqa: PLW0603
    with _SESSION_LOCK:
        if _SESSION is not None:
            try:
                _SESSION.close()
            except zenoh_error_types() as exc:
                # Same narrow surface as :func:`release_session`. DEBUG here
                # because this path makes no success claim to contradict and
                # runs during interpreter shutdown, matching the level
                # ``BridgeTransport.close`` uses for its per-backend close.
                logger.debug("Zenoh mesh session close failed at exit: %s", exc)
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
