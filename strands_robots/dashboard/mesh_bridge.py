"""Zenoh mesh <-> asyncio bridge for the dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, cast

from strands_robots.dashboard import safety_state
from strands_robots.mesh._zenoh_config import cmd_bytes_cap as _cmd_bytes_cap
from strands_robots.utils import finite_number_error

logger = logging.getLogger(__name__)


def _env_float(name: str, default: str) -> float:
    """Read an operator-tunable float out of the environment.

    Every knob below is resolved once, at import, from a string an operator
    typed, and ``float()`` accepts far more than a knob can honor: it raises on
    a typo, and it returns ``nan`` for ``"nan"`` and ``inf`` for both ``"inf"``
    and an overflowing ``"1e999"``. A bare coercion therefore has two failure
    modes, and neither is reported. A typo raises ``ValueError`` while the
    module body is executing, so the dashboard's mesh bridge does not fail to
    read one knob - it fails to import, with a traceback that names ``float``
    rather than the variable. A non-finite value is worse for being accepted:
    it reaches a comparison as one side of a bound and removes the bound
    instead of widening it. ``PEER_TTL_S`` is compared as ``age > ttl``, which
    is ``False`` for every age against ``nan`` and against ``inf``, so a robot
    that left the fleet is never aged out of the snapshot; each
    :data:`COALESCE_HZ` rate becomes a period through ``1.0 / hz``, and no
    elapsed span is under ``nan`` or under the ``0.0`` an ``inf`` rate yields,
    so the repeat ceiling that rate names never applies.

    Finiteness is decided by the shared numeric domain
    (:func:`~strands_robots.utils.finite_number_error`) rather than left to a
    comparison, which cannot express it: ``inf > 0`` is ``True``, so a
    positivity floor admits ``inf``, ``Infinity`` and ``1e999`` as resolved
    knobs and refuses ``nan`` only incidentally, because ``nan > 0`` is
    ``False``. This is the rule
    :func:`~strands_robots.simulation.isaac.simulation._env_float` and
    :func:`~strands_robots.mesh.core._parse_positive_float_env` already state
    and apply for the same kind of operator input, and every rejection is
    reported for the reason those two report theirs: substituting the default
    in silence leaves the operator's model of the knob wrong with nothing to
    correct it against.

    No floor is applied, deliberately, and for the reason
    :func:`~strands_robots.mesh.core._parse_positive_float_env` gives: what a
    zero means differs per knob, so it belongs to the consumer rather than to
    this domain. Both consumers here already decide it - ``prune_peers`` treats
    a non-positive ``ttl`` as "never prune" and
    :meth:`EventCoalescer.allow` treats a non-positive rate as "no ceiling".

    Args:
        name: Environment variable to read. Unset, blank or unusable falls back.
        default: Value applied when the variable names no usable number, held as
            the string it would have been read from so the log names what the
            fallback was.

    Returns:
        The override when it is a usable finite number, else ``default``.
    """
    raw = os.getenv(name, default)
    if raw.strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %r", name, raw, default)
        return float(default)
    reason = finite_number_error(value, name, "dashboard mesh bridge")
    if reason is not None:
        logger.warning("%s; using %r", reason, default)
        return float(default)
    return value


def _resolve_mesh_camera_hz() -> float:
    """Resolve the camera publish rate ``STRANDS_MESH_CAMERA_HZ`` names.

    This is not the dashboard's knob. ``STRANDS_MESH_CAMERA_HZ`` is the mesh's,
    read by :meth:`~strands_robots.mesh.core.Mesh._resolve_camera_hz` to
    decide whether the camera loop runs at all, and the dashboard is the surface
    that *writes* it: the settings panel holds ``camera_hz``, and
    :func:`~strands_robots.dashboard.settings.apply_mesh_env` pushes it into the
    environment the mesh peers then read. Reporting it back through a second,
    looser reading closed that loop onto a value nothing publishes at - an
    operator typing a rate the mesh refuses saw it echoed as the live one, on
    the same panel they had typed it into.

    So the question is asked of the mesh's own owner,
    :func:`~strands_robots.mesh.session.hz_from_env`, and the fallback mirrors
    the publisher's documented one: unset takes the
    :data:`~strands_robots.mesh.session.CAMERA_HZ` default, and non-positive or
    unusable is ``0.0``, camera publishing off. Frames are large, so an
    unusable override disables the loop rather than substituting a rate the
    operator did not ask for, and this surface has to say the same. The two
    answers are held equal across every spelling an operator can type, which is
    the invariant to keep: whatever this returns has to be what the camera loop
    resolves, so it is asked of the loop's own resolver rather than re-derived.

    Returns:
        The rate the mesh camera loop resolves for this host, in Hz, with
        ``0.0`` meaning camera publishing is off.
    """
    from strands_robots.mesh.session import CAMERA_HZ, hz_from_env

    hz, reason = hz_from_env("STRANDS_MESH_CAMERA_HZ")
    if reason is not None:
        logger.warning("%s; camera publishing off", reason)
        return 0.0
    if hz is None:
        hz = CAMERA_HZ
    return hz if hz > 0 else 0.0


PEER_STALE_S = 15.0  # presence heartbeat timeout before a card greys out

# : How long a peer may stay quiet before it is dropped from the fleet snapshot : entirely.
PEER_TTL_S = _env_float("STRANDS_DASHBOARD_PEER_TTL_S", "300")


def prune_peers(
    peers: dict[str, dict[str, Any]],
    now: float,
    ttl: float = PEER_TTL_S,
    protected_ids: set[str] | frozenset[str] | None = None,
    stale_after: float = PEER_STALE_S,
) -> dict[str, dict[str, Any]]:
    """Flag quiet peers stale and drop the ones that aged past ``ttl``."""
    protected = protected_ids or frozenset()
    out: dict[str, dict[str, Any]] = {}
    for pid, entry in peers.items():
        age = now - entry.get("last_seen", 0)
        # Child sim peers are named "<parent>__<robot>"; they live and die with
        # the parent's process, so the parent's protection covers them.
        parent = pid.partition("__")[0]
        if ttl > 0 and age > ttl and pid not in protected and parent not in protected:
            continue
        out[pid] = {**entry, "stale": age > stale_after}
    return out


# : Transport low-pass filter on ``**/cmd`` (_zenoh_config.DEFAULT_MAX_CMD_BYTES). : Anything
# larger is dropped pre-deserialise and the sender only ever sees a : timeout, so we check
# before publishing and return a real error instead.
# One parser, the SDK's: this was a hand copy of the default (16*1024) that
# would silently diverge the day _zenoh_config changes DEFAULT_MAX_CMD_BYTES -
# the pre-publish check here MUST agree with the transport's drop filter or a
# command passes the check and vanishes into the documented timeout anyway.
# cmd_bytes_cap() also refuses a garbage env value with an actionable message.
MAX_CMD_BYTES = _cmd_bytes_cap()

#: How many fleet actions to keep for the activity panel.
ACTIVITY_CAP = 300


def peer_is_known(
    peer_id: str,
    peers: Mapping[str, Any] | Iterable[str],
    managed_ids: Iterable[str] = (),
) -> bool:
    """Is this peer id something the fleet could plausibly answer for?"""
    if not peer_id:
        return False
    haystack = set(peers) | set(managed_ids)
    if peer_id in haystack:
        return True
    # Both halves must be non-empty, matching route_task_target's own condition exactly: "arm-1__"
    # does NOT get rerouted there, so calling it known here would hand it straight back to the
    # timeout this guard exists to avoid.
    parent, _, child = peer_id.partition("__")
    return bool(parent and child) and parent in haystack


def managed_without_presence(
    peers: Mapping[str, Mapping[str, Any]],
    managed_ids: Iterable[str] = (),
    *,
    spawn_times: Mapping[str, float] | None = None,
    now: float | None = None,
    grace_s: float = 20.0,
) -> list[str]:
    present = set(peers)
    stamps = spawn_times or {}
    clock = time.time() if now is None else now
    out: list[str] = []
    for pid in managed_ids:
        if not pid or pid in present:
            continue
        if any(p.partition("__")[0] == pid for p in present if "__" in p):
            continue
        parent, _, child = pid.partition("__")
        if parent and child and parent in present:
            continue
        started = stamps.get(pid)
        if started is not None and clock - started < grace_s:
            continue
        out.append(pid)
    return sorted(out)


def silent_arms(peers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    ids = list(peers)
    streaming: list[str] = []
    silent: list[str] = []
    hosts = 0
    stale = 0
    for pid in ids:
        rec = peers.get(pid) or {}
        state = rec.get("state") or {}
        joints = state.get("joints") or {}
        if joints:
            streaming.append(pid)
            continue
        if rec.get("stale"):
            stale += 1
            continue
        if any(other != pid and other.startswith(f"{pid}__") for other in ids):
            hosts += 1  # a host process, not an arm: silence is its normal state
            continue
        silent.append(pid)
    if not silent:
        return None
    return {
        "streaming": len(streaming),
        "silent": sorted(silent),
        **({"host_processes": hosts} if hosts else {}),
        **({"stale": stale} if stale else {}),
    }


def peer_origins(
    peer_ids: Mapping[str, Any] | Iterable[str],
    managed_ids: Iterable[str] = (),
) -> dict[str, str]:
    """Label each peer ``"managed"`` (this dashboard spawned it) or ``"external"``."""
    managed = set(managed_ids)

    def origin(pid: str) -> str:
        if pid in managed:
            return "managed"
        parent, _, child = pid.partition("__")
        if parent and child and parent in managed:
            return "managed"
        return "external"

    return {pid: origin(pid) for pid in peer_ids}


def absent_children(
    peers: Mapping[str, Any] | Iterable[str],
    children: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    present = set(peers)
    families = {pid.split("__", 1)[0] for pid in present if "__" in pid}
    out: list[dict[str, Any]] = []
    for child in children or ():
        if not isinstance(child, Mapping):
            continue
        peer_id = str(child.get("peer_id") or "")
        if not peer_id or child.get("alive"):
            continue
        if peer_id in present or peer_id in families:
            continue
        out.append(
            {
                "peer_id": peer_id,
                "robot_name": child.get("robot_name"),
                "mode": child.get("mode"),
                "returncode": child.get("returncode"),
                "started_at": child.get("started_at"),
            }
        )
    out.sort(key=lambda c: c["peer_id"])
    return out


def route_task_target(target: str, cmd: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Route commands aimed at a child sim peer to its parent Simulation peer."""
    if "__" in target and not cmd.get("robot_name"):
        parent, _, robot_name = target.partition("__")
        if parent and robot_name:
            cmd = {**cmd, "robot_name": robot_name}
            target = parent
    return target, cmd


def command_succeeded(response: dict[str, Any] | None) -> bool:
    """Did a peer actually carry the command out?"""
    if not isinstance(response, dict):
        return False
    if response.get("error") or response.get("type") == "error":
        return False
    if response.get("ok") is False:
        return False
    result = response.get("result")
    if isinstance(result, dict):
        if result.get("ok") is False:
            return False
        if str(result.get("status", "")).lower() in ("error", "failed"):
            return False
        # An ``error`` INSIDE the payload is the peer's own refusal, and it arrives with no ``ok`` key
        # and no ``status`` at all -- a plain ``{"error": "gripper jammed"}`` from a tool.
        if result.get("error"):
            return False
        # A peer that wraps a tool result nests it once more. Both depths are real -- the dashboard's
        # own reader has always looked at ``result.error ?? result.result.error``.
        inner = result.get("result")
        if isinstance(inner, dict) and inner.get("error"):
            return False
    return True


def _raw_to_jpeg(raw: bytes, shape: Any) -> tuple[bytes | None, str | None]:
    """Transcode raw pixel bytes to JPEG. Returns ``(jpeg, error)``. Only called for frames whose
    ``encoding`` is not JPEG.
    """
    if not (isinstance(shape, (list, tuple)) and len(shape) in (2, 3)):
        return None, f"encoding is not jpeg and shape {shape!r} is unusable"
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None, "raw frame received but numpy/Pillow are not installed to transcode it"
    try:
        dims = tuple(int(d) for d in shape)
        expected = 1
        for d in dims:
            expected *= d
        if len(raw) != expected:
            return None, f"raw frame is {len(raw)} B but shape {dims} needs {expected} B"
        array = np.frombuffer(raw, dtype=np.uint8).reshape(dims)
        if array.ndim == 3 and array.shape[2] == 4:
            image = Image.fromarray(array, "RGBA").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 3:
            image = Image.fromarray(array, "RGB")
        elif array.ndim == 2:
            image = Image.fromarray(array, "L")
        else:
            return None, f"unsupported raw frame shape {dims}"
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue(), None
    except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the sub
        return None, f"raw frame transcode failed: {type(exc).__name__}: {exc}"


def stop_outcome(response: dict[str, Any] | None) -> dict[str, Any]:
    """Classify one peer's answer to a stop into the three honest states. ``stopped`` / ``not_stopped``
    (the peer answered but could not stop) / ``no_answer`` (timeout).
    """
    if not isinstance(response, dict):
        return {"state": "no_answer", "detail": "no response"}
    error = response.get("error")
    if error and "timeout" in str(error).lower():
        return {"state": "no_answer", "detail": str(error)}
    if command_succeeded(response):
        return {"state": "stopped", "detail": ""}
    result = response.get("result")
    detail = ""
    if isinstance(result, dict):
        detail = str(result.get("error") or result.get("status") or "")
    return {"state": "not_stopped", "detail": detail or str(error or "refused")}


# : Per-type ceiling on UNCHANGED repeats, in Hz.
COALESCE_HZ: dict[str, float] = {
    "presence": _env_float("STRANDS_DASHBOARD_PRESENCE_HZ", "1.0"),
    "camera_meta": _env_float("STRANDS_DASHBOARD_CAMERA_META_HZ", "2.0"),
    # The SensorLoops rates these mirror are pose/imu/odom 10 Hz each and lidar
    # summary 5 Hz, so one rover publishing all of them is ~36 events/s -- the
    # order of the 34.7 Hz for a single arm this coalescer was built for. A
    # CHANGED reading is never delayed, so a moving robot still streams; the
    # rate is the floor for a repeat that says nothing new.
    "pose": _env_float("STRANDS_DASHBOARD_POSE_HZ", "1.0"),
    "imu": _env_float("STRANDS_DASHBOARD_IMU_HZ", "1.0"),
    "odom": _env_float("STRANDS_DASHBOARD_ODOM_HZ", "1.0"),
    "lidar": _env_float("STRANDS_DASHBOARD_LIDAR_HZ", "1.0"),
    # Health already publishes at 0.5 Hz upstream; the floor matches it rather
    # than inventing a slower one.
    "health": _env_float("STRANDS_DASHBOARD_HEALTH_HZ", "0.5"),
}

#: Fields that tick on their own and therefore say nothing about whether the
#: PAYLOAD changed. Compared-out so a per-frame timestamp cannot defeat coalescing.
_VOLATILE_FIELDS = frozenset(
    {
        "t",
        "ts",
        "time",
        "timestamp",
        "last_seen",
        "seq",
        "frame",
        "frames",
        "frame_id",
        "count",
        "uptime",
        "uptime_s",
        "elapsed",
        "fps",
    }
)


def _stable_content(data: Any) -> str:
    """A comparable rendering of an event payload, minus self-ticking fields."""

    def strip(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: strip(x) for k, x in sorted(v.items()) if k not in _VOLATILE_FIELDS}
        if isinstance(v, (list, tuple)):
            return [strip(x) for x in v]
        return v

    try:
        return json.dumps(strip(data), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001 - never let bookkeeping drop an event
        return repr(data)[:2000]


class EventCoalescer:
    """Decides whether an event is worth another JSON serialization."""

    def __init__(self, rates: dict | None = None) -> None:
        self.rates = dict(COALESCE_HZ if rates is None else rates)
        self._last: dict[tuple, tuple] = {}
        self.suppressed = 0
        self.forwarded = 0

    def key(self, event: dict) -> tuple:
        # `kind` separates two streams that share a type: lidar publishes both a
        # summary and a state document, and without it their alternating
        # payloads read as a change every tick and coalesce to nothing. A strict
        # no-op for the types that predate it -- `allow` returns before ever
        # calling this when a type carries no rate, and neither rated type
        # (presence, camera_meta) sets `kind`.
        return (event.get("type"), event.get("peer_id"), event.get("cam"), event.get("kind"))

    def allow(self, event: dict, now: float) -> bool:
        """Has enough time passed to forward an UNCHANGED repeat of *event*?

        Args:
            event: The event to decide about.
            now: Reading of a MONOTONIC clock. What is compared is ``now`` minus
                the previous reading against ``1.0 / hz``, an elapsed span, so a
                clock that can be stepped moves the decision by the size of the
                step rather than by the time that passed.

        Returns:
            ``True`` when the event is new, changed, or carries no rate, and when
            its rate is non-positive - which every consumer reads as "no ceiling".
        """
        hz = self.rates.get(event.get("type") or "")
        if not hz or hz <= 0:
            self.forwarded += 1
            return True
        k = self.key(event)
        content = _stable_content(event.get("data"))
        prev = self._last.get(k)
        if prev is not None and prev[0] == content and (now - prev[1]) < (1.0 / hz):
            self.suppressed += 1
            return False
        self._last[k] = (content, now)
        self.forwarded += 1
        return True

    def forget(self, peer_id: str) -> None:
        """Drop bookkeeping for a peer that left, so a respawn starts clean."""
        for k in [k for k in self._last if k[1] == peer_id]:
            self._last.pop(k, None)

    def stats(self) -> dict:
        total = self.forwarded + self.suppressed
        return {
            "forwarded": self.forwarded,
            "suppressed": self.suppressed,
            "suppressed_pct": round(100 * self.suppressed / total, 1) if total else 0.0,
            "rates_hz": dict(self.rates),
        }


class MeshBridge:
    """Dashboard-side mesh peer. One instance per server process."""

    def __init__(self, peer_id: str | None = None) -> None:
        self.peer_id = peer_id or f"dashboard-{socket.gethostname().split('.')[0]}-{uuid.uuid4().hex[:4]}"
        self._session: Any | None = None
        self._subs: list[Any] = []
        self._running = False

        # Fleet snapshot: peer_id -> {presence, state, stream, last_seen, cameras:{name:{t,shape}}}
        self.peers: dict[str, dict[str, Any]] = {}
        self._peers_lock = threading.Lock()

        # Latest camera frames: (peer_id, cam) -> {"t": float, "jpeg": bytes, "shape": [...]}
        self.frames: dict[tuple[str, str], dict[str, Any]] = {}
        self._frames_lock = threading.Lock()

        # Async fan-out. Subscribers get JSON-able event dicts.
        self._queues: set[asyncio.Queue] = set()
        self._queues_lock = threading.Lock()
        self._coalescer = EventCoalescer()
        self._coalesce_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._lockout = safety_state.Lockout()
        self._lockout_proof: dict[str, float] = {}
        # Signed safety rail (lazy - see _safety_mesh)
        self._safety: Any | None = None
        self._safety_lock = threading.Lock()

        # RPC correlation (mirrors Mesh.send)
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        # turn_id -> the ONE peer allowed to answer it (mirrors
        # Mesh._expected_responders): without this, any ACL-authorised peer
        # observing a turn_id could answer someone else's pending turn.
        self._expected_responders: dict[str, str] = {}
        self._rpc_lock = threading.Lock()

        # Fleet activity: every command this dashboard issued + safety
        # envelopes seen on the wire. Cheap forensics for "who moved that arm?"
        self.activity: deque[dict[str, Any]] = deque(maxlen=ACTIVITY_CAP)
        self._activity_lock = threading.Lock()

        # Resolved endpoints of the live session, for /api/mesh/config.
        self._endpoints: dict[str, Any] = {}

        # Set by the server to a callable returning peer ids that must never be
        # aged out (LIVE managed local processes). Optional by design: the
        # bridge stays usable standalone.
        self.protected_peer_ids: Any | None = None

        self.peer_annotations: Any | None = None
        self.managed_children: Any | None = None

    def _peer_annotations(self) -> dict[str, dict[str, Any]]:
        if self.peer_annotations is None:
            return {}
        try:
            data = self.peer_annotations()
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # never let a bad hook break the snapshot
            logger.warning("[mesh] peer annotation lookup failed (%r)", exc)
            return {}

    def _managed_children(self) -> list[Any]:
        # getattr, not self.managed_children: snapshot() must keep working on a bridge that predates
        # this hook.
        hook = getattr(self, "managed_children", None)
        if hook is None:
            return []
        try:
            data = hook()
            return list(data) if data else []
        except Exception as exc:  # a memorial may never break the fleet snapshot
            logger.warning("[mesh] managed children lookup failed (%r)", exc)
            return []

    def _protected_peer_ids(self) -> frozenset[str]:
        if self.protected_peer_ids is None:
            return frozenset()
        try:
            return frozenset(self.protected_peer_ids())
        except Exception as exc:  # never let a bad hook break the snapshot
            logger.warning("[mesh] protected peer lookup failed (%r)", exc)
            return frozenset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Join the mesh. Returns False when zenoh is unavailable."""
        from strands_robots.dashboard import settings
        from strands_robots.mesh.session import get_session

        # ZENOH_CONNECT / ZENOH_LISTEN / STRANDS_MESH_PORT are read *inside*
        # get_session(), so remote-endpoint settings have to be in the
        # environment before this call - not after.
        settings.apply_mesh_env()

        self._loop = loop
        from strands_robots.mesh.core import mesh_disabled_by_env

        if mesh_disabled_by_env():
            logger.warning(
                "STRANDS_MESH=false - not joining the mesh; the dashboard serves "
                "settings, devices and cameras but shows no peers",
            )
            return False
        session = get_session()
        if session is None:
            logger.warning("Mesh session unavailable (is eclipse-zenoh installed?) - dashboard runs offline")
            return False
        self._session = session
        self._running = True

        sub = session.declare_subscriber
        self._subs = [
            sub("strands/*/presence", self._on_presence),
            sub("strands/*/state", self._on_state),
            sub("strands/*/stream", self._on_stream),
            sub("strands/*/camera/**", self._on_camera),
            # SensorLoops publishes these and nothing here consumed them, so a
            # rover or a humanoid rendered as a name and a camera. Same
            # raw-zenoh shape as state: one subscriber per topic, the payload
            # forwarded as the SDK wrote it.
            sub("strands/*/pose", self._on_pose),
            sub("strands/*/health", self._on_health),
            sub("strands/*/imu", self._on_imu),
            sub("strands/*/odom", self._on_odom),
            sub("strands/*/lidar/**", self._on_lidar),
            sub("strands/safety/estop", self._on_safety),
            sub("strands/safety/resume", self._on_safety),
            sub(f"strands/{self.peer_id}/response/**", self._on_response),
        ]
        self._endpoints = self._read_endpoints()
        logger.info("MeshBridge online as %s", self.peer_id)
        return True

    def _read_endpoints(self) -> dict[str, Any]:
        """What the live session is actually talking to."""
        from strands_robots.dashboard import settings

        info: dict[str, Any] = {
            "connect": settings.as_list(os.getenv("ZENOH_CONNECT")),
            "listen": settings.as_list(os.getenv("ZENOH_LISTEN")),
            "port": os.getenv("STRANDS_MESH_PORT", "7447"),
            "backend": os.getenv("STRANDS_MESH_BACKEND", "zenoh"),
        }
        try:
            from strands_robots.mesh._zenoh_config import resolve_auth_mode

            info["auth_mode"] = resolve_auth_mode()
        except Exception:  # noqa: BLE001 - introspection only
            info["auth_mode"] = "unknown"
        return info

    def mesh_info(self) -> dict[str, Any]:
        """Mesh posture for /api/mesh/config and the settings panel."""
        from strands_robots.dashboard import settings

        local_dev = os.getenv("STRANDS_MESH_LOCAL_DEV", "") not in ("", "0", "false")
        info = {
            **self._endpoints,
            "online": self._running,
            "peer_id": self.peer_id,
            "peers": len(self.peers),
            "live_peers": len(self.live_peers()),
            "local_dev": local_dev,
            # STRANDS_MESH_LOCAL_DEV=1 runs the entire mesh with auth "none";
            # _build_config logs "WIRE SECURITY DISABLED". Surface it so a lab
            # posture can't be mistaken for a secured one.
            "wire_security": "DISABLED (local dev)" if local_dev else self._endpoints.get("auth_mode"),
            "camera_hz": _resolve_mesh_camera_hz(),
            "settings": settings.load()["mesh"],
            "multicast": os.getenv("STRANDS_MESH_MULTICAST", ""),
            "max_cmd_bytes": MAX_CMD_BYTES,
        }
        try:
            from strands_robots.mesh.security import _policy_type_allowlist

            info["policy_allow"] = sorted(_policy_type_allowlist())
        except Exception:  # noqa: BLE001
            info["policy_allow"] = []
        return info

    def restart(self) -> bool:
        """Re-open the mesh session against the current settings."""
        loop = self._loop
        if loop is None:
            return False
        self.stop()
        with self._peers_lock:
            self.peers.clear()
        with self._frames_lock:
            self.frames.clear()
        ok = self.start(loop)
        self.record_activity("mesh", "restart", detail=self._endpoints, ok=ok)
        self._emit({"type": "mesh_reconfigured", "ok": ok, "mesh": self.mesh_info()})
        return ok

    def stop(self) -> None:
        self._running = False
        if self._safety is not None:
            try:
                self._safety.stop()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self._safety = None
        for s in self._subs:
            with contextlib.suppress(Exception):
                s.undeclare()
        self._subs.clear()
        if self._session is not None:
            from strands_robots.mesh.session import release_session

            with contextlib.suppress(Exception):
                release_session()
            self._session = None

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    def attach_queue(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._queues_lock:
            self._queues.add(q)
        return q

    def detach_queue(self, q: asyncio.Queue) -> None:
        with self._queues_lock:
            self._queues.discard(q)

    def coalesce_stats(self) -> dict[str, Any]:
        """Forwarded vs suppressed /ws/mesh events since process start."""
        with self._coalesce_lock:
            return self._coalescer.stats()

    def _emit(self, event: dict[str, Any]) -> None:
        """Push an event to all consumer queues (thread -> loop safe)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # Coalesce BEFORE fan-out: one decision serves every client, and the
        # serialization it avoids is per-client. On the monotonic clock, because
        # what the coalescer compares is an elapsed span against a period: an NTP
        # step moves a wall-clock span by the size of the step, which either
        # suppresses a reading that was due or forwards one the rate had capped.
        with self._coalesce_lock:
            if not self._coalescer.allow(event, time.monotonic()):
                return
        with self._queues_lock:
            queues = list(self._queues)
        for q in queues:

            def _put(q: Any = q) -> None:
                if q.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        q.get_nowait()  # drop oldest - dashboards want latest
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(event)

            loop.call_soon_threadsafe(_put)

    # ------------------------------------------------------------------
    # Zenoh callbacks (zenoh worker threads)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(sample: Any) -> dict[str, Any] | None:
        try:
            data = json.loads(sample.payload.to_bytes().decode())
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _touch_peer(self, peer_id: str) -> dict[str, Any]:
        with self._peers_lock:
            entry = self.peers.setdefault(peer_id, {"peer_id": peer_id})
            now = time.time()
            # When this dashboard first saw the peer: a peer that appeared AFTER an
            # e-stop is a process that never received it (safety_state.peer_lockout).
            entry.setdefault("first_seen", now)
            entry["last_seen"] = now
            return entry

    def _on_presence(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("robot_id")
        if not isinstance(peer_id, str) or peer_id == self.peer_id:
            return
        entry = self._touch_peer(peer_id)
        entry["presence"] = data
        self._emit({"type": "presence", "peer_id": peer_id, "data": data})

    def _on_state(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        if not isinstance(peer_id, str):
            return
        entry = self._touch_peer(peer_id)
        entry["state"] = data
        self._emit({"type": "state", "peer_id": peer_id, "data": data})

    def _on_stream(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        if not isinstance(peer_id, str):
            return
        entry = self._touch_peer(peer_id)
        entry["stream"] = data
        self._emit({"type": "stream", "peer_id": peer_id, "data": data})

    def _on_camera(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        cam = data.get("cam")
        encoded = data.get("data")
        if not (isinstance(peer_id, str) and isinstance(cam, str) and isinstance(encoded, str)):
            return
        import base64

        try:
            raw: bytes | None = base64.b64decode(encoded)
        except Exception:
            return
        meta: dict[str, Any] = {
            "t": data.get("t"),
            "shape": data.get("shape"),
            "encoding": data.get("encoding"),
        }
        # A peer may publish raw pixel bytes instead of JPEG.
        if str(meta["encoding"] or "jpeg").lower() not in ("jpeg", "jpg"):
            raw, error = _raw_to_jpeg(cast(bytes, raw), meta.get("shape"))
            meta["converted"] = error is None
            if error:
                meta["error"] = error
        meta["displayable"] = raw is not None
        with self._frames_lock:
            self.frames[(peer_id, cam)] = {"jpeg": raw, **meta}
        entry = self._touch_peer(peer_id)
        cams = entry.setdefault("cameras", {})
        cams[cam] = meta
        # Lightweight notification (no pixels) so the UI knows a frame arrived.
        self._emit({"type": "camera_meta", "peer_id": peer_id, "cam": cam, "data": meta})

    def _sensor_sample(self, sample: Any, slot: str) -> tuple[str | None, dict[str, Any]]:
        """Decode a sensor topic and file its payload under ``slot`` on the peer.

        The four scalar sensor topics differ only in which key they land on, so
        the decode/attribute/store steps live here once. The frame itself is
        built at each call site with a literal type, because the guard asking
        for a frontend reader for every emitted frame type reads those literals
        out of the AST: a type assembled from a variable would leave that guard
        passing while covering nothing.

        Args:
            sample: The raw zenoh sample.
            slot: Key on the peer entry to store the payload under.

        Returns:
            ``(peer_id, data)``, or ``(None, {})`` when the sample carries no
            usable payload or names no peer to attribute it to.
        """
        data = self._decode(sample)
        if not data:
            return None, {}
        peer_id = data.get("peer_id")
        if not isinstance(peer_id, str):
            return None, {}
        entry = self._touch_peer(peer_id)
        entry[slot] = data
        return peer_id, data

    def _on_pose(self, sample: Any) -> None:
        peer_id, data = self._sensor_sample(sample, "pose")
        if peer_id is None:
            return
        self._emit({"type": "pose", "peer_id": peer_id, "data": data})

    def _on_health(self, sample: Any) -> None:
        peer_id, data = self._sensor_sample(sample, "health")
        if peer_id is None:
            return
        self._emit({"type": "health", "peer_id": peer_id, "data": data})

    def _on_imu(self, sample: Any) -> None:
        peer_id, data = self._sensor_sample(sample, "imu")
        if peer_id is None:
            return
        self._emit({"type": "imu", "peer_id": peer_id, "data": data})

    def _on_odom(self, sample: Any) -> None:
        peer_id, data = self._sensor_sample(sample, "odom")
        if peer_id is None:
            return
        self._emit({"type": "odom", "peer_id": peer_id, "data": data})

    def _on_lidar(self, sample: Any) -> None:
        """One type, two documents: ``lidar/summary`` and ``lidar/state``.

        They are kept apart under ``lidar`` rather than filed on one key,
        because a summary landing on top of a state (or the reverse) would show
        an operator a field the other document never carried.
        """
        data = self._decode(sample)
        if not data:
            return
        peer_id = data.get("peer_id")
        if not isinstance(peer_id, str):
            return
        key = str(getattr(sample, "key_expr", ""))
        kind = "state" if key.endswith("/state") else "summary"
        entry = self._touch_peer(peer_id)
        with self._peers_lock:
            lidar = dict(entry.get("lidar") or {})
            lidar[kind] = data
            entry["lidar"] = lidar
        self._emit({"type": "lidar", "kind": kind, "peer_id": peer_id, "data": data})

    def _on_safety(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        key = str(getattr(sample, "key_expr", ""))
        kind = "estop" if key.endswith("estop") else "resume"
        # A five-second flash in the header was the ONLY representation of a lockout in
        # this product, so a reload erased it while two arms stayed locked for ten hours.
        with self._peers_lock:
            self._lockout = safety_state.apply_event(self._lockout, kind=kind, data=data, now=time.time())
            if kind == "estop":
                self._lockout_proof.clear()
        self.record_activity("safety", kind, detail=data, ok=True)
        self._emit({"type": "safety", "kind": kind, "data": data})

    def _on_response(self, sample: Any) -> None:
        data = self._decode(sample)
        if not data:
            return
        turn = data.get("turn_id")
        if not isinstance(turn, str):
            return
        responder = data.get("responder_id")
        with self._rpc_lock:
            evt = self._pending.get(turn)
            if evt is None:
                return
            # Point-to-point scope check (mirrors Mesh._on_response): a
            # response is accepted only from the peer send_cmd addressed.
            # Without it, any ACL-authorised peer observing a turn_id could
            # answer someone else's pending turn and have its result taken
            # for the target's. Legacy peers that omit responder_id are
            # rejected the same way - an absent identity is not a match.
            expected = self._expected_responders.get(turn)
            if expected is not None and responder != expected:
                logger.warning(
                    "[mesh_bridge] response for turn %s rejected: responder %r != expected %r",
                    turn,
                    responder,
                    expected,
                )
                return
            self._responses[turn] = data
            evt.set()

    # ------------------------------------------------------------------ Commands (dashboard ->
    # robot).

    # ------------------------------------------------------------------ Signed safety rail (A6).

    def _safety_mesh(self) -> Any | None:
        """Lazily start the robot-less Mesh used for signed safety envelopes."""
        with self._safety_lock:
            if self._safety is not None and getattr(self._safety, "alive", False):
                return self._safety
            try:
                from strands_robots.mesh.core import Mesh, mesh_disabled_by_env

                # STRANDS_MESH=false is asked HERE and not only in start().
                # This is the second site in this process that can OPEN a
                # session, and mesh_disabled_by_env()'s own docstring states
                # the rule: an operator who set the switch asked for no Zenoh
                # session and no presence on the fleet, "so every path that
                # can open one answers this". #2515 closed the same gap for
                # the robot-less gateway in tools/robot_mesh after a direct
                # Mesh(...) construction put a live gateway-* peer on the
                # fleet past the inline check. Reaching it from the e-stop is
                # the worst case of that: a new peer appears on the fleet at
                # the moment the operator is trying to halt it, and it is the
                # action they are least likely to want to debug afterwards.
                if mesh_disabled_by_env():
                    logger.warning(
                        "STRANDS_MESH=false - signed safety rail not started; the "
                        "per-peer broadcast stop is unaffected and still fanned out",
                    )
                    return None

                m = Mesh(None, peer_id=f"{self.peer_id}-safety", peer_type="gateway")
                m.start()
                if not m.alive:
                    return None
                self._safety = m
                logger.info("signed safety rail online as %s", m.peer_id)
                return m
            except Exception as exc:  # noqa: BLE001 - rail is enrichment over broadcast-stop
                logger.warning("signed safety rail unavailable: %s", exc)
                return None

    def _rail_unavailable(self) -> dict[str, Any]:
        """The one answer for "there is no signed rail", switch-aware.

        "Switched off" and "broken" are different answers and an operator acts
        on them differently: one is the switch they set, the other is a fault to
        chase. Every rail verb answers through here so the two wordings have a
        single owner. A second copy is exactly how the resume half came to
        report a switched-off rail as a fault while the e-stop half reported it
        correctly, 38 lines apart in this file -- and the operator reading
        "unavailable" is sent to the two ``override_code`` causes the
        troubleshooting sheet lists for a refused resume, neither of which is
        the switch they set.
        """
        from strands_robots.mesh.core import mesh_disabled_by_env

        return {
            "signed": False,
            "error": (
                "signed safety rail disabled by STRANDS_MESH=false"
                if mesh_disabled_by_env()
                else "safety mesh unavailable"
            ),
        }

    def signed_estop(self) -> dict[str, Any]:
        """Fleet e-stop over the SIGNED rail."""
        m = self._safety_mesh()
        if m is None:
            return self._rail_unavailable()
        responses = m.emergency_stop()
        # Carried through rather than recomputed: _peers_that_did_not_stop
        # grades the four response shapes that report a stop which did NOT
        # happen, and Mesh.emergency_stop() already puts its verdict in the
        # strands/safety/estop envelope and the audit record. A second copy of
        # that grading on the safety path would be a second thing to keep
        # right. (Its name is private and this is its second caller; promoting
        # it is the follow-up if a third appears.)
        from strands_robots.mesh.core import _peers_that_did_not_stop

        return {
            "signed": True,
            "issuer": m.peer_id,
            "responses": responses,
            # The ISSUER's own latch. emergency_stop() sets it unconditionally
            # before it broadcasts, so True here is a true statement about THIS
            # rail -- "a resume is required to clear it" -- and not about the
            # fleet. The fleet half is the two fields below, and conflating them
            # is what let one value describe a stop that reached everybody and
            # one that reached nobody. It must also stay true whenever a resume
            # is genuinely needed: the operator's resume control is gated on it.
            "lockout_engaged": True,
            # What the peers actually said. responses_received keeps its
            # meaning from Mesh.emergency_stop (replies received, not stops
            # confirmed), so the two numbers can be compared rather than one
            # absorbing the other; peers that never answered are the gap
            # between this and the peer count.
            "responses_received": len(responses),
            "peers_not_stopped": sorted(_peers_that_did_not_stop(responses)),
        }

    def signed_resume(self, override_code: str) -> dict[str, Any]:
        """Clear the fleet lockout with the operator override code."""
        m = self._safety_mesh()
        if m is None:
            return self._rail_unavailable()
        res = m._resume_lockout(override_code)
        return {"signed": True, "issuer": m.peer_id, **(res or {})}

    def send_cmd(
        self,
        target: str,
        cmd: dict[str, Any],
        timeout: float = 30.0,
        *,
        source: str = "api",
    ) -> dict[str, Any]:
        """Send a command to a peer and wait for its response (blocking)."""
        from strands_robots.mesh.session import put

        if not self._running:
            return {"error": "mesh offline", "ok": False}
        # Mesh.send parity, missing from this clone until now (§2.1):
        # target sanity + client-side validate_command. Receiver-side
        # _exec_cmd still validates - this turns "the peer refused it" (a
        # timeout or opaque error narrated as a robot fault) into a
        # structured local error naming the actual defect.
        if not isinstance(target, str) or not target:
            return {"error": "send_cmd: target must be a non-empty string", "ok": False}
        if "\x00" in target:
            return {"error": "send_cmd: target may not contain NUL", "ok": False}
        from strands_robots.mesh import security as _security

        try:
            cmd = _security.validate_command(cmd)
        except _security.ValidationError as exc:
            result = {"ok": False, "error": f"validation: {exc}"}
            self.record_activity(source, cmd.get("action", "?"), target=target, detail=result, ok=False)
            return result
        turn = uuid.uuid4().hex
        envelope = {
            "sender_id": self.peer_id,
            "turn_id": turn,
            "command": cmd,
            "timestamp": time.time(),
        }
        # The transport silently drops cmd messages over the low-pass cap, so
        # the caller would otherwise see a bare timeout with nothing to act on.
        size = len(json.dumps(envelope, default=str).encode())
        if size > MAX_CMD_BYTES:
            result = {
                "ok": False,
                "error": f"command too large: {size} B > transport cap {MAX_CMD_BYTES} B "
                "(raise STRANDS_MESH_MAX_CMD_BYTES on every peer, or shrink the payload)",
            }
            self.record_activity(source, cmd.get("action", "?"), target=target, detail=result, ok=False)
            return result

        evt = threading.Event()
        with self._rpc_lock:
            self._pending[turn] = evt
            self._expected_responders[turn] = target
        started = time.time()
        try:
            put(f"strands/{target}/cmd", envelope)
            if not evt.wait(timeout):
                result = {"error": f"timeout after {timeout:g}s", "turn_id": turn, "ok": False}
            else:
                with self._rpc_lock:
                    result = self._responses.pop(turn, {"error": "response lost", "ok": False})
            if not result.get("error") and safety_state.proves_clear(str(cmd.get("action", ""))):
                with self._peers_lock:
                    self._lockout_proof[target] = time.time()
            self.record_activity(
                source,
                cmd.get("action", "?"),
                target=target,
                detail={"instruction": cmd.get("instruction"), "provider": cmd.get("policy_provider")},
                ok=command_succeeded(result),
                result=result,
                elapsed=time.time() - started,
            )
            return result
        finally:
            with self._rpc_lock:
                self._pending.pop(turn, None)
                self._responses.pop(turn, None)
                self._expected_responders.pop(turn, None)

    async def send_cmd_async(
        self,
        target: str,
        cmd: dict[str, Any],
        timeout: float = 30.0,
        *,
        source: str = "api",
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self.send_cmd, target, cmd, timeout, source=source)

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------

    def record_activity(
        self,
        source: str,
        action: str,
        *,
        target: str = "",
        detail: Any = None,
        ok: bool | None = None,
        result: Any = None,
        elapsed: float | None = None,
    ) -> None:
        entry = {
            "t": time.time(),
            "source": source,
            "action": action,
            "target": target,
            "ok": ok,
            "detail": detail,
            "elapsed": round(elapsed, 3) if elapsed is not None else None,
        }
        if result is not None:
            entry["result"] = json.dumps(result, default=str)[:400]
        with self._activity_lock:
            self.activity.append(entry)
        self._emit({"type": "activity", "data": entry})

    def activity_log(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._activity_lock:
            items = list(self.activity)
        return items[-limit:][::-1]

    # ------------------------------------------------------------------
    # Snapshot for initial page load
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        protected = self._protected_peer_ids()
        with self._peers_lock:
            peers = prune_peers(self.peers, now, PEER_TTL_S, protected)
            # Forget the aged-out peers for good: keeping them in self.peers
            # only feeds the same ghosts back on every later snapshot.
            for pid in set(self.peers) - set(peers):
                self.peers.pop(pid, None)
                # Drop coalescing bookkeeping too, so a peer that comes BACK with the same content as when it
                # left is forwarded at once instead of waiting out a rate window against a memory of its
                # former self.
                with self._coalesce_lock:
                    self._coalescer.forget(pid)
        stamps: dict[str, float] = {}
        for m in self._managed_children():
            mid = getattr(m, "peer_id", None)
            if mid:
                stamps[str(mid)] = float(getattr(m, "started_at", 0.0) or 0.0)
        quiet = managed_without_presence(peers, protected, spawn_times=stamps, now=now)
        for pid, origin in peer_origins(peers, protected).items():
            peer = peers.get(pid)
            if isinstance(peer, dict):
                peers[pid] = {**peer, "origin": origin}
        try:
            with self._peers_lock:
                fleet_lockout = getattr(self, "_lockout", None) or safety_state.Lockout()
                proofs = dict(getattr(self, "_lockout_proof", None) or {})
            for pid, peer in list(peers.items()):
                if not isinstance(peer, dict):
                    continue
                verdict = safety_state.resolve_peer(
                    fleet_lockout,
                    first_seen=peer.get("first_seen"),
                    proof_at=proofs.get(pid),
                )
                peers[pid] = {**peer, "lockout": verdict.as_fields()}
        except Exception as exc:  # pragma: no cover - an annotation must never break the fleet view
            logger.debug("lockout annotation skipped: %s", exc)
        for pid, fields in self._peer_annotations().items():
            peer = peers.get(pid)
            if isinstance(peer, dict) and isinstance(fields, dict):
                # A plain overlay, deliberately: ``joint_silence.merge`` grades a
                # ``joint_problem`` annotation against the peer's live joints
                # before applying it, but that module arrives with the server
                # slice of the #2848 decomposition (#2977) -- the same slice
                # that supplies the only ``peer_annotations`` hook, so until it
                # lands this loop has no input to grade. The server slice
                # upgrades this line to ``joint_silence.merge(peer, fields)``
                # when it brings both halves.
                peers[pid] = {**peer, **fields}
        return {
            "type": "snapshot",
            "managed_no_presence": quiet,
            "dashboard_peer_id": self.peer_id,
            "peers": peers,
            "mesh": self.mesh_info(),
            "absent_children": absent_children(peers, self._managed_children()),
            "t": now,
        }

    def live_peers(self) -> list[str]:
        """Peer ids with a fresh presence heartbeat."""
        now = time.time()
        with self._peers_lock:
            return [pid for pid, entry in self.peers.items() if (now - entry.get("last_seen", 0)) <= PEER_STALE_S]

    def latest_frame(self, peer_id: str, cam: str) -> dict[str, Any] | None:
        with self._frames_lock:
            return self.frames.get((peer_id, cam))
