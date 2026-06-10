"""Process-wide :class:`MeshTransport` factory and ref-counted singleton.

Mirrors the existing :func:`~strands_robots.mesh.session.get_session` /
:func:`~strands_robots.mesh.session.release_session` pair so :class:`Mesh`
can swap transports by setting ``STRANDS_MESH_BACKEND`` without changing
its lifecycle code.

Backend selection
-----------------
Selection is done at the first :func:`get_transport` call:

- ``zenoh`` (default) — :class:`ZenohTransport`
- ``iot``  — :class:`IotMqttTransport`
- ``bridge``  — :class:`BridgeTransport` (Zenoh + IoT)

Subsequent calls in the same process bump the refcount but do NOT switch
backends. To change the backend, every consumer must release first
(``release_transport`` until refcount is 0) and then a new selection is made.

The factory is **not** consulted by :mod:`strands_robots.mesh.session` —
that module owns the legacy zenoh path independently. The factory is the
new path used when :class:`Mesh` is configured for ``backend="iot"``.
"""

from __future__ import annotations

import logging
import os
import threading

from strands_robots.mesh.transport.base import MeshTransport

logger = logging.getLogger(__name__)


_TRANSPORT: MeshTransport | None = None
_TRANSPORT_REFS: int = 0
_TRANSPORT_BACKEND: str = ""
_LOCK = threading.Lock()


def _select_backend() -> str:
    """Resolve ``STRANDS_MESH_BACKEND``. Defaults to ``zenoh``.

    Unknown values fall back to ``zenoh`` with a warning — the policy is to
    keep the mesh running rather than crash the host on a typo.
    """
    raw = os.getenv("STRANDS_MESH_BACKEND", "zenoh").strip().lower()
    if raw not in ("zenoh", "iot", "bridge"):
        logger.warning("Unknown STRANDS_MESH_BACKEND=%r — falling back to 'zenoh'", raw)
        return "zenoh"
    return raw


def _construct(backend: str) -> MeshTransport:
    """Build a fresh transport for *backend*.

    Imports are deferred (inside this function) to avoid import-time
    circular dependencies: factory → zenoh_transport → session → factory.
    """
    if backend == "iot":
        from strands_robots.mesh.transport.iot_transport import IotMqttTransport

        return IotMqttTransport()
    if backend == "bridge":
        from strands_robots.mesh.transport.bridge_transport import BridgeTransport

        return BridgeTransport()

    from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

    return ZenohTransport()


def get_transport() -> MeshTransport | None:
    """Acquire (or reuse) the process-wide transport singleton.

    Increments the refcount each call. Returns ``None`` if the underlying
    backend's :meth:`connect` failed (Zenoh missing, certs missing, broker
    unreachable, etc.) — callers MUST treat ``None`` the same way they
    treated ``get_session() is None`` historically: skip mesh activity and
    move on without raising.
    """
    global _TRANSPORT, _TRANSPORT_REFS, _TRANSPORT_BACKEND  # noqa: PLW0603
    with _LOCK:
        if _TRANSPORT is not None:
            _TRANSPORT_REFS += 1
            return _TRANSPORT

        backend = _select_backend()
        transport = _construct(backend)

        # Try to connect; bail out and keep the singleton None on failure.
        ok = transport.connect()  # type: ignore[attr-defined]
        if not ok:
            logger.debug(
                "[mesh.transport] %s backend connect failed — staying off",
                backend,
            )
            return None

        _TRANSPORT = transport
        _TRANSPORT_REFS = 1
        _TRANSPORT_BACKEND = backend
        logger.info("[mesh.transport] %s backend ready", backend)
        return _TRANSPORT


def release_transport() -> None:
    """Release one reference. Closes when the last is gone. Idempotent."""
    global _TRANSPORT, _TRANSPORT_REFS, _TRANSPORT_BACKEND  # noqa: PLW0603
    with _LOCK:
        if _TRANSPORT_REFS <= 0:
            return
        _TRANSPORT_REFS -= 1
        if _TRANSPORT_REFS <= 0 and _TRANSPORT is not None:
            try:
                _TRANSPORT.close()
            except Exception:
                pass
            _TRANSPORT = None
            _TRANSPORT_REFS = 0
            _TRANSPORT_BACKEND = ""


def current_transport() -> MeshTransport | None:
    """Return the transport without bumping the refcount, or ``None``."""
    with _LOCK:
        return _TRANSPORT


def current_backend() -> str:
    """Return the backend name (``"zenoh"`` / ``"iot"`` / ``""`` if not running)."""
    with _LOCK:
        return _TRANSPORT_BACKEND
