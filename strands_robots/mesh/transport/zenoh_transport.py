"""Eclipse Zenoh transport — thin :class:`MeshTransport` adapter over the
existing :mod:`strands_robots.mesh.session` singleton.

This wrapper deliberately delegates to the legacy ``session.get_session()`` /
``session.put()`` / ``session.release_session()`` functions instead of
reimplementing them. That keeps:

1. **Zero behaviour change for existing callers.** Every test in
   ``tests/mesh/test_mesh_session.py`` that pokes
   ``session._SESSION`` / ``_SESSION_REFS`` keeps working — the legacy
   module is the single source of truth for Zenoh state.
2. **A single connect/teardown path.** Multiple :class:`ZenohTransport`
   instances in the same process all funnel into the same ref-counted
   ``zenoh.Session`` singleton.
3. **A clean migration story.** When the Zenoh implementation needs to
   evolve (e.g. to add per-key QoS or per-endpoint TLS), changes go into
   ``session.py`` and this wrapper benefits automatically.

The Zenoh dependency stays **lazy**: importing this module does not import
``zenoh``. The first :meth:`connect` call delegates to ``session.get_session``
which performs the actual lazy import.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ZenohTransport:
    """Concrete :class:`MeshTransport` backed by Eclipse Zenoh.

    Each instance holds **one** reference to the process-wide Zenoh session
    via ``mesh.session.get_session`` / ``release_session``. The underlying
    session is shared by all transport instances and only closes when the
    last reference releases.

    Thread safety: :meth:`connect` and :meth:`close` are guarded by an
    internal lock so concurrent ``Mesh.start()`` calls don't double-acquire.
    The hot path (:meth:`put`) is lock-free in steady-state — 50 Hz teleop
    loops never serialise on the transport.
    """

    def __init__(self) -> None:
        self._has_ref: bool = False
        self._lock = threading.Lock()

    # Lifecycle

    def connect(self) -> bool:
        """Acquire (or reuse) the shared Zenoh session.

        Returns ``True`` on success, ``False`` if Zenoh is unavailable. Calling
        :meth:`connect` twice on the same instance is a no-op — only one
        reference is held per :class:`ZenohTransport` instance regardless of
        how many times it's connected.
        """
        with self._lock:
            if self._has_ref:
                return True

            from strands_robots.mesh.session import _get_zenoh_session_directly

            session = _get_zenoh_session_directly()
            if session is None:
                return False
            self._has_ref = True
            return True

    def close(self) -> None:
        """Release this transport's reference to the shared session.

        Idempotent. The underlying ``zenoh.Session`` only closes when the
        last reference (across all transports and direct callers) releases.
        """
        with self._lock:
            if not self._has_ref:
                return
            from strands_robots.mesh.session import release_session

            release_session()
            self._has_ref = False

    # Inspection

    def is_alive(self) -> bool:
        """True if the underlying Zenoh session is open AND this instance
        currently holds a reference."""
        if not self._has_ref:
            return False
        from strands_robots.mesh.session import _session_alive_directly

        return _session_alive_directly()

    @property
    def raw_session(self) -> Any | None:
        """The underlying ``zenoh.Session``, or ``None`` if not open.

        Exposed for callers that need to perform Zenoh-specific operations
        not yet abstracted into the :class:`MeshTransport` protocol. New
        code should not depend on this — use :meth:`put` and
        :meth:`declare_subscriber`.
        """
        from strands_robots.mesh.session import _current_zenoh_session_directly

        return _current_zenoh_session_directly()

    # Pub/Sub

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish *data* (JSON-encoded) to *key*. Fire-and-forget.

        Delegates to :func:`strands_robots.mesh.session.put`.
        """
        from strands_robots.mesh.session import put

        put(key, data)

    def declare_subscriber(self, key_expr: str, handler: Callable[[Any], None]) -> Any:
        """Subscribe to *key_expr* and route inbound samples to *handler*.

        Returns the raw ``zenoh.Subscriber`` directly — already exposes
        ``.undeclare()`` so it satisfies the :class:`SubHandle` protocol.

        Raises ``RuntimeError`` if the session is not open.
        """
        session = self.raw_session
        if session is None:
            raise RuntimeError("Zenoh session not open")
        return session.declare_subscriber(key_expr, handler)
