"""Pluggable transport protocol for the strands-robots mesh.

Defines the Protocol that :class:`~strands_robots.mesh.core.Mesh` uses to
publish and subscribe. Every concrete backend (Zenoh, AWS IoT MQTT, Bridge)
implements this protocol.

The protocol is deliberately tiny — exactly what ``mesh.session`` already
exposed. Every behavioural enrichment (peer registry, RPC correlation, audit)
lives at the Mesh layer and is transport-agnostic.

Why ``Sample`` is duck-typed
----------------------------
Zenoh callbacks receive a ``zenoh.Sample`` with ``.key_expr`` and
``.payload.to_bytes()``. AWS IoT MQTT5 callbacks receive a topic string and
``bytes`` payload. Rather than pick a concrete type and force one transport
to adapt, we declare the **structural protocol** all callers actually use:

    sample.key_expr  # str  — the topic / key the message arrived on
    sample.payload.to_bytes()  # bytes — the raw payload

Concrete backends produce objects matching this shape. The MQTT backend ships
a tiny ``_MqttSample`` wrapper; the Zenoh backend passes ``zenoh.Sample``
through unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class _PayloadLike(Protocol):
    def to_bytes(self) -> bytes: ...


@runtime_checkable
class Sample(Protocol):
    """Structural protocol for messages delivered to subscriber callbacks.

    Concrete shape that every backend's callback delivery must satisfy.
    Mirrors ``zenoh.Sample`` so existing Mesh handlers (``_on_presence``,
    ``_on_cmd``, ``_on_response``) work unchanged regardless of transport.
    """

    key_expr: Any  # zenoh: KeyExpr; mqtt: str — both stringify
    payload: _PayloadLike


@runtime_checkable
class SubHandle(Protocol):
    """Opaque subscription handle — must support ``undeclare()`` for teardown.

    Mirrors ``zenoh.Subscriber.undeclare()``. MQTT-backed implementations wrap
    the broker's ``unsubscribe`` packet behind the same name so Mesh teardown
    code is transport-agnostic.
    """

    def undeclare(self) -> None: ...


@runtime_checkable
class MeshTransport(Protocol):
    """Pluggable transport for :class:`~strands_robots.mesh.core.Mesh`.

    Lifetime contract
    -----------------
    Implementations are ref-counted singletons per process, mirroring the
    existing :func:`~strands_robots.mesh.session.get_session` /
    :func:`~strands_robots.mesh.session.release_session` pair.

    - The first :class:`Mesh` to require a transport calls :func:`get_transport`
      which constructs (or returns) the singleton and increments its refcount.
    - Each :class:`Mesh.stop` calls :func:`release_transport` exactly once.
    - When the refcount reaches zero, :meth:`close` is invoked.

    Failure mode
    ------------
    A transport whose connection is dead returns ``False`` from
    :meth:`is_alive`. Callers (notably :func:`put`) treat a dead transport as
    a no-op rather than raising — preserving the current Mesh contract that
    publish failures never propagate up into hot control loops.
    """

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish a JSON-serialisable payload to the wire.

        Fire-and-forget. MUST NOT raise on transient failure. Implementations
        should log at debug and continue, matching the Zenoh behaviour today.

        Args:
            key: The topic / Zenoh key expression. For MQTT-backed transports
                this is the MQTT topic verbatim (no translation needed — our
                topic scheme is already MQTT-safe).
            data: A JSON-serialisable dictionary. Implementations are expected
                to encode it via ``json.dumps(...).encode()``.
        """
        ...

    def declare_subscriber(self, key_expr: str, handler: Callable[[Sample], None]) -> SubHandle:
        """Subscribe to a key expression and route inbound messages to *handler*.

        ``handler`` receives a :class:`Sample`-shaped object. For Zenoh that's
        ``zenoh.Sample`` directly. For MQTT it's a thin wrapper that exposes
        ``.key_expr`` (the topic string) and ``.payload.to_bytes()`` (the
        payload bytes).

        Wildcard translation:
            Zenoh ``*``  matches one segment → MQTT ``+``
            Zenoh ``**`` matches tail  → MQTT ``#``
            MQTT-backed implementations translate these on the fly.

        Args:
            key_expr: Zenoh-style key expression. Concrete patterns we use:
                ``strands/*/presence``, ``strands/{peer}/cmd``,
                ``strands/{peer}/response/**``, ``strands/broadcast``.
            handler: Callback invoked once per received message. Runs on the
                transport's IO thread; must NOT block.

        Returns:
            An opaque :class:`SubHandle` that the caller must keep alive for
            the duration of the subscription, and call ``.undeclare()`` on
            during teardown.
        """
        ...

    def is_alive(self) -> bool:
        """True if the transport's session is open and usable."""
        ...

    def close(self) -> None:
        """Tear down the transport. Idempotent.

        Called when the last :class:`Mesh` referencing this transport stops.
        Implementations should release sockets, drain queues, and close any
        underlying client.
        """
        ...
