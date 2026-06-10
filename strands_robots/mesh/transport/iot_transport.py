"""AWS IoT Core MQTT5 transport — :class:`MeshTransport` over mTLS.

Wraps an ``awscrt.mqtt5`` client behind the :class:`MeshTransport` Protocol so
:class:`~strands_robots.mesh.core.Mesh` can publish presence, state, RPCs, and
safety events to AWS IoT Core with X.509 mutual TLS.

The strands-robots topic scheme is already MQTT-safe — every Zenoh key like
``strands/{peer}/state`` translates verbatim. The only translations that
happen here are wildcard mapping (``*`` → ``+``, ``**`` → ``#``) and the
delivery shape — MQTT5's flat ``(topic_str, bytes)`` callback is wrapped in
a tiny ``_MqttSample`` so existing :class:`Mesh` handlers work unmodified.

Trust model
-----------
Each robot owns an X.509 cert tied to a Thing whose name **equals** the
:class:`Mesh` peer_id. AWS IoT Policy enforces topic-level ACLs via the
``${iot:Connection.Thing.ThingName}`` substitution. See
:doc:`../../research/IOT_SPIKE_FINDINGS` for the full policy templates.

Required environment
--------------------
``STRANDS_IOT_ENDPOINT``
    The AWS IoT Core ATS endpoint, e.g.
    ``a2acz9p1ge6619-ats.iot.us-west-2.amazonaws.com``.
``STRANDS_IOT_THING_NAME``
    The Thing name. MUST equal the cert's CN. The ``client_id`` used at
    connect time is set to this value so policy variable substitution works.
``STRANDS_IOT_CERT_DIR``
    Directory holding ``{thing}.cert.pem``, ``{thing}.private.key``,
    ``AmazonRootCA1.pem``. Defaults to ``~/.strands_robots/iot``.

Optional
--------
``STRANDS_IOT_CA_FILE``
    Path to the root CA file. Defaults to ``$STRANDS_IOT_CERT_DIR/AmazonRootCA1.pem``.

Failure mode
------------
If ``awsiotsdk`` is not installed, :meth:`connect` returns ``False`` and the
transport behaves as a silent no-op (matching :class:`ZenohTransport` when
Zenoh is missing). If the endpoint or cert files are missing, :meth:`connect`
logs at ERROR and returns ``False`` — the mesh stays off rather than crash
the host.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default per-topic QoS / retain map.
#
# Derived from the empirically-validated policy in the spike (§4.1 of
# AWS_IOT_MESH_INTEGRATION.md). Topics not listed default to QoS 0, no retain.
#
# QoS values: 0 = at most once, 1 = at least once, 2 = exactly once.
_TOPIC_POLICY: dict[str, tuple[int, bool]] = {
    # Pattern (suffix-matched against key after the peer_id segment) -> (qos, retain).
    "presence": (1, True),
    "state": (0, False),
    "cmd": (1, False),
    "broadcast": (1, False),
    "response": (1, False),  # matches strands/{peer}/response/{turn}
    "pose": (0, False),
    "imu": (0, False),
    "odom": (0, False),
    "health": (0, True),
    "lidar/summary": (0, False),
    "lidar/state": (0, True),
    "map/info": (0, True),
    "safety/event": (1, True),
    "safety/estop": (1, True),
    "safety/resume": (1, True),  # paired with safety/estop; closes incident windows
    "stream": (0, False),
    "stream/meta": (0, False),
    # Camera frames are too big for MQTT — IotMqttTransport drops them.
    # See the design doc §4.2 for the S3 offload pattern.
    "camera": ("DROP", False),  # type: ignore[dict-item]
}

# Topics we strictly never publish over MQTT, regardless of caller intent.
# Camera frames hit MQTT's 128 KB cap; teleop input has fatal latency over WAN.
_NEVER_BRIDGE_PREFIXES: tuple[str, ...] = (
    "camera/",  # JPEG frames — use S3 offload (Layer 3)
    "input/",  # 50 Hz teleop — LAN-only
    "hand/",  # 50 Hz hand control — LAN-only
)


class _MqttSample:
    """Zenoh-shaped Sample wrapper around an MQTT message.

    Mesh handlers (``_on_presence``, ``_on_cmd``, ``_on_response``) all access
    ``sample.key_expr`` and ``sample.payload.to_bytes()``. By exposing the
    same shape we avoid touching any handler when the transport changes.
    """

    __slots__ = ("key_expr", "payload")

    def __init__(self, topic: str, payload_bytes: bytes) -> None:
        self.key_expr = topic
        self.payload = _MqttPayload(payload_bytes)


class _MqttPayload:
    """``zenoh.Sample.payload``-shaped wrapper exposing ``to_bytes()``."""

    __slots__ = ("_bytes",)

    def __init__(self, b: bytes) -> None:
        self._bytes = b

    def to_bytes(self) -> bytes:
        return self._bytes


class _MqttSubHandle:
    """Subscription handle that calls ``unsubscribe`` on undeclare.

    Mirrors ``zenoh.Subscriber.undeclare()`` so :class:`Mesh` teardown code
    is transport-agnostic.
    """

    def __init__(self, transport: IotMqttTransport, topic_filter: str, handler: Any = None) -> None:
        self._transport = transport
        self._topic_filter = topic_filter
        self._handler = handler
        self._undeclared = False

    def undeclare(self) -> None:
        if self._undeclared:
            return
        self._undeclared = True
        self._transport._unsubscribe(self._topic_filter, self._handler)


def _zenoh_to_mqtt_filter(key_expr: str) -> str:
    """Translate a Zenoh key-expression to an MQTT topic filter.

    Zenoh uses ``*`` for "one or more characters within a segment" (in
    practice we only use it as a single-segment match) and ``**`` for
    "any number of segments". MQTT uses ``+`` and ``#`` respectively.

    Patterns we actually use in :class:`Mesh`::

        strands/*/presence  -> strands/+/presence
        strands/{peer}/response/**  -> strands/{peer}/response/#
        strands/broadcast  -> strands/broadcast (unchanged)
        strands/{peer}/cmd  -> strands/{peer}/cmd (unchanged)

    We do **not** support arbitrary Zenoh key-expression syntax here. Callers
    that pass anything we don't recognise get a faithful pass-through and a
    DEBUG log entry; the broker will then SUBACK-deny it cleanly.
    """
    # Walk segments: each '*' segment becomes '+', a trailing '**' becomes '#'.
    segments = key_expr.split("/")
    out: list[str] = []
    for i, seg in enumerate(segments):
        if seg == "**":
            # Tail wildcard. Must be last segment in MQTT; if it isn't, log at
            # DEBUG (caller is using a Zenoh idiom that doesn't translate) and
            # leave the rest verbatim — broker will reject.
            if i != len(segments) - 1:
                logger.debug("Zenoh '**' is not in tail position in %r — MQTT may reject", key_expr)
            out.append("#")
        elif seg == "*":
            out.append("+")
        else:
            out.append(seg)
    return "/".join(out)


def _qos_and_retain_for(topic: str) -> tuple[int, bool]:
    """Look up the default QoS and retain flag for a topic.

    Resolves the suffix that follows the ``strands/...`` prefix and matches
    it against :data:`_TOPIC_POLICY`. Handles three layouts:

    - ``strands/broadcast``  -> suffix ``broadcast``
    - ``strands/safety/estop``  -> suffix ``safety/estop``
    - ``strands/{peer}/{topic}/...``  -> suffix ``{topic}/...``

    Topics with no entry in the policy get ``(0, False)``. Topics flagged as
    ``"DROP"`` return ``(-1, False)`` so callers can short-circuit.
    """
    if not topic.startswith("strands/"):
        return 0, False

    rest = topic[len("strands/") :]
    if not rest:
        return 0, False

    rest_segments = rest.split("/")
    first = rest_segments[0]

    # Two distinct topic layouts in the strands-robots scheme. They MUST NOT
    # be tried as a fallback chain — a peer_id that happens to be named
    # "broadcast" or "safety" must NOT pick up the top-level policy entry.
    #
    #  (a) Top-level system topics — first segment IS the kind:
    #  strands/broadcast
    #  strands/safety/estop
    #
    #  (b) Per-peer topics — first segment is the peer_id, topic kind
    #  starts at segment 1:
    #  strands/{peer}/{kind}  (e.g. presence, state, cmd)
    #  strands/{peer}/{kind}/{sub}  (e.g. lidar/summary, response/{turn})
    #
    # We resolve the layout by checking whether *first* is one of the
    # reserved top-level kinds. The set is small and closed — extending
    # the topic scheme means extending this set.
    _TOP_LEVEL_KINDS = {"broadcast", "safety"}

    if first in _TOP_LEVEL_KINDS:
        # Layout (a) — match suffixes that include the first segment.
        for n in range(len(rest_segments), 0, -1):
            candidate = "/".join(rest_segments[:n])
            entry = _TOPIC_POLICY.get(candidate)
            if entry is not None:
                qos_or_drop, retain = entry
                if qos_or_drop == "DROP":
                    return -1, False
                return int(qos_or_drop), retain
        return 0, False

    # Layout (b) — first segment is a peer_id; skip it.
    if len(rest_segments) < 2:
        return 0, False

    # Match suffixes from rest_segments[1:] longest-first.
    for n in range(len(rest_segments), 1, -1):
        candidate = "/".join(rest_segments[1:n])
        entry = _TOPIC_POLICY.get(candidate)
        if entry is not None:
            qos_or_drop, retain = entry
            if qos_or_drop == "DROP":
                return -1, False
            return int(qos_or_drop), retain

    return 0, False


def _should_drop(topic: str) -> bool:
    """True if the topic's payload should never traverse MQTT (camera/input/hand)."""
    parts = topic.split("/", 2)
    suffix = parts[2] if len(parts) == 3 else topic
    return suffix.startswith(_NEVER_BRIDGE_PREFIXES)


class IotMqttTransport:
    """Concrete :class:`MeshTransport` backed by AWS IoT Core MQTT5/mTLS.

    One instance manages exactly one ``awscrt.mqtt5.Client``. The client's
    ``client_id`` MUST equal the Thing name attached to the cert — the
    constructor enforces this so policy variable substitution works.

    Subscriptions are tracked in a dict keyed by topic_filter so
    :meth:`_unsubscribe` can locate them on undeclare.

    Thread safety
    -------------
    The ``awscrt`` library calls handlers on its own IO thread. We protect
    the subscription dict with a small lock; ``put`` is lock-free.
    """

    def __init__(
        self,
        thing_name: str | None = None,
        endpoint: str | None = None,
        cert_dir: str | None = None,
        ca_file: str | None = None,
        connect_timeout: float = 15.0,
    ) -> None:
        self._thing_name = thing_name or os.getenv("STRANDS_IOT_THING_NAME", "")
        self._endpoint = endpoint or os.getenv("STRANDS_IOT_ENDPOINT", "")
        self._cert_dir = Path(cert_dir or os.getenv("STRANDS_IOT_CERT_DIR") or Path.home() / ".strands_robots" / "iot")
        self._ca_file = ca_file or os.getenv("STRANDS_IOT_CA_FILE") or str(self._cert_dir / "AmazonRootCA1.pem")
        self._connect_timeout = connect_timeout

        self._client: Any | None = None
        self._connected = threading.Event()
        self._lock = threading.Lock()
        # topic_filter -> list of handlers (multiple subs to same topic OK)
        self._handlers: dict[str, list[Callable[[Any], None]]] = {}

    # Lifecycle

    def connect(self) -> bool:
        """Open the MQTT5 client and wait for CONNACK.

        Returns ``True`` once connected, ``False`` if the SDK is missing,
        configuration is invalid, or the broker is unreachable within
        ``connect_timeout`` seconds.
        """
        with self._lock:
            if self._client is not None and self._connected.is_set():
                return True

            try:
                from awsiot import mqtt5_client_builder
            except ImportError:
                logger.error(
                    "awsiotsdk not installed — IoT transport disabled. "
                    "Install with: pip install 'strands-robots[mesh-iot]'"
                )
                return False

            # Validate config
            if not self._thing_name:
                logger.error(
                    "STRANDS_IOT_THING_NAME is required for IoT transport "
                    "(must match the AWS IoT Thing name attached to the cert)"
                )
                return False
            if not self._endpoint:
                logger.error("STRANDS_IOT_ENDPOINT is required for IoT transport")
                return False

            cert_path = self._cert_dir / f"{self._thing_name}.cert.pem"
            key_path = self._cert_dir / f"{self._thing_name}.private.key"
            ca_path = Path(self._ca_file)
            for p, label in [
                (cert_path, "certificate"),
                (key_path, "private key"),
                (ca_path, "CA file"),
            ]:
                if not p.exists():
                    logger.error("IoT %s not found: %s", label, p)
                    return False

            self._connected.clear()
            self._client = mqtt5_client_builder.mtls_from_path(
                endpoint=self._endpoint,
                cert_filepath=str(cert_path),
                pri_key_filepath=str(key_path),
                ca_filepath=str(ca_path),
                client_id=self._thing_name,  # MUST match Thing name
                on_lifecycle_connection_success=self._on_connection_success,
                on_lifecycle_connection_failure=self._on_connection_failure,
                on_lifecycle_disconnection=self._on_disconnection,
                on_publish_received=self._on_publish_received,
            )
            self._client.start()
            ok = self._connected.wait(self._connect_timeout)
            if not ok:
                logger.error(
                    "IoT connection to %s timed out after %.1fs",
                    self._endpoint,
                    self._connect_timeout,
                )
                self._client.stop()
                self._client = None
                return False

            logger.info(
                "IoT mesh session opened (thing=%s, endpoint=%s)",
                self._thing_name,
                self._endpoint,
            )
            return True

    def close(self) -> None:
        """Disconnect and tear down the MQTT5 client. Idempotent."""
        with self._lock:
            if self._client is None:
                return
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
            self._connected.clear()
            self._handlers.clear()
            logger.info("IoT mesh session closed (thing=%s)", self._thing_name)

    # Inspection

    def is_alive(self) -> bool:
        """True if the MQTT client is connected."""
        return self._client is not None and self._connected.is_set()

    @property
    def thing_name(self) -> str:
        return self._thing_name or ""

    # Pub/Sub

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Publish *data* to *key*. Fire-and-forget.

        Per-topic QoS and retain flags come from :data:`_TOPIC_POLICY`.
        Topics in :data:`_NEVER_BRIDGE_PREFIXES` (camera/input/hand) are
        silently dropped — they belong on Zenoh-LAN, not MQTT-WAN.
        """
        if self._client is None or not self._connected.is_set():
            return

        if _should_drop(key):
            return

        qos, retain = _qos_and_retain_for(key)
        if qos < 0:
            return  # explicit DROP

        try:
            from awscrt import mqtt5

            qos_enum = mqtt5.QoS.AT_MOST_ONCE if qos == 0 else mqtt5.QoS.AT_LEAST_ONCE
            self._client.publish(
                mqtt5.PublishPacket(
                    topic=key,
                    payload=json.dumps(data).encode(),
                    qos=qos_enum,
                    retain=retain,
                )
            )
        except Exception as exc:
            logger.debug("MQTT put error on %s: %s", key, exc)

    def declare_subscriber(self, key_expr: str, handler: Callable[[Any], None]) -> Any:
        """Subscribe to *key_expr* (Zenoh form) translated to an MQTT topic filter.

        Multiple subscribers to the same filter are allowed — handlers are
        appended to a per-filter list. Each :class:`_MqttSubHandle` only
        removes its own handler on undeclare.
        """
        if self._client is None or not self._connected.is_set():
            raise RuntimeError("IoT MQTT client not connected")

        from awscrt import mqtt5

        topic_filter = _zenoh_to_mqtt_filter(key_expr)

        with self._lock:
            already_subscribed = topic_filter in self._handlers
            self._handlers.setdefault(topic_filter, []).append(handler)

        if not already_subscribed:
            try:
                self._client.subscribe(
                    mqtt5.SubscribePacket(
                        subscriptions=[
                            mqtt5.Subscription(
                                topic_filter=topic_filter,
                                qos=mqtt5.QoS.AT_LEAST_ONCE,
                            )
                        ]
                    )
                ).result(timeout=5)
            except Exception as exc:
                # Roll back the handler registration so a retry works cleanly.
                with self._lock:
                    self._handlers.get(topic_filter, []).remove(handler)
                    if not self._handlers.get(topic_filter):
                        self._handlers.pop(topic_filter, None)
                raise RuntimeError(f"MQTT subscribe to {topic_filter!r} failed: {exc}") from exc

        return _MqttSubHandle(self, topic_filter, handler)

    # Internal

    def _unsubscribe(self, topic_filter: str, handler: Any = None) -> None:
        """Remove *handler* for *topic_filter*; unsubscribe if last."""
        with self._lock:
            handlers = self._handlers.get(topic_filter)
            if not handlers:
                return
            if handler is not None:
                try:
                    handlers.remove(handler)
                except ValueError:
                    pass  # handler already gone
            else:
                handlers.pop()  # legacy fallback: remove last
            if handlers:
                return  # other subscribers still active
            self._handlers.pop(topic_filter, None)

        # Last handler removed — unsubscribe at the broker.
        if self._client is None:
            return
        try:
            from awscrt import mqtt5

            self._client.unsubscribe(mqtt5.UnsubscribePacket(topic_filters=[topic_filter])).result(timeout=5)
        except Exception as exc:
            logger.debug("MQTT unsubscribe error on %s: %s", topic_filter, exc)

    # Callbacks

    def _on_connection_success(self, data: Any) -> None:
        logger.info("IoT MQTT connected (thing=%s)", self._thing_name)
        self._connected.set()

    def _on_connection_failure(self, data: Any) -> None:
        logger.warning("IoT MQTT connection failure: %s", data.exception)
        self._connected.clear()

    def _on_disconnection(self, data: Any) -> None:
        logger.info("IoT MQTT disconnected (thing=%s)", self._thing_name)
        self._connected.clear()

    def _on_publish_received(self, data: Any) -> None:
        """Route inbound messages to subscriber handlers via topic-filter match."""
        topic = data.publish_packet.topic
        payload = bytes(data.publish_packet.payload or b"")

        # Match topic against all registered filters. MQTT brokers route by
        # filter, but our handler-dict is keyed by the original filter — so
        # we need to test each registered filter for a topic match.
        with self._lock:
            matching = [(f, list(handlers)) for f, handlers in self._handlers.items() if _mqtt_topic_matches(f, topic)]

        if not matching:
            return

        sample = _MqttSample(topic, payload)
        for _filter, handlers in matching:
            for handler in handlers:
                try:
                    handler(sample)
                except Exception as exc:
                    logger.debug("IoT handler error on %s: %s", topic, exc)


def _mqtt_topic_matches(filter_: str, topic: str) -> bool:
    """True if MQTT *topic* matches the topic-filter *filter_*.

    Implements the standard MQTT v5 wildcard semantics:

    - ``+`` matches exactly one topic level
    - ``#`` matches zero or more trailing topic levels (must be at end)
    - other segments must match literally
    """
    f_parts = filter_.split("/")
    t_parts = topic.split("/")

    for i, fp in enumerate(f_parts):
        if fp == "#":
            # Tail wildcard — matches everything from here, including zero
            # remaining segments.
            return True
        if i >= len(t_parts):
            return False
        if fp == "+":
            continue
        if fp != t_parts[i]:
            return False

    # Filter exhausted — topic matches iff topic is also exhausted.
    return len(t_parts) == len(f_parts)
