"""The MQTT side of the IoT transport: a ``Link`` to AWS IoT Core, and ``FakeIotBroker`` with the same semantics.

AWS: MQTT5 over mTLS (``cert``/``key``) as a device, or over a SigV4 WebSocket with IAM credentials; commands are
``SendDirectMessage`` with confirmation (200 delivered · 404 the client is not connected, nothing is queued ·
504 unconfirmed). The fake keeps exactly that: a direct message to an absent client is a 404 and is dropped.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any, Protocol

from ..._extras import hint
from .base import key_matches

DIRECT_PAYLOAD_CAP = 128 * 1024
Handler = Callable[[str, bytes], None]
_FAKES: dict[str, FakeIotBroker] = {}


class Link(Protocol):
    client_id: str

    def publish(self, topic: str, payload: bytes, **props: str) -> None:
        """``props`` are MQTT5 request/response properties (``response_topic``, ``correlation_data``); a link
        without MQTT5 ignores them (the body carries the same addressing)."""
        ...

    def subscribe(self, topic_filter: str, handler: Handler) -> Callable[[], None]: ...

    def send_direct(self, client_id: str, topic: str, payload: bytes, *, timeout: float) -> int: ...

    def close(self) -> None: ...


def mqtt_matches(topic_filter: str, topic: str) -> bool:
    return key_matches(topic_filter.replace("+", "*").replace("#", "**"), topic)


class _Subs:
    """Topic-filter handlers, delivered on one dispatcher thread (never on the network thread)."""

    def __init__(self, name: str) -> None:
        self.subs: dict[int, tuple[str, Handler]] = {}
        self.n = 0
        self.lock = threading.Lock()
        self.inbox: queue.Queue[tuple[str, bytes] | None] = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name=f"iot-{name}").start()

    def add(self, topic_filter: str, handler: Handler) -> Callable[[], None]:
        with self.lock:
            self.n += 1
            n = self.n
            self.subs[n] = (topic_filter, handler)
        return lambda: self.subs.pop(n, None) and None

    def _run(self) -> None:
        while (item := self.inbox.get()) is not None:
            topic, payload = item
            for f, h in list(self.subs.values()):
                if mqtt_matches(f, topic):
                    try:
                        h(topic, payload)
                    except Exception:  # one bad message never stops the dispatcher
                        pass

    def close(self) -> None:
        self.inbox.put(None)


class FakeIotBroker:
    """An in-process broker with Direct Messaging semantics; ``iot://fake?broker=<name>`` shares one by name."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: dict[str, FakeLink] = {}
        self.direct: list[tuple[str, str, int]] = []

    @classmethod
    def named(cls, name: str) -> FakeIotBroker:
        return _FAKES.setdefault(name, cls())

    def publish(self, topic: str, payload: bytes) -> None:
        with self.lock:
            links = list(self.clients.values())
        for link in links:
            link.subs.inbox.put((topic, payload))

    def send_direct(self, client_id: str, topic: str, payload: bytes) -> int:
        with self.lock:
            link = self.clients.get(client_id)
        status = 413 if len(payload) > DIRECT_PAYLOAD_CAP else 404 if link is None else 200
        self.direct.append((client_id, topic, status))
        if link is not None and status == 200:
            link.subs.inbox.put((topic, payload))
        return status


class FakeLink:
    def __init__(self, broker: FakeIotBroker, client_id: str) -> None:
        self.broker, self.client_id, self.closed = broker, client_id, False
        self.subs = _Subs(client_id)
        with broker.lock:
            broker.clients[client_id] = self

    def publish(self, topic: str, payload: bytes, **props: str) -> None:
        if not self.closed:
            self.broker.publish(topic, payload)

    def subscribe(self, topic_filter: str, handler: Handler) -> Callable[[], None]:
        return self.subs.add(topic_filter, handler)

    def send_direct(self, client_id: str, topic: str, payload: bytes, *, timeout: float) -> int:
        if self.closed:
            raise ConnectionError(f"iot: {self.client_id} is disconnected")
        return self.broker.send_direct(client_id, topic, payload)

    def close(self) -> None:
        self.closed = True
        with self.broker.lock:
            if self.broker.clients.get(self.client_id) is self:
                self.broker.clients.pop(self.client_id)
        self.subs.close()


def _device_client(cert: str, key: str, **kw: Any) -> Any:
    """The MQTT5 client for a device certificate: a ``pkcs11:`` key is a token (mTLS with PKCS#11), any other key
    is a PEM file on disk (mTLS from path)."""
    from awsiot import mqtt5_client_builder

    from .pkcs11 import client as pkcs11_client
    from .pkcs11 import is_pkcs11

    if is_pkcs11(key):
        return pkcs11_client(cert, key, **kw)
    return mqtt5_client_builder.mtls_from_path(cert_filepath=cert, pri_key_filepath=key, **kw)


class AwsLink:
    """One MQTT5 connection to AWS IoT Core (mTLS with cert/key, else SigV4 WebSocket) + boto3 SendDirectMessage."""

    def __init__(
        self,
        endpoint: str,
        region: str,
        client_id: str,
        *,
        cert: str | None = None,
        key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        try:
            from awscrt import auth, mqtt5
            from awsiot import mqtt5_client_builder
        except ImportError as exc:
            raise ImportError(f"the iot mesh transport needs awsiotsdk + boto3: {hint('iot')}") from exc
        self.client_id, self.endpoint, self.region, self._m = client_id, endpoint, region, mqtt5
        self.subs = _Subs(client_id)
        self._up, self._down, self._why = threading.Event(), threading.Event(), ""
        self._boto: Any = None
        kw: dict[str, Any] = dict(
            endpoint=endpoint,
            client_id=client_id,
            clean_session=True,
            keep_alive_secs=30,
            on_publish_received=lambda d: self.subs.inbox.put(
                (str(d.publish_packet.topic), bytes(d.publish_packet.payload or b""))
            ),
            on_lifecycle_connection_success=lambda *_: self._up.set(),
            on_lifecycle_connection_failure=lambda d: setattr(self, "_why", str(getattr(d, "exception", d))),
            on_lifecycle_stopped=lambda *_: self._down.set(),
        )
        if cert and key:
            self.client = _device_client(cert, key, **kw)
        else:
            creds = auth.AwsCredentialsProvider.new_default_chain()
            self.client = mqtt5_client_builder.websockets_with_default_aws_signing(
                region=region, credentials_provider=creds, **kw
            )
        self.client.start()
        if not self._up.wait(timeout):
            raise TimeoutError(f"iot: {client_id} did not connect to {endpoint} within {timeout}s ({self._why})")

    def publish(self, topic: str, payload: bytes, **props: str) -> None:
        pkt = self._m.PublishPacket(topic=topic, payload=payload, qos=self._m.QoS.AT_MOST_ONCE)
        if props.get("response_topic"):
            pkt.response_topic = props["response_topic"]
            pkt.correlation_data_bytes = str(props.get("correlation_data", "")).encode()
        self.client.publish(pkt)

    def subscribe(self, topic_filter: str, handler: Handler) -> Callable[[], None]:
        m = self._m
        undo = self.subs.add(topic_filter, handler)
        sub = m.Subscription(topic_filter=topic_filter, qos=m.QoS.AT_LEAST_ONCE)
        codes = [int(c) for c in self.client.subscribe(m.SubscribePacket([sub])).result(10.0).reason_codes or []]
        if any(c >= 0x80 for c in codes):
            undo()
            raise PermissionError(f"iot: subscribe to {topic_filter!r} refused ({codes}): check iot:Subscribe")
        return undo

    def send_direct(self, client_id: str, topic: str, payload: bytes, *, timeout: float) -> int:
        from botocore.exceptions import ClientError

        if self._boto is None:
            import boto3
            from botocore.config import Config

            cfg = Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 1})
            self._boto = boto3.client(
                "iot-data", region_name=self.region, endpoint_url=f"https://{self.endpoint}", config=cfg
            )
            if not hasattr(self._boto, "send_direct_message"):
                raise ImportError(
                    "the iot mesh transport needs boto3>=1.43.17 for iot-data SendDirectMessage; "
                    f"this environment has an older boto3: {hint('iot')}"
                )
        if len(payload) > DIRECT_PAYLOAD_CAP:
            return 413
        try:
            self._boto.send_direct_message(
                clientId=client_id,
                topic=topic,
                payload=payload,
                confirmation=True,
                timeout=max(1, min(10, int(timeout))),
            )
        except ClientError as exc:
            return int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 500)
        return 200

    def close(self) -> None:
        try:
            self.client.stop()
            self._down.wait(5.0)
        except Exception:  # closing a link that already dropped: nothing left to stop
            pass
        self.subs.close()
