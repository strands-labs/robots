"""The AWS IoT Core transport: rpcs as confirmed Direct Messages, state and events over MQTT.

``iot://us-west-2`` connects with IAM credentials (SigV4 WebSocket); ``?cert=&key=`` as a device certificate;
``?endpoint=`` overrides ``aws iot describe-endpoint``; ``?client=`` fixes the MQTT client id (a device policy may
require its thing name). ``iot://fake?broker=<name>`` is the in-process FakeIotBroker with the same semantics.

A query goes to the ONE client that holds the device (learned from its presence heartbeats) as a Direct Message
with confirmation: a client that is not connected answers 404 and nothing is queued, so an e-stop never replays
later. The reply comes back on ``strands/<tenant>/@reply/<caller client>``. IoT has no liveliness, so a token is a
heartbeat every ``presence_ttl / 3`` and a watcher forgets a key it has not heard for ``presence_ttl``. A key is
held by the first client heard on it until that client's goodbye or silence; another client's claim is ignored.

Deploy requirement (a client id is whatever the client says unless the policy pins it): give each device a
certificate whose policy allows ``iot:Connect`` only as ``${iot:Connection.Thing.ThingName}`` and ``iot:Publish``
on ``strands/<tenant>/@presence/*`` only for the device ids that thing serves; callers (IAM) get no publish right on
``@presence``. A targeted e-stop is also published on the tenant ``@estop`` key, so a diverted Direct Message is never
the only stop.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

from ..iot.direct import reply_ok, rpc_mode
from .base import OnAlive, OnQuery, OnSample, RemoteError, Unsubscribe
from .iot_link import AwsLink, FakeIotBroker, FakeLink, Link


class Offline(LookupError):
    """The device's client is not connected (Direct Messaging 404): nothing was delivered or queued."""


def _filter(key_expr: str) -> str:
    parts = ["+" if p == "*" else "#" if p == "**" else p for p in key_expr.split("/")]
    return "/".join(parts)


def _link(url: str) -> Link:
    u = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    client = q.get("client") or f"strands-{uuid.uuid4().hex[:12]}"
    if u.netloc == "fake":
        return FakeLink(FakeIotBroker.named(q.get("broker", "default")), client)
    endpoint = q.get("endpoint")
    if not endpoint:
        import boto3

        iot = boto3.client("iot", region_name=u.netloc)
        endpoint = iot.describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]
    return AwsLink(endpoint, u.netloc, client, cert=q.get("cert"), key=q.get("key"))


class IotTransport:
    def __init__(self, url: str, presence_ttl: float = 3.0, link: Link | None = None) -> None:
        self.url = url
        self.ttl = presence_ttl
        self.link = link or _link(url)
        self.rpc = rpc_mode({k: v[0] for k, v in parse_qs(urlparse(url).query).items()})
        self._pending: dict[str, Future[bytes]] = {}
        self._clients: dict[str, str] = {}  # presence key's device id -> the client holding it
        self._replies: set[str] = set()
        self._holders: dict[str, tuple[str, float]] = {}  # liveliness key -> (client, last heard)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="iot-rpc")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._undo: list[Unsubscribe] = []

    def publish(self, key: str, payload: bytes) -> None:
        self.link.publish(key, payload)

    def subscribe(self, key_expr: str, cb: OnSample) -> Unsubscribe:
        return self._own(self.link.subscribe(_filter(key_expr), cb))

    def _own(self, undo: Unsubscribe) -> Unsubscribe:
        self._undo.append(undo)
        return undo

    def queryable(self, key_expr: str, handler: OnQuery, *, inline: bool = False) -> Unsubscribe:
        def answer(key: str, raw: bytes) -> None:
            req = json.loads(raw)
            if not reply_ok(str(req.get("reply", "")), key.split("/")[1]):
                return  # a reply address outside this tenant's @reply keys: never answered
            try:
                out = {"id": req["id"], "ok": handler(key, req["body"].encode()).decode()}
            except Exception as exc:
                out = {"id": req["id"], "err": f"{type(exc).__name__}: {exc}"}
            self.link.publish(req["reply"], json.dumps(out).encode())

        def on(key: str, raw: bytes) -> None:
            if inline:
                answer(key, raw)
            else:
                self._pool.submit(answer, key, raw)

        return self.subscribe(key_expr, on)

    def query(self, key: str, payload: bytes, timeout: float) -> list[bytes]:
        parts = key.split("/")
        tenant, did = parts[1], parts[2]
        client = self._clients.get(did)
        if client is None:
            return []
        reply = f"strands/{tenant}/@reply/{self.link.client_id}"
        with self._lock:
            if reply not in self._replies:
                self._replies.add(reply)
                self.subscribe(reply, self._on_reply)
        rid = uuid.uuid4().hex
        fut: Future[bytes] = Future()
        self._pending[rid] = fut
        body = json.dumps({"id": rid, "reply": reply, "body": payload.decode()}).encode()
        try:
            if self.rpc == "mqtt":  # request/response: the device's silence is the offline signal, presence says why
                self.link.publish(key, body, response_topic=reply, correlation_data=rid)
                return [fut.result(timeout)]
            status = self.link.send_direct(client, key, body, timeout=timeout)
            if status == 404:
                raise Offline(f"{did}: its client {client!r} is not connected (404); nothing was queued")
            if status != 200:
                raise RemoteError(f"{key}: direct message refused ({status})")
            return [fut.result(timeout)]
        except TimeoutError:
            return []
        finally:
            self._pending.pop(rid, None)

    def _on_reply(self, _key: str, raw: bytes) -> None:
        msg = json.loads(raw)
        fut = self._pending.get(msg.get("id", ""))
        if fut is None or fut.done():
            return
        if "err" in msg:
            fut.set_exception(RemoteError(msg["err"]))
        else:
            fut.set_result(str(msg["ok"]).encode())

    def liveliness(self, key: str) -> Unsubscribe:
        stop = threading.Event()
        beat = json.dumps({"alive": True, "client": self.link.client_id}).encode()

        def loop() -> None:
            while not stop.is_set() and not self._stop.is_set():
                self.link.publish(key, beat)
                stop.wait(self.ttl / 3)

        threading.Thread(target=loop, daemon=True, name="iot-beat").start()

        def drop() -> None:
            if not stop.is_set():
                stop.set()
                self.link.publish(key, json.dumps({"alive": False, "client": self.link.client_id}).encode())

        return self._own(drop)

    def _holder_ok(self, key: str, client: str, alive: bool) -> bool:
        """A key belongs to the client that holds it until that client says goodbye or goes silent for the ttl;
        nobody else may take it over or say goodbye for it (a forged heartbeat must never reroute an rpc)."""
        now = time.monotonic()
        with self._lock:
            held = self._holders.get(key)
            if held is not None and held[0] != client and now - held[1] <= self.ttl:
                return False
            if alive:
                self._holders[key] = (client, now)
                if "/@presence/" in key:
                    self._clients[key.rsplit("/", 1)[-1]] = client
            else:
                self._holders.pop(key, None)
        return True

    def watch_liveliness(self, key_expr: str, cb: OnAlive) -> Unsubscribe:
        seen: dict[str, float] = {}
        lock = threading.Lock()

        def on(key: str, raw: bytes) -> None:
            msg = json.loads(raw)
            alive, client = bool(msg.get("alive")), str(msg.get("client") or "")
            if not self._holder_ok(key, client, alive):
                return  # another client claiming a held key, or saying goodbye for it: ignored
            with lock:
                fresh = key not in seen
                seen[key] = time.monotonic() if alive else 0.0
                if not alive:
                    seen.pop(key, None)
            if alive and fresh or not alive:  # an explicit goodbye always counts, seen or not
                cb(key, alive)

        def expire() -> None:
            while not stop.wait(min(0.1, self.ttl / 10)):
                now = time.monotonic()
                with lock:
                    gone = [k for k, t in seen.items() if now - t > self.ttl]
                    for k in gone:
                        seen.pop(k)
                for k in gone:
                    cb(k, False)

        stop = threading.Event()
        threading.Thread(target=expire, daemon=True, name="iot-expire").start()
        undo = self.subscribe(key_expr, on)

        def unwatch() -> None:
            stop.set()
            undo()

        return self._own(unwatch)

    def close(self) -> None:
        for undo in reversed(self._undo):
            undo()
        self._stop.set()
        self.link.close()
        self._pool.shutdown(wait=False, cancel_futures=True)


def open(url: str, *, presence_ttl: float = 3.0) -> IotTransport:
    return IotTransport(url, presence_ttl)
