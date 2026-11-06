"""RosBus: the one rclpy node of this process, spun on its own thread, with cached endpoints.

Reads never block on ROS: a subscription is made once per topic and keeps the latest message, a count, an
EMA rate and a condition to wait on (``wait_for_message`` on a node another thread spins returns nothing
on rmw_zenoh). Publishers are cached too, so a 50 Hz writer pays discovery once. Endpoints are created
under one lock (concurrent creation crashes some DDS bindings). rclpy is imported only when the bus starts.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import discovery as _disc
from . import qos as _qos
from .convert import message_class
from .names import INSTALL_HINT, RosUnavailable, canonical


@dataclass
class Sub:
    topic: str
    type: str
    qos: str
    handle: Any = None
    msg: Any = None
    t: float = 0.0  # monotonic time of the latest message
    count: int = 0
    hz: float = 0.0
    listeners: list[Callable[[Any], None]] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition)


class RosBus:
    """One node + one executor thread. ``RosBus.shared()`` is the process-wide one."""

    _shared: RosBus | None = None
    _shared_lock = threading.Lock()

    def __init__(self, discovery: _disc.Discovery | None = None, *, node_name: str | None = None) -> None:
        self.discovery = discovery or _disc.Discovery()
        self.node_name = node_name or f"strands_robots_{os.getpid()}"
        self.node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._owns_context = False
        self._start_lock = threading.Lock()
        self._endpoint_lock = threading.Lock()
        self._subs: dict[str, Sub] = {}
        self._pubs: dict[tuple[str, str, str], Any] = {}
        self._stop = threading.Event()
        self.owners: dict[str, str] = {}  # command topic -> the Robot that streams it (one writer per channel)

    @classmethod
    def shared(cls, discovery: _disc.Discovery | None = None) -> RosBus:
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls(discovery)
            elif discovery is not None and discovery != cls._shared.discovery:
                _disc.apply(discovery, initialised=cls._shared.node is not None)  # refuses a different graph
            return cls._shared

    def start(self) -> RosBus:
        with self._start_lock:  # two nodes in one process: the first one's subscriptions went deaf
            if self.node is not None:
                return self
            try:
                import rclpy
                from rclpy.executors import SingleThreadedExecutor
            except ImportError:
                raise RosUnavailable(INSTALL_HINT) from None
            self._rclpy = rclpy
            _disc.apply(self.discovery, initialised=rclpy.ok())
            if not rclpy.ok():
                rclpy.init()
                self._owns_context = True
            self.node = rclpy.create_node(self.node_name)
            # one thread: rclpy's MultiThreadedExecutor pays a thread-pool hop per callback (100 Hz read as ~25 Hz)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self.node)
            self._thread = threading.Thread(target=self._spin, name="ros-bus", daemon=True)
            self._thread.start()
            import atexit

            atexit.register(self.close)  # an executor thread alive at exit aborts the process (exit 134)
            self._at_exit = True
        return self

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: BLE001 - a shutdown race ends the loop; anything else is retried
                if self._stop.is_set():
                    return
                time.sleep(0.01)

    def subscribe(
        self, topic: str, type: str, *, qos: str = "auto", listener: Callable[[Any], None] | None = None
    ) -> Sub:
        """The cached subscription for ``topic`` (made once). ``listener`` is called on the executor thread."""
        topic = canonical(topic)
        self.start()
        with self._endpoint_lock:
            sub = self._subs.get(topic)
            if sub is not None and sub.type != type:
                raise ValueError(f"{topic} is already read as {sub.type}; one type per topic (rmw matches by type)")
            if sub is None:
                q = _qos.resolve(self.node, topic, qos)
                sub = Sub(topic, type, q)
                sub.handle = self.node.create_subscription(
                    message_class(type), topic, lambda m, s=sub: self._on(s, m), _qos.profile(q)
                )
                self._subs[topic] = sub
            if listener is not None and listener not in sub.listeners:
                sub.listeners.append(listener)
        return sub

    def _on(self, sub: Sub, msg: Any) -> None:
        now = time.monotonic()
        with sub.cond:
            if sub.count and now > sub.t:
                sub.hz = 0.9 * sub.hz + 0.1 / (now - sub.t) if sub.hz else 1.0 / (now - sub.t)
            sub.msg, sub.t, sub.count = msg, now, sub.count + 1
            sub.cond.notify_all()
        for fn in list(sub.listeners):
            fn(msg)

    def unlisten(self, topic: str, listener: Callable[[Any], None]) -> None:
        sub = self._subs.get(canonical(topic))
        if sub is not None and listener in sub.listeners:
            sub.listeners.remove(listener)

    def latest(self, topic: str, *, max_age: float | None = None) -> tuple[Any, float] | None:
        """(message, age_s) of the newest sample, or None if none / older than ``max_age``."""
        sub = self._subs.get(canonical(topic))
        if sub is None or sub.msg is None:
            return None
        age = time.monotonic() - sub.t
        return None if max_age is not None and age > max_age else (sub.msg, age)

    def wait_newer(self, topic: str, after: int, timeout: float, pred: Callable[[Any], bool] | None = None) -> Any:
        """The first message with count > ``after`` (and ``pred`` true), or None after ``timeout``."""
        sub = self._subs[canonical(topic)]
        end = time.monotonic() + timeout
        with sub.cond:
            while True:
                if sub.count > after and sub.msg is not None and (pred is None or pred(sub.msg)):
                    return sub.msg
                left = end - time.monotonic()
                if left <= 0:
                    return None
                sub.cond.wait(left)

    def wait_first(self, topic: str, type: str, timeout: float, *, qos: str = "auto") -> Any:
        sub = self.subscribe(topic, type, qos=qos)
        return self.wait_newer(sub.topic, 0, timeout)

    def publisher(self, topic: str, type: str, *, qos: str = "reliable") -> Any:
        topic = canonical(topic)
        self.start()
        key = (topic, type, qos)
        with self._endpoint_lock:
            pub = self._pubs.get(key)
            if pub is None:
                pub = self.node.create_publisher(message_class(type), topic, _qos.profile(qos))
                self._pubs[key] = pub
        return pub

    def publish(self, topic: str, type: str, msg: Any, *, qos: str = "reliable") -> None:
        self.publisher(topic, type, qos=qos).publish(msg)

    def claim(self, topic: str, owner: str) -> None:
        """Mark ``topic`` as streamed by ``owner``; a second owner is refused (two writers fight the arbiter)."""
        topic = canonical(topic)
        held = self.owners.get(topic)
        if held is not None and held != owner:
            raise RuntimeError(f"{topic} is already streamed by {held}; one writer per command channel")
        self.owners[topic] = owner

    def release(self, topic: str, owner: str) -> None:
        if self.owners.get(canonical(topic)) == owner:
            del self.owners[canonical(topic)]

    def graph(self) -> dict[str, Any]:
        """Topics, services (name -> types) and nodes as the graph reports them right now."""
        n = self.start().node
        topics = {name: list(types) for name, types in n.get_topic_names_and_types()}
        services = {name: list(types) for name, types in n.get_service_names_and_types()}
        nodes = [f"{ns.rstrip('/')}/{name}" for name, ns in n.get_node_names_and_namespaces()]
        actions = sorted({s[: -len("/_action/send_goal")] for s in services if s.endswith("/_action/send_goal")})
        return {"topics": topics, "services": services, "nodes": sorted(nodes), "actions": actions}

    def stats(self) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        return {
            s.topic: {
                "type": s.type,
                "qos": s.qos,
                "count": s.count,
                "hz": round(s.hz, 2),
                "age_s": None if not s.count else round(now - s.t, 3),
            }
            for s in self._subs.values()
        }

    def heal(self, silent_s: float = 8.0) -> list[str]:
        """Re-create subscriptions that had traffic and went quiet for ``silent_s`` (rmw_zenoh deaf subs)."""
        now, healed = time.monotonic(), []
        with self._endpoint_lock:
            for s in self._subs.values():
                if s.count and now - s.t > silent_s:
                    self.node.destroy_subscription(s.handle)
                    s.handle = self.node.create_subscription(
                        message_class(s.type), s.topic, lambda m, s=s: self._on(s, m), _qos.profile(s.qos)
                    )
                    healed.append(s.topic)
        return healed

    def close(self) -> None:
        """Stop the spin thread, destroy the node, shut the context down only if this bus made it."""
        self._stop.set()
        if getattr(self, "_at_exit", False):
            import atexit

            atexit.unregister(self.close)
            self._at_exit = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            if self.node is not None:
                self.node.destroy_node()
        finally:
            if self._owns_context:
                self._rclpy.try_shutdown()  # the module we started with (a test may have swapped sys.modules since)
                self._owns_context = False
            self.node, self._subs, self._pubs = None, {}, {}
            if RosBus._shared is self:
                RosBus._shared = None
