"""The one writer of a robot's command topic, and the "never stall the stream" rule.

While the robot is armed, the stream publishes on every tick: one command per state frame (``trigger: state``,
frame-synced, for robots that expect an answer to each sensor frame; ``command.echo`` copies its id/stamps) or a
fixed-rate timer (``trigger: timer``, e.g. the G1's 500 Hz). What it sends:

- a fresh target (newer than ``target_max_age_ms``): that target;
- a late target: the last target again (counted as ``late``), the policy is slow but the robot is held;
- no target for ``deadman_ms`` (run loop died, agent gone): HOLD, the measured pose at that moment,
  with the hold gains. Never a damp: on a standing humanoid a damp is a fall;
- an exception building the frame: HOLD, counted in ``errors``.

Not armed: nothing is published and targets are dropped; the robot's own controller owns the body.
Shadow: frames are built and compared with a reference command topic (the vendor's own controller), never
published; ``stats["shadow_max_err"]`` is the largest joint gap seen.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ....ros.bus import RosBus
from .codec import Codec

LOG = logging.getLogger(__name__)


class CommandStream:
    def __init__(
        self,
        bus: RosBus,
        codec: Codec,
        cfg: dict[str, Any],
        *,
        owner: str,
        sender: str,
        gains: Callable[[], tuple[np.ndarray | None, np.ndarray | None]],
        hold_gains: Callable[[], tuple[np.ndarray | None, np.ndarray | None]],
        armed: Callable[[Any], tuple[bool, str]],
        measured: Callable[[Any], np.ndarray],
        latest_state: Callable[[], Any],
        shadow: dict[str, Any] | None = None,
    ) -> None:
        self.bus, self.codec, self.cfg, self.owner, self.sender = bus, codec, cfg, owner, sender
        self.gains, self.hold_gains, self.armed, self.measured, self.latest_state = (
            gains,
            hold_gains,
            armed,
            measured,
            latest_state,
        )
        self.topic, self.type, self.qos = cfg["topic"], cfg["type"], cfg.get("qos", "reliable")
        self.max_age = float(cfg.get("target_max_age_ms", 100)) / 1000.0
        self.deadman = float(cfg.get("deadman_ms", 300)) / 1000.0
        self.shadow = shadow
        self._lock = threading.Lock()
        self._target: np.ndarray | None = None
        self._t_target = 0.0
        self._hold: np.ndarray | None = None
        self._freeze = False
        self._running = False
        self._thread: threading.Thread | None = None
        self.stats: dict[str, Any] = {
            "sent": 0,
            "late": 0,
            "held": 0,
            "errors": 0,
            "skipped_unarmed": 0,
            "shadow_max_err": 0.0,
            "last_reason": "",
        }
        self._vendor: np.ndarray | None = None

    def start(self) -> None:
        if self._running:
            return
        if not self.shadow:
            self.bus.claim(self.topic, self.owner)
            self.bus.publisher(self.topic, self.type, qos=self.qos)
        else:
            self.bus.subscribe(
                self.shadow["topic"],
                self.shadow["type"],
                qos=self.shadow.get("qos", "reliable"),
                listener=self._on_vendor,
            )
        self._running = True
        if self.cfg.get("trigger", "timer") == "state":
            self.bus.subscribe(self.cfg["state_topic"], self.cfg["state_type"], listener=self._on_state)
        else:
            hz = float(self.cfg.get("rate_hz", 50))
            self._thread = threading.Thread(target=self._timer, args=(1.0 / hz,), name=f"{self.owner}-cmd", daemon=True)
            self._thread.start()

    def set_target(self, q: np.ndarray) -> None:
        if self._thread is not None and not self._thread.is_alive():  # a dead stream is never a quiet one
            raise ConnectionError(
                f"ros: the {self.topic} command stream stopped ({self.stats['last_reason']}); reconnect"
            )
        with self._lock:
            self._target, self._t_target, self._freeze, self._hold = (
                np.array(q, dtype=float),
                time.monotonic(),
                False,
                None,
            )

    def freeze(self) -> None:
        """Hold the measured pose from now until a new target arrives (hold(), estop(), deadman)."""
        with self._lock:
            self._freeze, self._hold = True, None

    def _timer(self, dt: float) -> None:
        nxt = time.monotonic()
        while self._running:
            try:
                self._tick(self.latest_state())
            except Exception as e:  # noqa: BLE001 - the timer is the only writer: it must outlive any one tick
                self._failed("tick", e)
            nxt += dt
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()  # behind: re-phase, never burst

    def _on_state(self, frame: Any) -> None:
        if self._running:
            self._tick(frame)

    def _tick(self, frame: Any) -> None:
        if frame is None:
            return
        try:
            ok, why = self.armed(frame)
            if not ok:
                with self._lock:
                    self._target, self._hold = None, None
                self.stats["skipped_unarmed"] += 1
                self.stats["last_reason"] = why
                return
            q, kp, kd = self._choose(frame)
            msg = self.codec.echo(self.codec.build(q, kp, kd, frame, self.sender), frame)
        except Exception as e:  # noqa: BLE001 - never stall: fall back to holding the measured pose
            self.stats["errors"] += 1
            self.stats["last_reason"] = f"{type(e).__name__}: {e}"
            try:
                msg = self._hold_frame(frame)
            except Exception:  # noqa: BLE001 - nothing safe to send; the robot's own timeout takes over
                LOG.exception("ros command stream: cannot build even a hold frame")
                return
        try:
            if self.shadow:
                self._compare(msg)
                return
            self.bus.publish(self.topic, self.type, msg, qos=self.qos)
            self.stats["sent"] += 1
        except Exception as e:  # noqa: BLE001 - an rmw error drops this frame, never the stream
            self._failed("publish", e)

    def _failed(self, what: str, e: Exception) -> None:
        self.stats["errors"] += 1
        self.stats["last_reason"] = f"{what}: {type(e).__name__}: {e}"
        if self.stats["errors"] == 1 or self.stats["errors"] % 500 == 0:  # once, then every 500th: never a log flood
            LOG.warning("ros command stream on %s: %s failed: %s", self.topic, what, e)

    def _choose(self, frame: Any) -> tuple[np.ndarray, Any, Any]:
        now = time.monotonic()
        with self._lock:
            target, age, freeze = self._target, now - self._t_target, self._freeze
        if target is None or freeze or age > self.deadman:
            self.stats["held"] += 1
            return (*self._hold_pose(frame),)
        if age > self.max_age:
            self.stats["late"] += 1
        kp, kd = self.gains()
        return target, kp, kd

    def _hold_pose(self, frame: Any) -> tuple[np.ndarray, Any, Any]:
        with self._lock:
            if self._hold is None:
                self._hold = np.array(self.measured(frame), dtype=float)
            hold = self._hold
        kp, kd = self.hold_gains()
        return hold, kp, kd

    def _hold_frame(self, frame: Any) -> Any:
        q, kp, kd = self._hold_pose(frame)
        return self.codec.echo(self.codec.build(q, kp, kd, frame, self.sender), frame)

    def _on_vendor(self, msg: Any) -> None:
        self._vendor = self.codec.positions(msg)

    def _compare(self, msg: Any) -> None:
        ours = self.codec.positions(msg)
        if ours is not None and self._vendor is not None and ours.shape == self._vendor.shape:
            err = float(np.nanmax(np.abs(ours - self._vendor)))
            self.stats["shadow_max_err"] = max(self.stats["shadow_max_err"], err)
        self.stats["shadow_frames"] = self.stats.get("shadow_frames", 0) + 1

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cfg.get("trigger") == "state":
            self.bus.unlisten(self.cfg["state_topic"], self._on_state)
        self.bus.release(self.topic, self.owner)
