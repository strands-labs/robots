"""Input device streaming over the mesh - publish and receive teleoperator actions.

Enables remote teleoperation: a leader arm on machine A publishes its joint
positions via :class:`InputPublisher`, and the follower arm on machine B
receives and applies them via :class:`InputReceiver`.

Topic schema for ``strands/{peer_id}/input/{device_name}``::

    {
        "peer_id": "<publisher-peer-id>",
        "device": "<device-name>",
        "method": "arm" | "gamepad" | "keyboard" | "phone",
        "t": <unix-timestamp>,
        "seq": <monotonic-frame-counter>,
        "action": {"motor.pos": float, ...},
        "events": {"terminate_episode": bool, ...} | null
    }
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from strands_robots.mesh.security import ValidationError, validate_input_frame

_log_safety_event: Callable[..., None] | None
try:  # audit is best-effort; never let an import issue break teleop apply
    from strands_robots.mesh.audit import log_safety_event as _log_safety_event
except Exception:  # pragma: no cover - defensive
    _log_safety_event = None

if TYPE_CHECKING:
    from strands_robots.mesh.core import Mesh

logger = logging.getLogger(__name__)

INPUT_HZ_DEFAULT = 50.0

#: Default ceiling on the rate at which an InputReceiver will
#: APPLY inbound teleop frames to the robot. The publisher streams at
#: INPUT_HZ_DEFAULT (50Hz); a malicious peer can stream far faster to slam
#: the servos (overcurrent / thermal / gear-strip). Frames arriving faster
#: than this cap are dropped-and-counted (``_rate_dropped``). Generous 2x
#: headroom over the default publish rate so legitimate jitter is never
#: rejected. Operator-tunable via ``STRANDS_MESH_INPUT_MAX_HZ`` (0 disables
#: the cap for trusted closed networks).
INPUT_MAX_HZ_DEFAULT = 100.0


def _input_max_hz() -> float:
    """Resolve ``STRANDS_MESH_INPUT_MAX_HZ`` (lazy, restart-free).

    Bad / missing input falls back to the default ceiling; an explicit
    non-positive value (0) disables the cap for trusted closed networks.
    """
    import os

    raw = os.getenv("STRANDS_MESH_INPUT_MAX_HZ")
    if raw is None:
        return INPUT_MAX_HZ_DEFAULT
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return INPUT_MAX_HZ_DEFAULT
    if val < 0:
        return INPUT_MAX_HZ_DEFAULT
    return val  # 0 => disabled


#: M-5: the teleop input path is high-rate (up to 50Hz),
#: so we cannot audit every applied frame without flooding the log. Instead we
#: record one ``input_stream_applied`` audit event every N applied frames (a
#: heartbeat that proves the stream was live + actuating, for post-incident
#: forensics of the "Invisible Puppeteer" chain). Operator-tunable via
#: ``STRANDS_MESH_INPUT_AUDIT_EVERY`` (0 disables input audit entirely).
INPUT_AUDIT_EVERY_DEFAULT = 100


def _input_audit_every() -> int:
    """Resolve ``STRANDS_MESH_INPUT_AUDIT_EVERY`` (lazy, restart-free).

    Bad/missing input falls back to the default sampling interval; an
    explicit non-positive value disables input-stream auditing.
    """
    import os

    raw = os.getenv("STRANDS_MESH_INPUT_AUDIT_EVERY")
    if raw is None:
        return INPUT_AUDIT_EVERY_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return INPUT_AUDIT_EVERY_DEFAULT
    return val if val > 0 else 0


class InputPublisher:
    """Publishes teleoperator actions to the mesh at a fixed rate.

    Runs in a background thread, polling the teleoperator and publishing
    normalized action dicts.
    """

    def __init__(
        self,
        mesh: Mesh,
        teleoperator: Any,
        device_name: str = "leader",
        method: str = "arm",
        hz: float = INPUT_HZ_DEFAULT,
    ) -> None:
        self.mesh = mesh
        self.teleoperator = teleoperator
        self.device_name = device_name
        self.method = method
        self.hz = hz
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._seq = 0
        self._error_count = 0
        self._frame_count = 0
        self._start_time = 0.0

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"InputPublisher(device={self.device_name!r}, method={self.method!r}, {state})"

    @property
    def topic(self) -> str:
        return f"strands/{self.mesh.peer_id}/input/{self.device_name}"

    @property
    def stats(self) -> dict[str, Any]:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "device": self.device_name,
            "method": self.method,
            "running": self._running,
            "frames": self._frame_count,
            "errors": self._error_count,
            "hz_actual": self._frame_count / elapsed if elapsed > 0 else 0,
            "hz_target": self.hz,
        }

    def start(self) -> None:
        """Start the input publishing loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._publish_loop,
            name=f"mesh-input-{self.device_name}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[mesh] input publisher started: %s (%s @ %.0fHz)",
            self.device_name,
            self.method,
            self.hz,
        )

    def stop(self) -> dict[str, Any]:
        """Stop the input publishing loop and return stats."""
        if not self._running:
            return self.stats
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info(
            "[mesh] input publisher stopped: %s (%d frames)",
            self.device_name,
            self._frame_count,
        )
        return self.stats

    def _publish_loop(self) -> None:
        period = 1.0 / self.hz
        while self._running and not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                action = self.teleoperator.get_action()
                action_dict = self._normalize_action(action)

                events = None
                if hasattr(self.teleoperator, "get_teleop_events"):
                    try:
                        events = self.teleoperator.get_teleop_events()
                    except Exception:
                        pass

                payload = {
                    "peer_id": self.mesh.peer_id,
                    "device": self.device_name,
                    "method": self.method,
                    "t": time.time(),
                    "seq": self._seq,
                    "action": action_dict,
                    "events": events,
                }
                # Route through Mesh.publish() -- the documented single
                # publish chokepoint -- so this teleop actuation stream is
                # covered by any audit/telemetry/compression hook landing
                # there, exactly like sensor/state/command publishers. The
                # receiver side already goes through self.mesh.subscribe().
                self.mesh.publish(self.topic, payload)
                self._seq += 1
                self._frame_count += 1
            except Exception as exc:
                self._error_count += 1
                if self._error_count <= 5:
                    logger.warning("[mesh] input publish error (%s): %s", self.device_name, exc)

            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    @staticmethod
    def _normalize_action(action: Any) -> dict[str, float]:
        """Convert action from any teleoperator format to a flat dict."""
        if isinstance(action, dict):
            result = {}
            for k, v in action.items():
                if hasattr(v, "item"):
                    result[k] = float(v.item())
                elif isinstance(v, (int, float)):
                    result[k] = float(v)
                else:
                    result[k] = float(v)
            return result
        elif hasattr(action, "tolist"):
            arr = action.tolist()
            return {f"j{i}": float(v) for i, v in enumerate(arr)}
        else:
            return {"raw": float(action)}


class InputReceiver:
    """Subscribes to a remote peer's input stream and applies actions locally.

    Listens on ``strands/{source_peer_id}/input/{device_name}`` and calls
    ``robot.send_action(action)`` for each received frame.
    """

    def __init__(
        self,
        mesh: Mesh,
        robot: Any,
        source_peer_id: str,
        device_name: str = "leader",
        apply_fn: Callable[[Any, dict[str, float]], None] | None = None,
    ) -> None:
        self.mesh = mesh
        self.robot = robot
        self.source_peer_id = source_peer_id
        self.device_name = device_name
        self._apply_fn = apply_fn or self._default_apply
        self._running = False
        self._sub_name: str | None = None
        self._frame_count = 0
        self._error_count = 0
        self._last_seq = -1
        self._drops = 0
        self._rejected = 0
        self._rate_dropped = 0
        self._last_apply_mono = 0.0
        self._start_time = 0.0

    def __repr__(self) -> str:
        state = "running" if self._running else "stopped"
        return f"InputReceiver(source={self.source_peer_id!r}, device={self.device_name!r}, {state})"

    @property
    def topic(self) -> str:
        return f"strands/{self.source_peer_id}/input/{self.device_name}"

    @property
    def stats(self) -> dict[str, Any]:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "source": self.source_peer_id,
            "device": self.device_name,
            "running": self._running,
            "frames_received": self._frame_count,
            "errors": self._error_count,
            "drops": self._drops,
            "rejected": self._rejected,
            "rate_dropped": self._rate_dropped,
            "hz_actual": self._frame_count / elapsed if elapsed > 0 else 0,
        }

    def start(self) -> None:
        """Start receiving input actions from the remote peer."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._sub_name = self.mesh.subscribe(
            self.topic,
            callback=self._on_input,
            name=f"input:{self.source_peer_id}/{self.device_name}",
        )
        if self._sub_name:
            logger.info(
                "[mesh] input receiver started: %s from %s",
                self.device_name,
                self.source_peer_id,
            )
        else:
            logger.warning("[mesh] input receiver failed to subscribe: %s", self.topic)
            self._running = False

    def stop(self) -> dict[str, Any]:
        """Stop receiving and return stats."""
        if not self._running:
            return self.stats
        self._running = False
        if self._sub_name:
            self.mesh.unsubscribe(self._sub_name)
        logger.info(
            "[mesh] input receiver stopped: %d frames from %s",
            self._frame_count,
            self.source_peer_id,
        )
        return self.stats

    def _on_input(self, topic: str, data: dict[str, Any]) -> None:
        if not self._running:
            return
        # E-stop lockout MUST gate the teleop input path the
        # same way it gates the command path (see Mesh._dispatch). Without
        # this check a LAN-adjacent peer could keep driving the follower's
        # joints via send_action() while an operator believes the robot is
        # safely locked out -- the "Safe Mode Illusion" / "Oscillation Kill"
        # exploit chains. The CMD path raises LockoutError; the input path is
        # a high-rate streaming loop, so we drop-and-count instead of raising
        # to avoid log/exception spam at 50Hz. Rejected frames are surfaced
        # via the ``rejected`` stat and a rate-limited warning.
        lockout = getattr(self.mesh, "_estop_lockout", None)
        if lockout is not None and lockout.is_set():
            self._rejected = getattr(self, "_rejected", 0) + 1
            if self._rejected <= 5:
                logger.warning(
                    "[mesh] input frame rejected during E-stop lockout from %s",
                    self.source_peer_id,
                )
            return
        try:
            action = data.get("action")
            if action is None:
                return
            seq = data.get("seq", 0)
            if self._last_seq >= 0 and seq > self._last_seq + 1:
                self._drops += seq - self._last_seq - 1
            self._last_seq = seq

            # Apply-rate ceiling. A peer streaming teleop far
            # above the nominal publish rate can slam servos into overcurrent
            # / thermal / gear damage. Enforce a minimum inter-apply interval
            # using a monotonic clock (immune to wall-clock/NTP skew). Frames
            # over the cap are dropped-and-counted (``_rate_dropped``) rather
            # than raising -- this is a 50Hz+ hot loop. 0 disables the cap.
            max_hz = _input_max_hz()
            if max_hz > 0:
                now_mono = time.perf_counter()
                min_interval = 1.0 / max_hz
                if self._last_apply_mono and (now_mono - self._last_apply_mono) < min_interval:
                    self._rate_dropped = getattr(self, "_rate_dropped", 0) + 1
                    if self._rate_dropped <= 5:
                        logger.warning(
                            "[mesh] input frame rate-limited from %s (> %.0fHz)",
                            self.source_peer_id,
                            max_hz,
                        )
                    return
                self._last_apply_mono = now_mono
            # B-04 / F-02: validate the teleop frame before it reaches
            # send_action(). A LAN-adjacent peer that discovers this
            # source peer_id could otherwise drive the follower's joints
            # directly with unbounded / non-finite values. validate_input_frame
            # bounds key count, key charset, and clamps each value to a
            # finite magnitude. Rejected frames are counted + logged and
            # dropped (never applied) rather than crashing the receiver.
            try:
                safe_action = validate_input_frame(action)
            except ValidationError as verr:
                self._rejected = getattr(self, "_rejected", 0) + 1
                if self._rejected <= 5:
                    logger.warning(
                        "[mesh] input frame rejected from %s: %s",
                        self.source_peer_id,
                        verr,
                    )
                return
            self._apply_fn(self.robot, safe_action)
            self._frame_count += 1
            # M-5: sampled positive audit of the live teleop stream so a
            # successful remote actuation is not invisible to forensics.
            _audit_every = _input_audit_every()
            if _log_safety_event is not None and _audit_every > 0 and self._frame_count % _audit_every == 0:
                try:
                    _log_safety_event(
                        "input_stream_applied",
                        getattr(self.mesh, "peer_id", "?"),
                        {
                            "source": self.source_peer_id,
                            "device": self.device_name,
                            "frames": self._frame_count,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    logger.debug("[mesh] input audit unavailable: %s", audit_exc)
        except Exception as exc:
            self._error_count += 1
            if self._error_count <= 5:
                logger.warning("[mesh] input apply error: %s", exc)

    @staticmethod
    def _default_apply(robot: Any, action: dict[str, float]) -> None:
        """Default: calls robot.send_action()."""
        if hasattr(robot, "send_action"):
            robot.send_action(action)
        elif hasattr(robot, "robot") and hasattr(robot.robot, "send_action"):
            robot.robot.send_action(action)
