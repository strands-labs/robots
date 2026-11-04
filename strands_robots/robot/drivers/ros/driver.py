"""RosDriver: any ROS 2 robot as a strands-robots Driver, driven by a ROS profile (data, not code).

``Robot("unitree_g1", driver="ros", host=...)`` or ``ros.connect("ros2_control", host=...)``. Reads come
from the bus cache (never block), writes set the target of the one CommandStream that owns the command
topic, and the stream keeps the robot fed at its own rate. Arming is the robot's (the profile's ``status.armed``
fields, a released vendor controller): this driver checks it on every frame and refuses writes when it is not.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

from ....description import Description
from ....ros import profile as _profile
from ....ros.bus import RosBus
from ....ros.discovery import Discovery
from ....types import JointState, Pose
from ..calibration import Calibration
from .codec import codec_class, gains_vector
from .images import to_rgb
from .stream import CommandStream


class RosDriver:
    calibration: Calibration | None = None
    resolution: int | None = None

    def __init__(
        self,
        description: Description,
        *,
        port: str | None = None,
        cameras: Mapping[str, Any] | None = None,
        host: str | None = None,
        profile: str | None = None,
        domain_id: int | None = None,
        shadow: bool = False,
        released: bool = False,
        kp: Any = None,
        kd: Any = None,
        timeout: float = 5.0,
        overrides: Mapping[str, Any] | None = None,
        client: RosBus | None = None,
    ) -> None:
        """``host``: the robot (None = this machine). ``profile``: a built-in name or a .json path (default: the
        robot's name). ``shadow``: compute commands, publish none, compare with the profile's ``shadow`` topic.
        ``released``: you released a vendor motion controller that would fight low-level commands (profiles with
        ``command.requires_release``, e.g. the G1). ``kp``/``kd``: per-joint gains in action_keys
        order (default: the policy's, then the robot's module defaults, then the profile's)."""
        self.description, self.names = description, tuple(description.action_keys)
        self.profile = _profile.load(profile or description.name, overrides=overrides)
        d = dict(self.profile.discovery)
        self.bus = client or RosBus.shared(
            Discovery(
                host=host or port,
                rmw=d.get("rmw"),
                domain_id=domain_id if domain_id is not None else d.get("domain_id"),
                zenoh_port=int(d.get("zenoh_port", 7447)),
                iface=d.get("iface"),
            )
        )
        self.codec = codec_class(self.profile)(self.profile, self.names)
        self.shadow, self.released = bool(shadow), bool(released)
        self.sender = f"strands-{uuid.uuid4().hex[:8]}"
        self.supports_position_hold = self.profile.family == "joint_state"  # a controller holds; others need us
        self._kp, self._kd = self._gain_arg(kp, "kp"), self._gain_arg(kd, "kd")
        self._policy_gains: tuple[np.ndarray, np.ndarray] | None = None
        self._lo = np.array([description.limits.get(n, (-np.inf, np.inf))[0] for n in self.names])
        self._hi = np.array([description.limits.get(n, (-np.inf, np.inf))[1] for n in self.names])
        self._subscribe(cameras or {})
        st = self.profile.need("state")
        if self.bus.wait_newer(st["topic"], 0, timeout) is None:
            raise ConnectionError(
                f"ros: no {st['type']} on {st['topic']} within {timeout}s. Check discovery (host={host or port!r}, "
                f"rmw={d.get('rmw')}), that the robot publishes it (`ros2 topic hz {st['topic']}`), "
                f"and QoS ({st.get('qos')})."
            )
        self.stream: CommandStream | None = None
        if self.profile.stream("command"):
            self._make_stream()

    def _gain_arg(self, v: Any, what: str) -> np.ndarray | None:
        return gains_vector(v, len(self.names), what)

    def _subscribe(self, cameras: Mapping[str, Any]) -> None:
        for key in ("state", "status", "odom", "gains_from", "telemetry"):
            s = self.profile.stream(key) if key != "gains_from" else self.profile.data.get("gains_from")
            if s:
                self.bus.subscribe(s["topic"], s["type"], qos=s.get("qos", "auto"))
        self._cams = {**self.profile.cameras, **{k: v for k, v in cameras.items() if isinstance(v, Mapping)}}
        for cam in self._cams.values():
            self.bus.subscribe(cam["topic"], cam["type"], qos=cam.get("qos", "sensor"))

    def _make_stream(self) -> None:
        cfg = dict(self.profile.need("command"))
        st = self.profile.need("state")
        cfg.update(state_topic=st["topic"], state_type=st["type"])
        self.stream = CommandStream(
            self.bus,
            self.codec,
            cfg,
            owner=f"{self.description.name}:{self.sender}",
            sender=self.sender,
            gains=self._gains,
            hold_gains=self._hold_gains,
            armed=self._armed,
            measured=lambda frame: self.codec.read(frame).positions,
            latest_state=self._latest_state,
            shadow=dict(self.profile.need("shadow")) if self.shadow else None,
        )
        self.stream.start()

    def _latest(self, key: str) -> Any:
        s = self.profile.data.get(key)
        if not s:
            return None
        age = s.get("max_age_ms")
        got = self.bus.latest(s["topic"], max_age=None if age is None else float(age) / 1000.0)
        return None if got is None else got[0]

    def _latest_state(self) -> Any:
        return self._latest("state")

    def _state(self) -> Any:
        msg = self._latest("state")
        if msg is None:
            st = self.profile.need("state")
            raise ConnectionError(f"ros: {st['topic']} went silent (no sample within {st.get('max_age_ms')} ms)")
        return msg

    def _armed(self, frame: Any) -> tuple[bool, str]:
        ok, why = self.codec.armed(self._latest("status"), frame)
        if ok and self.profile.need("command").get("requires_release") and not self.released:
            return False, "not armed: pass released=True once the vendor's motion controller is released"
        return ok, why

    def _gains(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._policy_gains is not None:
            return self._policy_gains
        if self._kp is not None and self._kd is not None:
            return self._kp, self._kd
        return self._hold_gains()

    def _hold_gains(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        info = self._latest("gains_from")
        g = self.codec.default_gains(info) if info is not None else None
        g = g or self.codec.default_gains(None)
        if g is not None:
            return g
        return (self._kp, self._kd)

    def set_gains(self, kp: Any, kd: Any) -> None:
        """kp/kd per joint in action_keys order (``Robot.set_gains``, a policy's stiffness/damping); the next frame
        sends them. ``hold()``, ``estop()`` and the deadman still hold with the robot's own default gains."""
        kp_, kd_ = self._gain_arg(kp, "kp"), self._gain_arg(kd, "kd")
        if kp_ is None or kd_ is None:
            raise ValueError("set_gains needs both kp and kd")
        self._policy_gains = (kp_, kd_)

    def read(self) -> JointState:
        return self.codec.read(self._state())

    def write(self, targets: np.ndarray) -> None:
        q = np.asarray(targets, dtype=float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError(f"ros: targets must be {len(self.names)} finite radians; nothing was sent")
        if ((q < self._lo - 1e-9) | (q > self._hi + 1e-9)).any():
            bad = [n for n, v, lo, hi in zip(self.names, q, self._lo, self._hi, strict=True) if not lo <= v <= hi]
            raise ValueError(f"ros: targets outside joint limits {bad}; the whole frame was refused")
        if self.stream is None:
            raise NotImplementedError(f"ROS profile {self.profile.name!r} has no command stream: observe-only")
        ok, why = self._armed(self._state())
        if not ok and not self.shadow:
            raise PermissionError(f"{self.description.name}: {why}; nothing was sent")
        if self.codec.needs_gains and any(
            g is None for g in (*self._gains(), *self._hold_gains())
        ):  # a stream that cannot build a hold frame would stall
            raise PermissionError(
                f"{self.description.name}: no kp/kd known yet (policy, robot module info or kp=/kd=); nothing was sent"
            )
        self.stream.set_target(q)

    def pose(self) -> Pose | None:
        return self.codec.pose(self._latest("state"), self._latest("odom"))

    def images(self) -> Mapping[str, np.ndarray]:
        out = {}
        for name, cam in self._cams.items():
            got = self.bus.latest(cam["topic"], max_age=float(cam.get("max_age_ms", 1000)) / 1000.0)
            img = None if got is None else to_rgb(got[0])
            if img is not None:
                out[name] = img
        return out

    def extras(self) -> Mapping[str, Any]:
        frame, status = self._latest("state"), self._latest("status")
        out = dict(self.codec.extras(frame, status))
        if frame is not None:
            out["armed"], out["armed_reason"] = self._armed(frame)
        if self.stream is not None:
            out["stream"] = dict(self.stream.stats, shadow=self.shadow)
        return out

    def hold(self) -> None:
        if self.stream is not None:
            self.stream.freeze()

    def estop(self) -> None:
        """Hold the measured pose, then send the profile's STOP (zero velocity, nav stop). Never damps."""
        if self.stream is not None:
            self.stream.freeze()
        from .stop import send_stop

        send_stop(self.bus, self.profile, self.sender)

    def calibrate(self, *, interactive: bool = True) -> Calibration:
        raise RuntimeError("ROS robots are calibrated by their vendor stack; nothing to calibrate here")

    def close(self) -> None:
        """Stop our stream. If the robot is still armed it is told so: a low-level robot faults without us."""
        if self.stream is not None:
            frame = self._latest("state")
            arms = self.profile.stream("status") or self.profile.need("command").get("requires_release")
            if arms and frame is not None and self._armed(frame)[0] and not self.shadow:
                import logging

                logging.getLogger(__name__).warning(
                    "%s is still armed; stopping the command stream now trips its command timeout. "
                    "Hand control back to the robot's own controller (or hang it on a gantry) first.",
                    self.description.name,
                )
            self.stream.stop()
