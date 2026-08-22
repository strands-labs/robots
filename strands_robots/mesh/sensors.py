"""Extended sensor topic publishing loops for the Mesh.

These loops are started conditionally by Mesh.start() and only publish when
the robot exposes the relevant attribute. Zero-cost when unused.

Topics published:
- strands/{peer_id}/pose - SE(3) from SLAM/odometry/VIO
- strands/{peer_id}/health - Battery, CPU, memory, disk, temps
- strands/{peer_id}/imu - Roll/pitch/yaw, gyro, accel
- strands/{peer_id}/odom - Dead-reckoning odometry
- strands/{peer_id}/lidar/summary - Point cloud stats
- strands/{peer_id}/lidar/state - Sensor state
- strands/{peer_id}/hand/{name}/state - End-effector joints/force
- strands/{peer_id}/map/info - Map metadata
- strands/{peer_id}/safety/event - On-demand safety events
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from strands_robots.bus_access import read_observation
from strands_robots.mesh.audit import log_safety_event
from strands_robots.mesh.pacing import Ticker
from strands_robots.mesh.session import (
    HAND_HZ,
    HEALTH_HZ,
    IMU_HZ,
    LIDAR_STATE_HZ,
    LIDAR_SUMMARY_HZ,
    MAP_INFO_HZ,
    ODOM_HZ,
    POSE_HZ,
    hz_from_env,
    put,  # noqa: F401  # re-exported so test fixtures can patch.object(sensors, "put")
)

logger = logging.getLogger(__name__)


def _quat_wxyz_from_rotmat(mat: Any) -> list[float]:
    """Rotation matrix -> unit quaternion, scalar-first ``[w, x, y, z]``.

    Shepperd's method: take the trace branch when the trace is positive,
    otherwise branch on the largest diagonal term. All four branches are needed.
    The trace branch alone divides by ``sqrt(trace + 1)``, which goes to zero as
    the trace approaches ``-1``, so it cannot serve the ``trace <= 0`` half of
    the domain - and that half is not an edge case. A rotation's trace is
    ``1 + 2 * cos(angle)``, so ``trace <= 0`` is exactly ``angle >= 120
    degrees``, which is 61% of SO(3) under the uniform measure: a robot that has
    turned to face back the way it came is in it. Substituting the identity
    quaternion there reports such a robot as one that has not turned at all, and
    reports it as a valid unit quaternion, so no consumer can tell.

    The result is normalized, so a matrix that is only approximately orthonormal
    - a pose integrated from odometry or handed over by a SLAM stack - still
    yields a unit quaternion rather than one whose length quietly carries the
    input's drift onto the wire. It is then sign-canonicalized to ``w >= 0``:
    ``q`` and ``-q`` encode the same rotation, so the canonical sign is safe for
    every consumer and keeps repeated reads of an unchanged pose identical.

    The MuJoCo and Isaac backends hold the same rule, and this is deliberately a
    third copy rather than an import: :mod:`strands_robots.mesh` must not depend
    on :mod:`strands_robots.simulation`, and the shared-domain module those two
    layers do have in common (:mod:`strands_robots.utils`) is the refusal-guard
    module, which holds a scanned invariant that no function in it converts a
    caller's value outside a ``try``. Giving all three one owner is a refactor
    of two already-correct implementations, separate from reporting a pose.

    Args:
        mat: A rotation matrix, or an SE(3) transform whose leading 3x3 block is
            one (the translation row and column are not read).

    Returns:
        ``[w, x, y, z]`` as plain floats: a unit quaternion with ``w >= 0``.
    """
    import numpy as np

    m = np.asarray(mat, dtype=np.float64)
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = float(np.sqrt(t + 1.0)) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(q)) or 1.0
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return [float(v) for v in q]


def _resolve_hz(env_name: str, default: float) -> float:
    """Read a publish rate (Hz) from the environment, falling back to default.

    Args:
        env_name: Environment variable holding the operator override.
        default: Rate to use when the variable is unset or unusable.

    Returns:
        The override when it names a rate this loop can pace itself with,
        *default* when the variable is unset or holds a value no loop can honor
        (including a non-finite one, which would otherwise make the caller's
        ``1.0 / hz`` period zero and busy-spin the publish loop), and ``0.0``
        for a non-positive override, which disables the loop.
    """
    hz, reason = hz_from_env(env_name)
    if reason is not None:
        logger.warning("%s; using default %.1f", reason, default)
        return default
    if hz is None:
        return default
    return hz if hz > 0 else 0.0


class SensorLoopsMixin:
    """Mixin providing all extended sensor publishing loops for Mesh.

    Requires the host class to have:
    - self.peer_id: str
    - self.robot: Any
    - self._running: bool
    - self._stop_event: threading.Event
    - self.publish(key, payload) -> None
    """

    # Type hints for attrs/methods provided by host class (Mesh).
    peer_id: str
    robot: Any
    _running: bool
    _stop_event: threading.Event

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        """Provided by host Mesh class. Publishes payload on the given key
        via the underlying Zenoh transport. Declared here so static
        type-checkers see the symbol on the mixin without duplicating logic.

        At runtime ``Mesh.publish`` shadows this stub via MRO
        (``class Mesh(SensorLoopsMixin)``); this body is never executed.

        Raises:
            NotImplementedError: if the mixin is used standalone (no host
                Mesh class). Replaces the bare ``...`` stub so static
                analysers (CodeQL #226) don't flag a no-effect statement.
        """
        raise NotImplementedError("SensorLoopsMixin.publish must be provided by a host class")

    def _paced(self, period: float) -> Iterator[None]:
        """Yield once per ``period`` seconds until the host stops.

        Every sensor loop in this mixin used to end with
        ``if self._stop_event.wait(period): break``. That wait is a delay where a
        rate needs a deadline: the time a read spends on a bus or in a driver was
        added to the period rather than subtracted from it, so the loop ran at
        ``1 / (period + read)`` while every consumer read the achieved rate as the
        sensor's own limit. :class:`~strands_robots.mesh.pacing.Ticker` paces on
        the selector timer instead, which treats the period as a deadline and
        still notices a stop within 10ms.

        Written as one generator rather than seven conversions on purpose: these
        loops differ only in what they read, so the pacing belongs in a single
        place where its ownership rules -- construct one ticker per loop, close it
        even when the body raises -- cannot be got right in six loops and wrong in
        the seventh.

        Args:
            period: Seconds per tick, forwarded to the ticker (which refuses a
                non-positive or non-finite value).

        Yields:
            Once per tick. The ticker is closed when the caller's ``for`` ends,
            breaks, or unwinds on an exception.
        """
        with Ticker(period, self._stop_event) as ticker:
            while self._running:
                yield
                if ticker.wait():
                    break

    # Pose

    def _pose_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_POSE_HZ", POSE_HZ)
        if hz <= 0:
            return
        period = 1.0 / hz
        for _ in self._paced(period):
            try:
                pose = self._read_pose()
                if pose:
                    self.publish(f"strands/{self.peer_id}/pose", pose)
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: pose tick error: %s", self.peer_id, exc)

    def _read_pose(self) -> dict[str, Any] | None:
        r = self.robot
        pose: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}

        # Explicit pose provider (highest priority)
        try:
            pose_data = getattr(r, "_pose", None)
            if pose_data is not None:
                if isinstance(pose_data, dict):
                    pose.update(pose_data)
                    pose.setdefault("source", "provider")
                    pose.setdefault("frame", "map")
                    return pose
                elif hasattr(pose_data, "shape") and getattr(pose_data, "shape", None) == (4, 4):
                    import numpy as np

                    mat = pose_data
                    pose["x"] = float(mat[0, 3])
                    pose["y"] = float(mat[1, 3])
                    pose["z"] = float(mat[2, 3])
                    pose["theta"] = float(np.arctan2(mat[1, 0], mat[0, 0]))
                    pose["quat"] = _quat_wxyz_from_rotmat(mat)
                    pose["source"] = "provider"
                    pose["frame"] = "map"
                    return pose
        except Exception:  # noqa: BLE001
            pass

        # SLAM pose
        try:
            slam_pose = getattr(r, "_slam_pose", None)
            if slam_pose is not None and isinstance(slam_pose, dict):
                pose.update(slam_pose)
                pose.setdefault("source", "slam")
                pose.setdefault("frame", "map")
                return pose
        except Exception:  # noqa: BLE001
            pass

        # Odometry pose
        try:
            odom_pose = getattr(r, "_odom_pose", None)
            if odom_pose is not None and isinstance(odom_pose, dict):
                pose.update(odom_pose)
                pose.setdefault("source", "odom")
                pose.setdefault("frame", "odom")
                return pose
        except Exception:  # noqa: BLE001
            pass

        return None

    # Health

    def _health_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_HEALTH_HZ", HEALTH_HZ)
        if hz <= 0:
            return
        period = 1.0 / hz
        for _ in self._paced(period):
            try:
                health = self._read_health()
                if health:
                    self.publish(f"strands/{self.peer_id}/health", health)
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: health tick error: %s", self.peer_id, exc)

    def _read_health(self) -> dict[str, Any] | None:
        r = self.robot
        health: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}
        has_data = False

        try:
            battery = getattr(r, "_battery", None)
            if battery is not None:
                if isinstance(battery, dict):
                    health["battery_pct"] = battery.get("pct", battery.get("percentage"))
                    health["charging"] = battery.get("charging", False)
                elif isinstance(battery, (int, float)):
                    health["battery_pct"] = float(battery)
                has_data = True
        except Exception:  # noqa: BLE001
            pass

        try:
            temps = getattr(r, "_temps", None)
            if temps is not None and isinstance(temps, dict):
                health["temps"] = temps
                has_data = True
        except Exception:  # noqa: BLE001
            pass

        try:
            load = os.getloadavg()
            health["cpu_load"] = round(load[0], 2)
            has_data = True
        except (OSError, AttributeError):
            pass

        try:
            import shutil

            _, _, free = shutil.disk_usage("/")
            health["disk_free_gb"] = round(free / (1024**3), 1)
            has_data = True
        except Exception:  # noqa: BLE001
            pass

        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_total = mem_avail = 0
            for line in lines:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1])
            if mem_total > 0:
                health["mem_pct"] = round(100.0 * (1.0 - mem_avail / mem_total), 1)
                has_data = True
        except (OSError, ValueError):
            pass

        try:
            with open("/proc/uptime") as f:
                health["uptime_s"] = round(float(f.read().split()[0]), 0)
                has_data = True
        except (OSError, ValueError):
            pass

        return health if has_data else None

    # IMU

    def _imu_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_IMU_HZ", IMU_HZ)
        if hz <= 0:
            return
        period = 1.0 / hz
        for _ in self._paced(period):
            try:
                imu = self._read_imu()
                if imu:
                    self.publish(f"strands/{self.peer_id}/imu", imu)
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: imu tick error: %s", self.peer_id, exc)

    def _read_imu(self) -> dict[str, Any] | None:
        r = self.robot
        imu: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}

        try:
            imu_data = getattr(r, "_imu", None)
            if imu_data is not None and isinstance(imu_data, dict):
                imu.update(imu_data)
                return imu
        except Exception:  # noqa: BLE001
            pass

        try:
            inner = getattr(r, "robot", None)
            if inner is not None and hasattr(inner, "get_observation") and getattr(inner, "is_connected", False):
                obs = read_observation(inner)
                for key in ("imu_rpy", "imu", "gyroscope", "accelerometer"):
                    if key in obs:
                        val = obs[key]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        if key in ("imu_rpy", "imu"):
                            imu["rpy"] = val[:3] if len(val) >= 3 else val
                        elif key == "gyroscope":
                            imu["gyro"] = val[:3] if len(val) >= 3 else val
                        elif key == "accelerometer":
                            imu["accel"] = val[:3] if len(val) >= 3 else val
                if "rpy" in imu or "gyro" in imu or "accel" in imu:
                    return imu
        except Exception:  # noqa: BLE001
            pass

        return None

    # Odometry

    def _odom_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_ODOM_HZ", ODOM_HZ)
        if hz <= 0:
            return
        period = 1.0 / hz
        for _ in self._paced(period):
            try:
                odom = self._read_odom()
                if odom:
                    self.publish(f"strands/{self.peer_id}/odom", odom)
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: odom tick error: %s", self.peer_id, exc)

    def _read_odom(self) -> dict[str, Any] | None:
        r = self.robot
        try:
            odom_data = getattr(r, "_odom", None)
            if odom_data is not None and isinstance(odom_data, dict):
                odom: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}
                odom.update(odom_data)
                odom.setdefault("frame", "odom")
                return odom
        except Exception:  # noqa: BLE001
            pass
        return None

    # LiDAR

    def _lidar_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_LIDAR_SUMMARY_HZ", LIDAR_SUMMARY_HZ)
        if hz <= 0:
            return
        summary_period = 1.0 / hz
        state_period = 1.0 / LIDAR_STATE_HZ
        # A publish interval is a duration, so it is measured on a clock that
        # cannot step. -inf rather than 0.0 because a monotonic reading is only
        # meaningful relative to another one: the first tick should be due
        # wherever a platform's monotonic epoch happens to sit.
        last_state_publish_mono = -math.inf

        for _ in self._paced(summary_period):
            try:
                now = time.monotonic()
                summary = self._read_lidar_summary()
                if summary:
                    self.publish(f"strands/{self.peer_id}/lidar/summary", summary)

                if now - last_state_publish_mono >= state_period:
                    state = self._read_lidar_state()
                    if state:
                        self.publish(f"strands/{self.peer_id}/lidar/state", state)
                    last_state_publish_mono = now
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: lidar tick error: %s", self.peer_id, exc)

    def _read_lidar_summary(self) -> dict[str, Any] | None:
        r = self.robot
        try:
            data = getattr(r, "_lidar_summary", None)
            if data is not None and isinstance(data, dict):
                summary: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}
                summary.update(data)
                return summary
        except Exception:  # noqa: BLE001
            pass
        return None

    def _read_lidar_state(self) -> dict[str, Any] | None:
        r = self.robot
        try:
            data = getattr(r, "_lidar_state", None)
            if data is not None and isinstance(data, dict):
                state: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}
                state.update(data)
                return state
        except Exception:  # noqa: BLE001
            pass
        return None

    # Hand / End-Effector

    def _hand_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_HAND_HZ", HAND_HZ)
        if hz <= 0:
            return
        period = 1.0 / hz
        for _ in self._paced(period):
            try:
                hands = self._read_hands()
                if hands:
                    for hand_name, hand_data in hands.items():
                        self.publish(f"strands/{self.peer_id}/hand/{hand_name}/state", hand_data)
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: hand tick error: %s", self.peer_id, exc)

    def _read_hands(self) -> dict[str, dict[str, Any]] | None:
        r = self.robot
        try:
            hands = getattr(r, "_hands", None)
            if hands is not None and isinstance(hands, dict):
                result = {}
                for name, data in hands.items():
                    if isinstance(data, dict):
                        state = {"peer_id": self.peer_id, "hand": name, "t": time.time()}
                        state.update(data)
                        result[name] = state
                return result if result else None
        except Exception:  # noqa: BLE001
            pass
        return None

    # Map Info

    def _map_info_loop(self) -> None:
        hz = _resolve_hz("STRANDS_MESH_MAP_INFO_HZ", MAP_INFO_HZ)
        if hz <= 0:
            return
        period = 1.0 / hz
        for _ in self._paced(period):
            try:
                info = self._read_map_info()
                if info:
                    self.publish(f"strands/{self.peer_id}/map/info", info)
            except NotImplementedError:
                # MRO contract violation: surface immediately rather than
                # silently dropping every sensor tick (issue #258).
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("[mesh] %s: map_info tick error: %s", self.peer_id, exc)

    def _read_map_info(self) -> dict[str, Any] | None:
        r = self.robot
        try:
            data = getattr(r, "_map_info", None)
            if data is not None and isinstance(data, dict):
                info: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}
                info.update(data)
                return info
        except Exception:  # noqa: BLE001
            pass
        return None

    # Safety events

    def publish_safety_event(
        self,
        event_type: str,
        severity: str = "warning",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish a safety event to the mesh AND write to audit log."""
        if not self._running:
            return

        event: dict[str, Any] = {
            "peer_id": self.peer_id,
            "type": event_type,
            # Issue #272: uniform on the wire so a subscriber on
            # strands/+/safety/event cannot use per-branch severity as a
            # content-channel oracle for the rejection reason. The real
            # severity is preserved only in the local audit record below.
            "severity": "info",
            "payload": payload or {},
            "t": time.time(),
        }

        self.publish(f"strands/{self.peer_id}/safety/event", event)

        try:
            log_safety_event(
                event_type=event_type,
                peer_id=self.peer_id,
                payload={"severity": severity, **(payload or {})},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[mesh] %s: audit log write failed: %s", self.peer_id, exc)
