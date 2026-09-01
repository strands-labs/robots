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


# A sensor payload is a plain record on the wire, so the values in it have to be
# ones ``json.dumps`` accepts. ``_JSONABLE_MAX_DEPTH`` bounds how far down this
# module will go looking for them.
#
# Past the bound the value is handed to the encoder unchanged. That is
# deliberate: a sensor record is shallow (a mapping of names to scalars and short
# vectors), so a structure deeper than this is not a reading but a pathological
# payload - a cycle, or a nested object graph. Those are exactly the payloads
# :func:`strands_robots.mesh.session._report_unencodable_payload` exists to
# report at ERROR, and recursing into them here would turn a reported failure
# into one absorbed by the reader's own handler.
_JSONABLE_MAX_DEPTH = 8


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Coerce *value* into something ``json.dumps`` accepts, or leave it alone.

    Every sensor topic is published as JSON
    (:func:`strands_robots.mesh.session._put_zenoh_directly` encodes the payload
    before it reaches the wire), and a payload the encoder refuses is not a
    transient failure: it fails identically on every tick, so the topic never
    publishes at all. A sensor pipeline reports its readings as numpy - a lidar
    summary's bounding box is whatever ``ndarray.min(axis=0)`` returned, an IMU's
    orientation is a ``float32`` - and none of those are JSON values.

    So this coerces the two things that are readings expressed in a foreign
    numeric type, and nothing else:

    - anything exposing ``tolist()`` (a numpy array or scalar, a torch tensor)
      becomes the equivalent Python list or number;
    - lists, tuples and sets are rebuilt with their contents coerced, and
      mappings with their keys coerced too, because a ``float32`` is no more
      encodable as a key than as a value.

    Anything else is returned unchanged rather than coerced to a string or
    dropped. A payload carrying an object that is genuinely not a reading cannot
    be repaired by guessing, and silently substituting something for it would
    publish a record that misreports what the sensor said. Handing it to the
    encoder untouched is what lets the transport report it by name.

    Args:
        value: A payload, or any value inside one.
        _depth: Recursion depth, bounded by :data:`_JSONABLE_MAX_DEPTH`.

    Returns:
        The coerced value; the original object when there is nothing to coerce.
    """
    if _depth >= _JSONABLE_MAX_DEPTH:
        return value
    if isinstance(value, dict):
        return {_jsonable(k, _depth + 1): _jsonable(v, _depth + 1) for k, v in value.items()}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, (list, tuple, set, frozenset)):
        # Named as concrete types rather than as ``Sequence``, because a ``str``
        # is a sequence and rebuilding one would take a name apart into
        # characters. A tuple has no JSON spelling of its own - ``json.dumps``
        # already renders one as an array - so rebuilding every sequence as a
        # list keeps the encoded form identical while coercing the contents.
        return [_jsonable(item, _depth + 1) for item in value]
    return value


def _coerce_record(payload: dict[str, Any]) -> None:
    """Coerce every entry of *payload* in place, so its declared type is kept.

    The readers build a payload and return it under a ``dict[str, Any] | None``
    signature. Rebuilding it through :func:`_jsonable` would return that
    function's ``Any``, so the coercion is applied entry by entry here instead
    and each reader keeps returning the object it built.

    Only the top level is rewritten: :func:`_jsonable` never mutates, it returns
    a new container, so a nested value the robot still owns - ``_read_health``
    stores the provider's ``_temps`` mapping by reference - is replaced in the
    payload rather than edited underneath the robot.

    Args:
        payload: The outgoing record, modified in place.
    """
    for key in list(payload):
        value = _jsonable(payload.pop(key))
        payload[_jsonable(key)] = value


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

    def _stamp_local_keys(self, record: dict[str, Any], **local: Any) -> dict[str, Any]:
        """Re-assert the keys this process decides, after a provider payload merged in.

        A sensor reader seeds a record, merges the robot's provider mapping over
        it and publishes the result to a topic it builds itself. Merged last, a
        provider mapping carrying one of those seeded names replaces the local
        reading -- so a record can be published to ``strands/{peer_id}/...``
        while naming a different peer inside, and a hand record can be published
        under one hand's name while naming another.

        The presence path already resolves this collision the other way:
        :meth:`strands_robots.mesh.session.PeerInfo.to_dict` spreads the peer's
        own payload *first* so the four keys that process decided win. This is
        the same precedence for the sensor records, applied after the merge
        rather than by ordering a single literal, because a reader merges from
        several provider attributes in turn.

        ``t`` is deliberately not re-asserted. It is a stamp rather than a
        locally computed duration, and a provider that stamps a reading when it
        decoded it is reporting something truer than the moment the loop got
        round to publishing it.

        Args:
            record: The outgoing record, modified in place.
            **local: Further keys this reader decided, such as the ``hand`` a
                hand record is published under.

        Returns:
            The same ``record``, for use as an expression.
        """
        record["peer_id"] = self.peer_id
        record.update(local)
        return record

    def _read_pose(self) -> dict[str, Any] | None:
        r = self.robot
        pose: dict[str, Any] = {"peer_id": self.peer_id, "t": time.time()}

        # Explicit pose provider (highest priority)
        try:
            pose_data = getattr(r, "_pose", None)
            if pose_data is not None:
                if isinstance(pose_data, dict):
                    pose.update(pose_data)
                    self._stamp_local_keys(pose)
                    pose.setdefault("source", "provider")
                    pose.setdefault("frame", "map")
                    _coerce_record(pose)
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
                    _coerce_record(pose)
                    return pose
        except Exception:  # noqa: BLE001
            pass

        # SLAM pose
        try:
            slam_pose = getattr(r, "_slam_pose", None)
            if slam_pose is not None and isinstance(slam_pose, dict):
                pose.update(slam_pose)
                self._stamp_local_keys(pose)
                pose.setdefault("source", "slam")
                pose.setdefault("frame", "map")
                _coerce_record(pose)
                return pose
        except Exception:  # noqa: BLE001
            pass

        # Odometry pose
        try:
            odom_pose = getattr(r, "_odom_pose", None)
            if odom_pose is not None and isinstance(odom_pose, dict):
                pose.update(odom_pose)
                self._stamp_local_keys(pose)
                pose.setdefault("source", "odom")
                pose.setdefault("frame", "odom")
                _coerce_record(pose)
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
                    if "charging" in battery:
                        # Only a record that carries the reading gets to make
                        # the claim.  Defaulting an absent key to False put a
                        # charge state on the health wire that no driver had
                        # reported, indistinguishable from a pack measured to
                        # be discharging.
                        health["charging"] = battery["charging"]
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

        _coerce_record(health)
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
                self._stamp_local_keys(imu)
                _coerce_record(imu)
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
                    _coerce_record(imu)
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
                self._stamp_local_keys(odom)
                odom.setdefault("frame", "odom")
                _coerce_record(odom)
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
                self._stamp_local_keys(summary)
                _coerce_record(summary)
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
                self._stamp_local_keys(state)
                _coerce_record(state)
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
                        self._stamp_local_keys(state, hand=name)
                        result[name] = state
                _coerce_record(result)
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
                self._stamp_local_keys(info)
                _coerce_record(info)
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
        """Publish a safety event to the mesh AND write to the audit log.

        The payload is coerced once through :func:`_coerce_record` and the same
        coerced mapping is handed to both halves, which is what makes the two
        halves report one event instead of two different ones.
        :func:`strands_robots.mesh.session._report_unencodable_payload` records
        why that matters: on a payload the JSON encoder refuses, the audit half
        writes a ``sig="SERIALISE_FAILED"`` poison record and logs at ERROR,
        while the wire half publishes nothing at all - and the failure is not
        transient, so no later tick recovers it. A reading expressed in a
        foreign numeric type is exactly what a safety payload carries (the joint
        value that tripped a limit, the distance that closed), and every other
        record this mixin sends to the wire is coerced before it goes.

        A failed audit write is reported at ERROR for the same reason
        :func:`~strands_robots.mesh.session._report_unencodable_payload` gives for
        the wire half: a transport fire-and-forget tolerance is scoped to a TRANSIENT
        failure that the next tick retries, and this is not that. A safety event is
        published once at one transition, so there is no later tick, and the audit
        copy is the only one carrying the real severity. Reporting the loss at DEBUG
        left the two halves of one call disagreeing about how a permanently lost
        safety record is announced, which is the disagreement that report was raised
        to ERROR to close. The report names the event type and the real severity
        because the wire copy names neither.

        Coerced into a copy rather than in place, so the caller keeps the mapping
        they passed unedited - the same reason :func:`_coerce_record` replaces a
        nested container instead of editing the provider's own.

        A value that is genuinely not a reading is still passed through
        untouched rather than stringified, so the transport still reports it by
        name: repairing an unrepresentable object by guessing would publish a
        record that misstates what happened.

        Args:
            event_type: Short, lowercase event identifier (e.g. ``"estop"``).
            severity: The real severity. It reaches the audit record only - the
                wire copy is uniformly ``"info"`` (issue #272) - so the audit
                record is the only surviving copy, and this parameter wins over
                a ``payload`` field of the same name rather than being replaced
                by it. A ``payload`` entry named ``severity`` is not a second
                channel for this argument; it is discarded from the audit copy.
            payload: Event-specific fields, or ``None`` for an event with none.
        """
        if not self._running:
            return

        record: dict[str, Any] = dict(payload) if payload is not None else {}
        _coerce_record(record)

        event: dict[str, Any] = {
            "peer_id": self.peer_id,
            "type": event_type,
            # Issue #272: uniform on the wire so a subscriber on
            # strands/+/safety/event cannot use per-branch severity as a
            # content-channel oracle for the rejection reason. The real
            # severity is preserved only in the local audit record below.
            "severity": "info",
            "payload": record,
            "t": time.time(),
        }

        self.publish(f"strands/{self.peer_id}/safety/event", event)

        try:
            log_safety_event(
                event_type=event_type,
                peer_id=self.peer_id,
                # ``severity`` last: the parameter is the documented channel for the
                # real severity, so an event-specific field that happens to be named
                # ``severity`` must not replace it. Same precedence as
                # ``session.PeerInfo.to_dict``, which spreads the peer's own payload
                # first so the keys that process decided win a name collision.
                payload={**record, "severity": severity},
            )
        except Exception as exc:  # noqa: BLE001 - see the reporting-level note in the docstring
            logger.error(
                "[mesh] %s: the audit record for a %s safety event could not be written, so the "
                "only copy carrying its real severity (%s) is lost - the wire copy is uniformly "
                "info-level and a safety event is published once, so no later call rewrites it: %s",
                self.peer_id,
                event_type,
                severity,
                exc,
            )
