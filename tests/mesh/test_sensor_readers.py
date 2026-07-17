"""Unit coverage for the Mesh sensor reader methods.

The threaded sensor loops (covered elsewhere) only exercise the happy path of
each ``_read_*`` method through a running Mesh. These tests drive the readers
directly through a minimal host object so the priority-branch logic - SE(3)
matrix decomposition, SLAM/odometry fallbacks, the inner-robot IMU observation
path, multi-source health aggregation and the uniform safety-event wire
severity - is asserted on its outputs rather than implicitly.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from strands_robots.mesh import sensors as mesh_sensors
from strands_robots.mesh.sensors import SensorLoopsMixin, _resolve_hz


class _Host(SensorLoopsMixin):
    """Minimal SensorLoopsMixin host that records published payloads."""

    def __init__(self, robot: Any, peer_id: str = "peer-1") -> None:
        self.robot = robot
        self.peer_id = peer_id
        self._running = True
        self._stop_event = threading.Event()
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, key: str, payload: dict[str, Any]) -> None:
        self.published.append((key, payload))


class _Robot:
    """Bare attribute bag standing in for a robot exposing sensor providers."""


def _host(**robot_attrs: Any) -> _Host:
    robot = _Robot()
    for name, value in robot_attrs.items():
        setattr(robot, name, value)
    return _Host(robot)


# _resolve_hz ---------------------------------------------------------------


def test_resolve_hz_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRANDS_MESH_TEST_HZ", raising=False)
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 7.5) == 7.5


def test_resolve_hz_blank_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH_TEST_HZ", "   ")
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 3.0) == 3.0


def test_resolve_hz_parses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH_TEST_HZ", "20")
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 5.0) == 20.0


def test_resolve_hz_invalid_warns_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH_TEST_HZ", "not-a-number")
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 4.0) == 4.0


def test_resolve_hz_non_positive_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRANDS_MESH_TEST_HZ", "0")
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 5.0) == 0.0
    monkeypatch.setenv("STRANDS_MESH_TEST_HZ", "-2")
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 5.0) == 0.0


# _read_pose ----------------------------------------------------------------


def test_read_pose_none_without_provider() -> None:
    assert _host()._read_pose() is None


def test_read_pose_dict_provider_sets_defaults() -> None:
    host = _host(_pose={"x": 1.0, "y": 2.0})
    pose = host._read_pose()
    assert pose is not None
    assert pose["x"] == 1.0
    assert pose["source"] == "provider"
    assert pose["frame"] == "map"
    assert pose["peer_id"] == "peer-1"


def test_read_pose_matrix_provider_decomposes_se3() -> None:
    # 90-degree yaw about Z at translation (1, 2, 3).
    mat = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = _host(_pose=mat)._read_pose()
    assert pose is not None
    assert pose["x"] == 1.0 and pose["y"] == 2.0 and pose["z"] == 3.0
    assert pose["theta"] == pytest.approx(np.pi / 2)
    # Unit quaternion, scalar-first ordering.
    quat = pose["quat"]
    assert len(quat) == 4
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)
    assert pose["source"] == "provider"


def test_read_pose_matrix_negative_trace_uses_identity_quat() -> None:
    # 180-degree rotation about X gives a non-positive trace branch.
    mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = _host(_pose=mat)._read_pose()
    assert pose is not None
    assert pose["quat"] == [1.0, 0.0, 0.0, 0.0]


def test_read_pose_slam_fallback() -> None:
    host = _host(_slam_pose={"x": 9.0})
    pose = host._read_pose()
    assert pose is not None
    assert pose["source"] == "slam"
    assert pose["frame"] == "map"


def test_read_pose_odom_fallback() -> None:
    host = _host(_odom_pose={"x": 5.0})
    pose = host._read_pose()
    assert pose is not None
    assert pose["source"] == "odom"
    assert pose["frame"] == "odom"


def test_read_pose_slam_accessor_fault_falls_through_to_odom() -> None:
    """A SLAM accessor that raises mid-tick must not abort pose resolution.

    ``_read_pose`` consults sources in priority order (explicit provider ->
    SLAM -> odometry). Each sub-source has its own fail-soft guard so a single
    faulty accessor degrades to the next source rather than crashing the pose
    publish loop. Here the SLAM accessor throws, so resolution falls through to
    the odometry pose.
    """

    class _SlamFaultRobot:
        _odom_pose = {"x": 5.0}

        @property
        def _slam_pose(self) -> Any:
            raise RuntimeError("sensor bus fault reading '_slam_pose'")

    pose = _Host(_SlamFaultRobot())._read_pose()
    assert pose is not None
    assert pose["source"] == "odom"
    assert pose["frame"] == "odom"
    assert pose["x"] == 5.0


def test_read_pose_slam_and_odom_accessor_faults_yield_none() -> None:
    """When both the SLAM and odometry accessors raise, pose resolution
    returns ``None`` (no sample) instead of propagating the driver fault, so
    the publish loop survives a fully faulted pose subsystem."""

    class _PoseFaultRobot:
        @property
        def _slam_pose(self) -> Any:
            raise RuntimeError("sensor bus fault reading '_slam_pose'")

        @property
        def _odom_pose(self) -> Any:
            raise RuntimeError("sensor bus fault reading '_odom_pose'")

    assert _Host(_PoseFaultRobot())._read_pose() is None


# _read_health --------------------------------------------------------------


def test_read_health_dict_battery_and_temps() -> None:
    host = _host(
        _battery={"pct": 80, "charging": True},
        _temps={"cpu": 55.0},
    )
    health = host._read_health()
    assert health is not None
    assert health["battery_pct"] == 80
    assert health["charging"] is True
    assert health["temps"] == {"cpu": 55.0}


def test_read_health_scalar_battery() -> None:
    health = _host(_battery=42)._read_health()
    assert health is not None
    assert health["battery_pct"] == 42.0


def test_read_health_system_stats_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # No robot-provided fields: still returns system stats (cpu/disk/mem/uptime)
    # on Linux, so has_data must be True and the payload non-None.
    health = _host()._read_health()
    assert health is not None
    assert health["peer_id"] == "peer-1"


# _read_imu -----------------------------------------------------------------


def test_read_imu_direct_dict() -> None:
    imu = _host(_imu={"rpy": [0.1, 0.2, 0.3]})._read_imu()
    assert imu is not None
    assert imu["rpy"] == [0.1, 0.2, 0.3]


def test_read_imu_none_without_data() -> None:
    assert _host()._read_imu() is None


def test_read_imu_from_inner_observation() -> None:
    class _Inner:
        is_connected = True

        def get_observation(self) -> dict[str, Any]:
            return {
                "imu_rpy": np.array([0.5, 0.6, 0.7, 9.9]),
                "gyroscope": [1.0, 2.0, 3.0],
                "accelerometer": [4.0, 5.0, 6.0],
            }

    imu = _host(robot=_Inner())._read_imu()
    assert imu is not None
    # ndarray converted via tolist and truncated to 3 elements.
    assert imu["rpy"] == [0.5, 0.6, 0.7]
    assert imu["gyro"] == [1.0, 2.0, 3.0]
    assert imu["accel"] == [4.0, 5.0, 6.0]


def test_read_imu_inner_not_connected_returns_none() -> None:
    class _Inner:
        is_connected = False

        def get_observation(self) -> dict[str, Any]:
            return {"imu_rpy": [0.1, 0.2, 0.3]}

    assert _host(robot=_Inner())._read_imu() is None


# _read_odom / _read_lidar_* / _read_hands / _read_map_info -----------------


def test_read_odom_sets_frame_default() -> None:
    odom = _host(_odom={"x": 0.5})._read_odom()
    assert odom is not None
    assert odom["frame"] == "odom"
    assert odom["x"] == 0.5


def test_read_odom_none_without_data() -> None:
    assert _host()._read_odom() is None


def test_read_lidar_summary_and_state() -> None:
    host = _host(_lidar_summary={"points": 1000}, _lidar_state={"status": "ok"})
    summary = host._read_lidar_summary()
    state = host._read_lidar_state()
    assert summary is not None and summary["points"] == 1000
    assert state is not None and state["status"] == "ok"
    assert _host()._read_lidar_summary() is None
    assert _host()._read_lidar_state() is None


def test_read_hands_wraps_each_hand() -> None:
    host = _host(_hands={"left": {"force": 1.0}, "bad": "not-a-dict"})
    hands = host._read_hands()
    assert hands is not None
    assert "left" in hands
    assert hands["left"]["hand"] == "left"
    assert hands["left"]["force"] == 1.0
    # Non-dict hand entries are skipped.
    assert "bad" not in hands


def test_read_hands_empty_returns_none() -> None:
    assert _host(_hands={})._read_hands() is None
    assert _host()._read_hands() is None


def test_read_map_info() -> None:
    info = _host(_map_info={"resolution": 0.05})._read_map_info()
    assert info is not None
    assert info["resolution"] == 0.05
    assert _host()._read_map_info() is None


# publish_safety_event ------------------------------------------------------


def test_publish_safety_event_uniform_wire_severity(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mesh_sensors,
        "log_safety_event",
        lambda **kw: logged.append(kw),
    )
    host = _host()
    host.publish_safety_event("estop", severity="critical", payload={"reason": "x"})

    assert len(host.published) == 1
    key, event = host.published[0]
    assert key == "strands/peer-1/safety/event"
    # Issue #272: wire severity is always "info" so subscribers cannot use it
    # as a content-channel oracle; true severity lives only in the audit log.
    assert event["severity"] == "info"
    assert event["type"] == "estop"
    assert event["payload"] == {"reason": "x"}
    assert logged[0]["payload"]["severity"] == "critical"


def test_publish_safety_event_noop_when_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mesh_sensors, "log_safety_event", lambda **kw: None)
    host = _host()
    host._running = False
    host.publish_safety_event("estop")
    assert host.published == []


def test_publish_safety_event_survives_audit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kw: Any) -> None:
        raise RuntimeError("audit backend down")

    monkeypatch.setattr(mesh_sensors, "log_safety_event", _boom)
    host = _host()
    # Audit failure must not propagate past the publish.
    host.publish_safety_event("estop", severity="warning")
    assert len(host.published) == 1


# sensor loop lifecycle -----------------------------------------------------
#
# Each ``_*_loop`` is a threaded publish loop with two contract behaviours the
# direct ``_read_*`` tests above do not touch:
#   1. a non-positive rate (``hz <= 0``) disables the loop -- it must return
#      immediately and never publish, so an operator can switch a topic off via
#      ``STRANDS_MESH_*_HZ=0`` without spawning an idle thread.
#   2. a failure inside one tick (e.g. a flaky ``_read_*`` or transport
#      ``publish``) must be swallowed so a single bad tick cannot kill the loop
#      and silently stop every future sample on that topic.
#
# ``(loop, hz_env, reader)`` covers every published sensor topic.
_SENSOR_LOOPS = [
    ("_pose_loop", "STRANDS_MESH_POSE_HZ", "_read_pose"),
    ("_health_loop", "STRANDS_MESH_HEALTH_HZ", "_read_health"),
    ("_imu_loop", "STRANDS_MESH_IMU_HZ", "_read_imu"),
    ("_odom_loop", "STRANDS_MESH_ODOM_HZ", "_read_odom"),
    ("_lidar_loop", "STRANDS_MESH_LIDAR_SUMMARY_HZ", "_read_lidar_summary"),
    ("_hand_loop", "STRANDS_MESH_HAND_HZ", "_read_hands"),
    ("_map_info_loop", "STRANDS_MESH_MAP_INFO_HZ", "_read_map_info"),
]


@pytest.mark.parametrize(("loop_name", "hz_env", "_reader"), _SENSOR_LOOPS)
def test_sensor_loop_disabled_rate_returns_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    loop_name: str,
    hz_env: str,
    _reader: str,
) -> None:
    """A non-positive rate disables the loop: it returns and publishes nothing."""
    monkeypatch.setenv(hz_env, "0")
    host = _host()
    # Would otherwise spin forever; the early return must fire before the loop.
    getattr(host, loop_name)()
    assert host.published == []


@pytest.mark.parametrize(("loop_name", "hz_env", "reader"), _SENSOR_LOOPS)
def test_sensor_loop_swallows_tick_error_and_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    loop_name: str,
    hz_env: str,
    reader: str,
) -> None:
    """A raising reader is caught per tick; the loop exits via the stop event
    instead of propagating, so one flaky sample cannot kill the topic."""
    monkeypatch.setenv(hz_env, "50")  # positive rate -> loop body runs
    host = _host()

    def _boom() -> dict[str, Any]:
        raise RuntimeError("transient sensor read failure")

    monkeypatch.setattr(host, reader, _boom)
    # Pre-set the stop event so the single error tick is the last iteration:
    # ``_stop_event.wait(period)`` returns True immediately and the loop breaks.
    host._stop_event.set()

    # Must not raise despite the reader blowing up on the only tick.
    getattr(host, loop_name)()
    assert host.published == []


@pytest.mark.parametrize(("loop_name", "hz_env", "reader"), _SENSOR_LOOPS)
def test_sensor_loop_reraises_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
    loop_name: str,
    hz_env: str,
    reader: str,
) -> None:
    """A ``NotImplementedError`` (MRO contract violation, issue #258) is the one
    failure that must surface immediately rather than be swallowed per tick."""
    monkeypatch.setenv(hz_env, "50")
    host = _host()

    def _mro_violation() -> dict[str, Any]:
        raise NotImplementedError("mixin used without a host class")

    monkeypatch.setattr(host, reader, _mro_violation)
    host._stop_event.set()

    with pytest.raises(NotImplementedError):
        getattr(host, loop_name)()


# reader fault resilience ---------------------------------------------------
#
# Each ``_read_*`` wraps its provider access in a fail-soft ``try/except`` so a
# robot whose sensor accessor raises (e.g. a driver probing a disconnected
# bus, a property that throws mid-read) degrades to "no sample this tick"
# rather than propagating. This is distinct from the loop-level swallowing
# above: here the *provider attribute access itself* raises, exercising the
# inner guard inside each reader. Without it a single faulty sensor accessor
# would crash the publish loop and silence the topic.


class _FaultyRobot:
    """Robot whose sensor-provider accessors raise, simulating a driver/bus
    fault on sensor read (a property that throws mid-tick).

    Each provider attribute the readers consult (``_pose``, ``_slam_pose``, ``_odom_pose``, ``_battery``,
    ``_temps``, ``_imu``, ``robot``, ``_odom``, ``_lidar_summary``,
    ``_lidar_state``, ``_hands``, ``_map_info``) is a property that raises
    ``RuntimeError`` -- a non-``AttributeError`` fault. This is deliberate:
    the readers fetch providers via ``getattr(r, name, None)``, which would
    *silently swallow* an ``AttributeError`` (returning the default before the
    reader's own ``try/except`` runs), so an ``AttributeError`` fixture would
    never exercise the inner fail-soft guard. A ``RuntimeError`` is not
    suppressed by ``getattr``'s default and therefore propagates into the
    guard under test."""

    @property
    def _pose(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_pose'")

    @property
    def _slam_pose(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_slam_pose'")

    @property
    def _odom_pose(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_odom_pose'")

    @property
    def _battery(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_battery'")

    @property
    def _temps(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_temps'")

    @property
    def _imu(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_imu'")

    @property
    def robot(self) -> Any:
        raise RuntimeError("sensor bus fault reading 'robot'")

    @property
    def _odom(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_odom'")

    @property
    def _lidar_summary(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_lidar_summary'")

    @property
    def _lidar_state(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_lidar_state'")

    @property
    def _hands(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_hands'")

    @property
    def _map_info(self) -> Any:
        raise RuntimeError("sensor bus fault reading '_map_info'")


def _faulty_host() -> _Host:
    return _Host(_FaultyRobot())


@pytest.mark.parametrize(
    "reader",
    [
        "_read_pose",
        "_read_imu",
        "_read_odom",
        "_read_lidar_summary",
        "_read_lidar_state",
        "_read_hands",
        "_read_map_info",
    ],
)
def test_reader_returns_none_when_provider_access_raises(reader: str) -> None:
    """A provider whose attribute access throws yields ``None`` (no sample),
    never a propagated exception, so the publish loop survives the fault."""
    host = _faulty_host()
    assert getattr(host, reader)() is None


def test_read_health_degrades_when_robot_and_system_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the robot's battery/temps accessors raise AND every system-stat
    source (loadavg, disk, /proc) is unavailable, ``_read_health`` collects no
    data and returns ``None`` instead of an empty-but-truthy payload or a
    crash."""
    import builtins
    import os
    import shutil

    def _raise_os(*_a: Any, **_k: Any) -> Any:
        raise OSError("source unavailable")

    monkeypatch.setattr(os, "getloadavg", _raise_os)
    monkeypatch.setattr(shutil, "disk_usage", _raise_os)
    monkeypatch.setattr(builtins, "open", _raise_os)  # blocks /proc/meminfo + /proc/uptime

    assert _faulty_host()._read_health() is None


def test_read_health_aggregates_system_stats_despite_faulty_robot() -> None:
    """Even when the robot's own providers (battery/temps) raise, the
    system-stat sources still populate health: the per-source guards isolate
    the robot fault from the host metrics so partial data is published."""
    health = _faulty_host()._read_health()
    # On a normal host at least one of loadavg / disk / meminfo / uptime
    # resolves, so health is a populated payload (not None) and carries no
    # robot-provided battery field (that source faulted).
    assert health is not None
    assert "battery_pct" not in health
    assert health["peer_id"] == "peer-1"
