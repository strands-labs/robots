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
import time
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


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e999"])
def test_resolve_hz_non_finite_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A non-finite override keeps the loop at its default rate.

    ``float()`` accepts "inf"/"nan" and overflows "1e999" to inf, so these slip
    past the unparsable-value guard. Each caller then computes
    ``period = 1.0 / hz``: inf gives a zero period, so ``_stop_event.wait(0)``
    returns immediately and the publish loop busy-spins; nan compares False
    against ``hz > 0`` and silently switches the topic off. Neither is a rate an
    operator can have meant, so the loop keeps the default it documents.
    """
    monkeypatch.setenv("STRANDS_MESH_TEST_HZ", value)
    assert _resolve_hz("STRANDS_MESH_TEST_HZ", 9.0) == 9.0


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


def _yaw_se3(degrees: float) -> np.ndarray:
    """SE(3) transform: yaw about Z at a fixed non-zero translation."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, -s, 0.0, 1.5],
            [s, c, 0.0, -0.25],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rotmat_from_quat_wxyz(q: list[float]) -> np.ndarray:
    """Scalar-first unit quaternion -> 3x3 rotation matrix (round-trip oracle)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_angle_degrees(q: list[float]) -> float:
    """Rotation angle a scalar-first quaternion encodes, in degrees."""
    return float(2.0 * np.degrees(np.arccos(min(1.0, abs(q[0])))))


def _axis_angle_rot(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues).

    Built from the angle rather than typed as decimal literals so the matrix is
    orthonormal to machine precision and a round-trip can be asserted tightly.
    """
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.radians(degrees)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(t) * k + (1.0 - np.cos(t)) * (k @ k)


@pytest.mark.parametrize("yaw", [5.0, 45.0, 90.0, 119.0, 121.0, 150.0, 179.0, 180.0, -135.0, -180.0])
def test_read_pose_matrix_quat_and_theta_report_the_same_rotation(yaw: float) -> None:
    """One pose payload cannot carry two different headings.

    ``theta`` and ``quat`` are decomposed from the same SE(3) matrix and are the
    only two orientation fields the payload has, so a consumer that reads either
    must get the same answer. A quaternion built from the trace branch alone is
    unusable once the trace is non-positive - which is exactly a rotation of 120
    degrees or more, 61% of SO(3) - and substituting the identity there makes
    this payload self-contradictory: the same message reported a robot turned
    180 degrees as ``theta`` = pi beside a ``quat`` meaning no rotation at all.
    """
    pose = _host(_pose=_yaw_se3(yaw))._read_pose()
    assert pose is not None
    theta_deg = float(np.degrees(pose["theta"]))
    assert theta_deg == pytest.approx(yaw, abs=1e-6)
    # A planar yaw is a rotation about Z alone, so the quaternion's angle is the
    # magnitude of that heading and its axis is +/-Z.
    assert _quat_angle_degrees(pose["quat"]) == pytest.approx(abs(yaw), abs=1e-6)
    assert pose["quat"][1] == pytest.approx(0.0, abs=1e-9)
    assert pose["quat"][2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "rot"),
    [
        ("yaw 180", np.diag([-1.0, -1.0, 1.0])),
        ("pitch 180", np.diag([-1.0, 1.0, -1.0])),
        ("roll 180", np.diag([1.0, -1.0, -1.0])),
    ],
)
def test_read_pose_matrix_reports_a_half_turn_as_a_half_turn(name: str, rot: np.ndarray) -> None:
    """A half turn about any axis is a half turn, not the identity.

    All three of these have trace ``-1``, the far end of the branch the trace
    formula cannot serve. Reporting the identity here tells a fleet operator a
    robot that has completely reversed its orientation has not moved.
    """
    mat = np.eye(4)
    mat[:3, :3] = rot
    pose = _host(_pose=mat)._read_pose()
    assert pose is not None
    assert _quat_angle_degrees(pose["quat"]) == pytest.approx(180.0, abs=1e-6), name


@pytest.mark.parametrize(
    ("axis", "degrees"),
    [
        ((0.0, 0.0, 1.0), 180.0),
        ((1.0, 0.0, 0.0), 180.0),
        ((0.0, 1.0, 0.0), 135.0),
        ((1.0, 1.0, 0.0), 180.0),
        ((1.0, 2.0, -3.0), 168.0),
        ((-2.0, 1.0, 0.5), 121.0),
    ],
)
def test_read_pose_matrix_quat_round_trips_to_the_input_rotation(
    axis: tuple[float, float, float], degrees: float
) -> None:
    """Recomposing the published quaternion returns the matrix it came from.

    The round trip is the contract independent of any convention: whatever
    ordering or sign the payload uses, the rotation it encodes must be the one
    the caller handed over. Each of these has a non-positive trace.
    """
    rot = _axis_angle_rot(axis, degrees)
    mat = np.eye(4)
    mat[:3, :3] = rot
    pose = _host(_pose=mat)._read_pose()
    assert pose is not None
    assert float(np.trace(rot)) <= 0.0, "premise: exercises the non-positive-trace branch"
    recovered = _rotmat_from_quat_wxyz(pose["quat"])
    assert np.allclose(recovered, rot, atol=1e-9)


@pytest.mark.parametrize("yaw", [5.0, 45.0, 90.0, 119.0])
def test_read_pose_matrix_quat_unchanged_below_the_trace_boundary(yaw: float) -> None:
    """Rotations the trace branch already served keep the quaternion they had.

    These yaws all have a positive trace, so they were already reported
    correctly. Completing the remaining branches must not perturb them - this
    fails if the branch selection is changed rather than extended.
    """
    pose = _host(_pose=_yaw_se3(yaw))._read_pose()
    assert pose is not None
    half = np.radians(yaw) / 2.0
    assert pose["quat"] == pytest.approx([np.cos(half), 0.0, 0.0, np.sin(half)], abs=1e-9)


@pytest.mark.parametrize("yaw", [45.0, 121.0, 180.0, -160.0])
def test_read_pose_matrix_quat_is_always_a_unit_quaternion(yaw: float) -> None:
    """Every published quaternion is unit length, on both sides of the branch."""
    pose = _host(_pose=_yaw_se3(yaw))._read_pose()
    assert pose is not None
    assert float(np.linalg.norm(pose["quat"])) == pytest.approx(1.0, abs=1e-12)


def test_read_pose_matrix_quat_round_trips_across_a_uniform_sample_of_so3() -> None:
    """The whole rotation group, not a hand-picked list of angles.

    Recomposing the published quaternion must return the matrix it came from for
    any orientation a robot can be in. The sample is asserted to straddle the
    branch boundary so a seed that happened to draw only small rotations cannot
    pass vacuously, and the split is reported in the failure message because the
    non-positive-trace side is the majority of the group, not a corner of it.
    """
    rng = np.random.default_rng(17)
    mats = []
    for _ in range(300):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        mats.append(q)
    traces = [float(np.trace(m)) for m in mats]
    n_nonpositive = sum(1 for t in traces if t <= 0.0)
    assert 0 < n_nonpositive < len(mats), (
        f"premise: sample must straddle the branch boundary, got {n_nonpositive}/{len(mats)} non-positive"
    )
    worst = 0.0
    for rot in mats:
        mat = np.eye(4)
        mat[:3, :3] = rot
        pose = _host(_pose=mat)._read_pose()
        assert pose is not None
        worst = max(worst, float(np.max(np.abs(_rotmat_from_quat_wxyz(pose["quat"]) - rot))))
    assert worst < 1e-9, (
        f"worst round-trip error {worst:.3e} over {len(mats)} rotations ({n_nonpositive} non-positive trace)"
    )


def test_read_pose_matrix_quat_is_unit_for_a_drifted_rotation_block() -> None:
    """A not-quite-orthonormal rotation block still yields a unit quaternion.

    A pose integrated from odometry or handed over by a SLAM stack accumulates
    scale drift. Without normalization the quaternion's length silently carries
    that drift onto the wire, where a consumer reading it as a unit quaternion
    has no way to notice.
    """
    mat = _yaw_se3(70.0)
    mat[:3, :3] *= 1.004
    pose = _host(_pose=mat)._read_pose()
    assert pose is not None
    assert float(np.linalg.norm(pose["quat"])) == pytest.approx(1.0, abs=1e-12)


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


def test_read_health_makes_no_charge_claim_a_record_did_not_carry() -> None:
    """A record with no charge reading gets no charge key on the wire.

    The reader used to default an absent ``charging`` key to ``False``, so
    every driver whose battery record carries no charge flag - the G1's
    ``BmsState_`` declares none - published a charge state indistinguishable
    from a pack measured to be discharging.  Absence is the honest encoding
    of a question the robot does not answer.
    """
    health = _host(_battery={"pct": 80})._read_health()
    assert health is not None
    assert health["battery_pct"] == 80
    assert "charging" not in health


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


@pytest.mark.parametrize(("loop_name", "hz_env", "_reader"), _SENSOR_LOOPS)
def test_sensor_loop_paces_itself_when_rate_env_is_non_finite(
    monkeypatch: pytest.MonkeyPatch,
    loop_name: str,
    hz_env: str,
    _reader: str,
) -> None:
    """A non-finite rate override must not turn the loop into a busy-spin.

    ``STRANDS_MESH_*_HZ=inf`` resolves through ``float()`` to a rate whose
    period is ``1.0 / inf == 0.0``, so ``_stop_event.wait(0.0)`` returns
    immediately and the loop publishes to the mesh as fast as the CPU allows --
    a flood, from one environment typo, on whichever topics the robot exposes.
    Falling back to the documented default instead keeps the loop paced.

    The bound is deliberately loose so the assertion does not depend on how
    fast the host is: the highest default sensor rate is 50 Hz, which is a
    handful of publishes in this window, while an unpaced loop reaches five to
    six orders of magnitude more.
    """
    monkeypatch.setenv(hz_env, "inf")
    host = _host(
        _pose={"x": 0.0, "y": 0.0, "z": 0.0},
        _imu={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        _odom={"x": 0.0, "y": 0.0},
        _lidar_summary={"points": 1},
        _hands={"left": {"joints": [0.0]}},
        _map_info={"resolution": 0.05},
    )
    thread = threading.Thread(target=getattr(host, loop_name), daemon=True)
    thread.start()
    try:
        time.sleep(0.15)
    finally:
        host._running = False
        host._stop_event.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    # A loop that published nothing would satisfy any upper bound, so pin both
    # ends: the topic stayed live *and* it was paced.
    assert host.published, f"{loop_name} published nothing; the bound below would be vacuous"
    assert len(host.published) < 500, f"{loop_name} published {len(host.published)} times unpaced"


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
