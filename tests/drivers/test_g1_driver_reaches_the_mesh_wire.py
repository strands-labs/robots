"""A native G1 driver's DDS readings reach the mesh wire, encoded.

Issue #354's fourth acceptance criterion: a :class:`~strands_robots.mesh.Mesh`
wrapping a :class:`~strands_robots.drivers.g1.G1Driver` fed mocked DDS publishes
``strands/<peer>/lidar/summary``, ``imu`` and ``health``. Every other criterion
was pinned when the driver landed; this one was verified by reading the two
sides and checking that the key names agreed, which is exactly the check a test
should be doing instead.

What makes it worth a test rather than an inspection is that the two halves are
joined by nothing but attribute names. ``SensorLoopsMixin`` reads sixteen
underscore-prefixed attributes straight off whatever object the mesh was handed;
the driver writes four of them from its DDS callbacks. Nothing declares that
relationship, so a rename on either side is silent - and issue #2749 records the
same seam failing in the other direction, where a contract-complete driver
publishes no joint telemetry at all because joints are reached through an inner
lerobot device this driver does not have. Pinning which topics *do* arrive is
what makes that boundary legible.

Three properties this file asserts that a reader-level test cannot:

- the composition, with a real ``Mesh`` rather than a mixin host, so the
  attribute names are checked against the object the mesh really reads;
- the *encoding*, because ``session._put_zenoh_directly`` runs ``json.dumps``
  before the bytes reach ``put`` and a payload it refuses is dropped for good -
  the failure is deterministic, so no later tick recovers it. The capture here
  encodes what it is handed, which is the same step moved to where a test can
  see which side of it a record landed;
- that the DDS side really is numpy. A Livox frame's header fields and an IMU's
  orientation arrive as ``int64``/``float32``, and ``json.dumps`` refuses both,
  so feeding plain Python floats would test a payload the robot never sends.

The loops are driven one tick each, deterministically and without sleeping:
``SensorLoopsMixin._paced`` yields *before* it waits, so a stop event that is
already set produces exactly one iteration and then breaks.
"""

from __future__ import annotations

import json
import types
from typing import Any

import numpy as np
import pytest

import strands_robots.mesh.core as mesh_core
from strands_robots.drivers.g1 import G1Driver
from strands_robots.mesh.core import Mesh

_PEER = "neon"

#: The publish rates the loops resolve from the environment. Cleared so a rate
#: configured on the host cannot turn a loop off (``_resolve_hz`` returning 0
#: makes the loop return before its first tick) and change what this file grades.
_RATE_ENV = (
    "STRANDS_MESH_POSE_HZ",
    "STRANDS_MESH_HEALTH_HZ",
    "STRANDS_MESH_IMU_HZ",
    "STRANDS_MESH_ODOM_HZ",
    "STRANDS_MESH_LIDAR_SUMMARY_HZ",
)

#: The loops that publish the criterion's three topics, plus the state topic the
#: lidar loop shares. Driven explicitly rather than through ``Mesh.start()`` so
#: no thread, clock or network is involved.
_LOOPS = ("_imu_loop", "_health_loop", "_lidar_loop")


def _lowstate() -> Any:
    """An ``rt/lowstate`` sample, with the IMU as the float32 the SDK reports."""
    return types.SimpleNamespace(
        imu_state=types.SimpleNamespace(
            rpy=np.array([0.01, -0.02, 0.5], dtype=np.float32),
            gyroscope=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            accelerometer=np.array([0.0, 0.0, 9.81], dtype=np.float32),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        mode_machine=np.int64(501),
    )


def _bms() -> Any:
    """An ``rt/lf/bmsstate`` sample, with the charge as a float32."""
    return types.SimpleNamespace(soc=np.float32(87.5), current=np.float32(-2.4), cycle=np.int64(42))


def _lidar_state() -> Any:
    """An ``rt/utlidar/lidar_state`` sample.

    The field names are the ones ``LidarState_`` actually declares:
    ``error_state`` for the fault code and ``cloud_frequency`` for the scan
    rate. A double that instead spells whatever name the decoder happens to
    read would agree with the decoder whatever that name is, which is how an
    undeclared read stayed invisible here in the first place.
    """
    return types.SimpleNamespace(error_state=0, cloud_frequency=np.float32(10.0), sys_rotation_speed=np.float32(3600.0))


def _lidar_cloud() -> Any:
    """A Mid-360 ``PointCloud2_`` header at 10 Hz: 24000 points in one row."""
    return types.SimpleNamespace(
        width=np.int64(24000), height=np.int64(1), point_step=np.int64(16), row_step=np.int64(384000)
    )


class _Capture:
    """Records what the transport would encode, and what it would refuse."""

    def __init__(self) -> None:
        self.wire: dict[str, Any] = {}
        self.dropped: dict[str, str] = {}

    def put(self, key: str, payload: dict[str, Any]) -> None:
        try:
            encoded = json.dumps(payload).encode()
        except (TypeError, ValueError) as exc:
            self.dropped[key] = f"{type(exc).__name__}: {exc}"
            return
        self.wire[key] = json.loads(encoded)


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    """Replace the module-level ``put`` the mesh publishes through."""
    recorder = _Capture()
    monkeypatch.setattr(mesh_core, "put", recorder.put)
    for name in _RATE_ENV:
        monkeypatch.delenv(name, raising=False)
    return recorder


def _driver(*, fed: bool = True) -> G1Driver:
    """A driver whose caches hold one DDS sample each, or nothing at all."""
    driver = G1Driver(tool_name=_PEER, port="192.168.1.172")
    if fed:
        driver._on_lowstate(_lowstate())
        driver._on_bms(_bms())
        driver._on_lidar_state(_lidar_state())
        driver._on_lidar_cloud(_lidar_cloud())
    return driver


def _tick(robot: Any, capture: _Capture) -> _Capture:
    """Run one tick of each sensor loop of a real Mesh wrapping *robot*."""
    mesh = Mesh(robot, _PEER)
    mesh._running = True
    # _paced yields before waiting, so an already-set stop event gives exactly
    # one iteration per loop and the ticker returns without sleeping.
    mesh._stop_event.set()
    for loop in _LOOPS:
        getattr(mesh, loop)()
    return capture


class TestTheCriterionTopicsArriveEncoded:
    """Issue #354 criterion 4, asserted at the transport boundary."""

    @pytest.mark.parametrize("topic", ["lidar/summary", "imu", "health"])
    def test_the_topic_is_published(self, capture: _Capture, topic: str) -> None:
        wire = _tick(_driver(), capture).wire
        assert f"strands/{_PEER}/{topic}" in wire

    def test_nothing_the_driver_reported_was_refused_by_the_encoder(self, capture: _Capture) -> None:
        """A numpy reading must survive the encode, not be dropped before the wire."""
        assert _tick(_driver(), capture).dropped == {}

    def test_the_imu_record_carries_the_dds_orientation(self, capture: _Capture) -> None:
        imu = _tick(_driver(), capture).wire[f"strands/{_PEER}/imu"]
        assert imu["rpy"] == pytest.approx([0.01, -0.02, 0.5], rel=1e-6)
        assert imu["accelerometer"] == pytest.approx([0.0, 0.0, 9.81], rel=1e-6)

    def test_the_health_record_carries_the_battery_percentage(self, capture: _Capture) -> None:
        health = _tick(_driver(), capture).wire[f"strands/{_PEER}/health"]
        assert health["battery_pct"] == pytest.approx(87.5, rel=1e-6)

    def test_the_health_record_claims_no_charge_state_the_pack_never_reported(self, capture: _Capture) -> None:
        """``BmsState_`` declares no charge flag, so the wire carries none.

        The reader used to default an absent ``charging`` key to ``False``,
        which put a charge state on the health wire for every G1 - a claim
        indistinguishable from a pack measured to be discharging.  An
        absent key is the honest encoding of a reading the robot does not
        publish.
        """
        health = _tick(_driver(), capture).wire[f"strands/{_PEER}/health"]
        assert "charging" not in health

    def test_the_lidar_summary_carries_the_cloud_size(self, capture: _Capture) -> None:
        summary = _tick(_driver(), capture).wire[f"strands/{_PEER}/lidar/summary"]
        assert summary["count"] == 24000
        assert summary["width"] == 24000

    def test_every_record_is_addressed_to_the_publishing_peer(self, capture: _Capture) -> None:
        """The topic names the mesh's peer, and so does the record inside it."""
        wire = _tick(_driver(), capture).wire
        assert wire, "no topic was published, so this asserts nothing"
        for topic, payload in wire.items():
            assert topic.startswith(f"strands/{_PEER}/")
            assert payload["peer_id"] == _PEER


class TestTheLidarStateTopicComesWithTheSummary:
    """The lidar loop publishes two topics; the criterion names one of them."""

    def test_the_state_topic_is_published(self, capture: _Capture) -> None:
        wire = _tick(_driver(), capture).wire
        assert f"strands/{_PEER}/lidar/state" in wire

    def test_the_state_record_carries_the_decoded_status_code(self, capture: _Capture) -> None:
        state = _tick(_driver(), capture).wire[f"strands/{_PEER}/lidar/state"]
        assert state["freq"] == pytest.approx(10.0)
        assert "OK" in state["code_text"]


class TestASilentDriverPublishesNoReading:
    """Non-vacuity: the topics above are the DDS samples, not the loop running."""

    @pytest.mark.parametrize("topic", ["lidar/summary", "lidar/state", "imu"])
    def test_an_unfed_driver_publishes_no_sensor_topic(self, capture: _Capture, topic: str) -> None:
        """Every sensor cache is optional, so a driver that never connected is quiet."""
        wire = _tick(_driver(fed=False), capture).wire
        assert f"strands/{_PEER}/{topic}" not in wire

    def test_health_still_publishes_without_a_battery_reading(self, capture: _Capture) -> None:
        """Health aggregates host metrics too, so it is the one topic that survives.

        Asserted so the parametrized cell above is read for what it is: those
        three topics are silent because their provider is absent, not because the
        loop failed to run.
        """
        health = _tick(_driver(fed=False), capture).wire[f"strands/{_PEER}/health"]
        assert "battery_pct" not in health


class TestTheDriverIsWhatTheMeshReads:
    """The seam itself: the mesh reads these attributes off the driver."""

    @pytest.mark.parametrize("attribute", ["_imu", "_battery", "_lidar_state", "_lidar_summary"])
    def test_the_driver_fills_the_attribute_the_mesh_reads(self, attribute: str) -> None:
        """A rename on either side is otherwise silent - nothing declares this."""
        assert getattr(_driver(), attribute) is not None

    def test_the_mesh_holds_the_driver_itself(self) -> None:
        """No inner lerobot device: the driver *is* the robot the mesh reads.

        This is the property issue #2749 is about. It is asserted here as the
        premise of the topics above, not as something desirable.
        """
        driver = _driver()
        mesh = Mesh(driver, _PEER)
        assert mesh.robot is driver
        assert getattr(driver, "robot", None) is None
