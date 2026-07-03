"""Tests for strands_robots.telemetry — TelemetryStream, events, batching."""

from strands_robots.telemetry import (
    BatchConfig,
    EventCategory,
    StreamTier,
    TelemetryStream,
)
from strands_robots.telemetry.stream import TransportConfig


class TestEventCategory:
    def test_categories_exist(self):
        assert hasattr(EventCategory, "JOINT_STATE")
        assert hasattr(EventCategory, "TASK_START")
        assert hasattr(EventCategory, "TASK_END")


class TestStreamTier:
    def test_tiers_exist(self):
        assert hasattr(StreamTier, "STREAM")
        assert hasattr(StreamTier, "BATCH")
        assert hasattr(StreamTier, "STORAGE")


class TestBatchConfig:
    def test_defaults(self):
        bc = BatchConfig()
        assert bc.max_count > 0
        assert bc.max_age_ms > 0


class TestTelemetryStream:
    def test_create(self):
        stream = TelemetryStream(robot_id="test_robot")
        assert stream is not None

    def test_start_stop(self):
        stream = TelemetryStream(robot_id="test_robot")
        stream.start()
        stream.stop()

    def test_emit(self):
        stream = TelemetryStream(robot_id="test_robot")
        stream.start()
        stream.emit(EventCategory.JOINT_STATE, {"q": [0.1, 0.2, 0.3]})
        stream.stop()


class TestTelemetryDelivery:
    """Exercise the emit -> buffer -> flush -> transport path end to end."""

    @staticmethod
    def _capturing_stream(**kwargs):
        delivered: list = []

        def handler(batch):
            # Small batches arrive as list[dict]; large ones as gzip bytes.
            if isinstance(batch, list):
                delivered.extend(batch)
            return True

        stream = TelemetryStream(robot_id="test_robot", **kwargs)
        stream.add_transport(TransportConfig(name="mem", tiers=list(StreamTier), handler=handler))
        return stream, delivered

    def test_stop_force_drains_sub_threshold_buffer(self):
        # Regression for the shutdown data-loss bug: events below the count/age
        # thresholds must still be delivered when stop() is called.
        stream, delivered = self._capturing_stream(flush_interval_s=5.0)
        stream.start()
        for i in range(3):  # far below max_count, and well under max_age
            stream.emit(EventCategory.JOINT_STATE, {"i": i})
        stream.stop()  # must force-drain the 3 buffered events

        assert len(delivered) == 3
        assert {e["data"]["i"] for e in delivered} == {0, 1, 2}
        assert all(e["robot_id"] == "test_robot" for e in delivered)
        assert all(e["tier"] == "BATCH" for e in delivered)  # JOINT_STATE routes to BATCH
