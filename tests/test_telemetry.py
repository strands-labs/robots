"""Tests for strands_robots.telemetry — TelemetryStream, events, batching."""

from strands_robots.telemetry import (
    BatchConfig,
    EventCategory,
    StreamTier,
    TelemetryStream,
)


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
