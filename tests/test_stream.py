"""
Comprehensive tests for the Leo Platform-inspired telemetry system.

Tests cover:
- Event creation and serialization
- Tier routing (strategy pattern)
- TelemetryStream lifecycle (start/stop)
- Non-blocking emit performance
- Auto-batching (count/size/age triggers)
- Gzip compression
- Correlation tracking (spans)
- LocalWALTransport
- StdoutTransport
- stream tool (all 7 actions)
- Robot integration simulation
"""

import gzip
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

# Pre-mock strands before importing any strands_robots modules
_mock_strands = MagicMock()
_mock_strands.tool = lambda f: f
sys.modules.setdefault("strands", _mock_strands)

# Pre-mock cv2 to prevent OpenCV 4.12 crash
import importlib.machinery as _im  # noqa: E402

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im.ModuleSpec("cv2", None)
_mock_cv2.dnn = MagicMock()
sys.modules.setdefault("cv2", _mock_cv2)
sys.modules.setdefault("cv2.dnn", _mock_cv2.dnn)

from strands_robots.telemetry.stream import TelemetryStream, TransportConfig  # noqa: E402
from strands_robots.telemetry.transports.local import LocalWALTransport  # noqa: E402
from strands_robots.telemetry.transports.stdout import StdoutTransport  # noqa: E402
from strands_robots.telemetry.types import (  # noqa: E402
    BatchConfig,
    EventCategory,
    StreamTier,
    TelemetryEvent,
)

# ============================================================================
# TelemetryEvent Tests
# ============================================================================


class TestTelemetryEvent:
    """Tests for TelemetryEvent dataclass."""

    def test_create_event(self):
        """Create a basic telemetry event."""
        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="so100",
            data={"q": [0.1, 0.2, 0.3]},
        )
        assert event.robot_id == "so100"
        assert event.category == EventCategory.JOINT_STATE
        assert event.data["q"] == [0.1, 0.2, 0.3]
        assert event.event_id  # auto-generated
        assert event.timestamp_ms > 0

    def test_effective_tier_default(self):
        """Default tier comes from category."""
        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="test",
            data={},
        )
        assert event.effective_tier == StreamTier.BATCH

    def test_effective_tier_override(self):
        """Explicit tier overrides category default."""
        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="test",
            data={},
            tier=StreamTier.STREAM,
        )
        assert event.effective_tier == StreamTier.STREAM

    def test_safety_events_are_stream_tier(self):
        """Safety events default to STREAM tier."""
        for cat in [
            EventCategory.EMERGENCY_STOP,
            EventCategory.COLLISION,
            EventCategory.JOINT_LIMIT,
            EventCategory.SAFETY_ALERT,
            EventCategory.ERROR,
        ]:
            event = TelemetryEvent(category=cat, robot_id="test", data={})
            assert event.effective_tier == StreamTier.STREAM, f"{cat} should be STREAM"

    def test_heavy_data_is_storage_tier(self):
        """Camera/point cloud events default to STORAGE tier."""
        for cat in [
            EventCategory.CAMERA_FRAME,
            EventCategory.POINT_CLOUD,
            EventCategory.DEPTH_MAP,
            EventCategory.EPISODE_CHECKPOINT,
        ]:
            event = TelemetryEvent(category=cat, robot_id="test", data={})
            assert event.effective_tier == StreamTier.STORAGE, f"{cat} should be STORAGE"

    def test_serialize_basic(self):
        """Serialize event to JSON bytes."""
        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="so100",
            data={"q": [0.1, 0.2]},
        )
        raw = event.serialize()
        assert isinstance(raw, bytes)
        parsed = json.loads(raw)
        assert parsed["category"] == "joint_state"
        assert parsed["robot_id"] == "so100"
        assert parsed["data"]["q"] == [0.1, 0.2]

    def test_serialize_with_numpy(self):
        """Serialize event with numpy arrays."""
        np = pytest.importorskip("numpy")
        event = TelemetryEvent(
            category=EventCategory.OBSERVATION,
            robot_id="g1",
            data={
                "q": np.array([0.1, 0.2, 0.3]),
                "vel": np.float32(1.5),
                "steps": np.int64(42),
            },
        )
        raw = event.serialize()
        parsed = json.loads(raw)
        assert parsed["data"]["q"] == [0.1, 0.2, 0.3]
        assert parsed["data"]["vel"] == pytest.approx(1.5, rel=1e-5)
        assert parsed["data"]["steps"] == 42

    def test_compress(self):
        """Serialize + gzip compress."""
        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="so100",
            data={"q": list(range(100))},
        )
        compressed = event.compress()
        assert isinstance(compressed, bytes)
        decompressed = gzip.decompress(compressed)
        parsed = json.loads(decompressed)
        assert parsed["data"]["q"] == list(range(100))

    def test_size_bytes_estimate(self):
        """Size estimate is reasonable."""
        event = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="so100",
            data={"q": [0.1] * 10},
        )
        est = event.size_bytes()
        actual = len(event.serialize())
        # Estimate should be within 5x of actual
        assert est > 0
        assert est < actual * 5

    def test_all_categories_have_tier(self):
        """Every EventCategory has a valid default tier."""
        for cat in EventCategory:
            assert isinstance(cat.default_tier, StreamTier)
            assert cat.category_name  # non-empty string

    def test_serialize_bytes_data(self):
        """Bytes in data are serialized as length reference."""
        event = TelemetryEvent(
            category=EventCategory.CAMERA_FRAME,
            robot_id="cam0",
            data={"frame": b"\x00" * 1000},
        )
        raw = event.serialize()
        parsed = json.loads(raw)
        assert parsed["data"]["frame"] == "<bytes:1000>"

    def test_metadata_included(self):
        """Metadata field is included in serialization."""
        event = TelemetryEvent(
            category=EventCategory.CUSTOM,
            robot_id="test",
            data={"x": 1},
            metadata={"source": "test_suite"},
        )
        raw = event.serialize()
        parsed = json.loads(raw)
        assert parsed["metadata"]["source"] == "test_suite"


# ============================================================================
# BatchConfig Tests
# ============================================================================


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.max_count == 100
        assert cfg.max_bytes == 256 * 1024
        assert cfg.max_age_ms == 500.0

    def test_custom(self):
        cfg = BatchConfig(max_count=10, max_bytes=1024, max_age_ms=50.0)
        assert cfg.max_count == 10
        assert cfg.max_bytes == 1024
        assert cfg.max_age_ms == 50.0


# ============================================================================
# TelemetryStream Tests
# ============================================================================


class TestTelemetryStream:
    """Tests for the core TelemetryStream class."""

    def test_create_stream(self):
        """Create a stream with default config."""
        stream = TelemetryStream(robot_id="so100")
        assert stream.robot_id == "so100"
        assert not stream._running
        assert len(stream._buffers) == 3  # STREAM, BATCH, STORAGE

    def test_start_stop(self):
        """Start and stop lifecycle."""
        stream = TelemetryStream(robot_id="test")
        stream.start()
        assert stream._running
        assert stream._flush_thread is not None
        assert stream._flush_thread.is_alive()

        stream.stop()
        assert not stream._running

    def test_emit_basic(self):
        """Emit events into buffer."""
        stream = TelemetryStream(robot_id="test")

        stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
        stream.emit(EventCategory.EMERGENCY_STOP, {"reason": "collision"})
        stream.emit(EventCategory.CAMERA_FRAME, {"size": 1000})

        assert len(stream._buffers[StreamTier.BATCH]) == 1
        assert len(stream._buffers[StreamTier.STREAM]) == 1
        assert len(stream._buffers[StreamTier.STORAGE]) == 1

    def test_emit_frame_counter(self):
        """Frame counter increments monotonically."""
        stream = TelemetryStream(robot_id="test")
        stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
        stream.emit(EventCategory.JOINT_STATE, {"q": [0.2]})
        stream.emit(EventCategory.JOINT_STATE, {"q": [0.3]})

        events = list(stream._buffers[StreamTier.BATCH])
        assert events[0].frame_id == 0
        assert events[1].frame_id == 1
        assert events[2].frame_id == 2

    def test_emit_ring_buffer(self):
        """Buffer drops oldest when full."""
        stream = TelemetryStream(robot_id="test", buffer_maxlen=5)

        for i in range(10):
            stream.emit(EventCategory.JOINT_STATE, {"i": i})

        buf = stream._buffers[StreamTier.BATCH]
        assert len(buf) == 5
        # Oldest should be dropped — newest 5 remain
        values = [e.data["i"] for e in buf]
        assert values == [5, 6, 7, 8, 9]

    def test_emit_dropped_count(self):
        """Dropped events are tracked in stats."""
        stream = TelemetryStream(robot_id="test", buffer_maxlen=3)

        for i in range(5):
            stream.emit(EventCategory.JOINT_STATE, {"i": i})

        stats = stream.get_stats()
        assert stats["emitted"] == 5
        assert stats["dropped"] == 2  # 4th and 5th events caused drops

    def test_emit_performance(self):
        """Emit should be <1ms average."""
        stream = TelemetryStream(robot_id="perf_test")

        times = []
        for _ in range(1000):
            start = time.perf_counter()
            stream.emit(EventCategory.JOINT_STATE, {"q": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_ms = sum(times) / len(times)
        p99_ms = sorted(times)[int(len(times) * 0.99)]

        assert avg_ms < 1.0, f"Average emit time {avg_ms:.3f}ms exceeds 1ms budget"
        assert p99_ms < 5.0, f"P99 emit time {p99_ms:.3f}ms exceeds 5ms budget"

    def test_emit_threadsafe(self):
        """Emit from multiple threads concurrently."""
        stream = TelemetryStream(robot_id="mt_test")
        errors = []

        def emitter(thread_id: int):
            try:
                for i in range(100):
                    stream.emit(EventCategory.JOINT_STATE, {"tid": thread_id, "i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emitter, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert stream.get_stats()["emitted"] == 400

    def test_correlation_span(self):
        """Span sets and restores correlation context."""
        stream = TelemetryStream(robot_id="test")

        with stream.span("pick_object") as trace_id:
            assert trace_id
            assert len(trace_id) == 16
            stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]})

            # Event should have correlation_id
            event = stream._buffers[StreamTier.BATCH][-1]
            assert event.correlation_id == trace_id

        # After span, correlation should be cleared
        assert getattr(stream._correlation, "trace_id", None) is None

    def test_nested_spans(self):
        """Nested spans track parent-child relationship."""
        stream = TelemetryStream(robot_id="test")

        with stream.span("outer") as outer_id:
            with stream.span("inner") as inner_id:
                assert inner_id != outer_id
                stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]})

            # After inner span, should restore outer
            assert getattr(stream._correlation, "trace_id", None) == outer_id

    def test_flush_with_transport(self):
        """Flush sends events to registered transport."""
        stream = TelemetryStream(robot_id="test", flush_interval_s=10.0)

        received: List[Any] = []

        def mock_handler(payload: Any) -> bool:
            received.append(payload)
            return True

        stream.add_transport(
            TransportConfig(
                name="mock",
                tiers=[StreamTier.BATCH],
                handler=mock_handler,
            )
        )

        # Emit enough events to trigger count-based flush
        for i in range(200):
            stream.emit(EventCategory.JOINT_STATE, {"i": i})

        stream._flush_all()

        assert len(received) > 0
        stats = stream.get_stats()
        assert stats["flushed"] > 0

    def test_flush_compression(self):
        """Large batches are gzip compressed."""
        stream = TelemetryStream(
            robot_id="test",
            compression_threshold=100,  # Low threshold for testing
        )

        # Override batch config to flush on small count
        stream._batch_configs[StreamTier.BATCH] = BatchConfig(max_count=20, max_bytes=256 * 1024, max_age_ms=500.0)

        received: List[Any] = []

        def mock_handler(payload: Any) -> bool:
            received.append(payload)
            return True

        stream.add_transport(
            TransportConfig(
                name="mock",
                tiers=[StreamTier.BATCH],
                handler=mock_handler,
            )
        )

        # Emit events that will exceed compression threshold AND count threshold
        for i in range(50):
            stream.emit(EventCategory.JOINT_STATE, {"q": list(range(20))})

        stream._flush_all()

        # At least one payload should be compressed (bytes, not list)
        compressed_payloads = [p for p in received if isinstance(p, bytes)]
        assert len(compressed_payloads) > 0, "Expected compressed payload"

        # Verify it decompresses correctly
        decompressed = gzip.decompress(compressed_payloads[0])
        events = json.loads(decompressed)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_retry_on_failure(self):
        """Transport failures trigger retries with backoff."""
        stream = TelemetryStream(robot_id="test")

        call_count = 0

        def failing_handler(payload: Any) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated failure")
            return True

        stream.add_transport(
            TransportConfig(
                name="retry_test",
                tiers=[StreamTier.BATCH],
                handler=failing_handler,
                retry_max=3,
                retry_base_ms=10.0,  # Short for testing
            )
        )

        for i in range(200):
            stream.emit(EventCategory.JOINT_STATE, {"i": i})

        stream._flush_all()

        assert call_count >= 3, "Should have retried"

    def test_retry_exhausted(self):
        """Exhausted retries increment error counter."""
        stream = TelemetryStream(robot_id="test")

        def always_fail(payload: Any) -> bool:
            raise ConnectionError("Always fails")

        stream.add_transport(
            TransportConfig(
                name="fail_test",
                tiers=[StreamTier.BATCH],
                handler=always_fail,
                retry_max=2,
                retry_base_ms=1.0,
            )
        )

        for i in range(200):
            stream.emit(EventCategory.JOINT_STATE, {"i": i})

        stream._flush_all()

        stats = stream.get_stats()
        assert stats["errors"] > 0

    def test_stats(self):
        """Stats reflect current state."""
        stream = TelemetryStream(robot_id="stats_test")
        stats = stream.get_stats()

        assert stats["emitted"] == 0
        assert stats["flushed"] == 0
        assert stats["running"] is False
        assert stats["robot_id"] == "stats_test"
        assert "BATCH" in stats["buffer_sizes"]

    def test_start_twice_warns(self):
        """Starting twice logs warning but doesn't crash."""
        stream = TelemetryStream(robot_id="test")
        stream.start()
        stream.start()  # Should warn, not crash
        stream.stop()

    def test_stop_without_start(self):
        """Stopping without starting is a no-op."""
        stream = TelemetryStream(robot_id="test")
        stream.stop()  # Should not raise

    def test_age_based_flush(self):
        """Events are flushed when age threshold exceeded."""
        stream = TelemetryStream(robot_id="test")

        received: List[Any] = []

        def mock_handler(payload: Any) -> bool:
            received.append(payload)
            return True

        stream.add_transport(
            TransportConfig(
                name="mock",
                tiers=[StreamTier.BATCH],
                handler=mock_handler,
            )
        )

        # Override batch config with very short age
        stream._batch_configs[StreamTier.BATCH] = BatchConfig(
            max_count=10000, max_bytes=10 * 1024 * 1024, max_age_ms=1.0
        )

        stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
        time.sleep(0.01)  # Wait for age threshold

        stream._flush_all()
        assert len(received) > 0

    def test_multiple_transports(self):
        """Events are sent to all matching transports."""
        stream = TelemetryStream(robot_id="test")

        received_a: List[Any] = []
        received_b: List[Any] = []

        stream.add_transport(
            TransportConfig(name="a", tiers=[StreamTier.BATCH], handler=lambda p: (received_a.append(p), True)[-1])
        )
        stream.add_transport(
            TransportConfig(name="b", tiers=[StreamTier.BATCH], handler=lambda p: (received_b.append(p), True)[-1])
        )

        for i in range(200):
            stream.emit(EventCategory.JOINT_STATE, {"i": i})

        stream._flush_all()

        assert len(received_a) > 0
        assert len(received_b) > 0

    def test_sim_or_real_tag(self):
        """sim_or_real tag is preserved."""
        stream = TelemetryStream(robot_id="test")

        stream.emit(EventCategory.JOINT_STATE, {"q": [0.1]}, sim_or_real="sim")
        event = stream._buffers[StreamTier.BATCH][-1]
        assert event.sim_or_real == "sim"


# ============================================================================
# LocalWALTransport Tests
# ============================================================================


class TestLocalWALTransport:
    """Tests for the Write-Ahead Log transport."""

    def test_write_events(self, tmp_path):
        """Write events to WAL."""
        wal = LocalWALTransport(wal_dir=str(tmp_path / "wal"))
        events = [
            {"category": "joint_state", "robot_id": "test", "data": {"q": [0.1]}},
            {"category": "joint_state", "robot_id": "test", "data": {"q": [0.2]}},
        ]
        result = wal.send(events)
        assert result is True

        # Verify files exist
        wal_files = list(Path(wal.wal_dir).glob("telemetry_*.jsonl"))
        assert len(wal_files) == 1

        # Read back
        with open(wal_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["data"]["q"] == [0.1]

        wal.close()

    def test_write_compressed_payload(self, tmp_path):
        """Write gzip-compressed payload."""
        wal = LocalWALTransport(wal_dir=str(tmp_path / "wal"))

        events = [{"category": "test", "data": {"i": i}} for i in range(5)]
        compressed = gzip.compress(json.dumps(events).encode())

        result = wal.send(compressed)
        assert result is True

        wal.close()

    def test_rotation(self, tmp_path):
        """File rotates when size exceeded."""
        wal = LocalWALTransport(
            wal_dir=str(tmp_path / "wal"),
            max_file_bytes=200,  # Very small for testing
            compress_rotated=False,
        )

        for i in range(20):
            wal.send([{"data": {"i": i, "padding": "x" * 50}}])

        wal.close()

        files = list(Path(wal.wal_dir).glob("telemetry_*"))
        assert len(files) > 1, "Expected rotation to create multiple files"

    def test_rotation_with_compression(self, tmp_path):
        """Rotated files are gzip compressed."""
        wal = LocalWALTransport(
            wal_dir=str(tmp_path / "wal"),
            max_file_bytes=200,
            compress_rotated=True,
        )

        for i in range(20):
            wal.send([{"data": {"i": i, "padding": "x" * 50}}])

        wal.close()

        gz_files = list(Path(wal.wal_dir).glob("*.gz"))
        assert len(gz_files) > 0, "Expected gzip files"

    def test_cleanup_old_files(self, tmp_path):
        """Old files are deleted when max_files exceeded."""
        wal = LocalWALTransport(
            wal_dir=str(tmp_path / "wal"),
            max_file_bytes=100,
            max_files=3,
            compress_rotated=False,
        )

        for i in range(30):
            wal.send([{"data": {"i": i, "padding": "x" * 50}}])

        wal.close()

        files = list(Path(wal.wal_dir).glob("telemetry_*"))
        assert len(files) <= 3

    def test_stats(self, tmp_path):
        """Stats return correct info."""
        wal = LocalWALTransport(wal_dir=str(tmp_path / "wal"))
        wal.send([{"test": True}])

        stats = wal.get_stats()
        assert stats["file_count"] > 0
        assert stats["total_written_bytes"] > 0
        assert stats["wal_dir"] == str(tmp_path / "wal")

        wal.close()

    def test_error_handling(self, tmp_path):
        """Graceful error handling on write failure."""
        wal = LocalWALTransport(wal_dir=str(tmp_path / "wal"))

        # Send non-serializable data
        result = wal.send(object())  # type: ignore[arg-type]
        assert result is False


# ============================================================================
# StdoutTransport Tests
# ============================================================================


class TestStdoutTransport:
    """Tests for the debug stdout transport."""

    def test_send_events(self):
        """Send events to stdout."""
        transport = StdoutTransport()
        events = [
            {"category": "joint_state", "robot_id": "test", "frame_id": 0},
            {"category": "task_start", "robot_id": "test", "frame_id": 1},
        ]
        result = transport.send(events)
        assert result is True
        assert transport._total_events == 2

    def test_send_compressed(self):
        """Send compressed payload."""
        transport = StdoutTransport()
        events = [{"category": "test", "data": {"i": i}} for i in range(3)]
        compressed = gzip.compress(json.dumps(events).encode())

        result = transport.send(compressed)
        assert result is True
        assert transport._total_events == 3

    def test_max_events_per_batch(self):
        """Respects max events per batch limit."""
        transport = StdoutTransport(max_events_per_batch=2)
        events = [{"category": "test"} for _ in range(10)]

        result = transport.send(events)
        assert result is True
        assert transport._total_events == 10  # All counted

    def test_never_fails(self):
        """Stdout transport never returns False."""
        transport = StdoutTransport()
        result = transport.send("not valid json")  # type: ignore[arg-type]
        assert result is True

    def test_stats(self):
        """Stats track total events."""
        transport = StdoutTransport()
        transport.send([{"test": True}])
        stats = transport.get_stats()
        assert stats["total_events_logged"] == 1


# ============================================================================
# stream Tool Tests
# ============================================================================


class TestStreamTool:
    """Tests for the Strands @tool wrapper."""

    @pytest.fixture(autouse=True)
    def clear_streams(self):
        """Clear global stream registry between tests."""
        from strands_robots.tools.stream import _STREAMS

        _STREAMS.clear()
        yield
        # Cleanup: stop any running streams
        for stream in list(_STREAMS.values()):
            try:
                stream.stop(timeout=1.0)
            except Exception:
                pass
        _STREAMS.clear()

    def test_start_action(self, tmp_path):
        """Start a telemetry stream."""
        from strands_robots.tools.stream import stream as stream_tool

        result = stream_tool(
            action="start",
            robot_id="test_bot",
            wal_dir=str(tmp_path / "wal"),
            enable_stdout=True,
        )
        assert result["status"] == "success"
        assert "started" in result["content"][0]["text"]

    def test_stop_action(self, tmp_path):
        """Stop a telemetry stream."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="stop_test", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="stop", robot_id="stop_test")
        assert result["status"] == "success"
        assert "stopped" in result["content"][0]["text"]

    def test_emit_action(self, tmp_path):
        """Emit an event via tool."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="emit_test", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(
            action="emit",
            robot_id="emit_test",
            category="joint_state",
            data='{"q": [0.1, 0.2]}',
        )
        assert result["status"] == "success"
        assert "emitted" in result["content"][0]["text"].lower()

    def test_status_action(self, tmp_path):
        """Get telemetry status."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="status_test", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="status", robot_id="status_test")
        assert result["status"] == "success"
        assert "status_test" in result["content"][0]["text"]

    def test_status_no_streams(self):
        """Status with no streams running."""
        from strands_robots.tools.stream import stream as stream_tool

        result = stream_tool(action="status")
        assert result["status"] == "success"
        assert "No telemetry streams" in result["content"][0]["text"]

    def test_flush_action(self, tmp_path):
        """Force flush events."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="flush_test", wal_dir=str(tmp_path / "wal"))
        stream_tool(action="emit", robot_id="flush_test", category="joint_state", data='{"q": [0.1]}')
        result = stream_tool(action="flush", robot_id="flush_test")
        assert result["status"] == "success"
        assert "flush" in result["content"][0]["text"].lower()

    def test_start_trace_action(self, tmp_path):
        """Start a correlation trace."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="trace_test", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="start_trace", robot_id="trace_test", trace_name="pick_cube")
        assert result["status"] == "success"
        assert "pick_cube" in result["content"][0]["text"]

    def test_end_trace_action(self, tmp_path):
        """End a correlation trace."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="trace_test2", wal_dir=str(tmp_path / "wal"))
        stream_tool(action="start_trace", robot_id="trace_test2", trace_name="place_cube")
        result = stream_tool(action="end_trace", robot_id="trace_test2")
        assert result["status"] == "success"
        assert "place_cube" in result["content"][0]["text"]

    def test_unknown_action(self):
        """Unknown action returns error."""
        from strands_robots.tools.stream import stream as stream_tool

        result = stream_tool(action="invalid_action")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_emit_without_start(self):
        """Emit without starting returns error."""
        from strands_robots.tools.stream import stream as stream_tool

        result = stream_tool(action="emit", robot_id="nonexistent")
        assert result["status"] == "error"
        assert "No stream running" in result["content"][0]["text"]

    def test_start_twice_fails(self, tmp_path):
        """Starting same robot twice returns error."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="dup_test", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="start", robot_id="dup_test", wal_dir=str(tmp_path / "wal"))
        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"]

    def test_end_trace_without_start(self, tmp_path):
        """End trace without starting returns error."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="no_trace", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="end_trace", robot_id="no_trace")
        assert result["status"] == "error"
        assert "No active trace" in result["content"][0]["text"]

    def test_emit_unparseable_data(self, tmp_path):
        """Emit with non-JSON data wraps as raw."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="raw_test", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(
            action="emit",
            robot_id="raw_test",
            data="not json at all",
        )
        assert result["status"] == "success"

    def test_start_trace_no_name(self, tmp_path):
        """Start trace without name returns error."""
        from strands_robots.tools.stream import stream as stream_tool

        stream_tool(action="start", robot_id="noname", wal_dir=str(tmp_path / "wal"))
        result = stream_tool(action="start_trace", robot_id="noname")
        assert result["status"] == "error"
        assert "trace_name is required" in result["content"][0]["text"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self, tmp_path):
        """Full pipeline: create stream → emit → flush → verify WAL."""
        wal_dir = tmp_path / "wal"

        stream = TelemetryStream(robot_id="integration_bot")

        wal = LocalWALTransport(wal_dir=str(wal_dir))
        stream.add_transport(
            TransportConfig(
                name="wal",
                tiers=[StreamTier.STREAM, StreamTier.BATCH, StreamTier.STORAGE],
                handler=wal.send,
            )
        )

        stream.start()

        # Simulate control loop
        with stream.span("pick_cube") as trace_id:
            for i in range(50):
                stream.emit(
                    EventCategory.JOINT_STATE,
                    {"q": [0.1 * i, 0.2 * i], "step": i},
                    sim_or_real="sim",
                )

        # Allow flush thread to run
        time.sleep(1.0)
        stream._flush_all()
        stream.stop()
        wal.close()

        # Verify WAL files
        wal_files = list(wal_dir.glob("telemetry_*"))
        assert len(wal_files) > 0

        # Read and verify events
        all_events = []
        for f in wal_files:
            if f.suffix == ".gz":
                with gzip.open(f, "rt") as gf:
                    for line in gf:
                        all_events.append(json.loads(line))
            elif f.suffix == ".jsonl":
                with open(f) as jf:
                    for line in jf:
                        all_events.append(json.loads(line))

        # Should have 50 joint states + 2 task events (start/end from span)
        joint_events = [e for e in all_events if e.get("category") == "joint_state"]
        assert len(joint_events) == 50

        # All should have correlation_id
        for e in joint_events:
            assert e["correlation_id"] == trace_id

        stats = stream.get_stats()
        assert stats["emitted"] == 52  # 50 joint + 2 task (start/end)

    def test_simulated_50hz_loop(self, tmp_path):
        """Simulate a 50Hz control loop with telemetry."""
        stream = TelemetryStream(robot_id="sim_50hz")

        event_count = 0

        def counter_handler(payload: Any) -> bool:
            nonlocal event_count
            if isinstance(payload, bytes):
                events = json.loads(gzip.decompress(payload))
            else:
                events = payload
            event_count += len(events)
            return True

        stream.add_transport(
            TransportConfig(
                name="counter",
                tiers=[StreamTier.BATCH],
                handler=counter_handler,
            )
        )

        stream.start()

        # Simulate 1 second at 50Hz
        for _ in range(50):
            stream.emit(
                EventCategory.JOINT_STATE,
                {"q": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
                sim_or_real="sim",
            )
            time.sleep(0.02)  # 50Hz

        time.sleep(1.0)  # Let flush thread process
        stream._flush_all()
        stream.stop()

        assert event_count == 50, f"Expected 50 events, got {event_count}"

    def test_numpy_roundtrip(self, tmp_path):
        """numpy arrays survive serialize → WAL → deserialize."""
        np = pytest.importorskip("numpy")

        wal_dir = tmp_path / "wal"
        stream = TelemetryStream(robot_id="np_test")

        wal = LocalWALTransport(wal_dir=str(wal_dir))
        stream.add_transport(
            TransportConfig(
                name="wal",
                tiers=[StreamTier.BATCH],
                handler=wal.send,
            )
        )

        stream.start()
        stream.emit(
            EventCategory.OBSERVATION,
            {
                "q": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "vel": np.float64(1.5),
                "steps": np.int32(42),
            },
        )

        time.sleep(0.6)
        stream._flush_all()
        stream.stop()
        wal.close()

        # Read back
        wal_files = list(wal_dir.glob("telemetry_*.jsonl"))
        assert len(wal_files) > 0

        with open(wal_files[0]) as f:
            events = [json.loads(line) for line in f]

        obs_events = [e for e in events if e.get("category") == "observation"]
        assert len(obs_events) >= 1
        data = obs_events[0]["data"]
        assert data["q"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
        assert data["vel"] == pytest.approx(1.5, rel=1e-5)
        assert data["steps"] == 42
